import json
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import Agent
from agent_tools.storage.db import get_db, init_db
from agent_tools.storage.db_tools import list_course_files, propose_course_update, read_course
from schemas.schemas import (
    ApplyChangesRequest,
    ChatRequest,
    ConversationSummary,
    ConversationUpdateRequest,
    CopyFileRequest,
    CourseFile,
    CreateFileRequest,
    DisplayMessage,
    FolderCreate,
    FolderEntry,
    FolderRenameRequest,
    ForkRequest,
    Proposal,
    RenameFileRequest,
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Mathesis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SYSTEM_PROMPT = """\
Tu es Mathesis, un tuteur de mathématiques expert. Tu rédiges des cours de maths \
clairs et rigoureux en Markdown (syntaxe compatible Obsidian). \
Lorsque l'utilisateur te demande de créer ou modifier un cours, utilise l'outil \
propose_course_update pour soumettre le contenu. L'utilisateur verra la diff et \
pourra accepter ou rejeter les changements. Pour voir le contenu actuel d'un fichier, \
utilise read_course. Pour lister les fichiers existants, utilise list_course_files.\
"""

agent = Agent(
    system_prompt=_SYSTEM_PROMPT,
    tools=[list_course_files, read_course, propose_course_update],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_display_messages(llm_history: list[dict]) -> list[dict]:
    """Derive frontend-displayable messages from the raw LLM chat history."""
    result = []
    for msg in llm_history:
        role = msg.get("role")
        if role == "user":
            content = msg.get("content")
            if isinstance(content, str):
                result.append({"role": "user", "content": content})
        elif role == "assistant":
            content = msg.get("content")
            if content:
                result.append({"role": "assistant", "content": content})
            for tc in msg.get("tool_calls") or []:
                name = (tc.get("function") or {}).get("name", "unknown")
                result.append({"role": "tool_call", "content": f"{name}"})
    return result


def _find_history_slice_end(history: list[dict], display_index: int) -> int:
    """Return exclusive end index into history for the given 0-based display message index.

    Extends the slice past any immediately-following role='tool' messages so that an
    assistant message with tool_calls is never left without its tool results.
    """

    def _end(i: int) -> int:
        end = i + 1
        while end < len(history) and history[end].get("role") == "tool":
            end += 1
        return end

    count = 0
    for i, msg in enumerate(history):
        role = msg.get("role")
        if role == "user":
            if isinstance(msg.get("content"), str):
                if count == display_index:
                    return _end(i)
                count += 1
        elif role == "assistant":
            if msg.get("content"):
                if count == display_index:
                    return _end(i)
                count += 1
            for _ in msg.get("tool_calls") or []:
                if count == display_index:
                    return _end(i)
                count += 1
    return len(history)


async def _fetchone_or_404(db: aiosqlite.Connection, query: str, params: tuple[Any, ...], detail: str) -> aiosqlite.Row:
    cursor = await db.execute(query, params)
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=detail)
    return row


async def _execute_or_404(db: aiosqlite.Connection, query: str, params: tuple[Any, ...], detail: str) -> None:
    result = await db.execute(query, params)
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=detail)


async def get_db_session() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with get_db() as db:
        yield db


async def _save_course(db: aiosqlite.Connection, filename: str, content: str) -> CourseFile:
    await _execute_or_404(
        db,
        "UPDATE course_files SET content = ?, updated_at = datetime('now') WHERE filename = ?",
        (content, filename),
        "Course file not found",
    )
    cursor = await db.execute(
        "SELECT filename, content, updated_at FROM course_files WHERE filename = ?",
        (filename,),
    )
    row = await cursor.fetchone()
    return CourseFile.model_validate(dict(row))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    conv_id = request.conversation_id or str(uuid.uuid4())

    history: list[dict] = []
    async with get_db() as db:
        cursor = await db.execute("SELECT llm_history FROM conversations WHERE id = ?", (conv_id,))
        row = await cursor.fetchone()
        if row:
            history = json.loads(row["llm_history"])

    # When rerun=True, strip the last user message and everything after it so the
    # agent replays from a clean slate for that turn.
    if request.rerun:
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "user":
                history = history[:i]
                break

    async def event_generator():
        async for event in agent.stream(request.message, history):
            yield event

        async with get_db() as db:
            if row:
                await db.execute(
                    "UPDATE conversations SET llm_history = ?, updated_at = datetime('now') WHERE id = ?",
                    (json.dumps(history), conv_id),
                )
            else:
                c = next((m.get("content", "") for m in history if m.get("role") == "user"), "")
                title = (c[:40] + ("..." if len(c) > 40 else "")) if c else "New conversation"
                await db.execute(
                    "INSERT INTO conversations (id, title, llm_history) VALUES (?, ?, ?)",
                    (conv_id, title, json.dumps(history)),
                )
            await db.commit()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@app.get("/api/conversations", response_model=list[ConversationSummary])
async def get_conversations(db: aiosqlite.Connection = Depends(get_db_session)):
    cursor = await db.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
    rows = await cursor.fetchall()
    return [ConversationSummary.model_validate(dict(r)) for r in rows]


@app.get("/api/conversations/{conv_id}/messages", response_model=list[DisplayMessage])
async def get_conversation_messages(conv_id: str, db: aiosqlite.Connection = Depends(get_db_session)):
    row = await _fetchone_or_404(
        db, "SELECT llm_history FROM conversations WHERE id = ?", (conv_id,), "Conversation not found"
    )
    history = json.loads(row["llm_history"])
    return _to_display_messages(history)


@app.delete("/api/conversations/{conv_id}", status_code=204)
async def delete_conversation(conv_id: str, db: aiosqlite.Connection = Depends(get_db_session)):
    await _execute_or_404(db, "DELETE FROM conversations WHERE id = ?", (conv_id,), "Conversation not found")
    await db.commit()


@app.patch("/api/conversations/{conv_id}", response_model=ConversationSummary)
async def rename_conversation(
    conv_id: str, body: ConversationUpdateRequest, db: aiosqlite.Connection = Depends(get_db_session)
):
    await _execute_or_404(
        db,
        "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
        (body.title, conv_id),
        "Conversation not found",
    )
    await db.commit()
    row = await _fetchone_or_404(
        db,
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
        (conv_id,),
        "Conversation not found",
    )
    return ConversationSummary.model_validate(dict(row))


@app.post("/api/conversations/{conv_id}/fork", response_model=ConversationSummary)
async def fork_conversation(conv_id: str, body: ForkRequest, db: aiosqlite.Connection = Depends(get_db_session)):
    row = await _fetchone_or_404(
        db, "SELECT llm_history FROM conversations WHERE id = ?", (conv_id,), "Conversation not found"
    )
    history: list[dict] = json.loads(row["llm_history"])
    forked_history = history[: _find_history_slice_end(history, body.message_index)]
    new_id = str(uuid.uuid4())
    c = next((m.get("content", "") for m in forked_history if m.get("role") == "user"), "")
    title = ("Fork: " + c[:35] + ("..." if len(c) > 35 else "")) if c else "Forked conversation"
    await db.execute(
        "INSERT INTO conversations (id, title, llm_history) VALUES (?, ?, ?)",
        (new_id, title, json.dumps(forked_history)),
    )
    await db.commit()
    row2 = await _fetchone_or_404(
        db, "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (new_id,), ""
    )
    return ConversationSummary.model_validate(dict(row2))


# ---------------------------------------------------------------------------
# Courses
# ---------------------------------------------------------------------------


@app.get("/api/courses", response_model=list[CourseFile])
async def get_courses(db: aiosqlite.Connection = Depends(get_db_session)):
    cursor = await db.execute("SELECT filename, content, updated_at FROM course_files ORDER BY filename")
    rows = await cursor.fetchall()
    return [CourseFile.model_validate(dict(r)) for r in rows]


@app.post("/api/courses", response_model=CourseFile, status_code=201)
async def create_course(body: CreateFileRequest, db: aiosqlite.Connection = Depends(get_db_session)):
    cursor = await db.execute("SELECT 1 FROM course_files WHERE filename = ?", (body.filename,))
    if await cursor.fetchone():
        raise HTTPException(status_code=409, detail="File already exists")
    await db.execute("INSERT INTO course_files (filename, content) VALUES (?, ?)", (body.filename, body.content))
    await db.commit()
    cursor2 = await db.execute(
        "SELECT filename, content, updated_at FROM course_files WHERE filename = ?", (body.filename,)
    )
    return CourseFile.model_validate(dict(await cursor2.fetchone()))


@app.get("/api/courses/{filename:path}", response_model=CourseFile)
async def get_course(filename: str, db: aiosqlite.Connection = Depends(get_db_session)):
    row = await _fetchone_or_404(
        db,
        "SELECT filename, content, updated_at FROM course_files WHERE filename = ?",
        (filename,),
        "Course file not found",
    )
    return CourseFile.model_validate(dict(row))


@app.post("/api/courses/{filename:path}", response_model=CourseFile)
async def apply_changes(filename: str, body: ApplyChangesRequest, db: aiosqlite.Connection = Depends(get_db_session)):
    updated = await _save_course(db, filename, body.content)
    await db.execute("DELETE FROM proposals WHERE filename = ?", (filename,))
    await db.commit()
    return updated


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


@app.get("/api/proposals", response_model=list[Proposal])
async def get_proposals(db: aiosqlite.Connection = Depends(get_db_session)):
    cursor = await db.execute(
        "SELECT filename, proposed_content, description, created_at FROM proposals ORDER BY filename"
    )
    rows = await cursor.fetchall()
    return [Proposal.model_validate(dict(r)) for r in rows]


@app.get("/api/proposals/{filename:path}", response_model=Proposal)
async def get_proposal(filename: str, db: aiosqlite.Connection = Depends(get_db_session)):
    row = await _fetchone_or_404(
        db,
        "SELECT filename, proposed_content, description, created_at FROM proposals WHERE filename = ?",
        (filename,),
        "Proposal not found",
    )
    return Proposal.model_validate(dict(row))


@app.delete("/api/proposals/{filename:path}", status_code=204)
async def reject_proposal(filename: str, db: aiosqlite.Connection = Depends(get_db_session)):
    await _execute_or_404(db, "DELETE FROM proposals WHERE filename = ?", (filename,), "Proposal not found")
    await db.commit()


# ---------------------------------------------------------------------------
# Course manual save (keeps proposal intact)
# ---------------------------------------------------------------------------


@app.patch("/api/courses/{filename:path}", response_model=CourseFile)
async def save_course(filename: str, body: ApplyChangesRequest, db: aiosqlite.Connection = Depends(get_db_session)):
    result = await _save_course(db, filename, body.content)
    await db.commit()
    return result


@app.delete("/api/courses/{filename:path}", status_code=204)
async def delete_course(filename: str, db: aiosqlite.Connection = Depends(get_db_session)):
    await _execute_or_404(db, "DELETE FROM course_files WHERE filename = ?", (filename,), "Course file not found")
    await db.execute("DELETE FROM proposals WHERE filename = ?", (filename,))
    await db.commit()


# ---------------------------------------------------------------------------
# File operations (rename / copy) — body-based to avoid :path routing conflicts
# ---------------------------------------------------------------------------


@app.post("/api/file-ops/rename", response_model=CourseFile)
async def rename_file(body: RenameFileRequest, db: aiosqlite.Connection = Depends(get_db_session)):
    row = await _fetchone_or_404(
        db, "SELECT content FROM course_files WHERE filename = ?", (body.old_filename,), "Course file not found"
    )
    cursor = await db.execute("SELECT 1 FROM course_files WHERE filename = ?", (body.new_filename,))
    if await cursor.fetchone():
        raise HTTPException(status_code=409, detail="A file with that name already exists")
    await db.execute("INSERT INTO course_files (filename, content) VALUES (?, ?)", (body.new_filename, row["content"]))
    await db.execute("DELETE FROM course_files WHERE filename = ?", (body.old_filename,))
    # Migrate proposal if one exists
    cursor2 = await db.execute(
        "SELECT proposed_content, description FROM proposals WHERE filename = ?", (body.old_filename,)
    )
    proposal_row = await cursor2.fetchone()
    if proposal_row:
        await db.execute("DELETE FROM proposals WHERE filename = ?", (body.old_filename,))
        await db.execute(
            "INSERT INTO proposals (filename, proposed_content, description) VALUES (?, ?, ?)",
            (body.new_filename, proposal_row["proposed_content"], proposal_row["description"]),
        )
    await db.commit()
    cursor3 = await db.execute(
        "SELECT filename, content, updated_at FROM course_files WHERE filename = ?", (body.new_filename,)
    )
    return CourseFile.model_validate(dict(await cursor3.fetchone()))


@app.post("/api/file-ops/copy", response_model=CourseFile)
async def copy_file(body: CopyFileRequest, db: aiosqlite.Connection = Depends(get_db_session)):
    row = await _fetchone_or_404(
        db, "SELECT content FROM course_files WHERE filename = ?", (body.filename,), "Course file not found"
    )
    if body.new_filename:
        new_name = body.new_filename
    else:
        # Auto-generate: strip extension, append " copy", re-add extension
        parts = body.filename.rsplit(".", 1)
        new_name = parts[0] + " copy." + parts[1] if len(parts) == 2 else body.filename + " copy"
    cursor = await db.execute("SELECT 1 FROM course_files WHERE filename = ?", (new_name,))
    if await cursor.fetchone():
        raise HTTPException(status_code=409, detail="A file with that name already exists")
    await db.execute("INSERT INTO course_files (filename, content) VALUES (?, ?)", (new_name, row["content"]))
    await db.commit()
    cursor2 = await db.execute("SELECT filename, content, updated_at FROM course_files WHERE filename = ?", (new_name,))
    return CourseFile.model_validate(dict(await cursor2.fetchone()))


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------


@app.get("/api/folders", response_model=list[FolderEntry])
async def get_folders(db: aiosqlite.Connection = Depends(get_db_session)):
    cursor = await db.execute("SELECT path, created_at FROM folders ORDER BY path")
    rows = await cursor.fetchall()
    return [FolderEntry.model_validate(dict(r)) for r in rows]


@app.post("/api/folders", response_model=FolderEntry, status_code=201)
async def create_folder(body: FolderCreate, db: aiosqlite.Connection = Depends(get_db_session)):
    try:
        await db.execute("INSERT INTO folders (path) VALUES (?)", (body.path,))
        await db.commit()
    except aiosqlite.IntegrityError:
        raise HTTPException(status_code=409, detail="Folder already exists")
    cursor = await db.execute("SELECT path, created_at FROM folders WHERE path = ?", (body.path,))
    return FolderEntry.model_validate(dict(await cursor.fetchone()))


@app.delete("/api/folders/{path:path}", status_code=204)
async def delete_folder(path: str, db: aiosqlite.Connection = Depends(get_db_session)):
    await _fetchone_or_404(db, "SELECT 1 FROM folders WHERE path = ?", (path,), "Folder not found")
    prefix = path + "/"
    await db.execute("DELETE FROM proposals WHERE filename LIKE ?", (prefix + "%",))
    await db.execute("DELETE FROM course_files WHERE filename LIKE ?", (prefix + "%",))
    await db.execute("DELETE FROM folders WHERE path = ? OR path LIKE ?", (path, prefix + "%"))
    await db.commit()


@app.post("/api/folder-ops/rename", response_model=list[CourseFile])
async def rename_folder(body: FolderRenameRequest, db: aiosqlite.Connection = Depends(get_db_session)):
    """Rename a folder: updates its entry and renames all files whose paths begin with old_path/."""
    await _fetchone_or_404(db, "SELECT 1 FROM folders WHERE path = ?", (body.old_path,), "Folder not found")
    # Preflight: ensure destination folder doesn't already exist before any mutations
    collision_folder = await db.execute("SELECT 1 FROM folders WHERE path = ?", (body.new_path,))
    if await collision_folder.fetchone():
        raise HTTPException(status_code=409, detail="A folder already exists at the destination")
    prefix = body.old_path + "/"
    cursor = await db.execute("SELECT filename, content FROM course_files WHERE filename LIKE ?", (prefix + "%",))
    affected = await cursor.fetchall()
    updated_files: list[CourseFile] = []
    for file_row in affected:
        old_fname = file_row["filename"]
        new_fname = body.new_path + "/" + old_fname[len(prefix) :]
        collision = await db.execute("SELECT 1 FROM course_files WHERE filename = ?", (new_fname,))
        if await collision.fetchone():
            raise HTTPException(status_code=409, detail=f"File already exists: {new_fname}")
        await db.execute("INSERT INTO course_files (filename, content) VALUES (?, ?)", (new_fname, file_row["content"]))
        await db.execute("DELETE FROM course_files WHERE filename = ?", (old_fname,))
        # Migrate proposal
        cur2 = await db.execute("SELECT proposed_content, description FROM proposals WHERE filename = ?", (old_fname,))
        prop = await cur2.fetchone()
        if prop:
            await db.execute("DELETE FROM proposals WHERE filename = ?", (old_fname,))
            await db.execute(
                "INSERT INTO proposals (filename, proposed_content, description) VALUES (?, ?, ?)",
                (new_fname, prop["proposed_content"], prop["description"]),
            )
        cur3 = await db.execute(
            "SELECT filename, content, updated_at FROM course_files WHERE filename = ?", (new_fname,)
        )
        updated_files.append(CourseFile.model_validate(dict(await cur3.fetchone())))
    # Also rename sub-folders (use prefix-aware substr to avoid corrupting paths containing the old name)
    await db.execute(
        "UPDATE folders SET path = ? || substr(path, length(?) + 1) WHERE path = ? OR path LIKE ?",
        (body.new_path, body.old_path, body.old_path, prefix + "%"),
    )
    await db.commit()
    return updated_files
