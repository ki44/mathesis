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
from schemas.schemas import ApplyChangesRequest, ChatRequest, ConversationSummary, CourseFile, DisplayMessage, Proposal

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


# ---------------------------------------------------------------------------
# Courses
# ---------------------------------------------------------------------------


@app.get("/api/courses", response_model=list[CourseFile])
async def get_courses(db: aiosqlite.Connection = Depends(get_db_session)):
    cursor = await db.execute("SELECT filename, content, updated_at FROM course_files ORDER BY filename")
    rows = await cursor.fetchall()
    return [CourseFile.model_validate(dict(r)) for r in rows]


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
