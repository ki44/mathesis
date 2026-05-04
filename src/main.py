from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import Agent
from agent_tools.storage.db_tools import list_files, read, write
from schemas.schemas import ChatRequest

load_dotenv()

app = FastAPI(title="Mathesis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state (single-user, replaced by DB in a later phase)
# ---------------------------------------------------------------------------
_chat_history: list[dict] = []
_course_content: str = ""
_course_proposal: str | None = None

_SYSTEM_PROMPT = """\
Tu es Mathesis, un tuteur de mathématiques expert. Tu rédiges des cours de maths \
clairs et rigoureux en Markdown (syntaxe compatible Obsidian). \
Lorsque l'utilisateur te demande de créer ou modifier un cours, utilise les outils \
disponibles pour lire et écrire le contenu dans la base de données.\
"""

agent = Agent(
    system_prompt=_SYSTEM_PROMPT,
    tools=[read, write, list_files],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream the agent's response as Server-Sent Events."""

    async def event_generator():
        async for event in agent.stream(request.message, _chat_history):
            yield event

    return StreamingResponse(event_generator(), media_type="text/event-stream")
