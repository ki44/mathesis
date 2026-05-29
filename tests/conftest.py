import json
import os

# Must be set before main is imported so ModelConfiguration() can resolve.
os.environ.setdefault("MODEL", "test")

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

import agent_tools.storage.db as db_module  # noqa: E402
from agent_tools.storage.db import get_db, init_db  # noqa: E402
from main import app  # noqa: E402


@pytest.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "_DB_PATH", tmp_path / "test.db")
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# DB helpers (usable directly in tests after client fixture patches _DB_PATH)
# ---------------------------------------------------------------------------


async def insert_folders(*paths: str) -> None:
    async with get_db() as db:
        for path in paths:
            await db.execute("INSERT OR IGNORE INTO folders (path) VALUES (?)", (path,))
        await db.commit()


async def insert_files(*name_content_pairs: tuple[str, str]) -> None:
    async with get_db() as db:
        for filename, content in name_content_pairs:
            await db.execute(
                "INSERT OR IGNORE INTO course_files (filename, content) VALUES (?, ?)", (filename, content)
            )
        await db.commit()


async def insert_conversation(conv_id: str, history: list) -> None:
    async with get_db() as db:
        await db.execute(
            "INSERT INTO conversations (id, title, llm_history) VALUES (?, ?, ?)",
            (conv_id, "Test", json.dumps(history)),
        )
        await db.commit()


async def fetch_conversation_history(conv_id: str) -> list:
    async with get_db() as db:
        cursor = await db.execute("SELECT llm_history FROM conversations WHERE id = ?", (conv_id,))
        row = await cursor.fetchone()
        return json.loads(row["llm_history"]) if row else []
