from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

_DB_PATH = Path(__file__).parent.parent.parent / "mathesis.db"


@asynccontextmanager
async def get_db():
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def init_db() -> None:
    async with get_db() as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS course_files (
                filename    TEXT PRIMARY KEY,
                content     TEXT NOT NULL DEFAULT '',
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS proposals (
                filename         TEXT PRIMARY KEY,
                proposed_content TEXT NOT NULL,
                description      TEXT NOT NULL DEFAULT '',
                created_at       TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New conversation',
                llm_history TEXT NOT NULL DEFAULT '[]',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        await db.commit()
