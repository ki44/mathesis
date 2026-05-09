from agent_tools.tools_utils import tool

from .db import get_db
from .schemas import ProposeCourseUpdateParams, ReadCourseParams


@tool(
    description="List all course file names currently stored in the database.",
)
async def list_course_files() -> list[str]:
    async with get_db() as db:
        cursor = await db.execute("SELECT filename FROM course_files ORDER BY filename")
        rows = await cursor.fetchall()
        return [row["filename"] for row in rows]


@tool(
    description="Read the current content of a course file, without any proposal you may have made.",
    parameters=ReadCourseParams,
)
async def read_course(filename: str) -> str:
    async with get_db() as db:
        cursor = await db.execute("SELECT content FROM course_files WHERE filename = ?", (filename,))
        row = await cursor.fetchone()
        if row is None:
            return ""
        return row["content"]


@tool(
    description=(
        "Propose an updated version of a course file. "
        "The user will review the diff and choose which changes to accept. "
        "Use this whenever you want to create or modify a course."
    ),
    parameters=ProposeCourseUpdateParams,
)
async def propose_course_update(filename: str, content: str, description: str) -> str:
    async with get_db() as db:
        # Ensure the course file entry exists (with empty content if new)
        await db.execute(
            """
            INSERT INTO course_files (filename, content)
            VALUES (?, '')
            ON CONFLICT(filename) DO NOTHING
            """,
            (filename,),
        )
        # Upsert the proposal
        await db.execute(
            """
            INSERT INTO proposals (filename, proposed_content, description)
            VALUES (?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                proposed_content = excluded.proposed_content,
                description      = excluded.description,
                created_at       = datetime('now')
            """,
            (filename, content, description),
        )
        await db.commit()
    return f"Proposal stored for '{filename}'. The user will review the diff."


if __name__ == "__main__":
    import asyncio

    asyncio.run(list_course_files())
    print(read_course.schema)
