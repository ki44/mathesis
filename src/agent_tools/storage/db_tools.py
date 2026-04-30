from agent_tools.tools_utils import tool

from .schemas import ReadParameters, WriteParameters


@tool(
    description="Writes a value to a document in the database.",
    parameters=WriteParameters,
)
def write(document: str, key: str, value: str) -> None:
    pass


@tool(
    description="Reads a value from a document in the database.",
    parameters=ReadParameters,
)
def read(document: str, key: str) -> str:
    return "This is a dummy function that will be implemented later."


@tool(
    description="Get list of files in the database.",
    parameters=None,
)
def list_files() -> list[str]:
    return []
