from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"


class ModelConfiguration(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8")
    # API call configuration
    model: str = Field(..., description="The name of the model to use")
    base_url: str | None = Field(None, description="Base URL for API calls (if applicable)")
    api_key: str | None = Field(None, description="API key for authentication (if applicable)")
    temperature: float | None = Field(None, description="Sampling temperature for text generation")
    max_tokens: int = Field(8192, description="Maximum number of tokens to generate")

    # Agent loop configuration
    max_iterations: int = Field(5, description="Maximum number of sequential tool calls allowed in a single completion")


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    conversation_id: str | None = Field(None, description="Conversation ID (optional)")
    rerun: bool = Field(False, description="If True, strip the last user turn from history before generating")


class CourseFile(BaseModel):
    filename: str
    content: str
    updated_at: str


class Proposal(BaseModel):
    filename: str
    proposed_content: str
    description: str
    created_at: str


class ApplyChangesRequest(BaseModel):
    content: str = Field(..., description="The final merged Markdown content to save.")


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class DisplayMessage(BaseModel):
    role: str
    content: str


class FolderEntry(BaseModel):
    path: str
    created_at: str


class FolderCreate(BaseModel):
    path: str = Field(..., description="Folder path, e.g. 'algebra' or 'algebra/linear'")


class FolderRenameRequest(BaseModel):
    old_path: str
    new_path: str


class RenameFileRequest(BaseModel):
    old_filename: str
    new_filename: str


class CopyFileRequest(BaseModel):
    filename: str
    new_filename: str | None = Field(None, description="Target filename; auto-appends ' copy' if omitted")


class ConversationUpdateRequest(BaseModel):
    title: str


class ForkRequest(BaseModel):
    message_index: int = Field(..., description="0-based index of the display message to fork from (inclusive)")


class CreateFileRequest(BaseModel):
    filename: str
    content: str = ""
