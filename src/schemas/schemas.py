from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    conversation_id: str | None = Field(None, description="Conversation ID (optional)")
    rerun: bool = Field(False, description="If True, strip the last user turn from history before generating")
    variant_override: list[dict] | None = Field(
        None,
        description="When set, replaces the last response in history with these messages (selected variant)",
    )


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
