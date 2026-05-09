from pydantic import BaseModel, Field


class ReadCourseParams(BaseModel):
    filename: str = Field(..., description="The course file name to read (e.g. 'derivatives.md').")


class ProposeCourseUpdateParams(BaseModel):
    filename: str = Field(..., description="The course file name to create or update (e.g. 'derivatives.md').")
    content: str = Field(..., description="The full Markdown content to propose for the course file.")
    description: str = Field(..., description="A short human-readable description of what changed and why.")
