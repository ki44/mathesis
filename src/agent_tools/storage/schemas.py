from pydantic import BaseModel, Field


class WriteParameters(BaseModel):
    document: str = Field(..., description="The name of the document to write to.")
    key: str = Field(..., description="The key to write the value under.")
    value: str = Field(..., description="The value to write.")


class ReadParameters(BaseModel):
    document: str = Field(..., description="The name of the document to read from.")
    key: str = Field(..., description="The key to read the value from.")
