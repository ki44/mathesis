from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"


class ModelConfiguration(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8")
    model_name: str = Field(..., description="The name of the model to use")
    temperature: float | None = Field(None, description="Sampling temperature for text generation")
    max_tokens: int = Field(8192, description="Maximum number of tokens to generate")
    max_iterations: int = Field(5, description="Maximum number of sequential tool calls allowed in a single completion")
