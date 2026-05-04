import functools
from typing import Any, Callable

from pydantic import BaseModel


class ToolFunction:
    def __init__(self, func: Callable, schema: dict[str, Any]):
        self.schema = schema
        self._func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def tool(description: str, parameters: type[BaseModel] | None = None):
    def decorator(func) -> ToolFunction:
        default_parameters = {"type": "object", "properties": {}}
        parameters_schema = parameters.model_json_schema() if parameters else None
        # Remove "title" metadata from the parameters schema to avoid issues with some LLMs that may not handle it well.
        cleaned_schema = removes_title_metadata(parameters_schema) if parameters_schema else default_parameters
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": cleaned_schema,
            },
        }
        return ToolFunction(func, schema)

    return decorator


def removes_title_metadata(schema: dict | list) -> dict | list:
    if isinstance(schema, dict):
        return {k: removes_title_metadata(v) for k, v in schema.items() if k != "title"}
    elif isinstance(schema, list):
        return [removes_title_metadata(item) for item in schema]
    else:
        return schema
