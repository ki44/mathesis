import functools
from typing import Any, Callable

from pydantic import BaseModel


class ToolFunction:
    def __init__(self, func: Callable, schema: dict[str, Any]):
        self.tool_schema = schema
        self._func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def tool(description: str, parameters: type[BaseModel] | None):
    def decorator(func) -> ToolFunction:
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters.model_json_schema() if parameters else None,
            },
        }
        return ToolFunction(func, schema)

    return decorator
