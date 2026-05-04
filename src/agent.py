import asyncio
import json
from collections.abc import AsyncGenerator

import litellm
from dotenv import load_dotenv
from litellm import Message, ModelResponse, acompletion

from agent_tools.storage.db_tools import read, write
from agent_tools.tools_utils import ToolFunction
from schemas.schemas import ModelConfiguration

load_dotenv()
# litellm._logging._turn_on_debug()


class Agent:
    def __init__(self, system_prompt: str | None = None, tools: list[ToolFunction] | None = None):
        self.system_prompt = system_prompt
        self.tools = tools
        self.config = ModelConfiguration()  # type: ignore[call-arg]

        self._tools_by_name = {tool.schema["function"]["name"]: tool for tool in tools} if tools else {}

    async def _acompletion(self, **kwargs) -> Message:
        response = await acompletion(**kwargs)
        assert isinstance(response, ModelResponse)
        print("Full response:", json.dumps(response.model_dump(), indent=2))
        return response.choices[0].message

    def _initialization(self, chat_history: list | None, prompt: str) -> list:
        """Prepare the initial chat history, add the system prompt if included."""
        formatted_history = list(chat_history) if chat_history else []

        if self.system_prompt and not any(m.get("role") == "system" for m in formatted_history if isinstance(m, dict)):
            formatted_history.insert(0, {"role": "system", "content": self.system_prompt})
        formatted_history.append({"role": "user", "content": prompt})
        return formatted_history

    async def acompletion(
        self,
        prompt: str,
        chat_history: list | None = None,
    ) -> Message:
        """Get the agent's response for the given prompt and chat history, handling tool calls iteratively.

        Args:
            prompt: The user's message to the agent.
            chat_history: Optional prior chat history to maintain context across interactions.
        Returns:
            The final message from the agent after processing the prompt and any tool calls.
        """
        chat_history = self._initialization(chat_history, prompt)
        config = self.config.model_dump(exclude={"max_iterations"})

        for _ in range(self.config.max_iterations + 1):
            message = await self._acompletion(
                **config,
                messages=chat_history,
                tools=[tool.schema for tool in self.tools] if self.tools else None,
                tool_choice="auto",
                reasoning_effort="low",
            )
            chat_history.append(message.model_dump(exclude_none=True))

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    tool = self._tools_by_name.get(fn_name)
                    if tool is None:
                        raise ValueError(f"Unknown tool: {fn_name}")
                    if _ >= self.config.max_iterations:
                        tool_output = f"Sys Error: Maximum tool iteration limit {self.config.max_iterations} reached."
                    else:
                        tool_output = await tool(**fn_args)
                    chat_history.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_output)}
                    )
            else:
                break
        return message  # pyright: ignore[reportPossiblyUnboundVariable]

    async def stream(
        self,
        prompt: str,
        chat_history: list | None = None,
    ) -> AsyncGenerator[str, None]:
        """Async generator that yields SSE-formatted strings for the given prompt."""
        chat_history = self._initialization(chat_history, prompt)
        config = self.config.model_dump(exclude={"max_iterations"})

        for _ in range(self.config.max_iterations + 1):
            collected_chunks: list = []
            try:
                response = await acompletion(
                    **config,
                    messages=chat_history,
                    tools=[t.schema for t in self.tools] if self.tools else None,
                    tool_choice="auto",
                    stream=True,
                )
                async for chunk in response:  # type: ignore[union-attr]
                    collected_chunks.append(chunk)
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield _sse("delta", {"text": delta.content})
            except Exception as e:
                yield _sse("error", {"message": str(e)})
                return

            full_response = litellm.stream_chunk_builder(collected_chunks, messages=chat_history)
            message = full_response.choices[0].message  # type: ignore[union-attr]
            chat_history.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                break

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                yield _sse("tool_call", {"name": fn_name})

                tool = self._tools_by_name.get(fn_name)
                if tool is None:
                    yield _sse("error", {"message": f"Unknown tool: {fn_name}"})
                    return
                if _ >= self.config.max_iterations:
                    tool_output = f"Sys Error: Maximum tool iteration limit {self.config.max_iterations} reached."
                else:
                    tool_output = await tool(**fn_args)
                chat_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_output),
                    }
                )
        yield _sse("done", {})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


if __name__ == "__main__":

    async def main():
        agent = Agent(
            system_prompt=("You are a helpful assistant. "),
            tools=[read, write],
        )
        prompt = "Utilise tes 2 tools, ce sont des dummy tool mais je veux tester déjà si ça marche comme ça."
        answer = await agent.acompletion(prompt)
        # print(json.dumps(answer.model_dump(), indent=2))
        print(answer.content)

    asyncio.run(main())
