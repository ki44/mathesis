import json

from dotenv import load_dotenv
from litellm import Message, ModelResponse, completion

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

    def _completion(self, **kwargs) -> Message:
        response = completion(**kwargs)
        assert isinstance(response, ModelResponse)
        return response.choices[0].message

    def completion(
        self, prompt: str, chat_history: list | None = None, tools: list[ToolFunction] | None = None
    ) -> Message:
        if chat_history is None:
            chat_history = []
        # If tools are provided in the call, use them. Otherwise, use the agent's tools.
        tools = tools or self.tools
        tools_by_name = {tool.schema["function"]["name"]: tool for tool in tools} if tools else {}

        if self.system_prompt and not any(m.get("role") == "system" for m in chat_history if isinstance(m, dict)):
            chat_history.insert(0, {"role": "system", "content": self.system_prompt})

        chat_history.append({"role": "user", "content": prompt})
        config = self.config.model_dump(exclude={"max_iterations"})
        message = self._completion(
            **config,
            messages=chat_history,
            tools=[tool.schema for tool in tools] if tools else None,
            tool_choice="auto",
            reasoning_effort="low",
        )
        chat_history.append(message.model_dump(exclude_none=True))

        for _ in range(self.config.max_iterations):
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    tool = tools_by_name.get(fn_name)
                    if tool is None:
                        raise ValueError(f"Unknown tool: {fn_name}")
                    tool_output = tool(**fn_args)

                    chat_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_output),
                        },
                    )
                message = self._completion(
                    **config,
                    messages=chat_history,
                    tools=[tool.schema for tool in tools] if tools else None,
                    tool_choice="auto",
                    reasoning_effort="low",
                )
                chat_history.append(message.model_dump(exclude_none=True))
            else:
                break
        else:
            print("Reached maximum number of iterations. Stopping.")
        return message


if __name__ == "__main__":
    agent = Agent(
        system_prompt=("You are a helpful assistant. "),
        tools=[read, write],
    )
    prompt = "Use the write and read tools once. Just test that you can call them."
    answer = agent.completion(prompt)
    # print(json.dumps(answer.model_dump(), indent=2))
    print(answer.content)
