from dotenv import load_dotenv
from litellm import Message, ModelResponse, completion

from agent_tools.storage.db_tools import read, write
from agent_tools.tools_utils import ToolFunction
from schemas.schemas import ModelConfiguration

load_dotenv()


class Agent:
    def __init__(self, tools: list[ToolFunction]):
        self.tools = tools
        self._tools_by_name = {tool.tool_schema["function"]["name"]: tool for tool in tools}
        self.config = self.get_configuration()

    def get_configuration(self) -> ModelConfiguration:
        return ModelConfiguration()  # type: ignore[call-arg]

    def _completion(self, **kwargs) -> Message:
        response = completion(**kwargs)
        assert isinstance(response, ModelResponse)
        print(f"[COMPLETE] : {json.dumps(response.model_dump(), indent=2)}")
        return response.choices[0].message

    def completion(self, prompt: str, chat_history: list = [], tools: list[ToolFunction] = []) -> Message:
        # TODO: Add system prompt
        # If tools are provided in the call, use them. Otherwise, use the agent's tools.
        tools = tools or self.tools
        tools_by_name = {tool.tool_schema["function"]["name"]: tool for tool in tools} if tools else self._tools_by_name

        chat_history.append({"role": "user", "content": prompt})
        message = self._completion(
            model=self.config.model_name,
            messages=chat_history,
            tools=[tool.tool_schema for tool in tools],
        )
        chat_history.append(message)

        iteration = 0
        while iteration < self.config.max_iterations:
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
                    model=self.config.model_name,
                    messages=chat_history,
                    tools=[tool.tool_schema for tool in tools],
                )
                chat_history.append(message)
            else:
                break
            iteration += 1
            if iteration >= self.config.max_iterations:
                print("Reached maximum number of iterations. Stopping.")
        return message


if __name__ == "__main__":
    import json

    agent = Agent(tools=[read, write])
    input = "hi"  # "Use the write tool tool once. Just test that you can call them."
    answer = agent.completion(input, tools=[read, write])
    print(json.dumps(answer.model_dump(), indent=2))
