def to_display_messages(llm_history: list[dict]) -> list[dict]:
    """Derive frontend-displayable messages from the raw LLM chat history."""
    result = []
    for msg in llm_history:
        role = msg.get("role")
        if role == "user":
            content = msg.get("content")
            if isinstance(content, str):
                result.append({"role": "user", "content": content})
        elif role == "assistant":
            content = msg.get("content")
            if content:
                result.append({"role": "assistant", "content": content})
            for tc in msg.get("tool_calls") or []:
                name = (tc.get("function") or {}).get("name", "unknown")
                result.append({"role": "tool_call", "content": f"{name}"})
    return result


def find_history_slice_end(history: list[dict], display_index: int) -> int:
    """Return exclusive end index into history for the given 0-based display message index.

    Extends the slice past any immediately-following role='tool' messages so that an
    assistant message with tool_calls is never left without its tool results.
    """

    def _end(i: int) -> int:
        end = i + 1
        while end < len(history) and history[end].get("role") == "tool":
            end += 1
        return end

    count = 0
    for i, msg in enumerate(history):
        role = msg.get("role")
        if role == "user":
            if isinstance(msg.get("content"), str):
                if count == display_index:
                    return _end(i)
                count += 1
        elif role == "assistant":
            if msg.get("content"):
                if count == display_index:
                    return _end(i)
                count += 1
            for _ in msg.get("tool_calls") or []:
                if count == display_index:
                    return _end(i)
                count += 1
    return len(history)
