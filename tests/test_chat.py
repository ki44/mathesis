from unittest.mock import patch

from main import agent
from tests.conftest import fetch_conversation_history, insert_conversation


async def _noop_stream(prompt, chat_history=None):
    """Simulates the user-message append that agent._initialization() does, without LLM calls."""
    if chat_history is not None:
        chat_history.append({"role": "user", "content": prompt})
    if False:  # makes this an async generator without any yields
        yield  # noqa: unreachable


async def test_chat_stream_no_variant_override(client):
    """Normal send: history is not modified before the new user turn is appended."""
    conv_id = "conv-normal"
    initial_history = [
        {"role": "user", "content": "turn1"},
        {"role": "assistant", "content": "answer1"},
    ]
    await insert_conversation(conv_id, initial_history)

    with patch.object(agent, "stream", new=_noop_stream):
        await client.post("/api/chat/stream", json={"message": "turn2", "conversation_id": conv_id})

    history = await fetch_conversation_history(conv_id)
    roles_and_contents = [(m["role"], m["content"]) for m in history]
    assert ("user", "turn1") in roles_and_contents
    assert ("assistant", "answer1") in roles_and_contents
    assert ("user", "turn2") in roles_and_contents


async def test_chat_stream_variant_override(client):
    """variant_override replaces the last response with the selected variant before continuing."""
    conv_id = "conv-variant"
    # DB has variant2 saved (the most recently run response)
    initial_history = [
        {"role": "user", "content": "turn5"},
        {"role": "assistant", "content": "variant2_response"},
    ]
    await insert_conversation(conv_id, initial_history)

    with patch.object(agent, "stream", new=_noop_stream):
        await client.post(
            "/api/chat/stream",
            json={
                "message": "new_msg",
                "conversation_id": conv_id,
                "variant_override": [{"role": "assistant", "content": "variant1_response"}],
            },
        )

    history = await fetch_conversation_history(conv_id)
    contents = [m["content"] for m in history]

    # Variant 2 is gone; variant 1 is now the permanent context
    assert "variant2_response" not in contents
    assert "variant1_response" in contents

    # Ordering: turn5 user → variant1 override → new_msg user
    idx = {c: i for i, c in enumerate(contents)}
    assert idx["turn5"] < idx["variant1_response"] < idx["new_msg"]


async def test_chat_stream_rerun(client):
    """rerun=True strips the last user turn from history so the agent replays from a clean slate."""
    conv_id = "conv-rerun"
    initial_history = [
        {"role": "user", "content": "turn4"},
        {"role": "assistant", "content": "answer4"},
        {"role": "user", "content": "turn5"},
        {"role": "assistant", "content": "answer5"},
    ]
    await insert_conversation(conv_id, initial_history)

    with patch.object(agent, "stream", new=_noop_stream):
        await client.post(
            "/api/chat/stream",
            json={"message": "turn5", "conversation_id": conv_id, "rerun": True},
        )

    history = await fetch_conversation_history(conv_id)
    contents = [m["content"] for m in history]

    # answer5 is gone — the last user turn was stripped before re-running
    assert "answer5" not in contents
    # Prior context is preserved
    assert "turn4" in contents
    assert "answer4" in contents
    # rerun strips the original turn5 (and answer5), then the mock appends it once
    assert contents.count("turn5") == 1
