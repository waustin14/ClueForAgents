import asyncio

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402

from agents.langchain_client import (  # noqa: E402
    LangChainLLMClient,
    anthropic_client,
    build_messages,
    cerebras_client,
    gemini_client,
    ollama_client,
    openai_client,
)
from agents.llm import PromptSegment  # noqa: E402


def run(coro):
    return asyncio.run(coro)


SEGMENTS = [
    PromptSegment("INSTRUCTIONS", cacheable=True),
    PromptSegment("HAND", cacheable=True),
    PromptSegment("HISTORY", cacheable=True),
    PromptSegment("VOLATILE turn info", cacheable=False),
]


def test_anthropic_gets_cache_control_on_last_cacheable_block(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = anthropic_client("claude-sonnet-5")

    messages = build_messages(client.model, SEGMENTS)

    system_blocks = messages[0].content
    assert system_blocks == [
        {"type": "text", "text": "INSTRUCTIONS"},
        {"type": "text", "text": "HAND"},
        {"type": "text", "text": "HISTORY", "cache_control": {"type": "ephemeral"}},
    ]
    assert messages[1].content == "VOLATILE turn info"


@pytest.mark.parametrize(
    "factory, model, env",
    [
        (openai_client, "gpt-4o-mini", {"OPENAI_API_KEY": "test-key"}),
        (gemini_client, "gemini-2.5-flash", {"GOOGLE_API_KEY": "test-key"}),
        (ollama_client, "llama3", {}),
        (cerebras_client, "llama-3.3-70b", {"CEREBRAS_API_KEY": "test-key"}),
    ],
)
def test_non_anthropic_providers_get_plain_concatenated_system_text(monkeypatch, factory, model, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    client = factory(model)

    messages = build_messages(client.model, SEGMENTS)

    assert messages[0].content == "INSTRUCTIONS\n\nHAND\n\nHISTORY"
    assert messages[1].content == "VOLATILE turn info"


class FakeChatModel:
    """Duck-typed stand-in for a BaseChatModel — no network, no SDK."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.received_messages = None

    async def ainvoke(self, messages):
        self.received_messages = messages
        return AIMessage(content=self.reply)


def test_complete_returns_plain_string_response():
    fake = FakeChatModel(reply='{"kind": "pass"}')
    client = LangChainLLMClient(fake)

    result = run(client.complete(SEGMENTS))

    assert result == '{"kind": "pass"}'
    assert fake.received_messages is not None


def test_complete_flattens_list_of_text_content_blocks():
    fake = FakeChatModel(reply=[{"type": "text", "text": '{"kind": "pass"}'}])
    client = LangChainLLMClient(fake)

    result = run(client.complete(SEGMENTS))

    assert result == '{"kind": "pass"}'
