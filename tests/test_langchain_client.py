import asyncio

import pytest

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402

from agents.langchain_client import (  # noqa: E402
    LangChainLLMClient,
    anthropic_client,
    build_messages,
    cerebras_client,
    create_llm_client,
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


@pytest.mark.parametrize(
    "provider, model, env",
    [
        ("anthropic", "claude-sonnet-5", {"ANTHROPIC_API_KEY": "test-key"}),
        ("openai", "gpt-4o-mini", {"OPENAI_API_KEY": "test-key"}),
        ("gemini", "gemini-2.5-flash", {"GOOGLE_API_KEY": "test-key"}),
        ("ollama", "llama3", {}),
        ("cerebras", "llama-3.3-70b", {"CEREBRAS_API_KEY": "test-key"}),
    ],
)
def test_create_llm_client_dispatches_to_every_provider(monkeypatch, provider, model, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    client = create_llm_client(provider, model)

    assert isinstance(client, LangChainLLMClient)


def test_create_llm_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_llm_client("watsonx", "some-model")


def test_ollama_client_defaults_keep_alive_to_avoid_evicting_the_kv_cache():
    client = ollama_client("llama3")

    assert client.model.keep_alive == "30m"


def test_ollama_client_keep_alive_override_is_respected():
    client = ollama_client("llama3", keep_alive="2h")

    assert client.model.keep_alive == "2h"


class FakeChatModel:
    """Duck-typed stand-in for a BaseChatModel — no network, no SDK."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.received_messages = None
        self.received_config = None

    async def ainvoke(self, messages, config=None):
        self.received_messages = messages
        self.received_config = config
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


def test_every_call_tells_langsmith_which_seat_and_model_made_it():
    # Exports before this carried nothing but the project name, so which
    # model played which seat had to be reconstructed from prompt text.
    model = FakeChatModel("ok")
    client = LangChainLLMClient(
        model, {"clue_run_id": "run-abc", "clue_player_id": 2, "clue_model": "claude-sonnet-5"}
    )

    run(client.complete([PromptSegment("hi")], {"clue_decision": "action", "clue_turn": 7}))

    config = model.received_config
    assert config["metadata"] == {
        "clue_run_id": "run-abc",
        "clue_player_id": 2,
        "clue_model": "claude-sonnet-5",
        "clue_decision": "action",
        "clue_turn": 7,
    }
    # Tags are the cheap filter in the UI; the turn number is deliberately
    # not one of them, or a run would carry hundreds of tags.
    assert set(config["tags"]) == {
        "run_id:run-abc",
        "player_id:2",
        "model:claude-sonnet-5",
        "decision:action",
    }
    # Otherwise every row in the LangSmith run list reads "FakeChatModel".
    assert config["run_name"] == "action p2 t7"


def test_a_client_with_nothing_to_say_sends_no_config():
    model = FakeChatModel("ok")

    run(LangChainLLMClient(model).complete([PromptSegment("hi")]))

    assert model.received_config == {}


def test_create_llm_client_records_the_provider_and_model_it_was_asked_for(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    client = create_llm_client("openai", "gpt-5", trace_metadata={"clue_player_id": 3})

    assert client.metadata == {
        "clue_provider": "openai",
        "clue_model": "gpt-5",
        "clue_player_id": 3,
    }
