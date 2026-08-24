"""An LLMClient backed by any LangChain BaseChatModel.

One adapter, five providers (Anthropic, OpenAI, Gemini, Ollama, Cerebras),
selected by which factory function you call. `PromptSegment.cacheable`
(see agents/llm.py) is honored per-provider, not uniformly, because prompt
caching itself isn't uniform across providers:

- Anthropic: explicit opt-in. Cacheable segments become separate content
  blocks in the system message, with `cache_control` attached to the last
  one — Anthropic caches everything up through that block.
- OpenAI: the platform caches matching prefixes >=1024 tokens
  automatically. There's nothing to annotate; PromptSegments are just
  concatenated.
- Gemini (ChatGoogleGenerativeAI): explicit caching there is a separate
  stored `CachedContent` resource referenced by ID, not an inline
  per-message flag — a fundamentally different shape this adapter does not
  attempt to bridge. Segments are concatenated like OpenAI.
- Ollama / Cerebras: no documented prompt-prefix caching to opt into.
  Segments are concatenated.

Requires the optional `langchain` dependency group:
`uv sync --extra langchain`.
"""

from typing import Any

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "LangChainLLMClient requires the 'langchain' optional dependency group. "
        "Install it with: uv sync --extra langchain"
    ) from exc

from agents.llm import PromptSegment

_ANTHROPIC_CACHE_CONTROL = {"type": "ephemeral"}


def _is_anthropic(model: BaseChatModel) -> bool:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return False
    return isinstance(model, ChatAnthropic)


def _system_message_for(model: BaseChatModel, cacheable: list[PromptSegment]) -> SystemMessage:
    if not _is_anthropic(model):
        return SystemMessage(content="\n\n".join(segment.text for segment in cacheable))

    blocks: list[dict] = [{"type": "text", "text": segment.text} for segment in cacheable]
    if blocks:
        blocks[-1] = {**blocks[-1], "cache_control": _ANTHROPIC_CACHE_CONTROL}
    return SystemMessage(content=blocks)


def build_messages(model: BaseChatModel, segments: list[PromptSegment]) -> list[BaseMessage]:
    cacheable = [segment for segment in segments if segment.cacheable]
    volatile = [segment for segment in segments if not segment.cacheable]
    return [
        _system_message_for(model, cacheable),
        HumanMessage(content="\n\n".join(segment.text for segment in volatile)),
    ]


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


class LangChainLLMClient:
    """Adapts a LangChain BaseChatModel to this project's LLMClient protocol."""

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    async def complete(self, segments: list[PromptSegment]) -> str:
        messages = build_messages(self.model, segments)
        response = await self.model.ainvoke(messages)
        return _text_of(response.content)


def anthropic_client(model: str, **kwargs: Any) -> LangChainLLMClient:
    from langchain_anthropic import ChatAnthropic

    return LangChainLLMClient(ChatAnthropic(model=model, **kwargs))


def openai_client(model: str, **kwargs: Any) -> LangChainLLMClient:
    from langchain_openai import ChatOpenAI

    return LangChainLLMClient(ChatOpenAI(model=model, **kwargs))


def gemini_client(model: str, **kwargs: Any) -> LangChainLLMClient:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return LangChainLLMClient(ChatGoogleGenerativeAI(model=model, **kwargs))


def ollama_client(model: str, **kwargs: Any) -> LangChainLLMClient:
    from langchain_ollama import ChatOllama

    return LangChainLLMClient(ChatOllama(model=model, **kwargs))


def cerebras_client(model: str, **kwargs: Any) -> LangChainLLMClient:
    from langchain_cerebras import ChatCerebras

    return LangChainLLMClient(ChatCerebras(model=model, **kwargs))
