"""An LLMClient backed by any LangChain BaseChatModel.

One adapter, five providers (Anthropic, OpenAI, Gemini, Ollama, Cerebras),
selected via `create_llm_client(provider, model, **kwargs)` or by calling a
provider's factory function directly. `PromptSegment.cacheable` (see
agents/llm.py) only matters to one of the five, because caching itself
isn't uniform:

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
- Cerebras: automatic and implicit — the platform hashes each request in
  128-token blocks and reuses any block matching a recent request, no
  opt-in required. Segments are concatenated; putting the stable content
  first (which PromptSegment already does) is what makes the prefix match.
- Ollama: also automatic and implicit, via the underlying llama.cpp
  engine's KV-cache reuse for requests sharing an exact byte-identical
  prefix — same principle, but it only holds while the model stays loaded
  in memory. Ollama's default is to unload after 5 minutes of idling,
  which would silently evict the cache mid-game, so `ollama_client`
  defaults `keep_alive` to a longer duration (override via kwargs).

Requires the optional `langchain` dependency group:
`uv sync --extra langchain`.
"""

from typing import Any, TypeVar

from pydantic import BaseModel

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "LangChainLLMClient requires the 'langchain' optional dependency group. "
        "Install it with: uv sync --extra langchain"
    ) from exc

from agents.llm import PromptSegment, parse_model

ModelT = TypeVar("ModelT", bound=BaseModel)

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

    def __init__(
        self, model: BaseChatModel, metadata: dict[str, Any] | None = None
    ) -> None:
        self.model = model
        # Seat-level identity (run id, player, provider, model), fixed for
        # the life of this client and merged into every call's config.
        self.metadata = dict(metadata or {})

    def _config(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        """The `RunnableConfig` that tells LangSmith who made this call.

        LangSmith records a run's `metadata` and `tags` and lets you filter
        and group on them, which is the difference between a trace export you
        can compare models with and an anonymous pile of chat completions —
        the earlier exports carried nothing but the project name, so which
        seat used which model had to be reconstructed by reading prompts.

        `run_name` is set for the same reason: without it every row in the
        LangSmith run list reads "ChatAnthropic".
        """
        merged = {**self.metadata, **(metadata or {})}
        if not merged:
            return {}

        config: dict[str, Any] = {"metadata": merged}
        config["tags"] = [
            f"{key.removeprefix('clue_')}:{value}"
            for key, value in merged.items()
            if key in _TAGGED_KEYS
        ]
        name_parts = [
            merged.get("clue_decision"),
            f"p{merged['clue_player_id']}" if "clue_player_id" in merged else None,
            f"t{merged['clue_turn']}" if "clue_turn" in merged else None,
        ]
        named = " ".join(part for part in name_parts if part)
        if named:
            config["run_name"] = named
        return config

    async def complete(
        self, segments: list[PromptSegment], metadata: dict[str, Any] | None = None
    ) -> str:
        messages = build_messages(self.model, segments)
        response = await self.model.ainvoke(messages, config=self._config(metadata))
        return _text_of(response.content)

    async def complete_structured(
        self,
        segments: list[PromptSegment],
        schema: type[ModelT],
        metadata: dict[str, Any] | None = None,
    ) -> ModelT:
        """Constrains the response to `schema` via the provider's own
        structured-output mode, falling back to parsing free text.

        Asking for JSON in the prompt and hoping is not reliable: across the
        traced game, 28% of responses came back fenced in markdown or with a
        paragraph of reasoning in front, all of which had to be salvaged by a
        regex. Binding the schema removes the failure mode at the source.

        The fallback exists because "supports structured output" is a
        property of the specific model, not just the provider — a small local
        model behind Ollama may not, and degrading to the text path is much
        better than failing the seat outright.
        """
        messages = build_messages(self.model, segments)
        config = self._config(metadata)

        try:
            structured = self.model.with_structured_output(schema)
        except (AttributeError, NotImplementedError, ValueError):
            structured = None

        if structured is not None:
            try:
                result = await structured.ainvoke(messages, config=config)
            except (NotImplementedError, ValueError):
                result = None
            if isinstance(result, schema):
                return result

        response = await self.model.ainvoke(messages, config=config)
        return parse_model(_text_of(response.content), schema)


def anthropic_client(
    model: str, trace_metadata: dict[str, Any] | None = None, **kwargs: Any
) -> LangChainLLMClient:
    from langchain_anthropic import ChatAnthropic

    return LangChainLLMClient(ChatAnthropic(model=model, **kwargs), trace_metadata)


def openai_client(
    model: str, trace_metadata: dict[str, Any] | None = None, **kwargs: Any
) -> LangChainLLMClient:
    from langchain_openai import ChatOpenAI

    return LangChainLLMClient(ChatOpenAI(model=model, **kwargs), trace_metadata)


def gemini_client(
    model: str, trace_metadata: dict[str, Any] | None = None, **kwargs: Any
) -> LangChainLLMClient:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return LangChainLLMClient(ChatGoogleGenerativeAI(model=model, **kwargs), trace_metadata)


def ollama_client(
    model: str, trace_metadata: dict[str, Any] | None = None, **kwargs: Any
) -> LangChainLLMClient:
    from langchain_ollama import ChatOllama

    kwargs.setdefault("keep_alive", "30m")
    return LangChainLLMClient(ChatOllama(model=model, **kwargs), trace_metadata)


def cerebras_client(
    model: str, trace_metadata: dict[str, Any] | None = None, **kwargs: Any
) -> LangChainLLMClient:
    from langchain_cerebras import ChatCerebras

    return LangChainLLMClient(ChatCerebras(model=model, **kwargs), trace_metadata)


# Metadata keys that are also worth having as LangSmith tags. Tags are the
# cheap filter in the UI ("show me everything seat 2 did"); the full metadata
# dict is what you group and aggregate on afterwards. Turn number is
# deliberately absent — one tag per turn would be hundreds per run.
_TAGGED_KEYS = frozenset(
    {"clue_run_id", "clue_player_id", "clue_provider", "clue_model", "clue_decision"}
)

_PROVIDER_FACTORIES = {
    "anthropic": anthropic_client,
    "openai": openai_client,
    "gemini": gemini_client,
    "ollama": ollama_client,
    "cerebras": cerebras_client,
}


def create_llm_client(
    provider: str, model: str, trace_metadata: dict[str, Any] | None = None, **kwargs: Any
) -> LangChainLLMClient:
    """Single entry point for all five providers: create_llm_client("anthropic", "claude-sonnet-5").

    Prefer this over calling a provider's factory directly when the
    provider itself is a variable in your experiment (e.g. sweeping
    providers/models from a config file or CLI flag).

    `trace_metadata` is attached to every call this client makes; it is
    separate from `**kwargs` (which go to the provider's constructor)
    precisely so a metadata key can never be mistaken for a model parameter.
    """
    try:
        factory = _PROVIDER_FACTORIES[provider]
    except KeyError:
        available = ", ".join(sorted(_PROVIDER_FACTORIES))
        raise ValueError(f"Unknown provider {provider!r}. Available: {available}") from None
    identity = {"clue_provider": provider, "clue_model": model, **(trace_metadata or {})}
    return factory(model, identity, **kwargs)
