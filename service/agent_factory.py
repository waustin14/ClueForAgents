"""Turns a `SeatConfig` into a live `ClueAgent`, and reports what's usable.

`PROVIDER_KEY_ENV` names, for each LangChain provider, the environment
variable `create_llm_client` needs populated for that provider to actually
work (`None` for `ollama`, which talks to a local server and needs no key).
It's used both to build `GET /providers` (so the front-end can grey out a
provider whose key isn't set, before the tester wastes a turn on it) and to
give `build_agent` a clear error message up front instead of a confusing
failure from deep inside the provider SDK.
"""

import os
from typing import Any

from agents.base import ClueAgent
from agents.random_agent import RandomClueAgent
from service.schemas import ProviderInfo, SeatConfig

PROVIDER_KEY_ENV: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "ollama": None,
}

# Anthropic requires the response cap to leave room for the thinking budget
# plus the answer itself, so a seat that sets a budget and no max_tokens
# still gets a workable ceiling.
_ANTHROPIC_ANSWER_HEADROOM = 1024

# OpenAI's reasoning models take an effort level, not a token budget. These
# thresholds map one onto the other so a single `thinking_budget` on the seat
# means roughly the same thing whichever provider is behind it.
_OPENAI_EFFORT_THRESHOLDS = ((1, "minimal"), (2048, "low"), (8192, "medium"))


def _openai_reasoning_effort(budget: int) -> str:
    for threshold, effort in _OPENAI_EFFORT_THRESHOLDS:
        if budget < threshold:
            return effort
    return "high"


def model_kwargs_for(provider: str, seat: SeatConfig) -> dict[str, object]:
    """Translates a seat's provider-neutral knobs into one SDK's parameters.

    Only keys the seat actually set are returned, so anything left unset
    stays at the provider default rather than being pinned to a value this
    layer invented.
    """
    kwargs: dict[str, object] = {}
    budget = seat.thinking_budget

    if seat.temperature is not None:
        kwargs["temperature"] = seat.temperature

    if provider == "anthropic":
        max_tokens = seat.max_tokens
        if budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            max_tokens = max(max_tokens or 0, budget + _ANTHROPIC_ANSWER_HEADROOM)
            # Anthropic rejects a temperature other than 1 while extended
            # thinking is on, so a seat asking for both loses the temperature.
            kwargs.pop("temperature", None)
        elif budget == 0:
            kwargs["thinking"] = {"type": "disabled"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
    elif provider == "gemini":
        if budget is not None:
            kwargs["thinking_budget"] = budget
        if seat.max_tokens is not None:
            kwargs["max_output_tokens"] = seat.max_tokens
    elif provider == "openai":
        if budget is not None:
            kwargs["reasoning_effort"] = _openai_reasoning_effort(budget)
        if seat.max_tokens is not None:
            kwargs["max_tokens"] = seat.max_tokens
    elif provider == "ollama":
        if seat.max_tokens is not None:
            kwargs["num_predict"] = seat.max_tokens
    else:  # cerebras and anything else added later
        if seat.max_tokens is not None:
            kwargs["max_tokens"] = seat.max_tokens

    return kwargs


def list_providers() -> list[ProviderInfo]:
    providers = [ProviderInfo(name="random", requires_key=None, available=True)]
    for name, key_env in PROVIDER_KEY_ENV.items():
        providers.append(
            ProviderInfo(
                name=name,
                requires_key=key_env,
                available=key_env is None or bool(os.environ.get(key_env)),
            )
        )
    return providers


def seat_identity(player_id: int, seat: SeatConfig, run_id: str | None) -> dict[str, Any]:
    """What was seated here, in a form both trace backends can carry.

    The same dict goes to LangSmith (as run metadata, via the client) and to
    OpenTelemetry (as span attributes, via the agent), because the question
    it answers is the same in both: which model played which seat, under
    which knobs. Reconstructing that from prompt text after the fact — the
    only option in the earlier exports — does not scale past one game.

    Keys use `clue_` underscores rather than dots because LangSmith metadata
    filtering is happier with them; `build_agent` converts to `clue.` for
    OTel, where dotted namespacing is the convention.
    """
    identity: dict[str, Any] = {
        "clue_player_id": player_id,
        "clue_agent_type": seat.agent_type,
    }
    if run_id:
        identity["clue_run_id"] = run_id
    if seat.agent_type == "llm":
        identity["clue_provider"] = seat.provider
        identity["clue_model"] = seat.model
        identity["clue_scratchpad"] = seat.scratchpad
        if seat.thinking_budget is not None:
            identity["clue_thinking_budget"] = seat.thinking_budget
        if seat.temperature is not None:
            identity["clue_temperature"] = seat.temperature
    return identity


def _span_attributes(identity: dict[str, Any]) -> dict[str, Any]:
    """`clue_provider` -> `clue.provider`, for OTel's dotted convention."""
    return {key.replace("_", ".", 1): value for key, value in identity.items()}


def build_agent(player_id: int, seat: SeatConfig, run_id: str | None = None) -> ClueAgent:
    identity = seat_identity(player_id, seat, run_id)

    if seat.agent_type == "random":
        return RandomClueAgent(player_id, _span_attributes(identity))

    if seat.provider not in PROVIDER_KEY_ENV:
        available = ", ".join(sorted(PROVIDER_KEY_ENV))
        raise ValueError(f"Unknown provider {seat.provider!r}. Available: {available}")

    key_env = PROVIDER_KEY_ENV[seat.provider]
    if key_env and not os.environ.get(key_env):
        raise ValueError(f"Provider {seat.provider!r} requires {key_env} to be set")

    try:
        # Imported lazily so the service still runs with random-only seats
        # when the optional `langchain` extra isn't installed.
        from agents.langchain_client import create_llm_client
        from agents.llm import LLMClueAgent
    except ImportError as exc:
        raise ValueError(
            "LLM seats require the 'langchain' extra: uv sync --extra langchain"
        ) from exc

    client = create_llm_client(
        seat.provider,
        seat.model,
        trace_metadata=identity,
        **model_kwargs_for(seat.provider, seat),
    )
    return LLMClueAgent(
        player_id,
        client,
        use_scratchpad=seat.scratchpad,
        trace_attributes=_span_attributes(identity),
    )
