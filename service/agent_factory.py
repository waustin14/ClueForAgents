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
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from agents.base import ClueAgent
from agents.human import HumanClueAgent, PendingDecision
from agents.random_agent import RandomClueAgent
from service.schemas import ProviderInfo, SeatConfig

OnDecisionRequest = Callable[[PendingDecision], Awaitable[None]]

PROVIDER_KEY_ENV: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "ollama": None,
}

# Anthropic requires the response cap to leave room for the thinking budget
# plus the answer itself, so a seat that sets a budget and no max_tokens
# still gets a workable ceiling. Only applies to the pre-4.6 models that
# still take a token budget at all — see `_anthropic_thinking_mode`.
_ANTHROPIC_ANSWER_HEADROOM = 1024

# Claude 4.6 and later dropped the fixed thinking budget: `thinking` is
# `{"type": "adaptive"}` and depth is steered by `output_config.effort`
# instead. Sending `{"type": "enabled", "budget_tokens": N}` to those models
# is a 400 ("Use thinking.type.adaptive and output_config.effort"), which is
# exactly what a Sonnet 5 seat hit in a traced game. These thresholds map the
# seat's provider-neutral `thinking_budget` onto an effort level the same way
# `_OPENAI_EFFORT_THRESHOLDS` does for OpenAI, so one number on the seat
# still means roughly the same thing across the table.
_ANTHROPIC_EFFORT_THRESHOLDS = ((2048, "low"), (8192, "medium"), (32768, "high"))

# Model names come in two shapes: family-first (`claude-sonnet-4-6`,
# `claude-opus-5`) and the older version-first (`claude-3-5-sonnet-20241022`).
# Both patterns capture (major, minor) so the cutover below is one comparison.
_ANTHROPIC_FAMILY_FIRST = re.compile(r"claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d+))?")
_ANTHROPIC_VERSION_FIRST = re.compile(r"claude-(\d+)(?:-(\d+))?-(?:opus|sonnet|haiku)")
_ANTHROPIC_ADAPTIVE_SINCE = (4, 6)


def _anthropic_effort(budget: int) -> str:
    for threshold, effort in _ANTHROPIC_EFFORT_THRESHOLDS:
        if budget < threshold:
            return effort
    return "xhigh"


def _anthropic_version(model: str) -> tuple[int, int] | None:
    for pattern in (_ANTHROPIC_FAMILY_FIRST, _ANTHROPIC_VERSION_FIRST):
        match = pattern.search(model)
        if match:
            return int(match.group(1)), int(match.group(2) or 0)
    return None


def _anthropic_thinking_mode(model: str) -> Literal["budget", "adaptive", "always_on"]:
    """Which thinking API a Claude model speaks.

    - `budget`: pre-4.6 models (Sonnet 4.5, Haiku 4.5, Opus 4.5 and older)
      take `thinking.type=enabled` with a `budget_tokens` cap.
    - `adaptive`: 4.6 and later, Sonnet 5, Opus 5 — `thinking.type=adaptive`
      plus `output_config.effort`; `budget_tokens` is rejected outright.
    - `always_on`: Fable / Mythos — thinking cannot be disabled or given a
      budget at all, so only `effort` is ever sent.

    A name that matches neither pattern is treated as current (`adaptive`):
    a brand-new model is far more likely to speak the current API than the
    one being retired, and guessing wrong the other way is the 400 that
    started this.
    """
    name = model.lower()
    if "fable" in name or "mythos" in name:
        return "always_on"
    version = _anthropic_version(name)
    if version is not None and version < _ANTHROPIC_ADAPTIVE_SINCE:
        return "budget"
    return "adaptive"


def _anthropic_sampling_supported(model: str) -> bool:
    """Opus 4.7+, Sonnet 5, Opus 5 and Fable reject `temperature` / `top_p`
    outright (400), not just alongside thinking. 4.6 and older still take
    them."""
    if _anthropic_thinking_mode(model) == "always_on":
        return False
    version = _anthropic_version(model.lower())
    return version is not None and version <= _ANTHROPIC_ADAPTIVE_SINCE

# OpenAI's reasoning models take an effort level, not a token budget. These
# thresholds map one onto the other so a single `thinking_budget` on the seat
# means roughly the same thing whichever provider is behind it.
_OPENAI_EFFORT_THRESHOLDS = ((1, "minimal"), (2048, "low"), (8192, "medium"))


def _openai_reasoning_effort(budget: int) -> str:
    for threshold, effort in _OPENAI_EFFORT_THRESHOLDS:
        if budget < threshold:
            return effort
    return "high"


def _anthropic_kwargs(model: str, seat: SeatConfig) -> dict[str, object]:
    """The Anthropic half of `model_kwargs_for`, split by thinking API.

    `effort` is `ChatAnthropic`'s own field (an alias for `reasoning_effort`)
    and lands on the wire as `output_config.effort`; the adapter adds
    `thinking.type=adaptive` itself when effort is set on a current model.
    """
    kwargs: dict[str, object] = {}
    budget = seat.thinking_budget
    mode = _anthropic_thinking_mode(model)
    max_tokens = seat.max_tokens
    thinking_on = bool(budget) or mode == "always_on"

    if mode == "budget":
        if budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            max_tokens = max(max_tokens or 0, budget + _ANTHROPIC_ANSWER_HEADROOM)
        elif budget == 0:
            kwargs["thinking"] = {"type": "disabled"}
    elif mode == "adaptive":
        if budget:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["effort"] = _anthropic_effort(budget)
        elif budget == 0:
            kwargs["thinking"] = {"type": "disabled"}
    else:  # always_on: no `thinking` key at all — any explicit config is a 400
        if budget:
            kwargs["effort"] = _anthropic_effort(budget)
        elif budget == 0:
            # Can't be switched off; the cheapest legal setting is the
            # closest thing to what the seat asked for.
            kwargs["effort"] = "low"

    # Anthropic rejects a temperature other than 1 while thinking is on, and
    # the newest models reject sampling parameters unconditionally — either
    # way a seat asking for both loses the temperature rather than the run.
    if seat.temperature is not None and not thinking_on and _anthropic_sampling_supported(model):
        kwargs["temperature"] = seat.temperature

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def model_kwargs_for(provider: str, seat: SeatConfig) -> dict[str, object]:
    """Translates a seat's provider-neutral knobs into one SDK's parameters.

    Only keys the seat actually set are returned, so anything left unset
    stays at the provider default rather than being pinned to a value this
    layer invented.
    """
    kwargs: dict[str, object] = {}
    budget = seat.thinking_budget

    if seat.temperature is not None and provider != "anthropic":
        kwargs["temperature"] = seat.temperature

    if provider == "anthropic":
        kwargs.update(_anthropic_kwargs(seat.model or "", seat))
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
    providers = [
        ProviderInfo(name="random", requires_key=None, available=True),
        ProviderInfo(name="human", requires_key=None, available=True),
    ]
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
    elif seat.agent_type == "human" and seat.name:
        identity["clue_name"] = seat.name
    return identity


def _span_attributes(identity: dict[str, Any]) -> dict[str, Any]:
    """`clue_provider` -> `clue.provider`, for OTel's dotted convention."""
    return {key.replace("_", ".", 1): value for key, value in identity.items()}


def build_agent(
    player_id: int,
    seat: SeatConfig,
    run_id: str | None = None,
    on_decision_request: OnDecisionRequest | None = None,
) -> ClueAgent:
    identity = seat_identity(player_id, seat, run_id)

    if seat.agent_type == "random":
        return RandomClueAgent(player_id, _span_attributes(identity))

    if seat.agent_type == "human":
        return HumanClueAgent(
            player_id,
            name=seat.name,
            decision_timeout=seat.decision_timeout,
            on_request=on_decision_request,
            trace_attributes=_span_attributes(identity),
        )

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
