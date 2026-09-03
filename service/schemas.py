from typing import Literal

from pydantic import BaseModel, model_validator


class SeatConfig(BaseModel):
    """One player seat's controller: either the no-LLM RandomClueAgent, or
    an LLMClueAgent backed by `create_llm_client(provider, model)`.

    The generation knobs are provider-neutral and optional; `agent_factory`
    translates them into whatever each SDK actually calls them, and leaves
    any left as `None` at the provider's own default.

    `thinking_budget` is the one worth setting deliberately. Left unset, a
    reasoning model will spend as much as it likes on a decision with three
    possible shapes: in the traced game one seat averaged 6,643 output tokens
    a turn, 99% of it reasoning, and accounted for 80% of the run's output
    tokens while taking 20% of the turns. `0` disables extended thinking
    where the provider supports disabling it.
    """

    agent_type: Literal["random", "llm"]
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None
    scratchpad: bool = False

    @model_validator(mode="after")
    def _llm_seats_need_provider_and_model(self) -> "SeatConfig":
        if self.agent_type == "llm" and not (self.provider and self.model):
            raise ValueError("agent_type 'llm' requires both 'provider' and 'model'")
        if self.thinking_budget is not None and self.thinking_budget < 0:
            raise ValueError("thinking_budget must be zero or positive")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        return self


class CreateGameRequest(BaseModel):
    n_players: int
    seats: list[SeatConfig]
    max_turns: int = 500


class CreateGameResponse(BaseModel):
    game_id: str
    run_id: str
    stream_url: str


class GameStatusResponse(BaseModel):
    game_id: str
    run_id: str
    status: str
    turn_number: int
    current_player_id: int
    winning_player_id: int | None
    seats: list[SeatConfig]


class ProviderInfo(BaseModel):
    name: str
    requires_key: str | None
    available: bool
