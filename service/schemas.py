from typing import Literal

from pydantic import BaseModel, Field, model_validator

from models.actions import PlayerAction
from models.card import Card, PersonValue, RoomValue, WeaponValue
from models.events import Suggestion
from models.observation import PlayerObservation


class SeatConfig(BaseModel):
    """One player seat's controller: the no-LLM RandomClueAgent, an
    LLMClueAgent backed by `create_llm_client(provider, model)`, or a
    HumanClueAgent answered by a person over HTTP.

    The generation knobs are provider-neutral and optional; `agent_factory`
    translates them into whatever each SDK actually calls them, and leaves
    any left as `None` at the provider's own default.

    `thinking_budget` is the one worth setting deliberately. Left unset, a
    reasoning model will spend as much as it likes on a decision with three
    possible shapes: in the traced game one seat averaged 6,643 output tokens
    a turn, 99% of it reasoning, and accounted for 80% of the run's output
    tokens while taking 20% of the turns. `0` disables extended thinking
    where the provider supports disabling it.

    A human seat ignores every LLM knob — there's no model to configure —
    and instead takes `name` (shown on its seat card and join link) and an
    optional `decision_timeout` (seconds a person has to answer before the
    seat auto-passes/auto-reveals; `None` waits forever, useful for testing
    without a real person walking away mid-game).
    """

    agent_type: Literal["random", "llm", "human"]
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None
    scratchpad: bool = False
    name: str | None = None
    decision_timeout: float | None = None

    @model_validator(mode="after")
    def _llm_seats_need_provider_and_model(self) -> "SeatConfig":
        if self.agent_type == "llm" and not (self.provider and self.model):
            raise ValueError("agent_type 'llm' requires both 'provider' and 'model'")
        if self.thinking_budget is not None and self.thinking_budget < 0:
            raise ValueError("thinking_budget must be zero or positive")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        return self

    @model_validator(mode="after")
    def _human_seats_have_no_model_knobs(self) -> "SeatConfig":
        if self.agent_type == "human":
            if self.provider is not None or self.model is not None:
                raise ValueError("agent_type 'human' may not set 'provider' or 'model'")
            if self.thinking_budget is not None:
                raise ValueError("agent_type 'human' may not set 'thinking_budget'")
            if self.scratchpad:
                raise ValueError("agent_type 'human' may not set 'scratchpad'")
        return self

    @model_validator(mode="after")
    def _decision_timeout_must_be_positive(self) -> "SeatConfig":
        if self.decision_timeout is not None and self.decision_timeout <= 0:
            raise ValueError("decision_timeout must be positive")
        return self


class CreateGameRequest(BaseModel):
    n_players: int
    seats: list[SeatConfig]
    max_turns: int = 500


class HumanSeatCredential(BaseModel):
    """A bearer credential for one human seat, minted at game creation.

    `join_url` is the whole point: pasting it into a browser (or sharing it
    with whoever is playing that seat) is enough to reach the game screen
    for exactly that seat, with no separate login step.
    """

    player_id: int
    name: str | None
    token: str
    join_url: str


class CreateGameResponse(BaseModel):
    game_id: str
    run_id: str
    stream_url: str
    human_seats: list[HumanSeatCredential] = Field(default_factory=list)


class WaitingOn(BaseModel):
    player_id: int
    kind: Literal["action", "reveal"]


class GameStatusResponse(BaseModel):
    game_id: str
    run_id: str
    status: str
    turn_number: int
    current_player_id: int
    winning_player_id: int | None
    seats: list[SeatConfig]
    # Set only while the run is genuinely parked waiting on a human seat, so
    # a spectator can see the game is stalled on a person without holding
    # that seat's token.
    waiting_on: WaitingOn | None = None


class ProviderInfo(BaseModel):
    name: str
    requires_key: str | None
    available: bool


class PendingDecisionView(BaseModel):
    """What a human seat's panel needs to render a decision form.

    `suggestion` / `suggesting_player_id` / `offered` are only populated for
    a `"reveal"` decision — the front-end needs them to render "player 2
    suggested X/Y/Z, which of these do you show them?" without a second
    request.
    """

    kind: Literal["action", "reveal"]
    turn_number: int
    suggestion: Suggestion | None = None
    suggesting_player_id: int | None = None
    offered: list[Card] = Field(default_factory=list)


class PlayerView(BaseModel):
    """The response to `GET /games/{id}/players/{pid}`: this seat's own
    `PlayerObservation` (hand, card log, event history it's entitled to,
    public deductions), plus whatever decision is currently pending for it,
    if any.
    """

    observation: PlayerObservation
    pending: PendingDecisionView | None = None
    status: str


# The body of `POST /games/{id}/players/{pid}/action` is just a `PlayerAction`
# — aliased here so the endpoint signature reads in terms of the service's
# own request/response vocabulary rather than reaching into `models` inline.
SubmitActionRequest = PlayerAction


class SubmitRevealRequest(BaseModel):
    card: Card


class CardCatalog(BaseModel):
    """Every legal card value, so the front-end never hardcodes names."""

    people: list[PersonValue]
    weapons: list[WeaponValue]
    rooms: list[RoomValue]
