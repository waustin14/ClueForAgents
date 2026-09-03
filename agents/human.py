"""A seat driven by a person instead of a model or `RandomClueAgent`.

The engine and `service/game_runner.py::_play` only ever `await
agent.choose_action(...)` / `agent.choose_reveal(...)` — nothing in that
loop cares how long the await takes or where the answer comes from. So a
human seat is just another `ClueAgent`: `_choose_action` / `_choose_reveal`
park on an `asyncio.Future` until an HTTP handler resolves it, instead of
computing an answer synchronously the way `RandomClueAgent` and
`LLMClueAgent` do.

Deliberately no transport import here: `agents/` stays ignorant of SSE the
same way it stays ignorant of FastAPI. `on_request` is how the service
layer gets told a decision is pending — it's just an async callback, wired
up by `service/game_runner.py`.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from random import choice
from typing import Any, Literal

from opentelemetry import trace

from agents.base import ClueAgent
from models.actions import PassTurn, PlayerAction
from models.card import Card
from models.events import Suggestion
from models.observation import PlayerObservation

OnRequest = Callable[["PendingDecision"], Awaitable[None]]


@dataclass
class PendingDecision:
    """One outstanding question this seat has been asked, and the future
    that `submit_action` / `submit_reveal` resolves once a person answers.

    `suggestion` / `suggesting_player_id` / `offered` are only populated for
    a `"reveal"` decision — they're what the panel needs to render "player 2
    suggested X/Y/Z, which of these do you show them?" without a second
    round trip.
    """

    kind: Literal["action", "reveal"]
    player_id: int
    turn_number: int
    future: "asyncio.Future[Any]"
    suggestion: Suggestion | None = None
    suggesting_player_id: int | None = None
    offered: list[Card] = field(default_factory=list)


class HumanClueAgent(ClueAgent):
    """A `ClueAgent` whose decisions are answered by a person over HTTP.

    Behaves exactly like any other seat from the engine's point of view: on
    its turn `_choose_action` is awaited, and when asked to disprove a
    suggestion `_choose_reveal` is awaited. Both build a `PendingDecision`,
    notify `on_request` (typically a closure that publishes a
    `decision_requested` SSE envelope), and then await the decision's
    future — which `submit_action` / `submit_reveal` resolve once the
    person answers, from a request handler running on the same event loop.
    """

    def __init__(
        self,
        player_id: int,
        *,
        name: str | None = None,
        decision_timeout: float | None = None,
        on_request: OnRequest | None = None,
        trace_attributes: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(player_id, trace_attributes)
        self.name = name
        self.decision_timeout = decision_timeout
        self.on_request = on_request
        self.pending: PendingDecision | None = None

    async def _choose_action(self, observation: PlayerObservation) -> PlayerAction:
        # Mirrors RandomClueAgent: nothing to ask when it isn't this seat's
        # turn, or the seat is inactive. The engine already skips eliminated
        # seats via next_player_id; this is belt and braces.
        if not observation.active or observation.current_player_id != self.player_id:
            return PassTurn()

        future: asyncio.Future[PlayerAction] = asyncio.get_running_loop().create_future()
        decision = PendingDecision(
            kind="action",
            player_id=self.player_id,
            turn_number=observation.turn_number,
            future=future,
        )
        self.pending = decision
        if self.on_request is not None:
            await self.on_request(decision)

        try:
            if self.decision_timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout=self.decision_timeout)
        except TimeoutError:
            trace.get_current_span().set_attribute("clue.human_timed_out", True)
            if not future.done():
                future.cancel()
            self.pending = None
            return PassTurn()

    async def _choose_reveal(
        self,
        observation: PlayerObservation,
        suggestion: Suggestion,
        matches: list[Card],
        suggesting_player_id: int,
    ) -> Card:
        future: asyncio.Future[Card] = asyncio.get_running_loop().create_future()
        decision = PendingDecision(
            kind="reveal",
            player_id=self.player_id,
            turn_number=observation.turn_number,
            future=future,
            suggestion=suggestion,
            suggesting_player_id=suggesting_player_id,
            offered=list(matches),
        )
        self.pending = decision
        if self.on_request is not None:
            await self.on_request(decision)

        try:
            if self.decision_timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout=self.decision_timeout)
        except TimeoutError:
            trace.get_current_span().set_attribute("clue.human_timed_out", True)
            if not future.done():
                future.cancel()
            self.pending = None
            return choice(matches)

    def submit_action(self, action: PlayerAction) -> None:
        """Resolves a pending `"action"` decision. Raises on nothing
        pending, the wrong kind pending, or a decision already resolved —
        the HTTP handler maps all three onto a 409.
        """
        decision = self._require_pending("action")
        decision.future.set_result(action)
        self.pending = None

    def submit_reveal(self, card: Card) -> None:
        """Resolves a pending `"reveal"` decision. Raises if the card isn't
        one of the ones actually offered — the handler maps that to a 400,
        leaving the decision pending so the person can try again.
        """
        decision = self._require_pending("reveal")
        if card not in decision.offered:
            raise ValueError(
                f"{card!r} is not among the offered cards for player {self.player_id}"
            )
        decision.future.set_result(card)
        self.pending = None

    def _require_pending(self, kind: Literal["action", "reveal"]) -> PendingDecision:
        decision = self.pending
        if decision is None:
            raise ValueError(f"No decision is pending for player {self.player_id}")
        if decision.kind != kind:
            raise ValueError(
                f"Player {self.player_id} has a {decision.kind!r} decision pending, not {kind!r}"
            )
        if decision.future.done():
            raise ValueError(f"Decision for player {self.player_id} was already submitted")
        return decision
