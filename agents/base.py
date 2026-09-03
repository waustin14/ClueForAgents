from abc import ABC, abstractmethod
from random import choice
from typing import Any

from models.actions import PlayerAction
from models.card import Card
from models.events import Suggestion
from models.observation import PlayerObservation
from telemetry import tracer


class ClueAgent(ABC):
    """A controller that plays one seat. Different implementations (LLM,
    rule-based, random, human) can drive the same Player without the game
    engine knowing or caring which.

    Two decisions belong to a seat. `choose_action` is what it does on its
    own turn; subclasses must implement `_choose_action`. `choose_reveal` is
    which card it shows when someone else's suggestion reaches it — a real
    strategic choice whenever it holds more than one of the named cards, so
    subclasses may override `_choose_reveal`, and get a random pick if they
    don't.

    Both public entry points are wrapped here so every agent — present and
    future, LLM-backed or not — gets a consistent "how long did this decision
    take, what did it decide" span for free, instead of each implementation
    instrumenting itself.
    """

    def __init__(self, player_id: int, trace_attributes: dict[str, Any] | None = None) -> None:
        self.player_id = player_id
        # Who is actually playing this seat — run id, provider, model. The
        # base class can't know it (a seat's identity comes from the config
        # that built it, not from the agent class), so it's injected and
        # merged into both spans. Without it a trace shows that "some LLM
        # agent" decided something, which is useless the moment you seat two
        # different models in one game to compare them.
        self.trace_attributes = dict(trace_attributes or {})

    async def choose_action(self, observation: PlayerObservation) -> PlayerAction:
        with tracer.start_as_current_span(
            "clue.agent.choose_action",
            attributes={
                "clue.player_id": self.player_id,
                "clue.agent_type": type(self).__name__,
                "clue.turn_number": observation.turn_number,
                **self.trace_attributes,
            },
        ) as span:
            action = await self._choose_action(observation)
            span.set_attribute("clue.action_type", type(action).__name__)
            return action

    async def choose_reveal(
        self,
        observation: PlayerObservation,
        suggestion: Suggestion,
        matches: list[Card],
        suggesting_player_id: int,
    ) -> Card:
        with tracer.start_as_current_span(
            "clue.agent.choose_reveal",
            attributes={
                "clue.player_id": self.player_id,
                "clue.agent_type": type(self).__name__,
                "clue.turn_number": observation.turn_number,
                "clue.suggesting_player_id": suggesting_player_id,
                "clue.match_count": len(matches),
                **self.trace_attributes,
            },
        ) as span:
            card = await self._choose_reveal(
                observation, suggestion, matches, suggesting_player_id
            )
            if card not in matches:
                raise ValueError(
                    f"{type(self).__name__} for player {self.player_id} chose to reveal "
                    f"{card!r}, which is not one of the cards that disprove {suggestion!r}"
                )
            span.set_attribute("clue.revealed_card", str(card.value.value))
            return card

    @abstractmethod
    async def _choose_action(self, observation: PlayerObservation) -> PlayerAction: ...

    async def _choose_reveal(
        self,
        observation: PlayerObservation,
        suggestion: Suggestion,
        matches: list[Card],
        suggesting_player_id: int,
    ) -> Card:
        """Default: show a uniformly random matching card.

        Not abstract, because "pick one at random" is a perfectly valid
        policy and every agent that doesn't care should get it for free.
        """
        return choice(matches)
