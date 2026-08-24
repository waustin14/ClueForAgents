from abc import ABC, abstractmethod

from models.actions import PlayerAction
from models.observation import PlayerObservation
from telemetry import tracer


class ClueAgent(ABC):
    """A controller that plays one seat. Different implementations (LLM,
    rule-based, random, human) can drive the same Player without the game
    engine knowing or caring which.

    `choose_action` is the public entry point; subclasses implement
    `_choose_action`. Wrapping it here means every agent — present and
    future, LLM-backed or not — gets a consistent "how long did this
    decision take, what did it decide" span for free, instead of each
    implementation instrumenting itself.
    """

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    async def choose_action(self, observation: PlayerObservation) -> PlayerAction:
        with tracer.start_as_current_span(
            "clue.agent.choose_action",
            attributes={
                "clue.player_id": self.player_id,
                "clue.agent_type": type(self).__name__,
                "clue.turn_number": observation.turn_number,
            },
        ) as span:
            action = await self._choose_action(observation)
            span.set_attribute("clue.action_type", type(action).__name__)
            return action

    @abstractmethod
    async def _choose_action(self, observation: PlayerObservation) -> PlayerAction: ...
