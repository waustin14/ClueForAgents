from abc import ABC, abstractmethod

from models.actions import PlayerAction
from models.observation import PlayerObservation


class ClueAgent(ABC):
    """A controller that plays one seat. Different implementations (LLM,
    rule-based, random, human) can drive the same Player without the game
    engine knowing or caring which.
    """

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    @abstractmethod
    async def choose_action(self, observation: PlayerObservation) -> PlayerAction: ...
