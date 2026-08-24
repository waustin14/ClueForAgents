from random import choice, random

from agents.base import ClueAgent
from models.actions import MakeAccusation, MakeSuggestion, PassTurn, PlayerAction
from models.card import PersonValue, RoomValue, WeaponValue
from models.observation import PlayerObservation

ACCUSATION_PROBABILITY = 0.1


class RandomClueAgent(ClueAgent):
    """Picks a uniformly random suggestion (occasionally an accusation) on
    its turn, otherwise passes.

    Has no notion of Clue strategy at all; it exists so the engine, setup,
    and transport layers can be exercised end-to-end without an LLM.
    """

    async def choose_action(self, observation: PlayerObservation) -> PlayerAction:
        if observation.current_player_id != self.player_id or not observation.active:
            return PassTurn()

        guess = dict(
            person=choice(list(PersonValue)),
            weapon=choice(list(WeaponValue)),
            room=choice(list(RoomValue)),
        )
        if random() < ACCUSATION_PROBABILITY:
            return MakeAccusation(**guess)
        return MakeSuggestion(**guess)
