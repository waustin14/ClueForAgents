from typing import Protocol

from pydantic import ValidationError

from agents.base import ClueAgent
from models.actions import MakeAccusation, MakeSuggestion, PassTurn, PlayerAction
from models.observation import PlayerObservation

_ACTION_TYPES: tuple[type[PlayerAction], ...] = (MakeAccusation, MakeSuggestion, PassTurn)


class LLMClient(Protocol):
    """Minimal interface an LLM agent needs from a model provider.

    Concrete implementations (Anthropic, OpenAI, a local model, a scripted
    stub for tests) live outside this module so the agent layer never
    depends on a specific SDK.
    """

    async def complete(self, prompt: str) -> str: ...


def build_prompt(observation: PlayerObservation, scratchpad: str = "") -> str:
    lines = [
        "You are playing Clue. Respond with a single JSON object matching one of:",
        f"  {MakeSuggestion.__name__}: {{'person', 'weapon', 'room'}}",
        f"  {MakeAccusation.__name__}: {{'person', 'weapon', 'room'}}",
        f"  {PassTurn.__name__}: {{}}",
        "",
        f"You are player {observation.player_id}. Current player: {observation.current_player_id}.",
        f"Turn: {observation.turn_number}. You are still active: {observation.active}.",
        f"Your hand: {[card.value.value for card in observation.own_cards]}",
        f"Your card knowledge log: {observation.card_log.model_dump_json()}",
        f"Public event history: {[e.model_dump_json() for e in observation.public_history]}",
        f"Private reveals you've seen: {[e.model_dump_json() for e in observation.private_reveals]}",
    ]
    if scratchpad:
        lines.append(f"Your private notes from previous turns: {scratchpad}")
    return "\n".join(lines)


def parse_action(raw: str) -> PlayerAction:
    for action_cls in _ACTION_TYPES:
        try:
            return action_cls.model_validate_json(raw)
        except ValidationError:
            continue
    raise ValueError(f"Could not parse a valid action from LLM output: {raw!r}")


class LLMClueAgent(ClueAgent):
    """Delegates action selection to an LLM.

    `scratchpad` is a private, subjective notepad the agent may grow across
    turns to hold its own deductions and strategy. It is never authoritative
    game state and the engine never reads it.
    """

    def __init__(self, player_id: int, client: LLMClient) -> None:
        super().__init__(player_id)
        self.client = client
        self.scratchpad = ""

    async def choose_action(self, observation: PlayerObservation) -> PlayerAction:
        prompt = build_prompt(observation, self.scratchpad)
        raw = await self.client.complete(prompt)
        return parse_action(raw)
