from dataclasses import dataclass
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from agents.base import ClueAgent
from models.actions import MakeAccusation, MakeSuggestion, PassTurn, PlayerAction
from models.observation import PlayerObservation

_player_action_adapter: TypeAdapter[PlayerAction] = TypeAdapter(PlayerAction)


@dataclass
class PromptSegment:
    """One piece of a prompt, tagged with whether its bytes are stable.

    `cacheable=True` segments are only ever appended to or left untouched
    turn-over-turn for a given agent, so a client backed by a provider with
    prefix caching (e.g. Anthropic's `cache_control`) can mark the boundary
    after the last cacheable segment and pay full price only for what
    changed. A client for a provider without explicit caching can simply
    ignore the flag and concatenate everything.
    """

    text: str
    cacheable: bool = False


class LLMClient(Protocol):
    """Minimal interface an LLM agent needs from a model provider.

    Concrete implementations (Anthropic, OpenAI, a local model, a scripted
    stub for tests) live outside this module so the agent layer never
    depends on a specific SDK.
    """

    async def complete(self, segments: list[PromptSegment]) -> str: ...


def build_prompt(observation: PlayerObservation, scratchpad: str = "") -> list[PromptSegment]:
    """Builds the prompt as stable-prefix-first, volatile-suffix-last.

    `player_id` and `own_cards` never change after deal; `public_history`
    and `private_reveals` only ever grow by appending. Those are cacheable.
    `card_log` mutates in place as facts are learned, and `current_player_id`
    /`turn_number`/`active`/`scratchpad` change every turn, so those go in
    the trailing, non-cacheable segment.
    """
    instructions = "\n".join(
        [
            "You are playing Clue. Respond with a single JSON object matching one of:",
            f"  {{'kind': 'suggestion', 'person', 'weapon', 'room'}}  ({MakeSuggestion.__name__})",
            f"  {{'kind': 'accusation', 'person', 'weapon', 'room'}}  ({MakeAccusation.__name__})",
            f"  {{'kind': 'pass'}}  ({PassTurn.__name__})",
            "The 'kind' field is required and selects which of the three you mean.",
            "",
            f"You are player {observation.player_id}.",
        ]
    )
    own_hand = f"Your hand: {[card.value.value for card in observation.own_cards]}"
    public_history = "Public event history:\n" + "\n".join(
        event.model_dump_json() for event in observation.public_history
    )
    private_reveals = "Private reveals you've seen:\n" + "\n".join(
        event.model_dump_json() for event in observation.private_reveals
    )

    volatile_lines = [
        f"Current player: {observation.current_player_id}.",
        f"Turn: {observation.turn_number}. You are still active: {observation.active}.",
        f"Your card knowledge log: {observation.card_log.model_dump_json()}",
    ]
    if scratchpad:
        volatile_lines.append(f"Your private notes from previous turns: {scratchpad}")
    volatile = "\n".join(volatile_lines)

    return [
        PromptSegment(instructions, cacheable=True),
        PromptSegment(own_hand, cacheable=True),
        PromptSegment(public_history, cacheable=True),
        PromptSegment(private_reveals, cacheable=True),
        PromptSegment(volatile, cacheable=False),
    ]


def parse_action(raw: str) -> PlayerAction:
    try:
        return _player_action_adapter.validate_json(raw)
    except ValidationError as exc:
        raise ValueError(f"Could not parse a valid action from LLM output: {raw!r}") from exc


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
        segments = build_prompt(observation, self.scratchpad)
        raw = await self.client.complete(segments)
        return parse_action(raw)
