import asyncio
from dataclasses import dataclass, field

from agents.base import ClueAgent
from agents.human import HumanClueAgent
from game.engine import GameEngine
from models.game_state import GameState
from service.schemas import SeatConfig
from transport.sse import SSEBroadcastTransport


@dataclass
class RunRecord:
    """Everything needed to serve one in-flight or finished game run.

    `status` starts at "running" and is set by `game_runner._play` to one
    of "finished" / "turn_limit_reached" / "error" once the loop exits, so
    `GameStatusResponse` can report which of those actually happened rather
    than a binary success/failure.
    """

    game_id: str
    run_id: str
    state: GameState
    engine: GameEngine
    transport: SSEBroadcastTransport
    agents: dict[int, ClueAgent]
    seats: list[SeatConfig]
    max_turns: int
    status: str = "running"
    task: asyncio.Task[None] | None = None
    # player_id -> bearer token, minted only for human seats. Player
    # endpoints check a request's `X-Player-Token` against this map rather
    # than any real auth — see docs/plans/human-players.md §3 for why that's
    # an acceptable boundary here (a test console, not a deployment).
    player_tokens: dict[int, str] = field(default_factory=dict)

    @property
    def human_player_ids(self) -> set[int]:
        """Derived from the live agents rather than `seats`, so it always
        reflects what's actually driving each seat.
        """
        return {
            player_id
            for player_id, agent in self.agents.items()
            if isinstance(agent, HumanClueAgent)
        }


_RUNS: dict[str, RunRecord] = {}


def add(record: RunRecord) -> None:
    _RUNS[record.game_id] = record


def get(game_id: str) -> RunRecord | None:
    return _RUNS.get(game_id)
