import asyncio
from dataclasses import dataclass

from agents.base import ClueAgent
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


_RUNS: dict[str, RunRecord] = {}


def add(record: RunRecord) -> None:
    _RUNS[record.game_id] = record


def get(game_id: str) -> RunRecord | None:
    return _RUNS.get(game_id)
