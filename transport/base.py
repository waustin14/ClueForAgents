from typing import Protocol

from models.events import PrivateGameEvent, PublicGameEvent


class AgentTransport(Protocol):
    """Delivers authoritative game events to agents.

    The game engine depends only on this abstraction, never on a concrete
    transport, so protocols like A2A or a websocket bridge can be swapped
    in without touching engine logic.
    """

    async def broadcast(self, event: PublicGameEvent) -> None: ...

    async def send(self, player_id: int, event: PrivateGameEvent) -> None: ...
