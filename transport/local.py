from collections import defaultdict

from models.events import PrivateGameEvent, PublicGameEvent


class LocalTransport:
    """In-process transport for running agents without any network hop.

    Delivered events are also kept here so tests and tooling can assert on
    what was actually sent, independent of the authoritative history the
    engine keeps on GameState.
    """

    def __init__(self) -> None:
        self.broadcast_log: list[PublicGameEvent] = []
        self.send_log: dict[int, list[PrivateGameEvent]] = defaultdict(list)

    async def broadcast(self, event: PublicGameEvent) -> None:
        self.broadcast_log.append(event)

    async def send(self, player_id: int, event: PrivateGameEvent) -> None:
        self.send_log[player_id].append(event)
