import asyncio
from collections import defaultdict
from typing import Any

from models.events import PrivateGameEvent, PublicGameEvent


class SSEBroadcastTransport:
    """In-process transport that also fans events out to SSE subscribers.

    Satisfies `AgentTransport` (`broadcast`/`send`) exactly like
    `LocalTransport`, and additionally keeps a full-history log of every
    envelope (game events plus narrative ones like `game_started` that the
    engine itself has no reason to emit) so a `GET /games/{id}/events`
    handler can replay everything-so-far to a client that connects mid-game,
    then keep streaming new envelopes as they happen.

    `subscribe_with_backlog()` is synchronous — it takes no `await` between
    reading `_history` and registering the new queue in `_subscribers` — so
    there's no window for a `publish()` running on the same event loop to
    land between the two and be silently missed.

    `publish_private_cards` controls whether the `private_reveal` envelope
    carries the revealed card's identity. Defaults on, which is fine for an
    all-AI game where the spectator feed doubles as a debugging aid; but the
    moment a human is seated at the table it turns into a cheat sheet any
    human can read by unticking "show reveals" in the console, since they
    share the same public SSE feed as every other reveal. The runner sets
    this to `False` whenever any seat is human — those players still see
    their own reveals through `GET /games/{id}/players/{pid}`, which reads
    from `PlayerObservation` rather than this feed.
    """

    def __init__(self, publish_private_cards: bool = True) -> None:
        self.broadcast_log: list[PublicGameEvent] = []
        self.send_log: dict[int, list[PrivateGameEvent]] = defaultdict(list)
        self.publish_private_cards = publish_private_cards
        self._history: list[dict[str, Any]] = []
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def subscribe_with_backlog(
        self,
    ) -> tuple[list[dict[str, Any]], "asyncio.Queue[dict[str, Any]]"]:
        backlog = list(self._history)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(queue)
        return backlog, queue

    async def publish(self, kind: str, data: dict[str, Any]) -> None:
        envelope = {"kind": kind, **data}
        self._history.append(envelope)
        for queue in self._subscribers:
            await queue.put(envelope)

    async def broadcast(self, event: PublicGameEvent) -> None:
        self.broadcast_log.append(event)
        await self.publish("public_event", {"event": event.model_dump(mode="json")})

    async def send(self, player_id: int, event: PrivateGameEvent) -> None:
        # send_log keeps the full event regardless of the privacy flag —
        # that's the authoritative record used elsewhere (and by tests),
        # only the *published* envelope is ever trimmed.
        self.send_log[player_id].append(event)
        payload = event.model_dump(mode="json")
        if not self.publish_private_cards:
            payload.pop("card", None)
        await self.publish(
            "private_reveal",
            {"player_id": player_id, "event": payload},
        )
