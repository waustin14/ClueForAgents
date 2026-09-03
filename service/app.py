"""FastAPI wrapper exposing the game engine over HTTP + SSE.

Endpoints are deliberately thin — validation and orchestration live in
`service/game_runner.py`, this module just translates HTTP <-> those calls
and formats the SSE wire format. The front-end reaches this service through
nginx's `/api/` reverse proxy (see docker/nginx.conf), so no CORS
configuration is needed here: browser and API share an origin.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from service import registry
from service.agent_factory import list_providers
from service.game_runner import start_game
from service.registry import RunRecord
from service.schemas import (
    CreateGameRequest,
    CreateGameResponse,
    GameStatusResponse,
    ProviderInfo,
)
from telemetry import configure_langsmith_tracing, configure_tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Distinct service_name from the library default ("clueforagents") so
    # traces from this container are visually separable in Jaeger from any
    # local `main.py` runs also exporting to the same collector.
    try:
        configure_tracing(service_name="clue-game-logic")
    except ImportError:
        pass
    try:
        configure_langsmith_tracing()
    except ImportError:
        pass
    yield


app = FastAPI(title="Clue for Agents - Game Logic Service", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/providers", response_model=list[ProviderInfo])
async def get_providers() -> list[ProviderInfo]:
    return list_providers()


@app.post("/games", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest) -> CreateGameResponse:
    try:
        record = start_game(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateGameResponse(
        game_id=record.game_id,
        run_id=record.run_id,
        stream_url=f"/games/{record.game_id}/events",
    )


@app.get("/games/{game_id}", response_model=GameStatusResponse)
async def get_game(game_id: str) -> GameStatusResponse:
    record = _require_record(game_id)
    return GameStatusResponse(
        game_id=record.game_id,
        run_id=record.run_id,
        status=record.status,
        turn_number=record.state.turn_number,
        current_player_id=record.state.current_player_id,
        winning_player_id=record.state.winning_player_id,
        seats=record.seats,
    )


@app.get("/games/{game_id}/events")
async def stream_game_events(game_id: str) -> StreamingResponse:
    record = _require_record(game_id)
    return StreamingResponse(
        _sse_envelopes(record),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _require_record(game_id: str) -> RunRecord:
    record = registry.get(game_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id {game_id!r}")
    return record


async def _sse_envelopes(record: RunRecord) -> AsyncIterator[str]:
    """Replays the run's backlog first, then streams live envelopes,
    closing the response right after `game_finished` — which `_play`
    always publishes exactly once, whether the run ended in a win, a
    turn-limit, or an error — so a client never hangs waiting past it.
    """
    backlog, queue = record.transport.subscribe_with_backlog()
    for envelope in backlog:
        yield _format_sse(envelope)
        if envelope["kind"] == "game_finished":
            return

    while True:
        envelope = await queue.get()
        yield _format_sse(envelope)
        if envelope["kind"] == "game_finished":
            return


def _format_sse(envelope: dict) -> str:
    """Formats one envelope as an SSE frame, deliberately *unnamed*.

    Setting an `event:` field would make the browser dispatch each frame to
    a listener registered for that exact name, and `EventSource.onmessage`
    — the obvious handler, and the one the console uses — fires only for
    frames without one. Naming them therefore delivers every frame to
    nobody unless the client registers a listener per kind, which is a
    silent, whole-feed failure the moment a new kind is added.

    The `kind` is already the first thing in the payload, so a client
    dispatches on that instead and nothing is lost. It also sidesteps the
    trap that the `error` kind would collide with `EventSource`'s own
    `error` event, making a game error indistinguishable from a dropped
    connection.
    """
    return f"data: {json.dumps(envelope)}\n\n"
