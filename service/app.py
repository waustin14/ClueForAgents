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

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from agents.human import HumanClueAgent
from models.card import PersonValue, RoomValue, WeaponValue
from service import registry
from service.agent_factory import list_providers
from service.game_runner import start_game
from service.registry import RunRecord
from service.schemas import (
    CardCatalog,
    CreateGameRequest,
    CreateGameResponse,
    GameStatusResponse,
    HumanSeatCredential,
    PendingDecisionView,
    PlayerView,
    ProviderInfo,
    SubmitActionRequest,
    SubmitRevealRequest,
    WaitingOn,
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


@app.get("/cards", response_model=CardCatalog)
async def get_cards() -> CardCatalog:
    # So the front-end (and anyone driving the API directly) never has to
    # hardcode card names — they come straight from the same enums the
    # engine itself uses.
    return CardCatalog(
        people=list(PersonValue),
        weapons=list(WeaponValue),
        rooms=list(RoomValue),
    )


@app.post("/games", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest) -> CreateGameResponse:
    try:
        record = start_game(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    human_seats = [
        HumanSeatCredential(
            player_id=player_id,
            name=record.seats[player_id].name,
            token=token,
            join_url=f"/?game={record.game_id}&player={player_id}&token={token}",
        )
        for player_id, token in sorted(record.player_tokens.items())
    ]
    return CreateGameResponse(
        game_id=record.game_id,
        run_id=record.run_id,
        stream_url=f"/games/{record.game_id}/events",
        human_seats=human_seats,
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
        waiting_on=_waiting_on(record),
    )


@app.get("/games/{game_id}/events")
async def stream_game_events(game_id: str) -> StreamingResponse:
    record = _require_record(game_id)
    return StreamingResponse(
        _sse_envelopes(record),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/games/{game_id}/players/{player_id}", response_model=PlayerView)
async def get_player_view(
    game_id: str, player_id: int, x_player_token: str | None = Header(default=None)
) -> PlayerView:
    record = _require_record(game_id)
    _authorize_player(record, player_id, x_player_token)
    return _player_view(record, player_id)


@app.post("/games/{game_id}/players/{player_id}/action", response_model=PlayerView)
async def submit_player_action(
    game_id: str,
    player_id: int,
    action: SubmitActionRequest,
    x_player_token: str | None = Header(default=None),
) -> PlayerView:
    record = _require_record(game_id)
    agent = _authorize_player(record, player_id, x_player_token)

    pending = agent.pending
    if pending is None or pending.kind != "action":
        raise HTTPException(status_code=409, detail="no action is pending for this seat")
    turn = pending.turn_number

    try:
        agent.submit_action(action)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await record.transport.publish(
        "decision_resolved",
        # `decision_kind`, not `kind` — see the matching comment in
        # `game_runner._make_on_decision_request` for why "kind" here would
        # collide with the envelope's own discriminator.
        {"player_id": player_id, "decision_kind": "action", "turn": turn},
    )
    return _player_view(record, player_id)


@app.post("/games/{game_id}/players/{player_id}/reveal", response_model=PlayerView)
async def submit_player_reveal(
    game_id: str,
    player_id: int,
    request: SubmitRevealRequest,
    x_player_token: str | None = Header(default=None),
) -> PlayerView:
    record = _require_record(game_id)
    agent = _authorize_player(record, player_id, x_player_token)

    pending = agent.pending
    if pending is None or pending.kind != "reveal":
        raise HTTPException(status_code=409, detail="no reveal is pending for this seat")
    turn = pending.turn_number

    if request.card not in pending.offered:
        raise HTTPException(status_code=400, detail="card is not among the offered cards")

    try:
        agent.submit_reveal(request.card)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await record.transport.publish(
        "decision_resolved",
        {"player_id": player_id, "decision_kind": "reveal", "turn": turn},
    )
    return _player_view(record, player_id)


def _require_record(game_id: str) -> RunRecord:
    record = registry.get(game_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id {game_id!r}")
    return record


def _authorize_player(record: RunRecord, player_id: int, token: str | None) -> HumanClueAgent:
    """Every player endpoint needs the same three checks in the same order:
    is this even a human seat (404 either way — a non-human seat isn't
    something a token could ever be right for), and if so, does the token
    presented match the one minted for it (403). Returns the seat's agent
    so callers don't have to look it up again.
    """
    expected = record.player_tokens.get(player_id)
    if expected is None:
        raise HTTPException(status_code=404, detail=f"player {player_id} is not a human seat")
    if token != expected:
        raise HTTPException(status_code=403, detail="missing or invalid X-Player-Token")
    agent = record.agents[player_id]
    assert isinstance(agent, HumanClueAgent)  # guaranteed by player_tokens above
    return agent


def _player_view(record: RunRecord, player_id: int) -> PlayerView:
    agent = record.agents[player_id]
    observation = record.engine.observation_for(player_id)
    pending_view = None
    if isinstance(agent, HumanClueAgent) and agent.pending is not None:
        decision = agent.pending
        pending_view = PendingDecisionView(
            kind=decision.kind,
            turn_number=decision.turn_number,
            suggestion=decision.suggestion,
            suggesting_player_id=decision.suggesting_player_id,
            offered=list(decision.offered),
        )
    return PlayerView(observation=observation, pending=pending_view, status=record.status)


def _waiting_on(record: RunRecord) -> WaitingOn | None:
    for agent in record.agents.values():
        if isinstance(agent, HumanClueAgent) and agent.pending is not None:
            return WaitingOn(player_id=agent.player_id, kind=agent.pending.kind)
    return None


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
