import json
from uuid import uuid4

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from agents.human import HumanClueAgent, PendingDecision  # noqa: E402
from agents.random_agent import RandomClueAgent  # noqa: E402
from game.engine import GameEngine  # noqa: E402
from game.setup import initialize_game  # noqa: E402
from models.card import PersonCard, PersonValue, RoomValue, WeaponValue  # noqa: E402
from models.events import Suggestion  # noqa: E402
from service import registry  # noqa: E402
from service.app import app  # noqa: E402
from service.registry import RunRecord  # noqa: E402
from service.schemas import SeatConfig  # noqa: E402
from transport.sse import SSEBroadcastTransport  # noqa: E402


@pytest.fixture(autouse=True)
def clear_registry():
    registry._RUNS.clear()
    yield
    registry._RUNS.clear()


@pytest.fixture(autouse=True)
def no_op_tracing_setup(monkeypatch):
    """Keeps the app's startup lifespan from touching real tracing state.

    Without this, if the `otel` extra happens to be installed and no other
    test has already claimed the process-global TracerProvider (OTel only
    allows one `set_tracer_provider` call to succeed per process), the
    lifespan's `configure_tracing()` call would install a real OTLP
    exporter pointed at `localhost:4317` and every test below would pay for
    real (failing) network retries — and which outcome you got would
    depend on unrelated test-file import order. These endpoint tests only
    care about the HTTP/SSE contract, so tracing setup is neutered outright
    rather than relying on ordering luck.
    """
    monkeypatch.setattr("service.app.configure_tracing", lambda **kwargs: None)
    monkeypatch.setattr("service.app.configure_langsmith_tracing", lambda **kwargs: None)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_providers_lists_random_human_and_all_five_llm_providers(client):
    response = client.get("/providers")
    assert response.status_code == 200
    names = {p["name"] for p in response.json()}
    assert names == {"random", "human", "anthropic", "openai", "gemini", "cerebras", "ollama"}
    random_entry = next(p for p in response.json() if p["name"] == "random")
    assert random_entry["available"] is True
    assert random_entry["requires_key"] is None
    human_entry = next(p for p in response.json() if p["name"] == "human")
    assert human_entry["available"] is True
    assert human_entry["requires_key"] is None


def test_create_game_rejects_seat_count_mismatch(client):
    response = client.post(
        "/games",
        json={"n_players": 3, "seats": [{"agent_type": "random"}, {"agent_type": "random"}]},
    )
    assert response.status_code == 400


def test_create_game_rejects_llm_seat_without_provider_and_model(client):
    response = client.post(
        "/games",
        json={"n_players": 1, "seats": [{"agent_type": "llm"}]},
    )
    assert response.status_code == 422


def test_create_game_rejects_unset_provider_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = client.post(
        "/games",
        json={
            "n_players": 2,
            "seats": [
                {"agent_type": "llm", "provider": "anthropic", "model": "claude-sonnet-5"},
                {"agent_type": "random"},
            ],
        },
    )
    assert response.status_code == 400
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_full_random_game_runs_to_completion_over_sse(client):
    create_response = client.post(
        "/games",
        json={
            "n_players": 3,
            "seats": [{"agent_type": "random"}] * 3,
            "max_turns": 200,
        },
    )
    assert create_response.status_code == 200
    body = create_response.json()
    game_id = body["game_id"]
    assert body["run_id"] == game_id
    assert body["stream_url"] == f"/games/{game_id}/events"

    with client.stream("GET", f"/games/{game_id}/events") as stream:
        kinds = []
        turns = []
        for line in stream.iter_lines():
            if line.startswith("data: "):
                envelope = json.loads(line[len("data: ") :])
                kinds.append(envelope["kind"])
                if envelope["kind"] == "turn_taken":
                    turns.append(envelope)
                if envelope["kind"] == "game_finished":
                    assert envelope["status"] in {"finished", "turn_limit_reached"}
                    break

    assert kinds[0] == "game_started"
    assert kinds[-1] == "game_finished"
    # The SSE feed is the spectator view, so an agent's private scratchpad
    # never reaches it — not even as an explicit null.
    assert turns
    assert all("notes" not in turn["action"] for turn in turns)

    status_response = client.get(f"/games/{game_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] in {"finished", "turn_limit_reached"}


def test_stream_frames_reach_the_default_message_handler(client):
    # An `event:` field makes a browser dispatch the frame to a listener
    # registered for that exact name, and `EventSource.onmessage` fires only
    # for frames without one. Naming frames therefore delivers the whole feed
    # to nobody unless the client registers a listener per kind — which is
    # what it did, and why the console rendered an empty feed while the game
    # ran to completion server-side. This asserts the dispatch contract the
    # console actually relies on; reading `data:` lines alone would not.
    create_response = client.post(
        "/games",
        json={"n_players": 3, "seats": [{"agent_type": "random"}] * 3, "max_turns": 200},
    )
    game_id = create_response.json()["game_id"]

    with client.stream("GET", f"/games/{game_id}/events") as stream:
        body = stream.read().decode()

    assert "event:" not in body
    kinds = [
        json.loads(line[len("data: ") :])["kind"]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert kinds[0] == "game_started"
    assert "turn_taken" in kinds
    assert kinds[-1] == "game_finished"


def test_create_game_rejects_a_negative_thinking_budget(client):
    response = client.post(
        "/games",
        json={
            "n_players": 2,
            "seats": [
                {
                    "agent_type": "llm",
                    "provider": "ollama",
                    "model": "llama3",
                    "thinking_budget": -1,
                },
                {"agent_type": "random"},
            ],
        },
    )
    assert response.status_code == 422


def test_get_unknown_game_returns_404(client):
    response = client.get("/games/does-not-exist")
    assert response.status_code == 404


def test_get_unknown_game_returns_404_for_players_endpoint(client):
    response = client.get("/games/does-not-exist/players/0")
    assert response.status_code == 404


class _FakeFuture:
    """Stands in for `asyncio.Future` in tests that build a `RunRecord`
    directly rather than through `start_game` — nothing here ever awaits
    the pending decision's future (no background `_play` task is running),
    so all `HumanClueAgent.submit_action`/`submit_reveal` need from it is
    `set_result`/`done`, not real event-loop binding.
    """

    def __init__(self) -> None:
        self._done = False
        self.result = None

    def set_result(self, value) -> None:
        self._done = True
        self.result = value

    def done(self) -> bool:
        return self._done


def _register_human_vs_random_record(token: str = "tok-0") -> tuple[RunRecord, HumanClueAgent]:
    """Registers a two-seat [human, random] run directly in the registry,
    with no background game loop driving it. Used by the player-endpoint
    contract tests below, which only care about how `service/app.py`
    translates authorization and decision state into status codes — not
    about real gameplay timing, which would make polling for a specific
    pending decision racy against the background `_play` task.
    """
    state = initialize_game(2)
    transport = SSEBroadcastTransport(publish_private_cards=False)
    engine = GameEngine(state, transport)
    human = HumanClueAgent(0)
    bot = RandomClueAgent(1)
    record = RunRecord(
        game_id=uuid4().hex,
        run_id=uuid4().hex,
        state=state,
        engine=engine,
        transport=transport,
        agents={0: human, 1: bot},
        seats=[SeatConfig(agent_type="human"), SeatConfig(agent_type="random")],
        max_turns=1,
        player_tokens={0: token},
    )
    registry.add(record)
    return record, human


def test_create_game_mints_one_credential_per_human_seat_and_no_others(client):
    response = client.post(
        "/games",
        json={
            "n_players": 3,
            "seats": [
                {"agent_type": "human", "name": "Alice"},
                {"agent_type": "random"},
                {"agent_type": "human", "name": "Bob"},
            ],
            "max_turns": 1,
        },
    )
    assert response.status_code == 200
    body = response.json()
    human_seats = {c["player_id"]: c for c in body["human_seats"]}
    assert set(human_seats) == {0, 2}
    assert human_seats[0]["name"] == "Alice"
    assert human_seats[2]["name"] == "Bob"
    for player_id, credential in human_seats.items():
        assert credential["token"]
        assert credential["join_url"] == (
            f"/?game={body['game_id']}&player={player_id}&token={credential['token']}"
        )


def test_create_game_rejects_provider_set_on_a_human_seat(client):
    response = client.post(
        "/games",
        json={
            "n_players": 2,
            "seats": [
                {"agent_type": "human", "provider": "anthropic"},
                {"agent_type": "random"},
            ],
        },
    )
    assert response.status_code == 422


def test_player_endpoint_requires_a_token(client):
    record, _human = _register_human_vs_random_record()
    response = client.get(f"/games/{record.game_id}/players/0")
    assert response.status_code == 403


def test_player_endpoint_rejects_another_seats_token(client):
    record, _human = _register_human_vs_random_record(token="right-token")
    response = client.get(
        f"/games/{record.game_id}/players/0", headers={"X-Player-Token": "wrong-token"}
    )
    assert response.status_code == 403


def test_player_endpoint_rejects_a_non_human_seat(client):
    record, _human = _register_human_vs_random_record(token="tok-0")
    response = client.get(
        f"/games/{record.game_id}/players/1", headers={"X-Player-Token": "tok-0"}
    )
    assert response.status_code == 404


def test_submit_action_with_nothing_pending_returns_409(client):
    record, _human = _register_human_vs_random_record(token="tok-0")
    response = client.post(
        f"/games/{record.game_id}/players/0/action",
        json={"kind": "pass"},
        headers={"X-Player-Token": "tok-0"},
    )
    assert response.status_code == 409


def test_submit_reveal_with_an_off_menu_card_returns_400_and_leaves_pending(client):
    record, human = _register_human_vs_random_record(token="tok-0")
    suggestion = Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)
    decision = PendingDecision(
        kind="reveal",
        player_id=0,
        turn_number=3,
        future=_FakeFuture(),
        suggestion=suggestion,
        suggesting_player_id=1,
        offered=[PersonCard(value=PersonValue.PLUM)],
    )
    human.pending = decision

    response = client.post(
        f"/games/{record.game_id}/players/0/reveal",
        json={"card": {"type": "weapon", "value": "Knife"}},
        headers={"X-Player-Token": "tok-0"},
    )

    assert response.status_code == 400
    assert human.pending is decision
    assert not decision.future.done()

    # The decision genuinely survived the bad submit — a correct follow-up
    # still resolves it.
    ok = client.post(
        f"/games/{record.game_id}/players/0/reveal",
        json={"card": {"type": "person", "value": "Professor Plum"}},
        headers={"X-Player-Token": "tok-0"},
    )
    assert ok.status_code == 200
    assert human.pending is None
    assert decision.future.done()


def test_end_to_end_human_and_random_game_via_the_api(client):
    create_response = client.post(
        "/games",
        json={
            "n_players": 2,
            "seats": [{"agent_type": "human", "name": "Alice"}, {"agent_type": "random"}],
            "max_turns": 60,
        },
    )
    assert create_response.status_code == 200
    body = create_response.json()
    game_id = body["game_id"]
    assert len(body["human_seats"]) == 1
    credential = body["human_seats"][0]
    assert credential["player_id"] == 0
    headers = {"X-Player-Token": credential["token"]}

    cards = client.get("/cards").json()

    status = "running"
    for _ in range(500):
        view_response = client.get(f"/games/{game_id}/players/0", headers=headers)
        assert view_response.status_code == 200
        view = view_response.json()
        status = view["status"]
        if status != "running":
            break
        pending = view["pending"]
        if pending is None:
            continue
        if pending["kind"] == "action":
            action = {
                "kind": "suggestion",
                "person": cards["people"][0],
                "weapon": cards["weapons"][0],
                "room": cards["rooms"][0],
            }
            submit = client.post(
                f"/games/{game_id}/players/0/action", json=action, headers=headers
            )
            assert submit.status_code == 200
        else:
            card = pending["offered"][0]
            submit = client.post(
                f"/games/{game_id}/players/0/reveal", json={"card": card}, headers=headers
            )
            assert submit.status_code == 200
    else:
        pytest.fail("game did not finish within the polling budget")

    assert status in {"finished", "turn_limit_reached", "error"}

    with client.stream("GET", f"/games/{game_id}/events") as stream:
        kinds = []
        reveals = []
        for line in stream.iter_lines():
            if line.startswith("data: "):
                envelope = json.loads(line[len("data: ") :])
                kinds.append(envelope["kind"])
                if envelope["kind"] == "private_reveal":
                    reveals.append(envelope)
                if envelope["kind"] == "game_finished":
                    break

    assert "decision_requested" in kinds
    assert "decision_resolved" in kinds
    # This is the whole point of `publish_private_cards=False`: a human is
    # seated, so the spectator feed must never leak a revealed card's
    # identity, even though the seat that made the reveal legitimately saw
    # it via its own `GET /games/{id}/players/{pid}`.
    assert all("card" not in r["event"] for r in reveals)


def test_all_random_game_still_publishes_card_identity_on_private_reveal(client):
    create_response = client.post(
        "/games",
        json={
            "n_players": 3,
            "seats": [{"agent_type": "random"}] * 3,
            "max_turns": 200,
        },
    )
    game_id = create_response.json()["game_id"]

    with client.stream("GET", f"/games/{game_id}/events") as stream:
        reveals = []
        for line in stream.iter_lines():
            if line.startswith("data: "):
                envelope = json.loads(line[len("data: ") :])
                if envelope["kind"] == "private_reveal":
                    reveals.append(envelope)
                if envelope["kind"] == "game_finished":
                    break

    # An all-AI game keeps the spectator feed as a debugging aid, so card
    # identities stay on `private_reveal` envelopes by default — this is a
    # regression guard for that default (`publish_private_cards=True`).
    assert reveals, "expected at least one private reveal in a 3-player random game"
    assert all("card" in r["event"] for r in reveals)
