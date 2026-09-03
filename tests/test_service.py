import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from service import registry  # noqa: E402
from service.app import app  # noqa: E402


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


def test_providers_lists_random_and_all_five_llm_providers(client):
    response = client.get("/providers")
    assert response.status_code == 200
    names = {p["name"] for p in response.json()}
    assert names == {"random", "anthropic", "openai", "gemini", "cerebras", "ollama"}
    random_entry = next(p for p in response.json() if p["name"] == "random")
    assert random_entry["available"] is True
    assert random_entry["requires_key"] is None


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
