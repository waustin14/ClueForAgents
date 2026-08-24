import asyncio

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from agents.random_agent import RandomClueAgent  # noqa: E402
from game.engine import GameEngine  # noqa: E402
from models.actions import MakeAccusation, MakeSuggestion, PassTurn  # noqa: E402
from models.card import (  # noqa: E402
    PersonCard,
    PersonValue,
    RoomCard,
    RoomValue,
    WeaponCard,
    WeaponValue,
)
from models.game_state import GameState, GameStatus, Solution  # noqa: E402
from models.player import Player  # noqa: E402
from transport.local import LocalTransport  # noqa: E402

# The global TracerProvider can only be installed once per process (the
# OTel API deliberately no-ops a second `set_tracer_provider` call), so this
# module wires it up once here rather than per test, and each test clears
# the shared in-memory exporter instead of reinstalling the provider.
_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_provider)


@pytest.fixture(autouse=True)
def clear_spans():
    _EXPORTER.clear()
    yield


def run(coro):
    return asyncio.run(coro)


def build_state() -> GameState:
    solution = Solution(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)
    players = [
        Player(id=0, cards=[PersonCard(value=PersonValue.GREEN)]),
        Player(id=1, cards=[WeaponCard(value=WeaponValue.KNIFE)]),
        Player(id=2, cards=[RoomCard(value=RoomValue.STUDY)]),
    ]
    return GameState(players=players, solution=solution)


def test_turn_and_suggestion_spans_nest_and_carry_attributes():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    suggestion = MakeSuggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)

    run(engine.take_turn(0, suggestion))

    spans = {span.name: span for span in _EXPORTER.get_finished_spans()}
    assert set(spans) == {"clue.turn", "clue.suggestion"}

    turn_span = spans["clue.turn"]
    suggestion_span = spans["clue.suggestion"]
    assert suggestion_span.parent.span_id == turn_span.context.span_id

    assert turn_span.attributes["clue.player_id"] == 0
    assert turn_span.attributes["clue.action_type"] == "MakeSuggestion"
    assert suggestion_span.attributes["clue.suggesting_player_id"] == 0
    assert suggestion_span.attributes["clue.disproving_player_id"] == 2
    assert suggestion_span.attributes["clue.unable_to_disprove_count"] == 1


def test_suggestion_span_omits_disproving_player_when_nobody_can_disprove():
    state = build_state()
    state.players[2].cards = [PersonCard(value=PersonValue.WHITE)]
    engine = GameEngine(state, LocalTransport())
    suggestion = MakeSuggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)

    run(engine.take_turn(0, suggestion))

    suggestion_span = next(s for s in _EXPORTER.get_finished_spans() if s.name == "clue.suggestion")
    assert "clue.disproving_player_id" not in suggestion_span.attributes
    assert suggestion_span.attributes["clue.unable_to_disprove_count"] == 2


def test_accusation_span_records_correctness():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    accusation = MakeAccusation(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)

    run(engine.take_turn(0, accusation))

    spans = {span.name: span for span in _EXPORTER.get_finished_spans()}
    assert spans["clue.accusation"].attributes["clue.correct"] is True
    assert spans["clue.accusation"].parent.span_id == spans["clue.turn"].context.span_id


def test_pass_turn_produces_only_the_turn_span():
    state = build_state()
    engine = GameEngine(state, LocalTransport())

    run(engine.take_turn(0, PassTurn()))

    names = [span.name for span in _EXPORTER.get_finished_spans()]
    assert names == ["clue.turn"]


def test_agent_choose_action_span_wraps_every_agent_subclass():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    agent = RandomClueAgent(player_id=0)
    observation = engine.observation_for(0)

    run(agent.choose_action(observation))

    span = next(s for s in _EXPORTER.get_finished_spans() if s.name == "clue.agent.choose_action")
    assert span.attributes["clue.player_id"] == 0
    assert span.attributes["clue.agent_type"] == "RandomClueAgent"
    assert span.attributes["clue.action_type"] in {"MakeSuggestion", "MakeAccusation", "PassTurn"}


def test_take_turn_error_marks_span_as_error_without_swallowing_it():
    state = build_state()
    state.status = GameStatus.FINISHED
    engine = GameEngine(state, LocalTransport())

    with pytest.raises(ValueError):
        run(engine.take_turn(0, PassTurn()))

    span = _EXPORTER.get_finished_spans()[0]
    assert span.name == "clue.turn"
    assert span.status.status_code.name == "ERROR"
