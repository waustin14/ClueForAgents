import asyncio
import contextlib

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from agents.human import HumanClueAgent  # noqa: E402
from agents.random_agent import RandomClueAgent  # noqa: E402
from game.engine import GameEngine  # noqa: E402
from game.setup import initialize_game  # noqa: E402
from models.actions import PassTurn  # noqa: E402
from models.card import PersonCard, PersonValue, RoomValue, WeaponCard, WeaponValue  # noqa: E402
from models.events import Suggestion  # noqa: E402
from models.game_state import GameStatus  # noqa: E402
from models.observation import PlayerObservation  # noqa: E402
from models.player import PlayerLog  # noqa: E402
from transport.local import LocalTransport  # noqa: E402

# The global TracerProvider can only be installed once per process (a second
# `set_tracer_provider` call is a deliberate no-op in the OTel API), so if
# another test module already installed one (e.g. tests/test_telemetry.py,
# depending on collection order) this just adds another processor to it
# instead of trying to replace it.
_EXPORTER = InMemorySpanExporter()
_provider = trace.get_tracer_provider()
if not isinstance(_provider, TracerProvider):
    _provider = TracerProvider()
    trace.set_tracer_provider(_provider)
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))


@pytest.fixture(autouse=True)
def clear_spans():
    _EXPORTER.clear()
    yield


def run(coro):
    return asyncio.run(coro)


def make_observation(
    *, player_id: int = 0, current_player_id: int = 0, active: bool = True, turn_number: int = 1
) -> PlayerObservation:
    return PlayerObservation(
        player_id=player_id,
        turn_number=turn_number,
        current_player_id=current_player_id,
        active=active,
        card_log=PlayerLog.empty(),
    )


def test_choose_action_asks_and_returns_what_is_submitted():
    requested = []

    async def on_request(decision):
        requested.append(decision)

    async def scenario():
        agent = HumanClueAgent(0, on_request=on_request)
        observation = make_observation(player_id=0, current_player_id=0)

        async def submit_soon():
            await asyncio.sleep(0)
            assert agent.pending is not None
            agent.submit_action(PassTurn())

        action, _ = await asyncio.gather(agent.choose_action(observation), submit_soon())
        return agent, action

    agent, action = run(scenario())
    assert isinstance(action, PassTurn)
    assert agent.pending is None
    assert requested[0].kind == "action"
    assert requested[0].player_id == 0


def test_submit_action_without_a_pending_decision_raises():
    agent = HumanClueAgent(0)
    with pytest.raises(ValueError):
        agent.submit_action(PassTurn())


def test_double_submit_action_raises():
    async def scenario():
        agent = HumanClueAgent(0)
        observation = make_observation(player_id=0, current_player_id=0)

        async def submit_twice():
            await asyncio.sleep(0)
            agent.submit_action(PassTurn())
            with pytest.raises(ValueError):
                agent.submit_action(PassTurn())

        await asyncio.gather(agent.choose_action(observation), submit_twice())

    run(scenario())


def test_choose_reveal_returns_the_submitted_card():
    async def scenario():
        agent = HumanClueAgent(1)
        observation = make_observation(player_id=1, current_player_id=0)
        suggestion = Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)
        matches = [PersonCard(value=PersonValue.PLUM), WeaponCard(value=WeaponValue.ROPE)]

        async def submit_soon():
            await asyncio.sleep(0)
            agent.submit_reveal(matches[1])

        card, _ = await asyncio.gather(
            agent.choose_reveal(observation, suggestion, matches, suggesting_player_id=0),
            submit_soon(),
        )
        return card, matches

    card, matches = run(scenario())
    assert card == matches[1]


def test_submit_reveal_off_menu_card_raises_and_leaves_decision_pending():
    async def scenario():
        agent = HumanClueAgent(1)
        observation = make_observation(player_id=1, current_player_id=0)
        suggestion = Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)
        matches = [PersonCard(value=PersonValue.PLUM)]
        off_menu = WeaponCard(value=WeaponValue.KNIFE)

        async def try_off_menu_then_correct():
            await asyncio.sleep(0)
            with pytest.raises(ValueError):
                agent.submit_reveal(off_menu)
            assert agent.pending is not None
            assert not agent.pending.future.done()
            agent.submit_reveal(matches[0])

        card, _ = await asyncio.gather(
            agent.choose_reveal(observation, suggestion, matches, suggesting_player_id=0),
            try_off_menu_then_correct(),
        )
        return card, matches

    card, matches = run(scenario())
    assert card == matches[0]


def test_inactive_seat_returns_pass_without_asking_anyone():
    called = False

    async def on_request(decision):
        nonlocal called
        called = True

    async def scenario():
        agent = HumanClueAgent(0, on_request=on_request)
        observation = make_observation(player_id=0, current_player_id=0, active=False)
        return await agent.choose_action(observation), agent

    action, agent = run(scenario())
    assert isinstance(action, PassTurn)
    assert called is False
    assert agent.pending is None


def test_not_this_seats_turn_returns_pass_without_asking_anyone():
    called = False

    async def on_request(decision):
        nonlocal called
        called = True

    async def scenario():
        agent = HumanClueAgent(0, on_request=on_request)
        observation = make_observation(player_id=0, current_player_id=1, active=True)
        return await agent.choose_action(observation), agent

    action, agent = run(scenario())
    assert isinstance(action, PassTurn)
    assert called is False
    assert agent.pending is None


def test_action_decision_times_out_to_pass_turn():
    async def scenario():
        agent = HumanClueAgent(0, decision_timeout=0.01)
        observation = make_observation(player_id=0, current_player_id=0)
        return await agent.choose_action(observation), agent

    action, agent = run(scenario())
    assert isinstance(action, PassTurn)
    assert agent.pending is None

    span = next(s for s in _EXPORTER.get_finished_spans() if s.name == "clue.agent.choose_action")
    assert span.attributes["clue.human_timed_out"] is True


def test_reveal_decision_times_out_to_one_of_the_matches():
    async def scenario():
        agent = HumanClueAgent(1, decision_timeout=0.01)
        observation = make_observation(player_id=1, current_player_id=0)
        suggestion = Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)
        matches = [PersonCard(value=PersonValue.PLUM), WeaponCard(value=WeaponValue.ROPE)]
        card = await agent.choose_reveal(observation, suggestion, matches, suggesting_player_id=0)
        return card, matches, agent

    card, matches, agent = run(scenario())
    assert card in matches
    assert agent.pending is None

    span = next(s for s in _EXPORTER.get_finished_spans() if s.name == "clue.agent.choose_reveal")
    assert span.attributes["clue.human_timed_out"] is True


def test_human_and_random_seats_play_to_completion_or_turn_cap():
    """Engine-level integration: no service, no HTTP — a background task
    plays the human seat by watching `agent.pending` and calling
    `submit_action` / `submit_reveal` directly, exactly the way an HTTP
    handler would, minus the transport hop.
    """

    async def scenario():
        state = initialize_game(2)
        transport = LocalTransport()
        engine = GameEngine(state, transport)
        human = HumanClueAgent(0)
        bot = RandomClueAgent(1)
        agents = {0: human, 1: bot}

        async def choose_reveal(player, suggestion, matches, suggesting_player_id):
            agent = agents[player.id]
            observation = engine.observation_for(player.id)
            return await agent.choose_reveal(
                observation, suggestion, matches, suggesting_player_id
            )

        engine.choose_reveal = choose_reveal

        async def drive_human():
            while state.status != GameStatus.FINISHED:
                decision = human.pending
                if decision is not None and not decision.future.done():
                    if decision.kind == "action":
                        human.submit_action(PassTurn())
                    else:
                        human.submit_reveal(decision.offered[0])
                await asyncio.sleep(0)

        driver = asyncio.create_task(drive_human())
        try:
            for _ in range(200):
                if state.status == GameStatus.FINISHED:
                    break
                agent = agents[state.current_player_id]
                observation = engine.observation_for(agent.player_id)
                action = await agent.choose_action(observation)
                await engine.take_turn(agent.player_id, action)
        finally:
            driver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await driver

        return state

    state = run(scenario())
    assert state.status in (GameStatus.FINISHED, GameStatus.IN_PROGRESS)
    assert state.turn_number > 0
