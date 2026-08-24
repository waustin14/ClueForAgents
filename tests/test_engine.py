import asyncio

from game.engine import GameEngine
from models.actions import MakeAccusation, MakeSuggestion, PassTurn
from models.card import PersonCard, PersonValue, RoomCard, RoomValue, WeaponCard, WeaponValue
from models.game_state import GameState, GameStatus, Solution
from models.player import Player
from transport.local import LocalTransport


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


def test_resolve_suggestion_finds_disprover_and_reveals_privately():
    state = build_state()
    transport = LocalTransport()
    engine = GameEngine(state, transport)
    suggestion_action = MakeSuggestion(
        person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY
    )

    event = run(engine.take_turn(0, suggestion_action))

    assert event.disproving_player_id == 2
    assert event.unable_to_disprove == [1]
    assert len(transport.broadcast_log) == 1
    # Only the suggester and the revealer received the private card reveal.
    assert len(transport.send_log[0]) == 1
    assert len(transport.send_log[2]) == 1
    assert 1 not in transport.send_log

    # The suggesting player now definitively knows who holds the room card.
    assert state.players[0].log.rooms[RoomValue.STUDY].who_has == 2
    assert state.players[0].log.rooms[RoomValue.STUDY].seen is True

    # Bystander player 1 only knows a disprove happened, not which card.
    assert state.players[1].log.rooms[RoomValue.STUDY].seen is False

    # Turn advances to the next player.
    assert state.current_player_id == 1
    assert state.turn_number == 1


def test_resolve_suggestion_with_no_disprovers():
    state = build_state()
    state.players[2].cards = [PersonCard(value=PersonValue.WHITE)]
    transport = LocalTransport()
    engine = GameEngine(state, transport)
    suggestion_action = MakeSuggestion(
        person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY
    )

    event = run(engine.take_turn(0, suggestion_action))

    assert event.disproving_player_id is None
    assert sorted(event.unable_to_disprove) == [1, 2]
    assert transport.send_log == {}


def test_correct_accusation_ends_game():
    state = build_state()
    transport = LocalTransport()
    engine = GameEngine(state, transport)
    accusation = MakeAccusation(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)

    run(engine.take_turn(0, accusation))

    assert state.status == GameStatus.FINISHED
    assert state.winning_player_id == 0
    assert transport.broadcast_log[-1].winning_player_id == 0


def test_wrong_accusation_eliminates_player_but_game_continues():
    state = build_state()
    transport = LocalTransport()
    engine = GameEngine(state, transport)
    accusation = MakeAccusation(person=PersonValue.GREEN, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)

    run(engine.take_turn(0, accusation))

    assert state.status == GameStatus.IN_PROGRESS
    assert state.players[0].active is False
    assert state.current_player_id == 1


def test_eliminated_player_cannot_suggest_or_accuse():
    state = build_state()
    state.players[0].active = False
    transport = LocalTransport()
    engine = GameEngine(state, transport)
    suggestion_action = MakeSuggestion(
        person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY
    )

    try:
        run(engine.take_turn(0, suggestion_action))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for eliminated player suggesting")


def test_pass_turn_advances_without_event():
    state = build_state()
    transport = LocalTransport()
    engine = GameEngine(state, transport)

    event = run(engine.take_turn(0, PassTurn()))

    assert event is None
    assert state.current_player_id == 1
    assert transport.broadcast_log == []


def test_observation_for_filters_private_reveals_to_participants():
    state = build_state()
    transport = LocalTransport()
    engine = GameEngine(state, transport)
    suggestion_action = MakeSuggestion(
        person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY
    )

    run(engine.take_turn(0, suggestion_action))

    obs_suggester = engine.observation_for(0)
    obs_bystander = engine.observation_for(1)
    obs_revealer = engine.observation_for(2)

    assert len(obs_suggester.private_reveals) == 1
    assert len(obs_revealer.private_reveals) == 1
    assert len(obs_bystander.private_reveals) == 0
    # Everyone sees the same public history.
    assert len(obs_bystander.public_history) == 1
