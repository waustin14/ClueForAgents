from game.rules import (
    can_player_disprove,
    cards_that_can_disprove,
    is_correct_accusation,
    next_player_id,
    turn_order_after,
)
from models.card import PersonCard, PersonValue, RoomCard, RoomValue, WeaponCard, WeaponValue
from models.events import Suggestion
from models.game_state import Solution
from models.player import Player


def make_player(pid: int, cards: list) -> Player:
    return Player(id=pid, cards=cards)


def test_cards_that_can_disprove_finds_matching_cards():
    suggestion = Suggestion(
        person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY
    )
    player = make_player(0, [PersonCard(value=PersonValue.PLUM), WeaponCard(value=WeaponValue.KNIFE)])

    matches = cards_that_can_disprove(player, suggestion)

    assert matches == [PersonCard(value=PersonValue.PLUM)]
    assert can_player_disprove(player, suggestion) is True


def test_cards_that_can_disprove_returns_empty_when_no_match():
    suggestion = Suggestion(
        person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY
    )
    player = make_player(0, [PersonCard(value=PersonValue.GREEN), WeaponCard(value=WeaponValue.KNIFE)])

    assert cards_that_can_disprove(player, suggestion) == []
    assert can_player_disprove(player, suggestion) is False


def test_is_correct_accusation():
    solution = Solution(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)
    correct = Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)
    wrong = Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.KITCHEN)

    assert is_correct_accusation(correct, solution) is True
    assert is_correct_accusation(wrong, solution) is False


def test_next_player_id_wraps_around():
    players = [make_player(i, []) for i in range(4)]

    assert next_player_id(0, players) == 1
    assert next_player_id(3, players) == 0


def test_turn_order_after_excludes_self_and_wraps():
    players = [make_player(i, []) for i in range(4)]

    assert [p.id for p in turn_order_after(1, players)] == [2, 3, 0]
