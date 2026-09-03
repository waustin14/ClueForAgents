from models.card import PersonCard, RoomCard, WeaponCard
from models.events import Suggestion
from models.game_state import Solution
from models.player import Player


def cards_that_can_disprove(player: Player, suggestion: Suggestion) -> list:
    matches = []
    for card in player.cards:
        if isinstance(card, PersonCard) and card.value == suggestion.person:
            matches.append(card)
        elif isinstance(card, WeaponCard) and card.value == suggestion.weapon:
            matches.append(card)
        elif isinstance(card, RoomCard) and card.value == suggestion.room:
            matches.append(card)
    return matches


def can_player_disprove(player: Player, suggestion: Suggestion) -> bool:
    return len(cards_that_can_disprove(player, suggestion)) > 0


def is_correct_accusation(accusation: Suggestion, solution: Solution) -> bool:
    return (
        accusation.person == solution.person
        and accusation.weapon == solution.weapon
        and accusation.room == solution.room
    )


def active_players(players: list[Player]) -> list[Player]:
    return [player for player in players if player.active]


def next_player_id(current_player_id: int, players: list[Player]) -> int:
    """The next *active* seat after `current_player_id`.

    Eliminated players still hold cards and are still asked to disprove
    suggestions (see `turn_order_after`), but they can never take a turn of
    their own again: the only action the engine accepts from them is a pass.
    Cycling through them burns a turn, and for an LLM-backed seat it burns a
    real provider call on a decision with exactly one legal answer.

    Falls back to `current_player_id` when nobody is active at all, which
    only happens in a game the engine has already finished.
    """
    ids = [player.id for player in players]
    start = ids.index(current_player_id)
    for offset in range(1, len(players) + 1):
        candidate = players[(start + offset) % len(players)]
        if candidate.active:
            return candidate.id
    return current_player_id


def turn_order_after(player_id: int, players: list[Player]) -> list[Player]:
    """All other players, in seating order starting right after player_id.

    Deliberately includes eliminated players: being out of the game stops you
    taking turns, not holding cards, and an eliminated player must still show
    a matching card when someone's suggestion reaches them.
    """
    ids = [player.id for player in players]
    start = ids.index(player_id)
    return players[start + 1 :] + players[:start]
