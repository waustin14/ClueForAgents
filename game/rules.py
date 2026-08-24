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


def next_player_id(current_player_id: int, players: list[Player]) -> int:
    ids = [player.id for player in players]
    return ids[(ids.index(current_player_id) + 1) % len(ids)]


def turn_order_after(player_id: int, players: list[Player]) -> list[Player]:
    """All other players, in seating order starting right after player_id."""
    ids = [player.id for player in players]
    start = ids.index(player_id)
    return players[start + 1 :] + players[:start]
