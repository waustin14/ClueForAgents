from random import choice, sample

from models.card import (
    Card,
    PersonValue,
    RoomValue,
    WeaponValue,
    card_for_value,
)
from models.game_state import GameState, Solution
from models.player import LogEntry, Player

MIN_PLAYERS = 2
MAX_PLAYERS = 6


def make_solution() -> Solution:
    return Solution(
        person=choice(list(PersonValue)),
        weapon=choice(list(WeaponValue)),
        room=choice(list(RoomValue)),
    )


def make_deck(solution: Solution) -> list[Card]:
    return [
        *(card_for_value(value) for value in PersonValue if value != solution.person),
        *(card_for_value(value) for value in WeaponValue if value != solution.weapon),
        *(card_for_value(value) for value in RoomValue if value != solution.room),
    ]


def shuffle_cards(cards: list[Card]) -> list[Card]:
    return sample(cards, k=len(cards))


def create_players(n_players: int) -> list[Player]:
    if not MIN_PLAYERS <= n_players <= MAX_PLAYERS:
        raise ValueError(
            f"Number of players must be between {MIN_PLAYERS} and {MAX_PLAYERS}"
        )
    return [Player(id=i) for i in range(n_players)]


def _record_own_card(player: Player, card: Card) -> None:
    entry = LogEntry(seen=True, who_has=player.id)
    if card.type == "person":
        player.log.people[card.value] = entry
    elif card.type == "weapon":
        player.log.weapons[card.value] = entry
    else:
        player.log.rooms[card.value] = entry


def deal_cards(deck: list[Card], players: list[Player]) -> None:
    hands = [deck[i :: len(players)] for i in range(len(players))]
    for player, hand in zip(players, hands):
        player.cards = hand
        for card in hand:
            _record_own_card(player, card)


def initialize_game(n_players: int) -> GameState:
    solution = make_solution()
    deck = shuffle_cards(make_deck(solution))
    players = create_players(n_players)
    deal_cards(deck, players)
    return GameState(players=players, solution=solution)
