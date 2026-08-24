from models.card import Card, PersonValue, WeaponValue, RoomValue
from models.game_state import GameState, Solution
from models.player import Player, PlayerLog
from random import choice, sample
from typing import Annotated, List, Union

def make_solution() -> Solution:
    return Solution(
        person=choice(list(PersonValue)),
        weapon=choice(list(WeaponValue)),
        room=choice(list(RoomValue))
    )

def make_deck(solution: Solution, shuffled: bool=True) -> List[Card]:
    deck = [
        *[Card(value=person) for person in PersonValue if person != solution.person],
        *[Card(value=weapon) for weapon in WeaponValue if weapon != solution.weapon],
        *[Card(value=room) for room in RoomValue if room != solution.room]
    ]
    if shuffled:
        shuffled_deck = shuffle_cards(deck)
        del deck
        return shuffled_deck
    else:
        return deck

def shuffle_cards(cards: List[Card]) -> List[Card]:
    return sample(cards, k=len(cards))

def init_log() -> PlayerLog:
    return PlayerLog(people={}, weapons={}, rooms={})

def init_players(n_players=6) -> List[Player]:
    if n_players not in range(2,7):
        raise ValueError("Number of players must be between 2 and 6")
    return[Player(id=i, name=None, piece=None, cards=[], log=init_log()) for i in range(n_players)]

def deal_cards(cards: List[Card], players: List[Player]) -> None:
    hands = [cards[i::len(players)] for i in range(len(players))]
    for i in range(len(players)):
        players[i].cards = hands[i]
    return

def init_game(n_players: int) -> GameState:
    solution = make_solution()
    deck = make_deck(solution, shuffled=True)
    players = init_players(n_players)
    deal_cards(deck, players)
    return GameState(players=players, solution=solution)
    