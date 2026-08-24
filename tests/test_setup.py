from game.setup import (
    create_players,
    deal_cards,
    make_deck,
    make_solution,
    shuffle_cards,
)
from models.card import PersonValue, RoomValue, WeaponValue


def test_make_deck_excludes_solution_cards():
    solution = make_solution()
    deck = make_deck(solution)

    values = {card.value for card in deck}
    assert solution.person not in values
    assert solution.weapon not in values
    assert solution.room not in values
    assert len(deck) == (len(PersonValue) + len(WeaponValue) + len(RoomValue) - 3)


def test_shuffle_cards_is_a_permutation():
    solution = make_solution()
    deck = make_deck(solution)
    shuffled = shuffle_cards(deck)

    assert len(shuffled) == len(deck)
    assert set(id(c) for c in deck) == set(id(c) for c in shuffled)


def test_create_players_rejects_bad_counts():
    for bad_count in (0, 1, 7, 10):
        try:
            create_players(bad_count)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad_count} players")


def test_deal_cards_distributes_all_cards_without_overlap():
    solution = make_solution()
    deck = shuffle_cards(make_deck(solution))
    players = create_players(4)

    deal_cards(deck, players)

    dealt = [card for player in players for card in player.cards]
    assert len(dealt) == len(deck)
    assert len(set(id(c) for c in dealt)) == len(deck)


def test_deal_cards_records_own_cards_in_log():
    solution = make_solution()
    deck = shuffle_cards(make_deck(solution))
    players = create_players(3)

    deal_cards(deck, players)

    for player in players:
        for card in player.cards:
            if card.type == "person":
                entry = player.log.people[card.value]
            elif card.type == "weapon":
                entry = player.log.weapons[card.value]
            else:
                entry = player.log.rooms[card.value]
            assert entry.seen is True
            assert entry.who_has == player.id


def test_undealt_log_entries_default_to_unseen():
    players = create_players(2)
    player = players[0]
    for entry in player.log.people.values():
        assert entry.seen is False
        assert entry.who_has is None
