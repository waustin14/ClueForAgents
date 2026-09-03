from game.deductions import derive_public_deductions
from models.card import PersonValue, RoomValue, WeaponValue
from models.events import AccusationEvent, Suggestion, SuggestionEvent


def suggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY):
    return Suggestion(person=person, weapon=weapon, room=room)


def by_id(deductions, player_id):
    return next(p for p in deductions.per_player if p.player_id == player_id)


def test_unable_to_disprove_becomes_cannot_have():
    history = [
        SuggestionEvent(
            turn=0,
            suggesting_player_id=0,
            suggestion=suggestion(),
            unable_to_disprove=[1],
            disproving_player_id=2,
        )
    ]

    deductions = derive_public_deductions(history, [0, 1, 2])

    assert by_id(deductions, 1).cannot_have == [
        PersonValue.PLUM,
        WeaponValue.ROPE,
        RoomValue.STUDY,
    ]
    # The disprover holds one of the three, but nobody public knows which.
    assert by_id(deductions, 2).holds_at_least_one_of == [
        [PersonValue.PLUM, WeaponValue.ROPE, RoomValue.STUDY]
    ]
    assert by_id(deductions, 2).cannot_have == []


def test_suggester_is_never_credited_with_anything():
    # The suggester is not asked to disprove their own suggestion, so a
    # suggestion nobody could disprove says nothing at all about them.
    history = [
        SuggestionEvent(
            turn=0,
            suggesting_player_id=0,
            suggestion=suggestion(),
            unable_to_disprove=[1, 2],
            disproving_player_id=None,
        )
    ]

    deductions = derive_public_deductions(history, [0, 1, 2])

    assert by_id(deductions, 0).cannot_have == []
    assert by_id(deductions, 0).holds_at_least_one_of == []


def test_cannot_have_accumulates_without_duplicates():
    history = [
        SuggestionEvent(
            turn=0,
            suggesting_player_id=0,
            suggestion=suggestion(),
            unable_to_disprove=[1],
            disproving_player_id=None,
        ),
        SuggestionEvent(
            turn=1,
            suggesting_player_id=2,
            suggestion=suggestion(room=RoomValue.HALL),
            unable_to_disprove=[1],
            disproving_player_id=None,
        ),
    ]

    deductions = derive_public_deductions(history, [0, 1, 2])

    assert by_id(deductions, 1).cannot_have == [
        PersonValue.PLUM,
        WeaponValue.ROPE,
        RoomValue.STUDY,
        RoomValue.HALL,
    ]


def test_failed_accusations_are_recorded_and_correct_ones_are_not():
    history = [
        AccusationEvent(
            turn=0, accusing_player_id=0, accusation=suggestion(), correct=False
        ),
        AccusationEvent(
            turn=1,
            accusing_player_id=1,
            accusation=suggestion(room=RoomValue.HALL),
            correct=True,
        ),
    ]

    deductions = derive_public_deductions(history, [0, 1])

    assert deductions.disproven_triples == [suggestion()]


def test_derivation_uses_only_the_public_log():
    # Every player id gets an entry even with an empty history, and none of
    # them carries a fact — the solution is never narrowed here, because
    # working it out is the job this benchmark exists to measure.
    deductions = derive_public_deductions([], [0, 1, 2])

    assert [p.player_id for p in deductions.per_player] == [0, 1, 2]
    assert all(not p.cannot_have and not p.holds_at_least_one_of for p in deductions.per_player)
    assert deductions.disproven_triples == []


def test_the_same_triple_disproved_twice_is_stated_once():
    # Two seats suggesting the identical triple early is common, and the
    # second disproof proves nothing the first didn't. In one traced game
    # half of all rendered rows were this duplication.
    event = SuggestionEvent(
        turn=0,
        suggesting_player_id=0,
        suggestion=suggestion(),
        unable_to_disprove=[],
        disproving_player_id=2,
    )
    history = [event, event.model_copy(update={"turn": 1, "suggesting_player_id": 1})]

    deductions = derive_public_deductions(history, [0, 1, 2])

    assert by_id(deductions, 2).holds_at_least_one_of == [
        [PersonValue.PLUM, WeaponValue.ROPE, RoomValue.STUDY]
    ]


def test_a_card_the_player_cannot_have_drops_out_of_their_own_groups():
    history = [
        # Player 2 could not disprove Plum/Rope/Lounge, so holds none of those.
        SuggestionEvent(
            turn=0,
            suggesting_player_id=0,
            suggestion=suggestion(room=RoomValue.LOUNGE),
            unable_to_disprove=[2],
            disproving_player_id=1,
        ),
        # ...then disproved Plum/Rope/Study, which can only have been Study.
        SuggestionEvent(
            turn=1,
            suggesting_player_id=0,
            suggestion=suggestion(),
            unable_to_disprove=[],
            disproving_player_id=2,
        ),
    ]

    deductions = derive_public_deductions(history, [0, 1, 2])

    # Leaving Plum and Rope in would make the reader re-collapse a line that
    # contradicts the `holds none of` clause beside it, every single turn.
    assert by_id(deductions, 2).holds_at_least_one_of == [[RoomValue.STUDY]]


def test_a_group_implied_by_a_narrower_one_is_dropped():
    history = [
        SuggestionEvent(
            turn=0,
            suggesting_player_id=0,
            suggestion=suggestion(),
            unable_to_disprove=[1],
            disproving_player_id=2,
        ),
        # Player 2 disproves a second triple sharing Plum and Rope with the
        # first. On its own that adds a real, separate fact.
        SuggestionEvent(
            turn=1,
            suggesting_player_id=0,
            suggestion=suggestion(room=RoomValue.LOUNGE),
            unable_to_disprove=[],
            disproving_player_id=2,
        ),
        # But then player 2 is proven to hold no Study, which narrows the
        # first group to [Plum, Rope] — and that now implies the second.
        SuggestionEvent(
            turn=2,
            suggesting_player_id=0,
            suggestion=suggestion(person=PersonValue.GREEN, weapon=WeaponValue.KNIFE),
            unable_to_disprove=[2],
            disproving_player_id=1,
        ),
    ]

    deductions = derive_public_deductions(history, [0, 1, 2])

    assert by_id(deductions, 2).holds_at_least_one_of == [[PersonValue.PLUM, WeaponValue.ROPE]]
