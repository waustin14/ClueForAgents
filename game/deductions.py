"""Bookkeeping over the public event log.

Everything derived here is already visible to every player — and to a
spectator — in `GameState.public_event_history`. The engine computes it once
per observation so that every agent doesn't have to re-derive the same facts
from raw event JSON on every single turn, which is both the most expensive
and the least interesting part of the reasoning an LLM seat does.

The line this module deliberately does not cross: every fact it records is
about one player, derived from the public log alone. It never cross-
references one player against another, never intersects anything with the
observer's hand, and never narrows the envelope. Working out what the
solution must be from these facts is the part of Clue the agents are here to
play, and doing it for them would make the benchmark measure nothing.

Within a single player, though, it does tidy up: see `simplify_groups`. That
is bookkeeping on facts already stated, not deduction — it removes lines the
reader would otherwise have to re-collapse from scratch on every turn.
"""

from models.card import CardValue
from models.deductions import PlayerDeduction, PublicDeductions
from models.events import AccusationEvent, PublicGameEvent, Suggestion, SuggestionEvent


def cards_in(suggestion: Suggestion) -> list[CardValue]:
    return [suggestion.person, suggestion.weapon, suggestion.room]


def simplify_groups(
    groups: list[list[CardValue]], cannot_have: list[CardValue]
) -> list[list[CardValue]]:
    """Collapses one player's "holds at least one of" groups to their
    smallest equivalent form, using only that same player's public facts.

    Three reductions, none of which adds information:

    - A card the player is publicly proven not to hold drops out of the
      group. `holds none of: Green` and `holds one of [Green, Rope, Study]`
      on the same line just means `holds one of [Rope, Study]`.
    - Identical groups collapse to one. A player who disproved the same
      suggestion twice — common, since two seats often suggest the same
      triple early — proves nothing the second time. In one traced game half
      the rendered rows were this.
    - A group that is a strict superset of another drops out, because the
      smaller group already implies it. Narrowing tends to create these.

    Kept here rather than in the prompt renderer so that every consumer sees
    the same collapsed form, and so the reduction is testable on its own.
    """
    excluded = set(cannot_have)
    narrowed: list[list[CardValue]] = []
    for group in groups:
        kept = [card for card in group if card not in excluded]
        # An empty result would mean the player is proven both to hold one of
        # a set and none of it. That is unreachable from a consistent log, so
        # it is dropped rather than given a rendering of its own.
        if kept and not any(set(kept) == set(seen) for seen in narrowed):
            narrowed.append(kept)
    return [
        group
        for group in narrowed
        if not any(other is not group and set(other) < set(group) for other in narrowed)
    ]


def derive_public_deductions(
    public_history: list[PublicGameEvent], player_ids: list[int]
) -> PublicDeductions:
    cannot_have: dict[int, list[CardValue]] = {player_id: [] for player_id in player_ids}
    holds_one_of: dict[int, list[list[CardValue]]] = {player_id: [] for player_id in player_ids}
    disproven: list[Suggestion] = []

    for event in public_history:
        if isinstance(event, SuggestionEvent):
            triple = cards_in(event.suggestion)
            for player_id in event.unable_to_disprove:
                known = cannot_have.setdefault(player_id, [])
                known.extend(card for card in triple if card not in known)
            if event.disproving_player_id is not None:
                holds_one_of.setdefault(event.disproving_player_id, []).append(triple)
        elif isinstance(event, AccusationEvent) and not event.correct:
            disproven.append(event.accusation)

    return PublicDeductions(
        per_player=[
            PlayerDeduction(
                player_id=player_id,
                cannot_have=cannot_have.get(player_id, []),
                holds_at_least_one_of=simplify_groups(
                    holds_one_of.get(player_id, []), cannot_have.get(player_id, [])
                ),
            )
            for player_id in player_ids
        ],
        disproven_triples=disproven,
    )
