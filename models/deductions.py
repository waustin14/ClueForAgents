from pydantic import BaseModel, Field

from models.card import CardValue
from models.events import Suggestion


class PlayerDeduction(BaseModel):
    """What the public event log alone proves about one player's hand.

    Both fields are facts a spectator could write down without seeing a
    single card: `cannot_have` comes from suggestions the player was asked
    about and could not disprove, `holds_at_least_one_of` from suggestions
    they did disprove (which card they showed stays private to the two
    players involved).
    """

    player_id: int
    cannot_have: list[CardValue] = Field(default_factory=list)
    holds_at_least_one_of: list[list[CardValue]] = Field(default_factory=list)


class PublicDeductions(BaseModel):
    per_player: list[PlayerDeduction] = Field(default_factory=list)
    disproven_triples: list[Suggestion] = Field(default_factory=list)
