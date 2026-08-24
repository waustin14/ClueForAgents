from pydantic import BaseModel, Field

from models.card import Card
from models.events import CardRevealEvent, PublicGameEvent
from models.player import PlayerLog


class PlayerObservation(BaseModel):
    player_id: int
    turn_number: int
    current_player_id: int
    active: bool = True
    own_cards: list[Card] = Field(default_factory=list)
    card_log: PlayerLog
    public_history: list[PublicGameEvent] = Field(default_factory=list)
    private_reveals: list[CardRevealEvent] = Field(default_factory=list)
