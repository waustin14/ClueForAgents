from enum import Enum

from pydantic import BaseModel, Field

from models.card import PersonValue, RoomValue, WeaponValue
from models.events import CardRevealEvent, PublicGameEvent
from models.player import Player


class Solution(BaseModel):
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue


class GameStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"


class GameState(BaseModel):
    players: list[Player] = Field(default_factory=list)
    solution: Solution
    current_player_id: int = 0
    turn_number: int = 0
    status: GameStatus = GameStatus.IN_PROGRESS
    winning_player_id: int | None = None
    public_event_history: list[PublicGameEvent] = Field(default_factory=list)
    private_reveal_history: list[CardRevealEvent] = Field(default_factory=list)
