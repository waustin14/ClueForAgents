from typing import Union

from pydantic import BaseModel, Field

from models.card import Card, PersonValue, RoomValue, WeaponValue


class Suggestion(BaseModel):
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue


class SuggestionEvent(BaseModel):
    turn: int
    suggesting_player_id: int
    suggestion: Suggestion
    unable_to_disprove: list[int] = Field(default_factory=list)
    disproving_player_id: int | None = None


class AccusationEvent(BaseModel):
    turn: int
    accusing_player_id: int
    accusation: Suggestion
    correct: bool


class GameOverEvent(BaseModel):
    turn: int
    winning_player_id: int | None
    solution: Suggestion


class CardRevealEvent(BaseModel):
    turn: int
    revealing_player_id: int
    receiving_player_id: int
    card: Card


PublicGameEvent = Union[SuggestionEvent, AccusationEvent, GameOverEvent]
PrivateGameEvent = CardRevealEvent
GameEvent = Union[PublicGameEvent, PrivateGameEvent]
