from pydantic import BaseModel

from .card import Card, PersonValue, WeaponValue, RoomValue

class Suggestion(BaseModel):
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue

class SuggestionEvent(BaseModel):
    turn: int
    suggesting_player_id: int
    suggestion: Suggestion
    unable_to_disprove: list[int]
    disproving_player_id: int | None = None

class CardRevealEvent(BaseModel):
    turn: int
    revealing_player_id: int
    receiving_player_id: int
    card: Card