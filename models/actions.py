from pydantic import BaseModel

from .card import PersonValue, WeaponValue, RoomValue

class MakeSuggestion(BaseModel):
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue


class MakeAccusation(BaseModel):
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue