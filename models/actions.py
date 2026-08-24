from typing import Union

from pydantic import BaseModel

from models.card import PersonValue, RoomValue, WeaponValue


class MakeSuggestion(BaseModel):
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue


class MakeAccusation(BaseModel):
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue


class PassTurn(BaseModel):
    pass


PlayerAction = Union[MakeSuggestion, MakeAccusation, PassTurn]
