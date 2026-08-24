from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from models.card import PersonValue, RoomValue, WeaponValue


class MakeSuggestion(BaseModel):
    kind: Literal["suggestion"] = "suggestion"
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue


class MakeAccusation(BaseModel):
    kind: Literal["accusation"] = "accusation"
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue


class PassTurn(BaseModel):
    kind: Literal["pass"] = "pass"


PlayerAction = Annotated[
    Union[MakeSuggestion, MakeAccusation, PassTurn], Field(discriminator="kind")
]
