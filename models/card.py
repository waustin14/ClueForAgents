from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class PersonValue(str, Enum):
    GREEN = "Mr. Green"
    MUSTARD = "Colonel Mustard"
    PEACOCK = "Mrs. Peacock"
    PLUM = "Professor Plum"
    SCARLET = "Miss Scarlet"
    WHITE = "Mrs. White"


class WeaponValue(str, Enum):
    CANDLESTICK = "Candlestick"
    KNIFE = "Knife"
    PIPE = "Lead Pipe"
    REVOLVER = "Revolver"
    ROPE = "Rope"
    WRENCH = "Wrench"


class RoomValue(str, Enum):
    BALLROOM = "Ball Room"
    BILLIARD = "Billiard Room"
    CONSERVATORY = "Conservatory"
    DINING = "Dining Room"
    HALL = "Hall"
    KITCHEN = "Kitchen"
    LIBRARY = "Library"
    LOUNGE = "Lounge"
    STUDY = "Study"


CardValue = Union[PersonValue, WeaponValue, RoomValue]


class PersonCard(BaseModel):
    type: Literal["person"] = "person"
    value: PersonValue


class WeaponCard(BaseModel):
    type: Literal["weapon"] = "weapon"
    value: WeaponValue


class RoomCard(BaseModel):
    type: Literal["room"] = "room"
    value: RoomValue


Card = Annotated[Union[PersonCard, WeaponCard, RoomCard], Field(discriminator="type")]


def card_for_value(value: CardValue) -> "PersonCard | WeaponCard | RoomCard":
    if isinstance(value, PersonValue):
        return PersonCard(value=value)
    if isinstance(value, WeaponValue):
        return WeaponCard(value=value)
    if isinstance(value, RoomValue):
        return RoomCard(value=value)
    raise TypeError(f"Unsupported card value: {value!r}")
