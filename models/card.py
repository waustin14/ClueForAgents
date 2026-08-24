from enum import Enum
from typing import Union
from pydantic import BaseModel

class PersonValue(str, Enum):
    GREEN = 'Mr. Green'
    MUSTARD = 'Colonel Mustard'
    PEACOCK = 'Mrs. Peacock'
    PLUM = 'Professor Plum'
    SCARLET = 'Miss Scarlet'
    WHITE = 'Mrs. White'

class WeaponValue(str, Enum):
    CANDLESTICK = 'Candlestick'
    KNIFE = 'Knife'
    PIPE = 'Lead Pipe'
    REVOLVER = 'Revolver'
    ROPE = 'Rope'
    WRENCH = 'Wrench'

class RoomValue(str, Enum):
    BALLROOM = 'Ball Room'
    BILLIARD = 'Billiard Room'
    CONSERVATORY = 'Conservatory'
    DINING = 'Dining Room'
    HALL = 'Hall'
    KITCHEN = 'Kitchen'
    LIBRARY = 'Library'
    LOUNGE = 'Lounge'
    STUDY = 'Study'

CardValue = Union[PersonValue, WeaponValue, RoomValue]

class Card(BaseModel):
    value: CardValue

    @property
    def type(self) -> str:
        if isinstance(self.value, PersonValue):
            return "person"
        if isinstance(self.value, WeaponValue):
            return "weapon"
        if isinstance(self.value, RoomValue):
            return "room"

        raise ValueError(f"Unknown card type: {self.value}")