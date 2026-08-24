from typing import List
from pydantic import BaseModel, Field

from models.player import Player
from card import PersonValue, WeaponValue, RoomValue

class Solution(BaseModel):
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue

class GameState(BaseModel):
    players: List[Player]
    solution: Solution