from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from card import Card, PersonValue, WeaponValue, RoomValue

class LogEntry(BaseModel):
    seen: bool
    who_has: int | None

class PlayerLog(BaseModel):
    people: Dict[PersonValue, LogEntry]
    weapons: Dict[WeaponValue, LogEntry]
    rooms: Dict[RoomValue, LogEntry]

class Player(BaseModel):
    id: int
    name: str | None
    piece: str | None
    cards: List[Card]
    log: PlayerLog
