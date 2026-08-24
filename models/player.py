from pydantic import BaseModel, Field

from models.card import Card, PersonValue, RoomValue, WeaponValue


class LogEntry(BaseModel):
    seen: bool = False
    who_has: int | None = None


class PlayerLog(BaseModel):
    people: dict[PersonValue, LogEntry] = Field(default_factory=dict)
    weapons: dict[WeaponValue, LogEntry] = Field(default_factory=dict)
    rooms: dict[RoomValue, LogEntry] = Field(default_factory=dict)

    @classmethod
    def empty(cls) -> "PlayerLog":
        return cls(
            people={value: LogEntry() for value in PersonValue},
            weapons={value: LogEntry() for value in WeaponValue},
            rooms={value: LogEntry() for value in RoomValue},
        )


class Player(BaseModel):
    id: int
    name: str | None = None
    piece: PersonValue | None = None
    active: bool = True
    cards: list[Card] = Field(default_factory=list)
    log: PlayerLog = Field(default_factory=PlayerLog.empty)
