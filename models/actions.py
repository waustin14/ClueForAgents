from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from models.card import PersonValue, RoomValue, WeaponValue

# Every action carries an optional `notes` field: the agent's own private
# working notes, handed back to it on its next turn as a scratchpad (see
# `agents.llm.LLMClueAgent`). It is subjective, never authoritative, and the
# engine neither reads it nor shows it to anyone else — the spectator feed
# strips it before publishing.
NOTES_FIELD = Field(
    default=None,
    description=(
        "Your own private notes to carry into your next turn: deductions you "
        "have made, what you are trying to find out, who you suspect. Only you "
        "ever see them."
    ),
)


class MakeSuggestion(BaseModel):
    kind: Literal["suggestion"] = "suggestion"
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue
    notes: str | None = NOTES_FIELD


class MakeAccusation(BaseModel):
    kind: Literal["accusation"] = "accusation"
    person: PersonValue
    weapon: WeaponValue
    room: RoomValue
    notes: str | None = NOTES_FIELD


class PassTurn(BaseModel):
    kind: Literal["pass"] = "pass"
    notes: str | None = NOTES_FIELD


PlayerAction = Annotated[
    Union[MakeSuggestion, MakeAccusation, PassTurn], Field(discriminator="kind")
]
