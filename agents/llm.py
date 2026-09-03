import re
from dataclasses import dataclass
from functools import lru_cache
from random import choice
from typing import Any, Literal, Protocol, TypeVar

from opentelemetry import trace
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, create_model

from agents.base import ClueAgent
from models.actions import MakeAccusation, MakeSuggestion, PassTurn, PlayerAction
from models.card import Card, PersonValue, RoomValue, WeaponValue
from models.deductions import PublicDeductions
from models.events import Suggestion
from models.observation import PlayerObservation
from models.player import PlayerLog

_player_action_adapter: TypeAdapter[PlayerAction] = TypeAdapter(PlayerAction)

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass
class PromptSegment:
    """One piece of a prompt, tagged with whether its bytes are stable.

    `cacheable=True` segments are only ever appended to or left untouched
    turn-over-turn for a given agent, so a client backed by a provider with
    prefix caching (e.g. Anthropic's `cache_control`) can mark the boundary
    after the last cacheable segment and pay full price only for what
    changed. A client for a provider without explicit caching can simply
    ignore the flag and concatenate everything.
    """

    text: str
    cacheable: bool = False


class LLMClient(Protocol):
    """Minimal interface an LLM agent needs from a model provider.

    Concrete implementations (Anthropic, OpenAI, a local model, a scripted
    stub for tests) live outside this module so the agent layer never
    depends on a specific SDK.

    `complete_structured` is the preferred path — it constrains the response
    to a schema instead of asking politely for JSON in the prompt and hoping.
    It is optional: a client that only implements `complete` still works, and
    `LLMClueAgent` falls back to parsing the raw text.

    `metadata` describes the decision being made — which seat, which turn,
    which of the two decision types — and exists so the trace backend can
    group calls by seat and model instead of showing an undifferentiated
    list of chat completions. A client is free to ignore it.
    """

    async def complete(
        self, segments: list[PromptSegment], metadata: dict[str, Any] | None = None
    ) -> str: ...

    async def complete_structured(
        self,
        segments: list[PromptSegment],
        schema: type[ModelT],
        metadata: dict[str, Any] | None = None,
    ) -> ModelT: ...


class TurnDecision(BaseModel):
    """Flat response schema for a turn decision.

    `PlayerAction` is a discriminated union, which serializes to a top-level
    `oneOf` JSON schema that several providers' structured-output modes
    reject outright. A single flat object with a `kind` selector is the shape
    every provider handles, so this is what goes over the wire; `to_action()`
    converts it back into the engine's typed action.
    """

    kind: Literal["suggestion", "accusation", "pass"]
    person: PersonValue | None = None
    weapon: WeaponValue | None = None
    room: RoomValue | None = None

    def scratchpad_notes(self) -> str | None:
        """The private memo, on the variants of this schema that carry one.

        Notes are not a field here because a seat playing without a
        scratchpad must not be offered one: the field's own description is
        an invitation, and a model handed that invitation writes a paragraph
        of deduction that is then thrown away. Absent the field, the
        provider's structured-output mode makes writing one impossible
        rather than merely pointless.
        """
        return getattr(self, "notes", None)

    def to_action(self) -> PlayerAction:
        notes = self.scratchpad_notes()
        if self.kind == "pass":
            return PassTurn(notes=notes)
        if self.person is None or self.weapon is None or self.room is None:
            raise ValueError(
                f"a {self.kind} needs person, weapon and room; got "
                f"{self.person!r}, {self.weapon!r}, {self.room!r}"
            )
        action_type = MakeSuggestion if self.kind == "suggestion" else MakeAccusation
        return action_type(person=self.person, weapon=self.weapon, room=self.room, notes=notes)


class TurnDecisionWithNotes(TurnDecision):
    """`TurnDecision` for a seat whose scratchpad is switched on."""

    notes: str | None = Field(
        default=None,
        description=(
            "Your own private notes to carry into your next turn: what you "
            "have deduced, what you are trying to find out, who you suspect. "
            "Only you ever see them."
        ),
    )


def turn_schema(with_notes: bool) -> type[TurnDecision]:
    """The turn-decision schema for a seat, with or without the scratchpad."""
    return TurnDecisionWithNotes if with_notes else TurnDecision


@lru_cache(maxsize=None)
def reveal_schema(offered: tuple[str, ...], with_notes: bool) -> type[BaseModel]:
    """Builds the "which card do you show?" schema for one specific reveal.

    `card` is a `Literal` over the names actually on offer, so the provider's
    structured-output mode cannot return anything else. It has to be built
    per call because the legal set is the two or three cards in this player's
    hand that disprove this suggestion — narrower than any fixed enum could
    express, and different every time.

    That constraint is not decorative. Left as a free string, models named a
    card they had already shown this opponent but that was not among the ones
    offered, and the resolver quietly substituted a card of its own choosing
    — leaking a fresh card to an opponent the seat was actively trying not to
    inform.

    Cached because the schema is derived purely from its arguments, and a
    provider re-deriving (and re-uploading) an identical tool schema on every
    reveal is pure waste.
    """
    fields: dict[str, Any] = {
        "card": (
            Literal[offered],  # type: ignore[valid-type]
            Field(description="The card you are showing. Must be one you were offered."),
        )
    }
    if with_notes:
        fields["notes"] = (
            str | None,
            Field(
                default=None,
                description="Your own private notes to carry into your next turn.",
            ),
        )
    return create_model("RevealDecision", **fields)


_RULES = [
    "You are playing Clue (Cluedo).",
    "",
    "How the game works:",
    "- The deck is 6 people, 6 weapons and 9 rooms — 21 cards. One person, one"
    " weapon and one room are removed at the start, face down, and are the"
    " secret solution. Every other card is dealt out to the players, so any"
    " card you can prove somebody holds is NOT part of the solution.",
    "- On your turn you may make a suggestion, make an accusation, or pass.",
    "- A suggestion names a person, a weapon and a room. Each other player is"
    " asked in seating order, starting with the one after you, whether they"
    " hold any of the three. The FIRST player who holds one shows you exactly"
    " one matching card, privately, and the questioning stops there — players"
    " after them are never asked.",
    "- IMPORTANT: you are never asked to disprove your own suggestion, so"
    " `unable_to_disprove` lists only the players who were actually asked and"
    " held none of the three. A suggestion nobody could disprove does NOT"
    " prove the solution: the suggester may be holding one of the cards"
    " themselves, and anyone after the disprover was never asked. Treating an"
    " undisproved suggestion as proof is how players lose this game.",
    "- An accusation is the only way to win. If it is right you win"
    " immediately. If it is wrong you are eliminated permanently — you still"
    " show cards to disprove other players, but you never take another turn.",
    "- A suggestion is only ever an information-gathering move. Once you can"
    " prove the solution, ACCUSE — do not suggest it first. Suggesting it"
    " announces your deduction to everyone still playing and lets one of them"
    " accuse it before your next turn comes round.",
]

_CARD_GROUPS: list[tuple[str, type, str]] = [
    ("people", PersonValue, "people"),
    ("weapons", WeaponValue, "weapons"),
    ("rooms", RoomValue, "rooms"),
]


def _render_card_knowledge(card_log: PlayerLog, own_player_id: int) -> str:
    """Renders the card log as accounted-for vs still-open.

    The log's own JSON dump spends most of its bytes restating that a card is
    unknown, and buries the one thing that matters — which cards are still
    candidates for the envelope — in a wall of `"seen":false`. This says the
    same thing in a fraction of the tokens, in the form the decision actually
    needs.
    """
    accounted: list[str] = []
    open_lines: list[str] = []

    for log_attr, enum_type, label in _CARD_GROUPS:
        entries = getattr(card_log, log_attr)
        unknown: list[str] = []
        for value in enum_type:
            entry = entries.get(value)
            if entry is not None and entry.seen:
                holder = (
                    "you" if entry.who_has == own_player_id else f"player {entry.who_has}"
                )
                accounted.append(f"{value.value} ({holder})")
            else:
                unknown.append(value.value)
        open_lines.append(f"  {label}: {', '.join(unknown) if unknown else '(none left)'}")

    lines = [
        "Cards you have accounted for — somebody holds these, so none of them is"
        " in the solution:",
        f"  {', '.join(accounted) if accounted else '(none yet)'}",
        "Cards still unaccounted for — the solution is exactly one person, one"
        " weapon and one room from these:",
        *open_lines,
    ]
    return "\n".join(lines)


def render_public_deductions(deductions: PublicDeductions, own_player_id: int) -> str:
    """Renders the facts the engine already derived from the public log.

    Public information only — every line here is something a spectator could
    write down from the event history. It is spelled out rather than left
    implicit because re-deriving it from raw event JSON, every turn, is what
    the traced agents were burning most of their reasoning budget on.
    """
    lines = ["Proven from the public log (no private information here):"]
    for player in deductions.per_player:
        who = f"player {player.player_id}" + (" (you)" if player.player_id == own_player_id else "")
        facts: list[str] = []
        # A group `simplify_groups` narrowed to a single card is no longer a
        # disjunction — it is a proven holding, and saying "at least one card
        # from each of: [Rope]" makes the reader work that out again.
        proven = [g[0].value for g in player.holds_at_least_one_of if len(g) == 1]
        undecided = [g for g in player.holds_at_least_one_of if len(g) > 1]
        if proven:
            facts.append(f"holds: {', '.join(proven)}")
        if player.cannot_have:
            facts.append(f"holds none of: {', '.join(v.value for v in player.cannot_have)}")
        if undecided:
            groups = "; ".join(
                "[" + ", ".join(v.value for v in group) + "]" for group in undecided
            )
            facts.append(f"holds at least one card from each of: {groups}")
        lines.append(f"  {who}: {'; '.join(facts) if facts else 'nothing proven yet'}")

    if deductions.disproven_triples:
        wrong = "; ".join(
            f"{t.person.value} / {t.weapon.value} / {t.room.value}"
            for t in deductions.disproven_triples
        )
        lines.append(f"  Proven NOT to be the solution by a failed accusation: {wrong}")
    return "\n".join(lines)


def _stable_segments(observation: PlayerObservation, instructions: str) -> list[PromptSegment]:
    own_hand = f"Your hand: {[card.value.value for card in observation.own_cards]}"
    public_history = "Public event history:\n" + "\n".join(
        event.model_dump_json() for event in observation.public_history
    )
    private_reveals = "Private reveals you've seen:\n" + "\n".join(
        event.model_dump_json() for event in observation.private_reveals
    )
    return [
        PromptSegment(instructions, cacheable=True),
        PromptSegment(own_hand, cacheable=True),
        PromptSegment(public_history, cacheable=True),
        PromptSegment(private_reveals, cacheable=True),
    ]


_NOTES_INSTRUCTION = (
    " You may also use `notes` to write yourself a private memo that will be"
    " handed back to you on your next turn."
)


def build_prompt(
    observation: PlayerObservation, scratchpad: str = "", with_notes: bool = False
) -> list[PromptSegment]:
    """Builds the turn prompt as stable-prefix-first, volatile-suffix-last.

    `player_id` and `own_cards` never change after the deal; `public_history`
    and `private_reveals` only ever grow by appending. Those are cacheable.
    The card knowledge summary, the public deductions and the per-turn state
    are all recomputed each turn, so they go in the trailing, non-cacheable
    segment along with the scratchpad.

    `with_notes` must match the schema the answer will be parsed against —
    see `build_reveal_prompt` for why offering a scratchpad a seat doesn't
    have is worse than saying nothing.
    """
    notes_instruction = _NOTES_INSTRUCTION if with_notes else ""
    instructions = "\n".join(
        [
            *_RULES,
            "",
            "Respond with a single decision object. `kind` is one of"
            " 'suggestion', 'accusation' or 'pass'; a suggestion or accusation"
            f" also needs `person`, `weapon` and `room`.{notes_instruction}",
            "",
            f"You are player {observation.player_id}.",
        ]
    )

    volatile_lines = [
        f"Current player: {observation.current_player_id}.",
        f"Turn: {observation.turn_number}. You are still active: {observation.active}.",
        "",
        _render_card_knowledge(observation.card_log, observation.player_id),
        "",
        render_public_deductions(observation.public_deductions, observation.player_id),
    ]
    if scratchpad:
        volatile_lines += ["", f"Your private notes from previous turns: {scratchpad}"]

    return [
        *_stable_segments(observation, instructions),
        PromptSegment("\n".join(volatile_lines), cacheable=False),
    ]


def _already_shown_to(observation: PlayerObservation, other_player_id: int) -> list[str]:
    """The cards this player has already shown that opponent, in order."""
    return [
        reveal.card.value.value
        for reveal in observation.private_reveals
        if reveal.revealing_player_id == observation.player_id
        and reveal.receiving_player_id == other_player_id
    ]


def _reveal_history_line(
    observation: PlayerObservation, suggesting_player_id: int, offered: list[str]
) -> str:
    """States what this player has already shown that opponent, and says
    plainly whether any of it can be shown again right now.

    The bare list on its own read as an instruction: told to keep showing the
    same card and shown a card it had shown before, a model named that card
    even when it wasn't among the ones offered. Doing the intersection here
    turns a hint the model has to reconcile against the offered list into a
    fact it can act on directly.
    """
    shown_before = _already_shown_to(observation, suggesting_player_id)
    prefix = f"Cards you have already shown player {suggesting_player_id}"
    if not shown_before:
        return f"{prefix}: none yet, so whichever you pick is new information for them."
    repeatable = [name for name in shown_before if name in offered]
    if repeatable:
        return (
            f"{prefix}: {shown_before}. These are in the offered list above, so showing "
            f"one of them again tells them nothing new: {repeatable}."
        )
    return (
        f"{prefix}: {shown_before}. None of those is in the offered list above, so you "
        "cannot show one again here — whichever you pick is new information for them."
    )


def build_reveal_prompt(
    observation: PlayerObservation,
    suggestion: Suggestion,
    matches: list[Card],
    suggesting_player_id: int,
    scratchpad: str = "",
    with_notes: bool = False,
) -> list[PromptSegment]:
    """Builds the prompt for "you can disprove this — which card do you show?".

    Same stable prefix shape as `build_prompt` so the two share as much of an
    agent's cached context as the provider allows.

    `suggesting_player_id` is passed in rather than read off the observation:
    the observation's `current_player_id` is whoever's turn it is, which is
    the suggester only until a suggestion has been resolved, and naming the
    wrong opponent here would poison the one decision this prompt exists for.

    `with_notes` must match the schema the answer will be parsed against: a
    seat with its scratchpad off has no `notes` field to write into, so
    inviting it to write one here only wastes output tokens.
    """
    notes_instruction = _NOTES_INSTRUCTION if with_notes else ""
    instructions = "\n".join(
        [
            *_RULES,
            "",
            "Right now it is not your turn. Another player has made a"
            " suggestion and you are the first player asked who can disprove"
            " it, so you must show them exactly one of your matching cards.",
            "- Only the suggester sees which card you show. Everyone else"
            " learns only that you disproved it.",
            "- This is the only information that player gets from you, so"
            " choose deliberately. The standard play is to keep showing the"
            " SAME card to a given player whenever you can: that way one"
            " opponent learns about one card, instead of several opponents"
            " each learning about a different one.",
            "- That only applies when a card you have already shown them is"
            " among the ones offered below. A card missing from that list is"
            " one you cannot show: either it is not in your hand or it does"
            " not disprove this suggestion. Choose from the offered list and"
            " nothing else.",
            "",
            "Respond with the exact name of one of the cards you are offered,"
            f" in `card`.{notes_instruction}",
            "",
            f"You are player {observation.player_id}.",
        ]
    )

    offered = [card.value.value for card in matches]
    volatile_lines = [
        f"Turn: {observation.turn_number}.",
        f"Player {suggesting_player_id} suggested: "
        f"{suggestion.person.value} / {suggestion.weapon.value} / {suggestion.room.value}",
        f"Cards in your hand that would disprove it: {offered}",
        _reveal_history_line(observation, suggesting_player_id, offered),
        "",
        render_public_deductions(observation.public_deductions, observation.player_id),
    ]
    if scratchpad:
        volatile_lines += ["", f"Your private notes from previous turns: {scratchpad}"]

    return [
        *_stable_segments(observation, instructions),
        PromptSegment("\n".join(volatile_lines), cacheable=False),
    ]


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _extract_json_object(raw: str) -> str:
    """Best-effort recovery of a single JSON object from LLM output that
    ignores the "respond with only JSON" instruction — e.g. wraps the object
    in a ```json ... ``` fence, or follows it with unrequested reasoning
    prose. Returns `raw` unchanged if nothing extractable is found, so the
    original text still ends up in the error message on failure.
    """
    text = raw.strip()

    fence_match = _CODE_FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def parse_action(raw: str) -> PlayerAction:
    try:
        return _player_action_adapter.validate_json(_extract_json_object(raw))
    except ValidationError as exc:
        raise ValueError(f"Could not parse a valid action from LLM output: {raw!r}") from exc


def parse_model(raw: str, schema: type[ModelT]) -> ModelT:
    """Text-path counterpart to a provider's structured output."""
    try:
        return schema.model_validate_json(_extract_json_object(raw))
    except ValidationError as exc:
        raise ValueError(
            f"Could not parse a valid {schema.__name__} from LLM output: {raw!r}"
        ) from exc


def resolve_reveal(named: str, matches: list[Card]) -> Card:
    """Maps a model's named card onto one of the cards it was actually offered.

    `reveal_schema` makes an off-menu answer impossible on the structured
    path, so this only runs when a provider ignores the constraint and the
    answer has to be salvaged from free text. It still doesn't raise: the
    agent is mid-way through disproving someone, every option is legal, and
    ending the whole run over a misspelled card name would be a worse outcome
    than showing the "wrong" one.

    The last resort is a *random* legal card rather than the first one.
    `matches[0]` is the lowest-numbered card in a fixed deck order, so a
    deterministic fallback both leaks the same card repeatedly and hands
    opponents a card that correlates with deck position rather than with the
    seat's play.
    """
    wanted = named.strip().casefold()
    for card in matches:
        if card.value.value.casefold() == wanted:
            return card
    for card in matches:
        if card.value.value.casefold() in wanted or wanted in card.value.value.casefold():
            return card
    return choice(matches)


def _off_menu_reveal(matches: list[Card], named: str | None) -> Card:
    """Recovers a legal card when the model's answer wasn't one, and records
    on the enclosing `clue.agent.choose_reveal` span that it happened.

    Silence here is what made the original defect invisible: the substituted
    card went into the game as if the seat had chosen it, and only a
    line-by-line reading of the prompts against the reveals showed otherwise.
    A span attribute makes the rate of it a thing you can query.
    """
    card = resolve_reveal(named, matches) if named else choice(matches)
    span = trace.get_current_span()
    span.set_attribute("clue.reveal_off_menu", True)
    span.set_attribute("clue.reveal_named", named or "<unparseable>")
    span.set_attribute("clue.reveal_substituted", str(card.value.value))
    return card


class LLMClueAgent(ClueAgent):
    """Delegates both of a seat's decisions — its turn, and which card to
    show when disproving — to an LLM.

    `use_scratchpad` turns on a private notepad: the model may return `notes`
    with any decision, and whatever it writes is handed back to it on its
    next decision. It is subjective, never authoritative game state, and the
    engine never reads it. Off by default, because it is a real variable in
    any experiment run against these agents — whether a model plays better
    with its own memory of previous turns is a thing you would want to
    measure, not something to switch on for every seat by default.
    """

    def __init__(
        self,
        player_id: int,
        client: LLMClient,
        use_scratchpad: bool = False,
        trace_attributes: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(player_id, trace_attributes)
        self.client = client
        self.use_scratchpad = use_scratchpad
        self.scratchpad = ""

    async def _decide(
        self,
        segments: list[PromptSegment],
        schema: type[ModelT],
        metadata: dict[str, Any],
    ) -> ModelT:
        complete_structured = getattr(self.client, "complete_structured", None)
        if complete_structured is not None:
            return await complete_structured(segments, schema, metadata=metadata)
        return parse_model(await self.client.complete(segments, metadata=metadata), schema)

    def _remember(self, notes: str | None) -> None:
        if self.use_scratchpad and notes:
            self.scratchpad = notes

    async def _choose_action(self, observation: PlayerObservation) -> PlayerAction:
        segments = build_prompt(observation, self.scratchpad, self.use_scratchpad)
        decision = await self._decide(
            segments,
            turn_schema(self.use_scratchpad),
            {"clue_decision": "action", "clue_turn": observation.turn_number},
        )
        self._remember(decision.scratchpad_notes())
        return decision.to_action()

    async def _choose_reveal(
        self,
        observation: PlayerObservation,
        suggestion: Suggestion,
        matches: list[Card],
        suggesting_player_id: int,
    ) -> Card:
        if len(matches) == 1:
            # No decision to make, and no reason to pay for a provider call.
            return matches[0]
        offered = tuple(card.value.value for card in matches)
        segments = build_reveal_prompt(
            observation,
            suggestion,
            matches,
            suggesting_player_id,
            self.scratchpad,
            self.use_scratchpad,
        )
        try:
            decision = await self._decide(
                segments,
                reveal_schema(offered, self.use_scratchpad),
                {"clue_decision": "reveal", "clue_turn": observation.turn_number},
            )
        except (ValidationError, ValueError):
            # `reveal_schema` makes an off-menu answer impossible whenever the
            # provider honours it, so reaching here means the text fallback
            # also failed to produce anything usable. A seat is mid-disproof
            # and every option is legal, so show one rather than ending the
            # run — but leave a mark on the span saying we had to.
            return _off_menu_reveal(matches, named=None)
        self._remember(getattr(decision, "notes", None))
        if decision.card in offered:
            return matches[offered.index(decision.card)]
        return _off_menu_reveal(matches, named=decision.card)
