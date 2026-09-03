import asyncio

import pytest
from pydantic import ValidationError

from agents.llm import (
    LLMClueAgent,
    PromptSegment,
    TurnDecision,
    TurnDecisionWithNotes,
    build_prompt,
    build_reveal_prompt,
    parse_action,
    render_public_deductions,
    resolve_reveal,
    reveal_schema,
    turn_schema,
)
from game.engine import GameEngine
from models.actions import MakeAccusation, MakeSuggestion, PassTurn
from models.card import PersonCard, PersonValue, RoomCard, RoomValue, WeaponCard, WeaponValue
from models.deductions import PlayerDeduction, PublicDeductions
from models.events import Suggestion
from models.game_state import GameState, Solution
from models.player import LogEntry, Player
from transport.local import LocalTransport


def run(coro):
    return asyncio.run(coro)


def build_state() -> GameState:
    solution = Solution(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY)
    players = [
        Player(id=0, cards=[PersonCard(value=PersonValue.GREEN)]),
        Player(id=1, cards=[WeaponCard(value=WeaponValue.KNIFE)]),
        Player(id=2, cards=[RoomCard(value=RoomValue.STUDY)]),
    ]
    return GameState(players=players, solution=solution)


def cacheable_prefix(segments: list[PromptSegment]) -> str:
    return "".join(segment.text for segment in segments if segment.cacheable)


def test_build_prompt_marks_instructions_hand_and_history_as_cacheable():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    observation = engine.observation_for(0)

    segments = build_prompt(observation)

    assert [s.cacheable for s in segments] == [True, True, True, True, False]
    # The volatile segment is the only one carrying per-turn state.
    volatile_text = segments[-1].text
    assert "Turn:" in volatile_text
    assert "Current player:" in volatile_text
    assert "Current player:" not in cacheable_prefix(segments)


def test_each_cacheable_segment_is_independently_append_only():
    # Each cacheable segment is its own cache block in a real request, so
    # the invariant that matters is per-segment prefix stability, not
    # stability of the segments concatenated together (concatenating two
    # independently-growing segments would shift where the second one's
    # text falls, even though each is individually append-only).
    state = build_state()
    transport = LocalTransport()
    engine = GameEngine(state, transport)

    before = build_prompt(engine.observation_for(0))

    run(engine.take_turn(0, PassTurn()))
    run(
        engine.take_turn(
            1,
            MakeSuggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY),
        )
    )

    after = build_prompt(engine.observation_for(0))

    for before_segment, after_segment in zip(before, after):
        if before_segment.cacheable:
            assert after_segment.text.startswith(before_segment.text)

    public_history_index = 2
    assert len(after[public_history_index].text) > len(before[public_history_index].text)


def test_scratchpad_only_affects_the_volatile_segment():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    observation = engine.observation_for(0)

    plain = build_prompt(observation)
    with_notes = build_prompt(observation, scratchpad="I suspect the Study.")

    assert cacheable_prefix(plain) == cacheable_prefix(with_notes)
    assert "I suspect the Study." in with_notes[-1].text
    assert "I suspect the Study." not in plain[-1].text


def test_parse_action_recovers_typed_action_from_json():
    action = parse_action(
        '{"kind": "suggestion", "person": "Professor Plum", "weapon": "Rope", "room": "Study"}'
    )
    assert isinstance(action, MakeSuggestion)
    assert action.room == RoomValue.STUDY


def test_parse_action_distinguishes_suggestion_from_accusation():
    suggestion = parse_action(
        '{"kind": "suggestion", "person": "Professor Plum", "weapon": "Rope", "room": "Study"}'
    )
    accusation = parse_action(
        '{"kind": "accusation", "person": "Professor Plum", "weapon": "Rope", "room": "Study"}'
    )
    assert isinstance(suggestion, MakeSuggestion)
    assert isinstance(accusation, MakeAccusation)


def test_parse_action_strips_markdown_code_fence():
    action = parse_action(
        '```json\n'
        '{"kind": "suggestion", "person": "Mr. Green", "weapon": "Candlestick", "room": "Hall"}\n'
        '```'
    )
    assert isinstance(action, MakeSuggestion)


def test_parse_action_ignores_trailing_reasoning_after_fence():
    # Real observed output: a fenced JSON object followed by unrequested
    # commentary the model appended despite being told to respond with only
    # the JSON object.
    action = parse_action(
        '```json\n'
        '{\n'
        '  "kind": "suggestion",\n'
        '  "person": "Mr. Green",\n'
        '  "weapon": "Candlestick",\n'
        '  "room": "Hall"\n'
        '}\n'
        '```\n'
        '\n'
        '**Reasoning:** On the first turn, I want to gather information about '
        "cards I haven't seen."
    )
    assert isinstance(action, MakeSuggestion)
    assert action.person == PersonValue.GREEN


def test_parse_action_ignores_trailing_prose_without_fence():
    action = parse_action(
        '{"kind": "pass"}\n\nI have nothing useful to suggest this turn.'
    )
    assert isinstance(action, PassTurn)


def test_rules_primer_warns_against_reading_an_undisproved_suggestion_as_proof():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    instructions = build_prompt(engine.observation_for(0))[0].text

    # The two mistakes that decided the traced games: treating a suggestion
    # nobody could disprove as proof of the solution, and suggesting a proven
    # solution instead of accusing it.
    assert "never asked to disprove your own suggestion" in instructions
    assert "does NOT" in instructions and "prove the solution" in instructions
    assert "ACCUSE" in instructions


def test_prompt_states_the_public_deductions_and_the_open_cards():
    state = build_state()
    # `initialize_game` seeds each player's log with their own hand; this
    # hand-built state has to do the same for the "(you)" rendering to show.
    state.players[0].log.people[PersonValue.GREEN] = LogEntry(seen=True, who_has=0)
    engine = GameEngine(state, LocalTransport())
    run(
        engine.take_turn(
            0,
            MakeSuggestion(person=PersonValue.PLUM, weapon=WeaponValue.ROPE, room=RoomValue.STUDY),
        )
    )

    volatile = build_prompt(engine.observation_for(0))[-1].text
    accounted, rest = volatile.split("Cards still unaccounted for")
    still_open = rest.split("Proven from the public log")[0]

    assert "Proven from the public log" in volatile
    # Player 1 was asked and held none of the three.
    assert "player 1: holds none of: Professor Plum, Rope, Study" in volatile
    # Own cards and cards shown to us are accounted for, so neither is still
    # a candidate for the envelope.
    assert "Mr. Green (you)" in accounted
    assert "Study (player 2)" in accounted
    assert "Study" not in still_open
    assert "Mr. Green" not in still_open


@pytest.mark.parametrize(
    "kind,expected",
    [("suggestion", MakeSuggestion), ("accusation", MakeAccusation)],
)
def test_agent_decision_converts_to_the_typed_action(kind, expected):
    decision = TurnDecisionWithNotes(
        kind=kind,
        person=PersonValue.PLUM,
        weapon=WeaponValue.ROPE,
        room=RoomValue.STUDY,
        notes="keep watching the Study",
    )

    action = decision.to_action()

    assert isinstance(action, expected)
    assert action.room == RoomValue.STUDY
    assert action.notes == "keep watching the Study"


def test_agent_decision_pass_needs_no_cards():
    assert isinstance(TurnDecisionWithNotes(kind="pass").to_action(), PassTurn)


def test_agent_decision_rejects_an_incomplete_suggestion():
    decision = TurnDecisionWithNotes(kind="suggestion", person=PersonValue.PLUM)

    with pytest.raises(ValueError):
        decision.to_action()


def test_agent_decision_schema_has_no_top_level_oneof():
    # A discriminated union serializes to a top-level `oneOf`, which several
    # providers reject outright in structured-output mode. The flat `kind`
    # selector exists precisely to avoid that.
    schema = TurnDecision.model_json_schema()

    assert "oneOf" not in schema
    assert schema["properties"]["kind"]["enum"] == ["suggestion", "accusation", "pass"]


def test_turn_schema_only_offers_notes_when_the_scratchpad_is_on():
    # A `notes` field is an invitation to write a paragraph of deduction. A
    # seat that has no scratchpad throws that paragraph away, so it must not
    # be asked for one in the first place.
    assert "notes" not in turn_schema(False).model_json_schema()["properties"]
    assert "notes" in turn_schema(True).model_json_schema()["properties"]


def test_reveal_schema_admits_only_the_cards_actually_offered():
    schema = reveal_schema(("Knife", "Study"), False)

    assert schema.model_json_schema()["properties"]["card"]["enum"] == ["Knife", "Study"]
    assert schema(card="Study").card == "Study"
    with pytest.raises(ValidationError):
        schema(card="Mrs. Peacock")


def test_reveal_schema_only_offers_notes_when_the_scratchpad_is_on():
    assert "notes" not in reveal_schema(("Knife",), False).model_json_schema()["properties"]
    assert "notes" in reveal_schema(("Knife",), True).model_json_schema()["properties"]


def test_resolve_reveal_matches_exactly_then_loosely_then_falls_back():
    knife = WeaponCard(value=WeaponValue.KNIFE)
    study = RoomCard(value=RoomValue.STUDY)
    matches = [knife, study]

    assert resolve_reveal("Study", matches) is study
    assert resolve_reveal("  knife  ", matches) is knife
    assert resolve_reveal("the Study card", matches) is study
    # Nothing recognisable: show *a* legal card rather than fail the run, and
    # pick it at random. Returning matches[0] would deterministically leak the
    # lowest card in deck order every time this path is hit.
    assert resolve_reveal("Mrs. Peacock", matches) in matches
    picks = {id(resolve_reveal("Mrs. Peacock", matches)) for _ in range(50)}
    assert picks == {id(knife), id(study)}


def test_reveal_prompt_names_the_suggester_not_whoever_is_on_turn():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    run(
        engine.take_turn(
            0,
            MakeSuggestion(person=PersonValue.PLUM, weapon=WeaponValue.KNIFE, room=RoomValue.STUDY),
        )
    )
    # It is now player 1's turn, but player 1 is the one being asked to
    # disprove player 0's suggestion.
    observation = engine.observation_for(1)
    assert observation.current_player_id == 1

    segments = build_reveal_prompt(
        observation,
        Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.KNIFE, room=RoomValue.STUDY),
        [WeaponCard(value=WeaponValue.KNIFE)],
        suggesting_player_id=0,
    )
    volatile = segments[-1].text

    assert "Player 0 suggested:" in volatile
    assert "Player 1 suggested:" not in volatile
    # And it recalls what this opponent has already been shown, so the agent
    # can keep showing the same card instead of leaking a second one.
    assert "Cards you have already shown player 0: ['Knife']" in volatile


def test_reveal_prompt_shares_the_stable_prefix_shape_with_the_turn_prompt():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    observation = engine.observation_for(1)

    segments = build_reveal_prompt(
        observation,
        Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.KNIFE, room=RoomValue.STUDY),
        [WeaponCard(value=WeaponValue.KNIFE)],
        suggesting_player_id=0,
    )

    assert [s.cacheable for s in segments] == [True, True, True, True, False]


class StubClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.received_segments: list[PromptSegment] | None = None
        self.received_metadata: dict | None = None
        self.calls = 0

    async def complete(
        self, segments: list[PromptSegment], metadata: dict | None = None
    ) -> str:
        self.received_segments = segments
        self.received_metadata = metadata
        self.calls += 1
        return self.response


def test_llm_clue_agent_delegates_to_client_and_parses_result():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    observation = engine.observation_for(0)
    client = StubClient(
        '{"kind": "suggestion", "person": "Mr. Green", "weapon": "Rope", "room": "Study"}'
    )
    agent = LLMClueAgent(player_id=0, client=client)

    action = run(agent.choose_action(observation))

    assert isinstance(action, MakeSuggestion)
    assert action.person == PersonValue.GREEN
    assert client.received_segments is not None
    assert len(client.received_segments) == 5


def test_scratchpad_is_off_by_default():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    client = StubClient('{"kind": "pass", "notes": "Plum is looking likely"}')
    agent = LLMClueAgent(player_id=0, client=client)

    run(agent.choose_action(engine.observation_for(0)))
    run(agent.choose_action(engine.observation_for(0)))

    assert agent.scratchpad == ""
    assert "Plum is looking likely" not in client.received_segments[-1].text


def test_scratchpad_carries_notes_into_the_next_turn_when_enabled():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    client = StubClient('{"kind": "pass", "notes": "Plum is looking likely"}')
    agent = LLMClueAgent(player_id=0, client=client, use_scratchpad=True)

    run(agent.choose_action(engine.observation_for(0)))
    assert "Plum is looking likely" not in client.received_segments[-1].text

    run(agent.choose_action(engine.observation_for(0)))

    assert agent.scratchpad == "Plum is looking likely"
    assert "Plum is looking likely" in client.received_segments[-1].text


def test_agent_chooses_which_card_to_reveal():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    client = StubClient('{"card": "Study"}')
    agent = LLMClueAgent(player_id=1, client=client)
    matches = [WeaponCard(value=WeaponValue.KNIFE), RoomCard(value=RoomValue.STUDY)]

    card = run(
        agent.choose_reveal(
            engine.observation_for(1),
            Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.KNIFE, room=RoomValue.STUDY),
            matches,
            suggesting_player_id=0,
        )
    )

    assert card is matches[1]
    assert client.calls == 1


def test_agent_does_not_call_the_model_when_only_one_card_can_disprove():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    client = StubClient('{"card": "Study"}')
    agent = LLMClueAgent(player_id=1, client=client)
    only = WeaponCard(value=WeaponValue.KNIFE)

    card = run(
        agent.choose_reveal(
            engine.observation_for(1),
            Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.KNIFE, room=RoomValue.STUDY),
            [only],
            suggesting_player_id=0,
        )
    )

    assert card is only
    assert client.calls == 0


def reveal_from(client: StubClient, matches: list) -> object:
    """Runs one reveal decision through an agent backed by `client`."""
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    agent = LLMClueAgent(player_id=1, client=client)
    return run(
        agent.choose_reveal(
            engine.observation_for(1),
            Suggestion(person=PersonValue.PLUM, weapon=WeaponValue.KNIFE, room=RoomValue.STUDY),
            matches,
            suggesting_player_id=0,
        )
    )


def test_a_card_that_was_not_offered_never_reaches_the_opponent():
    # Twice in a traced game a model named a card it had shown this opponent
    # before but that was not among the ones offered, and the resolver
    # silently substituted the first match — handing that opponent a *fresh*
    # card, the exact outcome the model was trying to avoid. Whatever comes
    # back now, only an offered card can be shown.
    matches = [WeaponCard(value=WeaponValue.KNIFE), RoomCard(value=RoomValue.STUDY)]

    for response in ('{"card": "Mrs. Peacock"}', '{"card": ""}', "not json at all"):
        assert reveal_from(StubClient(response), matches) in matches


def test_an_unusable_reveal_answer_does_not_end_the_run():
    # The seat is mid-disproof and every option is legal, so a garbled answer
    # is worth degrading over, not aborting the game over.
    matches = [WeaponCard(value=WeaponValue.KNIFE), RoomCard(value=RoomValue.STUDY)]
    picks = {id(reveal_from(StubClient("not json at all"), matches)) for _ in range(50)}

    # And the degraded pick is random, not always matches[0].
    assert picks == {id(matches[0]), id(matches[1])}


def test_the_prompt_offers_a_scratchpad_only_to_a_seat_that_has_one():
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    observation = engine.observation_for(0)
    suggestion = Suggestion(
        person=PersonValue.PLUM, weapon=WeaponValue.KNIFE, room=RoomValue.STUDY
    )
    matches = [WeaponCard(value=WeaponValue.KNIFE)]

    for builder, args in (
        (build_prompt, (observation,)),
        (build_reveal_prompt, (observation, suggestion, matches, 0)),
    ):
        assert "`notes`" not in "".join(s.text for s in builder(*args))
        assert "`notes`" in "".join(s.text for s in builder(*args, with_notes=True))


def test_reveal_prompt_says_when_no_previously_shown_card_can_be_shown_again():
    # The bare "already shown" list read as an instruction: told to keep
    # showing the same card, models named one that wasn't on offer. Stating
    # the intersection outright removes the conflict.
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    run(
        engine.take_turn(
            0,
            MakeSuggestion(person=PersonValue.PLUM, weapon=WeaponValue.KNIFE, room=RoomValue.STUDY),
        )
    )
    observation = engine.observation_for(1)
    suggestion = Suggestion(
        person=PersonValue.PLUM, weapon=WeaponValue.KNIFE, room=RoomValue.STUDY
    )

    unavailable = build_reveal_prompt(
        observation, suggestion, [RoomCard(value=RoomValue.LOUNGE)], 0
    )[-1].text
    available = build_reveal_prompt(
        observation, suggestion, [WeaponCard(value=WeaponValue.KNIFE)], 0
    )[-1].text

    assert "None of those is in the offered list" in unavailable
    assert "tells them nothing new" in available


def test_each_call_carries_which_decision_and_turn_it_is():
    # Without this, a LangSmith export is an undifferentiated pile of chat
    # completions: nothing on a run says which seat or turn produced it.
    state = build_state()
    engine = GameEngine(state, LocalTransport())
    client = StubClient('{"kind": "pass"}')
    agent = LLMClueAgent(player_id=0, client=client)

    run(agent.choose_action(engine.observation_for(0)))
    assert client.received_metadata == {"clue_decision": "action", "clue_turn": 0}

    reveal_from(client, [WeaponCard(value=WeaponValue.KNIFE), RoomCard(value=RoomValue.STUDY)])
    assert client.received_metadata["clue_decision"] == "reveal"


def test_a_group_narrowed_to_one_card_is_rendered_as_a_proven_holding():
    # "holds at least one card from each of: [Study]" makes the reader do a
    # collapse the panel has already done.
    deductions = PublicDeductions(
        per_player=[
            PlayerDeduction(
                player_id=1,
                cannot_have=[PersonValue.PLUM],
                holds_at_least_one_of=[[RoomValue.STUDY], [WeaponValue.ROPE, RoomValue.LOUNGE]],
            )
        ]
    )

    rendered = render_public_deductions(deductions, own_player_id=0)

    assert "player 1: holds: Study" in rendered
    assert "holds at least one card from each of: [Rope, Lounge]" in rendered
    assert "[Study]" not in rendered
