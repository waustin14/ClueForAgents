import asyncio

from agents.llm import LLMClueAgent, PromptSegment, build_prompt, parse_action
from game.engine import GameEngine
from models.actions import MakeAccusation, MakeSuggestion, PassTurn
from models.card import PersonCard, PersonValue, RoomCard, RoomValue, WeaponCard, WeaponValue
from models.game_state import GameState, Solution
from models.player import Player
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


class StubClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.received_segments: list[PromptSegment] | None = None

    async def complete(self, segments: list[PromptSegment]) -> str:
        self.received_segments = segments
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
