import pytest

from agents.human import HumanClueAgent
from service.agent_factory import build_agent, model_kwargs_for, seat_identity
from service.schemas import SeatConfig


def seat(**overrides) -> SeatConfig:
    base = {"agent_type": "llm", "provider": "anthropic", "model": "a-model"}
    return SeatConfig(**{**base, **overrides})


def test_unset_knobs_are_left_to_the_provider_default():
    assert model_kwargs_for("anthropic", seat()) == {}
    assert model_kwargs_for("gemini", seat(provider="gemini")) == {}
    assert model_kwargs_for("openai", seat(provider="openai")) == {}


LEGACY = "claude-sonnet-4-5"  # pre-4.6: still takes a token budget


def test_anthropic_legacy_model_enables_extended_thinking_and_makes_room_for_the_answer():
    kwargs = model_kwargs_for("anthropic", seat(model=LEGACY, thinking_budget=4096))

    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    # The response cap has to cover the thinking budget plus the answer.
    assert kwargs["max_tokens"] > 4096


def test_anthropic_legacy_model_keeps_an_explicit_max_tokens_when_it_is_already_big_enough():
    kwargs = model_kwargs_for(
        "anthropic", seat(model=LEGACY, thinking_budget=1024, max_tokens=16000)
    )

    assert kwargs["max_tokens"] == 16000


def test_anthropic_legacy_model_drops_temperature_while_thinking_is_on():
    # The API rejects any temperature other than 1 alongside extended
    # thinking, so a seat asking for both loses the temperature.
    with_thinking = model_kwargs_for(
        "anthropic", seat(model=LEGACY, temperature=0.2, thinking_budget=2048)
    )
    without = model_kwargs_for("anthropic", seat(model=LEGACY, temperature=0.2))

    assert "temperature" not in with_thinking
    assert without["temperature"] == 0.2


def test_anthropic_legacy_model_budget_of_zero_disables_thinking():
    assert model_kwargs_for("anthropic", seat(model=LEGACY, thinking_budget=0))["thinking"] == {
        "type": "disabled"
    }


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6"],
)
def test_anthropic_current_models_use_adaptive_thinking_and_effort_not_a_budget(model):
    # Sending `thinking.type=enabled` + `budget_tokens` to these is a 400:
    # '"thinking.type.enabled" is not supported for this model. Use
    # "thinking.type.adaptive" and "output_config.effort"'.
    kwargs = model_kwargs_for("anthropic", seat(model=model, thinking_budget=4096))

    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["effort"] == "medium"
    assert "budget_tokens" not in str(kwargs)
    # No headroom padding either: max_tokens is the seat's own or unset.
    assert "max_tokens" not in kwargs


@pytest.mark.parametrize(
    "budget,effort",
    [(1024, "low"), (4096, "medium"), (16384, "high"), (65536, "xhigh")],
)
def test_anthropic_budget_maps_onto_effort_levels(budget, effort):
    kwargs = model_kwargs_for("anthropic", seat(model="claude-sonnet-5", thinking_budget=budget))
    assert kwargs["effort"] == effort


def test_anthropic_current_model_budget_of_zero_disables_thinking():
    kwargs = model_kwargs_for("anthropic", seat(model="claude-sonnet-5", thinking_budget=0))
    assert kwargs["thinking"] == {"type": "disabled"}
    assert "effort" not in kwargs


@pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-4-7", "claude-fable-5-1"])
def test_anthropic_newest_models_drop_temperature_unconditionally(model):
    # These reject sampling parameters outright, thinking or not.
    kwargs = model_kwargs_for("anthropic", seat(model=model, temperature=0.2))
    assert "temperature" not in kwargs


def test_anthropic_4_6_keeps_temperature_when_thinking_is_off():
    kwargs = model_kwargs_for("anthropic", seat(model="claude-sonnet-4-6", temperature=0.2))
    assert kwargs["temperature"] == 0.2


def test_anthropic_fable_never_sends_a_thinking_block():
    # Thinking is always on and any explicit `thinking` config is a 400, so
    # a budget becomes effort and zero becomes the lowest effort.
    with_budget = model_kwargs_for("anthropic", seat(model="claude-fable-5-1", thinking_budget=4096))
    zero = model_kwargs_for("anthropic", seat(model="claude-fable-5-1", thinking_budget=0))

    assert "thinking" not in with_budget and with_budget["effort"] == "medium"
    assert "thinking" not in zero and zero["effort"] == "low"


def test_anthropic_version_first_names_are_recognised_as_legacy():
    kwargs = model_kwargs_for(
        "anthropic", seat(model="claude-3-5-sonnet-20241022", thinking_budget=2048)
    )
    assert kwargs["thinking"]["type"] == "enabled"


def test_anthropic_unknown_model_name_is_treated_as_current():
    kwargs = model_kwargs_for("anthropic", seat(model="a-model", thinking_budget=2048))
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_gemini_and_ollama_use_their_own_parameter_names():
    gemini = model_kwargs_for("gemini", seat(provider="gemini", thinking_budget=512, max_tokens=900))
    ollama = model_kwargs_for("ollama", seat(provider="ollama", max_tokens=900))

    assert gemini == {"thinking_budget": 512, "max_output_tokens": 900}
    assert ollama == {"num_predict": 900}


@pytest.mark.parametrize(
    "budget,effort",
    [(0, "minimal"), (1024, "low"), (4096, "medium"), (16384, "high")],
)
def test_openai_maps_a_token_budget_onto_a_reasoning_effort(budget, effort):
    kwargs = model_kwargs_for("openai", seat(provider="openai", thinking_budget=budget))

    assert kwargs["reasoning_effort"] == effort


def test_unknown_providers_get_the_plain_max_tokens():
    assert model_kwargs_for("cerebras", seat(provider="cerebras", max_tokens=256)) == {
        "max_tokens": 256
    }


def test_seat_rejects_nonsensical_generation_limits():
    with pytest.raises(ValueError):
        seat(thinking_budget=-1)
    with pytest.raises(ValueError):
        seat(max_tokens=0)


def test_scratchpad_defaults_to_off():
    assert seat().scratchpad is False


def test_seat_identity_records_who_is_playing_the_seat():
    # A trace whose only metadata is the project name can't answer "which
    # model played seat 2?" — that had to be reconstructed by reading prompt
    # text out of an export, which does not scale past a single game.
    identity = seat_identity(2, seat(model="claude-sonnet-5", thinking_budget=4096), "run-abc")

    assert identity == {
        "clue_player_id": 2,
        "clue_agent_type": "llm",
        "clue_run_id": "run-abc",
        "clue_provider": "anthropic",
        "clue_model": "claude-sonnet-5",
        "clue_scratchpad": False,
        "clue_thinking_budget": 4096,
    }


def test_seat_identity_of_a_random_seat_carries_no_model_fields():
    identity = seat_identity(0, SeatConfig(agent_type="random"), "run-abc")

    assert identity == {
        "clue_player_id": 0,
        "clue_agent_type": "random",
        "clue_run_id": "run-abc",
    }


def test_a_built_agent_carries_its_identity_onto_its_spans():
    agent = build_agent(1, SeatConfig(agent_type="random"), run_id="run-abc")

    # Dotted keys, because that's OTel's convention — LangSmith gets the
    # underscored form instead.
    assert agent.trace_attributes == {
        "clue.player_id": 1,
        "clue.agent_type": "random",
        "clue.run_id": "run-abc",
    }


def test_build_agent_human_seat_returns_a_human_clue_agent():
    agent = build_agent(0, SeatConfig(agent_type="human", name="Alice"), run_id="run-abc")

    assert isinstance(agent, HumanClueAgent)
    assert agent.name == "Alice"
    assert agent.trace_attributes["clue.agent_type"] == "human"


def test_build_agent_human_seat_wires_the_on_decision_request_callback():
    received = []

    async def on_decision_request(decision):
        received.append(decision)

    agent = build_agent(
        0,
        SeatConfig(agent_type="human"),
        run_id="run-abc",
        on_decision_request=on_decision_request,
    )

    assert isinstance(agent, HumanClueAgent)
    assert agent.on_request is on_decision_request
