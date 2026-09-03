"""Starts and drives one game run — a thin, service-shaped transplant of
`main.py::run_local_game`: same loop, same reused `GameEngine` /
`initialize_game` / agent classes, with three things added around it that a
one-shot CLI run doesn't need: a `run_id` tag on the root span so a tester
can find this run in Jaeger, envelope publishing so the SSE feed has
something to show for every turn (including a `PassTurn`, which produces no
`PublicGameEvent` of its own), and exception containment — an LLM output
that fails to parse, or a provider call that errors, must end the run with
an `error` envelope rather than crashing the whole service.
"""

import asyncio
from uuid import uuid4

from agents.base import ClueAgent
from game.engine import GameEngine, RevealChooser
from game.setup import MAX_PLAYERS, MIN_PLAYERS, initialize_game
from models.card import Card
from models.events import Suggestion
from models.game_state import GameStatus
from models.player import Player
from service import registry
from service.agent_factory import build_agent
from service.registry import RunRecord
from service.schemas import CreateGameRequest
from telemetry import tracer
from transport.sse import SSEBroadcastTransport

MAX_TURNS_CEILING = 2000


def agent_reveal_chooser(engine: GameEngine, agents: dict[int, ClueAgent]) -> RevealChooser:
    """Routes "which card do you show?" to the seat that has to show it.

    The engine deliberately knows nothing about agents, so the choice is
    injected here: the disproving player's own agent gets its normal
    observation plus the suggestion it is answering, and decides.
    """

    async def choose_reveal(
        player: Player,
        suggestion: Suggestion,
        matches: list[Card],
        suggesting_player_id: int,
    ) -> Card:
        agent = agents[player.id]
        observation = engine.observation_for(player.id)
        return await agent.choose_reveal(
            observation, suggestion, matches, suggesting_player_id
        )

    return choose_reveal


def start_game(request: CreateGameRequest) -> RunRecord:
    """Validates the request, builds every seat's agent, and schedules the
    game loop as a background task. Raises ValueError on any bad input —
    including a seat naming a provider whose API key isn't set, or an `llm`
    seat when the `langchain` extra isn't installed — before anything is
    registered, so a failed request never leaves a half-built run behind.
    """
    if len(request.seats) != request.n_players:
        raise ValueError(
            f"seats has {len(request.seats)} entries but n_players={request.n_players}"
        )
    if not (MIN_PLAYERS <= request.n_players <= MAX_PLAYERS):
        raise ValueError(f"n_players must be between {MIN_PLAYERS} and {MAX_PLAYERS}")
    if not (1 <= request.max_turns <= MAX_TURNS_CEILING):
        raise ValueError(f"max_turns must be between 1 and {MAX_TURNS_CEILING}")

    state = initialize_game(request.n_players)
    transport = SSEBroadcastTransport()
    engine = GameEngine(state, transport)
    # Minted before the agents so each one can be built already knowing which
    # run it belongs to — that id is what stitches a seat's LangSmith runs and
    # its OTel spans back together into one game.
    game_id = uuid4().hex
    agents = {
        player.id: build_agent(player.id, seat, run_id=game_id)
        for player, seat in zip(state.players, request.seats, strict=True)
    }
    engine.choose_reveal = agent_reveal_chooser(engine, agents)

    record = RunRecord(
        game_id=game_id,
        run_id=game_id,
        state=state,
        engine=engine,
        transport=transport,
        agents=agents,
        seats=request.seats,
        max_turns=request.max_turns,
    )
    registry.add(record)
    record.task = asyncio.create_task(_play(record))
    return record


async def _play(record: RunRecord) -> None:
    with tracer.start_as_current_span(
        "clue.game",
        attributes={"clue.run_id": record.run_id, "clue.n_players": len(record.agents)},
    ):
        await record.transport.publish("game_started", {"n_players": len(record.agents)})
        try:
            for _ in range(record.max_turns):
                if record.state.status == GameStatus.FINISHED:
                    break
                agent = record.agents[record.state.current_player_id]
                observation = record.engine.observation_for(agent.player_id)
                action = await agent.choose_action(observation)
                await record.engine.take_turn(agent.player_id, action)
                await record.transport.publish(
                    "turn_taken",
                    {
                        "turn": record.state.turn_number,
                        "player_id": agent.player_id,
                        # `notes` is the agent's private scratchpad. The SSE
                        # feed is the spectator view, so it gets stripped
                        # here rather than broadcast to everyone watching.
                        "action": action.model_dump(mode="json", exclude={"notes"}),
                    },
                )
            record.status = (
                "finished" if record.state.status == GameStatus.FINISHED else "turn_limit_reached"
            )
        except Exception as exc:  # noqa: BLE001 - must end the run, not crash the service
            record.status = "error"
            await record.transport.publish("error", {"message": str(exc)})

        await record.transport.publish(
            "game_finished",
            {
                "status": record.status,
                "turn_number": record.state.turn_number,
                "winning_player_id": record.state.winning_player_id,
            },
        )
