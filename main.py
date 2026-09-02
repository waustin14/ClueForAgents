import asyncio

from agents.random_agent import RandomClueAgent
from game.engine import GameEngine
from game.setup import initialize_game
from models.game_state import GameStatus
from telemetry import configure_langsmith_tracing, configure_tracing, tracer
from transport.local import LocalTransport


async def run_local_game(n_players: int = 3, max_turns: int = 500) -> None:
    state = initialize_game(n_players)
    transport = LocalTransport()
    engine = GameEngine(state, transport)
    agents = {player.id: RandomClueAgent(player.id) for player in state.players}

    with tracer.start_as_current_span("clue.game", attributes={"clue.n_players": n_players}):
        for _ in range(max_turns):
            if state.status == GameStatus.FINISHED:
                break
            agent = agents[state.current_player_id]
            observation = engine.observation_for(agent.player_id)
            action = await agent.choose_action(observation)
            await engine.take_turn(agent.player_id, action)

    if state.status == GameStatus.FINISHED:
        print(f"Game over after {state.turn_number} turns. Winner: player {state.winning_player_id}")
    else:
        print(f"Turn limit ({max_turns}) reached without a winner.")


def _configure_tracing_if_available() -> None:
    """Wires up OTLP export when the optional `otel` extra is installed.

    Without it, `configure_tracing()` raises ImportError and every span
    created via `telemetry.tracer` throughout the codebase just stays a
    no-op — tracing degrades silently rather than breaking the game.
    """
    try:
        configure_tracing()
    except ImportError:
        pass


def _configure_langsmith_tracing_if_available() -> None:
    """Wires up LangSmith tracing when the optional `langsmith` extra is installed.

    Same soft-opt-in shape as `_configure_tracing_if_available()` above:
    without the extra, `configure_langsmith_tracing()` raises ImportError
    and every LangChain-backed LLM call just runs untraced.
    """
    try:
        configure_langsmith_tracing()
    except ImportError:
        pass


def main() -> None:
    _configure_tracing_if_available()
    _configure_langsmith_tracing_if_available()
    asyncio.run(run_local_game())


if __name__ == "__main__":
    main()
