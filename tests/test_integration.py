import asyncio

from agents.random_agent import RandomClueAgent
from game.engine import GameEngine
from game.setup import initialize_game
from models.game_state import GameStatus
from transport.local import LocalTransport


def test_full_random_game_runs_to_completion_without_an_llm():
    async def play() -> None:
        state = initialize_game(4)
        transport = LocalTransport()
        engine = GameEngine(state, transport)
        agents = {p.id: RandomClueAgent(p.id) for p in state.players}

        for _ in range(2000):
            if state.status == GameStatus.FINISHED:
                break
            agent = agents[state.current_player_id]
            observation = engine.observation_for(agent.player_id)
            action = await agent.choose_action(observation)
            await engine.take_turn(agent.player_id, action)

        assert state.status == GameStatus.FINISHED
        assert state.public_event_history[-1].winning_player_id == state.winning_player_id

    asyncio.run(play())
