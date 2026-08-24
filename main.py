import asyncio

from agents.random_agent import RandomClueAgent
from game.engine import GameEngine
from game.setup import initialize_game
from models.game_state import GameStatus
from transport.local import LocalTransport


async def run_local_game(n_players: int = 3, max_turns: int = 500) -> None:
    state = initialize_game(n_players)
    transport = LocalTransport()
    engine = GameEngine(state, transport)
    agents = {player.id: RandomClueAgent(player.id) for player in state.players}

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


def main() -> None:
    asyncio.run(run_local_game())


if __name__ == "__main__":
    main()
