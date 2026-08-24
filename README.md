# Clue for Agents

A Clue (Cluedo) game environment for AI agents. The engine records
observable facts — suggestions, disproves, card reveals, accusations — and
leaves the deduction to the agent playing each seat, so the project can be
used to benchmark reasoning strategies and models against each other.

## Architecture

- `models/` — data: cards, player state, game state, actions, events,
  and the per-player `PlayerObservation` the engine exposes to agents.
- `game/` — rules (`rules.py`), setup/dealing (`setup.py`), and the
  authoritative orchestration engine (`engine.py`). No LLM or transport
  dependency lives here.
- `transport/` — `AgentTransport` protocol plus a `LocalTransport` for
  running everything in-process. Other transports (A2A, websockets, a
  human UI) can implement the same protocol without touching the engine.
- `agents/` — controllers that decide what to do given a
  `PlayerObservation`: `RandomClueAgent` (no LLM, used for testing) and
  `LLMClueAgent` (delegates to a pluggable `LLMClient`).

`GameState` is ground truth and is never handed to an agent directly.
Agents only ever see a `PlayerObservation`, derived by the engine, which
contains their own hand, their card-knowledge log, and the public/private
event history they're entitled to.

## Running

```bash
uv sync
uv run python main.py   # plays one local game with random agents
uv run pytest           # runs the test suite (no LLM or network required)
```
