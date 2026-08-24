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

### Prompt structure and caching

`agents/llm.py::build_prompt` returns an ordered list of `PromptSegment`s
rather than one flat string, split stable-first / volatile-last: the game
instructions, the agent's hand, and the public/private event histories only
ever grow by appending, so they're marked `cacheable=True`; per-turn state
(`current_player_id`, `turn_number`, the card log, the scratchpad) changes
every call and is marked `cacheable=False`. Rebuilding the whole prompt from
scratch each turn is otherwise the expensive choice — without this
ordering, every turn re-pays full price for the entire history instead of
only the new bytes. A `LLMClient` backed by a provider with explicit prompt
caching (e.g. Anthropic's `cache_control`) can place its cache boundary
right after the last cacheable segment; a client for a provider with only
automatic caching can ignore the flag and concatenate everything.

## Running

```bash
uv sync
uv run python main.py   # plays one local game with random agents
uv run pytest           # runs the test suite (no LLM or network required)
```
