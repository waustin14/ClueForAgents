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

### LangChain-backed LLM clients

`agents/langchain_client.py` implements `LLMClient` on top of any LangChain
`BaseChatModel`, for five providers, all reachable through one factory:

```python
from agents.langchain_client import create_llm_client

client = create_llm_client("anthropic", "claude-sonnet-5")
# or: create_llm_client("openai", "gpt-4o-mini")
#     create_llm_client("gemini", "gemini-2.5-flash")
#     create_llm_client("ollama", "llama3")
#     create_llm_client("cerebras", "llama-3.3-70b")
agent = LLMClueAgent(player_id=0, client=client)
```

`create_llm_client(provider, model, **kwargs)` is the one entry point to
reach for when the provider itself is a variable in an experiment (sweeping
providers/models from a config or CLI flag); each provider's own factory
(`anthropic_client`, `openai_client`, `gemini_client`, `ollama_client`,
`cerebras_client`) is still exported directly for when it isn't.

`PromptSegment.cacheable` (stable-first, volatile-last — see above) is what
makes caching work everywhere, but only one provider needs code to use it:

- **Anthropic** — explicit opt-in. Cacheable segments become separate
  content blocks in the system message, `cache_control` on the last one.
- **OpenAI** — the platform auto-caches matching prefixes ≥1024 tokens;
  segments are just concatenated, nothing to annotate.
- **Gemini** (`ChatGoogleGenerativeAI`) — explicit caching is a separate
  stored `CachedContent` resource referenced by ID, not an inline
  per-message flag, so this adapter doesn't attempt to bridge it; segments
  are concatenated like OpenAI.
- **Cerebras** — automatic and implicit: the platform hashes each request
  in 128-token blocks and reuses any block matching a recent request, no
  opt-in needed. Segments are concatenated; `PromptSegment` ordering is
  what makes the prefix actually match.
- **Ollama** — also automatic and implicit, via the underlying llama.cpp
  engine's KV-cache reuse for a byte-identical prefix — same principle,
  but only while the model stays loaded. Ollama unloads after 5 minutes
  idle by default, which would silently evict the cache mid-game, so
  `ollama_client` defaults `keep_alive="30m"` (override via kwargs).

Requires the optional dependency group: `uv sync --extra langchain`. Without
it, `agents/langchain_client.py` isn't imported by anything else, and its
test file skips itself via `pytest.importorskip` — the core game and its
tests never need LangChain or network access.

## Running

```bash
uv sync                          # core game only
uv sync --extra langchain        # + LangChain provider clients
uv run python main.py            # plays one local game with random agents
uv run pytest                    # runs the test suite (no LLM or network required)
```
