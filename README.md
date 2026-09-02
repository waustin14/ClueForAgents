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

### Tracing (OpenTelemetry)

`telemetry.py` exposes a single module-level `tracer`, imported and used
directly by `game/engine.py` and `agents/base.py` — nothing conditional,
no feature flag to thread through call sites. That's safe because
`opentelemetry-api` (a core dependency) ships a no-op tracer by default:
every `tracer.start_as_current_span(...)` call is inert until
`configure_tracing()` actually installs an exporting `TracerProvider`.

Spans, one turn of a 3-player game:

```
clue.game            (main.py, root span for the whole run)
└── clue.turn        (GameEngine.take_turn — player_id, turn_number, action_type)
    └── clue.suggestion   (or clue.accusation — disproving_player_id, correct, etc.)
clue.agent.choose_action   (agents/base.py — every ClueAgent subclass, for free;
                             wraps LLMClueAgent's LLM round-trip when that's the agent)
```

Centralizing the agent span in `ClueAgent.choose_action` (subclasses
implement `_choose_action`) means `RandomClueAgent`, `LLMClueAgent`, and any
future rule-based/human agent all get consistent per-decision latency and
outcome attributes without instrumenting themselves individually — exactly
the kind of cross-cutting concern that shouldn't live in each strategy.

`configure_tracing()` wires up an OTLP exporter, reading the standard
`OTEL_EXPORTER_OTLP_*` environment variables (defaults to
`localhost:4317`) rather than inventing a parallel config surface. It's
called from `main.py` guarded by a bare `try/except ImportError`, so the
game runs identically whether or not tracing is configured — with no
collector reachable, spans just fail to export in the background (logged,
not raised), though process exit is delayed a few seconds by the OTLP gRPC
exporter's own shutdown-flush retries, which is expected without a
collector, not a hang.

Requires the optional dependency group: `uv sync --extra otel`
(`opentelemetry-sdk` + `opentelemetry-exporter-otlp-proto-grpc`). Without
it, `configure_tracing()` raises `ImportError` and the game behaves exactly
as if tracing were never mentioned; `tests/test_telemetry.py` skips itself
via `pytest.importorskip` the same way the LangChain tests do.

### Tracing (LangSmith)

Every provider in `agents/langchain_client.py` — Anthropic, OpenAI, Gemini,
Ollama, Cerebras — is a LangChain `BaseChatModel`, so LLM tracing doesn't
need a per-provider or per-call-site hook the way the OTel spans above do:
LangChain's global callback manager already wraps every
`model.ainvoke(...)` call inside `LangChainLLMClient.complete`, and
LangSmith just needs to be told to listen. `configure_langsmith_tracing()`
in `telemetry.py` does that — it sets `LANGSMITH_TRACING=true` and a
default `LANGSMITH_PROJECT`, both only if unset, so an operator's own
environment always wins. `LANGSMITH_API_KEY` and `LANGSMITH_ENDPOINT` are
read straight from the environment, same as `configure_tracing()` does for
`OTEL_EXPORTER_OTLP_*` — no parallel configuration surface.

It's called from `main.py` guarded by a bare `try/except ImportError`,
same soft-opt-in shape as OTel: without the extra, or without
`LANGSMITH_API_KEY` set, LangChain just has nowhere to send runs and every
LLM call proceeds untraced rather than the game breaking. Once enabled,
each traced run captures the full prompt (including which `PromptSegment`s
were marked `cacheable`), the completion, token usage, latency, and — for
Anthropic — cache read/write token counts, viewable per-turn in the
LangSmith UI grouped under the project name.

Requires the optional dependency group: `uv sync --extra langsmith`
(`langsmith`) on top of `--extra langchain`, since there's nothing to trace
without a LangChain-backed client. Without it, `configure_langsmith_tracing()`
raises `ImportError`; `tests/test_langsmith.py` skips itself via
`pytest.importorskip` the same way.

## Running

```bash
uv sync                          # core game only
uv sync --extra langchain        # + LangChain provider clients
uv sync --extra otel             # + OpenTelemetry OTLP tracing
uv sync --extra langsmith        # + LangSmith LLM call tracing (needs --extra langchain too)
uv run python main.py            # plays one local game with random agents
uv run pytest                    # runs the test suite (no LLM or network required)
```
