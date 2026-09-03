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
  `PlayerObservation`: `RandomClueAgent` (no LLM, used for testing),
  `LLMClueAgent` (delegates to a pluggable `LLMClient`), and
  `HumanClueAgent` (parks on a decision until a person answers it over
  HTTP — see "Human seats" below). `agents/` never imports a transport, so
  `HumanClueAgent` knows it has an async `on_request` callback to call when
  a decision is pending, not what that callback does with it (in
  `service/game_runner.py`, it publishes an SSE envelope).

`GameState` is ground truth and is never handed to an agent directly.
Agents only ever see a `PlayerObservation`, derived by the engine, which
contains their own hand, their card-knowledge log, the public/private
event history they're entitled to, and `public_deductions` — the facts
`game/deductions.py` reads straight off the public event log (who was
asked and held none of a triple; who holds at least one of a triple;
which triples a failed accusation has ruled out). That derivation
deliberately stops short of playing: it never cross-references one player
against another, never touches the observer's hand, and never narrows the
envelope, because working out the solution is the thing these agents are
being measured on.

Within a single player it does tidy up — `simplify_groups` drops a card
that same player is publicly proven not to hold, collapses identical
groups, and drops a group that a narrower one already implies. That is
bookkeeping on facts already stated rather than deduction, and it matters:
in one traced game 62 of 122 rendered rows carried a duplicated triple,
because two seats had suggested the same thing and the same player had
disproved it twice.

A seat makes two decisions, and both belong to the agent. `choose_action`
is its turn. `choose_reveal` is which card it shows when someone else's
suggestion reaches it — a real strategic choice whenever it holds more
than one of the three named cards, since showing the same card to the
same opponent repeatedly leaks far less than spreading reveals around.
`ClueAgent` gives that a random default, `LLMClueAgent` asks the model,
and the engine routes it through an injected `RevealChooser` so it stays
ignorant of agents entirely.

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

### Response schemas

Every decision is bound to a Pydantic schema through the provider's own
structured-output mode rather than asked for in prose. Two of those schemas
are built per call, not declared once:

- **The reveal schema is generated from the cards actually on offer.**
  `reveal_schema(offered, with_notes)` makes `card` a `Literal` over the two
  or three cards in this hand that disprove this suggestion, which is
  narrower than any fixed enum could express. Left as a free string, models
  named a card they had shown that opponent before but that wasn't on offer,
  and the resolver quietly substituted one of its own choosing — leaking a
  *fresh* card to the one opponent the seat was trying not to inform. If a
  provider ignores the constraint anyway, `_off_menu_reveal` still shows a
  legal card (a random one, not `matches[0]`) rather than ending the run, and
  records `clue.reveal_off_menu` on the span so the rate is queryable instead
  of invisible.
- **`notes` only exists when the seat's scratchpad is on.** A `notes` field
  is an invitation, and a model handed one writes a paragraph of deduction
  that a scratchpad-less seat then throws away. `turn_schema(with_notes)` and
  the matching `with_notes` flag on the prompt builders remove both the field
  and the sentence describing it, so the tokens are never generated.

`TurnDecision` is deliberately one flat object with a `kind` selector rather
than a discriminated union: a union serializes to a top-level `oneOf`, which
several providers reject outright in structured-output mode. `to_action()`
converts it back to the engine's typed action.

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
clue.agent.choose_reveal   (same, for the "which card do I show?" decision)
```

Centralizing the agent spans in `ClueAgent.choose_action` /
`ClueAgent.choose_reveal` (subclasses implement the underscore-prefixed
versions) means `RandomClueAgent`, `LLMClueAgent`, and any
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

#### What each run is tagged with

Global tracing alone gets you the prompts and the completions but not who
produced them: an export of a four-seat game is a flat list of chat
completions, and which model played which seat has to be reconstructed by
reading prompt text. So every call also carries the seat's identity.
`service/agent_factory.py::seat_identity` builds it once per seat —
`clue_run_id`, `clue_player_id`, `clue_agent_type`, `clue_provider`,
`clue_model`, `clue_scratchpad`, and the thinking budget and temperature
when set — and it goes two places:

- to LangSmith as run `metadata`, plus `tags` for the cheap filters
  (`model:…`, `player_id:…`, `decision:…`) and a `run_name` like
  `action p2 t14` so the run list isn't 200 rows all reading `ChatAnthropic`.
  Per-call `clue_decision` (`action` or `reveal`) and `clue_turn` are merged
  in on top by `LLMClueAgent`.
- to OpenTelemetry as span attributes on `clue.agent.choose_action` and
  `clue.agent.choose_reveal`, dotted (`clue.provider`, `clue.model`) to match
  OTel convention. `clue.run_id` is the same id on both sides, which is what
  stitches a seat's LangSmith runs back to its spans in Jaeger.

Random seats get the same treatment minus the model fields, so a mixed game
is still groupable end to end.

## Running

```bash
uv sync                          # core game only
uv sync --extra langchain        # + LangChain provider clients
uv sync --extra otel             # + OpenTelemetry OTLP tracing
uv sync --extra langsmith        # + LangSmith LLM call tracing (needs --extra langchain too)
uv run python main.py            # plays one local game with random agents
uv run pytest                    # runs the test suite (no LLM or network required)
```

## Operational testing (containerized)

A small containerized stack lets you drive real games from a browser, with
per-seat model selection and full trace visibility, without touching Python.
It's for manual/exploratory testing of game play and tracing — not a
production deployment.

**Containers** (`docker-compose.yml`):

| Service         | Image / build                              | Port  | Role |
|-----------------|---------------------------------------------|-------|------|
| `frontend`      | `docker/frontend.Dockerfile` (nginx)         | 8080  | Static test console; reverse-proxies `/api/` to `game-logic` so the browser never deals with CORS |
| `game-logic`    | `docker/game-logic.Dockerfile` (FastAPI)     | 8000  | Wraps the existing engine/agents behind an HTTP + SSE API (`service/app.py`) |
| `otel-collector`| `otel/opentelemetry-collector-contrib:0.110.0` | —   | Receives OTLP from `game-logic`, forwards to Jaeger |
| `jaeger`        | `jaegertracing/all-in-one:1.57`              | 16686 | Trace storage + UI |

**Run it:**

```bash
cp .env.example .env             # blank is fine for random-only agents;
                                  # fill in provider keys to test LLM seats
docker compose up --build
```

Then open:
- **http://localhost:8080** — test console: pick player count, an agent
  type (`random`, `human`, or a provider) and model per seat, and a turn
  cap, then start a game and watch events stream in live. A human seat's
  panel (and its shareable join link, for hot-seat testing across
  browsers) appears on the game screen; see the `POST /games` and
  `/players/{player_id}` entries below for the endpoints it drives.
- **http://localhost:16686** — Jaeger UI. Each game run is tagged with a
  `clue.run_id` span attribute; the console's game screen includes a
  "View trace" link that deep-links straight to that run's spans
  (`clue.game` → `clue.turn` → `clue.suggestion`/`clue.accusation`, plus
  `clue.agent.choose_action` for LLM seats).

**API surface** (`service/app.py`), if you want to drive it directly instead
of through the console:

- `GET /providers` — which LLM providers are available (i.e. have their API
  key env var set) vs. usable regardless (`random`, `human`).
- `GET /cards` — every legal card value (`{people, weapons, rooms}`), so a
  client never has to hardcode the enum the engine itself uses. Used to
  populate a human seat's suggestion/accusation form.
- `POST /games` — `{n_players, seats: [...], max_turns}` →
  `{game_id, run_id, stream_url, human_seats: [...]}`. A seat is
  `{agent_type, provider?, model?, temperature?, max_tokens?, thinking_budget?, scratchpad?, name?, decision_timeout?}`;
  `agent_type` is `"random"`, `"llm"`, or `"human"` (llm seats require
  `provider` + `model`). The generation knobs are provider-neutral —
  `agent_factory` translates them into each SDK's own parameter names, and
  anything left unset stays at the provider default. Two are worth setting
  deliberately:
  - `thinking_budget` caps reasoning tokens per decision (`0` disables
    extended thinking where the provider allows it). Unset, a reasoning
    model will happily spend thousands of tokens choosing between three
    move shapes. The number is only sent literally to providers that take
    one: OpenAI's reasoning models and Claude 4.6+ (Sonnet 5, Opus 5, the
    4.6–4.8 family) take an effort level instead, so `agent_factory` maps
    the budget onto one (`<2048` low, `<8192` medium, `<32768` high, else
    xhigh) and, for Claude, switches to `thinking.type=adaptive` — the
    older `enabled` + `budget_tokens` form is a 400 on those models. Fable
    and Mythos can't turn thinking off at all, so `0` becomes `low` effort
    there. Pre-4.6 Claude models still get the literal budget.
  - `scratchpad` (default `false`) lets that seat write itself private
    notes and read them back on its next decision. It's per-seat because
    whether a model plays better carrying its own notes forward is a
    variable worth measuring, not a default.

  A `"human"` seat ignores every LLM knob (setting `provider`, `model`,
  `thinking_budget`, or `scratchpad` on one is a 422) and instead takes
  `name` (shown on its seat card and join link) and an optional
  `decision_timeout` in seconds (a person's answer window before the seat
  auto-passes or auto-reveals; unset waits forever, which is what the test
  console defaults to). `POST /games` mints one bearer token per human
  seat and returns it in `human_seats` as
  `{player_id, name, token, join_url}` — `join_url` is a
  `/?game=<id>&player=<pid>&token=<t>` link that takes whoever opens it
  straight to that seat's panel, no separate login step. These tokens are
  a test-console convenience, not a real auth boundary — see
  `docs/plans/human-players.md` §3.
- `GET /games/{game_id}` — snapshot: status, turn number, current player,
  winner (if any), and `waiting_on: {player_id, kind}` whenever the run is
  genuinely parked waiting on a human seat's decision (`null` otherwise).
- `GET /games/{game_id}/players/{player_id}` (requires an
  `X-Player-Token` header matching that seat's minted token — 403 if
  missing/wrong, 404 if the seat isn't human) — that seat's own
  `PlayerObservation` (hand, card-knowledge log, event history it's
  entitled to, public deductions) plus whatever decision is currently
  pending for it, if any.
- `POST /games/{game_id}/players/{player_id}/action` (same token
  requirement) — submits a `PlayerAction` (suggestion / accusation /
  pass) for a pending `"action"` decision. 409 if nothing (or the wrong
  kind) is pending for that seat.
- `POST /games/{game_id}/players/{player_id}/reveal` (same token
  requirement) — submits `{card}` for a pending `"reveal"` decision. 400
  if `card` isn't one of the offered cards, leaving the decision pending
  so the person can try again; 409 if nothing is pending.
- `GET /games/{game_id}/events` — Server-Sent Events stream of the game as
  it plays out (`game_started`, `turn_taken`, `public_event`,
  `private_reveal`, `decision_requested`, `decision_resolved`, `error`,
  `game_finished` envelopes); replays backlog on (re)connect and closes
  after `game_finished`. Frames are unnamed — every one carries its `kind`
  in the payload and clients dispatch on that, so `EventSource.onmessage`
  receives the whole feed (see `service/app.py::_format_sse` for why
  naming them is a trap). Because the backlog is replayed in full on
  reconnect, a client should rebuild its view from the stream rather than
  append to what it already has.

  This feed is the *spectator* view, shared by everyone watching the game
  — including any human seated at the table. An all-AI game keeps card
  identities on `private_reveal` as a debugging aid, but the moment any
  seat is human that becomes a cheat sheet readable by unticking "show
  private reveals" in the console: the same envelope reaches every
  connection. So `SSEBroadcastTransport` strips the `card` field from
  `private_reveal` whenever the game has a human seat, keeping only
  `revealing_player_id`/`receiving_player_id`; a human seat still sees its
  own reveals in full through `GET /games/{game_id}/players/{player_id}`,
  which reads from that seat's own `PlayerObservation` rather than this
  feed.

**Notes:**
- `game-logic` only has the providers you've given keys for in `.env`
  actually usable; everything else still runs fine with `random` seats.
- `docker compose down` tears the stack down; add `-v` to also drop the
  (currently unused) volumes.
