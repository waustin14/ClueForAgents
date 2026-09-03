# Human players — implementation plan

Branch: `dev/human-players`. Goal: any seat in a 2–6 player game can be
driven by a person instead of a model, with the game flow, engine, and
turn order untouched. First milestone is one human at a table of AI/random
seats; the design must not special-case that, so N humans works the same
way.

## 1. Core idea: a human is just another `ClueAgent`

The engine never sees agents. `service/game_runner.py::_play` awaits
`agent.choose_action(observation)` and the reveal chooser awaits
`agent.choose_reveal(...)`. Nothing in that loop cares how long the await
takes or where the answer comes from.

So a human seat is a `HumanClueAgent` whose `_choose_action` and
`_choose_reveal` each:

1. build a `PendingDecision` (what is being asked, what the legal answers are),
2. publish a `decision_requested` envelope on the run's SSE feed,
3. `await` an `asyncio.Future`,
4. return whatever an HTTP handler put into that future.

The HTTP layer resolves the future. `GameEngine`, `GameState`, `rules.py`,
`setup.py`, the `AgentTransport` protocol, and the loop in `_play` do not
change. Tracing comes for free: `ClueAgent.choose_action` already wraps the
call in a `clue.agent.choose_action` span, so a human turn shows up in
Jaeger with its wall-clock latency like any other seat.

Why not a separate "human game mode" in the runner: it would fork the loop,
and every future engine change would have to be made twice. Because
`HumanClueAgent` sits behind the same interface as `RandomClueAgent` and
`LLMClueAgent`, seat composition is arbitrary by construction: 1 human + 3
LLMs, 6 humans, 2 humans + 1 random, all identical code paths.

## 2. Changes by layer

### `agents/human.py` (new)

```python
@dataclass
class PendingDecision:
    kind: Literal["action", "reveal"]
    player_id: int
    turn_number: int
    future: asyncio.Future            # resolves to PlayerAction or Card
    # reveal only:
    suggestion: Suggestion | None = None
    suggesting_player_id: int | None = None
    offered: list[Card] = field(default_factory=list)

class HumanClueAgent(ClueAgent):
    def __init__(self, player_id, *, name=None, decision_timeout=None,
                 on_request=None, trace_attributes=None): ...
    pending: PendingDecision | None
    async def _choose_action(self, observation) -> PlayerAction
    async def _choose_reveal(self, observation, suggestion, matches,
                             suggesting_player_id) -> Card
    def submit_action(self, action: PlayerAction) -> None   # raises if none pending / wrong kind
    def submit_reveal(self, card: Card) -> None             # raises if card not in offered
```

Behaviour details:

- Mirrors `RandomClueAgent`: if `not observation.active` or it is not this
  seat's turn, return `PassTurn()` immediately without asking anyone. The
  engine already skips eliminated seats via `next_player_id`; this is belt
  and braces.
- `_choose_reveal` is only called when `matches` is non-empty. If there is
  exactly one match, still ask the human (it is their card to show and the
  UI should say so), but the plan below lets the front-end one-click it.
- `submit_*` validates against the pending decision and sets the future's
  result. A second submit while the future is done raises; the handler maps
  that to HTTP 409.
- `on_request` is an async callback the agent calls right after creating a
  `PendingDecision`. The service passes a closure that publishes the
  `decision_requested` envelope. Keeping publishing out of the agent keeps
  `agents/` free of transport imports, matching the existing layering.
- `decision_timeout` (seconds, `None` = wait forever). On timeout an action
  decision resolves to `PassTurn()` and a reveal to `choice(matches)` (the
  base class default). The span gets `clue.human_timed_out=True`. Default
  is `None`: for testing, a human walking away should pause the game, not
  silently forfeit turns.
- No `notes`/scratchpad: humans keep their own notes. `SeatConfig.scratchpad`
  is ignored for human seats.

### `service/schemas.py`

- `SeatConfig.agent_type: Literal["random", "llm", "human"]`.
- `SeatConfig.name: str | None` (display name, human seats only; also copied
  onto `Player.name`, which already exists and is unused).
- `SeatConfig.decision_timeout: float | None` (human seats only, must be > 0
  if set). Validator: `provider`/`model`/`thinking_budget`/`scratchpad`
  set on a human seat is a 422, same style as the existing llm check.
- `CreateGameResponse.human_seats: list[HumanSeatCredential]` where
  `HumanSeatCredential = {player_id, name, token, join_url}`. Tokens are
  `secrets.token_urlsafe(16)`, minted per human seat at game creation.
- New response models: `PendingDecisionView`, `PlayerView`
  (`observation: PlayerObservation` + `pending: PendingDecisionView | None`
  + `status`), `SubmitActionRequest` (= `PlayerAction`), `SubmitRevealRequest`
  (`{card: Card}`), `CardCatalog` (`{people, weapons, rooms}`).

### `service/agent_factory.py`

- `build_agent` gains a `human` branch returning `HumanClueAgent`. It needs
  the `on_request` callback, which needs the transport, so `build_agent`
  takes an optional `on_decision_request` argument that `start_game` passes.
- `seat_identity` emits `clue_agent_type: "human"` and `clue_name` so traces
  group by person the way they group by model.
- `list_providers()` adds `{"name": "human", "requires_key": None,
  "available": True}` so the existing seat-type dropdown picks it up with no
  front-end special-casing.

### `service/registry.py`

- `RunRecord.player_tokens: dict[int, str]` (player_id → token).
- `RunRecord.human_player_ids` property (derived from `agents`).

### `service/game_runner.py`

- `start_game` mints tokens for human seats, passes `on_decision_request`
  into `build_agent`, copies `seat.name` onto `state.players[i].name`.
- `_play` is unchanged. Note the loop's `max_turns` bound and error
  handling already cover the human path: a `ValueError` from the engine
  (should not happen, the handler validates first) still ends the run with
  an `error` envelope rather than crashing the service.
- `agent_reveal_chooser` is unchanged.

### `service/app.py` — new endpoints

All player endpoints require `X-Player-Token`. A bad or missing token is 403;
a token for a different seat is 403; an unknown game is 404; a non-human
seat is 404 ("player 2 is not a human seat").

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/cards` | Card catalog (enum values) so the UI never hardcodes names. |
| `GET` | `/games/{id}/players/{pid}` | `PlayerView`: full `PlayerObservation` for this seat (hand, card log, private reveals, public history, public deductions) plus the pending decision, if any. |
| `POST` | `/games/{id}/players/{pid}/action` | Body is a `PlayerAction`. 409 if no action is pending for this seat; 422 on shape errors. Returns the updated `PlayerView`. |
| `POST` | `/games/{id}/players/{pid}/reveal` | Body `{card}`. 400 if the card is not among the offered cards; 409 if nothing is pending. |

`GET /games/{id}` (`GameStatusResponse`) gains `waiting_on: {player_id,
kind} | null` so a spectator can see the game is parked on a human without
holding a token.

### `transport/sse.py` — new envelope, and one privacy fix

New envelope kinds published by the runner (public, safe for spectators):

- `decision_requested {player_id, decision_kind: "action"|"reveal", turn}` —
  emitted by the `on_request` callback. For reveals it deliberately does
  **not** include the suggestion's matching cards; the player fetches those
  through their token.
- `decision_resolved {player_id, decision_kind, turn}` — emitted by the action/reveal
  handler after the future is set, so UIs can clear their "waiting" state
  without polling. (`turn_taken` already follows an action; reveals have no
  public marker today, hence this.)

Privacy fix: `SSEBroadcastTransport.send` currently publishes
`private_reveal` **with the card identity** to every SSE subscriber. In an
all-AI game that is a debugging aid. With a human at the table it is a
cheat sheet visible to any human who unticks "show reveals" in the console.
Change: `SSEBroadcastTransport(publish_private_cards: bool)`; the runner
sets it to `False` when any seat is human. The `private_reveal` envelope
then carries `revealing_player_id` and `receiving_player_id` only. Humans
see their own reveals through `GET /games/{id}/players/{pid}`.

### `frontend/`

Setup screen:
- "human" appears in the seat-type select automatically via `/providers`.
- When a seat is human, show a `name` text input in place of model /
  thinking / notes fields (reuse the existing `syncModelVisibility`
  pattern).
- Optional per-seat `decision timeout (s)` input; blank = wait forever.

Game screen:
- After `POST /games`, stash `human_seats` in memory and in
  `localStorage` under the game id, so a reload keeps the tokens.
- Render one **player panel** per human seat the browser holds a token for.
  Hot-seat testing works out of the box (one browser holds every token);
  multi-browser works via `join_url` (`/?game=<id>&player=<pid>&token=<t>`),
  which the game screen shows next to each human seat as a copyable link.
  Opening a join URL enters the game screen directly with just that one
  panel.
- Panel contents, from `GET /games/{id}/players/{pid}`:
  - Hand (cards grouped by type).
  - Detective notebook: the seat's `card_log` rendered as a checklist
    (`seen` / `who_has`), plus `public_deductions.per_player` as
    "cannot have" / "holds one of" rows. This is what a person at a real
    table writes on paper; showing it is not deduction on their behalf,
    it is the same data an LLM seat gets in its prompt.
  - Private reveals list (who showed me what, and what I showed to whom).
- Decision form, rendered when `pending` is non-null:
  - `action`: radio for suggestion / accusation / pass; three selects
    populated from `/cards`; submit → `POST .../action`. Accusation gets a
    confirm step because a wrong one eliminates the seat.
  - `reveal`: the suggestion being answered, who asked, and one button per
    offered card; submit → `POST .../reveal`. Single offered card = single
    button, still one click.
  - Errors from the API (409/400) render inline on the panel.
- Triggering refresh: on `decision_requested` for a held seat, fetch the
  player view and show the form; on `turn_taken` / `decision_resolved` /
  `private_reveal` involving a held seat, refetch to refresh the notebook.
  The existing single SSE connection stays the only stream.
- Highlight: the seat card of a human that the game is waiting on gets a
  "your turn" badge; the header shows "Waiting on Alice (reveal)".

### `main.py` (optional, small)

A `TerminalHumanAgent(ClueAgent)` that prompts on stdin makes hot-seat play
possible without Docker (`uv run python main.py --humans 1`). It is ~60
lines, shares nothing with the service path except the base class, and is
useful for quick manual checks. Listed last in the build order; skip if the
console is enough.

## 3. Concurrency and failure notes

- Futures are created and resolved on the same event loop (FastAPI handlers
  and the `_play` task share uvicorn's loop), so `set_result` from a handler
  is safe. Handlers check `future.done()` first and return 409 on a
  duplicate submit.
- The engine's validity checks (active player, correct turn, card in
  matches) are duplicated in the handler so a bad submit is a 4xx, never an
  `error` envelope ending the run. `submit_reveal` checks `card in offered`;
  `submit_action` checks the kind matches and that the seat is active.
- A human who never answers parks the game indefinitely when
  `decision_timeout` is unset. That is intended for testing. `max_turns` is
  unaffected (it counts turns, not time).
- Service restart loses in-memory runs today; unchanged by this work.
- Tokens are bearer secrets in a test console, not auth. They stop one
  human trivially reading another's hand from the same origin; they are
  not a security boundary and the plan does not pretend otherwise.

## 4. Tests

`tests/test_human_agent.py` (no FastAPI needed):
- `_choose_action` publishes a pending decision and returns the submitted
  action; `submit_action` before a request raises; double submit raises.
- `_choose_reveal` with two matches returns the submitted card; off-menu
  card raises and leaves the decision pending.
- Inactive seat returns `PassTurn` without creating a pending decision.
- Timeout: action → `PassTurn`, reveal → one of `matches`.
- Engine integration: `GameEngine` + `LocalTransport`, seats `[Human,
  Random]`, drive the human from a task that reads `agent.pending` and
  submits; game reaches `FINISHED` or the turn cap.

`tests/test_service.py` additions:
- Creating a game with a human seat returns one credential per human seat
  and none for other seats.
- Human seat with `provider` set is 422.
- Player endpoints: missing token 403, wrong seat's token 403, non-human
  seat 404.
- `POST .../action` with nothing pending is 409.
- `POST .../reveal` with a card not offered is 400 and `pending` survives.
- End-to-end: 2 seats `[human, random]`, `max_turns=60`. Test loop polls
  `GET /games/{id}/players/0` until `pending`, submits a suggestion (or a
  reveal), repeats until `game_finished`. Asserts the SSE feed contains
  `decision_requested` and `decision_resolved` for player 0, and that
  `private_reveal` envelopes carry no `card` field.
- Spectator feed in an all-random game still includes `card` on
  `private_reveal` (regression guard for the flag default).

`tests/test_agent_factory.py`: `build_agent` for a human seat returns a
`HumanClueAgent` with `clue.agent_type == "human"` in trace attributes.

## 5. Build order

Each step leaves `uv run pytest` green.

1. `agents/human.py` + `tests/test_human_agent.py`. Pure Python, no service.
2. Schemas + factory + registry: `agent_type: "human"`, tokens, `name`,
   `decision_timeout`, `/providers` entry. Tests for validation.
3. Runner + transport: `on_decision_request` wiring, `decision_requested` /
   `decision_resolved` envelopes, `publish_private_cards` flag.
4. `app.py` endpoints + `/cards` + `waiting_on` on game status. End-to-end
   service test.
5. Front-end: setup changes, player panel, decision forms, join URLs.
   Manual check with `docker compose up --build`: 1 human + 2 random, then
   1 human + 1 LLM seat, then 2 humans in two browser tabs via join URLs.
6. README: new seat type, endpoints table, privacy note on the spectator
   feed.
7. Optional: `TerminalHumanAgent` in `main.py`.

## 6. Deliberately out of scope

- Reconnect/forfeit endpoints (a human leaving mid-game).
- Auth beyond per-seat bearer tokens.
- A per-player SSE stream. The single spectator stream plus a token-guarded
  GET is enough and avoids a second long-lived connection per human. If
  polling-on-event ever feels laggy, a filtered per-player stream is a
  drop-in addition to `SSEBroadcastTransport`.
- Board movement / dice. The engine has no board; suggestions are
  unconstrained by room, and this plan keeps it that way.
