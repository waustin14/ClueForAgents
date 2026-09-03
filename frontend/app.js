const API = "/api";
const JAEGER_BASE = "http://localhost:16686";

const els = {
  nPlayers: document.getElementById("n-players"),
  maxTurns: document.getElementById("max-turns"),
  seats: document.getElementById("seats"),
  startButton: document.getElementById("start-button"),
  setupError: document.getElementById("setup-error"),
  setupScreen: document.getElementById("setup-screen"),
  gameScreen: document.getElementById("game-screen"),
  gameId: document.getElementById("game-id"),
  gameStatus: document.getElementById("game-status"),
  turnNumber: document.getElementById("turn-number"),
  currentPlayer: document.getElementById("current-player"),
  waitingOnLine: document.getElementById("waiting-on-line"),
  waitingOn: document.getElementById("waiting-on"),
  jaegerLink: document.getElementById("jaeger-link"),
  newGameButton: document.getElementById("new-game-button"),
  seatsCard: document.getElementById("seats-card"),
  seatCards: document.getElementById("seat-cards"),
  feedCard: document.getElementById("feed-card"),
  feed: document.getElementById("feed"),
  showReveals: document.getElementById("show-reveals"),
  playerPanelsCard: document.getElementById("player-panels-card"),
  playerPanels: document.getElementById("player-panels"),
};

let providers = [];
let eventSource = null;
let seatConfigs = [];
const eliminated = new Set();

// Human-seat state. `heldSeats` maps player_id -> bearer token for every
// seat this browser can act as; it's the union of whatever was minted here
// (hot-seat testing: the creating browser holds every human seat's token)
// and whatever a join link or localStorage handed us. `joinMode` narrows
// rendering to a single panel when the game screen was entered via a
// `?game=&player=&token=` join link, per the plan: that link should feel
// like "you're playing this one seat", not a spectator console.
let currentGameId = null;
let heldSeats = new Map();
let joinMode = false;
let joinedPlayerId = null;
let cardCatalog = null;
// Per-seat counter of panel refreshes issued. Several envelopes that land
// within milliseconds of each other (`decision_resolved`, `private_reveal`,
// `turn_taken`) each trigger a refetch, and the one issued *before* the turn
// resolved can return *after* the ones issued afterwards — rendering a view
// with no reveal in it on top of one that had it. Only the latest issued
// request for a seat is allowed to render.
const panelRefreshSeq = new Map();

function storageKey(gameId) {
  return `clue:human-seats:${gameId}`;
}

function loadStoredCredentials(gameId) {
  let list = [];
  try {
    list = JSON.parse(localStorage.getItem(storageKey(gameId)) || "[]");
  } catch {
    list = [];
  }
  return new Map(list.map((s) => [s.player_id, s.token]));
}

function storeCredential(gameId, playerId, token) {
  const key = storageKey(gameId);
  let list = [];
  try {
    list = JSON.parse(localStorage.getItem(key) || "[]");
  } catch {
    list = [];
  }
  const idx = list.findIndex((s) => s.player_id === playerId);
  const entry = { player_id: playerId, token };
  if (idx >= 0) list[idx] = entry;
  else list.push(entry);
  localStorage.setItem(key, JSON.stringify(list));
}

function storeAllCredentials(gameId, humanSeats) {
  if (!humanSeats.length) return;
  localStorage.setItem(
    storageKey(gameId),
    JSON.stringify(humanSeats.map((s) => ({ player_id: s.player_id, token: s.token })))
  );
}

function buildJoinUrl(gameId, playerId, token) {
  return `${location.origin}/?game=${encodeURIComponent(gameId)}&player=${playerId}&token=${encodeURIComponent(token)}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

async function loadProviders() {
  const response = await fetch(`${API}/providers`);
  providers = await response.json();
  renderSeatRows();
}

function renderSeatRows() {
  const n = clamp(parseInt(els.nPlayers.value, 10) || 2, 2, 6);
  els.nPlayers.value = n;
  const existing = els.seats.querySelectorAll(".seat-row");

  els.seats.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const row = document.createElement("div");
    row.className = "seat-row";

    const label = document.createElement("span");
    label.textContent = `Seat ${i}`;

    const select = document.createElement("select");
    select.dataset.seat = String(i);
    select.dataset.role = "agent-type";
    for (const p of providers) {
      const option = document.createElement("option");
      option.value = p.name;
      option.textContent = p.name === "random" ? "Random (no LLM)" : p.name === "human" ? "Human (you)" : p.name;
      option.disabled = !p.available;
      select.appendChild(option);
    }
    // Preserve a previous selection at this seat index across resizes.
    const prior = existing[i]?.querySelector('[data-role="agent-type"]')?.value;
    if (prior) select.value = prior;

    const modelInput = document.createElement("input");
    modelInput.type = "text";
    modelInput.dataset.seat = String(i);
    modelInput.dataset.role = "model";
    modelInput.placeholder = "model name, e.g. claude-sonnet-5";
    const priorModel = existing[i]?.querySelector('[data-role="model"]')?.value;
    if (priorModel) modelInput.value = priorModel;

    const thinkingInput = document.createElement("input");
    thinkingInput.type = "number";
    thinkingInput.min = "0";
    thinkingInput.dataset.seat = String(i);
    thinkingInput.dataset.role = "thinking-budget";
    thinkingInput.placeholder = "think tokens";
    thinkingInput.title =
      "Reasoning budget in tokens. Left blank, the provider default applies; " +
      "0 turns extended thinking off where the provider supports it.";
    const priorThinking = existing[i]?.querySelector('[data-role="thinking-budget"]')?.value;
    if (priorThinking) thinkingInput.value = priorThinking;

    // The scratchpad is a per-seat experimental variable, not a global
    // setting: whether a model plays better carrying its own notes forward
    // is exactly the kind of thing you'd want to A/B across seats in one game.
    const scratchpadLabel = document.createElement("label");
    scratchpadLabel.className = "seat-toggle";
    const scratchpadInput = document.createElement("input");
    scratchpadInput.type = "checkbox";
    scratchpadInput.dataset.seat = String(i);
    scratchpadInput.dataset.role = "scratchpad";
    scratchpadInput.checked =
      existing[i]?.querySelector('[data-role="scratchpad"]')?.checked ?? false;
    scratchpadLabel.append(scratchpadInput, document.createTextNode("notes"));
    scratchpadLabel.title =
      "Let this seat write itself private notes and read them back next turn.";

    // A human seat has no model to configure — it takes a display name
    // instead (shown on its seat card and join link) and an optional
    // decision timeout instead of a thinking budget.
    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.dataset.seat = String(i);
    nameInput.dataset.role = "human-name";
    nameInput.placeholder = "name";
    const priorName = existing[i]?.querySelector('[data-role="human-name"]')?.value;
    if (priorName) nameInput.value = priorName;

    const timeoutInput = document.createElement("input");
    timeoutInput.type = "number";
    timeoutInput.min = "0";
    timeoutInput.step = "any";
    timeoutInput.dataset.seat = String(i);
    timeoutInput.dataset.role = "decision-timeout";
    timeoutInput.placeholder = "decision timeout (s)";
    timeoutInput.title = "Seconds this seat has to answer before it auto-passes/auto-reveals. Blank waits forever.";
    const priorTimeout = existing[i]?.querySelector('[data-role="decision-timeout"]')?.value;
    if (priorTimeout) timeoutInput.value = priorTimeout;

    const syncFieldVisibility = () => {
      const isHuman = select.value === "human";
      const isRandom = select.value === "random";
      const isLlm = !isHuman && !isRandom;
      for (const field of [modelInput, thinkingInput, scratchpadInput]) {
        field.disabled = !isLlm;
      }
      for (const field of [modelInput, thinkingInput, scratchpadLabel]) {
        field.style.visibility = isLlm ? "visible" : "hidden";
      }
      for (const field of [nameInput, timeoutInput]) {
        field.disabled = !isHuman;
        field.style.visibility = isHuman ? "visible" : "hidden";
      }
    };
    select.addEventListener("change", syncFieldVisibility);
    syncFieldVisibility();

    row.append(label, select, modelInput, thinkingInput, scratchpadLabel, nameInput, timeoutInput);
    els.seats.appendChild(row);
  }
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function readSeatConfigs() {
  const rows = [...els.seats.querySelectorAll(".seat-row")];
  return rows.map((row) => {
    const agentType = row.querySelector('[data-role="agent-type"]').value;
    if (agentType === "random") {
      return { agent_type: "random" };
    }
    if (agentType === "human") {
      const name = row.querySelector('[data-role="human-name"]').value.trim();
      const timeout = row.querySelector('[data-role="decision-timeout"]').value.trim();
      return {
        agent_type: "human",
        name: name || undefined,
        decision_timeout: timeout === "" ? undefined : Number(timeout),
      };
    }
    const model = row.querySelector('[data-role="model"]').value.trim();
    // A blank budget must stay absent rather than become 0 — 0 is a real,
    // different instruction ("no extended thinking"), not "unset".
    const thinking = row.querySelector('[data-role="thinking-budget"]').value.trim();
    return {
      agent_type: "llm",
      provider: agentType,
      model: model || undefined,
      thinking_budget: thinking === "" ? undefined : Number(thinking),
      scratchpad: row.querySelector('[data-role="scratchpad"]').checked,
    };
  });
}

async function startGame() {
  els.setupError.hidden = true;
  seatConfigs = readSeatConfigs();

  const body = {
    n_players: seatConfigs.length,
    seats: seatConfigs,
    max_turns: clamp(parseInt(els.maxTurns.value, 10) || 500, 1, 2000),
  };

  let response;
  try {
    response = await fetch(`${API}/games`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    showSetupError(`Could not reach the game-logic service: ${err}`);
    return;
  }

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    showSetupError(detail.detail ? JSON.stringify(detail.detail) : `HTTP ${response.status}`);
    return;
  }

  const { game_id, run_id, stream_url, human_seats } = await response.json();
  heldSeats = new Map(human_seats.map((s) => [s.player_id, s.token]));
  storeAllCredentials(game_id, human_seats);
  joinMode = false;
  joinedPlayerId = null;
  history.replaceState(null, "", `?game=${encodeURIComponent(game_id)}`);
  enterGameScreen(game_id, run_id, stream_url);
}

function showSetupError(message) {
  els.setupError.textContent = message;
  els.setupError.hidden = false;
}

async function resumeGame(gameId, urlPlayerId, urlToken) {
  let statusResponse;
  try {
    statusResponse = await fetch(`${API}/games/${gameId}`);
  } catch (err) {
    showSetupError(`Could not reach the game-logic service: ${err}`);
    await loadProviders();
    return;
  }
  if (!statusResponse.ok) {
    history.replaceState(null, "", location.pathname);
    await loadProviders();
    showSetupError(`Game ${gameId} was not found — it may have restarted since this link was made.`);
    return;
  }
  const status = await statusResponse.json();
  seatConfigs = status.seats;

  heldSeats = loadStoredCredentials(gameId);
  const hasJoinCredential = urlPlayerId !== null && urlPlayerId !== "" && Boolean(urlToken);
  if (hasJoinCredential) {
    const playerId = Number(urlPlayerId);
    heldSeats.set(playerId, urlToken);
    storeCredential(gameId, playerId, urlToken);
    joinMode = true;
    joinedPlayerId = playerId;
  } else {
    joinMode = false;
    joinedPlayerId = null;
  }

  enterGameScreen(gameId, status.run_id, `/games/${gameId}/events`);
}

function panelsToRender() {
  if (!joinMode) return [...heldSeats.keys()];
  return heldSeats.has(joinedPlayerId) ? [joinedPlayerId] : [];
}

function enterGameScreen(gameId, runId, streamUrl) {
  eliminated.clear();
  currentGameId = gameId;
  els.setupScreen.hidden = true;
  els.gameScreen.hidden = false;
  els.gameId.textContent = gameId;
  els.gameStatus.textContent = "running";
  els.turnNumber.textContent = "0";
  els.currentPlayer.textContent = "0";
  els.waitingOnLine.hidden = true;
  els.feed.innerHTML = "";
  els.jaegerLink.href =
    `${JAEGER_BASE}/search?service=clue-game-logic&tags=${encodeURIComponent(
      JSON.stringify({ "clue.run_id": runId })
    )}`;

  // A join link is meant to feel like "you're sitting down to play this one
  // seat" — the spectator seat grid and the raw event feed are noise for
  // that person, so both stay hidden and only their own panel shows.
  els.seatsCard.hidden = joinMode;
  els.feedCard.hidden = joinMode;

  renderSeatCards();
  renderPlayerPanels();
  refreshGameStatus();

  if (eventSource) eventSource.close();
  eventSource = new EventSource(`${API}${streamUrl}`);
  // The server replays the run's full backlog on every (re)connect, and
  // EventSource reconnects on its own after a drop — so the feed is rebuilt
  // from scratch each time rather than having the replay appended to what is
  // already on screen.
  eventSource.onopen = () => {
    els.feed.innerHTML = "";
  };
  // Frames are sent unnamed (see service/app.py::_format_sse) precisely so
  // they land here, and are dispatched on the envelope's own `kind`.
  eventSource.onmessage = (msg) => handleEnvelope(JSON.parse(msg.data));
  eventSource.onerror = () => {
    appendFeedEntry({ kind: "connection" }, "Connection to event stream lost.");
  };
}

function renderSeatCards() {
  els.seatCards.innerHTML = "";
  seatConfigs.forEach((seat, i) => {
    const card = document.createElement("div");
    card.className = "seat-card";
    card.id = `seat-card-${i}`;
    const title = document.createElement("div");
    title.className = "seat-title";
    title.textContent =
      seat.agent_type === "human" ? `Player ${i} — ${seat.name || "Human"}` : `Player ${i}`;
    const config = document.createElement("div");
    config.className = "seat-config";
    config.textContent =
      seat.agent_type === "random"
        ? "Random agent"
        : seat.agent_type === "human"
          ? "Human"
          : `${seat.provider} / ${seat.model}`;
    card.append(title, config);

    if (seat.agent_type === "human" && heldSeats.has(i)) {
      const url = buildJoinUrl(currentGameId, i, heldSeats.get(i));
      const joinRow = document.createElement("div");
      joinRow.className = "seat-join";
      const input = document.createElement("input");
      input.type = "text";
      input.readOnly = true;
      input.value = url;
      input.addEventListener("focus", () => input.select());
      const copyButton = document.createElement("button");
      copyButton.type = "button";
      copyButton.className = "secondary";
      copyButton.textContent = "Copy link";
      copyButton.addEventListener("click", () => {
        navigator.clipboard?.writeText(url).catch(() => {});
      });
      joinRow.append(input, copyButton);
      card.appendChild(joinRow);
    }

    els.seatCards.appendChild(card);
  });
}

function setCurrentSeat(playerId) {
  els.seatCards.querySelectorAll(".seat-card").forEach((c) => c.classList.remove("current"));
  document.getElementById(`seat-card-${playerId}`)?.classList.add("current");
}

function markEliminated(playerId) {
  eliminated.add(playerId);
  document.getElementById(`seat-card-${playerId}`)?.classList.add("eliminated");
}

function updateWaitingOn(waitingOn) {
  els.seatCards.querySelectorAll(".seat-card").forEach((c) => c.classList.remove("waiting"));
  if (!waitingOn) {
    els.waitingOnLine.hidden = true;
    return;
  }
  document.getElementById(`seat-card-${waitingOn.player_id}`)?.classList.add("waiting");
  const name = seatConfigs[waitingOn.player_id]?.name;
  els.waitingOn.textContent = name
    ? `${name} (${waitingOn.kind})`
    : `player ${waitingOn.player_id} (${waitingOn.kind})`;
  els.waitingOnLine.hidden = false;
}

async function refreshGameStatus() {
  if (!currentGameId) return;
  try {
    const response = await fetch(`${API}/games/${currentGameId}`);
    if (!response.ok) return;
    const status = await response.json();
    els.gameStatus.textContent = status.status;
    els.turnNumber.textContent = String(status.turn_number);
    els.currentPlayer.textContent = String(status.current_player_id);
    updateWaitingOn(status.waiting_on);
  } catch {
    // Best-effort refresh — the SSE feed is the source of truth for the log.
  }
}

function handleEnvelope(envelope) {
  switch (envelope.kind) {
    case "game_started":
      appendFeedEntry(envelope, `Game started with ${envelope.n_players} players.`);
      break;
    case "turn_taken": {
      els.turnNumber.textContent = String(envelope.turn);
      setCurrentSeat(envelope.player_id);
      appendFeedEntry(
        envelope,
        `Player ${envelope.player_id} played ${describeAction(envelope.action)}`
      );
      refreshAllHeldPanels();
      refreshGameStatus();
      break;
    }
    case "public_event":
      handlePublicEvent(envelope.event);
      break;
    case "private_reveal":
      if (els.showReveals.checked) {
        const e = envelope.event;
        appendFeedEntry(
          envelope,
          `[private] Player ${e.revealing_player_id} showed a card to player ${e.receiving_player_id}`
        );
      }
      refreshAllHeldPanels();
      break;
    case "decision_requested":
      appendFeedEntry(
        envelope,
        `Player ${envelope.player_id} needs to ${envelope.decision_kind === "reveal" ? "reveal a card" : "act"}.`
      );
      if (heldSeats.has(envelope.player_id) && panelsToRender().includes(envelope.player_id)) {
        refreshPlayerPanel(envelope.player_id);
      }
      refreshGameStatus();
      break;
    case "decision_resolved":
      appendFeedEntry(
        envelope,
        `Player ${envelope.player_id} resolved a ${envelope.decision_kind} decision.`
      );
      refreshAllHeldPanels();
      refreshGameStatus();
      break;
    case "error":
      els.gameStatus.textContent = "error";
      appendFeedEntry(envelope, `Error: ${envelope.message}`);
      break;
    case "game_finished":
      els.gameStatus.textContent = envelope.status;
      els.turnNumber.textContent = String(envelope.turn_number);
      appendFeedEntry(
        envelope,
        envelope.winning_player_id !== null && envelope.winning_player_id !== undefined
          ? `Game finished — winner: player ${envelope.winning_player_id}`
          : `Game finished — status: ${envelope.status}`
      );
      refreshAllHeldPanels();
      els.waitingOnLine.hidden = true;
      eventSource?.close();
      break;
    default:
      appendFeedEntry(envelope, JSON.stringify(envelope));
  }
}

function handlePublicEvent(event) {
  if ("suggesting_player_id" in event) {
    const s = event.suggestion;
    appendFeedEntry(
      { kind: "public_event" },
      `Player ${event.suggesting_player_id} suggested ${s.person} / ${s.weapon} / ${s.room}` +
        (event.disproving_player_id !== null
          ? ` — disproved by player ${event.disproving_player_id}`
          : " — nobody could disprove it")
    );
  } else if ("accusing_player_id" in event) {
    appendFeedEntry(
      { kind: "public_event" },
      `Player ${event.accusing_player_id} accused ${event.accusation.person} / ` +
        `${event.accusation.weapon} / ${event.accusation.room} — ${
          event.correct ? "CORRECT" : "wrong"
        }`
    );
    if (!event.correct) markEliminated(event.accusing_player_id);
  } else if ("solution" in event) {
    const sol = event.solution;
    appendFeedEntry(
      { kind: "public_event" },
      `Solution was: ${sol.person} / ${sol.weapon} / ${sol.room}`
    );
  } else {
    appendFeedEntry({ kind: "public_event" }, JSON.stringify(event));
  }
}

function describeAction(action) {
  if (action.kind === "pass") return "pass";
  if (action.kind === "suggestion") {
    return `a suggestion (${action.person} / ${action.weapon} / ${action.room})`;
  }
  if (action.kind === "accusation") {
    return `an accusation (${action.person} / ${action.weapon} / ${action.room})`;
  }
  return action.kind;
}

function appendFeedEntry(envelope, text) {
  const entry = document.createElement("div");
  entry.className = `feed-entry kind-${envelope.kind}`;
  const kindLabel = document.createElement("span");
  kindLabel.className = "feed-kind";
  kindLabel.textContent = envelope.kind;
  entry.append(kindLabel, document.createTextNode(text));
  els.feed.appendChild(entry);
  els.feed.scrollTop = els.feed.scrollHeight;
}

// --- Human player panels -----------------------------------------------
//
// Everything below renders `GET /games/{id}/players/{pid}` for every seat
// this browser holds a token for: hand, detective notebook (card_log +
// public_deductions — the same data an LLM seat gets in its prompt, not
// deduction on the person's behalf), private reveals, and a decision form
// when `pending` is non-null. Panels are plain innerHTML templates rebuilt
// on every refresh rather than incrementally patched — simple, and cheap
// enough for a one-to-few-human test console.

async function getCardCatalog() {
  if (!cardCatalog) {
    const response = await fetch(`${API}/cards`);
    cardCatalog = await response.json();
  }
  return cardCatalog;
}

function panelElementId(playerId) {
  return `player-panel-${playerId}`;
}

function ensurePanelSkeleton(playerId) {
  let panel = document.getElementById(panelElementId(playerId));
  if (!panel) {
    panel = document.createElement("div");
    panel.id = panelElementId(playerId);
    panel.className = "player-panel";
    els.playerPanels.appendChild(panel);
  }
  return panel;
}

function renderPlayerPanels() {
  els.playerPanels.innerHTML = "";
  const toRender = panelsToRender();
  els.playerPanelsCard.hidden = toRender.length === 0;
  for (const playerId of toRender) {
    ensurePanelSkeleton(playerId);
    refreshPlayerPanel(playerId);
  }
}

async function refreshAllHeldPanels() {
  for (const playerId of panelsToRender()) {
    await refreshPlayerPanel(playerId);
  }
}

async function refreshPlayerPanel(playerId) {
  const token = heldSeats.get(playerId);
  if (!token || !currentGameId) return;
  const panel = ensurePanelSkeleton(playerId);
  const seq = (panelRefreshSeq.get(playerId) ?? 0) + 1;
  panelRefreshSeq.set(playerId, seq);
  const isStale = () => panelRefreshSeq.get(playerId) !== seq;
  let response;
  try {
    response = await fetch(`${API}/games/${currentGameId}/players/${playerId}`, {
      headers: { "X-Player-Token": token },
    });
  } catch (err) {
    if (isStale()) return;
    panel.innerHTML = `<div class="error">Could not reach the service: ${escapeHtml(String(err))}</div>`;
    return;
  }
  if (isStale()) return;
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    panel.innerHTML = `<div class="error">HTTP ${response.status}${
      detail.detail ? `: ${escapeHtml(String(detail.detail))}` : ""
    }</div>`;
    return;
  }
  const view = await response.json();
  if (isStale()) return;
  await renderPlayerPanel(panel, playerId, view);
}

async function renderPlayerPanel(panel, playerId, view) {
  const obs = view.observation;
  const name = seatConfigs[playerId]?.name;
  const title = name ? `Player ${playerId} — ${escapeHtml(name)}` : `Player ${playerId}`;
  const badge = view.pending
    ? `<span class="badge">${view.pending.kind === "reveal" ? "Reveal needed" : "Your turn"}</span>`
    : "";

  panel.innerHTML = `
    <div class="panel-header"><h4>${title}</h4>${badge}</div>
    ${renderLastSuggestionOutcome(playerId, obs)}
    <div class="panel-section"><h5>Hand</h5>${renderHand(obs.own_cards)}</div>
    <div class="panel-section">
      <h5>Detective notebook</h5>
      ${renderCardLog(obs.card_log)}
      ${renderDeductions(obs.public_deductions)}
    </div>
    <div class="panel-section"><h5>Private reveals</h5>${renderPrivateReveals(playerId, obs.private_reveals)}</div>
    <div class="panel-section decision-form" id="${panelElementId(playerId)}-decision"></div>
  `;

  const formHost = panel.querySelector(`#${panelElementId(playerId)}-decision`);
  if (view.pending) {
    await renderDecisionForm(formHost, playerId, view.pending);
  }
}

function groupCardsByType(cards) {
  const groups = { person: [], weapon: [], room: [] };
  for (const card of cards) groups[card.type]?.push(card.value);
  return groups;
}

function renderCardListOrNone(values) {
  if (!values.length) return `<li class="muted">none</li>`;
  return values.map((v) => `<li>${escapeHtml(v)}</li>`).join("");
}

function renderHand(cards) {
  const g = groupCardsByType(cards);
  return `<div class="hand-groups">
    <div><strong>People</strong><ul>${renderCardListOrNone(g.person)}</ul></div>
    <div><strong>Weapons</strong><ul>${renderCardListOrNone(g.weapon)}</ul></div>
    <div><strong>Rooms</strong><ul>${renderCardListOrNone(g.room)}</ul></div>
  </div>`;
}

function renderCardLog(cardLog) {
  const categories = [
    ["People", cardLog.people],
    ["Weapons", cardLog.weapons],
    ["Rooms", cardLog.rooms],
  ];
  const columns = categories
    .map(([label, entries]) => {
      const items = Object.entries(entries)
        .map(([value, entry]) => {
          const status = entry.seen ? `held by player ${entry.who_has}` : "unknown";
          return `<li>${escapeHtml(value)} — ${status}</li>`;
        })
        .join("");
      return `<div><strong>${label}</strong><ul>${items}</ul></div>`;
    })
    .join("");
  return `<div class="card-log">${columns}</div>`;
}

function renderDeductions(deductions) {
  const perPlayer = deductions?.per_player || [];
  const rows = perPlayer
    .map((p) => {
      const parts = [];
      if (p.cannot_have.length) {
        parts.push(`cannot have: ${p.cannot_have.map(escapeHtml).join(", ")}`);
      }
      for (const group of p.holds_at_least_one_of) {
        parts.push(`holds one of: ${group.map(escapeHtml).join(", ")}`);
      }
      if (!parts.length) return "";
      return `<li>Player ${p.player_id}: ${parts.join(" — ")}</li>`;
    })
    .filter(Boolean)
    .join("");
  if (!rows) return "";
  return `<div class="deductions"><strong>Public deductions</strong><ul>${rows}</ul></div>`;
}

function renderLastSuggestionOutcome(playerId, obs) {
  // What came of this seat's most recent suggestion, pinned to the top of
  // the panel. The "Private reveals" list further down carries the same
  // fact, but a person who has just clicked "Suggest" is looking for one
  // answer — who showed me what — not a growing list to scan.
  const suggestions = obs.public_history.filter(
    (e) => "suggesting_player_id" in e && e.suggesting_player_id === playerId
  );
  if (!suggestions.length) return "";
  const last = suggestions[suggestions.length - 1];
  const s = last.suggestion;
  const triple = `${escapeHtml(s.person)} / ${escapeHtml(s.weapon)} / ${escapeHtml(s.room)}`;

  let outcome;
  let tone = "info";
  if (last.disproving_player_id === null) {
    outcome = "Nobody could disprove it.";
    tone = "strong";
  } else {
    const reveal = obs.private_reveals.find(
      (r) => r.turn === last.turn && r.receiving_player_id === playerId
    );
    outcome = reveal
      ? `Player ${last.disproving_player_id} showed you <strong>${escapeHtml(reveal.card.value)}</strong>.`
      : `Player ${last.disproving_player_id} is choosing a card to show you…`;
  }
  const passed = last.unable_to_disprove.length
    ? ` Could not disprove: player${last.unable_to_disprove.length > 1 ? "s" : ""} ${last.unable_to_disprove.join(", ")}.`
    : "";
  return `<div class="suggestion-outcome ${tone}">
    <span class="outcome-label">Your suggestion, turn ${last.turn}</span>
    <span>${triple} — ${outcome}${passed}</span>
  </div>`;
}

function renderPrivateReveals(playerId, reveals) {
  if (!reveals.length) return `<p class="muted">None yet.</p>`;
  const items = reveals
    .map((r) => {
      const cardText = r.card ? escapeHtml(r.card.value) : "a card";
      return r.revealing_player_id === playerId
        ? `<li>Turn ${r.turn}: you showed ${cardText} to player ${r.receiving_player_id}</li>`
        : `<li>Turn ${r.turn}: player ${r.revealing_player_id} showed you ${cardText}</li>`;
    })
    .join("");
  return `<ul>${items}</ul>`;
}

async function renderDecisionForm(host, playerId, pending) {
  if (pending.kind === "action") {
    await renderActionForm(host, playerId, pending);
  } else {
    renderRevealForm(host, playerId, pending);
  }
}

async function renderActionForm(host, playerId, pending) {
  const catalog = await getCardCatalog();
  const formId = `${panelElementId(playerId)}-action-form`;
  const optionsFor = (values) =>
    values.map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join("");

  host.innerHTML = `
    <h5>Your turn — turn ${pending.turn_number}</h5>
    <div class="error" id="${formId}-error" hidden></div>
    <form id="${formId}">
      <label><input type="radio" name="kind" value="suggestion" checked /> Suggestion</label>
      <label><input type="radio" name="kind" value="accusation" /> Accusation</label>
      <label><input type="radio" name="kind" value="pass" /> Pass</label>
      <div class="action-fields">
        <select name="person">${optionsFor(catalog.people)}</select>
        <select name="weapon">${optionsFor(catalog.weapons)}</select>
        <select name="room">${optionsFor(catalog.rooms)}</select>
      </div>
      <button type="submit">Submit</button>
    </form>
  `;

  const form = host.querySelector(`#${formId}`);
  const errorBox = host.querySelector(`#${formId}-error`);
  const fields = host.querySelector(".action-fields");

  const syncFields = () => {
    const kind = form.querySelector('input[name="kind"]:checked').value;
    fields.style.display = kind === "pass" ? "none" : "flex";
  };
  form.querySelectorAll('input[name="kind"]').forEach((r) => r.addEventListener("change", syncFields));
  syncFields();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    const kind = form.querySelector('input[name="kind"]:checked').value;
    const action =
      kind === "pass"
        ? { kind: "pass" }
        : {
            kind,
            person: form.person.value,
            weapon: form.weapon.value,
            room: form.room.value,
          };

    // A wrong accusation eliminates the seat for the rest of the game, so
    // it gets a confirm step the other two action kinds don't need.
    if (kind === "accusation") {
      const confirmed = window.confirm(
        `Accuse ${action.person} / ${action.weapon} / ${action.room}? A wrong accusation eliminates you.`
      );
      if (!confirmed) return;
    }

    const submitButton = form.querySelector('button[type="submit"]');
    submitButton.disabled = true;
    try {
      const response = await fetch(`${API}/games/${currentGameId}/players/${playerId}/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Player-Token": heldSeats.get(playerId) },
        body: JSON.stringify(action),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        errorBox.textContent = detail.detail ? String(detail.detail) : `HTTP ${response.status}`;
        errorBox.hidden = false;
        submitButton.disabled = false;
        return;
      }
      await refreshPlayerPanel(playerId);
    } catch (err) {
      errorBox.textContent = `Could not reach the service: ${err}`;
      errorBox.hidden = false;
      submitButton.disabled = false;
    }
  });
}

function renderRevealForm(host, playerId, pending) {
  const s = pending.suggestion;
  const formId = `${panelElementId(playerId)}-reveal-form`;
  const buttons = pending.offered
    .map(
      (card, i) =>
        `<button type="button" class="secondary" data-card-index="${i}">${escapeHtml(card.value)}</button>`
    )
    .join(" ");

  host.innerHTML = `
    <h5>Reveal needed — turn ${pending.turn_number}</h5>
    <p>Player ${pending.suggesting_player_id} suggested
      ${escapeHtml(s.person)} / ${escapeHtml(s.weapon)} / ${escapeHtml(s.room)}.
      Which do you show?</p>
    <div class="error" id="${formId}-error" hidden></div>
    <div class="reveal-buttons">${buttons}</div>
  `;

  const errorBox = host.querySelector(`#${formId}-error`);
  const buttonHost = host.querySelector(".reveal-buttons");
  buttonHost.querySelectorAll("[data-card-index]").forEach((button) => {
    button.addEventListener("click", async () => {
      const card = pending.offered[Number(button.dataset.cardIndex)];
      errorBox.hidden = true;
      buttonHost.querySelectorAll("button").forEach((b) => (b.disabled = true));
      try {
        const response = await fetch(`${API}/games/${currentGameId}/players/${playerId}/reveal`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Player-Token": heldSeats.get(playerId) },
          body: JSON.stringify({ card }),
        });
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          errorBox.textContent = detail.detail ? String(detail.detail) : `HTTP ${response.status}`;
          errorBox.hidden = false;
          buttonHost.querySelectorAll("button").forEach((b) => (b.disabled = false));
          return;
        }
        await refreshPlayerPanel(playerId);
      } catch (err) {
        errorBox.textContent = `Could not reach the service: ${err}`;
        errorBox.hidden = false;
        buttonHost.querySelectorAll("button").forEach((b) => (b.disabled = false));
      }
    });
  });
}

// --- Entry point ---------------------------------------------------------

els.nPlayers.addEventListener("change", renderSeatRows);
els.startButton.addEventListener("click", startGame);
els.newGameButton.addEventListener("click", () => {
  eventSource?.close();
  currentGameId = null;
  heldSeats = new Map();
  joinMode = false;
  joinedPlayerId = null;
  history.replaceState(null, "", location.pathname);
  els.gameScreen.hidden = true;
  els.setupScreen.hidden = false;
});

async function init() {
  const params = new URLSearchParams(location.search);
  const gameId = params.get("game");
  if (gameId) {
    await resumeGame(gameId, params.get("player"), params.get("token"));
  } else {
    await loadProviders();
  }
}

init();
