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
  jaegerLink: document.getElementById("jaeger-link"),
  newGameButton: document.getElementById("new-game-button"),
  seatCards: document.getElementById("seat-cards"),
  feed: document.getElementById("feed"),
  showReveals: document.getElementById("show-reveals"),
};

let providers = [];
let eventSource = null;
let seatConfigs = [];
const eliminated = new Set();

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
      option.textContent = p.name === "random" ? "Random (no LLM)" : p.name;
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

    const syncModelVisibility = () => {
      const isRandom = select.value === "random";
      for (const field of [modelInput, thinkingInput, scratchpadInput]) {
        field.disabled = isRandom;
      }
      for (const field of [modelInput, thinkingInput, scratchpadLabel]) {
        field.style.visibility = isRandom ? "hidden" : "visible";
      }
    };
    select.addEventListener("change", syncModelVisibility);
    syncModelVisibility();

    row.append(label, select, modelInput, thinkingInput, scratchpadLabel);
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
    const model = row.querySelector('[data-role="model"]').value.trim();
    if (agentType === "random") {
      return { agent_type: "random" };
    }
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

  const { game_id, run_id, stream_url } = await response.json();
  enterGameScreen(game_id, run_id, stream_url);
}

function showSetupError(message) {
  els.setupError.textContent = message;
  els.setupError.hidden = false;
}

function enterGameScreen(gameId, runId, streamUrl) {
  eliminated.clear();
  els.setupScreen.hidden = true;
  els.gameScreen.hidden = false;
  els.gameId.textContent = gameId;
  els.gameStatus.textContent = "running";
  els.turnNumber.textContent = "0";
  els.currentPlayer.textContent = "0";
  els.feed.innerHTML = "";
  els.jaegerLink.href =
    `${JAEGER_BASE}/search?service=clue-game-logic&tags=${encodeURIComponent(
      JSON.stringify({ "clue.run_id": runId })
    )}`;

  renderSeatCards();

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
    title.textContent = `Player ${i}`;
    const config = document.createElement("div");
    config.className = "seat-config";
    config.textContent =
      seat.agent_type === "random" ? "Random agent" : `${seat.provider} / ${seat.model}`;
    card.append(title, config);
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

els.nPlayers.addEventListener("change", renderSeatRows);
els.startButton.addEventListener("click", startGame);
els.newGameButton.addEventListener("click", () => {
  eventSource?.close();
  els.gameScreen.hidden = true;
  els.setupScreen.hidden = false;
});

loadProviders();
