/* neuro_int UI: WS client, gesture map editor, cursor prefs.
   The skeleton is drawn by the brain on the video frames themselves. */

"use strict";

const POSE_META = {
  index_up: { emoji: "☝", label: "1 finger up" },
  two_up: { emoji: "✌", label: "2 fingers up" },
  open_palm: { emoji: "🖐", label: "open palm" },
  fist: { emoji: "✊", label: "fist" },
  pinch: { emoji: "🤏", label: "pinch (cursor)" },
};
const MOTION_META = {
  right: "→", left: "←", up: "↑", down: "↓",
  hold: "◎ hold",
};

const $ = (id) => document.getElementById(id);
const video = $("video");

let app = {
  ws: null,
  actions: [],
  rules: [],
  customActions: [],
  recordingKeys: null,
};

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => console.log("UI connected");
  ws.onclose = () => {
    console.warn("UI disconnected, retrying");
    setTimeout(connect, 1500);
  };
  ws.onmessage = (event) => handle(JSON.parse(event.data));
  app.ws = ws;
}

function send(payload) {
  if (app.ws && app.ws.readyState === WebSocket.OPEN) {
    app.ws.send(JSON.stringify(payload));
  }
}

function handle(msg) {
  switch (msg.type) {
    case "state":
      app.actions = msg.actions;
      app.rules = msg.rules;
      app.customActions = msg.custom_actions;
      renderRules();
      renderCustoms();
      renderActions();
      fillForm();
      setRecordingState(msg.recording);
      if (msg.cursor_prefs) renderCursorPrefs(msg.cursor_prefs);
      if (msg.status) applyStatus(msg.status);
      break;
    case "landmarks":
      updatePoseBadge(msg);
      break;
    case "status":
      applyStatus(msg);
      break;
    case "landmarks_fsm": // future-proof: no-op
      break;
    case "cursor_prefs":
      renderCursorPrefs(msg);
      toast(`Cursor: ${msg.mode}, ${Number(msg.sensitivity).toFixed(1)}×`, "ok");
      break;
    case "recording":
      setRecordingState(msg.active);
      if (msg.error) toast(msg.error, "error");
      if (msg.keys) {
        app.recordingKeys = msg.keys;
        toast(`Recorded: ${msg.keys.join(" + ")}`, "ok");
        renderCustoms();
      }
      break;
    case "notify":
      toast(msg.text, msg.level);
      break;
    case "custom_added":
      toast(`${msg.action_id} created — pick it in the Action list`, "ok");
      break;
  }
}

function applyStatus(msg) {
  $("m-fps").textContent = msg.fps_inference.toFixed(1);
  $("m-hands").textContent = String(msg.hands_seen);
  $("badge-paused").classList.toggle("hidden", !msg.paused);
  $("pause-btn").textContent = msg.paused ? "Resume" : "Pause";
}

/* ---------- skeleton: drawn by the brain on the detected frame ----------
   No browser-side overlay: canvas coordinates detach from a moving hand
   (round-trip latency). The MJPEG stream already contains the skeleton. */

function updatePoseBadge(msg) {
  const el = $("pose-badge");
  const meta = POSE_META[msg.pose];
  el.textContent = msg.points.length ? (meta ? `${meta.emoji} ${msg.pose}` : msg.pose) : "no hand";
  el.classList.toggle("pinch", msg.pose === "pinch");

  const fsmEl = $("m-state");
  fsmEl.textContent = msg.fsm_state;
  fsmEl.style.color = msg.cursor_active ? "#8b9bf4" : "";

  const poseEl = $("m-pose");
  poseEl.textContent = msg.points.length && meta ? `${meta.emoji} ${meta.label}` : "—";

  const cursor = msg.cursor_active || msg.fsm_state === "cursor" || msg.fsm_state === "cursor_pending";
  $("cursor-banner").classList.toggle("hidden", !cursor);
  $("badge-cursor").classList.toggle("hidden", !cursor);
  document.querySelector(".video-wrap").classList.toggle("cursor-active", cursor);
}

/* ---------- cursor prefs ---------- */

function renderCursorPrefs(prefs) {
  const slider = $("sens-slider");
  slider.value = prefs.sensitivity;
  $("sens-value").textContent = `${Number(prefs.sensitivity).toFixed(1)}×`;
  document.querySelectorAll("#mode-switch button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === prefs.mode);
  });
}

$("sens-slider").addEventListener("input", (event) => {
  $("sens-value").textContent = `${Number(event.target.value).toFixed(1)}×`;
  sendCursorPrefsDebounced();
});
$("mode-switch").addEventListener("click", (event) => {
  if (event.target.dataset.mode) sendCursorPrefsDebounced(true);
});
document.querySelectorAll(".presets button").forEach((btn) => {
  btn.addEventListener("click", () => {
    $("sens-slider").value = btn.dataset.sens;
    $("sens-value").textContent = `${Number(btn.dataset.sens).toFixed(1)}×`;
    send({ type: "set_cursor", sensitivity: Number(btn.dataset.sens), mode: activeMode() });
  });
});

function activeMode() {
  const btn = document.querySelector("#mode-switch button.active");
  return btn ? btn.dataset.mode : "relative";
}

let _cursorTimer = null;
function sendCursorPrefsDebounced(immediate = false) {
  if (_cursorTimer) clearTimeout(_cursorTimer);
  _cursorTimer = setTimeout(
    () => send({
      type: "set_cursor",
      sensitivity: Number($("sens-slider").value),
      mode: activeMode(),
    }),
    immediate ? 0 : 250
  );
}

/* ---------- gesture map ---------- */

function keysChips(actionId) {
  const action = app.actions.find((a) => a.action === actionId);
  if (!action) return "";
  return action.keys
    .map((k) => `<span class="chip ${action.kind}">${k}</span>`)
    .join("");
}

function renderRules() {
  const list = $("rules-list");
  if (!app.rules.length) {
    list.innerHTML = `<div class="empty">No rules yet — add one below</div>`;
    return;
  }
  list.innerHTML = app.rules
    .map((r, i) => {
      const pose = POSE_META[r.pose] || { emoji: "?", label: r.pose };
      const motionText = r.hold ? "hold gesture" : `swipe ${r.direction}`;
      const motionIcon = r.hold ? MOTION_META.hold : MOTION_META[r.direction] || r.direction;
      const custom = (r.action || "").startsWith("custom_");
      return `
        <div class="rule ${custom ? "is-custom" : ""}">
          <span class="emoji" title="${pose.label}">${pose.emoji}</span>
          <span class="arrow" title="${motionText}">${motionIcon}</span>
          <span class="action-name">${r.action}<span class="motion-label">${motionText}</span></span>
          <span class="keys">${keysChips(r.action)}</span>
          <button class="delete" data-index="${i}" title="remove rule">&times;</button>
        </div>`;
    })
    .join("");
  list.querySelectorAll(".delete").forEach((btn) => {
    btn.onclick = () => {
      const r = app.rules[Number(btn.dataset.index)];
      send({ type: "remove_rule", pose: r.pose, motion: r.hold ? "hold" : r.direction, action: r.action });
    };
  });
}

function fillForm() {
  const poseSel = $("pose-select");
  const motionSel = $("motion-select");
  const actionSel = $("action-select");
  const keepPose = poseSel.value, keepMotion = motionSel.value, keepAction = actionSel.value;

  poseSel.innerHTML = Object.entries(POSE_META)
    .map(([id, m]) => `<option value="${id}">${m.emoji} ${id}</option>`)
    .join("");
  motionSel.innerHTML =
    ["left", "right", "up", "down"].map((d) => `<option value="${d}">${MOTION_META[d]} swipe</option>`).join("") +
    `<option value="hold">${MOTION_META.hold}</option>`;
  actionSel.innerHTML = app.actions
    .map((a) => `<option value="${a.action}">${a.action}${a.keys.length ? ` (${a.keys.join("+")})` : ""}</option>`)
    .join("");

  if (keepPose) poseSel.value = keepPose;
  if (keepMotion) motionSel.value = keepMotion;
  if (keepAction) actionSel.value = keepAction;
}

/* ---------- custom combos + actions ---------- */

function renderCustoms() {
  const list = $("custom-list");
  if (!app.customActions.length) {
    list.innerHTML = `<div class="empty">No custom combos yet</div>`;
    return;
  }
  list.innerHTML = app.customActions
    .map(
      (c) => `
      <div class="item">
        <span class="id">${c.id}</span>
        <span class="keys">${(c.keys || []).map((k) => `<span class="chip">${k}</span>`).join("")}</span>
        <button class="delete" data-id="${c.id}" title="delete combo and its rules">&times;</button>
      </div>`
    )
    .join("");
  list.querySelectorAll(".delete").forEach((btn) => {
    btn.onclick = () => send({ type: "delete_custom", action_id: btn.dataset.id });
  });
}

function renderActions() {
  const list = $("actions-list");
  list.innerHTML = app.actions
    .map((a) => {
      const chips = a.keys.length
        ? a.keys.map((k) => `<span class="chip ${a.kind}">${k}</span>`).join("")
        : `<span class="chip ${a.kind}">${a.kind}</span>`;
      return `<div class="item"><span class="id">${a.action}</span><span class="desc">${a.description || ""}</span><span class="keys">${chips}</span></div>`;
    })
    .join("");
}

function setRecordingState(active) {
  const btn = $("record-btn");
  const status = $("record-status");
  btn.classList.toggle("recording", active);
  btn.innerHTML = active ? "● Listening… (any combo)" : "● Record combo";
  status.textContent = active ? "press the shortcut now" : "";
}

/* ---------- toasts ---------- */

function toast(text, level = "info") {
  const el = document.createElement("div");
  el.className = `toast ${level}`;
  el.textContent = text;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

/* ---------- wiring ---------- */

$("add-rule-form").onsubmit = (event) => {
  event.preventDefault();
  const pose = $("pose-select").value;
  const motion = $("motion-select").value;
  const action = $("action-select").value;
  if (!action) {
    toast("pick an action first", "error");
    return;
  }
  send({ type: "add_rule", pose, direction: motion === "hold" ? "" : motion, hold: motion === "hold", action });
};

$("record-btn").onclick = () => {
  if ($("record-btn").classList.contains("recording")) {
    send({ type: "record_cancel" });
  } else {
    send({ type: "record_start" });
  }
};

$("pause-btn").onclick = () => send({ type: "pause_toggle" });

connect();
