import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

type Settings = {
  hotkey: string;
  injectMode: "paste" | "type" | "clipboard";
  language: string;
  modelReady: boolean;
  modelPath: string;
};

type DownloadProgress = {
  file: string;
  bytes: number;
  total: number;
  fraction: number;
};

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;

const els = {
  hotkey: $<HTMLButtonElement>("hotkey"),
  hotkeyToken: $<HTMLInputElement>("hotkeyToken"),
  language: $<HTMLInputElement>("language"),
  injectMode: $<HTMLSelectElement>("injectMode"),
  save: $<HTMLButtonElement>("save"),
  status: $<HTMLElement>("status"),
  lastTranscript: $<HTMLElement>("lastTranscript"),
  modelStatus: $<HTMLElement>("modelStatus"),
  modelProgress: $<HTMLProgressElement>("modelProgress"),
  modelProgressCaption: $<HTMLElement>("modelProgressCaption"),
  modelPath: $<HTMLElement>("modelPath"),
};

function setStatus(msg: string, kind: "ok" | "err" | "" = "") {
  els.status.textContent = msg;
  els.status.className = `status ${kind}`;
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// --- HIG-style shortcut rendering ----------------------------------------
// Backend stores tokens like "CmdOrCtrl+Shift+Space". The settings field
// renders them as ⌘⇧Space glyphs and captures a new combo on click.

const MOD_GLYPH: Record<string, string> = {
  cmd: "⌘",
  command: "⌘",
  cmdorctrl: "⌘",
  commandorcontrol: "⌘",
  ctrl: "⌃",
  control: "⌃",
  alt: "⌥",
  option: "⌥",
  shift: "⇧",
  meta: "⌘",
  super: "⌘",
};

const KEY_GLYPH: Record<string, string> = {
  space: "Space",
  enter: "⏎",
  return: "⏎",
  tab: "⇥",
  esc: "⎋",
  escape: "⎋",
  backspace: "⌫",
  delete: "⌦",
  arrowup: "↑",
  arrowdown: "↓",
  arrowleft: "←",
  arrowright: "→",
};

function tokenToGlyphs(token: string): string {
  // Tokens come in deterministic order from the backend but we sort modifiers
  // canonically for display: ⌃⌥⇧⌘ then the key, matching macOS menus.
  const parts = token.split("+").map((s) => s.trim().toLowerCase());
  const mods: string[] = [];
  let key = "";
  for (const p of parts) {
    if (p in MOD_GLYPH) {
      mods.push(MOD_GLYPH[p]);
    } else {
      key = KEY_GLYPH[p] ?? p.toUpperCase();
    }
  }
  const order = ["⌃", "⌥", "⇧", "⌘"];
  const sorted = mods.sort((a, b) => order.indexOf(a) - order.indexOf(b));
  return `${sorted.join("")}${key}`;
}

/// Render the stored token in the visible button.
function renderHotkey(token: string) {
  els.hotkeyToken.value = token;
  els.hotkey.textContent = tokenToGlyphs(token);
}

/// Capture the next key combo and write it back to the token field.
function beginCapture() {
  const original = els.hotkeyToken.value;
  els.hotkey.textContent = "Press a key combination…";
  els.hotkey.classList.add("recording");

  const onKey = (e: KeyboardEvent) => {
    e.preventDefault();
    if (e.key === "Escape") {
      end(original);
      return;
    }
    // Need at least one non-modifier key to commit a combo.
    if (["Control", "Meta", "Shift", "Alt"].includes(e.key)) {
      return;
    }
    const parts: string[] = [];
    if (e.metaKey) parts.push("CmdOrCtrl");
    if (e.altKey) parts.push("Alt");
    if (e.shiftKey) parts.push("Shift");
    if (e.ctrlKey && !e.metaKey) parts.push("Ctrl");
    parts.push(keyEventToToken(e));
    end(parts.join("+"));
  };

  const end = (token: string) => {
    document.removeEventListener("keydown", onKey, true);
    els.hotkey.classList.remove("recording");
    renderHotkey(token);
  };

  document.addEventListener("keydown", onKey, true);
}

function keyEventToToken(e: KeyboardEvent): string {
  if (e.code === "Space") return "Space";
  if (e.code === "Enter") return "Enter";
  if (e.code === "Tab") return "Tab";
  if (e.code === "Escape") return "Escape";
  if (e.code === "Backspace") return "Backspace";
  if (e.code.startsWith("Key")) return e.code.slice(3);
  if (e.code.startsWith("Digit")) return e.code.slice(5);
  // Fallback: uppercase printable key.
  return e.key.toUpperCase();
}

async function loadSettings() {
  const s = await invoke<Settings>("get_settings");
  renderHotkey(s.hotkey);
  els.language.value = s.language;
  els.injectMode.value = s.injectMode;
  els.modelPath.textContent = s.modelPath;
  if (s.modelReady) {
    els.modelStatus.textContent = "Ready — recognizer loaded.";
    els.modelStatus.className = "hint ok";
    els.modelProgress.hidden = true;
    els.modelProgressCaption.hidden = true;
  } else {
    els.modelStatus.textContent = "Preparing model…";
    els.modelStatus.className = "hint";
  }
}

async function save() {
  setStatus("Saving…");
  try {
    await invoke("save_settings", {
      hotkey: els.hotkeyToken.value,
      injectMode: els.injectMode.value,
      language: els.language.value,
    });
    setStatus("Saved.", "ok");
    await loadSettings();
  } catch (e) {
    setStatus(`Save failed: ${String(e)}`, "err");
  }
}

els.save.addEventListener("click", () => void save());
els.hotkey.addEventListener("click", beginCapture);

listen<string>("transcript", (e) => {
  els.lastTranscript.textContent = e.payload;
  setStatus("Transcribed.", "ok");
});

listen<string>("dictation-error", (e) => {
  setStatus(`Error: ${e.payload}`, "err");
});

listen<string>("model-status", (e) => {
  els.modelStatus.textContent = e.payload;
  els.modelStatus.className = "hint";
  if (e.payload === "Ready.") {
    els.modelStatus.className = "hint ok";
    els.modelProgress.hidden = true;
    els.modelProgressCaption.hidden = true;
    void loadSettings();
  }
});

listen<DownloadProgress>("model-download", (e) => {
  const p = e.payload;
  els.modelProgress.hidden = false;
  els.modelProgressCaption.hidden = false;
  els.modelProgress.value = p.fraction;
  const pct = (p.fraction * 100).toFixed(0);
  els.modelStatus.textContent = `Downloading ${p.file}…`;
  els.modelProgressCaption.textContent = `${fmtBytes(p.bytes)} of ${fmtBytes(p.total)} (${pct}%)`;
});

void loadSettings();
