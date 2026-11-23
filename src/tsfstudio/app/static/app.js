/* Forecast Studio interface: dependency-free (vanilla JS). Data is always written with textContent (innerHTML is never used). */
(() => {
"use strict";

// ================================================================================================ helpers
const TOKEN = document.querySelector('meta[name="tsf-token"]').content;
// replaceChildren/append: filter out null/false so they are not written to the DOM as the text "null"; flatten arrays.
for (const name of ["replaceChildren", "append"]) {
  const orig = Element.prototype[name];
  Element.prototype[name] = function (...kids) { return orig.apply(this, kids.flat(Infinity).filter(k => k != null && k !== false)); };
}
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return e;
}
function svg(tag, attrs, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    // Theme colours: SVG presentation attributes cannot use var(), the style property can, and it follows a theme change live.
    if ((k === "fill" || k === "stroke") && typeof v === "string" && v.startsWith("var(")) e.style.setProperty(k, v);
    else e.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
}

const nfSig = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 4 });
const nf0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const nf2 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const fmt = v => (v == null || Number.isNaN(v) ? "—" : nfSig.format(v));
const fint = v => (v == null ? "—" : nf0.format(v));
const fpct = (v, d) => (v == null ? "—" : (d === 0 ? nf0 : d === 2 ? nf2 : nf1).format(v * 100) + "%");
// Metric-aware numbers (UI review M1): a fixed number of decimals per measure so columns line up, and units where they apply.
const nf3fix = new Intl.NumberFormat("en-US", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
const nf1fix = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const nf2fix = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const PCT_METRICS = new Set(["WAPE", "MAPE", "PBias"]);
const SCALE_FREE = new Set(["SMAPE", "mSMAPE", "MASE", "RMSSE", "nRMSE_Mean", "nRMSE_Range", "MAAPE", "RMSLE"]);
function fmtM(k, v) {
  if (v == null || Number.isNaN(v)) return "—";
  if (PCT_METRICS.has(k)) return nf1fix.format(v) + "%";
  if (SCALE_FREE.has(k)) return nf3fix.format(v);
  return Math.abs(v) >= 1000 ? nf0.format(v) : nf2fix.format(v);      // data-unit measures (MAE, RMSE, MSE, MedAE, ME)
}
const FREQ = { "4_seconds": "4-second", minutely: "minutely", "10_minutes": "10-minute", "15_minutes": "15-minute", half_hourly: "half-hourly",
  hourly: "hourly", daily: "daily", weekly: "weekly", monthly: "monthly", quarterly: "quarterly", yearly: "yearly" };
const frq = f => FREQ[f] || f || "—";
const ENGINE_NAME = { python: "Python", r: "R", cpp: "C++" };
const ENGINES = ["python", "r", "cpp"];
const ENGINE_SHORT = { python: "Py", r: "R", cpp: "C++" };
const STAGE = { loading: "Reading data", prepare: "Preparing data", forecasting: "Generating forecasts", fitting: "Fitting models",
  metrics: "Computing error measures", writing: "Writing files", done: "Done", finished: "Forecasts ready", starting: "Starting" };

async function api(path, body) {
  const o = { method: body !== undefined ? "POST" : "GET", headers: { "X-TSF-Token": TOKEN } };
  if (body !== undefined) { o.headers["Content-Type"] = "application/json"; o.body = JSON.stringify(body); }
  const r = await fetch("/api/" + path, o);
  let j = null;
  try { j = await r.json(); } catch (_) { /* body is not JSON */ }
  if (!r.ok) throw new Error((j && j.error) || "Request failed (HTTP " + r.status + ")");
  return j;
}
const fileUrl = (runId, key) => `/api/runs/${encodeURIComponent(runId)}/file/${encodeURIComponent(key)}?t=${encodeURIComponent(TOKEN)}`;
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem("tsfs." + k)); } catch (_) { return null; } },
  set(k, v) { try { localStorage.setItem("tsfs." + k, JSON.stringify(v)); } catch (_) { /* storage may be disabled */ } },
};
let toastTimer = null;
function toast(msg, bad) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast" + (bad ? " bad" : ""); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, bad ? 7000 : 3500);
}
function bindNoteSheet() {
  const btn = $("#margin-toggle"), saved = store.get("noteOpen");
  const apply = open => {
    document.body.classList.toggle("note-collapsed", !open);
    $("#margin").classList.toggle("collapsed", !open);
    btn.textContent = open ? "Collapse note" : "Open note"; btn.setAttribute("aria-expanded", String(open));
  };
  let open = saved == null ? window.innerWidth > 760 : !!saved;
  apply(open);
  btn.addEventListener("click", () => { open = !open; store.set("noteOpen", open); apply(open); });
}
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

// ================================================================================================ durum
const S = {
  sys: null, models: null, datasets: null,
  tab: "all", filter: "",
  detail: null, detailErr: null,
  sel: { dataset: null, method: null, engine: "python" },
  settings: { horizon: "", lag: "", integer: "auto", missing: "error", max_series: "", seed: "1", workers: "1", metrics: null },
  params: {},
  est: null, estBusy: false, estSeq: 0, confirmLong: false,
  job: null, pollTimer: null,
  run: null,            // { id, detail, series, errors, hist, ... }
  view: "work",
  history: null, compare: new Set(), cmpMetric: "MASE", cmpStat: "mean",
  autoOff: false,
};
const model = id => S.models.models.find(m => m.id === id);
const curModel = () => (S.sel.method ? model(S.sel.method) : null);
const entry = id => (S.datasets ? S.datasets.items.find(d => d.id === id) : null);
const BIG = 40e6;

// ================================================================================================ kenar notu
const Note = {
  node: null, base: null, hovering: false,
  set(n) { this.node.replaceChildren(n); },
  restore() { this.hovering = false; this.set(this.base ? this.base() : introNote()); },
  refresh() { if (!this.hovering) this.restore(); },
};
// Roving tabindex for a radiogroup (UI review H3): the whole list is ONE Tab stop; Up/Down/Home/End move focus between rows,
// Enter or Space selects. (Selection does not follow focus, because selecting a dataset loads it.)
function roving(group, rowSel) {
  const rows = () => $$(rowSel, group).filter(r => r.getAttribute("aria-disabled") !== "true");
  const all = rows();
  if (!all.length) return group;
  const cur = all.find(r => r.getAttribute("aria-checked") === "true") || all[0];
  all.forEach(r => r.setAttribute("tabindex", r === cur ? "0" : "-1"));
  group.addEventListener("keydown", ev => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(ev.key)) return;
    const list = rows(), i = list.indexOf(document.activeElement);
    if (i < 0) return;
    ev.preventDefault();
    const j = ev.key === "Home" ? 0 : ev.key === "End" ? list.length - 1 : Math.max(0, Math.min(list.length - 1, i + (ev.key === "ArrowDown" ? 1 : -1)));
    list[i].setAttribute("tabindex", "-1"); list[j].setAttribute("tabindex", "0"); list[j].focus();
  });
  return group;
}
// Keep keyboard focus on the same row when a list is redrawn (otherwise it falls back to <body>).
function keptFocus(sel) { const a = document.activeElement; return a && a.matches && a.matches(sel) ? a.getAttribute("data-id") : null; }
function restoreFocus(sel, id) { if (!id) return; const r = $$(sel).find(x => x.getAttribute("data-id") === id); if (r) { r.setAttribute("tabindex", "0"); r.focus({ preventScroll: true }); } }
function bindNote(elm, build) {
  const on = () => { Note.hovering = true; Note.set(build()); };
  const off = () => Note.restore();
  elm.addEventListener("mouseenter", on);
  elm.addEventListener("focusin", on);
  elm.addEventListener("mouseleave", off);
  elm.addEventListener("focusout", off);
  return elm;
}
const runIn = (head, text, cls) => el("p", { class: cls || "" }, el("b", { class: "run-in" }, head + " "), text);
function refList(refs) {
  if (!refs || !refs.length) return null;
  return el("ul", { class: "refs" }, refs.map(r => el("li", {},
    r.url ? el("a", { href: r.url, target: "_blank", rel: "noopener noreferrer" }, r.text) : r.text,
    el("span", { class: "tag" + (r.verified ? " ok" : "") }, r.verified ? "citation verified" : "citation to be verified"))));
}
function introNote() {
  return el("div", {},
    el("h3", { text: "Margin note" }),
    el("p", { class: "lead", text: "Hover over (or keyboard-focus) a model, setting or measure and its explanation appears here." }),
    el("p", { text: "Work through the steps: choose the data, choose the model, change settings if needed, check the pre-run summary and start the run. Every run is saved together with its settings so it can be reproduced." }),
    el("p", { class: "hint", text: "All model explanations and references appear in this panel; references whose details have not been verified are flagged." }));
}
function modelNote(m, engine) {
  const e = engine || S.sel.engine;
  return el("div", {},
    el("h3", { text: m.name }),
    el("p", { class: "lead", text: m.summary }),
    runIn("How it works.", m.how),
    runIn("When to use.", m.use),
    runIn("Caution.", m.caution, "caution"),
    el("div", { class: "kv" },
      ENGINES.map(k => m.engines[k] && m.engines[k].supported ? el("div", {}, el("b", { text: ENGINE_NAME[k] + " engine: " }),
        (m.engines[k].available ? "" : "not available on this machine. ") + (m.engines[k].note || "")) : null),
      el("div", {}, el("b", { text: "Run time: " }), m.cost)),
    refList(m.refs));
}
function paramNote(p, m) {
  const rng = (p.min != null || p.max != null) ? `${p.min != null ? p.min : "…"} – ${p.max != null ? p.max : "…"}` : null;
  const def = p.default != null ? String(p.default) : (p.placeholder ? p.placeholder + " (library default)" : "empty");
  return el("div", {},
    el("h3", { text: p.label }), el("p", { class: "lead", text: m.name }),
    el("p", { text: p.help }),
    el("div", { class: "kv" }, el("div", {}, el("b", { text: "Default: " }), def), rng ? el("div", {}, el("b", { text: "Range: " }), rng) : null,
      el("div", {}, el("b", { text: "Applies to: " }), p.engines.map(x => ENGINE_NAME[x]).join(", "))));
}
function metricNote(k) {
  const mi = S.models.metrics[k];
  if (!mi) return introNote();
  return el("div", {}, el("h3", { text: mi.name }), el("p", { class: "lead", text: "Unit: " + mi.unit }), el("p", { text: mi.text }),
    runIn("Caution.", mi.caution, "caution"),
    refList(S.models.metric_refs));
}
const SETTING_NOTES = {
  horizon: ["Forecast horizon", "The LAST this-many observations of every series are held out as the test set; the model is trained on the rest and forecasts these steps. If left empty, the horizon from the dataset catalogue (or the file header) is used."],
  lag: ["Lag", "Global models learn the next value from the last 'lag' observations. For catalogue datasets the value comes from the reference study (or is marked 'suggested', which is not an evidence-based choice). Not used by local models."],
  max_series: ["Number of series", "Runs only the first N series (for a quick trial). If left empty, all series are run. A small trial before a long run is recommended."],
  integer: ["Round to integers", "Rounds forecasts to the nearest integer. For count data (car_parts, M5, ...) the catalogue turns this ON, as in the reference study. Intermittent-demand methods forecast a flat demand rate (often below 0.5); rounding turns it into 0 and every forecast becomes 0."],
  missing: ["Missing values", "What to do about missing values in the training part. By default the run does not start, because the choice affects the results and we do not change data silently."],
  seed: ["Seed", "Reproducibility for methods that involve randomness. Most classical methods are deterministic; the seed is still written to the manifest."],
  workers: ["Workers", "In the Python engine, series are run in this many parallel processes. The C++ engine uses this many threads (1 = all CPU cores). The R engine runs on a single core on Windows. With 8 GB of memory we suggest 1-2 for Python."],
  metrics: ["Measures", "The error measures to compute. Hover over a measure name to see its definition and what to watch out for."],
};
function settingNote(k, extra) {
  const [t, x] = SETTING_NOTES[k];
  return () => el("div", {}, el("h3", { text: t }), el("p", { text: x }), extra ? extra() : null);
}

// ================================================================================================ start-up
async function init() {
  Note.node = $("#margin-body");
  bindNoteSheet();
  Note.base = null; Note.restore();
  try {
    const [sys, models, ds] = await Promise.all([api("system"), api("models"), api("datasets")]);
    S.sys = sys; S.models = models; S.datasets = ds;
  } catch (e) {
    $("#sheet").replaceChildren(el("p", { class: "note err", text: "Could not reach the server: " + e.message + ". Reload the page; if that does not help, restart the application." }));
    return;
  }
  S.settings.metrics = new Set(S.models.metric_defaults);
  // A run started before a reload (or in another tab) is still going on the server: pick it up again (UI review H2).
  if (S.sys.job && S.sys.job.status === "running") S.job = S.sys.job;
  renderStatus(); renderVeri(); renderModel(); renderAyarlar(); renderKosu(); renderSonuc();
  if (S.job && S.job.status === "running") pollJob();
  bindRail();
  let lastW = window.innerWidth;
  window.addEventListener("resize", debounce(() => { if (window.innerWidth !== lastW) { lastW = window.innerWidth; if (S.run && S.run.series) renderChart(false); if (S.run && S.run.hist) renderHist(); } }, 250));
  const saved = store.get("sel");
  if (saved) {
    if (saved.method && model(saved.method)) { S.sel.engine = saved.engine || "python"; selectModel(saved.method, true); }
    if (saved.dataset && entry(saved.dataset) && ["ok", "user"].includes(entry(saved.dataset).status)) selectDataset(saved.dataset, true);
  }
  refreshHistoryCount();
}

// ================================================================================================ rail, status bar
function renderStatus() {
  const sys = S.sys;
  const eng = k => {
    const e = sys.engines[k];
    return bindNote(el("span", { class: "st", tabindex: 0 }, el("i", { class: "dot" + (e.available ? "" : " off") }),
      `${ENGINE_NAME[k]} engine ${e.available ? "ready" : "unavailable"}`), () => el("div", {}, el("h3", { text: ENGINE_NAME[k] + " engine" }),
      el("p", { text: e.message })));
  };
  const mem = sys.free_memory_gb;
  $("#status").replaceChildren(...ENGINES.filter(k => sys.engines[k]).map(eng),
    el("span", { class: "st" }, el("i", { class: "dot" + (mem != null && mem < 1.5 ? " warn" : "") }), mem == null ? "Free memory unknown" : `Free memory ${nf1.format(mem)} GB`),
    el("button", { type: "button", class: "linkish", onclick: async () => {
      try { S.sys = await api("system?refresh=1"); S.models = await api("models"); renderStatus(); renderModel(); renderKosuDebounced(); toast("Status refreshed."); } catch (e) { toast(e.message, true); } } }, "Refresh"));
}
function bindRail() {
  $("#view-history").addEventListener("click", () => setView("history"));
  $("#view-features").addEventListener("click", () => setView("features"));
  $("#view-work").addEventListener("click", () => setView("work"));
  let armed = false;
  $("#quit").addEventListener("click", async ev => {
    if (!armed) { armed = true; ev.target.textContent = "Sure? Quit"; setTimeout(() => { armed = false; ev.target.textContent = "Quit application"; }, 4000); return; }
    try { await api("shutdown", {}); } catch (_) { /* the connection drops while the server shuts down */ }
    document.body.replaceChildren(el("p", { class: "note", style: "margin:40px auto;max-width:520px", text: "Forecast Studio has been closed. You can close this tab." }));
  });
  const secs = $$(".sec");
  const io = new IntersectionObserver(es => {
    for (const e of es) if (e.isIntersecting) { $$(".steps a").forEach(a => a.classList.toggle("here", a.dataset.step === e.target.id)); }
  }, { rootMargin: "-20% 0px -65% 0px" });
  secs.forEach(s => io.observe(s));
}
function updateRail() {
  const d = S.sel.dataset ? entry(S.sel.dataset) : null, m = curModel();
  const done = { veri: !!S.sel.dataset, model: !!m, ayarlar: !!(S.sel.dataset && m), kosu: !!(S.run), sonuc: !!S.run };
  $$(".steps a").forEach(a => a.classList.toggle("done", !!done[a.dataset.step]));
  const p = $("#picked"); p.replaceChildren();
  if (d) p.append(el("div", {}, el("span", { text: "Data" }), el("b", { text: d.id.replace(/^upload:/, "") })));
  if (m) p.append(el("div", {}, el("span", { text: "Model" }), el("b", { text: m.id }), el("span", { text: ENGINE_NAME[S.sel.engine] + " engine" })));
}
function setView(v) {
  S.view = v;
  for (const k of ["work", "history", "features"]) { $(`#view-${k}-body`).hidden = v !== k; $(`#view-${k}`).hidden = v === k; }
  if (v === "history") renderHistory();
  if (v === "features") renderFeatures();
  window.scrollTo({ top: 0 });
}
async function refreshHistoryCount() {
  try { const r = await api("runs"); S.history = r; $("#history-count").textContent = r.runs.length ? `(${r.runs.length})` : ""; } catch (_) { /* not important */ }
}

// ================================================================================================ section heading
function secHead(id, n, title, lead) {
  return [el("h2", { id: "h-" + id }, el("span", { class: "n", text: String(n) }), title), el("p", { class: "lead", text: lead })];
}

// ================================================================================================ 1 DATA
function renderVeri() {
  const sec = $("#veri");
  const tabs = [["all", "All"], ["main", "Main set"], ["extra", "Extra"], ["m5", "M5"], ["user", "My uploads"]];
  const miss = S.datasets.missing;
  sec.replaceChildren(
    ...secHead("veri", 1, "Data", "Choose the dataset to work on, or upload your own."),
    el("div", { class: "toolbar" },
      el("input", { type: "search", class: "search", placeholder: "Search: name or domain", "aria-label": "Search datasets", value: S.filter,
        oninput: ev => { S.filter = ev.target.value; renderDatasetList(); } }),
      el("div", { class: "tabs", role: "tablist" }, tabs.map(([k, l]) => el("button", { type: "button", role: "tab", "aria-selected": String(S.tab === k),
        onclick: () => { S.tab = k; renderVeri(); } }, l))),
      miss ? el("span", { class: "hint", text: `${miss} files are not in the data folder.` }) : null),
    el("div", { id: "dlist" }),
    el("div", { id: "ddetail" }),
    uploadBlock());
  renderDatasetList(); renderDetail();
}
function renderDatasetList() {
  const q = S.filter.trim().toLowerCase();
  const rank = d => (d.status === "user" ? 0 : d.status === "ok" ? 1 : 2);
  const items = S.datasets.items.filter(d => (S.tab === "all" || d.bundle === S.tab) && (!q || (d.id + " " + d.title).toLowerCase().includes(q)))
    .sort((a, b) => rank(a) - rank(b));
  const list = el("div", { class: "list", role: "radiogroup", "aria-label": "Datasets" },
    el("div", { class: "drow head" }, el("span", { text: "Name" }), el("span", { class: "d", text: "Domain" }), el("span", { class: "m", text: "Frequency" }), el("span", { class: "m", text: "Series" }), el("span", { class: "m", text: "Status" })),
    items.length ? items.map(d => {
      const ok = d.status === "ok" || d.status === "user";
      const on = S.sel.dataset === d.id;
      const row = el("div", { class: "drow" + (ok ? "" : " dim"), role: "radio", "aria-checked": String(on), tabindex: ok ? 0 : -1, "aria-disabled": ok ? null : "true", "data-id": d.id,
        onclick: () => ok && selectDataset(d.id), onkeydown: ev => { if (ok && (ev.key === "Enter" || ev.key === " ")) { ev.preventDefault(); selectDataset(d.id); } } },
        el("span", { class: "t" }, on ? el("span", { class: "hl", text: d.id.replace(/^upload:/, "") }) : d.id.replace(/^upload:/, "")),
        el("span", { class: "d", text: d.title || "" }),
        el("span", { class: "m", text: d.frequency ? frq(d.frequency) : "—" }),
        el("span", { class: "m", text: d.n_series != null ? fint(d.n_series) : "—" }),
        el("span", { class: "m" + (ok ? "" : " pen"), text: d.status === "ok" ? "ready" : d.status === "user" ? "uploaded" : d.status === "missing" ? "file missing" : "size mismatch" }));
      return bindNote(row, () => datasetNote(d));
    }) : el("p", { class: "empty", text: "No matching dataset." }));
  const focusId = keptFocus("#dlist .drow");
  $("#dlist").replaceChildren(roving(list, ".drow[role=radio]"));
  restoreFocus("#dlist .drow", focusId);
}
function datasetNote(d) {
  const nodes = [el("h3", { text: d.id.replace(/^upload:/, "") }), el("p", { class: "lead", text: d.title || "" })];
  nodes.push(el("div", { class: "kv" }, el("div", {}, el("b", { text: "File: " }), d.file), el("div", {}, el("b", { text: "Size: " }), nf1.format(d.bytes / 1e6) + " MB")));
  if (d.status === "missing") nodes.push(runIn("File missing.", "This dataset was not found in the data folder; place it there under its catalogue name and reload the page.", "caution"));
  if (d.kind === "catalog") {
    nodes.push(el("p", {}, d.lag != null ? `Lag: ${d.lag} (${d.lag_origin === "study" ? "from the reference study" : "suggested, not in the reference study"}). ` : "No lag suggestion in the catalogue. ",
      d.integer_conversion ? "Rounding to integers is ON in the reference study." : "Rounding to integers is off."));
    if (d.bytes > BIG) nodes.push(runIn("Large file.", "The pre-run check reads the whole file; we only do this when you ask (Run step).", "caution"));
  }
  return el("div", {}, nodes);
}
async function selectDataset(id, silent) {
  S.sel.dataset = id; S.detail = null; S.detailErr = null; S.est = null; S.autoOff = false;
  store.set("sel", S.sel);
  renderDatasetList(); renderDetail(); updateRail(); refreshStale();
  try {
    S.detail = await api("dataset/" + encodeURIComponent(id));
  } catch (e) { S.detailErr = e.message; if (!silent) toast(e.message, true); }
  if (S.sel.dataset !== id) return;
  applyDatasetDefaults();
  renderDetail(); renderAyarlar(); renderKosu(); updateRail();
  if (!silent) $("#ddetail").scrollIntoView({ block: "nearest", behavior: "smooth" });
}
function applyDatasetDefaults() {
  const d = S.detail, m = curModel();
  if (!d || !m) return;
  if (m.task === "global" && !S.settings.lag && d.lag == null && d.lag_suggestion) S.settings.lag = String(d.lag_suggestion);
  const inter = m.group === "aralikli";
  if (inter && d.integer_conversion && S.settings.integer === "auto") { S.settings.integer = "off"; S.autoOff = true; }
}
const PROFILE = [["smooth", "smooth", "var(--ink)"], ["erratic", "erratic", "#6f93c9"], ["intermittent", "intermittent", "var(--hl)"], ["lumpy", "lumpy", "var(--pen)"], ["all_zero", "all zero", "var(--rule)"]];
function renderDetail() {
  const host = $("#ddetail");
  if (!S.sel.dataset) { host.replaceChildren(el("p", { class: "empty", text: "Choose a dataset; its summary and sample series appear here." })); return; }
  if (S.detailErr) { host.replaceChildren(el("p", { class: "note err", text: S.detailErr })); return; }
  if (!S.detail) { host.replaceChildren(el("p", { class: "empty", text: "Reading data summary…" })); return; }
  const d = S.detail, st = d.stats, pv = d.preview;
  const lens = st ? (st.len_min === st.len_max ? fint(st.len_min) : `${fint(st.len_min)}–${fint(st.len_max)}`) : (pv.len_min === pv.len_max ? fint(pv.len_min) : `${fint(pv.len_min)}–${fint(pv.len_max)}`);
  const fact = (k, v, sm) => el("div", {}, el("dt", { text: k }), el("dd", {}, v, sm ? el("small", { text: " " + sm }) : null));
  const prof = pv.profile;
  host.replaceChildren(
    el("dl", { class: "facts" },
      fact("Frequency", frq(d.frequency)), fact("Forecast horizon", d.horizon != null ? fint(d.horizon) : "none"),
      fact("Series", st ? fint(st.n_series) : "?", st ? null : "(no summary)"), fact("Length", lens, st ? null : "(first " + pv.n_read + " series)"),
      fact("Missing values", st ? fpct(st.missing_share, 2) : "?"), fact("Share of zeros", fpct(st ? st.zero_share : pv.zero_share, 1))),
    d.stats_note ? el("p", { class: "hint", text: d.stats_note }) : null,
    d.consistency.map(w => el("p", { class: "note soft", text: w })),
    prof.n ? bindNote(el("div", { class: "profile", tabindex: 0 },
      el("div", { class: "hint", text: `Demand pattern (first ${prof.n} series)` }),
      el("div", { class: "bar", role: "img", "aria-label": "Distribution of demand patterns" }, PROFILE.map(([k, , c]) => prof.share[k] > 0 ? el("i", { style: `width:${prof.share[k] * 100}%;background:${c}` }) : null)),
      el("div", { class: "legend" }, PROFILE.filter(([k]) => prof.counts[k] > 0).map(([k, l, c]) => el("span", {}, el("i", { class: "sw", style: `background:${c}` }), `${l} ${fpct(prof.share[k], 0)}`)))),
      () => profileNote()) : null,
    el("div", { class: "sparks" }, pv.series.map(s => spark(s))),
  );
}
function profileNote() {
  const r = S.models.ref_table.syntetos2005cat;
  return el("div", {}, el("h3", { text: "Demand pattern" }),
    el("p", { class: "lead", text: "Each series is assigned to one of four classes by its average demand interval (ADI) and the variability of its non-zero demands (CV²)." }),
    el("p", { text: "ADI < 1.32 and CV² < 0.49: smooth. ADI < 1.32, CV² ≥ 0.49: erratic. ADI ≥ 1.32, CV² < 0.49: intermittent. ADI ≥ 1.32, CV² ≥ 0.49: lumpy. Series that are all zero are counted separately." }),
    runIn("How to use it.", "If most series are intermittent or lumpy, try the Croston family (Intermittent demand group); for smooth series, classical methods such as ETS/Theta are the usual choice."),
    runIn("Caution.", "Only the first series of the dataset are scanned; the thresholds are the suggested cut-off values for classification, not evidence for choosing a model.", "caution"),
    refList([r]));
}
function spark(s) {
  const ys = s.y.filter(v => v != null);
  const lo = Math.min(...ys), hi = Math.max(...ys), W = 240, H = 44, n = s.y.length;
  let dd = "", pen = false;
  s.y.forEach((v, i) => { if (v == null) { pen = false; return; } const x = (i / Math.max(1, n - 1)) * W, y = H - 3 - ((v - lo) / (hi - lo || 1)) * (H - 8); dd += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1); pen = true; });
  return el("div", { class: "spark" }, svg("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img", "aria-label": "Series " + s.name },
    svg("path", { d: dd, fill: "none", stroke: "var(--ink)", "stroke-width": "1.3", "vector-effect": "non-scaling-stroke" })), `${s.name} · ${fint(s.n)} observations`.replace(" · ", ", "));
}

function uploadBlock() {
  const info = el("div", { id: "upinfo" });
  const fill = el("input", { type: "checkbox", id: "fillgaps" });
  const input = el("input", { type: "file", accept: ".tsf,.xlsx,.xlsm", hidden: true, onchange: ev => { if (ev.target.files[0]) doUpload(ev.target.files[0], fill.checked, info); ev.target.value = ""; } });
  const box = el("div", { class: "upload" },
    el("h3", { text: "Use your own data" }),
    el("p", { text: "Upload a .tsf file or an Excel (.xlsx) file that follows the template; Excel is converted to .tsf automatically and a validation report appears below." }),
    el("div", { class: "row-actions" },
      el("button", { type: "button", class: "btn ghost small", onclick: () => input.click() }, "Choose file"),
      el("a", { href: `/api/template.xlsx?t=${encodeURIComponent(TOKEN)}`, download: "forecast_studio_template.xlsx" }, "Download the Excel template"),
      el("label", { class: "hint", style: "display:inline-flex;gap:6px;align-items:center" }, fill, "In Excel, fill missing dates with missing values")),
    input, info);
  ["dragenter", "dragover"].forEach(t => box.addEventListener(t, ev => { ev.preventDefault(); box.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(t => box.addEventListener(t, ev => { ev.preventDefault(); box.classList.remove("drag"); }));
  box.addEventListener("drop", ev => { const f = ev.dataTransfer.files[0]; if (f) doUpload(f, fill.checked, info); });
  bindNote(box, () => el("div", {}, el("h3", { text: "Use your own data" }),
    el("p", { text: "TSF (Monash format) or Excel based on the template. Excel may be in long (series, date, value) or wide format; frequency and horizon are read from the 'Meta' sheet." }),
    runIn("Validation.", "The converter reports missing dates, ordering and name adjustments. If there is an error the file is not uploaded and the reason is shown."),
    runIn("Privacy.", "The file is only copied into the 'uploads' folder on this computer.")));
  return box;
}
async function doUpload(file, fill, info) {
  info.replaceChildren(el("p", { class: "hint", text: `Uploading ${file.name}…` }));
  try {
    const r = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}&fill_gaps=${fill ? 1 : 0}`, { method: "POST", headers: { "X-TSF-Token": TOKEN }, body: file });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Upload failed");
    const lines = [...(j.errors || []).map(e => "ERROR: " + e), ...(j.warnings || []).map(w => "WARNING: " + w)];
    if (!j.ok) { info.replaceChildren(el("p", { class: "note err", text: "The file was not uploaded." }), el("div", { class: "report bad", text: lines.join("\n") || "Unknown error." })); return; }
    info.replaceChildren(el("p", { class: "note soft", text: `Uploaded: ${j.id.replace(/^upload:/, "")}.` + (j.info && j.info.n_series ? ` ${j.info.n_series} series, ${frq(j.info.frequency)} frequency, horizon ${j.info.horizon}.` : "") }),
      lines.length ? el("div", { class: "report", text: lines.join("\n") }) : null);
    S.datasets = await api("datasets"); S.tab = "user";
    renderVeri(); $("#upinfo").replaceWith(info.cloneNode(true));
    selectDataset(j.id);
  } catch (e) { info.replaceChildren(el("p", { class: "note err", text: e.message })); }
}

// ================================================================================================ 2 MODEL
function engineBadges(m) {
  return ENGINES.filter(k => m.engines[k]).map(k => {
    const e = m.engines[k];
    if (!e.supported) return el("span", { class: "eng no", title: "Not available in this engine", text: ENGINE_SHORT[k] });
    return el("span", { class: "eng" + (e.available ? "" : " miss"), text: ENGINE_SHORT[k], title: e.available ? "" : "Not installed" });
  });
}
function renderModel() {
  const sec = $("#model");
  const focusId = keptFocus("#model .mrow");
  queueMicrotask(() => restoreFocus("#model .mrow", focusId));
  const studied = new Set(S.detail ? [...(S.detail.study_methods || []), ...(S.detail.recommended_global || [])] : []);
  sec.replaceChildren(
    ...secHead("model", 2, "Model", "Choose a method. Its explanation appears in the right margin; below you can also choose which engine runs it."),
    roving(el("div", { class: "groups", role: "radiogroup", "aria-label": "Models" }, S.models.groups.map(g => {
      const ms = S.models.models.filter(m => m.group === g.id);
      return el("div", { class: "group" }, el("h3", { text: g.title }), el("p", { text: g.text }),
        ms.map(m => {
          const on = S.sel.method === m.id;
          const any = ENGINES.some(k => m.engines[k] && m.engines[k].available);
          const row = el("div", { class: "mrow", role: "radio", "aria-checked": String(on), tabindex: 0, "data-id": m.id, onclick: () => selectModel(m.id),
            onkeydown: ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); selectModel(m.id); } } },
            el("span", { class: "t" }, on ? el("span", { class: "hl", text: m.name }) : m.name, studied.has(m.id) ? el("span", { class: "studied", text: "used in the reference study" }) : null),
            el("span", { class: "s" }, m.summary, any ? null : el("span", { class: "pen", text: " Not installed on this machine." })),
            el("span", { class: "e" }, engineBadges(m)));
          return bindNote(row, () => modelNote(m));
        }));
    })), ".mrow[role=radio]"),
    el("div", { id: "enginepick" }));
  renderEnginePick();
}
function renderEnginePick() {
  const host = $("#enginepick"); if (!host) return;
  const m = curModel();
  if (!m) { host.replaceChildren(); return; }
  const opts = ENGINES.filter(k => m.engines[k] && m.engines[k].supported);
  host.replaceChildren(el("div", { class: "enginepick" }, el("span", { text: "Engine:" }),
    el("div", { class: "seg", role: "group", "aria-label": "Engine" }, opts.map(k => bindNote(el("button", { type: "button", "aria-pressed": String(S.sel.engine === k),
      onclick: () => setEngine(k) }, ENGINE_NAME[k] + (m.engines[k].available ? "" : " (unavailable)")), () => el("div", {}, el("h3", { text: ENGINE_NAME[k] + " engine" }), el("p", { text: m.engines[k].note || "" }),
      m.engines[k].available ? null : runIn("Unavailable.", ((S.sys.engines[k].message) || "") , "caution"))))),
    opts.length === 1 ? el("span", { class: "hint", text: "This method exists only in the " + ENGINE_NAME[opts[0]] + " engine." }) : el("span", { class: "hint", text: "The R engine runs the original R code of the reference study; the Python engine uses its own libraries; the C++ engine is a fast native port (of the R code, or of the Python model where noted)." })));
}
function setEngine(k) {
  S.sel.engine = k; store.set("sel", S.sel);
  resetParams(); renderEnginePick(); renderModel(); renderAyarlarParams(); renderKosuDebounced(); updateRail(); refreshStale();
}
function selectModel(id, silent) {
  const m = model(id); if (!m) return;
  S.sel.method = id;
  const avail = ENGINES.filter(k => m.engines[k] && m.engines[k].supported && m.engines[k].available);
  if (!m.engines[S.sel.engine] || !m.engines[S.sel.engine].supported || (!m.engines[S.sel.engine].available && avail.length))
    S.sel.engine = avail[0] || ENGINES.find(k => m.engines[k] && m.engines[k].supported) || "python";
  S.autoOff = false;
  if (m.group !== "aralikli" && S.settings.integer === "off" && S.autoOff === false && S._autoOffSet) { S.settings.integer = "auto"; }
  S._autoOffSet = false;
  resetParams();
  if (m.group === "aralikli" && S.detail && S.detail.integer_conversion && S.settings.integer === "auto") { S.settings.integer = "off"; S.autoOff = true; S._autoOffSet = true; }
  applyDatasetDefaults();
  store.set("sel", S.sel);
  Note.hovering = false; Note.base = () => (curModel() ? modelNote(curModel()) : introNote()); Note.restore();
  renderModel(); renderAyarlar(); renderKosuDebounced(); updateRail(); refreshStale();
  if (!silent) $("#enginepick").scrollIntoView({ block: "nearest", behavior: "smooth" });
}
function resetParams() {
  const m = curModel(); S.params = {};
  if (!m) return;
  for (const p of m.params) if (p.engines.includes(S.sel.engine)) S.params[p.key] = p.default == null ? "" : String(p.default);
}

// ================================================================================================ 3 AYARLAR
function field(label, input, noteBuilder, changed, reset) {
  const f = el("div", { class: "field" + (changed ? " changed" : "") }, el("label", {}, el("span", { text: label }), reset || null), input);
  return noteBuilder ? bindNote(f, noteBuilder) : f;
}
function renderAyarlar() {
  const sec = $("#ayarlar");
  const m = curModel();
  if (!m) { sec.replaceChildren(...secHead("ayarlar", 3, "Settings", "Choose a model first."), el("p", { class: "empty", text: "Settings appear here according to the model you choose." })); return; }
  const d = S.detail, s = S.settings;
  const global = m.task === "global";
  const inp = (key, type, ph, extra) => el("input", { type, value: s[key], placeholder: ph || "", min: extra && extra.min, step: extra && extra.step || "1", inputmode: "numeric",
    oninput: ev => { s[key] = ev.target.value; ev.target.closest(".field").classList.toggle("changed", ev.target.value !== ""); renderKosuDebounced(); } });
  const intSeg = el("div", { class: "seg", role: "group", "aria-label": "Round to integers" }, [["auto", "Dataset default" + (d ? ` (${d.integer_conversion ? "on" : "off"})` : "")], ["on", "On"], ["off", "Off"]].map(([v, l]) =>
    el("button", { type: "button", "aria-pressed": String(s.integer === v), onclick: () => { s.integer = v; S.autoOff = false; renderAyarlar(); renderKosuDebounced(); } }, l)));
  const effRound = s.integer === "on" || (s.integer === "auto" && d && d.integer_conversion);
  const roundWarn = m.group === "aralikli" && effRound ? el("p", { class: "note", text: "Rounding is on: the Croston family produces a flat demand rate (often below 0.5); once rounded, every forecast becomes 0. Choose Off." }) : null;
  const autoNote = S.autoOff && s.integer === "off" ? el("p", { class: "note soft", text: "This method is meant for intermittent demand, so we turned rounding off for you. You can change it above if you wish." }) : null;
  const missSel = el("select", { onchange: ev => { s.missing = ev.target.value; renderKosuDebounced(); } },
    S.models.missing_policies.map(p => el("option", { value: p.value, selected: s.missing === p.value }, p.label)));
  const maxQuick = el("div", { class: "quick" }, [["20", "20"], ["100", "100"], ["", "All"]].map(([v, l]) => el("button", { type: "button", class: "btn ghost small",
    onclick: () => { s.max_series = v; renderAyarlar(); renderKosuDebounced(); } }, l)));
  const metricsBox = el("div", { class: "chips", role: "group", "aria-label": "Measures" }, (S.models.metric_all || S.models.metric_defaults).map(k => bindNote(el("label", {},
    el("input", { type: "checkbox", checked: s.metrics.has(k), onchange: ev => { ev.target.checked ? s.metrics.add(k) : s.metrics.delete(k); renderKosuDebounced(); } }), S.models.metrics[k].name), () => metricNote(k))));
  sec.replaceChildren(
    ...secHead("ayarlar", 3, "Settings", (d && d.kind === "catalog" ? "Defaults come from the reference study (dataset catalogue)" : "Defaults come from the data file header") + "; edit only what you want to change."),
    el("div", { class: "grid3" },
      field("Forecast horizon", inp("horizon", "number", d && d.horizon != null ? String(d.horizon) + " (data)" : "from dataset", { min: 1 }), settingNote("horizon"), s.horizon !== ""),
      global ? field("Lag", inp("lag", "number", d && d.lag != null ? String(d.lag) + (d.kind === "catalog" ? " (reference study)" : " (data)") : "required", { min: 1 }), settingNote("lag", () => d && d.lag_suggestion && d.lag == null ? el("p", { class: "hint", text: `Suggestion: ${d.lag_suggestion} (based on the seasonal period; not evidence-based).` }) : null), s.lag !== "") : null,
      field("Number of series", el("div", {}, inp("max_series", "number", "all", { min: 1 }), maxQuick), settingNote("max_series"), s.max_series !== "")),
    el("div", { class: "grid2 sub" },
      bindNote(el("div", { class: "field" }, el("div", { class: "lab", text: "Round to integers" }), intSeg, roundWarn, autoNote), settingNote("integer")),
      field("Missing values", missSel, () => {
        const pol = S.models.missing_policies.find(p => p.value === s.missing);
        return el("div", {}, el("h3", { text: SETTING_NOTES.missing[0] }), el("p", { text: SETTING_NOTES.missing[1] }), pol ? runIn(pol.label + ".", pol.help) : null);
      })),
    el("div", { class: "sub" }, bindNote(el("div", { class: "field" }, el("div", { class: "lab", text: "Measures" }), metricsBox), settingNote("metrics"))),
    el("details", { class: "adv" }, el("summary", { text: "Seed and workers" }),
      el("div", { class: "grid3 sub" },
        field("Seed", inp("seed", "number", "1", { min: 0 }), settingNote("seed"), s.seed !== "1"),
        field("Workers", inp("workers", "number", "1", { min: 1 }), settingNote("workers"), s.workers !== "1")),
      el("p", { class: "hint", text: `This computer has ${S.sys.cpu || "?"} cores.` })),
    el("div", { id: "mparams" }));
  renderAyarlarParams();
}
function renderAyarlarParams() {
  const host = $("#mparams"); if (!host) return;
  const m = curModel(); if (!m) { host.replaceChildren(); return; }
  const ps = m.params.filter(p => p.engines.includes(S.sel.engine));
  const mk = p => {
    const cur = S.params[p.key] ?? "";
    const dflt = p.default == null ? "" : String(p.default);
    const changed = cur !== dflt;
    let input;
    const set = v => { S.params[p.key] = v; renderKosuDebounced(); };
    const mark = ev => { const ch = ev.target.value !== dflt; ev.target.closest(".field").classList.toggle("changed", ch); reset.hidden = !ch; };
    if (p.kind === "choice") input = el("select", { onchange: ev => { set(ev.target.value); mark(ev); } }, p.choices.map(c => el("option", { value: c.value, selected: cur === c.value }, c.label)));
    else if (p.kind === "bool") input = el("select", { onchange: ev => { set(ev.target.value); mark(ev); } }, el("option", { value: "false", selected: cur !== "true" }, "No"), el("option", { value: "true", selected: cur === "true" }, "Yes"));
    else input = el("input", { type: "number", value: cur, placeholder: p.placeholder || "", min: p.min, max: p.max, step: p.step || (p.kind === "int" ? 1 : "any"), oninput: ev => { set(ev.target.value); mark(ev); } });
    const reset = el("button", { type: "button", class: "linkish reset", hidden: !changed, onclick: () => { S.params[p.key] = dflt; renderAyarlarParams(); renderKosuDebounced(); } }, "back to default");
    return field(p.label, input, () => paramNote(p, m), changed, reset);
  };
  const main = ps.filter(p => !p.advanced), adv = ps.filter(p => p.advanced);
  host.replaceChildren(
    el("h3", { text: "Model settings" }), el("p", { class: "hint", text: m.name + " (" + ENGINE_NAME[S.sel.engine] + " engine)" }),
    ps.length ? null : el("p", { class: "empty", text: "This method has no adjustable parameters." }),
    main.length ? el("div", { class: "grid2" }, main.map(mk)) : null,
    adv.length ? el("details", { class: "adv" }, el("summary", { text: "Advanced" }), el("div", { class: "grid2 sub" }, adv.map(mk))) : null);
}

// ================================================================================================ 4 RUN
function buildConfig() {
  const s = S.settings;
  const params = {};
  const m = curModel();
  if (m) for (const p of m.params) {
    if (!p.engines.includes(S.sel.engine)) continue;
    const v = S.params[p.key];
    if (v === undefined || v === "") continue;
    params[p.key] = p.kind === "bool" ? v === "true" : v;
  }
  return { dataset: S.sel.dataset, engine: S.sel.engine, method: S.sel.method, horizon: s.horizon, lag: s.lag, integer_conversion: s.integer, missing_policy: s.missing,
    max_series: s.max_series, seed: s.seed, workers: s.workers, metrics: [...s.metrics], params };
}
// Load a past run's configuration into the workspace (UI review M4): dataset, model, engine, settings and parameters.
async function loadSettings(m) {
  const c = m.config || {}, dsId = m.dataset && m.dataset.id;
  if (!dsId || !entry(dsId)) { toast(`The dataset of this run (${dsId || "?"}) is not available in this data folder.`, true); return; }
  if (!model(c.method)) { toast(`The method of this run (${c.method}) is not known to this version.`, true); return; }
  S.sel.engine = c.engine || "python";
  selectModel(c.method, true);
  await selectDataset(dsId, true);
  const str = v => (v == null ? "" : String(v));
  Object.assign(S.settings, {
    horizon: str(c.horizon), lag: str(c.lag), missing: c.missing_policy || "error", max_series: str(c.max_series),
    seed: str(c.seed == null ? 1 : c.seed), workers: str(c.workers == null ? 1 : c.workers),
    integer: c.integer_conversion == null ? "auto" : (c.integer_conversion ? "on" : "off"),
  });
  if (Array.isArray(c.metrics)) S.settings.metrics = new Set(c.metrics.filter(k => S.models.metrics[k]));
  S.params = {};
  for (const [k, v] of Object.entries(c.params || {})) {
    if (v && typeof v === "object" && !Array.isArray(v)) for (const [k2, v2] of Object.entries(v)) S.params[`${k}.${k2}`] = String(v2);
    else S.params[k] = typeof v === "boolean" ? String(v) : String(v);
  }
  setView("work"); renderModel(); renderAyarlar(); renderKosuDebounced(); updateRail(); refreshStale();
  $("#ayarlar").scrollIntoView({ behavior: "smooth" });
  toast(`Settings loaded from ${runLabel(m)}. Check them, then start a run.`);
}
const renderKosuDebounced = debounce(() => refreshEstimate(false), 550);
function ready() { return !!(S.sel.dataset && S.sel.method); }
function methodAvailable() { const m = curModel(); return !!(m && m.engines[S.sel.engine] && m.engines[S.sel.engine].available); }
async function refreshEstimate(manual, probe) {
  if (!ready()) { S.est = null; renderKosu(); return; }
  const ds = entry(S.sel.dataset);
  if (!manual && ds && ds.bytes > BIG) { S.est = { needsManual: true }; renderKosu(); return; }
  if (!methodAvailable()) { S.est = null; renderKosu(); return; }
  if (S.settings.metrics.size === 0) { S.est = { ok: false, error: "Select at least one measure." }; renderKosu(); return; }
  const seq = ++S.estSeq; S.estBusy = true; renderKosu();
  try {
    const r = await api("estimate", { config: buildConfig(), probe: !!probe });
    if (seq !== S.estSeq) return;
    S.est = r;
  } catch (e) { if (seq !== S.estSeq) return; S.est = { ok: false, error: e.message }; }
  S.estBusy = false; S.confirmLong = false; renderKosu();
}
function renderKosu() {
  const sec = $("#kosu");
  const m = curModel(), est = S.est, running = S.job && S.job.status === "running";
  const parts = [...secHead("kosu", 4, "Run", "Before you start, the pre-run check shows the estimated time and memory; long runs ask for your confirmation.")];
  if (!ready()) parts.push(el("p", { class: "empty", text: "Once a dataset and a model are chosen, the pre-run check appears here." }));
  else if (!methodAvailable()) {
    const e = m.engines[S.sel.engine];
    parts.push(el("p", { class: "note err", text: `${m.name} is not available in the ${ENGINE_NAME[S.sel.engine]} engine on this machine.` + (e && e.supported ? " " + ((S.sys.engines[S.sel.engine].message) || "") : " This method does not exist in that engine.") }));
  } else if (S.estBusy) parts.push(el("p", { class: "empty", text: "Preparing the pre-run check…" }));
  else if (est && est.needsManual) parts.push(el("p", { class: "note soft", text: "This data file is large; the pre-run check reads all of it (a few seconds to minutes, and it uses memory)." }),
    el("div", { class: "row-actions" }, el("button", { type: "button", class: "btn ghost", onclick: () => refreshEstimate(true) }, "Run pre-run check")));
  else if (est && !est.ok) parts.push(el("p", { class: "note err", text: est.error }));
  else if (est) {
    const dl = el("dl", {});
    const row = (k, v) => { dl.append(el("dt", { text: k }), el("dd", {}, v)); };
    const total = S.detail && S.detail.stats ? S.detail.stats.n_series : null;
    row("Series", `${fint(est.n_series_run)}` + (total && est.n_series_run < total ? ` (of ${fint(total)})` : "") + (est.excluded ? `, ${est.excluded} series excluded` : ""));
    row("Horizon", fint(est.horizon)); row("Seasonality", est.seasonality.map(x => nf2.format(x)).join(", "));
    row("Rounding", est.integer_conversion ? "on" : "off"); if (est.lag != null) row("Lag", fint(est.lag));
    if (est.runtime) row("Run time", est.runtime.message + (est.probed ? "" : ""));
    if (est.memory) row("Memory", est.memory.message);
    parts.push(el("div", { class: "pre" }, dl), est.warnings.map(w => el("p", { class: "note", text: w })),
      est.blocked ? el("p", { class: "note err", text: "Memory looks insufficient; the run cannot start. Reduce the number of series (Settings) or close other applications." }) : null);
  }
  const needConfirm = !!(est && est.ok && est.long_running);
  if (needConfirm) parts.push(el("label", { class: "confirm" }, el("input", { type: "checkbox", checked: S.confirmLong, onchange: ev => { S.confirmLong = ev.target.checked; renderKosu(); } }),
    el("span", { text: "This run will take long; I have read the time estimate above and confirm that it should start." })));
  const canRun = ready() && methodAvailable() && est && est.ok && !est.blocked && (!needConfirm || S.confirmLong) && !running;
  const canProbe = ready() && methodAvailable() && !running && !S.estBusy && (!est || !est.needsManual) && curModel().task === "local";
  parts.push(el("div", { class: "row-actions" },
    el("button", { type: "button", class: "btn big", disabled: !canRun, onclick: startRun }, "Start run"),
    el("button", { type: "button", class: "btn ghost", disabled: !canProbe, onclick: () => refreshEstimate(true, true),
      title: "Measures the time with a real trial run on 8 series" }, "Measure time (8 series)")));
  parts.push(el("div", { id: "progress" }));
  sec.replaceChildren(...parts);
  renderProgress();
}
async function startRun() {
  try {
    S.job = await api("run", { config: buildConfig(), confirm_long: S.confirmLong });
    S.run = null; renderSonuc(); renderKosu(); pollJob();
  } catch (e) { toast(e.message, true); }
}
// UI review M8: one failed poll no longer stops progress for good; retry with backoff, then say the server is unreachable.
function pollJob(failures = 0) {
  clearTimeout(S.pollTimer);
  S.pollTimer = setTimeout(async () => {
    let j;
    try { j = await api("jobs/" + S.job.id); }
    catch (e) {
      if (failures < 5) return pollJob(failures + 1);
      const host = $("#progress");
      if (host) host.replaceChildren(el("p", { class: "note err", text: "Lost contact with the server (" + e.message + "). Is the Forecast Studio window still open? " },
        el("button", { type: "button", class: "btn ghost small", onclick: () => pollJob(0) }, "Try again")));
      return;
    }
    S.job = j;
    renderProgress();
    if (S.job.status === "running") return pollJob(0);
    renderKosu();
    if (S.job.status === "done") { await loadRun(S.job.run_id); refreshHistoryCount(); $("#sonuc").scrollIntoView({ behavior: "smooth" }); }
  }, failures ? Math.min(600 * 2 ** failures, 8000) : 600);
}
// Durations (UI review M2): short runs are not "0:00": below 10 s one decimal, below a minute whole seconds, then m:ss.
function mmss(s) {
  if (s == null || Number.isNaN(s)) return "—";
  if (s < 0.05) return "< 0.1 s";
  if (s < 10) return nf1fix.format(s) + " s";
  if (s < 60) return Math.round(s) + " s";
  s = Math.round(s); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function renderProgress() {
  const host = $("#progress"); if (!host) return;
  const j = S.job;
  if (!j) { host.replaceChildren(); return; }
  if (j.status === "running") {
    const pct = j.total > 1 ? Math.min(100, (j.done / j.total) * 100) : null;
    const what = j.dataset ? `${String(j.dataset).replace(/^upload:/, "")}, ${j.method} (${ENGINE_NAME[j.engine] || j.engine})` : "";
    host.replaceChildren(el("div", { class: "progress", role: "status" },
      what ? el("div", { class: "hint", text: "Running: " + what }) : null,
      el("div", {}, `${STAGE[j.stage] || j.stage}` + (j.total > 1 ? `: ${fint(j.done)} / ${fint(j.total)} series` : "") + `, elapsed ${mmss(j.elapsed)}`),
      el("div", { class: "track" + (pct == null ? " indet" : ""), "aria-hidden": "true" }, el("i", { style: pct == null ? "" : `width:${pct}%` })),
      el("div", { class: "row-actions", style: "margin-top:10px" }, el("button", { type: "button", class: "btn ghost small", disabled: j.cancelling, onclick: () => api(`jobs/${j.id}/cancel`, {}).then(v => { S.job = v; renderProgress(); }).catch(e => toast(e.message, true)) }, j.cancelling ? "Cancelling…" : "Cancel"))));
  } else if (j.status === "error") host.replaceChildren(el("p", { class: "note err", text: "Run failed: " + j.error }));
  else if (j.status === "cancelled") host.replaceChildren(el("p", { class: "note soft", text: "Run cancelled." }));
  else if (runIsStale()) host.replaceChildren(el("p", { class: "note", text: `The last run (${runLabel(S.run.detail.manifest)}) finished. Its results below do not match your current selection; start a run to get results for ${selLabel()}.` }));
  else host.replaceChildren(el("p", { class: "note soft", text: `Run finished (${mmss(j.elapsed)}). Results are below.` }));
}

// ================================================================================================ 5 RESULTS
function runLabel(m) { const c = m.config, d = m.dataset; return `${d.id.replace(/^upload:/, "")}, ${c.method} (${ENGINE_NAME[c.engine] || c.engine})`; }
// True when the shown run was made with a different dataset/method/engine than the current selection (UI review H4).
function runIsStale() {
  if (!S.run || !S.run.detail) return false;
  const m = S.run.detail.manifest;
  return m.dataset.id !== S.sel.dataset || m.config.method !== S.sel.method || m.config.engine !== S.sel.engine;
}
function selLabel() { return `${(S.sel.dataset || "?").replace(/^upload:/, "")}, ${S.sel.method || "?"} (${ENGINE_NAME[S.sel.engine] || S.sel.engine || "?"})`; }
function renderStaleNote() {
  const host = $("#stalenote"); if (!host) return;
  host.replaceChildren(...(runIsStale() ? [el("p", { class: "note err", text: `These results are from an earlier run (${runLabel(S.run.detail.manifest)}), not from your current selection (${selLabel()}). Start a run to replace them.` })] : []));
}
function refreshStale() { renderStaleNote(); if (S.job) renderProgress(); }
async function loadRun(id) {
  try {
    const detail = await api("runs/" + encodeURIComponent(id));
    const names = (await api(`runs/${encodeURIComponent(id)}/names`)).names;
    S.run = { id, detail, names, series: null, errors: null, sort: null, desc: true, q: "", offset: 0, hist: null, histMetric: null, cur: null };
    const cols = detail.metrics_columns;
    S.run.histMetric = cols.includes("MASE") ? "MASE" : cols[0];
    S.run.shortMetric = S.run.histMetric;
    renderSonuc(); updateRail();
    await Promise.all([loadErrors(), loadHist(), showSeries(names[0])]);
  } catch (e) { toast(e.message, true); }
}
async function loadErrors() {
  const r = S.run; if (!r) return;
  const qs = new URLSearchParams({ offset: r.offset, limit: 20, q: r.q || "" });
  if (r.sort) { qs.set("sort", r.sort); qs.set("dir", r.desc ? "desc" : "asc"); }
  r.errors = await api(`runs/${encodeURIComponent(r.id)}/errors?${qs}`);
  renderErrorTable();
}
async function loadHist() {
  const r = S.run; if (!r) return;
  r.hist = await api(`runs/${encodeURIComponent(r.id)}/hist?metric=${encodeURIComponent(r.histMetric)}`);
  renderHist();
}
async function showSeries(name) {
  const r = S.run; if (!r || !name) return;
  try { r.series = await api(`runs/${encodeURIComponent(r.id)}/series?name=${encodeURIComponent(name)}`); r.cur = name; renderChart(true); } catch (e) { toast(e.message, true); }
}
function renderSonuc() {
  const sec = $("#sonuc");
  const r = S.run;
  const head = secHead("sonuc", 5, "Results", "When a run finishes, the summary, diagnostics, a sample-series chart and per-series errors appear here.");
  if (!r) { sec.replaceChildren(...head, el("p", { class: "empty", text: S.job && S.job.status === "running" ? "Run in progress…" : "No results yet. Start a run or open one from Past runs." })); return; }
  const m = r.detail.manifest, d = r.detail, dg = d.diagnostics || {};
  const fact = (k, v, flag) => el("div", { class: flag ? "flag" : "" }, el("dt", { text: k }), el("dd", {}, v));
  const fbs = d.fallbacks || [], nfb = (m.fallbacks && m.fallbacks.count) || 0;
  const cfg = m.config || {}, eng = m.engine || {}, sw = m.software || {};
  const kv = (k, v) => (v == null || v === "" ? null : el("div", {}, el("dt", { text: k }), el("dd", { text: String(v) })));
  const params = cfg.params && Object.keys(cfg.params).length ? Object.entries(cfg.params).map(([k, v]) => `${k} = ${typeof v === "object" ? JSON.stringify(v) : v}`).join(", ") : "library defaults";
  const versions = Object.entries(eng.versions || {}).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(", ");
  const warns = (m.warnings || []);
  const flagZero = dg.series_all_zero_forecast_share != null && dg.series_all_zero_forecast_share > 0.5;
  const diagItem = (k, v, flag, note) => bindNote(el("div", { class: flag ? "flag" : "", tabindex: 0 }, el("dt", { text: k }), el("dd", { text: v })), note);
  sec.replaceChildren(...head,
    el("div", { id: "stalenote", role: "status" }),
    el("p", { class: "rtitle" }, el("span", { class: "hl", text: runLabel(m) })),
    el("p", { class: "rsub", text: r.id }),
    el("div", { class: "row-actions" }, el("button", { type: "button", class: "btn ghost small", onclick: () => loadSettings(m) }, "Load these settings into the workspace")),
    el("dl", { class: "facts" }, fact("Series", fint(m.dataset.n_series_run)), fact("Horizon", fint(m.config.effective_horizon)), fact("Run time", mmss(m.engine.elapsed_sec || 0)),
      fact("Fallback forecasts", fint(nfb), nfb > 0), fact("Rounding", m.config.effective_integer_conversion ? "on" : "off")),
    // UI review M3: which series fell back to the seasonal-naive forecast, and why.
    nfb > 0 ? el("details", { class: "fbdetails" },
      el("summary", { text: `${fint(nfb)} series received the fallback forecast (seasonal naive) instead of ${cfg.method}. Show which and why` }),
      el("div", { class: "tablewrap" }, el("table", { class: "t" }, el("thead", {}, el("tr", {}, el("th", { text: "Series" }), el("th", { text: "Reason" }))),
        el("tbody", {}, fbs.map(f => el("tr", {}, el("td", { class: "name", text: f.series_name || String(f.series_index) }), el("td", { text: f.message || "" })))))),
      nfb > fbs.length ? el("p", { class: "hint", text: `Showing the first ${fbs.length}. The full list is in fallbacks.csv (Files, below).` }) : null) : null,
    // UI review M4: what produced these numbers, without opening manifest.json.
    el("details", { class: "runcfg" }, el("summary", { text: "Run configuration (for reproducing this result)" }),
      el("dl", { class: "kvlist" },
        kv("Method / engine", `${cfg.method} / ${ENGINE_NAME[cfg.engine] || cfg.engine}`), kv("Parameters", params),
        kv("Horizon", cfg.effective_horizon), kv("Seasonal periods", (cfg.effective_seasonality || []).map(p => +(+p).toFixed(4)).join(", ")),
        kv("Lag", cfg.lag), kv("Seed", cfg.seed), kv("Workers", cfg.workers), kv("Missing values", cfg.missing_policy),
        kv("Series limit", cfg.max_series || "all"), kv("Measures", (cfg.metrics || []).join(", ")),
        kv("Data file SHA-256", m.dataset && m.dataset.sha256), kv("Engine versions", versions),
        kv("Forecast Studio", sw.tsfstudio && `${sw.tsfstudio} (Python ${sw.python}, ${sw.platform || ""})`), kv("Created (UTC)", (m.created_utc || "").replace("T", " ").slice(0, 19)))),
    warns.map(w => el("p", { class: "note", text: w })),
    flagZero ? el("p", { class: "note err", text: `${fpct(dg.series_all_zero_forecast_share, 0)} of the series received only 0 forecasts. MAE, MASE and WAPE can make this look good; SMAPE approaches exactly 2.0.` + (m.config.effective_integer_conversion ? " Turn off rounding to integers and run again." : " Rounding is already off, so the forecasts themselves are zero (typical for a naive method on sparse series); do not rely on these measures alone.") }) : null,
    el("h3", { text: "Summary" }),
    el("div", { class: "tablewrap" }, summaryTable(d.summary)),
    el("h3", { text: "Forecast diagnostics" }),
    el("dl", { class: "diag" },
      diagItem("Share of zero forecasts", fpct(dg.zero_forecast_share, 1), flagZero, () => diagNote("zero_f")),
      diagItem("Share of zero actuals", fpct(dg.zero_actual_share, 1), false, () => diagNote("zero_a")),
      diagItem("Total bias", dg.bias_pct == null ? "—" : (dg.bias_pct > 0 ? "+" : "") + nf1.format(dg.bias_pct) + "%", false, () => diagNote("bias")),
      diagItem("Mean forecast / actual", `${fmt(dg.mean_forecast)} / ${fmt(dg.mean_actual)}`, false, () => diagNote("mean"))),
    el("h3", { text: "Sample series" }), el("div", { id: "chartbar" }), el("div", { id: "chart", class: "chart" }),
    el("h3", { text: "Error distribution" }), el("div", { id: "histbar" }), el("div", { id: "hist", class: "chart" }),
    el("h3", { text: "Errors by series" }), el("div", { id: "errtable" }),
    el("h3", { text: "Files" }),
    el("div", { class: "files" }, d.files.map(f => el("a", { href: fileUrl(r.id, f.key), download: f.name }, f.name))));
  renderStaleNote();
  renderChartBar();
}
function diagNote(k) {
  const T = {
    zero_f: ["Share of zero forecasts", "The proportion of forecasts that equal 0 in the test window. With intermittent demand, when forecasts are rounded or a flat rate stays small, this share approaches 1; such a model can look good on error measures but is useless for an inventory decision."],
    zero_a: ["Share of zero actuals", "The proportion of test-window points where actual demand is 0; it indicates how intermittent the data are."],
    bias: ["Total bias", "(Σ forecast − Σ actual) / Σ actual. Negative: systematic under-forecasting (stock-out risk); positive: over-forecasting (excess-stock risk). This is a preliminary indicator; inventory outcomes such as service level and stock cost are not computed in this version."],
    mean: ["Mean forecast / actual", "The mean of forecasts and of actuals over the test points; useful for seeing the scale of the bias."],
  }[k];
  return el("div", {}, el("h3", { text: T[0] }), el("p", { text: T[1] }), refList([S.models.ref_table.hyndman2006]));
}
function summaryTable(rows) {
  const best = {};
  const t = el("table", { class: "t" }, el("thead", {}, el("tr", {}, ["Measure", "Mean", "Median", "Unit", "Valid series", "Excluded"].map(h => el("th", { text: h })))),
    el("tbody", {}, rows.map(r => bindNote(el("tr", { tabindex: 0 }, el("td", { class: "name", text: (S.models.metrics[r.Metric] || { name: r.Metric }).name }),
      el("td", { text: fmtM(r.Metric, r.Mean) }), el("td", { text: fmtM(r.Metric, r.Median) }), el("td", { class: "unit", text: (S.models.metrics[r.Metric] || {}).unit || "" }),
      el("td", { text: fint(r.N_valid) }), el("td", { text: fint(r.N_excluded) })), () => metricNote(r.Metric)))));
  return t;
}
// ---- series chart
function niceTicks(lo, hi, n) {
  const span = hi - lo, raw = span / Math.max(1, n), p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
const axisFmt = v => new Intl.NumberFormat("en-US", { maximumSignificantDigits: 4 }).format(v);
function renderChartBar() {
  const r = S.run; if (!r) return;
  const cols = r.detail.metrics_columns;
  const datalist = el("datalist", { id: "snames" }, r.names.slice(0, 3000).map(n => el("option", { value: n })));
  const input = el("input", { type: "text", class: "search", style: "width:150px", list: "snames", value: r.cur || "", "aria-label": "Series name",
    onchange: ev => { if (r.names.includes(ev.target.value)) showSeries(ev.target.value); else toast("No series with this name.", true); } });
  const step = k => { const i = r.names.indexOf(r.cur); showSeries(r.names[(i + k + r.names.length) % r.names.length]); };
  const pick = async which => {
    const nv = r.detail.summary.find(x => x.Metric === r.shortMetric)?.N_valid || 1;
    const off = which === "max" ? 0 : which === "min" ? 0 : Math.floor(nv / 2);
    const res = await api(`runs/${encodeURIComponent(r.id)}/errors?sort=${encodeURIComponent(r.shortMetric)}&dir=${which === "min" || which === "med" ? "asc" : "desc"}&offset=${off}&limit=1`);
    if (res.rows[0]) showSeries(res.rows[0].series_name);
  };
  $("#chartbar").replaceChildren(el("div", { class: "chartbar" },
    el("button", { type: "button", class: "btn ghost small", onclick: () => step(-1), "aria-label": "Previous series" }, "‹"), input,
    el("button", { type: "button", class: "btn ghost small", onclick: () => step(1), "aria-label": "Next series" }, "›"), datalist,
    el("span", { class: "hint", text: "By:" }),
    el("select", { "aria-label": "Measure", onchange: ev => { r.shortMetric = ev.target.value; } }, cols.map(c => el("option", { value: c, selected: c === r.shortMetric }, (S.models.metrics[c] || { name: c }).name))),
    el("button", { type: "button", class: "btn ghost small", onclick: () => pick("max") }, "highest error"),
    el("button", { type: "button", class: "btn ghost small", onclick: () => pick("med") }, "median"),
    el("button", { type: "button", class: "btn ghost small", onclick: () => pick("min") }, "lowest error")),
    el("div", { class: "legend-line" },
      el("span", {}, svg("svg", { width: 26, height: 10 }, svg("line", { x1: 0, x2: 26, y1: 5, y2: 5, stroke: "var(--pencil)", "stroke-width": 1.5 })), "history"),
      el("span", {}, svg("svg", { width: 26, height: 10 }, svg("line", { x1: 0, x2: 26, y1: 5, y2: 5, stroke: "var(--ink)", "stroke-width": 2 }), svg("circle", { cx: 13, cy: 5, r: 3, fill: "var(--ink)" })), "actual (test)"),
      el("span", {}, svg("svg", { width: 26, height: 10 }, svg("line", { x1: 0, x2: 26, y1: 5, y2: 5, stroke: "var(--blue)", "stroke-width": 2, "stroke-dasharray": "5 3" }), svg("circle", { cx: 13, cy: 5, r: 3, fill: "var(--white)", stroke: "var(--blue)", "stroke-width": 1.6 })), "forecast"),
      el("span", {}, svg("svg", { width: 18, height: 10 }, svg("rect", { x: 0, y: 0, width: 18, height: 10, fill: "var(--hl)" })), "forecast horizon")));
  if (r.series) renderChart(false);
}
function renderChart(sweep) {
  const r = S.run, d = r && r.series, host = $("#chart"); if (!host || !d) return;
  const input = $("#chartbar input"); if (input) input.value = d.name;
  const W = Math.round(Math.max(330, Math.min(780, (host.clientWidth || 780) - 2))), narrow = W < 520, H = narrow ? 270 : 310, m = { l: narrow ? 46 : 58, r: 14, t: 22, b: 34 };
  const hist = d.history, nh = hist.length, H_ = d.horizon, N = nh + H_;
  const vals = [...hist, ...d.test, ...d.forecast].filter(v => v != null);
  if (!vals.length) { host.replaceChildren(el("p", { class: "empty", text: "No data to plot for this series." })); return; }
  let lo = Math.min(...vals), hi = Math.max(...vals); if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const iw = W - m.l - m.r, step = iw / N;
  const x = i => m.l + (i + 0.5) * step, y = v => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  const path = (arr, off) => { let s = "", pen = false; arr.forEach((v, i) => { if (v == null) { pen = false; return; } s += (pen ? "L" : "M") + x(i + off).toFixed(1) + " " + y(v).toFixed(1); pen = true; }); return s; };
  const yt = niceTicks(lo, hi, 5);
  const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Series ${d.name}: history, test actuals and a ${H_}-step forecast` });
  // forecast-horizon band (highlighter pen)
  g.append(svg("rect", { class: "band", x: x(nh) - step / 2, y: m.t, width: H_ * step, height: H - m.t - m.b }));
  yt.forEach(v => { g.append(svg("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--rule-soft)", "stroke-width": 1 }), svg("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, axisFmt(v))); });
  g.append(svg("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--ink)", "stroke-width": 1 }));
  // x axis: steps relative to the cut-off (t-10, t-20 ...), horizon end t+H
  const stepx = Math.max(1, Math.round(nh / 5 / 5) * 5 || 1), cutX = x(nh) - step / 2;
  for (let j = 1; j * stepx < nh; j++) { const i = nh - j * stepx;
    g.append(svg("line", { x1: x(i) - step / 2, x2: x(i) - step / 2, y1: H - m.b, y2: H - m.b + 4, stroke: "var(--ink)" }), svg("text", { x: x(i) - step / 2, y: H - m.b + 17, "text-anchor": "middle" }, "t−" + (j * stepx))); }
  g.append(svg("text", { x: cutX - 4, y: H - m.b + 17, "text-anchor": "end" }, "cut-off"));
  g.append(svg("text", { x: x(nh + H_ - 1) + step / 2 - 2, y: H - m.b + 17, "text-anchor": "end" }, "t+" + H_));
  g.append(svg("line", { x1: x(nh) - step / 2, x2: x(nh) - step / 2, y1: m.t, y2: H - m.b, stroke: "var(--ink)", "stroke-width": 1, "stroke-dasharray": "2 3" }));
  g.append(svg("path", { d: path(hist, 0), fill: "none", stroke: "var(--pencil)", "stroke-width": 1.5 }));
  g.append(svg("path", { d: path(d.test, nh), fill: "none", stroke: "var(--ink)", "stroke-width": 2 }));
  d.test.forEach((v, i) => v != null && g.append(svg("circle", { cx: x(nh + i), cy: y(v), r: 3, fill: "var(--ink)" })));
  g.append(svg("path", { d: path(d.forecast, nh), fill: "none", stroke: "var(--blue)", "stroke-width": 2.2, "stroke-dasharray": "6 4" }));
  d.forecast.forEach((v, i) => v != null && g.append(svg("circle", { cx: x(nh + i), cy: y(v), r: 3.2, fill: "var(--white)", stroke: "var(--blue)", "stroke-width": 1.7 })));
  // interaction
  const cross = svg("line", { y1: m.t, y2: H - m.b, stroke: "var(--blue)", "stroke-width": 1, opacity: 0 });
  const read = svg("text", { class: "readout", x: m.l + 8, y: H - m.b - 8 }, "");
  g.append(cross, read);
  const ov = svg("rect", { x: m.l, y: m.t, width: iw, height: H - m.t - m.b, fill: "transparent" });
  const move = ev => {
    const bb = g.getBoundingClientRect(), px = (ev.clientX - bb.left) / bb.width * W;
    const i = Math.max(0, Math.min(N - 1, Math.floor((px - m.l) / step)));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 1);
    if (i < nh) read.textContent = `t−${nh - i}: ${hist[i] == null ? "—" : fmt(hist[i])}`;
    else { const k = i - nh, a = d.test[k], f = d.forecast[k]; read.textContent = `t+${k + 1}: actual ${fmt(a)}, forecast ${fmt(f)}` + (a != null && f != null ? `, error ${fmt(f - a)}` : ""); }
  };
  ov.addEventListener("mousemove", move); ov.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); read.textContent = ""; });
  g.append(ov);
  host.className = "chart" + (sweep ? " sweep" : "");
  host.replaceChildren(g);
  if (d.note) host.append(el("p", { class: "note soft", text: d.note }));
  const er = d.errors || {};
  host.append(el("p", { class: "hint", style: "margin:6px 10px 8px", text: Object.keys(er).length ? "This series: " + Object.entries(er).map(([k, v]) => `${(S.models.metrics[k] || { name: k }).name} ${fmtM(k, v)}`).join(", ") : "" }));
}
// ---- distribution
function renderHist() {
  const r = S.run, h = r && r.hist, host = $("#hist"); if (!host) return;
  const cols = r.detail.metrics_columns;
  $("#histbar").replaceChildren(el("div", { class: "chartbar" }, el("span", { class: "hint", text: "Measure:" }),
    el("select", { "aria-label": "Distribution measure", onchange: ev => { r.histMetric = ev.target.value; loadHist(); } }, cols.map(c => el("option", { value: c, selected: c === r.histMetric }, (S.models.metrics[c] || { name: c }).name))),
    h && h.n ? el("span", { class: "hint", text: `${fint(h.n)} series` + (h.clipped ? `; the highest ${fint(h.clipped)} series are left out of view (above the 98th percentile)` : "") }) : null));
  if (!h || !h.n) { host.replaceChildren(el("p", { class: "empty", text: "No valid values for this measure." })); return; }
  const W = Math.round(Math.max(330, Math.min(780, (host.clientWidth || 780) - 2))), H = W < 520 ? 200 : 220, m = { l: 40, r: 14, t: 20, b: 34 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const mx = Math.max(...h.counts), n = h.counts.length, bw = iw / n, lo = h.edges[0], hi = h.edges[n];
  const x = v => m.l + (v - lo) / (hi - lo) * iw, y = c => m.t + (1 - c / mx) * ih;
  const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Distribution of ${r.histMetric}` });
  niceTicks(0, mx, 4).forEach(v => g.append(svg("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--rule-soft)" }), svg("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, fint(v))));
  h.counts.forEach((c, i) => g.append(svg("rect", { x: m.l + i * bw + 1, y: y(c), width: Math.max(1, bw - 2), height: ih - (y(c) - m.t), fill: "var(--ink)", opacity: 0.78 }, svg("title", {}, `${fmt(h.edges[i])} – ${fmt(h.edges[i + 1])}: ${c} series`))));
  g.append(svg("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: "var(--ink)" }));
  niceTicks(lo, hi, 6).forEach(v => g.append(svg("text", { x: x(v), y: H - m.b + 17, "text-anchor": "middle" }, axisFmt(v))));
  const mark = (v, label, dy, color) => { if (v >= lo && v <= hi) g.append(svg("line", { x1: x(v), x2: x(v), y1: m.t, y2: m.t + ih, stroke: color, "stroke-width": 2, "stroke-dasharray": label === "median" ? "4 3" : "" }), svg("text", { x: x(v) + 5, y: m.t + dy, fill: color, style: "fill:" + color + ";font-weight:700" }, `${label} ${fmt(v)}`)); };
  mark(h.mean, "mean", 12, "var(--pen)"); mark(h.median, "median", 26, "var(--blue)");
  host.replaceChildren(g);
}
// ---- error table
function renderErrorTable() {
  const r = S.run, e = r && r.errors, host = $("#errtable"); if (!host || !e) return;
  const cols = e.columns;
  const th = c => el("th", { class: "sortable", "aria-sort": r.sort === c ? (r.desc ? "descending" : "ascending") : null, tabindex: 0,
    onclick: () => { if (r.sort === c) r.desc = !r.desc; else { r.sort = c; r.desc = true; } r.offset = 0; loadErrors(); },
    onkeydown: ev => { if (ev.key === "Enter") ev.target.click(); } }, c === "series_name" ? "Series" : (S.models.metrics[c] || { name: c }).name, r.sort === c ? (r.desc ? " ↓" : " ↑") : "");
  const page = Math.floor(r.offset / 20) + 1, pages = Math.max(1, Math.ceil(e.total / 20));
  host.replaceChildren(
    el("div", { class: "chartbar" }, el("input", { type: "search", class: "search", style: "width:190px", placeholder: "Search series name", value: r.q || "", "aria-label": "Search series",
      onchange: ev => { r.q = ev.target.value; r.offset = 0; loadErrors(); } }), el("span", { class: "hint", text: `${fint(e.total)} series; click a header to sort` })),
    el("div", { class: "tablewrap" }, el("table", { class: "t" }, el("thead", {}, el("tr", {}, cols.map(th))),
      el("tbody", {}, e.rows.map(row => el("tr", { class: "clickrow", style: "cursor:pointer", tabindex: 0, onclick: () => { showSeries(row.series_name); $("#chartbar").scrollIntoView({ behavior: "smooth", block: "center" }); },
        onkeydown: ev => { if (ev.key === "Enter") ev.target.click(); } }, cols.map(c => el("td", { class: c === "series_name" ? "name" : "", text: c === "series_name" ? row[c] : fmtM(c, row[c]) }))))))),
    el("div", { class: "pager" }, el("button", { type: "button", class: "btn ghost small", disabled: r.offset <= 0, onclick: () => { r.offset = Math.max(0, r.offset - 20); loadErrors(); } }, "‹ Previous"),
      el("span", { text: `Page ${page} / ${pages}` }),
      el("button", { type: "button", class: "btn ghost small", disabled: page >= pages, onclick: () => { r.offset += 20; loadErrors(); } }, "Next ›"),
      el("span", { class: "hint", text: "Click a row to open the series chart." })));
}

// ================================================================================================ HISTORY and comparison
async function renderHistory() {
  const host = $("#view-history-body");
  host.replaceChildren(el("section", { class: "sec" }, el("h2", {}, "Past runs"), el("p", { class: "empty", text: "Reading…" })));
  try { S.history = await api("runs"); } catch (e) { host.replaceChildren(el("p", { class: "note err", text: e.message })); return; }
  const runs = S.history.runs;
  $("#history-count").textContent = runs.length ? `(${runs.length})` : "";
  const sel = runs.filter(r => S.compare.has(r.run_id));
  const metricKeys = [...new Set(runs.flatMap(r => Object.keys(r.summary || {})))];
  const head = el("div", { class: "runrow head" }, el("span", {}), el("span", { text: "Data, method" }), el("span", { text: "Date" }), el("span", { class: "m", text: "Series" }),
    el("span", { class: "m", text: "SMAPE" }), el("span", { class: "m", text: "MASE" }), el("span", { class: "m", text: "Run time" }), el("span", {}));
  const rows = runs.map(r => {
    const cb = el("input", { type: "checkbox", checked: S.compare.has(r.run_id), "aria-label": "Compare", onchange: ev => { ev.target.checked ? S.compare.add(r.run_id) : S.compare.delete(r.run_id); renderCompare(runs, metricKeys); } });
    return el("div", { class: "runrow" }, cb,
      el("span", { class: "t" }, `${(r.dataset || "?").replace(/^upload:/, "")}, ${r.method} (${ENGINE_NAME[r.engine] || r.engine})`, r.n_warnings ? el("span", { class: "pen", text: `  ${r.n_warnings} warnings` }) : null),
      el("span", { text: (r.created || "").replace("T", " ").slice(0, 16) }), el("span", { class: "m", text: fint(r.n_series) }),
      el("span", { class: "m", text: fmtM("SMAPE", r.summary.SMAPE && r.summary.SMAPE.mean) }), el("span", { class: "m", text: fmtM("MASE", r.summary.MASE && r.summary.MASE.mean) }),
      el("span", { class: "m", text: r.elapsed != null ? mmss(r.elapsed) : "—" }),
      el("button", { type: "button", class: "linkish", onclick: async () => { await loadRun(r.run_id); setView("work"); $("#sonuc").scrollIntoView(); } }, "Open"));
  });
  host.replaceChildren(el("section", { class: "sec" },
    el("h2", {}, "Past runs"),
    el("p", { class: "lead", text: `Runs on this computer (${S.history.runs_dir}). Tick the boxes to compare.` }),
    runs.length ? el("div", { class: "list tall" }, head, rows) : el("p", { class: "empty", text: "No runs yet. Start your first run in the workspace." }),
    el("div", { id: "compare" })));
  renderCompare(runs, metricKeys);
}
function renderCompare(runs, metricKeys) {
  const host = $("#compare"); if (!host) return;
  const sel = runs.filter(r => S.compare.has(r.run_id));
  if (sel.length < 2) { host.replaceChildren(el("p", { class: "empty", text: sel.length ? "Tick at least two runs to compare." : "" })); return; }
  const metrics = (S.models.metric_all || S.models.metric_defaults).filter(k => metricKeys.includes(k));
  const stat = S.cmpStat;
  const SIGNED = new Set(["ME", "PBias"]);                   // bias measures: closest to 0 is best
  const score = (k, v) => (SIGNED.has(k) ? Math.abs(v) : v);
  const best = {}; for (const k of metrics) { let b = null; for (const r of sel) { const v = r.summary[k] && r.summary[k][stat]; if (v != null && (b == null || score(k, v) < score(k, b))) b = v; } best[k] = b; }
  const sameData = new Set(sel.map(r => r.dataset)).size === 1;
  // UI review M5: a "best" marker only means something when the runs share the data and the horizon.
  const comparable = sameData && new Set(sel.map(r => r.horizon)).size === 1;
  const label = r => `${(r.dataset || "?").replace(/^upload:/, "")}, ${r.method} (${ENGINE_NAME[r.engine] || r.engine})`;
  const cm = metrics.includes(S.cmpMetric) ? S.cmpMetric : metrics[0];
  const vals = sel.map(r => r.summary[cm] ? r.summary[cm][stat] : null);
  const mx = Math.max(...vals.filter(v => v != null).map(Math.abs), 1e-12);
  host.replaceChildren(
    el("h3", { text: "Comparison" }),
    sameData ? null : el("p", { class: "note", text: "The selected runs use different datasets; comparing the measures directly may not be meaningful." }),
    new Set(sel.map(r => r.horizon)).size > 1 ? el("p", { class: "note", text: "The runs have different forecast horizons." }) : null,
    el("div", { class: "chartbar" }, el("div", { class: "seg", role: "group" }, [["mean", "Mean"], ["median", "Median"]].map(([k, l]) => el("button", { type: "button", "aria-pressed": String(stat === k), onclick: () => { S.cmpStat = k; renderCompare(runs, metricKeys); } }, l))),
      el("span", { class: "hint", text: comparable ? "The lowest value in each measure is highlighted (lower is better; for ME and Percent bias, closer to 0 is better)." : "No value is highlighted: the runs do not share the same data and horizon, so they cannot be ranked." })),
    el("div", { class: "tablewrap" }, el("table", { class: "t" }, el("thead", {}, el("tr", {}, el("th", { text: "Run" }), metrics.map(k => el("th", { text: (S.models.metrics[k] || { name: k }).name })))),
      el("tbody", {}, sel.map(r => el("tr", {}, el("td", { class: "name", text: label(r) }),
        metrics.map(k => { const v = r.summary[k] && r.summary[k][stat]; return el("td", { class: comparable && v != null && v === best[k] ? "best" : "", text: fmtM(k, v) }); })))))),
    el("div", { class: "chartbar", style: "margin-top:14px" }, el("span", { class: "hint", text: "Chart measure:" }),
      el("select", { onchange: ev => { S.cmpMetric = ev.target.value; renderCompare(runs, metricKeys); } }, metrics.map(k => el("option", { value: k, selected: k === cm }, (S.models.metrics[k] || { name: k }).name)))),
    el("div", { class: "bars" }, sel.map((r, i) => el("div", { class: "b" + (comparable && vals[i] != null && vals[i] === best[cm] ? " best" : "") },
      el("span", { text: label(r) }), el("div", {}, el("i", { style: `width:${vals[i] == null ? 0 : Math.max(0.5, Math.abs(vals[i]) / mx * 100)}%` })), el("span", { text: fmtM(cm, vals[i]) })))));
}

// ================================================================================================ series features
const FSETS = [["basic", "Basic", "Length, level, spread, share of zeros, ADI and CV² with the demand class, Box-Cox λ."],
  ["catch22", "catch22", "The 22 canonical time-series characteristics of Lubba et al. (2019), computed by the C++ library."],
  ["tsfeatures", "tsfeatures", "The tsfeatures set of the ranking study (Hyndman et al.): shifts, autocorrelations, entropy, Holt parameters, Hurst, unit-root tests, nonlinearity, lumpiness, stability and STL trend/seasonal strength, ported from R and checked against the study's values. Slower: about a second per long series."]];
const FSTAGE = { loading: "Reading data", basic: "Basic features", catch22: "catch22 features", tsfeatures: "tsfeatures" };
S.feat = { sets: new Set(["basic", "catch22"]), study: false, data: null, job: null, err: null, busy: false };
const featQuery = () => `dataset=${encodeURIComponent(S.sel.dataset)}&sets=${[...S.feat.sets].join(",")}&study=${S.feat.study ? 1 : 0}`;
function featuresNote() {
  return el("div", {}, el("h3", { text: "Series features" }),
    el("p", { class: "lead", text: "One row of descriptive features per series of the selected dataset, for describing the data and relating it to forecast accuracy." }),
    el("p", { text: "catch22 (Lubba et al., 2019, Data Mining and Knowledge Discovery 33) is a set of 22 features selected from the hctsa library for being informative and little redundant. Each is computed on the z-scored series." }),
    runIn("Ranking study values.", "The catch22 values stored with the ranking study came from an Rcatch22 build whose FFT was faulty (every twiddle factor equal to 1). Tick “Reproduce the study’s catch22 values” to get exactly those numbers; leave it off for correct catch22. Seven features differ between the two.", "caution"),
    el("p", { text: "Series shorter than 10 points, constant series and series with missing values get empty catch22 values." }));
}
async function renderFeatures() {
  const host = $("#view-features-body");
  const head = [el("h2", {}, "Series features"), el("p", { class: "lead", text: "Compute features for every series of the dataset chosen in step 1, look at their summary and download them." })];
  if (!S.sel.dataset) {
    host.replaceChildren(el("section", { class: "sec" }, head, el("p", { class: "empty", text: "Choose a dataset in the workspace first (step 1, Data)." })));
    return;
  }
  const ds = S.sel.dataset.replace(/^upload:/, "");
  const setBoxes = FSETS.map(([k, l, help]) => bindNote(el("label", { class: "chk" }, el("input", { type: "checkbox", checked: S.feat.sets.has(k),
    onchange: ev => { ev.target.checked ? S.feat.sets.add(k) : S.feat.sets.delete(k); loadFeatures(); } }), " " + l),
    () => el("div", {}, el("h3", { text: l }), el("p", { text: help }))));
  const study = bindNote(el("label", { class: "chk" }, el("input", { type: "checkbox", checked: S.feat.study, disabled: !S.feat.sets.has("catch22"),
    onchange: ev => { S.feat.study = ev.target.checked; loadFeatures(); } }), " Reproduce the study’s catch22 values"), featuresNote);
  host.replaceChildren(el("section", { class: "sec" }, head,
    el("p", {}, "Dataset: ", el("b", { text: ds })),
    el("div", { class: "toolbar" }, setBoxes, study,
      el("button", { type: "button", class: "btn", id: "feat-go", disabled: !S.feat.sets.size || S.feat.busy, onclick: startFeatures }, "Compute features")),
    el("div", { id: "feat-progress" }), el("div", { id: "feat-body" })));
  Note.base = featuresNote; Note.restore();
  await loadFeatures();
}
async function loadFeatures() {
  const btn = $("#feat-go"); if (btn) btn.disabled = !S.feat.sets.size || S.feat.busy;
  const chk = $$("#view-features-body label.chk input"); if (chk.length) chk[chk.length - 1].disabled = !S.feat.sets.has("catch22");
  S.feat.data = null; S.feat.err = null;
  if (S.feat.sets.size) {
    try { S.feat.data = await api("features?" + featQuery()); } catch (e) { S.feat.err = e.message; }
  }
  renderFeatureBody();
}
async function startFeatures() {
  try {
    S.feat.job = await api("features", { dataset: S.sel.dataset, sets: [...S.feat.sets], study_fft: S.feat.study });
    S.feat.busy = true; $("#feat-go").disabled = true; pollFeatures();
  } catch (e) { toast(e.message, true); }
}
async function pollFeatures() {
  const j = S.feat.job; if (!j) return;
  let v;
  try { v = await api("jobs/" + j.id); } catch (e) { S.feat.busy = false; toast(e.message, true); return; }
  S.feat.job = v;
  const host = $("#feat-progress");
  if (v.status === "running") {
    if (host) host.replaceChildren(el("p", { class: "hint" }, `${FSTAGE[v.stage] || v.stage}${v.total ? ` ${fint(v.done)}/${fint(v.total)}` : ""}, ${mmss(v.elapsed)} `,
      el("button", { type: "button", class: "linkish", onclick: () => api(`jobs/${v.id}/cancel`, {}).catch(() => {}) }, v.cancelling ? "Cancelling…" : "Cancel")));
    setTimeout(pollFeatures, 700);
    return;
  }
  S.feat.busy = false;
  if (host) host.replaceChildren(v.status === "error" ? el("p", { class: "note err", text: v.error }) :
    v.status === "cancelled" ? el("p", { class: "hint", text: "Cancelled." }) : el("p", { class: "hint", text: `Done in ${mmss(v.elapsed)}.` }));
  if (S.view === "features") loadFeatures();
}
function renderFeatureBody() {
  const host = $("#feat-body"); if (!host) return;
  const d = S.feat.data;
  if (!S.feat.sets.size) { host.replaceChildren(el("p", { class: "empty", text: "Tick at least one feature set." })); return; }
  if (S.feat.err) { host.replaceChildren(el("p", { class: "note err", text: S.feat.err })); return; }
  if (!d) { host.replaceChildren(el("p", { class: "empty", text: "Reading…" })); return; }
  if (d.status !== "ready") { host.replaceChildren(el("p", { class: "empty", text: "Not computed yet for this choice of feature sets. Press “Compute features”." })); return; }
  const byName = Object.fromEntries(d.summary.map(s => [s.name, s]));
  const sumRow = n => {
    const s = byName[n];
    if (s.levels) return el("tr", {}, el("td", { class: "name", text: n }), el("td", { text: fint(s.missing) }),
      el("td", { colspan: 6, text: Object.entries(s.levels).map(([k, c]) => `${k} ${fint(c)}`).join(", ") }));
    return el("tr", {}, el("td", { class: "name", text: n }), el("td", { text: fint(s.missing) }),
      ["mean", "sd", "min", "median", "max"].map(k => el("td", { text: fmt(s[k]) })), el("td", { text: fint(s.distinct) }));
  };
  const groups = FSETS.filter(([k]) => d.groups[k] && d.groups[k].length);
  const nPrev = Math.min(d.rows.length, 200);
  host.replaceChildren(
    el("p", {}, `${fint(d.n_series)} series, ${d.columns.length} features. Stored ${new Date(d.updated * 1000).toLocaleString()}. `,
      el("a", { href: `/api/features/file?${featQuery()}&t=${encodeURIComponent(TOKEN)}`, download: "" }, "Download CSV")),
    el("h3", { text: "Summary" }),
    el("div", { class: "tablewrap" }, el("table", { class: "t" },
      el("thead", {}, el("tr", {}, ["Feature", "Missing", "Mean", "SD", "Min", "Median", "Max", "Distinct"].map(h => el("th", { text: h })))),
      groups.map(([k, l]) => el("tbody", {}, el("tr", { class: "grp" }, el("th", { colspan: 8, text: l })), d.groups[k].map(sumRow))))),
    el("h3", { text: `Values (first ${fint(nPrev)} series)` }),
    el("div", { class: "tablewrap tall" }, el("table", { class: "t" },
      el("thead", {}, el("tr", {}, el("th", { text: "Series" }), d.columns.map(c => el("th", { text: c })))),
      el("tbody", {}, d.rows.slice(0, nPrev).map((r, i) => el("tr", {}, el("td", { class: "name", text: d.index[i] }),
        r.map(v => el("td", { text: typeof v === "number" ? fmt(v) : (v == null ? "—" : String(v)) }))))))),
    el("div", { id: "pca" }));
  loadPca();
}

// ------------------------------------------------------------------------------------------------ PCA of the features
// Palette: validated (dataviz validator, all-pairs, app surfaces) for at most three categorical slots in a scatter;
// further levels fold into "Other". Sequential: one blue ramp, light-to-dark on light, receding-to-bright on dark.
const PCA_CAT = { light: ["#2a78d6", "#eb6834", "#1baf7a"], dark: ["#3987e5", "#d95926", "#199e70"] };
const PCA_SEQ = { light: ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"], dark: ["#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"] };
const PCA_OTHER = "#898781";
const CAT_ORDER = ["smooth", "intermittent", "erratic", "lumpy"];
S.pca = { data: null, err: null, color: null, mode: "2d", ax: [0, 1], yaw: -0.6, pitch: 0.35, focus: null, arrows: false, pts: [], seq: 0 };
const cssVar = k => getComputedStyle(document.body).getPropertyValue(k).trim();
const isDark = () => { const c = cssVar("--white").replace("#", ""); if (c.length < 6) return false; return (parseInt(c.slice(0, 2), 16) + parseInt(c.slice(2, 4), 16) + parseInt(c.slice(4, 6), 16)) / 3 < 128; };
async function loadPca() {
  const host = $("#pca"); if (!host) return;
  if (S.pca.color == null) S.pca.color = S.feat.sets.has("basic") ? "feature:demand_class" : "";
  const seq = ++S.pca.seq;
  try {
    const d = await api(`features/pca?${featQuery()}&color=${encodeURIComponent(S.pca.color)}`);
    if (seq !== S.pca.seq) return;
    S.pca.data = d; S.pca.err = null;
  } catch (e) {
    if (seq !== S.pca.seq) return;
    if (S.pca.color) { S.pca.color = ""; toast(e.message, true); return loadPca(); }
    S.pca.data = null; S.pca.err = e.message;
  }
  renderPca();
}
function pcaColorOptions(d) {
  const o = d.options, opts = [el("option", { value: "" }, "None")];
  const cats = o.categorical.map(c => el("option", { value: "feature:" + c }, c));
  opts.push(el("optgroup", { label: "Feature" }, cats, o.numeric.map(c => el("option", { value: "feature:" + c }, c))));
  if (o.runs.length) {
    opts.push(el("optgroup", { label: "Forecast error of a run" }, o.runs.flatMap(r => r.metrics.map(m =>
      el("option", { value: `run:${r.run_id}:${m}` }, `${m}, ${r.method} (${ENGINE_NAME[r.engine] || r.engine}) ${(r.created || "").slice(0, 10)}`)))));
    const models = new Set(o.runs.map(r => r.method + "|" + r.engine));
    if (models.size >= 2) {
      const ms = [...new Set(o.runs.flatMap(r => r.metrics))];
      opts.push(el("optgroup", { label: "Most accurate model per series" }, ms.map(m => el("option", { value: "best:" + m }, `lowest ${m}`))));
    }
  }
  for (const g of opts) for (const op of (g.tagName === "OPTGROUP" ? [...g.children] : [g])) op.selected = op.value === S.pca.color;
  return opts;
}
// The colour of every point plus the legend, from the server's colour column.
function pcaColors(d) {
  const mode = isDark() ? "dark" : "light", c = d.color, n = d.names.length;
  if (!c) return { col: new Array(n).fill(cssVar("--ink") || "#12294b"), legend: null };
  if (c.kind === "cat") {
    const counts = {}; c.values.forEach(v => { if (v != null) counts[v] = (counts[v] || 0) + 1; });
    const order = c.models || CAT_ORDER;
    const levels = Object.keys(counts).sort((a, b) => { const ia = order.indexOf(a), ib = order.indexOf(b); return (ia < 0 ? 1e9 : ia) - (ib < 0 ? 1e9 : ib) || a.localeCompare(b); });
    const slot = {}; levels.slice(0, 3).forEach((l, i) => { slot[l] = PCA_CAT[mode][i]; });
    const other = levels.slice(3);
    const col = c.values.map(v => (v == null ? null : slot[v] || PCA_OTHER));
    const items = levels.slice(0, 3).map(l => ({ key: l, label: l, color: slot[l], n: counts[l] }));
    if (other.length) items.push({ key: "__other", label: "Other: " + other.join(", "), color: PCA_OTHER, n: other.reduce((s, l) => s + counts[l], 0), members: other });
    return { col, legend: { kind: "cat", title: c.label, items, missing: c.values.filter(v => v == null).length } };
  }
  const v = c.values, idx = v.map((x, i) => [x, i]).filter(([x]) => x != null && Number.isFinite(x)).sort((a, b) => a[0] - b[0]);
  const stops = PCA_SEQ[mode], rgb = h => [1, 3, 5].map(k => parseInt(h.slice(k, k + 2), 16));
  const ramp = t => { const p = t * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(p)), f = p - k, a = rgb(stops[k]), b = rgb(stops[k + 1]); return `rgb(${a.map((x, j) => Math.round(x + (b[j] - x) * f)).join(",")})`; };
  const col = new Array(n).fill(null);
  idx.forEach(([, i], r) => { col[i] = ramp(idx.length > 1 ? r / (idx.length - 1) : 0.5); });
  const q = p => (idx.length ? idx[Math.min(idx.length - 1, Math.round(p * (idx.length - 1)))][0] : null);
  return { col, legend: { kind: "num", title: c.label, stops, lo: q(0), mid: q(0.5), hi: q(1), missing: n - idx.length } };
}
function renderPca() {
  const host = $("#pca"); if (!host) return;
  const d = S.pca.data;
  const head = el("h3", { text: "Principal components" });
  if (S.pca.err) { host.replaceChildren(head, el("p", { class: "note err", text: S.pca.err })); return; }
  if (!d || d.status !== "ready") { host.replaceChildren(head, el("p", { class: "empty", text: "Reading…" })); return; }
  const pcs = d.explained.map((e, i) => `PC${i + 1}`);
  const axSel = k => el("select", { "aria-label": k ? "Vertical axis" : "Horizontal axis", onchange: ev => { S.pca.ax[k] = +ev.target.value; drawPca(); } },
    pcs.map((p, i) => el("option", { value: i, selected: S.pca.ax[k] === i }, p)));
  const seg = el("div", { class: "seg", role: "group", "aria-label": "View" }, [["2d", "2D"], ["3d", "3D"]].map(([k, l]) =>
    el("button", { type: "button", "aria-pressed": String(S.pca.mode === k), onclick: () => { S.pca.mode = k; renderPca(); } }, l)));
  const colorSel = el("select", { "aria-label": "Colour by", onchange: ev => { S.pca.color = ev.target.value; S.pca.focus = null; loadPca(); } }, pcaColorOptions(d));
  const ctl = el("div", { class: "chartbar" }, seg,
    S.pca.mode === "2d" ? [el("span", { class: "hint", text: "x" }), axSel(0), el("span", { class: "hint", text: "y" }), axSel(1),
      el("label", { class: "chk" }, el("input", { type: "checkbox", checked: S.pca.arrows, onchange: ev => { S.pca.arrows = ev.target.checked; drawPca(); } }), " Loadings")] :
      el("span", { class: "hint", text: "Drag, or use the arrow keys, to turn the cloud." }),
    el("span", { class: "hint", text: "Colour" }), colorSel);
  const canvas = el("canvas", { class: "pca-canvas", tabindex: 0, role: "img",
    "aria-label": `Scatter of the series on the principal components, ${d.names.length} points` });
  const tip = el("div", { class: "pca-tip", hidden: true });
  const wrap = el("div", { class: "pca-wrap" }, canvas, tip);
  bindPcaCanvas(canvas, tip);
  const top = (j, k = 6) => d.features.map((f, i) => [f, d.loadings[i][j]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, k);
  host.replaceChildren(head,
    el("p", { class: "hint", text: `Standardised features (correlation PCA), ${d.features.length} features used. Explained variance: ${d.explained.map((e, i) => `PC${i + 1} ${fpct(e, 1)}`).join(", ")}.` + (d.dropped.length ? ` Left out (constant or mostly missing): ${d.dropped.join(", ")}.` : "") }),
    ctl, wrap, el("div", { id: "pca-legend", class: "pca-legend" }),
    d.color && d.color.matched != null && d.color.matched < d.names.length ? el("p", { class: "note soft", text: `${d.names.length - d.color.matched} series have no value in that run (it covered fewer series or names differ); they are drawn hollow.` }) : null,
    el("h3", { text: "Strongest loadings" }),
    el("div", { class: "tablewrap" }, el("table", { class: "t" },
      el("thead", {}, el("tr", {}, pcs.map(p => el("th", { colspan: 2, text: p })))),
      el("tbody", {}, [0, 1, 2, 3, 4, 5].map(r => el("tr", {}, pcs.map((p, j) => { const t = top(j)[r]; return t ? [el("td", { class: "name", text: t[0] }), el("td", { text: nf2fix.format(t[1]) })] : [el("td"), el("td")]; })))))),
    el("p", {}, el("button", { type: "button", class: "btn ghost small", onclick: downloadScores }, "Download scores (CSV)")));
  S.pca.colors = pcaColors(d);
  renderPcaLegend();
  drawPca();
}
function renderPcaLegend() {
  const host = $("#pca-legend"), lg = S.pca.colors && S.pca.colors.legend; if (!host) return;
  if (!lg) { host.replaceChildren(); return; }
  const miss = lg.missing ? el("span", { class: "hint" }, el("i", { class: "sw hollow" }), ` no value ${fint(lg.missing)}`) : null;
  if (lg.kind === "cat") {
    host.replaceChildren(el("span", { class: "hint", text: lg.title + ":" }), lg.items.map(it => el("button", { type: "button", class: "lg", "aria-pressed": String(S.pca.focus === it.key),
      title: "Show this group on top, dim the rest", onclick: () => { S.pca.focus = S.pca.focus === it.key ? null : it.key; renderPcaLegend(); drawPca(); } },
      el("i", { class: "sw", style: `background:${it.color}` }), `${it.label} ${fint(it.n)}`)), miss);
  } else {
    host.replaceChildren(el("span", { class: "hint", text: lg.title + ":" }), el("span", { class: "hint", text: fmt(lg.lo) }),
      el("i", { class: "ramp", style: `background:linear-gradient(90deg,${lg.stops.join(",")})` }), el("span", { class: "hint", text: fmt(lg.hi) }),
      el("span", { class: "hint", text: `(colour by rank; median ${fmt(lg.mid)})` }), miss);
  }
}
function downloadScores() {
  const d = S.pca.data; if (!d) return;
  const q = s => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  const lines = ["series," + d.explained.map((e, i) => "PC" + (i + 1)).join(",") + (d.color ? "," + q(d.color.label) : "")];
  d.names.forEach((nm, i) => lines.push([q(nm), ...d.scores[i], ...(d.color ? [d.color.values[i] == null ? "" : q(String(d.color.values[i]))] : [])].join(",")));
  const a = el("a", { href: URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" })), download: `${S.sel.dataset.replace(/^upload:/, "")}_pca_scores.csv` });
  document.body.append(a); a.click(); a.remove();
}
function bindPcaCanvas(cv, tip) {
  let drag = null;
  cv.addEventListener("pointerdown", ev => { if (S.pca.mode !== "3d") return; drag = [ev.clientX, ev.clientY]; cv.setPointerCapture(ev.pointerId); });
  cv.addEventListener("pointerup", () => { drag = null; });
  cv.addEventListener("pointermove", ev => {
    if (drag) {
      S.pca.yaw += (ev.clientX - drag[0]) * 0.01; S.pca.pitch = Math.max(-1.5, Math.min(1.5, S.pca.pitch + (ev.clientY - drag[1]) * 0.01));
      drag = [ev.clientX, ev.clientY]; tip.hidden = true; drawPca(); return;
    }
    const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = null, bd = 81;
    for (const p of S.pca.pts) { const dd = (p.x - x) ** 2 + (p.y - y) ** 2; if (dd < bd) { bd = dd; best = p; } }
    if (!best) { tip.hidden = true; return; }
    const d = S.pca.data, i = best.i, c = d.color;
    tip.replaceChildren(el("b", { text: d.names[i] }), el("div", { text: d.scores[i].map((s, j) => `PC${j + 1} ${nf2fix.format(s)}`).join("  ") }),
      c ? el("div", { text: `${c.label}: ${c.values[i] == null ? "no value" : typeof c.values[i] === "number" ? fmt(c.values[i]) : c.values[i]}` }) : null);
    tip.hidden = false;
    tip.style.left = Math.min(x + 12, r.width - 200) + "px"; tip.style.top = (y + 12) + "px";
  });
  cv.addEventListener("pointerleave", () => { tip.hidden = true; });
  cv.addEventListener("keydown", ev => {
    if (S.pca.mode !== "3d") return;
    const k = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] }[ev.key];
    if (!k) return;
    ev.preventDefault(); S.pca.yaw += k[0]; S.pca.pitch = Math.max(-1.5, Math.min(1.5, S.pca.pitch + k[1])); drawPca();
  });
}
function drawPca() {
  const cv = $(".pca-canvas"), d = S.pca.data; if (!cv || !d || !S.pca.colors) return;
  const W = cv.parentElement.clientWidth, H = Math.round(Math.min(520, Math.max(300, W * 0.62))), dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const ink = cssVar("--ink"), muted = cssVar("--pencil"), grid = cssVar("--rule-soft") || cssVar("--rule"), axis = cssVar("--rule"), surf = cssVar("--white");
  g.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
  const { col, legend } = S.pca.colors;
  const focusOf = i => { if (!S.pca.focus || !legend || legend.kind !== "cat") return true; const v = d.color.values[i]; const it = legend.items.find(t => t.key === S.pca.focus); return it.members ? it.members.includes(v) : v === S.pca.focus; };
  let pts = [];
  if (S.pca.mode === "2d") {
    const [a, b] = S.pca.ax, L = 52, R = 16, T = 14, B = 40;
    const xs = d.scores.map(s => s[a]), ys = d.scores.map(s => s[b]);
    const pad = (lo, hi) => { const r = (hi - lo) || 1; return [lo - r * 0.04, hi + r * 0.04]; };
    const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
    const X = v => L + (v - x0) / (x1 - x0) * (W - L - R), Y = v => H - B - (v - y0) / (y1 - y0) * (H - T - B);
    g.strokeStyle = grid; g.lineWidth = 1; g.fillStyle = muted; g.textAlign = "right"; g.textBaseline = "middle";
    for (const t of niceTicks(y0, y1, 5)) { if (t < y0 || t > y1) continue; g.beginPath(); g.moveTo(L, Y(t) + 0.5); g.lineTo(W - R, Y(t) + 0.5); g.stroke(); g.fillText(nfSig.format(t), L - 6, Y(t)); }
    g.textAlign = "center"; g.textBaseline = "top";
    for (const t of niceTicks(x0, x1, 6)) { if (t < x0 || t > x1) continue; g.beginPath(); g.moveTo(X(t) + 0.5, T); g.lineTo(X(t) + 0.5, H - B); g.stroke(); g.fillText(nfSig.format(t), X(t), H - B + 6); }
    g.strokeStyle = axis; g.beginPath(); g.moveTo(L, H - B + 0.5); g.lineTo(W - R, H - B + 0.5); g.stroke();
    g.fillStyle = muted; g.fillText(`PC${a + 1} (${fpct(d.explained[a], 1)})`, (L + W - R) / 2, H - 16);
    g.save(); g.translate(14, (T + H - B) / 2); g.rotate(-Math.PI / 2); g.textBaseline = "middle"; g.fillText(`PC${b + 1} (${fpct(d.explained[b], 1)})`, 0, 0); g.restore();
    pts = d.scores.map((s, i) => ({ i, x: X(s[a]), y: Y(s[b]), z: 0 }));
    drawPoints(g, pts, col, focusOf, surf, muted);
    if (S.pca.arrows) {
      const cx = X(0), cy = Y(0), ext = 0.8 * Math.min(Math.max(Math.abs(x0), Math.abs(x1)) * (W - L - R) / (x1 - x0), Math.max(Math.abs(y0), Math.abs(y1)) * (H - T - B) / (y1 - y0));
      const ld = d.features.map((f, i) => [f, d.loadings[i][a], d.loadings[i][b]]).sort((p, q) => Math.hypot(q[1], q[2]) - Math.hypot(p[1], p[2])).slice(0, 6);
      const m = Math.max(...ld.map(t => Math.hypot(t[1], t[2])), 1e-9);
      g.strokeStyle = ink; g.fillStyle = ink; g.lineWidth = 1.5; g.textBaseline = "middle";
      for (const [f, u, v] of ld) {
        const ex = cx + u / m * ext, ey = cy - v / m * ext;
        g.beginPath(); g.moveTo(cx, cy); g.lineTo(ex, ey); g.stroke();
        g.textAlign = u >= 0 ? "left" : "right"; g.fillText(f, ex + (u >= 0 ? 4 : -4), ey);
      }
    }
  } else {
    const cy = Math.cos(S.pca.yaw), sy = Math.sin(S.pca.yaw), cp = Math.cos(S.pca.pitch), sp = Math.sin(S.pca.pitch);
    const rot = ([x, y, z]) => { const x1 = x * cy + z * sy, z1 = -x * sy + z * cy; return [x1, y * cp - z1 * sp, y * sp + z1 * cp]; };
    const ext = Math.max(...d.scores.flatMap(s => s.slice(0, 3).map(Math.abs)), 1e-9);
    const sc = Math.min(W, H) * 0.42 / ext, ox = W / 2, oy = H / 2;
    const P = v => { const r = rot(v); return [ox + r[0] * sc, oy - r[1] * sc, r[2]]; };
    g.lineWidth = 1; g.textAlign = "center"; g.textBaseline = "middle";
    for (let j = 0; j < 3; j++) {
      const e = [0, 0, 0]; e[j] = ext; const [ax2, ay2] = P(e), n = [0, 0, 0]; n[j] = -ext; const [bx, by] = P(n);
      g.strokeStyle = axis; g.beginPath(); g.moveTo(bx, by); g.lineTo(ax2, ay2); g.stroke();
      g.fillStyle = muted; g.fillText(`PC${j + 1} (${fpct(d.explained[j], 1)})`, ax2 + (ax2 - ox) * 0.06, ay2 + (ay2 - oy) * 0.06);
    }
    pts = d.scores.map((s, i) => { const [x, y, z] = P([s[0], s[1], s[2] || 0]); return { i, x, y, z }; }).sort((p, q) => p.z - q.z);
    drawPoints(g, pts, col, focusOf, surf, muted);
  }
  S.pca.pts = pts;
}
function drawPoints(g, pts, col, focusOf, surf, muted) {
  const pass = on => { for (const p of pts) {
    if (focusOf(p.i) !== on) continue;
    g.globalAlpha = on ? 1 : 0.15; g.beginPath(); g.arc(p.x, p.y, 4, 0, 2 * Math.PI);
    const c = col[p.i];
    if (c) { g.fillStyle = c; g.fill(); g.lineWidth = 1; g.strokeStyle = surf; g.stroke(); }
    else { g.lineWidth = 1.2; g.strokeStyle = muted; g.stroke(); }
  } };
  pass(false); pass(true); g.globalAlpha = 1;
}
window.addEventListener("resize", debounce(() => { if (S.view === "features") drawPca(); }, 250));

init();
})();
