// Forecast Studio, web edition: the page. Data are read and split here (core.js); the forecasts are computed in Web
// Workers running the WebAssembly build of tsfcore (worker.js), one worker per CPU core. Nothing is sent to a server.
import { FREQUENCIES, seasonalityFor, defaultLag, defaultHorizon, parseFile, prepare, seriesMetrics, summarize, meanRanks,
         METRICS, toCSV, DataError } from "./core.js";
import { METHODS } from "./methods.js";

const $ = s => document.querySelector(s);
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  e.append(...kids.filter(k => k != null));
  return e;
}
const SVG = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}
const fmt = v => Number.isFinite(v) ? v.toLocaleString("en-US", { maximumSignificantDigits: 4 }) : "–";
const secs = ms => ms < 950 ? `${Math.max(1, Math.round(ms))} ms` : `${(ms / 1000).toFixed(ms < 9950 ? 1 : 0)} s`;
const byId = Object.fromEntries(METHODS.map(m => [m.id, m]));

const PRESETS = {
  fast: ["naive", "snaive", "naive2", "ses", "theta", "ets", "comb_shd"],
  study: ["ses", "theta", "ets", "arima", "dhr_arima", "tbats", "pooled_regression"],     // the Monash archive's statistical baselines
  none: [],
};
const FREQ_LABEL = { "4_seconds": "Every 4 seconds", minutely: "Minutely", "10_minutes": "Every 10 minutes", "15_minutes": "Every 15 minutes",
  half_hourly: "Half-hourly", hourly: "Hourly", daily: "Daily", weekly: "Weekly", monthly: "Monthly", quarterly: "Quarterly",
  yearly: "Yearly, or no seasonality" };

const S = {
  data: null, prep: null, selected: new Set(PRESETS.fast), results: new Map(), stat: "mean", sort: "MASE",
  plotted: new Map(),          // method id -> colour slot 1..4; a method keeps its colour while it is shown
  userPlotted: false, series: 0, running: false, pool: null,
};

// ---------------------------------------------------------------------------------------------------------------------
// 1. Data
// ---------------------------------------------------------------------------------------------------------------------
function showDataError(msg) { const e = $("#data-error"); e.textContent = msg; e.hidden = !msg; }

function loadData(data) {
  S.data = data;
  const freq = data.frequency || "yearly";
  $("#frequency").value = freq;
  const shortest = Math.min(...data.series.map(s => s.length));
  $("#horizon").value = data.horizon || Math.max(1, Math.min(defaultHorizon(freq), Math.floor(shortest / 4)));
  $("#lag").value = defaultLag(seasonalityFor(freq));
  showDataError("");
  $("#data-panel").hidden = false;
  for (const id of ["#s-methods", "#s-run"]) $(id).hidden = false;
  split();
}

function split() {
  const d = S.data, lens = d.series.map(s => s.length).sort((a, b) => a - b);
  const missing = d.series.reduce((k, s) => k + s.reduce((j, v) => j + Number.isNaN(v), 0), 0);
  const fact = (k, v) => el("div", {}, el("dt", { text: k }), el("dd", { text: v }));
  $("#data-facts").replaceChildren(fact("Data set", d.name), fact("Series", d.series.length.toLocaleString("en-US")),
    fact("Length", lens[0] === lens.at(-1) ? String(lens[0]) : `${lens[0]} to ${lens.at(-1)}`), fact("Missing values", missing.toLocaleString("en-US")));
  clearResults();
  try {
    S.prep = prepare(d, Number($("#horizon").value), $("#missing").value);
    const p = S.prep, note = [`The last ${p.h} observations of each series are held out as the test part.`];
    if (p.excluded.length) note.push(`${p.excluded.length} series left out (${p.excluded[0].reason}).`);
    if (p.imputed) note.push(`${p.imputed} missing training values filled.`);
    if (p.testMissing) note.push(`${p.testMissing} missing test values are skipped by the measures.`);
    if (!d.frequency) note.push("The file does not state a frequency: check the choice above, it sets the seasonal period.");
    $("#split-note").textContent = note.join(" ");
    $("#split-note").classList.remove("err");
  } catch (e) {
    S.prep = null;
    $("#split-note").textContent = e.message;
    $("#split-note").classList.add("err");
  }
  updateRunButton();
}

async function readFile(file) {
  try {
    if (file.size > 300e6) throw new DataError("The file is larger than 300 MB, which is more than a browser tab can comfortably hold.");
    loadData(parseFile(file.name, await file.text()));
  } catch (e) { showDataError(e instanceof DataError ? e.message : `Could not read the file: ${e.message}`); }
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Methods
// ---------------------------------------------------------------------------------------------------------------------
function renderMethods() {
  const families = [...new Set(METHODS.map(m => m.family))];
  $("#method-list").replaceChildren(...families.map(f => el("fieldset", {}, el("legend", { text: f }),
    el("div", { class: "methods" }, ...METHODS.filter(m => m.family === f).map(m => el("label", { class: "method" },
      el("input", { type: "checkbox", value: m.id, checked: S.selected.has(m.id), onchange: ev => {
        ev.target.checked ? S.selected.add(m.id) : S.selected.delete(m.id); methodsChanged(); } }),
      el("b", {}, m.name, m.slow ? el("span", { class: "tag", text: "slower" }) : null),
      el("small", { text: m.summary })))))));
  methodsChanged();
}

function methodsChanged() {
  $("#method-count").textContent = `${S.selected.size} selected`;
  $("#lag-field").hidden = !S.selected.has("pooled_regression");
  updateRunButton();
}

function updateRunButton() {
  $("#btn-run").disabled = S.running || !S.prep || !S.selected.size;
  if (!S.running) $("#run-status").textContent = !S.prep ? "" : !S.selected.size ? "Select at least one method." :
    `${S.prep.train.length.toLocaleString("en-US")} series × ${S.selected.size} method${S.selected.size > 1 ? "s" : ""}`;
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Run: a pool of workers, each with its own instance of the forecasting core
// ---------------------------------------------------------------------------------------------------------------------
class Pool {
  constructor(size) { this.size = size; this.workers = []; this.waiting = new Map(); this.seq = 0; this.version = ""; }
  start() {
    return Promise.all(Array.from({ length: this.size }, () => new Promise((resolve, reject) => {
      const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
      w.onerror = ev => reject(new Error(ev.message || "The forecasting worker could not be started."));
      w.onmessage = ev => {
        const m = ev.data;
        if ("ready" in m) { if (m.ready) { this.version = m.version; resolve(); } else reject(new Error(m.error)); return; }
        const p = this.waiting.get(m.id); this.waiting.delete(m.id);
        if (p) m.ok ? p.resolve(m) : p.reject(new Error(m.error));
      };
      this.workers.push(w);
    })));
  }
  call(w, msg) { return new Promise((resolve, reject) => { const id = ++this.seq; this.waiting.set(id, { resolve, reject }); w.postMessage({ id, ...msg }); }); }
  stop() {
    for (const w of this.workers) w.terminate();
    for (const p of this.waiting.values()) p.reject(new Stopped());
    this.waiting.clear(); this.workers = [];
  }
}
class Stopped extends Error {}

async function forecastLocal(pool, method, prep, periods, onProgress) {
  const n = prep.train.length, h = prep.h, out = new Float64Array(n * h), flags = new Int32Array(n);
  let next = 0, done = 0, chunk = byId[method].slow ? 1 : 4, firstMsg = "";
  await Promise.all(pool.workers.map(async w => {
    while (next < n) {
      const a = next, b = Math.min(n, a + chunk); next = b;
      const t = performance.now();
      const r = await pool.call(w, { kind: "local", method, h, periods, series: prep.train.slice(a, b) });
      out.set(r.forecasts, a * h); flags.set(r.flags, a);
      if (!firstMsg && r.msg) firstMsg = r.msg;
      done += b - a;
      chunk = Math.max(1, Math.min(64, Math.round(200 / Math.max((performance.now() - t) / (b - a), 0.01))));   // about 0.2 s per request
      onProgress(done / n);
    }
  }));
  const fallbacks = flags.reduce((k, v) => k + v, 0);
  return { forecasts: out, fallbacks, note: fallbacks ? `${fallbacks} series fell back to the seasonal naive forecast (${firstMsg}).` : "" };
}

async function forecastGlobal(pool, prep, lag) {
  const short = prep.train.filter(s => s.length <= lag).length;
  if (short) throw new Error(`${short} series are not longer than the ${lag} lags; lower the number of lags.`);
  const r = await pool.call(pool.workers[0], { kind: "global", lag, h: prep.h, series: prep.train });
  return { forecasts: r.forecasts, fallbacks: 0, note: r.aliased.length ? `${r.aliased.length} linearly dependent lags were given a zero coefficient, as R's glm does.` : "" };
}

async function run() {
  const prep = S.prep, periods = seasonalityFor($("#frequency").value), lag = Number($("#lag").value);
  const todo = METHODS.filter(m => S.selected.has(m.id));
  S.running = true; clearResults(); updateRunButton();
  $("#btn-cancel").hidden = false; $("#run-error").hidden = true;
  const rows = new Map(todo.map(m => [m.id, el("li", {}, el("span", { class: "name", text: m.name }),
    el("span", { class: "bar" }, el("i")), el("span", { class: "t", text: "waiting" }))]));
  $("#progress").replaceChildren(...rows.values());
  const t0 = performance.now();
  try {
    $("#run-status").textContent = "Starting the forecasting core…";
    S.pool = new Pool(Math.max(1, Math.min(navigator.hardwareConcurrency || 2, 8, prep.train.length)));
    await S.pool.start();
    $("#core-version").textContent = `Forecasting core: ${S.pool.version}, ${S.pool.size} worker${S.pool.size > 1 ? "s" : ""} on this device.`;
    for (const [k, m] of todo.entries()) {
      const row = rows.get(m.id), bar = row.querySelector("i"), tcell = row.querySelector(".t"), t = performance.now();
      $("#run-status").textContent = `Method ${k + 1} of ${todo.length}: ${m.name}`;
      tcell.textContent = "running";
      try {
        const r = m.task === "global" ? await forecastGlobal(S.pool, prep, lag)
          : await forecastLocal(S.pool, m.id, prep, periods, f => { bar.style.width = `${(f * 100).toFixed(1)}%`; });
        const ms = performance.now() - t;
        const perSeries = prep.test.map((a, i) => seriesMetrics(r.forecasts.subarray(i * prep.h, (i + 1) * prep.h), a, prep.train[i], periods));
        S.results.set(m.id, { ...r, perSeries, summary: summarize(perSeries), ms });
        bar.style.width = "100%"; tcell.textContent = secs(ms);
        renderResults();
      } catch (e) {
        if (e instanceof Stopped) throw e;
        tcell.textContent = "failed"; tcell.title = e.message;
        const box = $("#run-error"); box.hidden = false; box.textContent = `${m.name}: ${e.message}`;
      }
    }
    $("#run-status").textContent = `Finished in ${secs(performance.now() - t0)}.`;
  } catch (e) {
    if (e instanceof Stopped) $("#run-status").textContent = "Stopped. Finished methods are kept below.";
    else {
      $("#run-status").textContent = "";
      const box = $("#run-error"); box.hidden = false;
      box.textContent = `The forecasting core could not start in this browser (${e.message}). A current version of Chrome, Edge, Firefox or Safari is needed.`;
    }
  } finally {
    if (S.pool) S.pool.stop();
    S.pool = null; S.running = false; $("#btn-cancel").hidden = true;
    const status = $("#run-status").textContent;
    updateRunButton();
    if (status) $("#run-status").textContent = status;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. Results
// ---------------------------------------------------------------------------------------------------------------------
function clearResults() {
  S.results.clear(); S.plotted.clear(); S.userPlotted = false; S.series = 0;
  $("#s-results").hidden = true; $("#progress").replaceChildren(); $("#run-error").hidden = true;
}

function renderResults() {
  const ids = [...S.results.keys()];
  if (!ids.length) return;
  $("#s-results").hidden = false;
  const stat = S.stat, val = (id, key) => S.results.get(id).summary[key][stat];
  const ranks = ids.length > 1 ? meanRanks(Object.fromEntries(ids.map(id => [id, S.results.get(id).perSeries])), "MASE") : null;
  const cols = [...METRICS.map(m => ({ key: m.key, label: m.label, title: m.note, get: id => val(id, m.key) })),
    ...(ranks ? [{ key: "rank", label: "Mean rank", title: `Mean rank by MASE over ${ranks.series} series (1 = best).`, get: id => ranks.ranks[id] }] : [])];
  const sortCol = cols.find(c => c.key === S.sort) || cols[0];
  const order = ids.slice().sort((a, b) => (Number.isNaN(sortCol.get(a)) ? Infinity : sortCol.get(a)) - (Number.isNaN(sortCol.get(b)) ? Infinity : sortCol.get(b)));
  const best = Object.fromEntries(cols.map(c => [c.key, Math.min(...ids.map(c.get).filter(Number.isFinite))]));
  $("#summary").replaceChildren(
    el("thead", {}, el("tr", {}, el("th", { scope: "col", text: "Method" }),
      ...cols.map(c => el("th", { scope: "col", title: c.title, "aria-sort": c === sortCol ? "ascending" : null },
        el("button", { type: "button", text: c.label, onclick: () => { S.sort = c.key; renderResults(); } }))),
      el("th", { scope: "col", text: "Fallbacks" }), el("th", { scope: "col", text: "Time" }))),
    el("tbody", {}, ...order.map(id => { const r = S.results.get(id); return el("tr", {}, el("td", { text: byId[id].name }),
      ...cols.map(c => el("td", { class: ids.length > 1 && c.get(id) === best[c.key] ? "best" : null, text: fmt(c.get(id)) })),
      el("td", { class: r.fallbacks ? null : "muted", text: String(r.fallbacks) }), el("td", { class: "muted", text: secs(r.ms) })); })));

  const notes = [];
  const undefinedCols = METRICS.filter(m => ids.some(id => S.results.get(id).summary[m.key].excluded)).map(m => m.label);
  if (undefinedCols.length) notes.push(`${undefinedCols.join(", ")}: undefined on some series (for example a zero denominator); those series are left out of that column.`);
  for (const id of ids) if (S.results.get(id).note) notes.push(`${byId[id].name}: ${S.results.get(id).note}`);
  $("#result-notes").textContent = notes.join(" ");
  $("#result-notes").hidden = !notes.length;

  if (!S.userPlotted) {                       // until the reader chooses: the three methods with the lowest mean MASE
    const top = ids.slice().sort((a, b) => S.results.get(a).summary.MASE.mean - S.results.get(b).summary.MASE.mean).slice(0, 3);
    for (const id of [...S.plotted.keys()]) if (!top.includes(id)) S.plotted.delete(id);
    for (const id of top) if (!S.plotted.has(id)) S.plotted.set(id, [1, 2, 3, 4].find(k => ![...S.plotted.values()].includes(k)));
  }
  const sel = $("#series");
  if (sel.options.length !== S.prep.names.length) sel.replaceChildren(...S.prep.names.map((nm, i) => el("option", { value: i, text: nm })));
  sel.value = S.series;
  renderChips(); renderChart();
}

function renderChips() {
  const full = S.plotted.size >= 4;
  $("#chips").replaceChildren(...[...S.results.keys()].map(id => {
    const slot = S.plotted.get(id);
    return el("button", { type: "button", class: "chip", "aria-pressed": slot ? "true" : "false", disabled: !slot && full, style: slot ? `--k: var(--s${slot})` : null,
      onclick: () => {
        S.userPlotted = true;
        if (slot) S.plotted.delete(id);
        else S.plotted.set(id, [1, 2, 3, 4].find(k => ![...S.plotted.values()].includes(k)));
        renderChips(); renderChart();
      } }, el("span", { class: "key" }), byId[id].name);
  }));
}

function niceTicks(lo, hi, n = 4) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(s => s >= raw);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function renderChart() {
  const host = $("#chart"), p = S.prep, i = S.series, h = p.h, tr = p.train[i], te = p.test[i];
  const shown = Math.min(tr.length, Math.max(4 * h, 60)), first = tr.length - shown, total = shown + h;
  const lines = [...S.plotted.entries()].sort((a, b) => a[1] - b[1]).map(([id, slot]) => ({ id, slot, y: S.results.get(id).forecasts.subarray(i * h, (i + 1) * h) }));
  const all = [...tr.subarray(first), ...te, ...lines.flatMap(l => [...l.y])].filter(Number.isFinite);
  const ticks = niceTicks(Math.min(...all), Math.max(...all)), y0 = Math.min(ticks[0], ...all), y1 = Math.max(ticks.at(-1), ...all);
  const W = Math.max(300, host.clientWidth), H = W < 520 ? 250 : 320, L = 54, R = 12, T = 10, B = 28;
  const x = k => L + (total === 1 ? 0 : k / (total - 1)) * (W - L - R), y = v => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B);
  const path = (vals, k0) => { let d = "", pen = false; vals.forEach((v, k) => { if (!Number.isFinite(v)) { pen = false; return; }
    d += `${pen ? "L" : "M"}${x(k0 + k).toFixed(1)},${y(v).toFixed(1)}`; pen = true; }); return d; };

  const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Series ${p.names[i]}: history, held-out values and forecasts over ${h} steps` });
  g.append(svg("rect", { class: "band", x: x(shown) - (x(1) - x(0)) / 2, y: T, width: Math.max(2, x(total - 1) - x(shown) + (x(1) - x(0)) / 2), height: H - T - B }));
  for (const t of ticks) {
    g.append(svg("line", { class: "grid", x1: L, x2: W - R, y1: y(t), y2: y(t) }));
    g.append(svg("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end" }, fmt(t)));
  }
  const xt = niceTicks(first + 1, first + total, W < 520 ? 3 : 6).filter(v => Number.isInteger(v) && v >= first + 1 && v <= first + total);
  for (const t of xt) g.append(svg("text", { x: x(t - first - 1), y: H - 8, "text-anchor": "middle" }, t.toLocaleString("en-US")));
  g.append(svg("path", { class: "ln", d: path([...tr.subarray(first)], 0), stroke: "var(--history)" }));
  g.append(svg("path", { class: "ln", d: path([tr.at(-1), ...te], shown - 1), stroke: "var(--actual)" }));
  for (const l of lines) {
    g.append(svg("path", { class: "ln", d: path([...l.y], shown), stroke: `var(--s${l.slot})` }));
    if (h === 1) g.append(svg("circle", { cx: x(shown), cy: y(l.y[0]), r: 4, fill: `var(--s${l.slot})`, stroke: "var(--white)", "stroke-width": 2 }));
  }
  const cross = svg("line", { class: "cross", y1: T, y2: H - B, visibility: "hidden" });
  g.append(cross);
  host.replaceChildren(g);

  const key = (c, t) => el("span", {}, el("span", { class: "key", style: `--k: ${c}` }), t);
  $("#legend").replaceChildren(key("var(--history)", "History"), key("var(--actual)", "Held-out values"), ...lines.map(l => key(`var(--s${l.slot})`, byId[l.id].name)));
  $("#chart-note").textContent = `Observation number on the horizontal axis; the shaded band is the forecast horizon (${h} steps)` +
    (first ? `. The first ${first.toLocaleString("en-US")} observations are not shown.` : ".");

  const tip = $("#tip");
  const move = ev => {
    const box = g.getBoundingClientRect(), px = (ev.clientX - box.left) / box.width * W;
    const k = Math.max(0, Math.min(total - 1, Math.round((px - L) / ((W - L - R) / Math.max(1, total - 1)))));
    cross.setAttribute("x1", x(k)); cross.setAttribute("x2", x(k)); cross.setAttribute("visibility", "visible");
    const line = (c, t, v) => el("span", {}, el("span", { class: "key", style: `--k: ${c}` }), t, el("em", { text: fmt(v) }));
    tip.replaceChildren(el("b", { text: `Observation ${(first + k + 1).toLocaleString("en-US")}` + (k >= shown ? ` · step ${k - shown + 1} of ${h}` : "") }),
      ...(k < shown ? [line("var(--history)", "History", tr[first + k])]
        : [line("var(--actual)", "Held-out value", te[k - shown]), ...lines.map(l => line(`var(--s${l.slot})`, byId[l.id].name, l.y[k - shown]))]));
    tip.hidden = false;
    const tb = tip.getBoundingClientRect();
    tip.style.left = `${Math.max(8, Math.min(window.innerWidth - tb.width - 8, ev.clientX + 14))}px`;
    tip.style.top = `${Math.max(8, ev.clientY - tb.height - 12)}px`;
  };
  g.addEventListener("pointermove", move);
  g.addEventListener("pointerdown", move);
  g.addEventListener("pointerleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
}

function download(kind) {
  const p = S.prep, ids = [...S.results.keys()], keys = METRICS.map(m => m.key);
  let rows;
  if (kind === "summary") rows = [["method", "statistic", ...keys, "fallbacks", "seconds"],
    ...ids.flatMap(id => ["mean", "median"].map(st => [id, st, ...keys.map(k => S.results.get(id).summary[k][st]), S.results.get(id).fallbacks, S.results.get(id).ms / 1000]))];
  else if (kind === "errors") rows = [["method", "series", ...keys],
    ...ids.flatMap(id => S.results.get(id).perSeries.map((r, i) => [id, p.names[i], ...keys.map(k => r[k])]))];
  else rows = [["method", "series", ...Array.from({ length: p.h }, (_, k) => `h${k + 1}`)],
    ...p.names.map((nm, i) => ["actual", nm, ...p.test[i]]),
    ...ids.flatMap(id => p.names.map((nm, i) => [id, nm, ...S.results.get(id).forecasts.subarray(i * p.h, (i + 1) * p.h)]))];
  const a = el("a", { href: URL.createObjectURL(new Blob([toCSV(rows)], { type: "text/csv" })), download: `${S.data.name}_${kind}.csv` });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------------------------------------------------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------------------------------------------------
$("#frequency").replaceChildren(...FREQUENCIES.slice().reverse().map(f => el("option", { value: f, text: FREQ_LABEL[f] })));
renderMethods();
$("#btn-sample").addEventListener("click", async () => {
  try {
    const r = await fetch(new URL("./samples/nn5_weekly.tsf", import.meta.url));
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    loadData(parseFile("nn5_weekly.tsf", await r.text()));
  } catch (e) { showDataError(`Could not load the sample: ${e.message}`); }
});
$("#file").addEventListener("change", ev => { if (ev.target.files[0]) readFile(ev.target.files[0]); ev.target.value = ""; });
$("#frequency").addEventListener("change", () => { $("#lag").value = defaultLag(seasonalityFor($("#frequency").value)); split(); });
$("#horizon").addEventListener("change", split);
$("#missing").addEventListener("change", split);
for (const b of document.querySelectorAll("[data-preset]")) b.addEventListener("click", () => {
  S.selected = new Set(PRESETS[b.dataset.preset]);
  for (const c of document.querySelectorAll("#method-list input")) c.checked = S.selected.has(c.value);
  methodsChanged();
});
$("#btn-run").addEventListener("click", run);
$("#btn-cancel").addEventListener("click", () => { if (S.pool) S.pool.stop(); });
for (const b of document.querySelectorAll("[data-stat]")) b.addEventListener("click", () => {
  S.stat = b.dataset.stat;
  for (const o of document.querySelectorAll("[data-stat]")) o.setAttribute("aria-pressed", String(o === b));
  renderResults();
});
for (const b of document.querySelectorAll("[data-download]")) b.addEventListener("click", () => download(b.dataset.download));
$("#series").addEventListener("change", ev => { S.series = Number(ev.target.value); renderChart(); });
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (S.results.size && !$("#s-results").hidden) renderChart(); }, 150); });

let dragDepth = 0;
window.addEventListener("dragenter", ev => { if (ev.dataTransfer?.types.includes("Files")) { dragDepth++; document.body.classList.add("dragging"); } });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); } });
window.addEventListener("dragover", ev => ev.preventDefault());
window.addEventListener("drop", ev => {
  ev.preventDefault(); dragDepth = 0; document.body.classList.remove("dragging");
  if (ev.dataTransfer?.files[0]) readFile(ev.dataTransfer.files[0]);
});
