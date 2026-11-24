// Forecast Studio in the browser: the page logic. The engine (tsfcore, WebAssembly) runs in worker.js.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const SVG = "http://www.w3.org/2000/svg";
  const V = "?v=__BUILD__";   // the commit id once published (see index.html): every file is fetched with it
  const worker = new Worker("worker.js" + V);

  let engineReady = false, running = false;
  let data = null;         // { name, id, integer, stats, header } of the file the worker holds
  let examples = [];
  let learn = null;        // learn.json
  let result = null;       // the last run: files, manifest, rows, ...
  let pendingRun = null;   // timing of the run in progress
  let seriesToken = 0;

  // ------------------------------------------------------------------------------------------- small helpers
  function el(tag, attrs, ...kids) {
    const node = document.createElement(tag);
    for (const k in attrs || {}) {
      if (k === "text") node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
    for (const kid of kids) if (kid) node.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    return node;
  }
  function svg(tag, attrs, ...kids) {
    const node = document.createElementNS(SVG, tag);
    for (const k in attrs || {}) {
      if (k === "text") node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
    for (const kid of kids) if (kid) node.appendChild(kid);
    return node;
  }
  function fmt(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    if (!Number.isFinite(v)) return String(v);
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1e6 || a < 1e-3) return v.toExponential(2);
    return Number(v.toPrecision(4)).toString();
  }
  function seconds(s) {
    if (s < 1) return "under a second";
    if (s < 90) return "about " + Math.round(s) + " s";
    if (s < 5400) return "about " + Math.round(s / 60) + " min";
    return "about " + (s / 3600).toFixed(1) + " h";
  }
  function status(text, isError) {
    $("status").textContent = text;
    $("status").className = "status " + (isError ? "error" : "note");
  }
  function refresh() {
    $("run").disabled = !(engineReady && data && !running);
    // say why the button is off
    const why = running ? "" : !engineReady ? "The button turns on when the engine has loaded." :
      !data ? (pendingLoad ? "The button turns on when the data has been read." : "Choose an example dataset or a .tsf file first.") : "";
    $("runWhy").textContent = why;
    $("runWhy").hidden = !why;
  }

  function parseCsv(text) {
    const rows = [];
    let row = [], cell = "", quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
        else if (c === '"') quoted = false;
        else cell += c;
      } else if (c === '"') quoted = true;
      else if (c === ",") { row.push(cell); cell = ""; }
      else if (c === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
      else if (c !== "\r") cell += c;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    return rows;
  }
  const num = (s) => (s === "" ? NaN : Number(s));

  // "Nice" axis ticks covering [lo, hi]
  function ticks(lo, hi, count) {
    if (!(hi > lo)) { hi = lo + 1; }
    const raw = (hi - lo) / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
    return { values: out, step };
  }

  // ------------------------------------------------------------------------------------------- tabs
  function showTab(name) {
    for (const t of ["run", "learn"]) {
      $("tab-" + t).setAttribute("aria-selected", String(t === name));
      $("panel-" + t).hidden = t !== name;
    }
    if (location.hash !== "#" + name) history.replaceState(null, "", name === "learn" ? "#learn" : location.pathname + location.search);
  }
  $("tab-run").addEventListener("click", () => showTab("run"));
  $("tab-learn").addEventListener("click", () => showTab("learn"));
  if (location.hash === "#learn") showTab("learn");

  // ------------------------------------------------------------------------------------------- data
  function sendFile(name, bytes, info) {
    data = null;
    refresh();
    $("dataInfo").textContent = "Reading " + name + "…";
    const mine = (pendingLoad = { name, info, mb: bytes.byteLength / 1048576 });
    worker.postMessage({ type: "load", name, bytes }, [bytes]);
    refresh();
    setTimeout(() => {   // no answer: the engine did not start, or the browser holds files of an older version of the page
      if (pendingLoad === mine && !data && engineReady) {
        $("dataInfo").textContent = "The data could not be read. Reload the page; if that does not help, clear this site's data in the browser.";
        $("dataInfo").className = "note error";
      }
    }, 20000);
  }
  let pendingLoad = null;

  function onLoaded(msg) {
    const info = pendingLoad.info, st = msg.stats;
    data = { name: msg.name, id: info ? info.id : msg.name.replace(/\.tsf$/i, ""), integer: !!(info && info.integer_conversion),
             stats: st, header: msg.header };
    const len = st.len_min === st.len_max ? st.len_min + " observations each" : st.len_min + "–" + st.len_max + " observations";
    $("dataInfo").textContent = (info ? info.title + ": " : "") + st.n_series + " " + (st.frequency || "") + " series, " + len + ", " +
      pendingLoad.mb.toFixed(2) + " MB" + (st.missing_share > 0 ? ", " + (100 * st.missing_share).toFixed(1) + "% missing" : "") +
      (data.integer ? "; forecasts are rounded to whole numbers" : "");
    const h = info && info.horizon ? info.horizon : st.horizon;
    $("horizon").value = h || "";
    if (!h) $("dataInfo").textContent += ". The file names no horizon: enter one below.";
    refresh();
  }

  $("file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    $("example").value = "";
    sendFile(f.name, await f.arrayBuffer(), null);
  });

  async function loadExample(id) {
    const info = examples.find((x) => x.id === id);
    if (!info) return;
    $("dataInfo").textContent = "Loading " + info.id + "…";
    try {
      const r = await fetch("data/" + info.file);
      if (!r.ok) throw new Error("HTTP " + r.status);
      $("file").value = "";
      sendFile(info.file, await r.arrayBuffer(), info);
    } catch (err) {
      $("dataInfo").textContent = "The example could not be loaded (" + err.message + "); choose a .tsf file instead.";
    }
  }
  $("example").addEventListener("change", (e) => loadExample(e.target.value));

  const examplesReady = fetch("data/index.json" + V).then((r) => (r.ok ? r.json() : [])).catch(() => []).then((list) => {
    examples = list;
    const sel = $("example");
    sel.textContent = "";
    sel.appendChild(el("option", { value: "", text: list.length ? "choose…" : "no examples are published with this page" }));
    for (const x of list) sel.appendChild(el("option", { value: x.id, text: x.id + " – " + x.title + " (" + x.n_series + " series)" }));
  });

  // ------------------------------------------------------------------------------------------- methods + Learn tab
  function methodNote() {
    const m = learn && learn.models.find((x) => x.id === $("method").value);
    const p = $("methodInfo");
    p.textContent = "";
    if (!m) return;
    p.appendChild(document.createTextNode(m.summary + " "));
    const a = el("a", { href: "#learn", text: "More in Learn" });
    a.addEventListener("click", (e) => {
      e.preventDefault();
      showTab("learn");
      const d = $("learn-" + m.id);
      if (d) { d.open = true; d.scrollIntoView({ block: "start" }); }
    });
    p.appendChild(a);
  }
  $("method").addEventListener("change", methodNote);

  function refLine(key) {
    const r = learn.references[key];
    const li = el("li", { id: "ref-" + key });
    li.appendChild(document.createTextNode(r.text + " "));
    if (r.url) li.appendChild(el("a", { href: r.url, target: "_blank", rel: "noopener", text: "link" }));
    li.appendChild(el("span", { class: "tag", text: r.verified ? "checked" : "not checked" }));
    return li;
  }

  function buildLearn() {
    const order = Object.keys(learn.references);
    const cite = (keys) => {
      const span = el("span", {});
      keys.forEach((k, i) => {
        if (i) span.appendChild(document.createTextNode(", "));
        span.appendChild(el("a", { href: "#ref-" + k, text: "[" + (order.indexOf(k) + 1) + "]" }));
      });
      return span;
    };
    const models = $("learnModels");
    models.textContent = "";
    for (const g of learn.groups) {
      models.appendChild(el("h3", { text: g.name }));
      models.appendChild(el("p", { class: "note", style: "margin-top:0", text: g.text }));
      for (const m of learn.models.filter((x) => x.group === g.id)) {
        const d = el("details", { id: "learn-" + m.id });
        const s = el("summary", {});
        s.appendChild(document.createTextNode(m.name + " "));
        s.appendChild(el("span", { text: "(" + m.id + ")" }));
        d.appendChild(s);
        const dl = el("dl", {});
        const add = (k, v) => { dl.appendChild(el("dt", { text: k })); dl.appendChild(typeof v === "string" ? el("dd", { text: v }) : el("dd", {}, v)); };
        add("In short", m.summary);
        add("How it works", m.how);
        add("When to use it", m.use);
        add("Watch for", m.caution);
        if (m.refs.length) add("References", cite(m.refs));
        d.appendChild(dl);
        models.appendChild(d);
      }
    }
    const metrics = $("learnMetrics");
    metrics.textContent = "";
    for (const m of learn.metrics) {
      const d = el("details", { id: "learn-metric-" + m.id });
      const s = el("summary", {});
      s.appendChild(document.createTextNode(m.name + " "));
      s.appendChild(el("span", { text: "(" + m.unit + ")" }));
      d.appendChild(s);
      const dl = el("dl", {});
      dl.appendChild(el("dt", { text: "Definition" }));
      dl.appendChild(el("dd", { text: m.text }));
      dl.appendChild(el("dt", { text: "Watch for" }));
      dl.appendChild(el("dd", { text: m.caution }));
      d.appendChild(dl);
      metrics.appendChild(d);
    }
    const p = el("p", { class: "note" });
    p.appendChild(document.createTextNode("Sources of the definitions: "));
    p.appendChild(cite(learn.metric_refs));
    metrics.appendChild(p);
    const refs = $("learnRefs");
    refs.textContent = "";
    for (const k of order) refs.appendChild(refLine(k));
  }

  const learnReady = fetch("learn.json" + V).then((r) => (r.ok ? r.json() : null)).catch(() => null).then((j) => {
    learn = j;
    const sel = $("method");
    sel.textContent = "";
    if (!learn) { sel.appendChild(el("option", { value: "ets", text: "ets" })); return; }
    for (const g of learn.groups) {
      const og = el("optgroup", { label: g.name });
      for (const m of learn.models.filter((x) => x.group === g.id)) og.appendChild(el("option", { value: m.id, text: m.name }));
      sel.appendChild(og);
    }
    sel.value = "ets";
    methodNote();
    buildLearn();
  });

  // ------------------------------------------------------------------------------------------- run
  const PILOT = 5;   // series timed first, to estimate the full run

  $("run").addEventListener("click", () => {
    const config = ["method=" + $("method").value, "missing_policy=" + $("missing").value, "threads=1", "eol=lf",
                    "dataset_id=" + data.id, "integer_conversion=" + (data.integer ? 1 : 0)];
    if (learn) config.push("metrics=" + learn.metrics.map((x) => x.id).join(","));
    if ($("horizon").value) config.push("horizon=" + parseInt($("horizon").value, 10));
    const limit = $("maxSeries").value ? Math.min(parseInt($("maxSeries").value, 10), data.stats.n_series) : data.stats.n_series;
    if ($("maxSeries").value) config.push("max_series=" + limit);
    const pilot = limit > 4 * PILOT ? PILOT : 0;
    running = true;
    refresh();
    pendingRun = { limit, started: performance.now(), estimate: null, timer: null };
    $("progress").hidden = false;
    $("progress").firstElementChild.style.width = "0";
    status(pilot ? "Timing " + pilot + " series to estimate the run…" : "Running " + limit + " series…");
    worker.postMessage({ type: "run", config, pilot });
  });

  function onEstimate(msg) {
    const r = pendingRun;
    r.estimate = (msg.seconds / msg.series) * r.limit;
    r.fullStarted = performance.now();
    const tick = () => {
      const done = (performance.now() - r.fullStarted) / 1000;
      status("Estimated time: " + seconds(r.estimate) + " for " + r.limit + " series (from " + msg.series + " timed series). Running: " +
             Math.round(done) + " s.");
      $("progress").firstElementChild.style.width = Math.min(97, (100 * done) / Math.max(r.estimate, 0.2)) + "%";
    };
    tick();
    r.timer = setInterval(tick, 500);
  }

  function endRun() {
    if (pendingRun && pendingRun.timer) clearInterval(pendingRun.timer);
    running = false;
    $("progress").hidden = true;
    refresh();
  }

  // ------------------------------------------------------------------------------------------- results
  function showResults(msg) {
    const files = msg.files;
    const manifest = JSON.parse(files["manifest.json"].replace(/\bNaN\b|-?\bInfinity\b/g, "null"));
    const fc = files["forecasts.csv"].split("\n").filter((l) => l), ac = files["test_actuals.csv"].split("\n").filter((l) => l);
    const errors = parseCsv(files["errors_per_series.csv"]);
    const excluded = new Set(manifest.split.excluded.map((e) => e.index));
    const limit = manifest.config.max_series || manifest.dataset.n_series_total;
    const origIndex = [];
    for (let i = 0; i < limit; i++) if (!excluded.has(i)) origIndex.push(i);
    result = { files, manifest, fc, ac, errors, origIndex, H: manifest.split.horizon, estimate: pendingRun && pendingRun.estimate };

    $("runInfo").textContent = manifest.dataset.id + ", " + manifest.config.method + ": " + manifest.dataset.n_series_run + " of " +
      manifest.dataset.n_series_total + " series, horizon " + result.H + ". Time on this device: " + msg.seconds.toFixed(1) + " s" +
      (result.estimate !== null && result.estimate !== undefined ? " (estimated " + seconds(result.estimate) + ")." : ".");
    const w = $("warnings");
    w.textContent = "";
    for (const text of manifest.warnings) w.appendChild(el("p", { class: "warning", text }));

    const table = $("summary");
    table.textContent = "";
    parseCsv(files["summary.csv"]).forEach((cells, r) => {
      const tr = el("tr", {});
      cells.forEach((cell, c) => {
        const text = r === 0 ? cell.replace("N_valid", "Series").replace("N_excluded", "Left out") : c === 0 ? cell : c > 2 ? cell : fmt(num(cell));
        tr.appendChild(el(r === 0 ? "th" : "td", { text }));
      });
      table.appendChild(tr);
    });

    const sSel = $("seriesSel");
    sSel.textContent = "";
    for (let i = 1; i < errors.length; i++) sSel.appendChild(el("option", { value: String(i - 1), text: errors[i][0] }));
    const mSel = $("metricSel");
    mSel.textContent = "";
    errors[0].slice(1).forEach((name, j) => mSel.appendChild(el("option", { value: String(j + 1), text: name })));
    const mase = errors[0].indexOf("MASE");
    if (mase > 0) mSel.value = String(mase);

    const list = $("files");
    list.textContent = "";
    for (const name of Object.keys(files).sort()) {
      const a = el("a", { download: name, text: name });
      a.href = URL.createObjectURL(new Blob([files[name]], { type: name.endsWith(".json") ? "application/json" : "text/csv" }));
      list.appendChild(a);
    }
    $("results").hidden = false;
    requestSeries(0);
    drawHistogram();
  }

  // ---- forecast chart
  function requestSeries(row) {
    if (!result) return;
    row = Math.max(0, Math.min(row, result.origIndex.length - 1));
    $("seriesSel").value = String(row);
    $("prevSeries").disabled = row === 0;
    $("nextSeries").disabled = row === result.origIndex.length - 1;
    result.row = row;
    worker.postMessage({ type: "series", index: result.origIndex[row], token: ++seriesToken });
  }
  $("seriesSel").addEventListener("change", (e) => requestSeries(parseInt(e.target.value, 10)));
  $("prevSeries").addEventListener("click", () => requestSeries(result.row - 1));
  $("nextSeries").addEventListener("click", () => requestSeries(result.row + 1));

  // Charts are drawn at the width they are shown at, so that text keeps its size on a phone; redrawn when that changes.
  function chartWidth(box) { return Math.max(280, Math.round(box.clientWidth || 640)); }
  let resizeTimer = null, lastWidth = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!result || $("results").hidden || $("panel-run").hidden) return;
      const w = chartWidth($("histChart"));
      if (w === lastWidth) return;
      lastWidth = w;
      if (result.values) drawForecast(result.values);
      drawHistogram();
    }, 150);
  });

  function linePath(points) {   // points: [x, y] or null (gap)
    let d = "", pen = false;
    for (const p of points) {
      if (!p) { pen = false; continue; }
      d += (pen ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1);
      pen = true;
    }
    return d;
  }

  function drawForecast(values) {
    const H = result.H, row = result.row;
    const name = parseCsv(result.fc[row + 1])[0][0];
    const forecast = parseCsv(result.fc[row + 1])[0].slice(1).map(num);
    const actual = parseCsv(result.ac[row + 1])[0].slice(1).map(num);
    const train = Array.from(values.subarray(0, values.length - H));
    const shown = Math.min(train.length, Math.max(3 * H, 36));
    const hist = train.slice(train.length - shown);
    $("fcTitle").textContent = "Actual values and forecast: " + name;

    result.values = values;
    const W = chartWidth($("fcChart")), Hh = W < 480 ? 230 : 270, m = { l: 48, r: 12, t: 10, b: 30 };
    const all = hist.concat(actual, forecast).filter(Number.isFinite);
    let lo = Math.min(...all), hi = Math.max(...all);
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.06;
    const ty = ticks(lo - pad, hi + pad, 4);
    const y0 = Math.min(lo - pad, ty.values[0]), y1 = Math.max(hi + pad, ty.values[ty.values.length - 1]);
    const n = shown + H;
    const x = (i) => m.l + ((W - m.l - m.r) * i) / Math.max(1, n - 1);       // i = 0 .. n-1; forecast step h sits at shown + h - 1
    const y = (v) => m.t + (Hh - m.t - m.b) * (1 - (v - y0) / (y1 - y0));

    const root = svg("svg", { viewBox: "0 0 " + W + " " + Hh, role: "img",
                              "aria-label": "Line chart of series " + name + ": the last " + shown + " training values, the " + H +
                                            " actual test values and the forecast." });
    for (const v of ty.values) {
      root.appendChild(svg("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--line)", "stroke-width": 1 }));
      root.appendChild(svg("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end", text: fmt(v) }));
    }
    // the forecast window
    const xo = (x(shown - 1) + x(shown)) / 2;
    root.appendChild(svg("rect", { x: xo, y: m.t, width: W - m.r - xo, height: Hh - m.t - m.b, fill: "var(--band)" }));
    root.appendChild(svg("line", { x1: xo, x2: xo, y1: m.t, y2: Hh - m.b, stroke: "var(--axis)", "stroke-width": 1 }));
    root.appendChild(svg("line", { x1: m.l, x2: W - m.r, y1: Hh - m.b, y2: Hh - m.b, stroke: "var(--axis)", "stroke-width": 1 }));
    const xt = ticks(-shown, H, W < 480 ? 4 : 6).values.filter((v) => Number.isInteger(v) && v !== 0 && v >= -shown + 1 && v <= H);
    for (const v of xt) {
      const xi = v < 0 ? x(shown + v) : x(shown + v - 1);
      root.appendChild(svg("text", { x: xi, y: Hh - m.b + 15, "text-anchor": "middle", text: (v > 0 ? "+" : "−") + Math.abs(v) }));
    }
    root.appendChild(svg("text", { x: (m.l + W - m.r) / 2, y: Hh - 2, "text-anchor": "middle", class: "lbl",
                                   text: "steps before (−) and after (+) the forecast start" }));

    const pts = (arr, offset) => arr.map((v, i) => (Number.isFinite(v) ? [x(offset + i), y(v)] : null));
    const last = Number.isFinite(hist[shown - 1]) ? [[x(shown - 1), y(hist[shown - 1])]] : [null];
    root.appendChild(svg("path", { d: linePath(pts(hist, 0).concat(pts(actual, shown))), fill: "none", stroke: "var(--ink2)", "stroke-width": 2,
                                   "stroke-linejoin": "round", "stroke-linecap": "round" }));
    root.appendChild(svg("path", { d: linePath(last.concat(pts(forecast, shown))), fill: "none", stroke: "var(--series-1)", "stroke-width": 2,
                                   "stroke-linejoin": "round", "stroke-linecap": "round" }));
    if (H <= 24) {
      forecast.forEach((v, i) => {
        if (Number.isFinite(v)) root.appendChild(svg("circle", { cx: x(shown + i), cy: y(v), r: 3, fill: "var(--series-1)", stroke: "var(--panel)", "stroke-width": 1.5 }));
      });
    }
    // hover: crosshair + tooltip
    const cross = svg("line", { y1: m.t, y2: Hh - m.b, stroke: "var(--muted)", "stroke-width": 1, visibility: "hidden" });
    root.appendChild(cross);
    const box = $("fcChart");
    box.textContent = "";
    box.appendChild(root);
    const tip = el("div", { class: "tip", hidden: "" });
    box.appendChild(tip);
    const move = (ev) => {
      const r = root.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1))));
      cross.setAttribute("x1", x(i));
      cross.setAttribute("x2", x(i));
      cross.setAttribute("visibility", "visible");
      tip.textContent = "";
      if (i < shown) {
        tip.appendChild(el("b", { text: (shown - i) + " before the forecast start" }));
        tip.appendChild(el("div", { text: "Actual " + fmt(hist[i]) }));
      } else {
        const h = i - shown;
        tip.appendChild(el("b", { text: "Step +" + (h + 1) }));
        tip.appendChild(el("div", { text: "Actual " + fmt(actual[h]) }));
        tip.appendChild(el("div", { text: "Forecast " + fmt(forecast[h]) }));
      }
      tip.hidden = false;
      const left = ((x(i) / W) * r.width);
      tip.style.left = Math.max(0, Math.min(r.width - tip.offsetWidth, left + (left > r.width / 2 ? -tip.offsetWidth - 10 : 10))) + "px";
      tip.style.top = "8px";
    };
    root.addEventListener("pointermove", move);
    root.addEventListener("pointerdown", move);
    root.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });

    const e = result.errors[row + 1], head = result.errors[0];
    $("fcNote").textContent = "Showing the last " + shown + " of " + train.length + " training values. This series: " +
      head.slice(1, 5).map((nm, j) => nm + " " + fmt(num(e[j + 1]))).join(", ") + ".";
  }

  // ---- distribution of a measure over the series
  function drawHistogram() {
    if (!result) return;
    const col = parseInt($("metricSel").value, 10), name = result.errors[0][col];
    const raw = result.errors.slice(1).map((r) => num(r[col]));
    const vals = raw.filter(Number.isFinite).sort((a, b) => a - b);
    const box = $("histChart");
    box.textContent = "";
    $("histTitle").textContent = "How " + name + " is spread over the series";
    if (!vals.length) { $("histNote").textContent = name + " is undefined for every series of this run."; return; }
    const q = (p) => vals[Math.min(vals.length - 1, Math.floor(p * (vals.length - 1)))];
    const median = vals.length % 2 ? vals[(vals.length - 1) / 2] : (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2;
    let lo = vals[0], hi = vals[vals.length - 1];
    let capped = false;
    if (vals.length >= 20 && hi - lo > 4 * (q(0.9) - lo) && q(0.98) > lo) { hi = q(0.98); capped = true; }   // a long right tail: last bar collects it
    if (hi === lo) { hi = lo + 1; }
    const tk = ticks(lo, hi, Math.max(8, Math.min(24, Math.round(2 * Math.sqrt(vals.length)))));
    const step = tk.step, start = Math.floor(lo / step) * step;
    const nb = Math.max(1, Math.ceil((hi - start) / step - 1e-9));
    const counts = new Array(nb).fill(0);
    for (const v of vals) counts[Math.min(nb - 1, Math.max(0, Math.floor((v - start) / step)))]++;

    const W = chartWidth(box), Hh = W < 480 ? 210 : 240, m = { l: 40, r: 12, t: 12, b: 34 };
    const cmax = Math.max(...counts), cy = ticks(0, cmax, 4);
    const top = Math.max(cmax, cy.values[cy.values.length - 1]);
    const x = (v) => m.l + ((W - m.l - m.r) * (v - start)) / (nb * step);
    const y = (c) => m.t + (Hh - m.t - m.b) * (1 - c / top);
    const root = svg("svg", { viewBox: "0 0 " + W + " " + Hh, role: "img",
                              "aria-label": "Histogram of " + name + " over " + vals.length + " series; median " + fmt(median) + "." });
    for (const c of cy.values) {
      if (!Number.isInteger(c)) continue;
      root.appendChild(svg("line", { x1: m.l, x2: W - m.r, y1: y(c), y2: y(c), stroke: "var(--line)", "stroke-width": 1 }));
      root.appendChild(svg("text", { x: m.l - 6, y: y(c) + 4, "text-anchor": "end", text: String(c) }));
    }
    const tip = el("div", { class: "tip", hidden: "" });
    counts.forEach((c, b) => {
      const x0 = x(start + b * step) + 1, x1 = x(start + (b + 1) * step) - 1, yt = y(c), yb = y(0), r = Math.min(4, (x1 - x0) / 2, yb - yt);
      const label = fmt(start + b * step) + (capped && b === nb - 1 ? " and above" : " to " + fmt(start + (b + 1) * step));
      if (c > 0) {
        root.appendChild(svg("path", { fill: "var(--series-1)",
          d: "M" + x0 + " " + yb + "V" + (yt + r) + "Q" + x0 + " " + yt + " " + (x0 + r) + " " + yt + "H" + (x1 - r) + "Q" + x1 + " " + yt + " " + x1 + " " + (yt + r) + "V" + yb + "Z" }));
      }
      const hit = svg("rect", { x: x0 - 1, y: m.t, width: x1 - x0 + 2, height: Hh - m.t - m.b, fill: "transparent" });
      const show = () => {
        tip.textContent = "";
        tip.appendChild(el("b", { text: name + " " + label }));
        tip.appendChild(el("div", { text: c + " series" }));
        tip.hidden = false;
        const rect = root.getBoundingClientRect(), left = (((x0 + x1) / 2) / W) * rect.width;
        tip.style.left = Math.max(0, Math.min(rect.width - tip.offsetWidth, left - tip.offsetWidth / 2)) + "px";
        tip.style.top = "4px";
      };
      hit.addEventListener("pointerenter", show);
      hit.addEventListener("pointerdown", show);
      hit.addEventListener("pointerleave", () => { tip.hidden = true; });
      root.appendChild(hit);
    });
    root.appendChild(svg("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: "var(--axis)", "stroke-width": 1 }));
    const every = Math.max(1, Math.ceil(nb / (W < 480 ? 5 : 7)));
    for (let b = 0; b <= nb; b += every) {
      root.appendChild(svg("text", { x: x(start + b * step), y: Hh - m.b + 15, "text-anchor": "middle", text: fmt(start + b * step) }));
    }
    root.appendChild(svg("text", { x: (m.l + W - m.r) / 2, y: Hh - 3, "text-anchor": "middle", class: "lbl", text: name + " (bars: number of series)" }));
    if (median >= start && median <= start + nb * step) {
      const xm = x(median);
      root.appendChild(svg("line", { x1: xm, x2: xm, y1: m.t, y2: y(0), stroke: "var(--ink)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
      root.appendChild(svg("text", { x: xm + (xm > W * 0.75 ? -5 : 5), y: m.t + 10, "text-anchor": xm > W * 0.75 ? "end" : "start", class: "lbl",
                                     text: "median " + fmt(median) }));
    }
    box.appendChild(root);
    box.appendChild(tip);
    const left = raw.length - vals.length, mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    $("histNote").textContent = vals.length + " series, mean " + fmt(mean) + ", median " + fmt(median) + ", from " + fmt(vals[0]) + " to " +
      fmt(vals[vals.length - 1]) + "." + (capped ? " The last bar also holds the series above " + fmt(start + (nb - 1) * step) + "." : "") +
      (left ? " " + left + " series left out: the measure is undefined for them." : "");
  }
  $("metricSel").addEventListener("change", drawHistogram);

  // ---- all result files in one .zip (stored, no compression)
  const CRC = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();
  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }
  function zip(files) {
    const enc = new TextEncoder(), parts = [], central = [];
    const now = new Date();
    const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
    const date = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
    let offset = 0;
    for (const name of Object.keys(files).sort()) {
      const nm = enc.encode(name), body = enc.encode(files[name]), crc = crc32(body);
      const head = new DataView(new ArrayBuffer(30));
      head.setUint32(0, 0x04034b50, true); head.setUint16(4, 20, true); head.setUint16(6, 0x0800, true); head.setUint16(8, 0, true);
      head.setUint16(10, time, true); head.setUint16(12, date, true); head.setUint32(14, crc, true);
      head.setUint32(18, body.length, true); head.setUint32(22, body.length, true); head.setUint16(26, nm.length, true); head.setUint16(28, 0, true);
      parts.push(new Uint8Array(head.buffer), nm, body);
      const cd = new DataView(new ArrayBuffer(46));
      cd.setUint32(0, 0x02014b50, true); cd.setUint16(4, 20, true); cd.setUint16(6, 20, true); cd.setUint16(8, 0x0800, true); cd.setUint16(10, 0, true);
      cd.setUint16(12, time, true); cd.setUint16(14, date, true); cd.setUint32(16, crc, true);
      cd.setUint32(20, body.length, true); cd.setUint32(24, body.length, true); cd.setUint16(28, nm.length, true);
      cd.setUint32(42, offset, true);
      central.push(new Uint8Array(cd.buffer), nm);
      offset += 30 + nm.length + body.length;
    }
    const size = central.reduce((s, p) => s + p.length, 0), count = central.length / 2;
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, count, true); end.setUint16(10, count, true);
    end.setUint32(12, size, true); end.setUint32(16, offset, true);
    return new Blob(parts.concat(central, [new Uint8Array(end.buffer)]), { type: "application/zip" });
  }
  $("downloadAll").addEventListener("click", () => {
    if (!result) return;
    const a = el("a", { download: result.manifest.run_id + ".zip" });
    a.href = URL.createObjectURL(zip(result.files));
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 30000);
  });

  // ------------------------------------------------------------------------------------------- worker messages
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === "ready") {
      engineReady = true;
      $("method").disabled = false;
      status("Engine ready (" + msg.version + ").");
      examplesReady.then(() => { if (examples.length && !data && !pendingLoad) { $("example").value = examples[0].id; loadExample(examples[0].id); } });
    } else if (msg.type === "loaded") {
      onLoaded(msg);
    } else if (msg.type === "estimate") {
      onEstimate(msg);
    } else if (msg.type === "done") {
      endRun();
      status("Finished in " + msg.seconds.toFixed(1) + " s.");
      showResults(msg);
    } else if (msg.type === "series") {
      if (msg.token === seriesToken && result) drawForecast(msg.values);
    } else if (msg.type === "error") {
      if (msg.stage === "load") { $("dataInfo").textContent = msg.message; $("dataInfo").className = "note error"; }
      else if (msg.stage === "run") { endRun(); status(msg.message, true); }
    }
    if (msg.type === "loaded") $("dataInfo").className = "note";
    refresh();
  };
  worker.onerror = (e) => { endRun(); status("The engine could not be loaded: " + (e.message || "unknown error"), true); };
  if (location.protocol === "file:") {
    // a browser does not start the engine for a page opened from a file
    status("This page was opened as a file, so the engine cannot start. Open it from its web address instead.", true);
    $("runWhy").textContent = "";
  }
  learnReady.then(refresh);
})();
