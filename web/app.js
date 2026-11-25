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
      !data ? (pendingLoad ? "The button turns on when the data has been read." : "Choose an example dataset or a file of your own first.") : "";
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
    result = null;                 // results and characteristics belong to the previous file
    $("results").hidden = true;
    resetCharacteristics();
    $("charStatus").textContent = "Reading the data…";
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
             stats: st, header: msg.header, noDates: !!pendingLoad.noDates };
    const len = st.len_min === st.len_max ? st.len_min + " observations each" : st.len_min + "–" + st.len_max + " observations";
    $("dataInfo").textContent = (info ? info.title + ": " : "") + st.n_series + " " + (st.frequency || "") + " series, " + len + ", " +
      pendingLoad.mb.toFixed(2) + " MB" + (st.missing_share > 0 ? ", " + (100 * st.missing_share).toFixed(1) + "% missing" : "") +
      (data.integer ? "; forecasts are rounded to whole numbers" : "");
    const h = info && info.horizon ? info.horizon : st.horizon;
    $("horizon").value = h || "";
    if (!h) $("dataInfo").textContent += ". The file names no horizon: enter one below.";
    refresh();
    afterLoadCharacteristics();
  }

  // A table (Excel, CSV) is converted to .tsf on this device (convert.js) and then read like any .tsf file.
  let table = null;            // { name, bytes } of the table chosen last, kept to convert it again at another frequency
  let converted = null;        // { name, text } of the .tsf made from it
  const LAYOUT = { long: "one row per observation", wide: "one row per date, one column per series", columns: "columns of numbers without dates" };
  const freqLabel = (f) => f.replace("_", " ");
  (function () {
    const sel = $("convFreq");
    sel.appendChild(el("option", { value: "", text: "work it out from the dates" }));
    for (const f of TsfConvert.FREQUENCIES) sel.appendChild(el("option", { value: f, text: freqLabel(f) }));
  })();

  function resetConversion() {
    table = null;
    converted = null;
    $("convBox").hidden = true;
    $("convSave").hidden = true;
    $("convFreq").value = "";
  }

  function conversionText(rep) {
    const parts = [
      "Read as a table with " + LAYOUT[rep.layout] + (rep.sheet ? " (sheet '" + rep.sheet + "')" : "") + ": " + rep.nSeries + " series, " +
        freqLabel(rep.frequency) + (rep.noDates ? "" : ", " + rep.first + " to " + rep.last) + "."];
    if (rep.noDates) parts.push(rep.frequencySource === "assumed"
      ? "The file has no dates: the data is taken as monthly, oldest value first. Choose the frequency below if it is another one."
      : "The file has no dates: the oldest value is taken to come first.");
    else if (rep.frequencySource === "detected") parts.push("The frequency was worked out from the dates; change it below if it is wrong.");
    if (rep.dateOrder) parts.push("Dates were read as " + rep.dateOrder + ".");
    if (rep.filled) parts.push(rep.filled + " dates absent inside the series became missing values.");
    if (rep.dropped) parts.push(rep.dropped + " series without any number were left out.");
    if (rep.renamed) parts.push(rep.renamed + " series names were simplified (letters, digits and underscores only).");
    if (rep.ignored.length) parts.push("Columns not used: " + rep.ignored.slice(0, 6).join(", ") + (rep.ignored.length > 6 ? " and " + (rep.ignored.length - 6) + " more" : "") + ".");
    return parts.join(" ");
  }

  async function convertTable() {
    const mine = table;
    $("convBox").hidden = false;
    $("convSave").hidden = true;
    $("convInfo").className = "note";
    $("convInfo").textContent = "Converting " + mine.name + "…";
    try {
      const r = await TsfConvert.convert(mine.name, mine.bytes, { frequency: $("convFreq").value || undefined });
      if (table !== mine) return;
      converted = { name: r.name, text: r.text };
      $("convInfo").textContent = conversionText(r.report);
      $("convSave").hidden = false;
      const bytes = new TextEncoder().encode(r.text).buffer;
      sendFile(r.name, bytes, null);
      pendingLoad.noDates = !!r.report.noDates;      // a table without dates: forecasts are numbered, not dated
    } catch (err) {
      if (table !== mine) return;
      converted = null;
      data = null;
      result = null;
      $("results").hidden = true;
      resetCharacteristics();
      $("dataInfo").textContent = "";
      $("convInfo").className = "note error";
      $("convInfo").textContent = err instanceof TsfConvert.ConvertError ? err.message
        : "The file could not be converted (" + (err && err.message ? err.message : err) + ").";
      refresh();
    }
  }

  $("convFreq").addEventListener("change", () => { if (table) convertTable(); });
  $("convSave").addEventListener("click", () => {
    if (!converted) return;
    const a = el("a", { download: converted.name });
    a.href = URL.createObjectURL(new Blob([converted.text], { type: "text/plain" }));
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  $("file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    $("example").value = "";
    const bytes = await f.arrayBuffer();
    resetConversion();
    if (TsfConvert.isTsf(f.name, bytes)) { sendFile(f.name, bytes, null); return; }
    table = { name: f.name, bytes };
    convertTable();
  });

  async function loadExample(id) {
    const info = examples.find((x) => x.id === id);
    if (!info) return;
    $("dataInfo").textContent = "Loading " + info.id + "…";
    try {
      const r = await fetch("data/" + info.file);
      if (!r.ok) throw new Error("HTTP " + r.status);
      $("file").value = "";
      resetConversion();
      sendFile(info.file, await r.arrayBuffer(), info);
    } catch (err) {
      $("dataInfo").textContent = "The example could not be loaded (" + err.message + "); choose a file of your own instead.";
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
    if (r.url) li.appendChild(el("a", { href: r.url, target: "_blank", rel: "noopener", text: /doi\.org/.test(r.url) ? "DOI" : "link" }));
    if (!r.verified) li.appendChild(el("span", { class: "tag", text: "not checked" }));   // only what is still open is marked
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
    // data characteristics and the map
    const ch = learn.characteristics, chars = $("learnChars");
    chars.textContent = "";
    if (ch) {
      const block = (title, note, items, refKeys, open) => {
        const d = el("details", open ? { open: "" } : {});
        d.appendChild(el("summary", { text: title }));
        if (note) d.appendChild(el("p", { text: note }));
        const dl = el("dl", {});
        for (const it of items) { dl.appendChild(el("dt", { text: it.id })); dl.appendChild(el("dd", { text: it.text })); }
        d.appendChild(dl);
        const p = el("p", { class: "note" });
        p.appendChild(document.createTextNode("Sources: "));
        p.appendChild(cite(refKeys));
        d.appendChild(p);
        chars.appendChild(d);
      };
      block("Demand pattern (four classes)", "Every series is put in one of four classes from two numbers: how often demand occurs (ADI) and " +
            "how much its size varies (CV²). The class says which kind of method suits the series: the intermittent-demand methods are " +
            "made for the intermittent and lumpy classes.", ch.classes, ch.class_refs, true);
      block("Descriptive characteristics (10)", "Simple summaries of each series, computed after leaving out missing values.", ch.basic, ch.basic_refs);
      block("catch22 (22 characteristics)", "A set of 22 characteristics chosen from several thousand candidates for telling time series " +
            "apart while overlapping little. Each is computed on the standardised series, so it describes the shape of the series, " +
            "not its level or scale. Series with fewer than 10 observations, constant series and series with missing values get no value.",
            ch.catch22, ch.catch22_refs);
      if (ch.tsfeatures) {
        block("tsfeatures (about 40 characteristics, optional)", "The characteristics of the R package tsfeatures, computed as in the " +
              "ranking study this application comes from: on the scaled series, with the seasonal periods of the data's frequency. " +
              "The page computes them when the box under “Data characteristics” is ticked; they take longer than the others. " +
              "A characteristic that cannot be computed for some series of a file (the seasonal ones for a series shorter than two " +
              "periods, for example) is left out for the whole file, and series with fewer than 4 observations get no values.",
              ch.tsfeatures, ch.tsfeatures_refs);
      }
      const d = el("details", {});
      d.appendChild(el("summary", { text: "How to read the map (principal components)" }));
      for (const t of ch.map) d.appendChild(el("p", { text: t }));
      const p2 = el("p", { class: "note" });
      p2.appendChild(document.createTextNode("Sources: "));
      p2.appendChild(cite(ch.map_refs));
      d.appendChild(p2);
      chars.appendChild(d);
    }
    const refs = $("learnRefs");
    refs.textContent = "";
    for (const k of order) refs.appendChild(refLine(k));
    const open = order.filter((k) => !learn.references[k].verified).length;
    $("learnRefsNote").textContent = open ? "Entries marked “not checked” have not been compared with a bibliographic record yet; check them yourself before citing them."
      : "Every entry was compared with the publisher's or a bibliographic record (authors, year, title, journal, volume, issue and pages).";
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

  const PI_MIN_SERIES = 10;   // prediction intervals pool the calibration errors over the series; fewer give no usable spread
  let purpose = "test";       // "test": hold the last values back and measure the error; "future": forecast beyond the data

  function setPurpose(p) {
    purpose = p;
    const future = p === "future";
    $("modeTest").setAttribute("aria-pressed", String(!future));
    $("modeFuture").setAttribute("aria-pressed", String(future));
    $("horizonLabel").textContent = future ? "Steps ahead to forecast" : "Horizon (blank = file's)";
    $("modeNote").textContent = future
      ? "The method sees the whole of each series and forecasts the steps after its last observation. No error is computed, since " +
        "there is nothing to compare with yet. With " + PI_MIN_SERIES + " series or more you also get 80% and 95% prediction intervals " +
        "(the run then takes about twice as long)."
      : "The last values of each series are held back, the method forecasts them from the rest, and the forecast is compared " +
        "with what really happened. Use this to see how well a method works on your data.";
    $("run").textContent = future ? "Forecast" : "Start run";
  }
  $("modeTest").addEventListener("click", () => setPurpose("test"));
  $("modeFuture").addEventListener("click", () => setPurpose("future"));
  setPurpose("test");

  $("run").addEventListener("click", () => {
    const future = purpose === "future";
    const h = $("horizon").value ? parseInt($("horizon").value, 10) : 0;
    if (future && !(h >= 1)) { status("Enter how many steps ahead to forecast.", true); $("horizon").focus(); return; }
    const config = ["method=" + $("method").value, "missing_policy=" + $("missing").value, "threads=1", "eol=lf",
                    "dataset_id=" + data.id, "integer_conversion=" + (data.integer ? 1 : 0)];
    if (learn) config.push("metrics=" + (future ? "MAE" : learn.metrics.map((x) => x.id).join(",")));
    if (h >= 1) config.push("horizon=" + h);
    const limit = $("maxSeries").value ? Math.min(parseInt($("maxSeries").value, 10), data.stats.n_series) : data.stats.n_series;
    if ($("maxSeries").value) config.push("max_series=" + limit);
    const intervals = future && limit >= PI_MIN_SERIES;
    if (intervals) config.push("intervals=80,95");
    const pilot = limit > 4 * PILOT ? PILOT : 0;
    running = true;
    refresh();
    pendingRun = { limit, started: performance.now(), estimate: null, timer: null, future, intervals };
    $("progress").hidden = false;
    $("progress").firstElementChild.style.width = "0";
    status(pilot ? "Timing " + pilot + " series to estimate the run…" : (future ? "Forecasting " : "Running ") + limit + " series…");
    worker.postMessage({ type: "run", config, pilot, future: future ? h : 0 });
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
  // The date of a position in a series (0 = its first observation), from the series' start timestamp; null without one.
  function stepDate(start, position, frequency) {
    const t = start ? TsfConvert.parseDate(start) : null;
    if (t == null || !TsfConvert.FREQUENCIES.includes(frequency)) return null;
    return TsfConvert.shortDate(TsfConvert.advance(t, position, frequency), frequency);
  }

  // A forecast of the future: the run's forecasts (and bounds) without anything that compares them with stand-in values.
  function showFuture(msg) {
    const run = msg.files;
    const manifest = JSON.parse(run["manifest.json"].replace(/\bNaN\b|-?\bInfinity\b/g, "null"));
    const fc = run["forecasts.csv"].split("\n").filter((l) => l);
    const excluded = new Set(manifest.split.excluded.map((e) => e.index));
    const limit = manifest.config.max_series || manifest.dataset.n_series_total;
    const origIndex = [];
    for (let i = 0; i < limit; i++) if (!excluded.has(i)) origIndex.push(i);
    const H = manifest.split.horizon, frequency = data.header.frequency, dated = !data.noDates;
    const rows = fc.slice(1).map((l) => parseCsv(l)[0]);
    const names = rows.map((r) => r[0]);
    // bounds: intervals.csv holds one block of series rows per quantile
    let bounds = null;
    if (run["intervals.csv"]) {
      bounds = {};
      for (const r of parseCsv(run["intervals.csv"]).slice(1)) {
        const q = String(Number(r[1]));
        if (r.length > 2) (bounds[q] = bounds[q] || []).push(r.slice(2).map(num));
      }
      if (!["0.025", "0.1", "0.9", "0.975"].every((q) => bounds[q] && bounds[q].length === rows.length)) bounds = null;
    }
    const start = (r) => msg.info.starts[origIndex[r]], length = (r) => msg.info.lengths[origIndex[r]];
    const date = (r, h) => (dated ? stepDate(start(r), length(r) + h, frequency) : null);     // h = 0 .. H-1
    const anyDate = dated && rows.length > 0 && date(0, 0) !== null;
    const lines = ["series,step" + (anyDate ? ",date" : "") + ",forecast" + (bounds ? ",lower_95,lower_80,upper_80,upper_95" : "")];
    rows.forEach((r, i) => {
      for (let h = 0; h < H; h++) {
        const cells = [csvCell(r[0]), String(h + 1)];
        if (anyDate) cells.push(date(i, h) || "");
        cells.push(r[h + 1]);
        if (bounds) for (const q of ["0.025", "0.1", "0.9", "0.975"]) cells.push(Number.isFinite(bounds[q][i][h]) ? String(bounds[q][i][h]) : "");
        lines.push(cells.join(","));
      }
    });
    const info = {
      purpose: "forecast beyond the end of the data", dataset: manifest.dataset.id, method: manifest.config.method, frequency, horizon: H,
      series_forecast: rows.length, series_in_file: manifest.dataset.n_series_total,
      left_out: manifest.split.excluded.map((e) => ({ name: e.name, reason: /missing/.test(e.reason) ? e.reason : "too short for this horizon" })),
      missing_values: manifest.config.missing_policy,
      prediction_intervals: bounds ? "80% and 95%, split conformal: calibrated on the last " + H + " observations of the series" : null,
      engine: manifest.engine || null, warnings: manifest.warnings,
    };
    const files = { "forecast_table.csv": lines.join("\n") + "\n", "forecasts.csv": run["forecasts.csv"],
                    "forecast_info.json": JSON.stringify(info, null, 2) + "\n" };
    if (run["fallbacks.csv"] && run["fallbacks.csv"].split("\n").filter((l) => l).length > 1) files["fallbacks.csv"] = run["fallbacks.csv"];
    result = { future: true, files, manifest, fc, bounds, origIndex, H, date, anyDate, errors: [["series"]].concat(names.map((n) => [n])),
               estimate: pendingRun && pendingRun.estimate, zipName: manifest.run_id + "_forecast" };

    const span = anyDate ? " (" + date(0, 0) + (H > 1 ? " to " + date(0, H - 1) : "") + " for " + names[0] + ")" : "";
    $("runInfo").textContent = manifest.dataset.id + ", " + manifest.config.method + ": forecast of " + rows.length + " of " +
      manifest.dataset.n_series_total + " series, " + H + " step" + (H > 1 ? "s" : "") + " after the last observation" + span +
      ". Time on this device: " + msg.seconds.toFixed(1) + " s.";
    const w = $("warnings");
    w.textContent = "";
    if (manifest.split.excluded.length)
      w.appendChild(el("p", { class: "warning", text: manifest.split.excluded.length + " series were left out (too short for this horizon" +
        (bounds ? " with prediction intervals" : "") + ", or without values): " +
        manifest.split.excluded.slice(0, 5).map((e) => e.name).join(", ") + (manifest.split.excluded.length > 5 ? ", …" : "") + "." }));
    for (const text of manifest.warnings) if (/fallback/i.test(text)) w.appendChild(el("p", { class: "warning", text }));
    if (!bounds) w.appendChild(el("p", { class: "note", text: pendingRun && pendingRun.intervals
      ? "Prediction intervals could not be computed for this run."
      : "No prediction intervals: they need at least " + PI_MIN_SERIES + " series, because the spread is learned from the errors of all series together." }));
    showCommon(files, names);
  }

  // What both kinds of result share: which blocks show, the series list, the files.
  function showCommon(files, names) {
    const future = !!result.future;
    $("summaryBlock").hidden = future;
    $("histBlock").hidden = future;
    const sSel = $("seriesSel");
    sSel.textContent = "";
    names.forEach((n, i) => sSel.appendChild(el("option", { value: String(i), text: n })));
    const list = $("files");
    list.textContent = "";
    for (const name of Object.keys(files).sort()) {
      const a = el("a", { download: name, text: name });
      a.href = URL.createObjectURL(new Blob([files[name]], { type: name.endsWith(".json") ? "application/json" : "text/csv" }));
      list.appendChild(a);
    }
    $("results").hidden = false;
    requestSeries(0);
    if (!future) drawHistogram();
    updatePcaColor();              // the map can now be coloured by this run's errors
  }

  function showResults(msg) {
    if (msg.future) { showFuture(msg); return; }
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

    const mSel = $("metricSel");
    mSel.textContent = "";
    errors[0].slice(1).forEach((name, j) => mSel.appendChild(el("option", { value: String(j + 1), text: name })));
    const mase = errors[0].indexOf("MASE");
    if (mase > 0) mSel.value = String(mase);
    showCommon(files, errors.slice(1).map((r) => r[0]));
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
    const H = result.H, row = result.row, future = !!result.future;
    const name = parseCsv(result.fc[row + 1])[0][0];
    const forecast = parseCsv(result.fc[row + 1])[0].slice(1).map(num);
    const actual = future ? [] : parseCsv(result.ac[row + 1])[0].slice(1).map(num);
    const train = future ? Array.from(values) : Array.from(values.subarray(0, values.length - H));
    const shown = Math.min(train.length, Math.max(3 * H, 36));
    const hist = train.slice(train.length - shown);
    const pi = future && result.bounds ? { lo95: result.bounds["0.025"][row], lo80: result.bounds["0.1"][row],
                                           hi80: result.bounds["0.9"][row], hi95: result.bounds["0.975"][row] } : null;
    $("fcTitle").textContent = (future ? "Forecast of the future: " : "Actual values and forecast: ") + name;
    const legend = $("fcLegend");
    legend.textContent = "";
    legend.appendChild(el("span", {}, el("i", { style: "border-color:var(--ink2)" }), document.createTextNode(future ? "Observed" : "Actual")));
    legend.appendChild(el("span", {}, el("i", { style: "border-color:var(--series-1)" }), document.createTextNode("Forecast")));
    if (pi) {
      legend.appendChild(el("span", {}, el("i", { class: "box", style: "background:var(--pi80)" }), document.createTextNode("80% interval")));
      legend.appendChild(el("span", {}, el("i", { class: "box", style: "background:var(--pi95)" }), document.createTextNode("95% interval")));
    }

    result.values = values;
    const W = chartWidth($("fcChart")), Hh = W < 480 ? 230 : 270, m = { l: 48, r: 12, t: 10, b: 30 };
    const all = hist.concat(actual, forecast, pi ? pi.lo95.concat(pi.hi95) : []).filter(Number.isFinite);
    let lo = Math.min(...all), hi = Math.max(...all);
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.06;
    const ty = ticks(lo - pad, hi + pad, 4);
    const y0 = Math.min(lo - pad, ty.values[0]), y1 = Math.max(hi + pad, ty.values[ty.values.length - 1]);
    const n = shown + H;
    const x = (i) => m.l + ((W - m.l - m.r) * i) / Math.max(1, n - 1);       // i = 0 .. n-1; forecast step h sits at shown + h - 1
    const y = (v) => m.t + (Hh - m.t - m.b) * (1 - (v - y0) / (y1 - y0));

    const root = svg("svg", { viewBox: "0 0 " + W + " " + Hh, role: "img",
                              "aria-label": future
                                ? "Line chart of series " + name + ": its last " + shown + " observations and the forecast of the next " + H + " steps" +
                                  (pi ? " with 80% and 95% prediction intervals." : ".")
                                : "Line chart of series " + name + ": the last " + shown + " training values, the " + H +
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
    if (pi) {       // the wider band first; both fan out from the last observation
      for (const [lo, hi, fill] of [[pi.lo95, pi.hi95, "var(--pi95)"], [pi.lo80, pi.hi80, "var(--pi80)"]]) {
        const up = [], down = [];
        for (let h = 0; h < H; h++) if (Number.isFinite(lo[h]) && Number.isFinite(hi[h])) { up.push([x(shown + h), y(hi[h])]); down.push([x(shown + h), y(lo[h])]); }
        if (!up.length) continue;
        const ring = (last[0] ? [last[0]] : []).concat(up, down.reverse());
        root.appendChild(svg("path", { d: ring.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join("") + "Z", fill }));
      }
    }
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
        const when = future && result.anyDate ? result.date(row, i - shown) : null;
        tip.appendChild(el("b", { text: when || (shown - i) + " before the forecast start" }));
        tip.appendChild(el("div", { text: (future ? "Observed " : "Actual ") + fmt(hist[i]) }));
      } else {
        const h = i - shown, when = future && result.anyDate ? result.date(row, h) : null;
        tip.appendChild(el("b", { text: "Step +" + (h + 1) + (when ? " · " + when : "") }));
        if (!future) tip.appendChild(el("div", { text: "Actual " + fmt(actual[h]) }));
        tip.appendChild(el("div", { text: "Forecast " + fmt(forecast[h]) }));
        if (pi) {
          tip.appendChild(el("div", { text: "80%: " + fmt(pi.lo80[h]) + " to " + fmt(pi.hi80[h]) }));
          tip.appendChild(el("div", { text: "95%: " + fmt(pi.lo95[h]) + " to " + fmt(pi.hi95[h]) }));
        }
      }
      tip.hidden = false;
      const left = ((x(i) / W) * r.width);
      tip.style.left = Math.max(0, Math.min(r.width - tip.offsetWidth, left + (left > r.width / 2 ? -tip.offsetWidth - 10 : 10))) + "px";
      tip.style.top = "8px";
    };
    root.addEventListener("pointermove", move);
    root.addEventListener("pointerdown", move);
    root.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });

    if (future) {
      const d0 = result.anyDate ? result.date(row, 0) : null, d1 = result.anyDate ? result.date(row, H - 1) : null;
      $("fcNote").textContent = "Showing the last " + shown + " of " + train.length + " observations and the forecast of the next " + H + " step" +
        (H > 1 ? "s" : "") + (d0 ? " (" + d0 + (H > 1 ? " to " + d1 : "") + ")" : "") + "." +
        (pi ? " The bands say how far such forecasts were off on the last " + H + " observations of the series; they are a guide, not a guarantee." : "");
      return;
    }
    const e = result.errors[row + 1], head = result.errors[0];
    $("fcNote").textContent = "Showing the last " + shown + " of " + train.length + " training values. This series: " +
      head.slice(1, 5).map((nm, j) => nm + " " + fmt(num(e[j + 1]))).join(", ") + ".";
  }

  // ---- distribution of a measure over the series
  function drawHistogram() {
    if (!result || result.future) return;
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
    const a = el("a", { download: (result.zipName || result.manifest.run_id) + ".zip" });
    a.href = URL.createObjectURL(zip(feat ? Object.assign({ "characteristics.csv": featuresCsv() }, result.files) : result.files));
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 30000);
  });

  // ------------------------------------------------------------------------------------------- data characteristics
  // As the desktop application's "Features" view: basic features + catch22 (and, when asked, the tsfeatures set) per
  // series, and a PCA map of the series.
  const CLASSES = ["smooth", "intermittent", "erratic", "lumpy"];
  const PCA_CAT = { light: ["#2a78d6", "#eb6834", "#1baf7a"], dark: ["#3987e5", "#d95926", "#199e70"] };
  const PCA_SEQ = { light: ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"], dark: ["#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"] };
  const PCA_OTHER = "#898781";
  const AUTO_VALUES = 3e6;   // files up to this many observations are analysed without asking
  const AUTO_TSF = 3e5;      // ... and up to this many with the (slower) tsfeatures set as well
  let feat = null;           // the worker's answer for the loaded file
  let featToken = 0;
  const PCA_3D = { yaw: -0.6, pitch: 0.35, zoom: 1 };
  const pcaView = Object.assign({ ax: [0, 1], arrows: true, color: "class", focus: null, pts: [], colors: null, dim: 2 }, PCA_3D);
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const themeMode = () => (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

  function resetCharacteristics() {
    feat = null;
    featToken++;
    $("charBody").hidden = true;
    $("charRun").hidden = true;
    $("charTsfRow").hidden = true;
  }
  function requestFeatures() {
    $("charRun").hidden = true;
    $("charTsfRow").hidden = false;
    $("charStatus").textContent = "Computing the characteristics of " + data.stats.n_series + " series…";
    worker.postMessage({ type: "features", token: ++featToken, tsfeatures: $("charTsf").checked });
  }
  $("charRun").addEventListener("click", requestFeatures);
  $("charTsf").addEventListener("change", () => { if (data && $("charRun").hidden) requestFeatures(); });

  function afterLoadCharacteristics() {
    resetCharacteristics();
    const approx = data.stats.n_series * data.stats.len_median;
    $("charTsf").checked = approx <= AUTO_TSF;
    $("charTsfRow").hidden = false;
    if (approx <= AUTO_VALUES) { requestFeatures(); return; }
    $("charStatus").textContent = "This file has about " + (approx / 1e6).toFixed(1) + " million observations; analysing it can take a while.";
    $("charRun").hidden = false;
  }

  function column(name) {
    const j = feat.columns.indexOf(name), p = feat.columns.length, out = new Array(feat.names.length);
    for (let i = 0; i < out.length; i++) out[i] = feat.table[i * p + j];
    return out;
  }

  function pcaColorOptions() {
    const sel = $("pcaColor"), keep = pcaView.color;
    sel.textContent = "";
    sel.appendChild(el("option", { value: "class", text: "Demand pattern" }));
    if (result && !result.future) {
      const og = el("optgroup", { label: "Error of the last run (" + result.manifest.config.method + ")" });
      result.errors[0].slice(1).forEach((m) => og.appendChild(el("option", { value: "run:" + m, text: m })));
      sel.appendChild(og);
    }
    const og2 = el("optgroup", { label: "A characteristic" });
    feat.columns.forEach((c) => og2.appendChild(el("option", { value: "feature:" + c, text: c })));
    sel.appendChild(og2);
    sel.value = keep;
    if (sel.value !== keep) { sel.value = "class"; pcaView.color = "class"; }
  }

  // the values the points are coloured by: { kind: "cat" | "num", label, values }
  function pcaColorData() {
    const [kind, rest] = pcaView.color.split(/:(.*)/);
    if (kind === "feature") return { kind: "num", label: rest, values: column(rest) };
    if (kind === "run" && result && !result.future) {
      const j = result.errors[0].indexOf(rest), by = new Map();
      for (let i = 1; i < result.errors.length; i++) by.set(result.errors[i][0], num(result.errors[i][j]));
      return { kind: "num", label: rest + " of " + result.manifest.config.method, values: feat.names.map((n) => (by.has(n) ? by.get(n) : NaN)),
               partial: by.size < feat.names.length };
    }
    return { kind: "cat", label: "Demand pattern", values: feat.classes };
  }

  function pcaColors(c) {
    const mode = themeMode(), n = c.values.length;
    if (c.kind === "cat") {
      const counts = {};
      c.values.forEach((v) => { if (v != null) counts[v] = (counts[v] || 0) + 1; });
      const levels = CLASSES.filter((l) => counts[l]);
      const slot = {};
      levels.slice(0, 3).forEach((l, i) => { slot[l] = PCA_CAT[mode][i]; });
      const other = levels.slice(3);
      const items = levels.slice(0, 3).map((l) => ({ key: l, label: l, color: slot[l], n: counts[l] }));
      if (other.length) items.push({ key: "__other", label: other.join(", "), color: PCA_OTHER, n: other.reduce((s, l) => s + counts[l], 0), members: other });
      return { col: c.values.map((v) => (v == null ? null : slot[v] || PCA_OTHER)),
               legend: { kind: "cat", title: c.label, items, missing: c.values.filter((v) => v == null).length } };
    }
    const idx = c.values.map((x, i) => [x, i]).filter(([x]) => Number.isFinite(x)).sort((a, b) => a[0] - b[0]);
    const stops = PCA_SEQ[mode], rgb = (h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16));
    const ramp = (t) => {
      const p = t * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(p)), f = p - k, a = rgb(stops[k]), b = rgb(stops[k + 1]);
      return "rgb(" + a.map((x, j) => Math.round(x + (b[j] - x) * f)).join(",") + ")";
    };
    const col = new Array(n).fill(null);
    idx.forEach(([, i], r) => { col[i] = ramp(idx.length > 1 ? r / (idx.length - 1) : 0.5); });
    const q = (p) => (idx.length ? idx[Math.min(idx.length - 1, Math.round(p * (idx.length - 1)))][0] : NaN);
    return { col, legend: { kind: "num", title: c.label, stops, lo: q(0), mid: q(0.5), hi: q(1), missing: n - idx.length } };
  }

  function renderPcaLegend() {
    const host = $("pcaLegend"), lg = pcaView.colors.legend;
    host.textContent = "";
    host.appendChild(el("span", { text: lg.title + ":" }));
    if (lg.kind === "cat") {
      for (const it of lg.items) {
        const b = el("button", { type: "button", "aria-pressed": String(pcaView.focus === it.key), title: "Show this group on top, dim the rest" },
                     el("i", { class: "sw", style: "background:" + it.color }), it.label + " " + it.n);
        b.addEventListener("click", () => { pcaView.focus = pcaView.focus === it.key ? null : it.key; renderPcaLegend(); drawPca(); });
        host.appendChild(b);
      }
    } else {
      host.appendChild(el("span", { text: fmt(lg.lo) }));
      host.appendChild(el("i", { class: "ramp", style: "background:linear-gradient(90deg," + lg.stops.join(",") + ")" }));
      host.appendChild(el("span", { text: fmt(lg.hi) + " (colour by rank; median " + fmt(lg.mid) + ")" }));
    }
    if (lg.missing) host.appendChild(el("span", {}, el("i", { class: "sw hollow" }), " no value " + lg.missing));
  }

  // The three first components as a cloud that can be turned: PC1 to the right, PC2 up, PC3 into the screen, each
  // scaled to its own range (as the flat map scales its two axes), drawn back to front with a mild perspective.
  // quick: while the cloud is being turned, a large one is thinned so that it follows the finger.
  function drawPca3d(g, W, H, d, quick) {
    const k = d.k, n = feat.names.length;
    const ink = cssVar("--ink"), muted = cssVar("--muted"), grid = cssVar("--line"), axis = cssVar("--axis"), surf = cssVar("--panel");
    const rng = [0, 1, 2].map((c) => {
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i++) { const v = d.scores[i * k + c]; if (v < lo) lo = v; if (v > hi) hi = v; }
      return { mid: (lo + hi) / 2, half: ((hi - lo) / 2 || 1) * 1.05 };
    });
    const unit = (c, v) => (v - rng[c].mid) / rng[c].half;
    const cosY = Math.cos(pcaView.yaw), sinY = Math.sin(pcaView.yaw), cosP = Math.cos(pcaView.pitch), sinP = Math.sin(pcaView.pitch);
    const S = Math.min(W, H) * 0.3 * pcaView.zoom, cx = W / 2, cy = H / 2;
    const proj = (x, y, z) => {
      const x1 = x * cosY + z * sinY, z1 = -x * sinY + z * cosY, y2 = y * cosP - z1 * sinP, z2 = y * sinP + z1 * cosP, s = 4 / (4 - z2);
      return [cx + x1 * S * s, cy - y2 * S * s, z2, s];
    };
    // the box of the data, then its three edges from the low corner as the axes
    g.lineWidth = 1; g.strokeStyle = grid;
    for (let c = 0; c < 8; c++) for (let b = 0; b < 3; b++) {
      if (c & (1 << b)) continue;
      const p = proj(c & 1 ? 1 : -1, c & 2 ? 1 : -1, c & 4 ? 1 : -1), e = c | (1 << b), q = proj(e & 1 ? 1 : -1, e & 2 ? 1 : -1, e & 4 ? 1 : -1);
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
    }
    const o = proj(-1, -1, -1);
    g.strokeStyle = axis; g.fillStyle = muted; g.textBaseline = "middle";
    for (let c = 0; c < 3; c++) {
      const q = proj(c === 0 ? 1 : -1, c === 1 ? 1 : -1, c === 2 ? 1 : -1);
      g.lineWidth = 1.5; g.beginPath(); g.moveTo(o[0], o[1]); g.lineTo(q[0], q[1]); g.stroke();
      const label = "PC" + (c + 1) + " (" + (100 * d.explained[c]).toFixed(1) + "%)", tw = g.measureText(label).width;
      const dx = q[0] - o[0], dy = q[1] - o[1], len = Math.hypot(dx, dy) || 1;
      const lx = Math.max(tw / 2 + 2, Math.min(W - tw / 2 - 2, q[0] + (dx / len) * (tw / 2 + 6))), ly = Math.max(8, Math.min(H - 8, q[1] + (dy / len) * 12));
      g.textAlign = "center";
      g.lineWidth = 3; g.strokeStyle = surf; g.lineJoin = "round"; g.strokeText(label, lx, ly);
      g.fillText(label, lx, ly);
      g.strokeStyle = axis;
    }

    const lg = pcaView.colors.legend, col = pcaView.colors.col, cd = pcaView.colorData;
    const focusItem = pcaView.focus && lg.kind === "cat" ? lg.items.find((t) => t.key === pcaView.focus) : null;
    const focused = (i) => !focusItem || (focusItem.members ? focusItem.members.includes(cd.values[i]) : cd.values[i] === pcaView.focus);
    const step = quick && n > 2500 ? Math.ceil(n / 2500) : 1, pts = [];
    for (let i = 0; i < n; i += step) {
      const p = proj(unit(0, d.scores[i * k]), unit(1, d.scores[i * k + 1]), unit(2, d.scores[i * k + 2]));
      pts.push({ i, x: p[0], y: p[1], z: p[2], s: p[3] });
    }
    pts.sort((u, v) => u.z - v.z);                         // far points first, near ones on top
    const r = n > 1500 ? 2.5 : n > 400 ? 3.2 : 4;
    for (const on of [false, true]) {
      for (const p of pts) {
        if (focused(p.i) !== on) continue;
        g.globalAlpha = on ? 1 : 0.15;
        g.beginPath(); g.arc(p.x, p.y, Math.max(1.5, r * p.s * Math.sqrt(pcaView.zoom)), 0, 2 * Math.PI);
        if (col[p.i]) { g.fillStyle = col[p.i]; g.fill(); g.lineWidth = 1; g.strokeStyle = surf; g.stroke(); }
        else { g.lineWidth = 1.2; g.strokeStyle = muted; g.stroke(); }
      }
    }
    g.globalAlpha = 1;
    if (pcaView.arrows) {
      const z = [0, 1, 2].map((c) => unit(c, 0));           // where the origin of the components sits in the box
      // the strongest characteristics, leaving out those that point the same way as one already drawn (level, spread,
      // minimum and maximum of a series pull together and would cover each other)
      const all = d.features.map((f, j) => [f, d.loadings[j * k], d.loadings[j * k + 1], d.loadings[j * k + 2]])
        .sort((u, v) => Math.hypot(v[1], v[2], v[3]) - Math.hypot(u[1], u[2], u[3]));
      const ld = [];
      for (const t of all) {
        if (ld.length >= (W < 480 ? 4 : 6)) break;
        const nt = Math.hypot(t[1], t[2], t[3]);
        if (nt > 0 && ld.every((u) => (t[1] * u[1] + t[2] * u[2] + t[3] * u[3]) / (nt * Math.hypot(u[1], u[2], u[3])) < 0.9)) ld.push(t);
      }
      const m = Math.max(...ld.map((t) => Math.hypot(t[1], t[2], t[3])), 1e-9);
      const fit = 0.8;              // the longest arrow, in half box widths; arrows may leave the box, the labels stay on the canvas
      const c0 = proj(z[0], z[1], z[2]);
      g.strokeStyle = ink; g.fillStyle = ink; g.textBaseline = "middle";
      for (const [f, u, v, w] of ld) {
        const e = proj(z[0] + (u / m) * fit, z[1] + (v / m) * fit, z[2] + (w / m) * fit);
        g.lineWidth = 1.5; g.strokeStyle = ink;
        g.beginPath(); g.moveTo(c0[0], c0[1]); g.lineTo(e[0], e[1]); g.stroke();
        const label = W < 480 && f.length > 18 ? f.slice(0, 17) + "…" : f, tw = g.measureText(label).width;
        const right = e[0] >= c0[0] ? e[0] + 4 + tw <= W - 2 : e[0] - 4 - tw < 2;
        g.textAlign = right ? "left" : "right";
        g.lineWidth = 3; g.strokeStyle = surf; g.lineJoin = "round";
        const ty = Math.max(8, Math.min(H - 8, e[1]));
        g.strokeText(label, e[0] + (right ? 4 : -4), ty);
        g.fillText(label, e[0] + (right ? 4 : -4), ty);
      }
    }
    pcaView.pts = pts;
  }

  function drawPca(quick) {
    if (!feat || !feat.pca || !pcaView.colors) return;
    const cv = $("pcaCanvas"), d = feat.pca, k = d.k;
    const three = pcaView.dim === 3 && k >= 3;
    const W = Math.max(260, cv.parentElement.clientWidth), dpr = window.devicePixelRatio || 1;
    const H = Math.round(three ? Math.min(520, Math.max(300, W * 0.85)) : Math.min(480, Math.max(280, W * 0.7)));
    cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + "px";
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const ink = cssVar("--ink"), muted = cssVar("--muted"), grid = cssVar("--line"), axis = cssVar("--axis"), surf = cssVar("--panel");
    g.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    if (three) { drawPca3d(g, W, H, d, quick === true); return; }
    const [a, b] = pcaView.ax, L = 46, R = 14, T = 12, B = 40, n = feat.names.length;
    const xs = new Array(n), ys = new Array(n);
    for (let i = 0; i < n; i++) { xs[i] = d.scores[i * k + a]; ys[i] = d.scores[i * k + b]; }
    const pad = (lo, hi) => { const r = hi - lo || 1; return [lo - r * 0.05, hi + r * 0.05]; };
    const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
    const X = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R), Y = (v) => H - B - ((v - y0) / (y1 - y0)) * (H - T - B);
    g.strokeStyle = grid; g.lineWidth = 1; g.fillStyle = muted; g.textAlign = "right"; g.textBaseline = "middle";
    for (const t of ticks(y0, y1, 5).values) { g.beginPath(); g.moveTo(L, Y(t) + 0.5); g.lineTo(W - R, Y(t) + 0.5); g.stroke(); g.fillText(fmt(t), L - 6, Y(t)); }
    g.textAlign = "center"; g.textBaseline = "top";
    for (const t of ticks(x0, x1, W < 480 ? 4 : 6).values) { g.beginPath(); g.moveTo(X(t) + 0.5, T); g.lineTo(X(t) + 0.5, H - B); g.stroke(); g.fillText(fmt(t), X(t), H - B + 6); }
    g.strokeStyle = axis; g.beginPath(); g.moveTo(L, H - B + 0.5); g.lineTo(W - R, H - B + 0.5); g.stroke();
    const pct = (e) => (100 * e).toFixed(1) + "%";
    g.fillStyle = muted; g.fillText("PC" + (a + 1) + " (" + pct(d.explained[a]) + ")", (L + W - R) / 2, H - 16);
    g.save(); g.translate(12, (T + H - B) / 2); g.rotate(-Math.PI / 2); g.textBaseline = "middle"; g.fillText("PC" + (b + 1) + " (" + pct(d.explained[b]) + ")", 0, 0); g.restore();

    const lg = pcaView.colors.legend, col = pcaView.colors.col, cd = pcaView.colorData;
    const focused = (i) => {
      if (!pcaView.focus || lg.kind !== "cat") return true;
      const it = lg.items.find((t) => t.key === pcaView.focus);
      return it.members ? it.members.includes(cd.values[i]) : cd.values[i] === pcaView.focus;
    };
    const pts = xs.map((x, i) => ({ i, x: X(x), y: Y(ys[i]) }));
    const r = n > 1500 ? 2.5 : n > 400 ? 3.2 : 4;
    for (const on of [false, true]) {
      for (const p of pts) {
        if (focused(p.i) !== on) continue;
        g.globalAlpha = on ? 1 : 0.15;
        g.beginPath(); g.arc(p.x, p.y, r, 0, 2 * Math.PI);
        if (col[p.i]) { g.fillStyle = col[p.i]; g.fill(); g.lineWidth = 1; g.strokeStyle = surf; g.stroke(); }
        else { g.lineWidth = 1.2; g.strokeStyle = muted; g.stroke(); }
      }
    }
    g.globalAlpha = 1;
    if (pcaView.arrows) {
      const p = d.features.length, cx = X(0), cy = Y(0);
      const ext = 0.8 * Math.min((Math.max(Math.abs(x0), Math.abs(x1)) * (W - L - R)) / (x1 - x0), (Math.max(Math.abs(y0), Math.abs(y1)) * (H - T - B)) / (y1 - y0));
      const ld = d.features.map((f, j) => [f, d.loadings[j * k + a], d.loadings[j * k + b]])
        .sort((u, v) => Math.hypot(v[1], v[2]) - Math.hypot(u[1], u[2])).slice(0, W < 480 ? 4 : 6);
      const m = Math.max(...ld.map((t) => Math.hypot(t[1], t[2])), 1e-9);
      // the longest arrow length for which every arrow stays inside the plot
      let fit = ext;
      for (const [, u, v] of ld) {
        const dx = u / m, dy = -v / m;
        if (Math.abs(dx) > 1e-9) fit = Math.min(fit, 0.92 * (dx > 0 ? W - R - cx : cx - L) / Math.abs(dx));
        if (Math.abs(dy) > 1e-9) fit = Math.min(fit, 0.92 * (dy > 0 ? H - B - cy : cy - T) / Math.abs(dy));
      }
      g.strokeStyle = ink; g.fillStyle = ink; g.lineWidth = 1.5; g.textBaseline = "middle";
      for (const [f, u, v] of ld) {
        const ex = cx + (u / m) * fit, ey = cy - (v / m) * fit;
        g.beginPath(); g.moveTo(cx, cy); g.lineTo(ex, ey); g.stroke();
        const label = W < 480 && f.length > 18 ? f.slice(0, 17) + "…" : f, tw = g.measureText(label).width;
        const right = u >= 0 ? ex + 4 + tw <= W - 2 : ex - 4 - tw < 2;      // keep the label inside the canvas
        g.textAlign = right ? "left" : "right";
        g.lineWidth = 3; g.strokeStyle = surf; g.lineJoin = "round";        // a halo keeps the label readable over the points
        g.strokeText(label, ex + (right ? 4 : -4), ey);
        g.fillText(label, ex + (right ? 4 : -4), ey);
        g.lineWidth = 1.5; g.strokeStyle = ink;
      }
    }
    pcaView.pts = pts;
  }

  function renderCharacteristics() {
    const n = feat.names.length;
    $("charStatus").textContent = n + " series analysed in " + (feat.seconds < 1 ? "under a second" : feat.seconds.toFixed(1) + " s") + ": " +
      feat.columns.length + " characteristics per series (" + feat.sets.basic + " descriptive ones, the " + feat.sets.catch22 + " of catch22" +
      (feat.tsfeatures ? (feat.sets.tsfeatures ? " and " + feat.sets.tsfeatures + " of tsfeatures" : "; the tsfeatures set gave no values for this file") : "") + ").";
    const counts = {};
    feat.classes.forEach((c) => { const key = c || "undefined"; counts[key] = (counts[key] || 0) + 1; });
    const tiles = $("classTiles");
    tiles.textContent = "";
    for (const c of CLASSES.concat(counts.undefined ? ["undefined"] : [])) {
      tiles.appendChild(el("div", { class: "tile" }, el("b", { text: String(counts[c] || 0) }),
                           el("span", { text: c + " (" + ((100 * (counts[c] || 0)) / n).toFixed(0) + "%)" })));
    }
    // summary of every characteristic
    const ft = $("featTable");
    ft.textContent = "";
    ft.appendChild(el("tr", {}, ...["Characteristic", "Mean", "Median", "Minimum", "Maximum", "No value"].map((t) => el("th", { text: t }))));
    for (const name of feat.columns) {
      const v = column(name).filter(Number.isFinite).sort((x, y) => x - y), m = v.length;
      const med = m ? (m % 2 ? v[(m - 1) / 2] : (v[m / 2 - 1] + v[m / 2]) / 2) : NaN;
      const cells = [name, fmt(m ? v.reduce((s, x) => s + x, 0) / m : NaN), fmt(med), fmt(m ? v[0] : NaN), fmt(m ? v[m - 1] : NaN), String(n - m)];
      ft.appendChild(el("tr", {}, ...cells.map((t) => el("td", { text: t }))));
    }
    $("charBody").hidden = false;
    const d = feat.pca;
    $("pcaBox").hidden = !d;
    $("pcaCsv").hidden = !d;
    if (!d) { $("pcaHint").textContent = "A map cannot be drawn for this file: " + feat.pcaError + "."; return; }
    $("pcaHint").textContent = "Each point is one series; series with similar characteristics lie close together. The axes are the principal " +
      "components of the " + d.features.length + " standardised characteristics. Explained variance: " +
      d.explained.map((e, i) => "PC" + (i + 1) + " " + (100 * e).toFixed(1) + "%").join(", ") + "." +
      (d.dropped.length ? " Left out (constant or mostly missing): " + d.dropped.join(", ") + "." : "");
    for (const [id, which] of [["pcaX", 0], ["pcaY", 1]]) {
      const sel = $(id);
      sel.textContent = "";
      for (let i = 0; i < d.k; i++) sel.appendChild(el("option", { value: String(i), text: "PC" + (i + 1) }));
      pcaView.ax[which] = Math.min(pcaView.ax[which], d.k - 1);
      sel.value = String(pcaView.ax[which]);
    }
    $("pcaDimRow").hidden = d.k < 3;                       // a third component needs at least three usable characteristics
    setPcaDim(d.k < 3 ? 2 : pcaView.dim, true);
    const lt = $("loadTable");
    lt.textContent = "";
    const head = el("tr", {});
    for (let c = 0; c < d.k; c++) { head.appendChild(el("th", { text: "PC" + (c + 1) })); head.appendChild(el("th", { text: "loading" })); }
    lt.appendChild(head);
    const tops = [];
    for (let c = 0; c < d.k; c++) {
      tops.push(d.features.map((f, j) => [f, d.loadings[j * d.k + c]]).sort((u, v) => Math.abs(v[1]) - Math.abs(u[1])).slice(0, 6));
    }
    for (let r = 0; r < 6; r++) {
      const tr = el("tr", {});
      for (let c = 0; c < d.k; c++) {
        const t = tops[c][r];
        tr.appendChild(el("td", { text: t ? t[0] : "" }));
        tr.appendChild(el("td", { text: t ? t[1].toFixed(2) : "" }));
      }
      lt.appendChild(tr);
    }
    updatePcaColor();
  }

  function updatePcaColor() {
    if (!feat || !feat.pca) return;
    pcaColorOptions();
    pcaView.colorData = pcaColorData();
    pcaView.colors = pcaColors(pcaView.colorData);
    pcaView.focus = null;
    const three = pcaView.dim === 3 && feat.pca.k >= 3;
    $("pcaCanvas").setAttribute("aria-label", "Scatter of " + feat.names.length + " series on " + (three ? "three" : "two") +
      " principal components, coloured by " + pcaView.colorData.label + ".");
    $("pcaNote").textContent = (three ? "Drag to turn the map, pinch (or Ctrl and the mouse wheel) to zoom. " : "") +
      (pcaView.colorData.partial ? "Series outside the last run have no value and are drawn hollow. " : "") +
      (result ? "Tap a point to see that series in the forecast chart below." : "After a test run you can also colour the map by the error of each series.");
    renderPcaLegend();
    drawPca();
  }

  function setPcaDim(dim, quiet) {
    pcaView.dim = dim;
    const three = dim === 3;
    $("pca2d").setAttribute("aria-pressed", String(!three));
    $("pca3d").setAttribute("aria-pressed", String(three));
    $("pcaReset").hidden = !three;
    $("pcaXBox").hidden = three;
    $("pcaYBox").hidden = three;
    $("pcaCanvas").style.touchAction = three ? "none" : "pan-y";     // in 3D a finger on the map turns it
    $("pcaCanvas").style.cursor = three ? "grab" : "crosshair";
    $("pcaTip").hidden = true;
    if (!quiet) updatePcaColor();
  }
  $("pca2d").addEventListener("click", () => setPcaDim(2));
  $("pca3d").addEventListener("click", () => setPcaDim(3));
  $("pcaReset").addEventListener("click", () => { Object.assign(pcaView, PCA_3D); drawPca(); });

  // turning and zooming the 3D map
  const pcaDrag = { at: new Map(), moved: 0, pinch: 0, frame: 0 };
  // one redraw per animation frame; the timer covers a frame the browser drops (a page in the background), which
  // would otherwise leave the map waiting for it for good
  const pcaSoon = () => {
    if (pcaDrag.frame) return;
    const run = () => {
      if (!pcaDrag.frame) return;
      cancelAnimationFrame(pcaDrag.frame); clearTimeout(pcaDrag.timer);
      pcaDrag.frame = 0;
      drawPca(pcaDrag.at.size > 0);
    };
    pcaDrag.frame = requestAnimationFrame(run);
    pcaDrag.timer = setTimeout(run, 150);
  };
  const pcaSpan = () => { const p = [...pcaDrag.at.values()]; return Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y); };
  $("pcaCanvas").addEventListener("pointerdown", (ev) => {
    if (pcaView.dim !== 3) return;
    try { ev.currentTarget.setPointerCapture(ev.pointerId); } catch (e) { /* the pointer is already gone */ }
    pcaDrag.at.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    if (pcaDrag.at.size === 1) pcaDrag.moved = 0;
    if (pcaDrag.at.size === 2) pcaDrag.pinch = pcaSpan();
  });
  function pcaTurn(ev) {            // -> true when the pointer is turning or zooming the map
    const was = pcaDrag.at.get(ev.pointerId);
    if (pcaView.dim !== 3 || !was) return false;
    const dx = ev.clientX - was.x, dy = ev.clientY - was.y;
    pcaDrag.at.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    pcaDrag.moved += Math.abs(dx) + Math.abs(dy);
    if (pcaDrag.at.size >= 2) {
      const span = pcaSpan();
      if (pcaDrag.pinch > 0 && span > 0) pcaView.zoom = Math.max(0.5, Math.min(4, (pcaView.zoom * span) / pcaDrag.pinch));
      pcaDrag.pinch = span;
      pcaDrag.moved += 10;
    } else {
      pcaView.yaw += dx * 0.01;
      pcaView.pitch = Math.max(-1.5, Math.min(1.5, pcaView.pitch + dy * 0.01));
    }
    if (pcaDrag.moved > 4) { $("pcaTip").hidden = true; pcaSoon(); }
    return pcaDrag.moved > 4;
  }
  for (const type of ["pointerup", "pointercancel"]) $("pcaCanvas").addEventListener(type, (ev) => {
    if (pcaDrag.at.delete(ev.pointerId) && !pcaDrag.at.size && pcaDrag.moved > 4) pcaSoon();     // the full cloud again
  });
  $("pcaCanvas").addEventListener("wheel", (ev) => {
    if (pcaView.dim !== 3 || !ev.ctrlKey) return;
    ev.preventDefault();
    pcaView.zoom = Math.max(0.5, Math.min(4, pcaView.zoom * Math.exp(-ev.deltaY * 0.002)));
    pcaSoon();
  }, { passive: false });
  $("pcaColor").addEventListener("change", (e) => { pcaView.color = e.target.value; updatePcaColor(); });
  $("pcaX").addEventListener("change", (e) => { pcaView.ax[0] = parseInt(e.target.value, 10); drawPca(); });
  $("pcaY").addEventListener("change", (e) => { pcaView.ax[1] = parseInt(e.target.value, 10); drawPca(); });
  $("pcaArrows").addEventListener("change", (e) => { pcaView.arrows = e.target.checked; drawPca(); });
  window.addEventListener("resize", () => { clearTimeout(drawPca.timer); drawPca.timer = setTimeout(() => drawPca(), 150); });
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", updatePcaColor);

  function nearestPoint(ev) {
    const cv = $("pcaCanvas"), r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = null, bd = ev.pointerType === "touch" ? 324 : 100;
    for (const p of pcaView.pts) { const dd = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y); if (dd < bd) { bd = dd; best = p; } }
    return { best, x, y, width: r.width };
  }
  function pcaHover(ev) {
    const tip = $("pcaTip"), hit = nearestPoint(ev);
    if (!hit.best || !feat || !feat.pca) { tip.hidden = true; return null; }
    const i = hit.best.i, d = feat.pca, c = pcaView.colorData, v = c.values[i];
    tip.textContent = "";
    tip.appendChild(el("b", { text: feat.names[i] }));
    const axes = pcaView.dim === 3 && d.k >= 3 ? [0, 1, 2] : pcaView.ax;
    tip.appendChild(el("div", { text: axes.map((c) => "PC" + (c + 1) + " " + d.scores[i * d.k + c].toFixed(2)).join("   ") }));
    tip.appendChild(el("div", { text: c.label + ": " + (v == null || (typeof v === "number" && !Number.isFinite(v)) ? "no value" : typeof v === "number" ? fmt(v) : v) }));
    tip.hidden = false;
    tip.style.left = Math.max(0, Math.min(hit.width - tip.offsetWidth, hit.x + (hit.x > hit.width / 2 ? -tip.offsetWidth - 12 : 12))) + "px";
    tip.style.top = Math.max(0, hit.y - 10) + "px";
    return i;
  }
  $("pcaCanvas").addEventListener("pointermove", (ev) => { if (!pcaTurn(ev)) pcaHover(ev); });
  $("pcaCanvas").addEventListener("pointerleave", () => { $("pcaTip").hidden = true; });
  $("pcaCanvas").addEventListener("click", (ev) => {
    if (pcaView.dim === 3 && pcaDrag.moved > 4) return;       // the end of a turn, not a tap on a point
    const i = pcaHover(ev);
    if (i === null || !result) return;
    for (let r = 1; r < result.errors.length; r++) {
      if (result.errors[r][0] === feat.names[i]) { requestSeries(r - 1); $("pcaNote").textContent = feat.names[i] + " is now shown in the forecast chart below."; return; }
    }
    $("pcaNote").textContent = feat.names[i] + " was not part of the last run.";
  });

  function csvCell(s) { return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }
  function featuresCsv() {
    const p = feat.columns.length, lines = ["series," + feat.columns.join(",") + ",demand_class"];
    feat.names.forEach((nm, i) => {
      const row = [csvCell(nm)];
      for (let j = 0; j < p; j++) { const v = feat.table[i * p + j]; row.push(Number.isFinite(v) ? String(v) : ""); }
      row.push(feat.classes[i] || "");
      lines.push(row.join(","));
    });
    return lines.join("\n") + "\n";
  }
  function saveText(name, text) {
    const a = el("a", { download: name });
    a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 30000);
  }
  $("featCsv").addEventListener("click", () => { if (feat) saveText(data.id + "_characteristics.csv", featuresCsv()); });
  $("pcaCsv").addEventListener("click", () => {
    if (!feat || !feat.pca) return;
    const d = feat.pca, lines = ["series," + d.explained.map((e, i) => "PC" + (i + 1)).join(",")];
    feat.names.forEach((nm, i) => lines.push([csvCell(nm)].concat(Array.from(d.scores.subarray(i * d.k, (i + 1) * d.k), String)).join(",")));
    saveText(data.id + "_pca_scores.csv", lines.join("\n") + "\n");
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
    } else if (msg.type === "features") {
      if (msg.token === featToken) { feat = msg; renderCharacteristics(); }
    } else if (msg.type === "featuresProgress") {
      if (msg.token === featToken) $("charStatus").textContent = "Computing the tsfeatures set: series " + msg.done + " of " + msg.total + "…";
    } else if (msg.type === "error") {
      if (msg.stage === "features") { $("charStatus").textContent = "The characteristics could not be computed: " + msg.message; $("charRun").hidden = false; }
      else if (msg.stage === "load") { $("dataInfo").textContent = msg.message; $("dataInfo").className = "note error"; }
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
