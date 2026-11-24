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
             stats: st, header: msg.header };
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
    updatePcaColor();              // the map can now be coloured by this run's errors
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
  const pcaView = { ax: [0, 1], arrows: true, color: "class", focus: null, pts: [], colors: null };
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
    if (result) {
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
    if (kind === "run" && result) {
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

  function drawPca() {
    if (!feat || !feat.pca || !pcaView.colors) return;
    const cv = $("pcaCanvas"), d = feat.pca, k = d.k;
    const W = Math.max(260, cv.parentElement.clientWidth), H = Math.round(Math.min(480, Math.max(280, W * 0.7))), dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + "px";
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const ink = cssVar("--ink"), muted = cssVar("--muted"), grid = cssVar("--line"), axis = cssVar("--axis"), surf = cssVar("--panel");
    g.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
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
    $("pcaCanvas").setAttribute("aria-label", "Scatter of " + feat.names.length + " series on two principal components, coloured by " + pcaView.colorData.label + ".");
    $("pcaNote").textContent = (pcaView.colorData.partial ? "Series outside the last run have no value and are drawn hollow. " : "") +
      (result ? "Tap a point to see that series in the forecast chart below." : "After a run you can also colour the map by the error of each series.");
    renderPcaLegend();
    drawPca();
  }
  $("pcaColor").addEventListener("change", (e) => { pcaView.color = e.target.value; updatePcaColor(); });
  $("pcaX").addEventListener("change", (e) => { pcaView.ax[0] = parseInt(e.target.value, 10); drawPca(); });
  $("pcaY").addEventListener("change", (e) => { pcaView.ax[1] = parseInt(e.target.value, 10); drawPca(); });
  $("pcaArrows").addEventListener("change", (e) => { pcaView.arrows = e.target.checked; drawPca(); });
  window.addEventListener("resize", () => { clearTimeout(drawPca.timer); drawPca.timer = setTimeout(drawPca, 150); });
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
    tip.appendChild(el("div", { text: [0, 1].map((w) => "PC" + (pcaView.ax[w] + 1) + " " + d.scores[i * d.k + pcaView.ax[w]].toFixed(2)).join("   ") }));
    tip.appendChild(el("div", { text: c.label + ": " + (v == null || (typeof v === "number" && !Number.isFinite(v)) ? "no value" : typeof v === "number" ? fmt(v) : v) }));
    tip.hidden = false;
    tip.style.left = Math.max(0, Math.min(hit.width - tip.offsetWidth, hit.x + (hit.x > hit.width / 2 ? -tip.offsetWidth - 12 : 12))) + "px";
    tip.style.top = Math.max(0, hit.y - 10) + "px";
    return i;
  }
  $("pcaCanvas").addEventListener("pointermove", pcaHover);
  $("pcaCanvas").addEventListener("pointerleave", () => { $("pcaTip").hidden = true; });
  $("pcaCanvas").addEventListener("click", (ev) => {
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
