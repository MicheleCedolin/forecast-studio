// Runs tsfcore (WebAssembly) off the main thread. Messages from the page:
//   { type: "load", bytes, name }        keep a .tsf file (in the module's in-memory file system) and report its summary
//   { type: "series", index, token }     the values of one series of the loaded file (for the chart)
//   { type: "features", token, tsfeatures }   characteristics of every series of the loaded file and their PCA;
//                                        tsfeatures adds that set (slower; "featuresProgress" messages report it)
//   { type: "run", config, pilot }       tsf_run with the "key=value" lines in config; pilot > 0 first times that many
//                                        series and reports it, so the page can show an estimate before the full run
const V = self.location.search;   // "?v=<commit id>" from the page: the engine files are fetched with the same version
importScripts("tsfcore.js" + V, "features.js" + V);

let core = null;
let handle = 0;        // tsf_io handle of the loaded file
let dataPath = null;

const ready = createTsfCore({ locateFile: (path) => path + V }).then((m) => {
  core = m;
  for (const dir of ["/data", "/runs"]) {
    try { m.FS.mkdir(dir); } catch (e) { /* exists */ }
  }
  postMessage({ type: "ready", version: m.UTF8ToString(m._tsf_version()).split(" (")[0] });
});

function cstr(text) {
  const n = core.lengthBytesUTF8(text) + 1;
  const p = core._malloc(n);
  core.stringToUTF8(text, p, n);
  return p;
}

function load(bytes, name) {
  if (handle) { core._tsf_io_close(handle); handle = 0; }
  if (dataPath) { try { core.FS.unlink(dataPath); } catch (e) { /* gone */ } }
  dataPath = "/data/" + name.replace(/[^A-Za-z0-9._-]/g, "_");
  core.FS.writeFile(dataPath, new Uint8Array(bytes));
  const msg = core._malloc(1024), p = cstr(dataPath);
  try {
    handle = core._tsf_io_open(p, -1n, msg, 1024);
    if (!handle) throw new Error(core.UTF8ToString(msg) || "The file could not be read.");
    return { stats: JSON.parse(core.UTF8ToString(core._tsf_io_stats_json(handle))),
             header: JSON.parse(core.UTF8ToString(core._tsf_io_header_json(handle))) };
  } finally {
    core._free(msg);
    core._free(p);
  }
}

function series(index) {
  const lenPtr = core._malloc(8);
  try {
    const ptr = core._tsf_io_series(handle, BigInt(index), lenPtr);
    const len = Number(core.HEAP64[lenPtr >> 3]);
    return ptr ? new Float64Array(core.HEAPF64.buffer, ptr, len).slice() : new Float64Array(0);
  } finally {
    core._free(lenPtr);
  }
}

// One tsf_run; returns { name: text } of the files it wrote and removes them from the in-memory file system.
function runOnce(lines) {
  const runDir = core._malloc(1024), msg = core._malloc(2048);
  const pPath = cstr(dataPath), pOut = cstr("/runs"), pCfg = cstr(lines.join("\n"));
  try {
    const rc = core._tsf_run(pPath, pOut, pCfg, runDir, 1024, msg, 2048);
    if (rc !== 0) throw new Error(core.UTF8ToString(msg) || "The run failed (code " + rc + ").");
    const dir = core.UTF8ToString(runDir);
    const files = {};
    for (const f of core.FS.readdir(dir)) {
      if (f === "." || f === "..") continue;
      files[f] = core.FS.readFile(dir + "/" + f, { encoding: "utf8" });
      core.FS.unlink(dir + "/" + f);
    }
    core.FS.rmdir(dir);
    return files;
  } finally {
    for (const p of [runDir, msg, pPath, pOut, pCfg]) core._free(p);
  }
}

onmessage = async (event) => {
  await ready;
  const m = event.data;
  try {
    if (m.type === "load") {
      postMessage(Object.assign({ type: "loaded", name: m.name }, load(m.bytes, m.name)));
    } else if (m.type === "series") {
      const values = series(m.index);
      postMessage({ type: "series", token: m.token, index: m.index, values }, [values.buffer]);
    } else if (m.type === "features") {
      // characteristics of every series and their principal components (features.js)
      const t0 = performance.now();
      const header = JSON.parse(core.UTF8ToString(core._tsf_io_header_json(handle)));
      let f = TsfFeatures.compute(core, handle, header);
      const sets = { basic: TsfFeatures.BASIC.length, catch22: f.columns.length - TsfFeatures.BASIC.length, tsfeatures: 0 };
      if (m.tsfeatures) {
        const t = TsfFeatures.tsfeatures(core, handle, header, (done, total) => postMessage({ type: "featuresProgress", token: m.token, done, total }));
        sets.tsfeatures = t.columns.length;
        f = TsfFeatures.join(f, t);
      }
      let pca = null, pcaError = null;
      try { pca = TsfFeatures.pca(f.table, f.names.length, f.columns); } catch (err) { pcaError = String(err.message || err); }
      postMessage({ type: "features", token: m.token, names: f.names, columns: f.columns, table: f.table, classes: f.classes, pca, pcaError, sets, tsfeatures: !!m.tsfeatures,
                    seconds: (performance.now() - t0) / 1000 });
    } else if (m.type === "run") {
      if (m.pilot > 0) {
        const t0 = performance.now();
        runOnce(m.config.filter((l) => !l.startsWith("max_series=")).concat("max_series=" + m.pilot));
        postMessage({ type: "estimate", seconds: (performance.now() - t0) / 1000, series: m.pilot });
      }
      const started = performance.now();
      const files = runOnce(m.config);
      postMessage({ type: "done", files, seconds: (performance.now() - started) / 1000 });
    }
  } catch (err) {
    postMessage({ type: "error", stage: m.type, message: String(err && err.message ? err.message : err) });
  }
};
