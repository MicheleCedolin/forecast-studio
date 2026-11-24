// Runs tsfcore (WebAssembly) off the main thread: the page posts a .tsf file and the settings, the worker writes the file
// into the module's in-memory file system, calls tsf_run and posts the result files back.
importScripts("tsfcore.js");

let core = null;
const ready = createTsfCore().then((m) => {
  core = m;
  for (const dir of ["/data", "/runs"]) {
    try { m.FS.mkdir(dir); } catch (e) { /* exists */ }
  }
  postMessage({ type: "ready", version: m.UTF8ToString(m._tsf_version()).split(" (")[0], methods: m.UTF8ToString(m._tsf_methods()).split(",") });
});

function cstr(text) {
  const n = core.lengthBytesUTF8(text) + 1;
  const p = core._malloc(n);
  core.stringToUTF8(text, p, n);
  return p;
}

onmessage = async (event) => {
  await ready;
  const { bytes, name, config } = event.data;
  const path = "/data/" + name.replace(/[^A-Za-z0-9._-]/g, "_");
  const started = performance.now();
  const runDir = core._malloc(1024), msg = core._malloc(2048);
  const pPath = cstr(path), pOut = cstr("/runs"), pCfg = cstr(config.join("\n"));
  try {
    core.FS.writeFile(path, new Uint8Array(bytes));
    const rc = core._tsf_run(pPath, pOut, pCfg, runDir, 1024, msg, 2048);
    if (rc !== 0) {
      postMessage({ type: "error", message: core.UTF8ToString(msg) || "The run failed (code " + rc + ")." });
      return;
    }
    const dir = core.UTF8ToString(runDir);
    const files = {};
    for (const f of core.FS.readdir(dir)) {
      if (f === "." || f === "..") continue;
      files[f] = core.FS.readFile(dir + "/" + f, { encoding: "utf8" });
      core.FS.unlink(dir + "/" + f);
    }
    core.FS.rmdir(dir);
    postMessage({ type: "done", files, seconds: (performance.now() - started) / 1000 });
  } catch (err) {
    postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
  } finally {
    try { core.FS.unlink(path); } catch (e) { /* not written */ }
    for (const p of [runDir, msg, pPath, pOut, pCfg]) core._free(p);
  }
};
