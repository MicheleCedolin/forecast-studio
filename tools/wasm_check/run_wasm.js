// Runs jobs.json through web/tsfcore.js (WebAssembly) in Node, calling tsf_run exactly as web/worker.js runOnce does.
// usage: node run_wasm.js jobs.json outdir [job-name-filter]
const fs = require("fs"), path = require("path");
const REPO = "/home/user/forecast-studio";
const createTsfCore = require(REPO + "/web/tsfcore.js");
const [jobsFile, outRoot, filt] = process.argv.slice(2);
const jobs = JSON.parse(fs.readFileSync(jobsFile, "utf8")).filter((j) => !filt || new RegExp(filt).test(j.name));
(async () => {
  const core = await createTsfCore();
  for (const dir of ["/data", "/runs"]) { try { core.FS.mkdir(dir); } catch (e) {} }
  const cstr = (t) => { const n = core.lengthBytesUTF8(t) + 1, p = core._malloc(n); core.stringToUTF8(t, p, n); return p; };
  const info = { version: core.UTF8ToString(core._tsf_version()), methods: core.UTF8ToString(core._tsf_methods()) };
  // which registry names does tsf_forecast_batch know? (rc -1 = unknown method)
  const names = fs.readFileSync(path.join(path.dirname(jobsFile), "registry_names.txt"), "utf8").split(",").filter(Boolean)
    .concat(["catboost", "lightgbm", "no_such_method"]);
  const n = 40, y = core._malloc(8 * n), off = core._malloc(16), per = core._malloc(16), out = core._malloc(8 * 4), fl = core._malloc(4), msg = core._malloc(256);
  for (let i = 0; i < n; i++) core.HEAPF64[(y >> 3) + i] = 10 + (i % 4) + 0.1 * i;
  core.HEAP64[off >> 3] = 0n; core.HEAP64[(off >> 3) + 1] = BigInt(n); core.HEAPF64[per >> 3] = 4; core.HEAPF64[(per >> 3) + 1] = 8;
  info.batch_rc = {};
  for (const nm of names) { const p = cstr(nm); info.batch_rc[nm] = core._tsf_forecast_batch(p, y, off, 1, 4, 0, per, nm === "dshw" ? 2 : 1, 1, out, fl, msg, 256); core._free(p); }
  fs.mkdirSync(outRoot, { recursive: true });
  fs.writeFileSync(path.join(outRoot, "info.json"), JSON.stringify(info, null, 1));
  const loaded = {};
  for (const job of jobs) {
    const dst = "/data/" + path.basename(job.file);
    if (!loaded[dst]) { core.FS.writeFile(dst, new Uint8Array(fs.readFileSync(job.file))); loaded[dst] = 1; }
    const runDir = core._malloc(1024), m = core._malloc(2048), pPath = cstr(dst), pOut = cstr("/runs"), pCfg = cstr(job.config.join("\n"));
    const t0 = performance.now();
    const rc = core._tsf_run(pPath, pOut, pCfg, runDir, 1024, m, 2048);
    const sec = (performance.now() - t0) / 1000;
    const od = path.join(outRoot, job.name); fs.mkdirSync(od, { recursive: true });
    const res = { rc, seconds: sec, message: core.UTF8ToString(m) };
    if (rc === 0) {
      const dir = core.UTF8ToString(runDir);
      for (const f of core.FS.readdir(dir)) {
        if (f === "." || f === "..") continue;
        if (["forecasts.csv", "fallbacks.csv", "manifest.json"].includes(f)) fs.writeFileSync(path.join(od, f), core.FS.readFile(dir + "/" + f, { encoding: "utf8" }));
        core.FS.unlink(dir + "/" + f);
      }
      core.FS.rmdir(dir);
    }
    fs.writeFileSync(path.join(od, "result.json"), JSON.stringify(res));
    console.log(job.name, rc, sec.toFixed(2) + "s", res.message);
    for (const p of [runDir, m, pPath, pOut, pCfg]) core._free(p);
  }
})();
