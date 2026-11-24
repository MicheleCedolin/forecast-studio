// Forecast Studio, web edition: forecasting worker. Loads the WebAssembly build of tsfcore (cpp/build_wasm.sh) and
// answers one request at a time. The page starts several of these and hands each a few series at a time, so all CPU
// cores are used and a run can be stopped by terminating the workers.
//   { id, kind: "local",  method, h, periods, series: [Float64Array] } -> { id, ok, forecasts (n x h), flags, msg }
//   { id, kind: "global", lag, h, series }                              -> { id, ok, forecasts (n x h), aliased }
import createModule from "./tsfcore.mjs";
import { forecastLocal, forecastGlobal } from "./engine.js";

const ready = createModule();
ready.then(M => postMessage({ ready: true, version: M.UTF8ToString(M._tsf_version()).split(" (")[0] }),
           e => postMessage({ ready: false, error: String(e && e.message || e) }));

onmessage = async ev => {
  const req = ev.data;
  try {
    const M = await ready;
    const res = req.kind === "global" ? forecastGlobal(M, req) : forecastLocal(M, req);
    postMessage({ id: req.id, ok: true, ...res }, [res.forecasts.buffer]);
  } catch (e) {
    postMessage({ id: req.id, ok: false, error: String(e && e.message || e) });
  }
};
