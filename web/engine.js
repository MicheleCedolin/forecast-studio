// Forecast Studio, web edition: calls into the WebAssembly build of tsfcore. `M` is the module instance created by
// tsfcore.mjs. Used by the worker in the page and by the Node tests (tools/web/check.mjs).
function pack(M, series) {
  const n = series.length;
  let total = 0; for (const s of series) total += s.length;
  const pv = M._malloc(Math.max(8, total * 8)), po = M._malloc((n + 1) * 8);
  const values = new Float64Array(M.HEAPU8.buffer, pv, total), offsets = new BigInt64Array(M.HEAPU8.buffer, po, n + 1);
  let k = 0;
  series.forEach((s, i) => { offsets[i] = BigInt(k); values.set(s, k); k += s.length; });
  offsets[n] = BigInt(k);
  return { pv, po };
}

const cstr = (M, s) => { const n = M.lengthBytesUTF8(s) + 1, p = M._malloc(n); M.stringToUTF8(s, p, n); return p; };

export function forecastLocal(M, { method, h, periods, series }) {
  const n = series.length, { pv, po } = pack(M, series);
  const pp = M._malloc(periods.length * 8), pout = M._malloc(n * h * 8), pf = M._malloc(Math.max(4, n * 4)), pm = M._malloc(512), pn = cstr(M, method);
  new Float64Array(M.HEAPU8.buffer, pp, periods.length).set(periods);
  try {
    const rc = M._tsf_forecast_batch(pn, pv, po, n, h, 0, pp, periods.length, 1, pout, pf, pm, 512);
    if (rc === -1) throw new Error(`The forecasting core does not know the method '${method}'.`);
    if (rc !== 0) throw new Error(`The forecasting core rejected the job (code ${rc}).`);
    // Views are taken after the call: the WebAssembly memory may have grown, which replaces its buffer.
    return { forecasts: new Float64Array(M.HEAPU8.buffer, pout, n * h).slice(), flags: new Int32Array(M.HEAPU8.buffer, pf, n).slice(), msg: M.UTF8ToString(pm) };
  } finally { for (const p of [pv, po, pp, pout, pf, pm, pn]) M._free(p); }
}

export function forecastGlobal(M, { lag, h, series }) {
  const n = series.length, { pv, po } = pack(M, series);
  const pout = M._malloc(n * h * 8), pa = M._malloc(Math.max(4, lag * 4)), pna = M._malloc(4), pm = M._malloc(512);
  try {
    const rc = M._tsf_pooled_regression(pv, po, n, lag, h, 0, pout, pa, pna, pm, 512);
    if (rc !== 0) throw new Error(M.UTF8ToString(pm) || "Pooled regression failed.");
    const na = new Int32Array(M.HEAPU8.buffer, pna, 1)[0];
    return { forecasts: new Float64Array(M.HEAPU8.buffer, pout, n * h).slice(), aliased: Array.from(new Int32Array(M.HEAPU8.buffer, pa, na)) };
  } finally { for (const p of [pv, po, pout, pa, pna, pm]) M._free(p); }
}
