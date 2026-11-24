// Forecast Studio, web edition: data reading, train/test split and error measures.
// Pure functions with no browser dependency, so the same file runs in the page and in the Node tests
// (tests/test_web_edition.py compares every function here with the Python package).
// Definitions follow src/tsfstudio (tsf/reader.py, seasonality.py, runner.py, metrics.py).

export const FREQUENCIES = ["4_seconds", "minutely", "10_minutes", "15_minutes", "half_hourly", "hourly",
  "daily", "weekly", "monthly", "quarterly", "yearly"];

// Seasonal periods per frequency (the study's SEASONALITY_VALS; several periods below daily).
export const SEASONALITY = {
  "4_seconds": [21600, 151200, 7889400], minutely: [1440, 10080, 525960], "10_minutes": [144, 1008, 52596],
  "15_minutes": [96, 672, 35064], half_hourly: [48, 336, 17532], hourly: [24, 168, 8766],
  daily: [7], weekly: [365.25 / 7], monthly: [12], quarterly: [4], yearly: [1],
};

// Horizon proposed when the file does not declare one (the horizons of the M competitions for these frequencies).
const DEFAULT_HORIZON = { yearly: 6, quarterly: 8, monthly: 18, weekly: 13, daily: 14, hourly: 48 };

export const seasonalityFor = f => SEASONALITY[f] ? SEASONALITY[f].slice() : [1];
export const primaryPeriod = periods => Math.max(1, Math.trunc(Math.min(...periods)));
// Lag count of the global models: 1.25 x the seasonal period, as in the study (monthly 15, weekly 65, daily 9, yearly 2).
export const defaultLag = periods => Math.ceil(1.25 * primaryPeriod(periods));
export const defaultHorizon = f => DEFAULT_HORIZON[f] || 12;

export class DataError extends Error {}

const num = s => {
  const t = s.trim();
  if (t === "" || t === "?" || /^(na|nan|null)$/i.test(t)) return NaN;
  const v = Number(t);
  if (Number.isNaN(v)) throw new DataError(`'${t.slice(0, 30)}' is not a number`);
  return v;
};

// Monash .tsf: "@" header lines, then one series per line as <attribute values separated by ':'>:<v1,v2,...>; '?' = missing.
export function parseTSF(text) {
  const head = { relation: null, frequency: null, horizon: null, attributes: [] };
  const names = [], series = [];
  let inData = false, lineno = 0;
  for (const raw of text.split(/\r?\n/)) {
    lineno++;
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    if (!inData) {
      if (!line.startsWith("@")) throw new DataError(`Line ${lineno}: expected a header line starting with '@' before @data.`);
      const parts = line.split(/\s+/), key = parts[0].toLowerCase();
      if (key === "@data") { inData = true; continue; }
      if (key === "@attribute") {
        if (parts.length !== 3) throw new DataError(`Line ${lineno}: @attribute must have the form 'name type'.`);
        head.attributes.push(parts[1]);
      } else if (key === "@relation") head.relation = parts.slice(1).join(" ");
      else if (key === "@frequency") head.frequency = parts[1] || null;
      else if (key === "@horizon") {
        head.horizon = parseInt(parts[1], 10);
        if (!Number.isFinite(head.horizon)) throw new DataError(`Line ${lineno}: @horizon must be an integer.`);
      }
      continue;
    }
    const cells = line.split(":");
    if (cells.length !== head.attributes.length + 1)
      throw new DataError(`Line ${lineno}: expected ${head.attributes.length} attribute values and one value list, found ${cells.length} parts.`);
    const toks = cells[cells.length - 1].split(",");
    const y = new Float64Array(toks.length);
    try { for (let i = 0; i < toks.length; i++) y[i] = num(toks[i]); }
    catch (e) { throw new DataError(`Line ${lineno}: ${e.message}.`); }
    const k = head.attributes.indexOf("series_name");
    names.push(k >= 0 ? cells[k] : `T${series.length + 1}`);
    series.push(y);
  }
  if (!inData) throw new DataError("The file has no @data section; it is not a .tsf file.");
  if (!series.length) throw new DataError("The file contains no series.");
  if (head.frequency && !SEASONALITY[head.frequency]) head.frequency = null;
  return { name: head.relation || "dataset", frequency: head.frequency, horizon: head.horizon, names, series };
}

// Delimited text in wide layout: one row per time step, one column per series, a header row with the series names.
// A first column that is not numeric (dates, labels) is skipped. With ';' or tab as the separator a decimal comma is read
// as a decimal point (spreadsheets with a Turkish or German locale export this way).
export function parseCSV(text, name = "dataset") {
  const lines = text.split(/\r?\n/).filter(l => l.trim() !== "");
  if (lines.length < 3) throw new DataError("The file needs a header row and at least two rows of values.");
  const count = ch => lines[0].split(ch).length;
  const sep = count("\t") > 1 ? "\t" : count(";") > count(",") ? ";" : ",";
  const fix = sep === "," ? s => s : s => s.replace(",", ".");
  const header = lines[0].split(sep).map(s => s.trim().replace(/^"|"$/g, ""));
  const rows = lines.slice(1).map(l => l.split(sep));
  const isNum = s => { const t = fix(s.trim()); return t === "" || t === "?" || !Number.isNaN(Number(t)); };
  const first = rows.every(r => isNum(r[0] ?? "")) ? 0 : 1;
  if (header.length - first < 1) throw new DataError("No value columns found.");
  const names = [], series = [];
  for (let c = first; c < header.length; c++) {
    const col = new Float64Array(rows.length);
    for (let r = 0; r < rows.length; r++) {
      try { col[r] = num(fix(rows[r][c] ?? "")); }
      catch (e) { throw new DataError(`Row ${r + 2}, column '${header[c]}': ${e.message}.`); }
    }
    let a = 0, b = col.length;                       // series of different length: empty cells at either end are dropped
    while (a < b && Number.isNaN(col[a])) a++;
    while (b > a && Number.isNaN(col[b - 1])) b--;
    if (b - a < 1) continue;
    names.push(header[c] || `T${c - first + 1}`);
    series.push(col.slice(a, b));
  }
  if (!series.length) throw new DataError("Every column is empty.");
  return { name, frequency: null, horizon: null, names, series };
}

export function parseFile(fileName, text) {
  const base = fileName.replace(/\.[^.]*$/, "");
  if (/\.tsf$/i.test(fileName) || /^\s*[@#]/.test(text)) { const d = parseTSF(text); if (d.name === "dataset") d.name = base; return d; }
  return parseCSV(text, base);
}

export const MISSING_POLICIES = ["interpolate", "locf", "zero"];

function fillMissing(y, policy) {
  const out = Float64Array.from(y), n = out.length;
  if (policy === "zero") { for (let i = 0; i < n; i++) if (Number.isNaN(out[i])) out[i] = 0; return out; }
  const idx = []; for (let i = 0; i < n; i++) if (!Number.isNaN(y[i])) idx.push(i);
  for (let i = 0; i < idx[0]; i++) out[i] = y[idx[0]];                                  // leading gap: first value
  for (let i = idx[idx.length - 1] + 1; i < n; i++) out[i] = y[idx[idx.length - 1]];    // trailing gap: last value
  for (let k = 0; k + 1 < idx.length; k++) {
    const a = idx[k], b = idx[k + 1];
    for (let i = a + 1; i < b; i++) out[i] = policy === "locf" ? y[a] : y[a] + (y[b] - y[a]) * (i - a) / (b - a);
  }
  return out;
}

// The last h observations of each series are the test part (fixed origin, as in the study). A series shorter than h + 2 is
// left out. Missing values in the training part are filled by `policy`; missing test values stay and are skipped by the measures.
export function prepare(data, h, policy = "interpolate") {
  if (!Number.isInteger(h) || h < 1) throw new DataError("The horizon must be a positive integer.");
  if (!MISSING_POLICIES.includes(policy)) throw new DataError(`Unknown missing-value policy: ${policy}`);
  const names = [], train = [], test = [], excluded = [];
  let imputed = 0, testMissing = 0;
  data.series.forEach((y, i) => {
    const nm = data.names[i];
    if (y.length < h + 2) { excluded.push({ name: nm, reason: `too short (length ${y.length} < horizon + 2 = ${h + 2})` }); return; }
    let tr = y.slice(0, y.length - h);
    const te = y.slice(y.length - h);
    let miss = 0; for (const v of tr) if (Number.isNaN(v)) miss++;
    if (miss === tr.length) { excluded.push({ name: nm, reason: "training part entirely missing" }); return; }
    if (miss) { tr = fillMissing(tr, policy); imputed += miss; }
    for (const v of te) if (Number.isNaN(v)) testMissing++;
    names.push(nm); train.push(tr); test.push(te);
  });
  if (!train.length) throw new DataError(`No series is long enough for a horizon of ${h}.`);
  return { names, train, test, h, excluded, imputed, testMissing };
}

// ---------------------------------------------------------------------------------------------------------------------
// Error measures, one value per series (metrics.py). Pairs with a missing forecast or actual are skipped; an undefined
// value is NaN and is left out of the summary (counted in `excluded`).
// ---------------------------------------------------------------------------------------------------------------------
export const METRICS = [
  { key: "MASE", label: "MASE", note: "Mean absolute error divided by the in-sample seasonal naive error." },
  { key: "SMAPE", label: "sMAPE", note: "Symmetric MAPE as a fraction (0 to 2)." },
  { key: "mSMAPE", label: "msMAPE", note: "Modified sMAPE (epsilon 0.1), stable near zero." },
  { key: "MAE", label: "MAE", note: "Mean absolute error, in the units of the data." },
  { key: "RMSE", label: "RMSE", note: "Root mean squared error, in the units of the data." },
  { key: "RMSSE", label: "RMSSE", note: "Root mean squared error scaled by the in-sample one-step naive error (M5)." },
  { key: "WAPE", label: "WAPE %", note: "Sum of absolute errors over the sum of actuals, in per cent." },
];

const mean = a => { let s = 0; for (const v of a) s += v; return a.length ? s / a.length : NaN; };

function maseScale(tr, lag) {
  if (tr.length <= lag) return NaN;
  let s = 0; for (let i = lag; i < tr.length; i++) s += Math.abs(tr[i] - tr[i - lag]);
  return s / (tr.length - lag);
}

export function seriesMetrics(f, a, train, periods) {
  const ae = [], se = [], sm = [], ms = [];
  let sumAbsA = 0, sumAe = 0;
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(f[i]) || Number.isNaN(a[i])) continue;
    const e = Math.abs(f[i] - a[i]), d = Math.abs(f[i]) + Math.abs(a[i]);
    ae.push(e); se.push(e * e); sumAe += e; sumAbsA += Math.abs(a[i]);
    const s = 2 * e / d;                                  // 0/0 is NaN and is skipped, as rowMeans(na.rm = TRUE) does
    if (!Number.isNaN(s)) sm.push(s);
    ms.push(2 * e / Math.max(0.6, d + 0.1));
  }
  const out = { MASE: NaN, SMAPE: NaN, mSMAPE: NaN, MAE: NaN, RMSE: NaN, RMSSE: NaN, WAPE: NaN };
  if (!ae.length) return out;
  out.MAE = mean(ae); out.RMSE = Math.sqrt(mean(se)); out.SMAPE = mean(sm); out.mSMAPE = mean(ms);
  out.WAPE = sumAbsA === 0 ? NaN : sumAe / sumAbsA * 100;
  const tr = train.filter(v => !Number.isNaN(v));
  let m = out.MAE / maseScale(tr, primaryPeriod(periods));  // seasonal scale; the lag-1 scale when that is undefined
  if (Number.isNaN(m)) m = out.MAE / maseScale(tr, 1);
  if (Number.isFinite(m)) out.MASE = m;
  if (tr.length >= 2) {
    let s = 0; for (let i = 1; i < tr.length; i++) s += (tr[i] - tr[i - 1]) ** 2;
    s /= tr.length - 1;
    if (s > 0) out.RMSSE = Math.sqrt(mean(se) / s);
  }
  return out;
}

export function median(values) {
  const v = values.filter(x => !Number.isNaN(x)).sort((x, y) => x - y);
  if (!v.length) return NaN;
  return v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
}

// Mean and median of every measure over the series, with the number of series on which it is undefined.
export function summarize(perSeries) {
  const out = {};
  for (const { key } of METRICS) {
    const v = perSeries.map(r => r[key]).filter(x => !Number.isNaN(x));
    out[key] = { mean: mean(v), median: median(v), excluded: perSeries.length - v.length };
  }
  return out;
}

// Mean rank of each method over the series for one measure (1 = lowest error; ties share the average rank). A series on
// which the measure is undefined for any method is left out, so every method is ranked on the same series.
export function meanRanks(perSeriesByMethod, key) {
  const methods = Object.keys(perSeriesByMethod);
  const n = methods.length ? perSeriesByMethod[methods[0]].length : 0;
  const sums = new Array(methods.length).fill(0);
  let used = 0;
  for (let s = 0; s < n; s++) {
    const v = methods.map(m => perSeriesByMethod[m][s][key]);
    if (v.some(Number.isNaN)) continue;
    used++;
    const order = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]);
    for (let i = 0; i < order.length;) {
      let j = i; while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
      for (let k = i; k <= j; k++) sums[order[k][1]] += (i + j) / 2 + 1;
      i = j + 1;
    }
  }
  return { ranks: Object.fromEntries(methods.map((m, i) => [m, used ? sums[i] / used : NaN])), series: used };
}

// ---------------------------------------------------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------------------------------------------------
const cell = v => typeof v === "number" ? (Number.isNaN(v) ? "" : String(v)) : /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
export const toCSV = rows => rows.map(r => r.map(cell).join(",")).join("\n") + "\n";
