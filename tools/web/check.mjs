// Prints, as JSON, what the web edition computes for a golden case, for tests/test_web_edition.py to compare with the
// Python package and the study's R forecasts:   node tools/web/check.mjs <golden folder> <sample.tsf> <methods,comma-separated>
import fs from "fs";
import { parseFile, parseCSV, prepare, seriesMetrics, summarize, meanRanks, seasonalityFor, defaultLag } from "../../web/core.js";
import { forecastLocal, forecastGlobal } from "../../web/engine.js";
import createModule from "../../web/tsfcore.mjs";

const [golden, sample, methods] = process.argv.slice(2);
const data = parseFile("nn5_weekly.tsf", fs.readFileSync(sample, "utf8"));
const periods = seasonalityFor(data.frequency), prep = prepare(data, data.horizon), h = prep.h;
const rows = f => fs.readFileSync(`${golden}/${f}`, "utf8").trim().split("\n").map(l => l.split(",").filter(Boolean).map(Number));

// measures of the study's R ETS forecasts, computed by core.js
const ets = rows("forecasts_ets.csv");
const perSeries = ets.map((f, i) => seriesMetrics(f, prep.test[i], prep.train[i], periods));

// a table with a date column, a decimal comma, series of unequal length and a gap
const csv = parseCSV("date;a;b\n2020-01;1,5;\n2020-02;2,5;10\n2020-03;;11\n2020-04;4,5;12\n2020-05;5,5;\n", "t");
const filled = Object.fromEntries(["interpolate", "locf", "zero"].map(p => [p, Array.from(prepare({ names: ["a"], series: [Float64Array.of(NaN, 1, NaN, NaN, 4, 5, 6, 7)] }, 2, p).train[0])]));

const M = await createModule();
const forecasts = {};
for (const m of methods.split(",").filter(Boolean)) {
  const r = m === "pooled_regression" ? forecastGlobal(M, { lag: defaultLag(periods), h, series: prep.train })
    : forecastLocal(M, { method: m, h, periods, series: prep.train });
  forecasts[m] = { values: Array.from(r.forecasts), fallbacks: r.flags ? r.flags.reduce((a, b) => a + b, 0) : 0 };
}
const nan = (k, v) => typeof v === "number" && !Number.isFinite(v) ? null : v;
console.log(JSON.stringify({
  data: { name: data.name, frequency: data.frequency, horizon: data.horizon, names: data.names.slice(0, 3), n: data.series.length,
          lengths: data.series.map(s => s.length), train0: Array.from(prep.train[0]), test0: Array.from(prep.test[0]) },
  periods, lag: defaultLag(periods), perSeries, summary: summarize(perSeries),
  ranks: meanRanks({ a: perSeries, b: perSeries.map(r => ({ ...r, MASE: r.MASE + (r.MAE > 15 ? 1 : -1) })) }, "MASE"),
  csv: { names: csv.names, series: csv.series.map(s => Array.from(s)) }, filled, forecasts,
}, nan));
