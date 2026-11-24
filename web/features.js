// Series characteristics and their principal components, as the desktop application computes them
// (src/tsfstudio/features.py): the basic descriptive features, catch22 (in tsfcore) and a PCA on the standardised
// features. Used by worker.js; tools/validate_web_features.py runs it under Node and compares it with Python.
(function (root) {
  "use strict";
  const BASIC = ["length", "mean", "sd", "cv", "min", "max", "zero_share", "adi", "cv2", "boxcox_lambda"];
  const CLASSES = ["smooth", "intermittent", "erratic", "lumpy"];   // Syntetos, Boylan & Croston (2005)
  const ADI_CUT = 1.32, CV2_CUT = 0.49;
  // seasonality.SEASONALITY_MAP
  const SEASONS = { "4_seconds": [21600, 151200, 7889400], minutely: [1440, 10080, 525960], "10_minutes": [144, 1008, 52596],
                    "15_minutes": [96, 672, 35064], half_hourly: [48, 336, 17532], hourly: [24, 168, 8766], daily: [7],
                    weekly: [365.25 / 7], monthly: [12], quarterly: [4], yearly: [1] };

  // seasonality.primary_period(seasonality_for(frequency))
  function primaryPeriod(frequency) {
    const s = SEASONS[frequency] || [1];
    return Math.max(1, Math.trunc(Math.min.apply(null, s)));
  }

  // features.basic_features without boxcox_lambda: x holds the finite values of one series
  function basic(x) {
    const n = x.length, out = { length: n, mean: NaN, sd: NaN, cv: NaN, min: NaN, max: NaN, zero_share: NaN, adi: NaN, cv2: NaN, cls: null };
    if (!n) return out;
    let sum = 0, lo = Infinity, hi = -Infinity, zeros = 0, nzSum = 0;
    for (let i = 0; i < n; i++) {
      const v = x[i];
      sum += v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
      if (v === 0) zeros++; else nzSum += v;
    }
    const mu = sum / n;
    let ss = 0;
    for (let i = 0; i < n; i++) ss += (x[i] - mu) * (x[i] - mu);
    out.mean = mu;
    out.sd = n > 1 ? Math.sqrt(ss / (n - 1)) : NaN;
    out.cv = mu !== 0 ? out.sd / Math.abs(mu) : NaN;
    out.min = lo;
    out.max = hi;
    out.zero_share = zeros / n;
    const nz = n - zeros;
    if (nz) {
      out.adi = n / nz;
      const nzMu = nzSum / nz;
      let s2 = 0;
      for (let i = 0; i < n; i++) if (x[i] !== 0) s2 += (x[i] - nzMu) * (x[i] - nzMu);
      out.cv2 = nzMu !== 0 ? Math.pow(Math.sqrt(s2 / nz) / nzMu, 2) : NaN;
    }
    if (Number.isFinite(out.adi) && Number.isFinite(out.cv2)) out.cls = CLASSES[(out.adi >= ADI_CUT ? 1 : 0) + 2 * (out.cv2 >= CV2_CUT ? 1 : 0)];
    return out;
  }

  // All features of the file behind `handle` (a tsf_io handle of the tsfcore module `core`).
  // Returns { names, columns, table (Float64Array, n x columns.length, row-major), classes }.
  function compute(core, handle, header) {
    const n = Number(core._tsf_io_n_series(handle));
    const c22 = core._tsf_catch22_count();
    const columns = BASIC.slice();
    for (let j = 0; j < c22; j++) columns.push(core.UTF8ToString(core._tsf_catch22_name(j)));
    const p = columns.length, m = primaryPeriod(header.frequency);
    const nameAttr = header.attributes.findIndex((a) => a[0] === "series_name");
    const lenPtr = core._malloc(8);
    const lengths = new Array(n);
    let total = 0, longest = 0;
    for (let i = 0; i < n; i++) {
      core._tsf_io_series(handle, BigInt(i), lenPtr);
      lengths[i] = Number(core.HEAP64[lenPtr >> 3]);
      total += lengths[i];
      longest = Math.max(longest, lengths[i]);
    }
    const values = core._malloc(8 * Math.max(1, total)), offsets = core._malloc(8 * (n + 1));
    const out = core._malloc(8 * Math.max(1, n * c22)), tmp = core._malloc(8 * Math.max(1, longest));
    const table = new Float64Array(n * p), names = new Array(n), classes = new Array(n);
    try {
      let pos = 0;
      core.HEAP64[offsets >> 3] = 0n;
      for (let i = 0; i < n; i++) {
        const ptr = core._tsf_io_series(handle, BigInt(i), lenPtr);
        core.HEAPF64.copyWithin((values >> 3) + pos, ptr >> 3, (ptr >> 3) + lengths[i]);
        pos += lengths[i];
        core.HEAP64[(offsets >> 3) + i + 1] = BigInt(pos);
      }
      core._tsf_catch22(values, offsets, n, 0, 1, out);
      pos = 0;
      for (let i = 0; i < n; i++) {
        // the finite values, for the basic features and the Box-Cox lambda
        let k = 0;
        const src = (values >> 3) + pos, dst = tmp >> 3;
        for (let t = 0; t < lengths[i]; t++) {
          const v = core.HEAPF64[src + t];
          if (Number.isFinite(v)) core.HEAPF64[dst + k++] = v;
        }
        pos += lengths[i];
        const b = basic(core.HEAPF64.subarray(dst, dst + k));
        const lam = k ? core._tsf_boxcox_lambda(tmp, k, -1, 2, m) : NaN;
        const row = i * p;
        BASIC.forEach((key, j) => { table[row + j] = key === "boxcox_lambda" ? (Number.isFinite(lam) ? lam : NaN) : b[key]; });
        for (let j = 0; j < c22; j++) table[row + BASIC.length + j] = core.HEAPF64[(out >> 3) + i * c22 + j];
        classes[i] = b.cls;
        names[i] = nameAttr >= 0 ? core.UTF8ToString(core._tsf_io_attr(handle, BigInt(i), nameAttr)) : "T" + (i + 1);
      }
    } finally {
      for (const q of [lenPtr, values, offsets, out, tmp]) core._free(q);
    }
    return { names, columns, table, classes };
  }

  // Eigenvalues and eigenvectors of a symmetric matrix (cyclic Jacobi). a: p x p row-major, overwritten.
  function eigenSym(a, p) {
    const v = new Float64Array(p * p);
    for (let i = 0; i < p; i++) v[i * p + i] = 1;
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) off += a[i * p + j] * a[i * p + j];
      if (off < 1e-22) break;
      for (let i = 0; i < p; i++) {
        for (let j = i + 1; j < p; j++) {
          const aij = a[i * p + j];
          if (Math.abs(aij) < 1e-300) continue;
          const theta = (a[j * p + j] - a[i * p + i]) / (2 * aij);
          const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < p; k++) {
            const aki = a[k * p + i], akj = a[k * p + j];
            a[k * p + i] = c * aki - s * akj;
            a[k * p + j] = s * aki + c * akj;
          }
          for (let k = 0; k < p; k++) {
            const aik = a[i * p + k], ajk = a[j * p + k];
            a[i * p + k] = c * aik - s * ajk;
            a[j * p + k] = s * aik + c * ajk;
          }
          for (let k = 0; k < p; k++) {
            const vki = v[k * p + i], vkj = v[k * p + j];
            v[k * p + i] = c * vki - s * vkj;
            v[k * p + j] = s * vki + c * vkj;
          }
        }
      }
    }
    const values = new Float64Array(p);
    for (let i = 0; i < p; i++) values[i] = a[i * p + i];
    return { values, vectors: v };
  }

  // features.pca: columns with more than maxMissing missing values or no variance are dropped, the other missing
  // values are replaced by the column median, every column is standardised (a PCA on the correlation matrix); the
  // largest-magnitude loading of each component is made positive.
  function pca(table, n, columns, nComponents, maxMissing) {
    const p0 = columns.length;
    nComponents = nComponents || 3;
    maxMissing = maxMissing === undefined ? 0.5 : maxMissing;
    const keep = [], dropped = [], med = new Float64Array(p0), mean = new Float64Array(p0), sd = new Float64Array(p0);
    for (let j = 0; j < p0; j++) {
      const col = [];
      for (let i = 0; i < n; i++) { const v = table[i * p0 + j]; if (Number.isFinite(v)) col.push(v); }
      col.sort((a, b) => a - b);
      const k = col.length;
      med[j] = k ? (k % 2 ? col[(k - 1) / 2] : (col[k / 2 - 1] + col[k / 2]) / 2) : NaN;
      let s = 0;
      for (let i = 0; i < n; i++) { const v = table[i * p0 + j]; s += Number.isFinite(v) ? v : med[j]; }
      mean[j] = s / n;
      let ss = 0;
      for (let i = 0; i < n; i++) { const v = table[i * p0 + j]; const z = (Number.isFinite(v) ? v : med[j]) - mean[j]; ss += z * z; }
      sd[j] = n > 1 ? Math.sqrt(ss / (n - 1)) : 0;
      const ok = (n ? (n - k) / n : 1) <= maxMissing && Number.isFinite(sd[j]) && sd[j] > 1e-12 * Math.max(1, Math.abs(med[j]));
      (ok ? keep : dropped).push(j);
    }
    const p = keep.length;
    if (p < 2 || n < 3) throw new Error("too few usable features or series for a PCA");
    const Z = new Float64Array(n * p);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < p; c++) {
        const j = keep[c], v = table[i * p0 + j];
        Z[i * p + c] = ((Number.isFinite(v) ? v : med[j]) - mean[j]) / sd[j];
      }
    }
    const C = new Float64Array(p * p);   // Z'Z: its eigenvalues are the squared singular values of Z
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < p; a++) {
        const za = Z[i * p + a];
        if (za === 0) continue;
        for (let b = a; b < p; b++) C[a * p + b] += za * Z[i * p + b];
      }
    }
    for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) C[a * p + b] = C[b * p + a];
    const eig = eigenSym(C, p);
    const order = Array.from({ length: p }, (_, i) => i).sort((x, y) => eig.values[y] - eig.values[x]);
    const k = Math.min(nComponents, p, n);
    let totalVar = 0;
    for (let i = 0; i < p; i++) totalVar += Math.max(0, eig.values[i]);
    const loadings = new Float64Array(p * k), explained = new Array(k);
    for (let c = 0; c < k; c++) {
      const e = order[c];
      let big = 0, sign = 1;
      for (let j = 0; j < p; j++) { const v = eig.vectors[j * p + e]; if (Math.abs(v) > big) { big = Math.abs(v); sign = v < 0 ? -1 : 1; } }
      for (let j = 0; j < p; j++) loadings[j * k + c] = sign * eig.vectors[j * p + e];
      explained[c] = Math.max(0, eig.values[e]) / totalVar;
    }
    const scores = new Float64Array(n * k);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < k; c++) {
        let s = 0;
        for (let j = 0; j < p; j++) s += Z[i * p + j] * loadings[j * k + c];
        scores[i * k + c] = s;
      }
    }
    return { k, scores, explained, loadings, features: keep.map((j) => columns[j]), dropped: dropped.map((j) => columns[j]) };
  }

  const api = { BASIC, CLASSES, primaryPeriod, basic, compute, pca };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TsfFeatures = api;
})(typeof self !== "undefined" ? self : globalThis);
