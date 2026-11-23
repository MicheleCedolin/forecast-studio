"""Çalıştırmadan ÖNCE kaynak (bellek) ve süre tahmini.

Amaç: Uzun/ağır bir koşuya başlamadan kullanıcıya dürüst bir öngörü göstermek (ör. M5 üzerinde TBATS günlerce sürer;
M5 L12 havuzlu regresyon ~13 GB'lık tasarım matrisi ister).

Süre tahmini iki kaynaktan gelir:
  1. `probe_runtime`: gerçek motoru seçili serilerin küçük bir örneğinde çalıştırıp doğrusal ölçekler (tercih edilen).
  2. `runtime_prior`: kullanıcının NN5 üzerindeki kendi ölçümlerinden (111 seri) türetilmiş seri başına saniye
     öncülleri. KABA bir kılavuzdur: makine, seri uzunluğu ve veriye göre sapma büyüktür.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# ------------------------------------------------------------------------------------------------
# Bellek
# ------------------------------------------------------------------------------------------------
_GB = 1024 ** 3


def _read_int(path: str) -> int | None:
    try:
        txt = Path(path).read_text().strip()
        return None if txt in ("", "max") else int(txt)
    except (OSError, ValueError):
        return None


def _windows_available_memory() -> int | None:
    """Windows: GlobalMemoryStatusEx ile kullanılabilir fiziksel bellek (yalnızca standart kütüphane). Diğer sistemlerde None."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
            return None
        return int(st.ullAvailPhys)
    except Exception:  # noqa: BLE001
        return None


def available_memory_bytes() -> int | None:
    """Kullanılabilir bellek: Windows'ta GlobalMemoryStatusEx; Linux'ta cgroup sınırı (Docker/WSL2) varsa `sınır - kullanılan`,
    yoksa /proc/meminfo MemAvailable.

    Docker Desktop'ta sınır WSL2 sanal makinesinin belleğidir (`.wslconfig`); bu yüzden cgroup değeri kullanılır.
    """
    if os.name == "nt":
        return _windows_available_memory()
    cands: list[int] = []
    lim = _read_int("/sys/fs/cgroup/memory.max")              # cgroup v2
    if lim is not None:
        cur = _read_int("/sys/fs/cgroup/memory.current") or 0
        cands.append(max(lim - cur, 0))
    else:
        lim1 = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")  # cgroup v1
        if lim1 is not None and lim1 < (1 << 60):
            cur1 = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes") or 0
            cands.append(max(lim1 - cur1, 0))
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                cands.append(int(ln.split()[1]) * 1024)
                break
    except OSError:
        pass
    return min(cands) if cands else None


@dataclass
class MemoryEstimate:
    rows: int
    cols: int
    matrix_bytes: int
    peak_bytes: int
    available_bytes: int | None
    verdict: str                         # "ok" | "warn" | "block" | "unknown"
    message: str


# Tepe bellek katsayıları (tasarım matrisi X'in kaç katı): glm.fit (QR + ağırlıklar + artıklar) ağırdır;
# catboost R matrisi + Pool kopyası. Katsayılar muhafazakâr kaba değerlerdir (ölçüm değil).
_PEAK_FACTOR = {"pooled_regression": 4.5, "catboost": 3.0, "lightgbm": 1.5, "lightgbm_feat": 2.0}


def global_rows(lengths: Sequence[int], lag: int, max_obs: int | None = None) -> int:
    """`embed(lag+1)` sonrası toplam satır: seri başına (L - lag), yalnızca pozitif olanlar; max_obs pencereler."""
    total = 0
    for L in lengths:
        L = min(int(L), int(max_obs)) if max_obs else int(L)
        total += max(L - lag, 0)
    return total


def estimate_global_memory(lengths: Sequence[int], lag: int, method: str, max_obs: int | None = None,
                           available: int | None = -1, peak_factor: float | None | str = "default") -> MemoryEstimate:
    """`peak_factor`: tasarım matrisinin kaç katı tepe bellek (varsayılan: R tablosu); None = akış hâlinde çözüm
    (matris bellekte tutulmaz: tepe ≈ blok + normalize edilmiş seri kopyası)."""
    rows = global_rows(lengths, lag, max_obs)
    cols = lag + 1
    mat = rows * cols * 8
    if peak_factor == "default":
        peak_factor = _PEAK_FACTOR.get(method, 4.0)
    if peak_factor is None:
        total_obs = sum(min(int(L), int(max_obs)) if max_obs else int(L) for L in lengths)
        peak = int(3 * 200_000 * cols * 8 + 2 * total_obs * 8)
    else:
        peak = int(mat * float(peak_factor))
    avail = available_memory_bytes() if available == -1 else available
    if avail is None:
        return MemoryEstimate(rows, cols, mat, peak, None, "unknown",
                              f"Estimated peak memory ≈ {peak / _GB:.1f} GB (available memory could not be read).")
    if peak > 0.9 * avail:
        verdict = "block"
        hint = (f"Options: use `max_obs` to keep only the last observations of each series (fewer rows), lower the lag, "
                f"limit the number of series, or allocate more memory (Docker: .wslconfig).")
    elif peak > 0.6 * avail:
        verdict, hint = "warn", "The system may start swapping; closing other applications is recommended."
    else:
        verdict, hint = "ok", ""
    kind = "streaming solution (the matrix is not held in memory)" if peak_factor is None else f"design matrix ≈ {mat / _GB:.2f} GB"
    msg = (f"{rows:,} rows × {cols} columns; {kind}; estimated peak ≈ {peak / _GB:.1f} GB; "
           f"available ≈ {avail / _GB:.1f} GB. {hint}").strip()
    return MemoryEstimate(rows, cols, mat, peak, avail, verdict, msg)


# ------------------------------------------------------------------------------------------------
# Süre
# ------------------------------------------------------------------------------------------------
# Kullanıcının NN5 ölçümleri (111 seri). Değer: (referans seri uzunluğu, seri başına saniye).
# Kaynak: NN5weekly/NN5daily "Execution Time" dosyaları. Not: tek koşu; R başlatma ve paket yükleme dahil olabilir.
_PRIORS: dict[str, tuple[tuple[int, float], tuple[int, float]]] = {
    #            haftalık (L=105)           günlük (L=735)
    "ses":       ((105, 8.405 / 111),       (735, 6.672 / 111)),
    "theta":     ((105, 8.548 / 111),       (735, 7.987 / 111)),
    "ets":       ((105, 10.509 / 111),      (735, 1.698 * 60 / 111)),
    "arima":     ((105, 41.015 / 111),      (735, 30.026 * 60 / 111)),
    "dhr_arima": ((105, 42.442 / 111),      (735, 30.026 * 60 / 111)),     # günlük ölçülmedi; arima ile aynı varsayıldı
    "tbats":     ((105, 11.759 * 60 / 111), (735, 13.085 * 60 / 111)),
    "snaive":    ((105, 0.001),             (735, 0.001)),
    "naive":     ((105, 0.001),             (735, 0.001)),
}


@dataclass
class RuntimeEstimate:
    seconds: float | None
    basis: str                      # "probe" | "prior" | "none"
    message: str
    low: float | None = None
    high: float | None = None


def format_duration(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 5400:
        return f"{sec / 60:.0f} min"
    if sec < 2 * 86400:
        return f"{sec / 3600:.1f} h"
    return f"{sec / 86400:.1f} days"


def runtime_prior(method: str, n_series: int, mean_length: float) -> RuntimeEstimate:
    """Seri başına süreyi iki NN5 noktası arasında (L=105, L=735) log-log doğrusal olarak ekstrapole eder."""
    if method not in _PRIORS:
        return RuntimeEstimate(None, "none", f"No run-time prior exists for '{method}'; a small trial run is recommended.")
    (l1, s1), (l2, s2) = _PRIORS[method]
    L = max(float(mean_length), 1.0)
    slope = math.log(s2 / s1) / math.log(l2 / l1)
    per = s1 * (L / l1) ** slope
    total = per * n_series
    return RuntimeEstimate(
        total, "prior",
        f"≈ {format_duration(total)} (rough projection from NN5 measurements; ≈ {per:.2f} s per series, "
        f"may deviate by a factor of 3 depending on machine and data).",
        low=total / 3, high=total * 3)


def probe_runtime(engine, job, n_probe: int = 8, seed: int = 1) -> RuntimeEstimate:
    """Gerçek motoru rastgele `n_probe` seride çalıştırıp doğrusal ölçekler (yalnızca yerel modeller için anlamlı)."""
    from .engines.base import EngineJob

    n = len(job.train)
    if job.task != "local" or n <= n_probe:
        return RuntimeEstimate(None, "none", "A trial run is only made for local models when the number of series exceeds the sample.")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=n_probe, replace=False))
    sub = EngineJob(task=job.task, method=job.method, train=[job.train[i] for i in idx], horizon=job.horizon,
                    seasonality=job.seasonality, integer_conversion=job.integer_conversion, params=job.params,
                    seed=job.seed, workers=job.workers)
    t0 = time.time()
    res = engine.run(sub)
    wall = time.time() - t0
    per = max(res.elapsed_sec, 1e-9) / n_probe
    # R başlatma sabit maliyet olabilir; duvar süresi ile çalışma süresi arasındaki fark sabit sayılır.
    fixed = max(wall - res.elapsed_sec, 0.0)
    total = fixed + per * n / max(job.workers, 1)
    return RuntimeEstimate(
        total, "probe",
        f"≈ {format_duration(total)} (from a trial run on {n_probe} series: ≈ {per:.2f} s per series; "
        f"deviates if series length/complexity differs).",
        low=total / 2, high=total * 2)


@dataclass
class PreflightReport:
    memory: MemoryEstimate | None = None
    runtime: RuntimeEstimate | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.memory and self.memory.verdict == "block")

    @property
    def long_running(self) -> bool:
        return bool(self.runtime and self.runtime.seconds and self.runtime.seconds > 15 * 60)


def preflight(job, n_total_series: int | None = None, engine=None, probe: bool = False) -> PreflightReport:
    """`EngineJob` için bellek + süre raporu. `engine` verilir ve `probe=True` ise deneme koşusu yapılır."""
    rep = PreflightReport()
    lens = [len(s) for s in job.train]
    n = n_total_series or len(lens)
    if job.task == "global":
        factors = getattr(engine, "MEMORY_FACTOR", None) or {}
        pf = factors[job.method] if job.method in factors else "default"
        rep.memory = estimate_global_memory(lens, int(job.lag), job.method, (job.params or {}).get("max_obs"), peak_factor=pf)
        if rep.memory.verdict in ("warn", "block"):
            rep.warnings.append(rep.memory.message)
        rep.runtime = RuntimeEstimate(None, "none", "Run-time projection for global models: depends on the design matrix size; "
                                      "try a small subset of series first.")
    else:
        if probe and engine is not None:
            try:
                rep.runtime = probe_runtime(engine, job)
            except Exception as exc:  # noqa: BLE001
                rep.warnings.append(f"The trial run could not be made: {exc}")
        if rep.runtime is None or rep.runtime.basis == "none":
            rep.runtime = runtime_prior(job.method, n, float(np.mean(lens)) if lens else 0.0)
        if rep.long_running:
            rep.warnings.append(f"Long-run warning: {rep.runtime.message}")
    return rep
