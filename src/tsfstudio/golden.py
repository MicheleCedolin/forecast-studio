"""Altın karşılaştırma: bir motorun tahminleri, kullanıcının KENDİ R çıktılarıyla (NN5) ne kadar örtüşüyor?

İki düzeyde raporlanır:
  * tahmin düzeyi : bağıl fark (ortanca / %95 / maks.), >1e-3 fark eden seri sayısı;
  * karar düzeyi  : ortalama SMAPE ve MASE (motor vs R) ve yüzde değişim — yani fark doğruluk ölçülerini ne kadar oynatıyor.
Veri: `tests/golden/<ad>/` (tools/make_nn5_fixture.py üretir). Kendi çıktılarınızı eklemek için aynı klasör yapısını kullanın.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .engines.base import Engine, EngineJob
from .metrics import mase, smape
from .seasonality import seasonality_for

REL_FLOOR = 1e-9
THRESH = 1e-3


def golden_root() -> Path:
    env = os.environ.get("TSFSTUDIO_GOLDEN_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "tests" / "golden"


def available_cases(root: Path | None = None) -> list[str]:
    r = root or golden_root()
    return sorted(p.name for p in r.iterdir() if (p / "meta.json").is_file()) if r.is_dir() else []


def _load(d: Path, stem: str) -> np.ndarray:
    for ext in (".csv", ".csv.gz"):
        p = d / f"{stem}{ext}"
        if p.is_file():
            return np.loadtxt(p, delimiter=",", ndmin=2)
    raise FileNotFoundError(f"{d / stem}.csv[.gz] not found")


@dataclass
class GoldenCase:
    name: str
    meta: dict
    train: np.ndarray
    actuals: np.ndarray
    forecasts: dict[str, np.ndarray]


def load_case(name: str, root: Path | None = None) -> GoldenCase:
    d = (root or golden_root()) / name
    if not (d / "meta.json").is_file():
        raise FileNotFoundError(f"Golden dataset does not exist: {d}. Available: {available_cases(root)}")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    fc = {}
    for m in meta["methods"]:
        key = "pooled_regression" if m.startswith("pooled_regression") else m
        fc[key] = _load(d, f"forecasts_{m}")
    return GoldenCase(name, meta, _load(d, "train"), _load(d, "actuals"), fc)


def compare(engine: Engine, case: GoldenCase, methods: list[str] | None = None, workers: int = 1,
            progress=None, limit: int | None = None) -> pd.DataFrame:
    """Motorun desteklediği ve R referansı olan her yöntem için bir satır.

    `limit`: yalnızca ilk N seri (hızlı deneme/süre ölçümü; yavaş yöntemleri tam koşudan önce yoklamak için)."""
    if limit:
        k = int(limit)
        case = GoldenCase(case.name, case.meta, case.train[:k], case.actuals[:k], {m: f[:k] for m, f in case.forecasts.items()})
    ms = engine.methods()
    seas = seasonality_for(case.meta["frequency"])
    H = int(case.meta["horizon"])
    train = [r for r in case.train]
    rows = []
    for method, ref in case.forecasts.items():
        task = "global" if method == "pooled_regression" else "local"
        if methods and method not in methods:
            continue
        if method not in ms[task]:
            rows.append({"method": method, "status": "not in engine"})
            continue
        if progress:
            progress(method)
        job = EngineJob(task=task, method=method, train=train, horizon=H, seasonality=seas,
                        lag=int(case.meta["lag"]) if task == "global" else None, workers=workers)
        try:
            res = engine.run(job)
        except Exception as exc:  # noqa: BLE001
            rows.append({"method": method, "status": f"HATA: {str(exc).strip().splitlines()[0][:120]}"})
            continue
        f = res.forecasts
        rel = np.abs(f - ref) / np.maximum(np.abs(ref), REL_FLOOR)
        per = np.nanmax(rel, axis=1)
        sm_py, sm_r = float(np.nanmean(smape(f, case.actuals))), float(np.nanmean(smape(ref, case.actuals)))
        ma_py = float(np.nanmean(mase(f, case.actuals, train, seas)))
        ma_r = float(np.nanmean(mase(ref, case.actuals, train, seas)))
        rows.append({
            "method": method, "status": "tamam", "median_rel": float(np.nanmedian(rel)),
            "p95_rel": float(np.nanquantile(rel, 0.95)), "max_rel": float(np.nanmax(rel)),
            "n_gt_1e-3": int((per > THRESH).sum()), "n_series": int(f.shape[0]), "fallbacks": len(res.fallbacks),
            "fb_reason": res.fallbacks[0]["message"][:300] if res.fallbacks else "",
            "notes": " | ".join(res.notes),
            "SMAPE_engine": sm_py, "SMAPE_R": sm_r, "dSMAPE_%": 100 * (sm_py - sm_r) / sm_r,
            "MASE_engine": ma_py, "MASE_R": ma_r, "dMASE_%": 100 * (ma_py - ma_r) / ma_r,
            "sec": res.elapsed_sec,
        })
    return pd.DataFrame(rows)
