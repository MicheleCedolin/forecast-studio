"""Deep-learning and foundation models for the Python engine (optional packages).

nhits    N-HiTS (Challu et al., 2023) via Nixtla neuralforecast: one global network trained on all series; input window =
         the job's lag, robust scaling per window, deterministic seed. Needs torch + neuralforecast.
chronos  Chronos-Bolt (Ansari et al., 2024) via chronos-forecasting: a pretrained model used zero-shot (no training);
         the forecast is the median of its quantile forecast. The weights are downloaded once from Hugging Face
         (amazon/chronos-bolt-small by default). Its pretraining corpus includes public benchmark data (M4 and Monash
         datasets among them), so scores on those datasets may be optimistic (data leakage) and should be reported as such.
Both run on the CPU.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

CHRONOS_DEFAULT = "amazon/chronos-bolt-small"
CHRONOS_MODELS = ("amazon/chronos-bolt-tiny", "amazon/chronos-bolt-mini", "amazon/chronos-bolt-small",
                  "amazon/chronos-bolt-base")


def _has(*mods: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in mods)


def nhits_available() -> bool:
    return _has("torch", "neuralforecast")


def chronos_available() -> bool:
    return _has("torch", "chronos")


def versions() -> dict:
    v = {}
    for mod in ("torch", "neuralforecast", "chronos", "transformers"):
        if _has(mod):
            try:
                from importlib.metadata import version
                v[mod] = version("chronos-forecasting" if mod == "chronos" else mod)
            except Exception:  # noqa: BLE001
                v[mod] = "installed"
    return v


def nhits_forecast(train: list[np.ndarray], H: int, input_size: int, params: dict | None, seed: int,
                   workers: int) -> np.ndarray:
    import logging

    import pandas as pd
    import torch
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    p = dict(params or {})
    for name in ("lightning", "pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)
    torch.set_num_threads(max(1, int(workers)) if workers and workers > 1 else max(1, (os.cpu_count() or 2) // 2))
    rows = []
    for i, y in enumerate(train):
        y = np.asarray(y, dtype=np.float64)
        rows.append(pd.DataFrame({"unique_id": i, "ds": np.arange(len(y)), "y": y}))
    df = pd.concat(rows, ignore_index=True)
    model = NHITS(h=H, input_size=int(input_size), max_steps=int(p.get("max_steps", 500)),
                  learning_rate=float(p.get("learning_rate", 1e-3)), scaler_type="robust", random_seed=int(seed),
                  start_padding_enabled=True, enable_progress_bar=False, enable_model_summary=False, logger=False,
                  accelerator="cpu", devices=1)
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df=df)
    fc = nf.predict()
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
    out = np.full((len(train), H), np.nan)
    for uid, g in fc.groupby("unique_id", sort=False):
        out[int(uid), :] = g.sort_values("ds")[col].to_numpy(dtype=np.float64)[:H]
    return out


_PIPE: dict = {}


def chronos_forecast(train: list[np.ndarray], H: int, params: dict | None, workers: int) -> np.ndarray:
    import torch
    from chronos import BaseChronosPipeline

    p = dict(params or {})
    model_id = p.get("model") or CHRONOS_DEFAULT
    if model_id not in CHRONOS_MODELS:
        raise ValueError(f"Unknown Chronos model: {model_id}")
    torch.set_num_threads(max(1, int(workers)) if workers and workers > 1 else max(1, (os.cpu_count() or 2) // 2))
    if model_id not in _PIPE:
        _PIPE[model_id] = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
    pipe = _PIPE[model_id]
    ctx_len = int(p.get("context_length", 2048))
    batch = int(p.get("batch_size", 64))
    out = np.full((len(train), H), np.nan)
    for s in range(0, len(train), batch):
        ctx = [torch.tensor(np.asarray(y, dtype=np.float32)[-ctx_len:]) for y in train[s:s + batch]]
        q, _mean = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5])
        out[s:s + len(ctx), :] = q[:, :, 0].numpy().astype(np.float64)
    return out
