"""GluonTS models for the Python engine (optional package gluonts[torch]).

deepar_gluonts  gluonts.torch.model.deepar.DeepAREstimator (Salinas et al., 2020)
feed_forward    gluonts.torch.model.simple_feedforward.SimpleFeedForwardEstimator
wavenet         gluonts.torch.model.wavenet.WaveNetEstimator (van den Oord et al., 2016)

The calls follow the Monash archive script (reference/experiments/deep_learning_experiments.py): the library defaults,
context_length = lag (WaveNet has none), a dummy start of 1900-01-01, the frequency of the dataset, and the median
of 100 sample paths as the point forecast. That script used the old MXNet GluonTS (0.8); this is the PyTorch
GluonTS, so its forecasts are not those published values. A job carries no frequency, so it is inferred from the
shortest seasonal period the way the Monash maps pair them (24 -> hourly, 7 -> daily, 12 -> monthly, ...).
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np

# shortest seasonal period -> GluonTS frequency (inverse of SEASONALITY_MAP / FREQUENCY_MAP in the Monash script)
_FREQ_BY_PERIOD = ((1440, "1min"), (144, "10min"), (48, "30min"), (24, "1h"), (7, "1D"), (365.25 / 7, "1W"), (12, "1M"),
                   (4, "1Q"), (1, "1Y"))
METHODS = ("deepar_gluonts", "feed_forward", "wavenet")


def available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "gluonts", "lightning"))


def version() -> str | None:
    if importlib.util.find_spec("gluonts") is None:
        return None
    from importlib.metadata import version as v
    return v("gluonts")


def freq_for(seasonality) -> tuple[str, bool]:
    """GluonTS frequency for the job's seasonal periods; second value False when it had to fall back to daily."""
    periods = [float(s) for s in np.asarray(seasonality, dtype=np.float64).reshape(-1) if np.isfinite(s) and s >= 1]
    m = min(periods) if periods else 1.0
    for p, f in _FREQ_BY_PERIOD:
        if abs(m - p) < 0.5:
            return f, True
    return "1D", False


def forecast(method: str, train: list[np.ndarray], H: int, lag: int, seasonality, params: dict | None, seed: int,
             workers: int) -> tuple[np.ndarray, list[str]]:
    import logging
    import warnings

    import lightning
    import pandas as pd
    import torch
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName

    if method not in METHODS:
        raise ValueError(f"Unknown GluonTS method: {method}")
    p = dict(params or {})
    for name in ("lightning", "pytorch_lightning", "lightning.pytorch", "lightning_fabric", "gluonts"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="gluonts")
    warnings.filterwarnings("ignore", module="lightning")
    torch.set_num_threads(max(1, int(workers)) if workers and workers > 1 else max(1, (os.cpu_count() or 2) // 2))
    lightning.seed_everything(int(seed), verbose=False)

    freq, known = freq_for(seasonality)
    start = pd.Period("1900-01-01 00:00", freq=freq)
    ds = ListDataset([{FieldName.TARGET: np.asarray(y, dtype=np.float32), FieldName.START: start} for y in train], freq=freq)
    # GluonTS keeps the best epoch through a ModelCheckpoint callback; its files go to a temporary folder, not the cwd
    tmp = tempfile.TemporaryDirectory(prefix="tsfstudio_gluonts_")
    trainer = dict(max_epochs=int(p.get("epochs", 100)), accelerator="cpu", devices=1, enable_progress_bar=False,
                   enable_model_summary=False, logger=False, default_root_dir=tmp.name)
    common = dict(prediction_length=int(H), lr=float(p.get("learning_rate", 1e-3)), trainer_kwargs=trainer)
    if method == "deepar_gluonts":
        from gluonts.torch.model.deepar import DeepAREstimator
        est = DeepAREstimator(freq=freq, context_length=int(lag), **common)
    elif method == "feed_forward":
        from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
        est = SimpleFeedForwardEstimator(context_length=int(lag), **common)
    else:
        from gluonts.torch.model.wavenet import WaveNetEstimator
        est = WaveNetEstimator(freq=freq, **common)
    try:
        predictor = est.train(training_data=ds)
        out = np.full((len(train), H), np.nan)
        for i, f in enumerate(predictor.predict(ds, num_samples=100)):
            out[i, :] = np.asarray(f.median, dtype=np.float64)[:H]
    finally:
        tmp.cleanup()
    notes = [f"GluonTS {version()} (PyTorch), frequency {freq}"
             + ("" if known else " (no Monash frequency matches the seasonal period; daily was assumed)")
             + "; point forecast = median of 100 sample paths, as in the Monash archive study (which used MXNet GluonTS 0.8)."]
    return out, notes
