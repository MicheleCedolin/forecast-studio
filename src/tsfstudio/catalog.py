"""Paketlenmiş TSF veri kümeleri kataloğu.

`data/catalog.yaml` kullanıcının R/Python deney betiklerinden çıkarılan varsayılanları (lag, ufuk,
tamsayı dönüşümü, çalışmada kullanılan yöntemler) ve Drive'daki gerçek dosya adı/boyutlarını içerir.
Frekans, seri sayısı gibi veri-bağımlı bilgiler UYDURULMAZ; dosyanın kendisinden okunur
(`read_tsf_header` hızlı, `dataset_stats` tam geçiş).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .seasonality import seasonality_for
from .tsf import TSFDataset, TSFHeader, read_tsf, read_tsf_header

LOCAL_METHODS = ("ses", "ets", "theta", "tbats", "arima", "dhr_arima", "snaive")
GLOBAL_METHODS = ("pooled_regression", "catboost")
DEEP_METHODS = ("feed_forward", "deepar", "nbeats", "wavenet", "transformer")
INDEX_FILENAME = "_index.json"


@dataclass(frozen=True)
class DatasetEntry:
    id: str
    file: str
    bytes: int
    bundle: str                 # main | extra | m5 | user
    domain: str
    lag: int | None
    lag_origin: str             # study | suggested
    horizon_override: int | None
    integer_conversion: bool
    use_timestamps: bool
    local_methods: tuple[str, ...] = ()
    global_methods: tuple[str, ...] = ()
    deep_methods: tuple[str, ...] = ()
    wrmsse: bool = False

    def effective_horizon(self, header: TSFHeader) -> int | None:
        """R: external_forecast_horizon (varsa) aksi halde @horizon."""
        return self.horizon_override if self.horizon_override is not None else header.horizon

    def recommended_local_methods(self, header: TSFHeader | None = None) -> tuple[str, ...]:
        """Çalışmada kullanılan yerel yöntemler; yoksa kural: çoklu/uzun mevsimsellikte DHR-ARIMA, değilse ARIMA."""
        if self.local_methods:
            return self.local_methods
        base = ["ses", "ets", "theta", "tbats"]
        freq = header.frequency if header else None
        long_season = freq is not None and (len(seasonality_for(freq)) > 1 or freq in ("weekly", "daily"))
        return tuple(base + (["dhr_arima"] if long_season else ["arima"]))


@dataclass
class Catalog:
    entries: list[DatasetEntry]
    not_bundled: list[dict] = field(default_factory=list)

    def get(self, ds_id: str) -> DatasetEntry:
        for e in self.entries:
            if e.id == ds_id:
                return e
        raise KeyError(ds_id)

    def ids(self) -> list[str]:
        return [e.id for e in self.entries]


def load_catalog() -> Catalog:
    text = resources.files("tsfstudio").joinpath("data/catalog.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    entries = []
    for d in raw["datasets"]:
        d = dict(d)
        for k in ("local_methods", "global_methods", "deep_methods"):
            d[k] = tuple(d.get(k) or ())
        d.setdefault("wrmsse", False)
        entries.append(DatasetEntry(**d))
    return Catalog(entries=entries, not_bundled=list(raw.get("referenced_but_not_bundled") or []))


def default_data_dir() -> Path:
    return Path(os.environ.get("TSFSTUDIO_DATA_DIR", "data/tsf"))


def _dir(data_dir: "str | os.PathLike | None") -> Path:
    """str/Path/None kabul eder."""
    return Path(data_dir) if data_dir is not None else default_data_dir()


# ---------------------------------------------------------------------------------------------
# Yerel dosya durumu
# ---------------------------------------------------------------------------------------------
def availability(entry: DatasetEntry, data_dir: "str | os.PathLike | None" = None) -> str:
    """'ok' | 'missing' | 'size_mismatch' (beklenen bayt sayısı ile karşılaştırma)."""
    p = _dir(data_dir) / entry.file
    if not p.is_file():
        return "missing"
    return "ok" if p.stat().st_size == entry.bytes else "size_mismatch"


def check_bundle(catalog: Catalog | None = None, data_dir: "str | os.PathLike | None" = None) -> dict[str, list[str]]:
    catalog = catalog or load_catalog()
    out: dict[str, list[str]] = {"ok": [], "missing": [], "size_mismatch": []}
    for e in catalog.entries:
        out[availability(e, data_dir)].append(e.id)
    return out


def load_entry(entry: DatasetEntry, data_dir: "str | os.PathLike | None" = None, **kw) -> TSFDataset:
    path = _dir(data_dir) / entry.file
    if not path.is_file():
        raise FileNotFoundError(f"{entry.file} not found: {path}. Use `make data-check` to see the package status.")
    return read_tsf(path, **kw)


def read_entry_header(entry: DatasetEntry, data_dir: "str | os.PathLike | None" = None) -> TSFHeader:
    return read_tsf_header(_dir(data_dir) / entry.file)


# ---------------------------------------------------------------------------------------------
# Veri özeti (tam geçiş; sonuç _index.json içinde önbelleğe alınır)
# ---------------------------------------------------------------------------------------------
def dataset_stats(ds: TSFDataset) -> dict:
    L = ds.lengths
    n_vals = int(L.sum())
    n_nan = int(sum(int(np.isnan(v).sum()) for v in ds.values))
    n_zero = int(sum(int((v == 0).sum()) for v in ds.values))
    return {
        "relation": ds.header.relation,
        "frequency": ds.header.frequency,
        "horizon": ds.header.horizon,
        "n_series": ds.n_series,
        "len_min": int(L.min()),
        "len_median": float(np.median(L)),
        "len_max": int(L.max()),
        "equal_length": ds.is_equal_length(),
        "missing_share": n_nan / n_vals,
        "zero_share": n_zero / max(1, n_vals - n_nan),
        "attributes": ds.header.attribute_names,
    }


def build_index(
    data_dir: "str | os.PathLike | None" = None,
    ids: list[str] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Mevcut dosyaların özetini çıkarıp `<data_dir>/_index.json` dosyasına yazar."""
    data_dir = _dir(data_dir)
    cat = load_catalog()
    idx_path = data_dir / INDEX_FILENAME
    index = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.is_file() else {}
    todo = [e for e in cat.entries if (ids is None or e.id in ids) and availability(e, data_dir) != "missing"]
    for i, e in enumerate(todo, start=1):
        if progress:
            progress(e.id, i, len(todo))
        ds = load_entry(e, data_dir)
        index[e.id] = dataset_stats(ds) | {"bytes": e.bytes}
    idx_path.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return index


def load_index(data_dir: "str | os.PathLike | None" = None) -> dict:
    p = _dir(data_dir) / INDEX_FILENAME
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
