"""Motor kaydı: ad -> Engine. Yeni motorlar (Python/statsforecast, PyTorch GluonTS) buraya eklenir."""
from __future__ import annotations

from .base import Engine
from .baseline import PythonBaselineEngine
from .py_engine import PythonEngine
from .r_engine import REngine

ENGINES: dict[str, type[Engine]] = {
    "python": PythonEngine,
    "r": REngine,
    "python-baseline": PythonBaselineEngine,
}


def get_engine(name: str) -> Engine:
    try:
        return ENGINES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown engine: {name!r}. Registered engines: {', '.join(sorted(ENGINES))}") from exc


def engine_status() -> dict[str, tuple[bool, str]]:
    """Arayüz için: motor -> (kullanılabilir mi, sürüm/açıklama)."""
    return {n: cls().available() for n, cls in ENGINES.items()}
