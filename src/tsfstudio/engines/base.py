"""Motor arayüzü: R motoru, Python motoru ve referans (baseline) motoru aynı sözleşmeyi paylaşır."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

Task = Literal["local", "global"]
ProgressFn = Callable[[str, int, int], None]  # (aşama, tamamlanan, toplam)


class EngineError(RuntimeError):
    """Motor çalışırken hata (ayrıntı mesajda)."""


class EngineCancelled(EngineError):
    """Kullanıcı işi iptal etti."""


@dataclass
class EngineJob:
    task: Task
    method: str
    train: list[np.ndarray]                 # eğitim serileri (NaN içermez)
    horizon: int                            # en büyük ufuk
    seasonality: tuple[float, ...]
    horizons: list[int] | None = None       # seri başına ufuk (yoksa hepsi `horizon`)
    integer_conversion: bool = False
    lag: int | None = None                  # yalnızca global modeller
    params: dict = field(default_factory=dict)
    seed: int = 1
    workers: int = 1

    def validate(self) -> None:
        if not self.train:
            raise EngineError("There are no training series.")
        if self.horizon < 1:
            raise EngineError("The horizon must be at least 1.")
        if self.horizons is not None and len(self.horizons) != len(self.train):
            raise EngineError("The length of horizons does not match the number of series.")
        if self.task == "global" and (self.lag is None or self.lag < 1):
            raise EngineError("Global models require lag >= 1.")
        for i, s in enumerate(self.train):
            if len(s) == 0:
                raise EngineError(f"Series {i} is empty.")


@dataclass
class EngineResult:
    forecasts: np.ndarray                   # (n_seri, horizon); kısa ufuklu serilerde sağı NaN
    elapsed_sec: float
    fallbacks: list[dict] = field(default_factory=list)   # {"index": int, "message": str}
    na_replaced: int = 0                    # NA -> 0 yapılan tahmin sayısı (orijinal davranış)
    versions: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class Engine(ABC):
    name: str = "engine"
    LOCAL_METHODS: tuple[str, ...] = ()
    GLOBAL_METHODS: tuple[str, ...] = ()

    def methods(self) -> dict[str, tuple[str, ...]]:
        return {"local": self.LOCAL_METHODS, "global": self.GLOBAL_METHODS}

    def supports(self, task: str, method: str) -> bool:
        return method in self.methods().get(task, ())

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(kullanılabilir mi, açıklama/sürüm)."""

    @abstractmethod
    def run(self, job: EngineJob, progress: ProgressFn | None = None,
            cancel: threading.Event | None = None) -> EngineResult:
        ...
