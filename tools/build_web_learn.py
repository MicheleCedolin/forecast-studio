"""Writes web/learn.json: the texts of the browser page's method list and Learn tab.

Usage:  python tools/build_web_learn.py

Everything comes from the interface's own registry (src/tsfstudio/app/models_info.py), so the browser page and the desktop
application describe a method, a measure and a reference in the same words. Only the local methods of the C++ engine are
listed, because those are what the browser build can run. Run this again after models_info.py changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfstudio.app import models_info as MI  # noqa: E402
from tsfstudio.engines.cpp_engine import CppEngine  # noqa: E402
from tsfstudio.metrics import ALL_METRICS, DEFAULT_METRICS  # noqa: E402


def main() -> int:
    used_refs: list[str] = []

    def refs(keys) -> list[str]:
        for k in keys:
            if k not in used_refs:
                used_refs.append(k)
        return list(keys)

    groups = [{"id": g, "name": name, "text": text} for g, name, text in MI.GROUPS]
    models = []
    for mid, m in MI.MODELS.items():
        if m.get("task") != "local" or mid not in CppEngine.LOCAL_METHODS:
            continue
        models.append({"id": mid, "name": m["name"], "group": m["group"], "summary": m["summary"], "how": m["how"],
                       "use": m["use"], "caution": m["caution"], "refs": refs(m.get("refs", []))})
    metrics = []
    for key in ALL_METRICS:
        info = MI.METRICS_INFO[key]
        metrics.append({"id": key, "name": info["name"], "unit": info["unit"], "text": info["text"], "caution": info["caution"],
                        "default": key in DEFAULT_METRICS})
    metric_refs = refs(MI.METRIC_REFS)
    refs(["godahewa2021"])
    references = {k: MI.REFS[k] for k in used_refs}
    out = {"groups": [g for g in groups if any(m["group"] == g["id"] for m in models)], "models": models, "metrics": metrics,
           "metric_refs": metric_refs, "references": references}
    path = ROOT / "web" / "learn.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path}: {len(models)} methods, {len(metrics)} measures, {len(references)} references "
          f"({sum(1 for r in references.values() if r['verified'])} checked against a bibliographic record)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
