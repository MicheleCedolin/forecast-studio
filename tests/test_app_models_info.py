"""Consistency between the interface model schema (app/models_info.py) and what the engines actually provide."""
from __future__ import annotations

import numpy as np
import pytest

from tsfstudio.app import models_info as MI
from tsfstudio.engines.py_engine import PythonEngine, forecast_one
from tsfstudio.engines.r_engine import REngine
from tsfstudio.metrics import DEFAULT_METRICS


def test_every_engine_method_has_a_ui_entry():
    for cls, eng in ((PythonEngine, "python"), (REngine, "r")):
        for m in cls.LOCAL_METHODS + cls.GLOBAL_METHODS:
            assert m in MI.MODELS, f"'{m}' of the {eng} engine is missing from the interface registry"
            assert eng in MI.MODELS[m]["engines"], f"the {eng} note is missing for '{m}'"


def test_every_ui_model_is_known_to_its_engines():
    py = set(PythonEngine.LOCAL_METHODS + PythonEngine.GLOBAL_METHODS)
    r = set(REngine.LOCAL_METHODS + REngine.GLOBAL_METHODS)
    for mid, info in MI.MODELS.items():
        if "python" in info["engines"]:
            assert mid in py, f"{mid}: the interface says it exists in Python, the engine does not know it"
        if "r" in info["engines"]:
            assert mid in r, f"{mid}: the interface says it exists in R, the engine does not know it"


def test_task_matches_engine_family():
    for mid, info in MI.MODELS.items():
        fam = "global" if mid in PythonEngine.GLOBAL_METHODS or mid in REngine.GLOBAL_METHODS else "local"
        assert info["task"] == fam, mid


def test_groups_text_and_refs_are_complete():
    gids = {g[0] for g in MI.GROUPS}
    for mid, info in MI.MODELS.items():
        assert info["group"] in gids, mid
        for k in ("name", "summary", "how", "use", "caution", "cost"):
            assert isinstance(info[k], str) and info[k].strip(), f"{mid}.{k} is empty"
        for r in info["refs"]:
            assert r in MI.REFS, f"{mid}: bilinmeyen kaynak {r}"
        for eng, note in info["engines"].items():
            assert isinstance(note, str) and note.strip(), f"{mid}.engines.{eng} is empty"
    for k, r in MI.REFS.items():
        assert r["text"].strip() and isinstance(r["verified"], bool), k
    for r in MI.METRIC_REFS:
        assert r in MI.REFS


def test_metrics_info_covers_defaults():
    assert set(DEFAULT_METRICS) <= set(MI.METRICS_INFO)
    for k, v in MI.METRICS_INFO.items():
        assert {"name", "unit", "text", "caution"} <= set(v), k


def test_param_schema_is_valid():
    for mid, info in MI.MODELS.items():
        for eng in ("python", "r", "cpp"):
            keys = MI.model_param_keys(mid, eng)
            assert len(keys) == len(set(keys)), f"{mid}/{eng}: yinelenen parametre"
        for p in info["params"]:
            assert p["kind"] in ("int", "float", "bool", "choice"), (mid, p["key"])
            assert set(p["engines"]) <= {"python", "r", "cpp"} and p["engines"]
            if "cpp" in p["engines"]:                       # C++ shows a setting only where it accepts every value
                assert MI.CPP_PARAMS.get(mid, set()) is None or p["key"] in MI.CPP_PARAMS[mid], (mid, p["key"])
            assert p["help"].strip() and p["label"].strip()
            if p["kind"] == "choice":
                vals = [c["value"] for c in p["choices"]]
                assert vals and (p["default"] is None or str(p["default"]) in vals), (mid, p["key"])
            if p["kind"] in ("int", "float") and p["default"] is not None:
                if p["min"] is not None:
                    assert p["default"] >= p["min"], (mid, p["key"])
                if p["max"] is not None:
                    assert p["default"] <= p["max"], (mid, p["key"])
            if p["default"] is None:
                assert p["kind"] != "bool" and p["placeholder"], f"{mid}.{p['key']}: an empty default needs a placeholder"


def test_ui_default_params_reach_native_python_models():
    """The interface default parameters must really be passable to the built-in Python methods (no key-name mismatch)."""
    rng = np.random.default_rng(0)
    y = 50 + np.cumsum(rng.normal(size=60))
    for mid in ("naive", "snaive", "ses", "theta"):
        info = MI.MODELS[mid]
        params = {p["key"]: p["default"] for p in info["params"] if "python" in p["engines"] and p["default"] is not None}
        f = np.asarray(forecast_one(mid, y, 6, (12.0,), params))
        assert f.shape == (6,) and np.isfinite(f).all(), mid


def test_build_catalog_marks_availability():
    cat = MI.build_catalog({"python": {"local": ("naive",), "global": ()}, "r": None})
    by = {m["id"]: m for m in cat}
    assert by["naive"]["engines"]["python"]["available"] is True
    assert by["naive"]["engines"]["r"]["available"] is False
    assert by["ses"]["engines"]["python"]["available"] is False
    assert by["tbats"]["engines"]["python"]["supported"] is False        # TBATS is R-only
    assert by["tbats"]["engines"]["r"]["supported"] is True


@pytest.mark.parametrize("mid", [m for m in MI.MODELS if MI.MODELS[m]["group"] == "aralikli"])
def test_intermittent_group_warns_about_rounding(mid):
    txt = (MI.MODELS[mid]["caution"] + MI.MODELS[mid]["how"] + MI.MODELS[mid]["use"]).lower()
    assert "rounding" in txt, f"{mid}: no warning about rounding to integers"

def test_cpp_engine_is_offered_for_its_methods():
    """Every method of the C++ engine appears in the interface catalogue with a C++ note, and the catalogue marks it
    available when the native library is present."""
    from tsfstudio.engines.cpp_engine import CppEngine
    meths = tuple(CppEngine.LOCAL_METHODS) + tuple(CppEngine.GLOBAL_METHODS)
    for mid in meths:
        assert "cpp" in MI.MODELS[mid]["engines"], mid
    cat = {m["id"]: m for m in MI.build_catalog({"cpp": {"local": CppEngine.LOCAL_METHODS, "global": CppEngine.GLOBAL_METHODS}})}
    assert all(cat[m]["engines"]["cpp"]["available"] for m in meths)
