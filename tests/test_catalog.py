from pathlib import Path

import pytest

from tsfstudio import catalog as C
from tsfstudio.tsf import TSFHeader


def test_catalog_shape_and_uniqueness():
    cat = C.load_catalog()
    assert len(cat.entries) == 43
    assert len({e.id for e in cat.entries}) == 43 and len({e.file for e in cat.entries}) == 43
    by = {b: sum(1 for e in cat.entries if e.bundle == b) for b in ("main", "extra", "m5")}
    assert by == {"main": 24, "extra": 17, "m5": 2}
    assert sum(e.bytes for e in cat.entries) == pytest.approx(1.98e9, rel=0.01)


def test_defaults_taken_from_user_scripts():
    cat = C.load_catalog()
    e = cat.get("m4_hourly")
    assert (e.lag, e.lag_origin, e.integer_conversion) == (210, "study", False)
    assert cat.get("tourism_yearly").lag == 2
    assert cat.get("kdd_cup").horizon_override == 168
    assert cat.get("car_parts").integer_conversion is True and cat.get("car_parts").horizon_override == 12
    assert cat.get("nn5_weekly").local_methods[-1] == "dhr_arima"        # DHR-ARIMA yalnızca yüksek frekanslılarda
    assert "arima" in cat.get("m4_daily").local_methods and "dhr_arima" not in cat.get("m4_daily").local_methods
    assert "wavenet" not in cat.get("solar_10_minutes").deep_methods     # orijinalde WaveNet bu veride koşturulmamış
    assert cat.get("dominick").use_timestamps is False


def test_suggested_defaults_are_flagged():
    cat = C.load_catalog()
    for i in ("m5_item_store", "m5_item_state", "sample", "london_smart_meters"):
        assert cat.get(i).lag_origin == "suggested"
    assert cat.get("m5_item_store").wrmsse and not cat.get("m5_item_state").wrmsse
    assert {m["id"] for m in cat.not_bundled} == {"kaggle_web_traffic_daily", "saugeen_river_flow", "sunspot", "us_births"}


def test_effective_horizon_and_recommendation_rule():
    cat = C.load_catalog()
    assert cat.get("kdd_cup").effective_horizon(TSFHeader(horizon=48)) == 168      # dış ufuk önceliklidir
    assert cat.get("m4_daily").effective_horizon(TSFHeader(horizon=14)) == 14
    rec = cat.get("m5_item_store").recommended_local_methods(TSFHeader(frequency="daily"))
    assert rec == ("ses", "ets", "theta", "tbats", "dhr_arima")
    assert cat.get("london_smart_meters").recommended_local_methods(TSFHeader(frequency="half_hourly"))[-1] == "dhr_arima"


def test_availability_states(tmp_path: Path):
    cat = C.load_catalog()
    e = cat.get("m1_yearly")
    assert C.availability(e, tmp_path) == "missing"
    (tmp_path / e.file).write_bytes(b"x" * 10)
    assert C.availability(e, tmp_path) == "size_mismatch"
    (tmp_path / e.file).write_bytes(b"x" * e.bytes)
    assert C.availability(e, tmp_path) == "ok"
    rep = C.check_bundle(cat, tmp_path)
    assert rep["ok"] == ["m1_yearly"] and len(rep["missing"]) == 42
