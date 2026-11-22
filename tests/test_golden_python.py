"""Altın testler: Python motoru, kullanıcının KENDİ R çıktılarını (NN5 haftalık + günlük, 111 seri) yeniden üretiyor mu?

  * pooled_regression : R `glm` ile makine duyarlılığında aynı (1e-9).
  * ses / theta       : aynı algoritma; farklar yalnızca R'ın Nelder-Mead'inin SES alfa'sını sınırda bırakamadığı
                        serilerde (R'ın SSE'si daha YÜKSEK). Burada bu açıklama da test edilir.
  * ets, arima, dhr_arima (statsforecast; tbats devre dışı) : kurulu değilse atlanır; kurulu ise yalnızca KARAR DÜZEYİ
    eşiği sınanır (ortalama SMAPE farkı) ve bağıl fark raporlanır. Bu yöntemler R ile birebir aynı algoritma değildir.
"""
import numpy as np
import pytest

from tsfstudio.engines import PythonEngine
from tsfstudio.engines.py_models import ALPHA_LO, ses_fit
from tsfstudio.golden import compare, load_case

CASES = ["nn5_weekly", "nn5_daily"]


@pytest.fixture(scope="module", params=CASES)
def case(request):
    return load_case(request.param)


@pytest.fixture(scope="module")
def table(case):
    return compare(PythonEngine(), case).set_index("method")


def test_pooled_regression_equals_r_glm(table):
    r = table.loc["pooled_regression"]
    assert r["max_rel"] < 1e-9 and abs(r["dSMAPE_%"]) < 1e-9


@pytest.mark.parametrize("method", ["ses", "theta"])
def test_ses_theta_match_r_except_optimizer_artifacts(table, method):
    r = table.loc[method]
    assert r["median_rel"] < 1e-4                          # algoritma aynı
    assert r["n_gt_1e-3"] <= 0.08 * r["n_series"]          # sapma yalnızca az sayıda seride
    assert abs(r["dSMAPE_%"]) < 1.0 and abs(r["dMASE_%"]) < 1.0   # karar düzeyi etkisi < %1 (göreli)


def test_ses_deviations_occur_only_at_alpha_lower_bound(case):
    ref = case.forecasts["ses"]
    h = ref.shape[1]
    dev = 0
    for i, y in enumerate(case.train):
        alpha, level = ses_fit(y)
        rel = abs(level - ref[i, 0]) / abs(ref[i, 0])
        if rel > 1e-3:
            dev += 1
            assert alpha <= 1e-3, f"seri {i + 1}: alfa={alpha:.4f} sınırda değil ama R'dan {rel:.1e} farklı"
    assert dev <= 0.08 * len(case.train)


@pytest.mark.parametrize("method", ["ets", "arima", "dhr_arima"])
def test_statsforecast_methods_decision_level(case, method):
    eng = PythonEngine()
    if method not in eng.methods()["local"]:
        pytest.skip("statsforecast kurulu değil")
    if method not in case.forecasts:
        pytest.skip("bu veri setinde R referansı yok")
    r = compare(eng, case, methods=[method]).iloc[0]
    print(f"[altın] {case.name}/{method}: median_rel={r['median_rel']:.2e} dSMAPE={r['dSMAPE_%']:.2f}% "
          f"dMASE={r['dMASE_%']:.2f}% yedek={r['fallbacks']}")
    assert r["status"] == "tamam" and r["fallbacks"] <= 0.05 * r["n_series"]
    assert abs(r["dSMAPE_%"]) < 10.0              # R'dan %10'dan fazla farklı ortalama SMAPE: algoritma farkı sorgulanmalı


def test_limit_restricts_to_first_n_series():
    from tsfstudio.engines import PythonEngine
    from tsfstudio.golden import compare, load_case
    df = compare(PythonEngine(), load_case("nn5_weekly"), ["ses"], limit=8)
    assert int(df.iloc[0]["n_series"]) == 8
