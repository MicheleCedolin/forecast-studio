import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tsfstudio.excel import build_template, excel_to_tsf, sanitize_name
from tsfstudio.tsf import read_tsf


def make_book(path, meta, long=None, wide=None):
    """meta: dict, long/wide: list-of-rows (ilk satır başlık)."""
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame([["key", "value"]] + [[k, v] for k, v in meta.items()]).to_excel(xw, "Meta", header=False, index=False)
        pd.DataFrame(long if long else [["series_name", "date", "value"]]).to_excel(xw, "Veri_Uzun", header=False, index=False)
        pd.DataFrame(wide if wide else [["date", "S1"]]).to_excel(xw, "Veri_Genis", header=False, index=False)
    return path


META_M = {"dataset_name": "t", "frequency": "monthly", "horizon": 2}


def months(n, start=(2020, 1), day=1):
    y, m = start
    out = []
    for _ in range(n):
        out.append(dt.datetime(y, m, day))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def long_rows(names, n=12, **kw):
    rows = [["series_name", "date", "value"]]
    for k, nm in enumerate(names):
        for i, d in enumerate(months(n, **kw)):
            rows.append([nm, d, float(i + 10 * k)])
    return rows


def test_template_roundtrip_default_example(tmp_path):
    p = tmp_path / "tpl.xlsx"
    p.write_bytes(build_template())
    out = tmp_path / "out.tsf"
    ds, rep = excel_to_tsf(p, out)
    assert rep.ok, rep.text()
    assert (ds.n_series, ds.frequency, ds.horizon) == (2, "monthly", 6)
    assert ds.lengths.tolist() == [36, 36] and ds.attrs["region"].tolist() == ["north", "south"]
    back = read_tsf(out)                       # Monash biçimi kurallarına uyuyor mu?
    assert back.n_series == 2 and back.header.attribute_names == ["series_name", "region", "start_timestamp"]
    assert back.attrs["start_timestamp"].iloc[0] == pd.Timestamp("2021-01-01")
    np.testing.assert_allclose(back.values[0], ds.values[0])
    assert back.consistency_warnings() == []


def test_template_is_bytes_or_file(tmp_path):
    assert isinstance(build_template(), bytes)
    assert build_template(tmp_path / "x.xlsx") == tmp_path / "x.xlsx"


def test_wide_format_with_missing_cells(tmp_path):
    wide = [["date", "A", "B"]] + [[d, float(i), (None if i == 3 else float(i * 2))] for i, d in enumerate(months(8))]
    p = make_book(tmp_path / "w.xlsx", META_M, wide=wide)
    ds, rep = excel_to_tsf(p)
    assert rep.ok, rep.text()
    assert ds.n_series == 2 and np.isnan(ds.values[1][3]) and ds.header.missing is True
    assert any("eksik değer" in w for w in rep.warnings)


def test_both_sheets_filled_is_error(tmp_path):
    p = make_book(tmp_path / "b.xlsx", META_M, long=long_rows(["A"]), wide=[["date", "A"], [dt.datetime(2020, 1, 1), 1.0]])
    ds, rep = excel_to_tsf(p)
    assert ds is None and "ikisi" not in rep.text() and "Yalnızca birini" in rep.text()


@pytest.mark.parametrize("meta,needle", [
    ({**META_M, "frequency": "aylik"}, "frequency='aylik' geçersiz"),
    ({**META_M, "horizon": 0}, "pozitif bir tamsayı"),
    ({**META_M, "horizon": 2.5}, "pozitif bir tamsayı"),
    ({"dataset_name": "t", "frequency": "monthly"}, "horizon"),
])
def test_meta_errors(tmp_path, meta, needle):
    p = make_book(tmp_path / "m.xlsx", meta, long=long_rows(["A"]))
    ds, rep = excel_to_tsf(p)
    assert ds is None and needle in rep.text()


def test_gap_is_error_unless_fill_requested(tmp_path):
    rows = long_rows(["A"], 12)
    del rows[5]                                  # bir ayı sil
    p = make_book(tmp_path / "g.xlsx", META_M, long=rows)
    ds, rep = excel_to_tsf(p)
    assert ds is None and "1 eksik tarih" in rep.text() and "2020-05-01" in rep.text()
    ds, rep = excel_to_tsf(p, fill_gaps=True)
    assert rep.ok and ds.lengths.tolist() == [12] and np.isnan(ds.values[0][4]) and ds.header.missing


def test_duplicate_date_and_text_value_report_excel_cells(tmp_path):
    # Hata raporu iki aşamalıdır: önce hücre hataları (yanıltıcı art arda hataları önlemek için),
    # hücreler düzelince yapısal hatalar (yinelenen tarih, boşluk, düzensiz frekans...).
    rows = long_rows(["A"], 12)
    rows[3][1] = rows[2][1]                      # yinelenen tarih
    rows[6][2] = "1,5"                           # ondalık virgül
    ds, rep = excel_to_tsf(make_book(tmp_path / "d.xlsx", META_M, long=rows))
    assert ds is None and "Veri_Uzun!C7" in rep.text() and "'1,5' sayı değil" in rep.text()
    rows[6][2] = 1.5                             # hücre hatası giderildi
    ds, rep = excel_to_tsf(make_book(tmp_path / "d2.xlsx", META_M, long=rows))
    assert ds is None and "aynı tarih birden fazla" in rep.text()


def test_month_end_dates_are_normalised_with_warning(tmp_path):
    rows = [["series_name", "date", "value"]]
    ends = [dt.datetime(2020, 1, 31), dt.datetime(2020, 2, 29), dt.datetime(2020, 3, 31), dt.datetime(2020, 4, 30), dt.datetime(2020, 5, 31), dt.datetime(2020, 6, 30)]
    rows += [["A", d, float(i)] for i, d in enumerate(ends)]
    p = make_book(tmp_path / "e.xlsx", META_M, long=rows)
    ds, rep = excel_to_tsf(p)
    assert rep.ok, rep.text()
    assert ds.attrs["start_timestamp"].iloc[0] == pd.Timestamp("2020-01-01")
    assert any("1. gününe" in w for w in rep.warnings)


def test_irregular_weekly_and_yearly_integers(tmp_path):
    wk = [["series_name", "date", "value"]] + [["A", dt.datetime(2020, 1, 1) + dt.timedelta(days=7 * i + (1 if i == 4 else 0)), float(i)] for i in range(8)]
    ds, rep = excel_to_tsf(make_book(tmp_path / "wk.xlsx", {**META_M, "frequency": "weekly"}, long=wk))
    assert ds is None and "frekansına uymuyor" in rep.text()
    yr = [["series_name", "date", "value"]] + [["A", 2010 + i, float(i)] for i in range(8)]
    ds, rep = excel_to_tsf(make_book(tmp_path / "yr.xlsx", {**META_M, "frequency": "yearly"}, long=yr))
    assert rep.ok and ds.attrs["start_timestamp"].iloc[0] == pd.Timestamp("2010-01-01") and any("tamsayı yıllar" in w for w in rep.warnings)


def test_series_too_short_all_missing_and_name_collisions(tmp_path):
    ds, rep = excel_to_tsf(make_book(tmp_path / "s.xlsx", META_M, long=long_rows(["A"], 2)))
    assert ds is None and "eğitim verisi kalmıyor" in rep.text()
    rows = long_rows(["A"], 6)
    for r in rows[1:]:
        r[2] = None
    ds, rep = excel_to_tsf(make_book(tmp_path / "am.xlsx", META_M, long=rows))
    assert ds is None and "tüm değerler eksik" in rep.text()
    both = long_rows(["Şeker", "Seker"], 6)
    ds, rep = excel_to_tsf(make_book(tmp_path / "c.xlsx", META_M, long=both))
    assert ds is None and "aynı olan seri adları" in rep.text()


def test_turkish_names_become_ascii(tmp_path):
    assert sanitize_name("Çiğdem Işık: Şişli/İstanbul") == "Cigdem_Isik__Sisli_Istanbul"
    ds, rep = excel_to_tsf(make_book(tmp_path / "t.xlsx", META_M, long=long_rows(["Ürün Ağacı"], 8)))
    assert rep.ok and ds.series_names == ["Urun_Agaci"] and any("ASCII" in w for w in rep.warnings)


def test_unequal_lengths_flagged_and_english_messages(tmp_path):
    rows = long_rows(["A"], 10) + long_rows(["B"], 7)[1:]
    ds, rep = excel_to_tsf(make_book(tmp_path / "u.xlsx", META_M, long=rows))
    assert rep.ok and ds.header.equal_length is False
    ds, rep = excel_to_tsf(make_book(tmp_path / "en.xlsx", {**META_M, "frequency": "x"}, long=long_rows(["A"])), lang="en")
    assert ds is None and "is invalid" in rep.text()


def test_unreadable_file_reports_error(tmp_path):
    p = tmp_path / "bad.xlsx"
    p.write_bytes(b"not an excel file")
    ds, rep = excel_to_tsf(p)
    assert ds is None and "okunamadı" in rep.text()


def test_error_listing_is_capped(tmp_path):
    rows = long_rows(["A"], 30)
    for r in rows[1:]:
        r[2] = "x"
    ds, rep = excel_to_tsf(make_book(tmp_path / "many.xlsx", META_M, long=rows))
    assert ds is None and len(rep.errors) == 13 and "ve 18 tane daha" in rep.errors[-1]
