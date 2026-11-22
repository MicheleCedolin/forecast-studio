import filecmp
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfstudio.seasonality import timestamps
from tsfstudio.tsf import TSFFormatError, read_tsf, read_tsf_header, write_dataset, write_tsf

SAMPLE = """# yorum satiri
@relation test
@attribute series_name string
@attribute n numeric
@attribute start_timestamp date
@frequency monthly
@horizon 3
@missing true
@equallength false
@data
T1:7:2020-01-01 00-00-00:1,2,?,4.5
T2:8:2019-05-01 00-00-00:3,3,3
"""


def _write(tmp_path, text, name="a.tsf"):
    p = tmp_path / name
    p.write_text(text, encoding="cp1252")
    return p


def test_header_only(tmp_path):
    h = read_tsf_header(_write(tmp_path, SAMPLE))
    assert (h.relation, h.frequency, h.horizon, h.missing, h.equal_length) == ("test", "monthly", 3, True, False)
    assert h.attributes == [("series_name", "string"), ("n", "numeric"), ("start_timestamp", "date")]
    assert h.comments == ["# yorum satiri"]


def test_read_values_missing_and_attrs(tmp_path):
    ds = read_tsf(_write(tmp_path, SAMPLE))
    assert ds.n_series == 2 and ds.series_names == ["T1", "T2"]
    assert np.isnan(ds.values[0][2]) and ds.values[0][3] == 4.5
    assert ds.attrs["n"].tolist() == [7, 8]
    assert ds.attrs["start_timestamp"].iloc[0] == pd.Timestamp("2020-01-01")
    assert ds.consistency_warnings() == []


def test_consistency_warning_when_header_lies(tmp_path):
    ds = read_tsf(_write(tmp_path, SAMPLE.replace("@missing true", "@missing false").replace("@equallength false", "@equallength true")))
    assert len(ds.consistency_warnings()) == 2


def test_max_series(tmp_path):
    assert read_tsf(_write(tmp_path, SAMPLE), max_series=1).n_series == 1


@pytest.mark.parametrize("bad,needle", [
    (SAMPLE.replace("T2:8:2019-05-01 00-00-00:3,3,3", "T2:8:3,3,3"), "fields separated"),
    (SAMPLE.replace("T2:8:2019-05-01 00-00-00:3,3,3", "T2:8:2019-05-01 00-00-00:?,?"), "all values in the series are missing"),
    (SAMPLE.replace("2019-05-01 00-00-00", "2019/05/01"), "invalid value"),
    (SAMPLE.replace("T1:7:", "T1:x:"), "invalid value"),
    (SAMPLE.replace("1,2,?,4.5", "1,abc,3"), "non-numeric"),
    (SAMPLE.replace("@data\n", ""), "@data"),
    (SAMPLE.replace("@attribute n numeric", "@attribute n integer"), "invalid attribute type"),
])
def test_format_errors_are_explicit(tmp_path, bad, needle):
    with pytest.raises(TSFFormatError, match=needle):
        read_tsf(_write(tmp_path, bad))


def test_roundtrip_is_byte_stable(tmp_path):
    p = _write(tmp_path, SAMPLE)
    ds = read_tsf(p)
    q = write_dataset(tmp_path / "b.tsf", ds)
    assert filecmp.cmp(p, q, shallow=False)


def test_writer_rejects_bad_names_and_colons(tmp_path):
    with pytest.raises(TSFFormatError):
        write_tsf(tmp_path / "x.tsf", relation="r", attributes=[("bad name", "string")], attr_rows=[["a"]],
                  values=[np.array([1.0])], frequency="daily", horizon=1, missing=False, equal_length=True)
    with pytest.raises(TSFFormatError):
        write_tsf(tmp_path / "x.tsf", relation="r", attributes=[("series_name", "string")], attr_rows=[["a:b"]],
                  values=[np.array([1.0])], frequency="daily", horizon=1, missing=False, equal_length=True)


def test_timestamps_follow_R_seq_semantics():
    assert list(timestamps("2020-01-01", 3, "monthly").strftime("%Y-%m-%d")) == ["2020-01-01", "2020-02-01", "2020-03-01"]
    assert list(timestamps("2020-01-01", 3, "quarterly").strftime("%Y-%m-%d")) == ["2020-01-01", "2020-04-01", "2020-07-01"]
    assert list(timestamps("2020-01-02", 2, "weekly").strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-09"]  # 7 gün, Pazar'a sabitleme yok
    assert str(timestamps("2020-01-01 00:00:00", 2, "half_hourly")[1]) == "2020-01-01 00:30:00"


M5_DIR = Path(os.environ.get("TSFSTUDIO_DATA_DIR", "data/tsf"))


@pytest.mark.skipif(not (M5_DIR / "m5_item_state_daily_dataset.tsf").is_file(), reason="M5 TSF dosyası yok")
def test_m5_level11_reads_and_roundtrips(tmp_path):
    p = M5_DIR / "m5_item_state_daily_dataset.tsf"
    ds = read_tsf(p)
    assert (ds.n_series, ds.frequency, ds.horizon, ds.header.missing, ds.header.equal_length) == (9147, "daily", 28, False, False)
    q = write_dataset(tmp_path / "rt.tsf", ds)
    assert filecmp.cmp(p, q, shallow=False)
