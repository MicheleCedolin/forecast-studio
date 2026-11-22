"""Excel -> TSF dönüştürücü.

Beklenen çalışma kitabı (şablon `build_template` ile üretilir):
  * `Meta`       : anahtar/değer satırları — dataset_name, frequency, horizon, (opsiyonel) description
  * `Veri_Uzun`  : uzun biçim — series_name | date | value | [ek öznitelik sütunları]
  * `Veri_Genis` : geniş biçim — ilk sütun date, diğer sütun başlıkları seri adı
İkisinden yalnızca biri dolu olmalıdır.

Doğrulamalar (hatalar satır/sütun numarasıyla Excel sayfasına göre bildirilir):
  frekans geçerli mi, ufuk pozitif tamsayı mı, tarihler çözümlenebilir/tekil/düzenli mi, değerler sayısal mı,
  seri tamamen eksik mi, eğitim kısmı kalıyor mu, seri adları (ASCII'ye çevrilir) tekil mi.
Aylık/çeyreklik/yıllık veride tarihler dönemin 1. gününe çekilir (R `seq(by="1 month")` ay sonu tarihlerinde
kayar ve tsibble indeksini bozar; bu nedenle normalize edilir ve uyarı verilir).
"""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from .. import __version__
from ..seasonality import FREQUENCIES, HIGH_FREQUENCIES, has_time_of_day, timestamps
from ..tsf import TSFDataset, TSFHeader
from ..tsf.writer import write_dataset

SHEET_META = "Meta"
SHEET_LONG = "Veri_Uzun"
SHEET_WIDE = "Veri_Genis"
RESERVED_ATTRS = {"series_name", "start_timestamp", "date", "value"}
MISSING_TOKENS = {"", "?", "na", "n/a", "nan", "null", "none"}
_PERIOD = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}
_TR_MAP = str.maketrans({"ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G", "ü": "u", "Ü": "U",
                         "ö": "o", "Ö": "O", "ç": "c", "Ç": "C"})
MAX_LISTED = 12  # her hata/uyarı kodu için en çok kaç örnek yazılır

MSG = {
    "tr": {
        "no_sheet": "'{sheet}' sayfası bulunamadı. Şablondaki sayfa adlarını değiştirmeyin.",
        "no_data": "Veri_Uzun ve Veri_Genis sayfalarının ikisi de boş. Verinizi birine yazın.",
        "both_data": "Hem Veri_Uzun hem Veri_Genis dolu. Yalnızca birini kullanın (diğerini temizleyin).",
        "meta_missing": "Meta sayfasında '{key}' değeri eksik.",
        "bad_freq": "Meta: frequency='{v}' geçersiz. Geçerli değerler: {valid}.",
        "bad_horizon": "Meta: horizon='{v}' pozitif bir tamsayı olmalı.",
        "long_cols": "Veri_Uzun başlığı 'series_name', 'date', 'value' sütunlarını içermeli (bulunan: {found}).",
        "bad_attr_name": "Veri_Uzun: '{v}' ek sütun adı yalnızca harf/rakam/alt çizgi içermeli (ve ayrılmış ad olmamalı).",
        "attr_not_const": "Seri '{s}': '{a}' ek sütunu seri içinde sabit olmalı (farklı değerler: {v}).",
        "bad_date": "{where}: '{v}' tarih olarak çözümlenemedi. Excel tarih hücresi ya da YYYY-AA-GG biçimi kullanın.",
        "blank_date": "{where}: tarih boş.",
        "blank_series": "{where}: series_name boş.",
        "bad_value": "{where}: '{v}' sayı değil (ondalık için nokta veya sayısal hücre kullanın; eksik için hücreyi boş bırakın).",
        "inf_value": "{where}: sonsuz değer kabul edilmez.",
        "dup_dates": "Seri '{s}': aynı tarih birden fazla kez var (ilk: {d}).",
        "gap": "Seri '{s}': {n} eksik tarih var (ilk: {d}). Boşlukları doldurun ya da dönüştürücüde 'boşlukları eksik değerle doldur' seçeneğini açın.",
        "gap_filled": "Seri '{s}': {n} eksik tarih eksik değer olarak eklendi (ilk: {d}).",
        "irregular": "Seri '{s}': tarihler '{f}' frekansına uymuyor (ilk uyumsuz: {d}).",
        "all_missing": "Seri '{s}': tüm değerler eksik.",
        "too_short": "Seri '{s}': uzunluk ({n}) ufuktan ({h}) uzun olmalı; eğitim verisi kalmıyor.",
        "short_warn": "Seri '{s}': uzunluk ({n}) ufkun iki katından ({h2}) kısa; sonuçlar güvenilir olmayabilir.",
        "dup_names": "Sadeleştirmeden sonra aynı olan seri adları: {v}.",
        "renamed": "Seri adı ASCII/TSF kurallarına göre değiştirildi: {v}.",
        "period_normalized": "Seri '{s}': tarihler dönemin 1. gününe çekildi (ilk: {a} -> {b}).",
        "time_dropped": "Seri '{s}': {f} frekansında saat bilgisi yok sayıldı.",
        "year_int": "Yıllık veri: tamsayı yıllar ({v}...) 1 Ocak olarak yorumlandı.",
        "unsorted": "Seri '{s}': satırlar tarihe göre sıralı değildi; sıralandı.",
        "missing_vals": "Veride eksik değer var ({n}). Modeller çalışmadan önce eksik değer işleme (ör. doğrusal enterpolasyon) gerekir.",
        "unequal": "Seriler farklı uzunlukta (@equallength=false olarak yazıldı).",
        "empty": "Dönüştürülecek seri bulunamadı.",
        "read_fail": "Excel dosyası okunamadı: {v}",
        "more": "... ve {n} tane daha",
    },
    "en": {
        "no_sheet": "Sheet '{sheet}' not found. Do not rename template sheets.",
        "no_data": "Both Veri_Uzun and Veri_Genis are empty. Enter your data in one of them.",
        "both_data": "Both Veri_Uzun and Veri_Genis contain data. Use only one (clear the other).",
        "meta_missing": "Meta sheet: value for '{key}' is missing.",
        "bad_freq": "Meta: frequency='{v}' is invalid. Valid values: {valid}.",
        "bad_horizon": "Meta: horizon='{v}' must be a positive integer.",
        "long_cols": "Veri_Uzun header must contain 'series_name', 'date', 'value' (found: {found}).",
        "bad_attr_name": "Veri_Uzun: extra column name '{v}' may only contain letters/digits/underscore (and must not be a reserved name).",
        "attr_not_const": "Series '{s}': extra column '{a}' must be constant within a series (values: {v}).",
        "bad_date": "{where}: '{v}' could not be parsed as a date. Use an Excel date cell or YYYY-MM-DD.",
        "blank_date": "{where}: date is empty.",
        "blank_series": "{where}: series_name is empty.",
        "bad_value": "{where}: '{v}' is not a number (use a dot or a numeric cell; leave the cell empty for missing).",
        "inf_value": "{where}: infinite values are not allowed.",
        "dup_dates": "Series '{s}': the same date appears more than once (first: {d}).",
        "gap": "Series '{s}': {n} dates are missing (first: {d}). Fill the gaps or enable 'fill gaps with missing values'.",
        "gap_filled": "Series '{s}': {n} missing dates were added as missing values (first: {d}).",
        "irregular": "Series '{s}': dates do not follow frequency '{f}' (first mismatch: {d}).",
        "all_missing": "Series '{s}': all values are missing.",
        "too_short": "Series '{s}': length ({n}) must exceed the horizon ({h}); no training data left.",
        "short_warn": "Series '{s}': length ({n}) is below twice the horizon ({h2}); results may be unreliable.",
        "dup_names": "Series names that collide after sanitising: {v}.",
        "renamed": "Series names were changed to fit ASCII/TSF rules: {v}.",
        "period_normalized": "Series '{s}': dates were moved to the first day of the period (first: {a} -> {b}).",
        "time_dropped": "Series '{s}': time of day ignored for frequency {f}.",
        "year_int": "Yearly data: integer years ({v}...) were interpreted as 1 January.",
        "unsorted": "Series '{s}': rows were not sorted by date; they were sorted.",
        "missing_vals": "The data contain missing values ({n}). A missing-value treatment (e.g. linear interpolation) is needed before running models.",
        "unequal": "Series have different lengths (written as @equallength=false).",
        "empty": "No series to convert.",
        "read_fail": "Could not read the Excel file: {v}",
        "more": "... and {n} more",
    },
}


@dataclass
class ConversionReport:
    lang: str = "tr"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    _counts: dict = field(default_factory=dict)
    _extra_pos: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def _add(self, bucket: list[str], code: str, **kw) -> None:
        key = (id(bucket), code)
        n = self._counts.get(key, 0)
        self._counts[key] = n + 1
        if n < MAX_LISTED:
            bucket.append(MSG[self.lang][code].format(**kw))
        elif n == MAX_LISTED:
            bucket.append("")  # yer tutucu; finalize() kalan sayıyı yazar
            self._extra_pos[key] = len(bucket) - 1

    def error(self, code: str, **kw) -> None:
        self._add(self.errors, code, **kw)

    def warn(self, code: str, **kw) -> None:
        self._add(self.warnings, code, **kw)

    def finalize(self) -> "ConversionReport":
        for bucket in (self.errors, self.warnings):
            for (bid, code), pos in self._extra_pos.items():
                if bid == id(bucket):
                    bucket[pos] = MSG[self.lang]["more"].format(n=self._counts[(bid, code)] - MAX_LISTED)
        return self

    def text(self) -> str:
        out = [("HATA: " if self.lang == "tr" else "ERROR: ") + e for e in self.errors]
        out += [("UYARI: " if self.lang == "tr" else "WARNING: ") + w for w in self.warnings]
        return "\n".join(out)


# ----------------------------------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    s = str(name).strip().translate(_TR_MAP)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.\-]", "_", s)
    return s


def _ascii_text(text: str) -> str:
    """Serbest metin (açıklama) için: Türkçe karakterleri ASCII'ye çevir, noktalama ve boşlukları koru."""
    t = str(text).translate(_TR_MAP)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip()


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "") or v is pd.NaT


def _parse_value(v):
    """-> (float | nan, error_kind | None)"""
    if _is_blank(v):
        return np.nan, None
    if isinstance(v, str):
        if v.strip().lower() in MISSING_TOKENS:
            return np.nan, None
        try:
            x = float(v.strip())
        except ValueError:
            return np.nan, "bad"
    elif isinstance(v, (bool, np.bool_)):
        return np.nan, "bad"
    else:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return np.nan, "bad"
    if np.isinf(x):
        return np.nan, "inf"
    return x, None


_ISO = re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2})[:\-](\d{2})(?:[:\-](\d{2}))?)?\s*$")


def _parse_date(v, frequency: str | None, report_flags: dict):
    if _is_blank(v):
        return None, "blank"
    if isinstance(v, pd.Timestamp):
        return v, None
    if hasattr(v, "year") and hasattr(v, "month") and hasattr(v, "day"):  # datetime/date
        return pd.Timestamp(v), None
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
        y = int(v)
        if frequency == "yearly" and 1000 <= y <= 9999:
            report_flags["year_int"] = y
            return pd.Timestamp(year=y, month=1, day=1), None
        return None, "bad"
    if isinstance(v, str):
        m = _ISO.match(v)
        if not m:
            return None, "bad"
        y, mo, d, hh, mi, ss = m.groups()
        try:
            return pd.Timestamp(int(y), int(mo), int(d), int(hh or 0), int(mi or 0), int(ss or 0)), None
        except ValueError:
            return None, "bad"
    return None, "bad"


def _read_sheets(src, report: ConversionReport) -> dict[str, pd.DataFrame] | None:
    try:
        if isinstance(src, (bytes, bytearray)):
            src = io.BytesIO(src)
        return pd.read_excel(src, sheet_name=None, header=None, dtype=object, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001 — kullanıcıya anlaşılır mesaj
        report.error("read_fail", v=str(exc)[:200])
        return None


def _parse_meta(df: pd.DataFrame, report: ConversionReport) -> dict:
    kv = {}
    for _, row in df.iterrows():
        k = row.iloc[0]
        if _is_blank(k):
            continue
        v = row.iloc[1] if len(row) > 1 else None
        kv[str(k).strip().lower()] = None if _is_blank(v) else (v.strip() if isinstance(v, str) else v)
    return kv


def _non_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    body = df.iloc[1:] if len(df) else df
    return body.dropna(how="all")


def excel_to_tsf(
    src: str | Path | bytes | BinaryIO,
    out_path: str | Path | None = None,
    *,
    fill_gaps: bool = False,
    lang: str = "tr",
) -> tuple[TSFDataset | None, ConversionReport]:
    """Şablona uygun Excel'i TSFDataset'e çevirir; `out_path` verilir ve hata yoksa .tsf dosyası da yazılır."""
    rep = ConversionReport(lang=lang)
    sheets = _read_sheets(src, rep)
    if sheets is None:
        return None, rep.finalize()
    for name in (SHEET_META,):
        if name not in sheets:
            rep.error("no_sheet", sheet=name)
    if rep.errors:
        return None, rep.finalize()

    meta = _parse_meta(sheets[SHEET_META], rep)
    relation = sanitize_name(str(meta.get("dataset_name") or "user_dataset")) or "user_dataset"
    freq = meta.get("frequency")
    freq = str(freq).strip().lower() if freq is not None else None
    if freq is None:
        rep.error("meta_missing", key="frequency")
    elif freq not in FREQUENCIES:
        rep.error("bad_freq", v=freq, valid=", ".join(FREQUENCIES))
        freq = None
    horizon = None
    hv = meta.get("horizon")
    if hv is None:
        rep.error("meta_missing", key="horizon")
    else:
        try:
            horizon = int(float(hv))
            if horizon <= 0 or float(hv) != horizon:
                raise ValueError
        except (TypeError, ValueError):
            rep.error("bad_horizon", v=hv)
            horizon = None

    long_df = _non_empty_rows(sheets[SHEET_LONG]) if SHEET_LONG in sheets else pd.DataFrame()
    wide_df = _non_empty_rows(sheets[SHEET_WIDE]) if SHEET_WIDE in sheets else pd.DataFrame()
    if long_df.empty and wide_df.empty:
        rep.error("no_data")
    elif not long_df.empty and not wide_df.empty:
        rep.error("both_data")
    if rep.errors:
        return None, rep.finalize()

    flags: dict = {}
    series_rows = _collect_long(sheets[SHEET_LONG], rep, freq, flags) if not long_df.empty else _collect_wide(sheets[SHEET_WIDE], rep, freq, flags)
    if rep.errors or not series_rows:
        if not series_rows and not rep.errors:
            rep.error("empty")
        return None, rep.finalize()
    if "year_int" in flags:
        rep.warn("year_int", v=flags["year_int"])

    ds = _build_dataset(series_rows, relation, freq, horizon, meta, rep, fill_gaps)
    rep.finalize()
    if ds is None or rep.errors:
        return None, rep
    if out_path is not None:
        write_dataset(out_path, ds)
    rep.info.update({"n_series": ds.n_series, "len_min": int(ds.lengths.min()), "len_max": int(ds.lengths.max()),
                     "frequency": freq, "horizon": horizon})
    return ds, rep


# ----------------------------------------------------------------------------------------------
# Satır toplama (uzun / geniş)
# ----------------------------------------------------------------------------------------------
@dataclass
class _Raw:
    name: str
    attrs: dict
    dates: list
    values: list
    rows: list  # Excel satır numaraları (hata mesajları için)


def _collect_long(df: pd.DataFrame, rep: ConversionReport, freq, flags) -> list[_Raw]:
    header = [("" if _is_blank(h) else str(h).strip()) for h in df.iloc[0].tolist()]
    low = [h.lower() for h in header]
    need = {"series_name", "date", "value"}
    if not need.issubset(low):
        rep.error("long_cols", found=", ".join(h for h in header if h) or "-")
        return []
    ci = {n: low.index(n) for n in need}
    extra = [(i, h) for i, h in enumerate(header) if h and h.lower() not in need]
    for _, h in extra:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", h) or h.lower() in RESERVED_ATTRS:
            rep.error("bad_attr_name", v=h)
    if rep.errors:
        return []
    groups: dict[str, _Raw] = {}
    order: list[str] = []
    for r in range(1, len(df)):
        row = df.iloc[r].tolist()
        if all(_is_blank(c) for c in row):
            continue
        xl = r + 1  # Excel satırı (başlık = 1)
        name = row[ci["series_name"]]
        if _is_blank(name):
            rep.error("blank_series", where=f"{SHEET_LONG}!A{xl}")
            continue
        name = str(name).strip()
        d, derr = _parse_date(row[ci["date"]], freq, flags)
        v, verr = _parse_value(row[ci["value"]])
        if derr == "blank":
            rep.error("blank_date", where=f"{SHEET_LONG}!B{xl}")
        elif derr:
            rep.error("bad_date", where=f"{SHEET_LONG}!B{xl}", v=row[ci["date"]])
        if verr == "bad":
            rep.error("bad_value", where=f"{SHEET_LONG}!C{xl}", v=row[ci["value"]])
        elif verr == "inf":
            rep.error("inf_value", where=f"{SHEET_LONG}!C{xl}")
        g = groups.get(name)
        if g is None:
            g = groups[name] = _Raw(name, {h: set() for _, h in extra}, [], [], [])
            order.append(name)
        for i, h in extra:
            if not _is_blank(row[i]):
                g.attrs[h].add(str(row[i]).strip())
        g.dates.append(d)
        g.values.append(v)
        g.rows.append(xl)
    out = []
    for n in order:
        g = groups[n]
        for a, vals in g.attrs.items():
            if len(vals) > 1:
                rep.error("attr_not_const", s=n, a=a, v=", ".join(sorted(vals)[:4]))
        g.attrs = {a: (next(iter(v)) if v else "") for a, v in g.attrs.items()}
        out.append(g)
    return out


def _collect_wide(df: pd.DataFrame, rep: ConversionReport, freq, flags) -> list[_Raw]:
    header = df.iloc[0].tolist()
    names = [("" if _is_blank(h) else str(h).strip()) for h in header[1:]]
    cols = [(j + 1, n) for j, n in enumerate(names) if n]
    raws = {n: _Raw(n, {}, [], [], []) for _, n in cols}
    if len(raws) != len(cols):
        rep.error("dup_names", v=", ".join(sorted({n for _, n in cols if [m for _, m in cols].count(n) > 1})[:6]))
        return []
    from openpyxl.utils import get_column_letter
    for r in range(1, len(df)):
        row = df.iloc[r].tolist()
        if all(_is_blank(c) for c in row):
            continue
        xl = r + 1
        d, derr = _parse_date(row[0], freq, flags)
        if derr == "blank":
            rep.error("blank_date", where=f"{SHEET_WIDE}!A{xl}")
            continue
        if derr:
            rep.error("bad_date", where=f"{SHEET_WIDE}!A{xl}", v=row[0])
            continue
        for j, n in cols:
            v, verr = _parse_value(row[j])
            where = f"{SHEET_WIDE}!{get_column_letter(j + 1)}{xl}"
            if verr == "bad":
                rep.error("bad_value", where=where, v=row[j])
            elif verr == "inf":
                rep.error("inf_value", where=where)
            raws[n].dates.append(d)
            raws[n].values.append(v)
            raws[n].rows.append(xl)
    # geniş biçimde boş hücre = eksik değer; baştaki/sondaki tamamen boş kısımlar kırpılır
    out = []
    for _, n in cols:
        g = raws[n]
        vals = np.array(g.values, dtype=float)
        if vals.size and not np.isnan(vals).all():
            first, last = np.flatnonzero(~np.isnan(vals))[[0, -1]]
            g.dates, g.values, g.rows = g.dates[first:last + 1], g.values[first:last + 1], g.rows[first:last + 1]
        out.append(g)
    return out


# ----------------------------------------------------------------------------------------------
# Düzenlilik denetimi ve TSFDataset üretimi
# ----------------------------------------------------------------------------------------------
def _regularize(raw: _Raw, freq: str, rep: ConversionReport, fill_gaps: bool):
    """-> (start Timestamp, değer dizisi) ya da None (hata)."""
    s = raw.name
    dates = pd.DatetimeIndex(raw.dates)
    vals = np.array(raw.values, dtype=float)
    if dates.has_duplicates:
        rep.error("dup_dates", s=s, d=str(dates[dates.duplicated()][0].date()))
        return None
    if not dates.is_monotonic_increasing:
        order = np.argsort(dates.values, kind="stable")
        dates, vals = dates[order], vals[order]
        rep.warn("unsorted", s=s)
    if freq in _PERIOD:
        per = dates.to_period(_PERIOD[freq])
        if per.has_duplicates:
            rep.error("dup_dates", s=s, d=str(dates[per.duplicated()][0].date()))
            return None
        starts = per.start_time
        if (dates != starts).any():
            rep.warn("period_normalized", s=s, a=str(dates[dates != starts][0].date()), b=str(starts[dates != starts][0].date()))
        full = pd.period_range(per[0], per[-1], freq=_PERIOD[freq])
        idx = np.array([p.ordinal for p in per]) - full[0].ordinal
        n_full = len(full)
        start = full[0].start_time
    else:
        if freq in HIGH_FREQUENCIES and (dates != dates.normalize()).any():
            rep.warn("time_dropped", s=s, f=freq)
            dates = dates.normalize()
            if dates.has_duplicates:
                rep.error("dup_dates", s=s, d=str(dates[dates.duplicated()][0].date()))
                return None
        step = timestamps(dates[0], 2, freq)[1] - dates[0]
        diffs = (dates[1:] - dates[:-1]) if len(dates) > 1 else pd.TimedeltaIndex([])
        bad = np.flatnonzero((diffs % step) != pd.Timedelta(0)) if len(diffs) else np.array([], dtype=int)
        if bad.size:
            rep.error("irregular", s=s, f=freq, d=str(dates[bad[0] + 1]))
            return None
        idx = np.asarray((dates - dates[0]) // step, dtype=np.int64)
        n_full = int(idx[-1]) + 1
        start = dates[0]
    if n_full != len(dates):
        missing_pos = np.setdiff1d(np.arange(n_full), idx)
        first_missing = (timestamps(start, n_full, freq)[missing_pos[0]] if not freq in _PERIOD
                         else (pd.Period(start, freq=_PERIOD[freq]) + int(missing_pos[0])).start_time)
        if not fill_gaps:
            rep.error("gap", s=s, n=len(missing_pos), d=str(first_missing.date() if freq in HIGH_FREQUENCIES else first_missing))
            return None
        rep.warn("gap_filled", s=s, n=len(missing_pos), d=str(first_missing.date() if freq in HIGH_FREQUENCIES else first_missing))
        full_vals = np.full(n_full, np.nan)
        full_vals[idx] = vals
        vals = full_vals
    return start, vals


def _build_dataset(raws: list[_Raw], relation, freq, horizon, meta, rep: ConversionReport, fill_gaps: bool) -> TSFDataset | None:
    if freq is None or horizon is None:
        return None
    # seri adları
    san = [sanitize_name(r.name) for r in raws]
    changed = [f"{r.name!r}->{s!r}" for r, s in zip(raws, san) if r.name != s]
    if changed:
        rep.warn("renamed", v="; ".join(changed[:5]) + (" ..." if len(changed) > 5 else ""))
    dup = sorted({s for s in san if san.count(s) > 1})
    if dup or any(s == "" for s in san):
        rep.error("dup_names", v=", ".join(dup[:6]) or "''")
        return None
    starts, arrs, kept = [], [], []
    for raw, name in zip(raws, san):
        res = _regularize(raw, freq, rep, fill_gaps)
        if res is None:
            continue
        start, vals = res
        if np.isnan(vals).all():
            rep.error("all_missing", s=raw.name)
            continue
        if len(vals) <= horizon:
            rep.error("too_short", s=raw.name, n=len(vals), h=horizon)
            continue
        if len(vals) < 2 * horizon:
            rep.warn("short_warn", s=raw.name, n=len(vals), h2=2 * horizon)
        starts.append(start)
        arrs.append(vals)
        kept.append((raw, name))
    if rep.errors:
        return None
    attr_names = list(kept[0][0].attrs) if kept and kept[0][0].attrs else []
    attrs = {"series_name": [n for _, n in kept]}
    for a in attr_names:
        attrs[a] = [sanitize_name(raw.attrs.get(a, "")) or "NA" for raw, _ in kept]
    attrs["start_timestamp"] = [pd.Timestamp(s) for s in starts]
    df = pd.DataFrame(attrs)
    n_missing = int(sum(int(np.isnan(v).sum()) for v in arrs))
    lengths = {len(v) for v in arrs}
    if n_missing:
        rep.warn("missing_vals", n=n_missing)
    if len(lengths) > 1:
        rep.warn("unequal")
    comments = ["# Dataset Information", f"# Converted from a user-supplied Excel workbook by Forecast Studio {__version__}."]
    desc = meta.get("description")
    if desc:
        comments.append("# " + _ascii_text(desc)[:200])
    header = TSFHeader(
        relation=relation,
        attributes=[("series_name", "string")] + [(a, "string") for a in attr_names] + [("start_timestamp", "date")],
        frequency=freq, horizon=horizon, missing=bool(n_missing), equal_length=len(lengths) == 1, comments=comments,
    )
    return TSFDataset(header=header, attrs=df, values=arrs, source="excel")
