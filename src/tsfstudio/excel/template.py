"""Excel template for users (Instructions + Meta + Veri_Uzun + Veri_Genis)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

from ..seasonality import FREQUENCIES, SEASONALITY_MAP
from .converter import SHEET_LONG, SHEET_META, SHEET_WIDE

_STEP_TEXT = {
    "4_seconds": "every 4 seconds", "minutely": "every minute", "10_minutes": "every 10 minutes",
    "15_minutes": "every 15 minutes", "half_hourly": "every 30 minutes", "hourly": "hourly",
    "daily": "daily", "weekly": "exactly 7 days apart", "monthly": "monthly (the 1st of the month is recommended)",
    "quarterly": "quarterly (1st day of the period)", "yearly": "yearly (1 January)",
}

INSTRUCTIONS = [
    ("FORECAST STUDIO — DATA TEMPLATE", "title"),
    ("This workbook is prepared so that you can upload your own time series to the application. The application converts it to the Monash .tsf format and then treats it like any other dataset.", None),
    ("", None),
    ("1) Sheets", "h"),
    ("• Meta: the dataset name, the frequency (choose from the drop-down list) and the forecast horizon (how many steps ahead to forecast).", None),
    ("• Veri_Uzun (\"long data\", recommended): one observation per row — series_name | date | value. You may add extra columns to the right (e.g. region, store); they become series-level attributes and must be constant within each series.", None),
    ("• Veri_Genis (\"wide data\", alternative): the first column is date and every following column is one series (header = series name). An empty cell = missing value.", None),
    ("• Write your data in ONE of the two data sheets only. Delete the example rows in the template; leave the data sheet you do not use empty (header row only). The sheet names are kept as they are for compatibility with files made earlier.", None),
    ("", None),
    ("2) Rules", "h"),
    ("• Dates: an Excel date cell or YYYY-MM-DD (YYYY-MM-DD HH:MM for sub-daily frequencies). The dates of every series must match the chosen frequency, with no duplicates and no gaps.", None),
    ("• For monthly/quarterly/yearly data, month-end dates (31.01.2021 etc.) may be entered; the application moves them to the 1st day of the period. For yearly data, writing only the year (2015) is enough.", None),
    ("• Values: must be numbers (use a numeric Excel cell for the decimal separator; the text '1,5' is not accepted). Leave the cell empty for a missing value (or write ? / NA).", None),
    ("• Series names: non-ASCII characters are converted to ASCII (for example ş→s, ı→i), and spaces become underscores. Two series that convert to the same name cause an error.", None),
    ("• Horizon: every series must be longer than the horizon; a length of at least 2×horizon is recommended.", None),
    ("• A completely missing series is not accepted. Missing values at the beginning/end produce a warning; a missing-value policy is chosen before running a model.", None),
    ("• Date gaps: an error by default. If you switch on the option to fill missing dates with missing values (in the converter), the missing dates are added as missing values.", None),
    ("", None),
    ("3) Frequency options and default seasonality (same as Monash/TSForecasting)", "h"),
]


def _example_long() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=36, freq="MS")
    rows = []
    for name, base, trend, region in (("example_A", 100.0, 1.2, "north"), ("example_B", 220.0, 0.6, "south")):
        seas = 12 * np.sin(2 * np.pi * (np.arange(36) % 12) / 12)
        vals = np.round(base + trend * np.arange(36) + seas + rng.normal(0, 2, 36), 1)
        rows += [(name, d.to_pydatetime(), float(v), region) for d, v in zip(dates, vals)]
    return pd.DataFrame(rows, columns=["series_name", "date", "value", "region"])


def build_template(path: str | Path | None = None) -> bytes | Path:
    """Builds the template. If `path` is given it writes the file and returns the path; otherwise it returns bytes (for the download button)."""
    buf = io.BytesIO() if path is None else None
    wb = xlsxwriter.Workbook(buf if buf is not None else str(path), {"in_memory": buf is not None, "default_date_format": "yyyy-mm-dd"})
    f_title = wb.add_format({"bold": True, "font_size": 16})
    f_h = wb.add_format({"bold": True, "font_size": 12, "font_color": "#1F4E79"})
    f_wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    f_head = wb.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1})
    f_date = wb.add_format({"num_format": "yyyy-mm-dd"})
    f_input = wb.add_format({"bg_color": "#FFF2CC", "border": 1})
    f_note = wb.add_format({"italic": True, "font_color": "#7F7F7F", "text_wrap": True, "valign": "top"})

    # ---- Instructions
    ws = wb.add_worksheet("Instructions")
    ws.set_column(0, 0, 120)
    r = 0
    for text, kind in INSTRUCTIONS:
        ws.write(r, 0, text, {"title": f_title, "h": f_h}.get(kind, f_wrap))
        r += 1
    ws.write(r, 0, "frequency  |  date step  |  default seasonality", f_head)
    r += 1
    for fr in FREQUENCIES:
        seas = ", ".join(f"{s:g}" if float(s).is_integer() else f"{s:.2f}" for s in SEASONALITY_MAP[fr])
        ws.write(r, 0, f"{fr}  |  {_STEP_TEXT[fr]}  |  {seas}", f_wrap)
        r += 1

    # ---- Meta
    wm = wb.add_worksheet(SHEET_META)
    wm.set_column(0, 0, 18)
    wm.set_column(1, 1, 24)
    wm.set_column(2, 2, 90)
    for c, t in enumerate(("key", "value", "description")):
        wm.write(0, c, t, f_head)
    meta = [
        ("dataset_name", "example_monthly", "Dataset name (letters/digits/_). Used in the output files."),
        ("frequency", "monthly", "Choose from the drop-down list."),
        ("horizon", 6, "Forecast horizon: how many steps ahead to forecast (positive integer)."),
        ("description", "Example data in the template (2 series, 36 months)", "Optional short description."),
    ]
    for i, (k, v, d) in enumerate(meta, start=1):
        wm.write(i, 0, k)
        wm.write(i, 1, v, f_input)
        wm.write(i, 2, d, f_note)
    wm.data_validation(2, 1, 2, 1, {"validate": "list", "source": list(FREQUENCIES),
                                    "error_title": "frequency", "error_message": "Choose a frequency from the list."})
    wm.data_validation(3, 1, 3, 1, {"validate": "integer", "criteria": ">", "value": 0,
                                    "error_title": "horizon", "error_message": "Enter a positive integer."})

    # ---- Veri_Uzun (filled with an example)
    wl = wb.add_worksheet(SHEET_LONG)
    ex = _example_long()
    wl.set_column(0, 0, 18)
    wl.set_column(1, 1, 14, f_date)
    wl.set_column(2, 2, 12)
    wl.set_column(3, 3, 16)
    for c, t in enumerate(ex.columns):
        wl.write(0, c, t, f_head)
    for i, row in enumerate(ex.itertuples(index=False), start=1):
        wl.write(i, 0, row.series_name)
        wl.write_datetime(i, 1, row.date, f_date)
        wl.write_number(i, 2, row.value)
        wl.write(i, 3, row.region)
    wl.freeze_panes(1, 0)
    # No free text is written in the header row (the converter treats extra columns as attributes); hints are given as cell comments.
    wl.write_comment(0, 0, "Delete the example rows and enter your own data.\nColumns: series_name | date | value | (optional constant attributes, e.g. region).", {"x_scale": 2.2, "y_scale": 1.6})
    wl.write_comment(0, 3, "'region' is an optional attribute column; delete it if you do not need it.", {"x_scale": 2, "y_scale": 1})

    # ---- Veri_Genis (header only)
    ww = wb.add_worksheet(SHEET_WIDE)
    ww.set_column(0, 0, 14, f_date)
    ww.set_column(1, 20, 14)
    ww.write(0, 0, "date", f_head)
    ww.write(0, 1, "series_1", f_head)
    ww.write(0, 2, "series_2", f_head)
    ww.freeze_panes(1, 1)
    ww.write_comment(0, 0, "If you use the wide format, clear Veri_Uzun (leave only its header row).\nReplace the headers B1, C1... with your series names.", {"x_scale": 2.2, "y_scale": 1.6})

    wb.close()
    if buf is not None:
        return buf.getvalue()
    return Path(path)
