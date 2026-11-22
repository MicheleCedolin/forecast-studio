"""Monash .tsf yazıcı (ASCII, `\\n` satır sonu; eksik değer `?`)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .reader import TSF_DATE_FORMAT, TSFDataset, TSFFormatError, VALID_ATTRIBUTE_TYPES

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _fmt_values(values: np.ndarray) -> str:
    if np.issubdtype(values.dtype, np.integer):
        return ",".join(map(str, values.tolist()))
    out = []
    for v in values.tolist():
        if v != v:  # NaN
            out.append("?")
        elif v == int(v) and abs(v) < 1e15:
            out.append(str(int(v)))
        else:
            out.append(repr(v))
    return ",".join(out)


def _fmt_attr(value, typ: str) -> str:
    if typ == "date":
        ts = pd.Timestamp(value)
        return ts.strftime(TSF_DATE_FORMAT)
    s = str(int(value)) if typ == "numeric" else str(value)
    if ":" in s or "\n" in s or "\r" in s:
        raise TSFFormatError(f"An attribute value cannot contain ':' or a line break: {s!r}")
    return s


def write_tsf(
    path: str | Path,
    *,
    relation: str,
    attributes: Sequence[tuple[str, str]],
    attr_rows: Iterable[Sequence],
    values: Iterable[np.ndarray],
    frequency: str | None,
    horizon: int | None,
    missing: bool,
    equal_length: bool,
    comments: Sequence[str] = (),
) -> Path:
    """Düşük seviyeli yazıcı. `attr_rows[i]` ve `values[i]` aynı seriye aittir."""
    path = Path(path)
    for name, typ in attributes:
        if not _NAME_RE.match(name):
            raise TSFFormatError(f"An attribute name may only contain letters/digits/underscore: {name!r}")
        if typ not in VALID_ATTRIBUTE_TYPES:
            raise TSFFormatError(f"Invalid attribute type: {typ!r}")
    if " " in relation:
        raise TSFFormatError("The @relation value cannot contain spaces")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        for c in comments:
            f.write((c if c.startswith("#") else "# " + c) + "\n")
        f.write(f"@relation {relation}\n")
        for name, typ in attributes:
            f.write(f"@attribute {name} {typ}\n")
        if frequency is not None:
            f.write(f"@frequency {frequency}\n")
        if horizon is not None:
            f.write(f"@horizon {int(horizon)}\n")
        f.write(f"@missing {'true' if missing else 'false'}\n")
        f.write(f"@equallength {'true' if equal_length else 'false'}\n")
        f.write("@data\n")
        for attrs, vals in zip(attr_rows, values):
            fields = [_fmt_attr(a, t) for a, (_, t) in zip(attrs, attributes)]
            f.write(":".join(fields) + ":" + _fmt_values(np.asarray(vals)) + "\n")
    return path


def write_dataset(path: str | Path, ds: TSFDataset, comments: Sequence[str] | None = None) -> Path:
    """Okunmuş bir TSFDataset'i yeniden yazar (tam geri dönüşüm testi için)."""
    h = ds.header
    rows = ds.attrs.itertuples(index=False, name=None)
    return write_tsf(
        path,
        relation=h.relation or "dataset",
        attributes=h.attributes,
        attr_rows=rows,
        values=ds.values,
        frequency=h.frequency,
        horizon=h.horizon,
        missing=ds.has_missing(),
        equal_length=ds.is_equal_length(),
        comments=comments if comments is not None else h.comments,
    )
