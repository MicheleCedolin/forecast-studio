"""Monash .tsf okuyucu.

Monash TSForecasting `convert_tsf_to_dataframe` ile aynı biçim kurallarını uygular, ayrıca:
  * Python >= 3.12 uyumlu (distutils.util.strtobool yerel olarak yeniden yazıldı),
  * satır numaralı, anlaşılır hata mesajları,
  * yalnızca başlık okuma (`read_tsf_header`) ve seri sayısı sınırlama (`max_series`),
  * değer tipi seçilebilir (varsayılan float64; R ile aynı duyarlılık).

Biçim: `@relation`, `@attribute <ad> <numeric|string|date>`, `@frequency`, `@horizon`,
`@missing`, `@equallength`, `@data`; veri satırı `<öznitelikler ':' ile>:<v1,v2,...>`;
eksik değer `?`; `date` öznitelikleri `%Y-%m-%d %H-%M-%S`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

TSF_DATE_FORMAT = "%Y-%m-%d %H-%M-%S"
VALID_ATTRIBUTE_TYPES = ("numeric", "string", "date")


class TSFFormatError(ValueError):
    """Geçersiz .tsf içeriği (satır numarası mesajda belirtilir)."""


def strtobool(value: str) -> bool:
    v = value.strip().lower()
    if v in ("y", "yes", "t", "true", "on", "1"):
        return True
    if v in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"Expected a boolean value but found {value!r}")


@dataclass
class TSFHeader:
    relation: str | None = None
    attributes: list[tuple[str, str]] = field(default_factory=list)  # (ad, tip)
    frequency: str | None = None
    horizon: int | None = None
    missing: bool | None = None
    equal_length: bool | None = None
    comments: list[str] = field(default_factory=list)

    @property
    def attribute_names(self) -> list[str]:
        return [a for a, _ in self.attributes]


@dataclass
class TSFDataset:
    header: TSFHeader
    attrs: pd.DataFrame            # seri başına öznitelikler (satır sırası = seri sırası)
    values: list[np.ndarray]       # her seri için 1-B dizi (eksik = NaN)
    source: str | None = None

    # ---- kolay erişim
    @property
    def frequency(self) -> str | None:
        return self.header.frequency

    @property
    def horizon(self) -> int | None:
        return self.header.horizon

    @property
    def n_series(self) -> int:
        return len(self.values)

    @property
    def lengths(self) -> np.ndarray:
        return np.fromiter((len(v) for v in self.values), dtype=np.int64, count=len(self.values))

    @property
    def series_names(self) -> list[str]:
        if "series_name" in self.attrs.columns:
            return self.attrs["series_name"].astype(str).tolist()
        return [f"T{i + 1}" for i in range(self.n_series)]

    def has_missing(self) -> bool:
        return any(bool(np.isnan(v).any()) for v in self.values)

    def is_equal_length(self) -> bool:
        L = self.lengths
        return bool(L.size) and bool((L == L[0]).all())

    def consistency_warnings(self) -> list[str]:
        """Do the @missing/@equallength declarations in the header match the real state of the data?"""
        out: list[str] = []
        if self.header.missing is not None and self.header.missing != self.has_missing():
            out.append(f"@missing={self.header.missing} is declared but the data {'contain' if self.has_missing() else 'do not contain'} missing values.")
        if self.header.equal_length is not None and self.header.equal_length != self.is_equal_length():
            out.append(f"@equallength={self.header.equal_length} is declared but the series have {'equal' if self.is_equal_length() else 'different'} lengths.")
        return out


def _open(path: Path, encoding: str):
    return open(path, "r", encoding=encoding, newline=None)


def _parse_meta_line(line: str, header: TSFHeader, lineno: int) -> None:
    parts = line.split()
    key = parts[0].lower()
    if key == "@attribute":
        if len(parts) != 3:
            raise TSFFormatError(f"Line {lineno}: @attribute must have the form 'name type': {line!r}")
        if parts[2] not in VALID_ATTRIBUTE_TYPES:
            raise TSFFormatError(f"Line {lineno}: invalid attribute type {parts[2]!r} (numeric|string|date)")
        header.attributes.append((parts[1], parts[2]))
        return
    if len(parts) < 2:
        raise TSFFormatError(f"Line {lineno}: header value is missing: {line!r}")
    value = line.split(None, 1)[1].strip()
    if key == "@relation":
        header.relation = value
    elif key == "@frequency":
        header.frequency = value
    elif key == "@horizon":
        try:
            header.horizon = int(value)
        except ValueError as exc:
            raise TSFFormatError(f"Line {lineno}: @horizon must be an integer: {value!r}") from exc
    elif key == "@missing":
        header.missing = strtobool(value)
    elif key == "@equallength":
        header.equal_length = strtobool(value)
    # bilinmeyen @etiketler yok sayılır (ileriye dönük uyumluluk)


def _iter_lines(path: Path, encoding: str) -> Iterator[tuple[int, str]]:
    with _open(path, encoding) as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if line:
                yield lineno, line


def read_tsf_header(path: str | Path, encoding: str = "cp1252") -> TSFHeader:
    """Yalnızca üstbilgiyi okur (büyük dosyalarda hızlı)."""
    path = Path(path)
    try:
        return _read_header(path, encoding)
    except UnicodeDecodeError:
        # same fallback as read_tsf, for bytes that cp1252 does not define
        return _read_header(path, "utf-8-sig")


def _read_header(path: Path, encoding: str) -> TSFHeader:
    header = TSFHeader()
    found_data = False
    for lineno, line in _iter_lines(path, encoding):
        if line.startswith("@"):
            if line.lower().startswith("@data"):
                found_data = True
                break
            _parse_meta_line(line, header, lineno)
        elif line.startswith("#"):
            header.comments.append(line)
        else:
            raise TSFFormatError(f"Line {lineno}: a data line was found before the @data tag")
    if not found_data:
        raise TSFFormatError("The @data tag was not found")
    if not header.attributes:
        raise TSFFormatError("There is no attribute section (@attribute); it must come before the data")
    return header


def _convert_attr(value: str, typ: str, name: str, lineno: int):
    try:
        if typ == "numeric":
            return int(value)
        if typ == "string":
            return str(value)
        return datetime.strptime(value, TSF_DATE_FORMAT)
    except ValueError as exc:
        raise TSFFormatError(f"Line {lineno}: invalid value {value!r} for the attribute '{name}' ({typ})") from exc


def read_tsf(
    path: str | Path,
    encoding: str = "cp1252",
    dtype=np.float64,
    max_series: int | None = None,
) -> TSFDataset:
    """Bir .tsf dosyasını okur.

    `max_series` verilirse ilk N seri okunur (önizleme/süre tahmini için).
    Bir seride tüm değerler eksikse Monash yükleyicisi gibi hata verilir.
    """
    path = Path(path)
    try:
        return _read(path, encoding, dtype, max_series)
    except UnicodeDecodeError:
        # cp1252'de tanımsız baytlar için güvenli geri dönüş
        return _read(path, "utf-8-sig", dtype, max_series)


def _read(path: Path, encoding: str, dtype, max_series: int | None) -> TSFDataset:
    header = TSFHeader()
    in_data = False
    cols: list[list] = []
    series: list[np.ndarray] = []
    n_attr = 0
    for lineno, line in _iter_lines(path, encoding):
        if not in_data:
            if line.startswith("@"):
                if line.lower().startswith("@data"):
                    if not header.attributes:
                        raise TSFFormatError("There is no attribute section (@attribute); it must come before the data")
                    in_data = True
                    n_attr = len(header.attributes)
                    cols = [[] for _ in range(n_attr)]
                else:
                    _parse_meta_line(line, header, lineno)
            elif line.startswith("#"):
                header.comments.append(line)
            else:
                raise TSFFormatError(f"Line {lineno}: a data line was found before the @data tag")
            continue
        if line.startswith("#"):
            continue
        if max_series is not None and len(series) >= max_series:
            break
        parts = line.split(":")
        if len(parts) != n_attr + 1:
            raise TSFFormatError(
                f"Line {lineno}: expected {n_attr} attributes + values, found {len(parts)} fields separated by ':'"
            )
        raw = parts[-1].replace("?", "nan")
        try:
            arr = np.array(raw.split(","), dtype=dtype)
        except ValueError as exc:
            raise TSFFormatError(f"Line {lineno}: non-numeric series value ({exc})") from exc
        if arr.size == 0:
            raise TSFFormatError(f"Line {lineno}: empty series")
        if np.isnan(arr).all():
            raise TSFFormatError(f"Line {lineno}: all values in the series are missing")
        for i, (name, typ) in enumerate(header.attributes):
            cols[i].append(_convert_attr(parts[i], typ, name, lineno))
        series.append(arr)
    if not in_data:
        raise TSFFormatError("The @data tag was not found")
    if not series:
        raise TSFFormatError("The file contains no series")
    attrs = pd.DataFrame({name: cols[i] for i, (name, _) in enumerate(header.attributes)})
    return TSFDataset(header=header, attrs=attrs, values=series, source=str(path))
