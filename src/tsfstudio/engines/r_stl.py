"""R `stats::stl()` (robust = FALSE) in plain Python: a line-by-line port of R's stl.c (the f2c translation of stl.f,
Cleveland et al. 1990), the same routine the C++ engine ports (cpp/src/autoarima.cpp). statsmodels' STL is the same algorithm but refuses R's default
low-pass window for odd periods (it requires low_pass > period, R uses nextodd(period) = period), so the R settings
cannot be passed to it. Arithmetic order follows the Fortran, so results equal R's to rounding.

1-based helpers as in the Fortran: y[i - 1] is y(i).
"""
from __future__ import annotations

import math

import numpy as np


def _est(y, n, length, ideg, xs, nleft, nright, w, userw, rw):
    """stlest: returns (ys, ok)."""
    rng = float(n) - 1.0
    h = max(xs - nleft, nright - xs)
    if length > n:
        h += float((length - n) // 2)
    h9, h1 = 0.999 * h, 0.001 * h
    a = 0.0
    for j in range(nleft, nright + 1):
        w[j - 1] = 0.0
        r = abs(j - xs)
        if r <= h9:
            if r <= h1:
                w[j - 1] = 1.0
            else:
                u = r / h                                   # R_pow_di(1 - R_pow_di(r / h, 3), 3): x * (x * x)
                v = 1.0 - u * (u * u)
                w[j - 1] = v * (v * v)
            if userw:
                w[j - 1] *= rw[j - 1]
            a += w[j - 1]
    if a <= 0.0:
        return 0.0, False
    for j in range(nleft, nright + 1):
        w[j - 1] /= a
    if h > 0.0 and ideg > 0:
        a = 0.0
        for j in range(nleft, nright + 1):
            a += w[j - 1] * j
        b = xs - a
        c = 0.0
        for j in range(nleft, nright + 1):
            d = j - a
            c += w[j - 1] * (d * d)
        if math.sqrt(c) > 0.001 * rng:
            b /= c
            for j in range(nleft, nright + 1):
                w[j - 1] *= (b * (j - a) + 1.0)
    s = 0.0
    for j in range(nleft, nright + 1):
        s += w[j - 1] * y[j - 1]
    return s, True


def _ess(y, n, length, ideg, njump, userw, rw, ys, off, res):
    """stless: smooths y[0:n] into ys[off:off + n]."""
    if n < 2:
        ys[off] = y[0]
        return
    newnj = min(njump, n - 1)
    nleft, nright = 1, n
    if length >= n:
        nleft, nright = 1, n
        for i in range(1, n + 1, newnj):
            v, ok = _est(y, n, length, ideg, float(i), nleft, nright, res, userw, rw)
            ys[off + i - 1] = v if ok else y[i - 1]
    elif newnj == 1:
        nsh = (length + 1) // 2
        nleft, nright = 1, length
        for i in range(1, n + 1):
            if i > nsh and nright != n:
                nleft += 1
                nright += 1
            v, ok = _est(y, n, length, ideg, float(i), nleft, nright, res, userw, rw)
            ys[off + i - 1] = v if ok else y[i - 1]
    else:
        nsh = (length + 1) // 2
        for i in range(1, n + 1, newnj):
            if i < nsh:
                nleft, nright = 1, length
            elif i >= n - nsh + 1:
                nleft, nright = n - length + 1, n
            else:
                nleft, nright = i - nsh + 1, length + i - nsh
            v, ok = _est(y, n, length, ideg, float(i), nleft, nright, res, userw, rw)
            ys[off + i - 1] = v if ok else y[i - 1]
    if newnj != 1:
        for i in range(1, n - newnj + 1, newnj):
            delta = (ys[off + i + newnj - 1] - ys[off + i - 1]) / newnj
            for j in range(i + 1, i + newnj):
                ys[off + j - 1] = ys[off + i - 1] + delta * (j - i)
        k = ((n - 1) // newnj) * newnj + 1
        if k != n:
            v, ok = _est(y, n, length, ideg, float(n), nleft, nright, res, userw, rw)
            ys[off + n - 1] = v if ok else y[n - 1]
            if k != n - 1:
                delta = (ys[off + n - 1] - ys[off + k - 1]) / (n - k)
                for j in range(k + 1, n):
                    ys[off + j - 1] = ys[off + k - 1] + delta * (j - k)


def _ma(x, n, length):
    """stlma: moving average of length `length` (running sum as in the Fortran)."""
    newn = n - length + 1
    flen = float(length)
    v = 0.0
    for i in range(length):
        v += x[i]
    ave = [0.0] * max(newn, 1)
    ave[0] = v / flen
    k, m = length, 0
    for j in range(2, newn + 1):
        k += 1
        m += 1
        v += x[k - 1] - x[m - 1]                               # stl.c: v += x[k] - x[m]
        ave[j - 1] = v / flen
    return ave


def _ss(y, n, np_, ns, isdeg, nsjump, season):
    """stlss (no user weights): cycle-subseries smoothing into season[0 : n + 2 np]."""
    for j in range(1, np_ + 1):
        k = (n - j) // np_ + 1
        work1 = [y[(i - 1) * np_ + j - 1] for i in range(1, k + 1)]
        work2 = [0.0] * (k + 2)
        work4 = [0.0] * max(k, ns)
        _ess(work1, k, ns, isdeg, nsjump, False, None, work2, 1, work4)
        nright = min(ns, k)
        v, ok = _est(work1, k, ns, isdeg, 0.0, 1, nright, work4, False, None)
        work2[0] = v if ok else work2[1]
        nleft = max(1, k - ns + 1)
        v, ok = _est(work1, k, ns, isdeg, k + 1.0, nleft, k, work4, False, None)
        work2[k + 1] = v if ok else work2[k]
        for m in range(1, k + 3):
            season[(m - 1) * np_ + j - 1] = work2[m - 1]


def stl_core(x, period: int, s_win: int, t_window: int, l_window: int, isdeg: int, itdeg: int, ildeg: int,
             nsjump: int, ntjump: int, nljump: int, ni: int) -> tuple[np.ndarray, np.ndarray]:
    x = [float(v) for v in x]
    n = len(x)
    ns, nt, nl = max(3, s_win), max(3, t_window), max(3, l_window)
    ns += ns % 2 == 0
    nt += nt % 2 == 0
    nl += nl % 2 == 0
    np_ = max(2, period)
    nw = n + 2 * np_
    seasonal, trend = [0.0] * n, [0.0] * n
    for _ in range(ni):
        w1 = [x[i] - trend[i] for i in range(n)]
        c = [0.0] * nw
        _ss(w1, n, np_, ns, isdeg, nsjump, c)
        t1 = _ma(c, nw, np_)                                   # stlfts
        t2 = _ma(t1, nw - np_ + 1, np_)
        low = _ma(t2, nw - 2 * np_ + 2, 3)
        lw = [0.0] * n
        _ess(low, n, nl, ildeg, nljump, False, None, lw, 0, [0.0] * max(n, nl))
        seasonal = [c[np_ + i] - lw[i] for i in range(n)]
        d = [x[i] - seasonal[i] for i in range(n)]
        _ess(d, n, nt, itdeg, ntjump, False, None, trend, 0, [0.0] * max(n, nt))
    return np.asarray(seasonal), np.asarray(trend)


def _nextodd(v: float) -> int:
    r = int(round(v))                                          # R round(): half to even, like Python round()
    return r + 1 if r % 2 == 0 else r


def stl(x, period: float, s_window: int) -> tuple[np.ndarray, np.ndarray]:
    """stats::stl(ts(x, frequency = period), s.window) with R's defaults (robust = FALSE): (seasonal, trend)."""
    n = len(x)
    if period < 2 or n <= 2 * period:
        raise ValueError("series is not periodic or has less than two periods")
    win = lambda k: max(3, _nextodd(k))                        # noqa: E731
    tw = win(math.ceil(1.5 * period / (1.0 - 1.5 / s_window)))
    sw, lw = win(s_window), win(_nextodd(period))
    jump = lambda w: max(1, math.ceil(w / 10.0))               # noqa: E731
    return stl_core(x, int(period), sw, tw, lw, 0, 1, 1, jump(sw), jump(tw), jump(lw), 2)
