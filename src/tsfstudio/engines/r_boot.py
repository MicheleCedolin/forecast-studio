"""R pieces for bagged ETS, reproduced exactly in plain Python (the C++ engine has the same ports, cpp/src/rstats.hpp
and cpp/src/bagged.cpp):

- R's default random number generator: Mersenne-Twister seeded as set.seed() does, unif_rand() and sample()'s
  "Rejection" index (R >= 3.6), so sample(1:k, 1) = 1 + unif_index(k);
- mean() / colMeans() / sd() with R's long double (x87, 64-bit significand) accumulation, emulated exactly;
- optimize() (Brent_fmin), BoxCox(), InvBoxCox(), forecast::BoxCox.lambda("guerrero");
- forecast::bld.mbb.bootstrap(): Box-Cox, stl(x, "periodic"), moving-block bootstrap of the remainder.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

_M32 = 0xFFFFFFFF


class RRng:
    N, M = 624, 397

    def __init__(self, seed: int):
        seed &= _M32
        for _ in range(50):
            seed = (69069 * seed + 1) & _M32
        mt = []
        for j in range(625):
            seed = (69069 * seed + 1) & _M32
            if j:
                mt.append(seed)
        self.mt = mt
        self.mti = self.N

    def _genrand(self) -> float:
        mt, N, M = self.mt, self.N, self.M
        if self.mti >= N:
            for kk in range(N - M):
                y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7FFFFFFF)
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            for kk in range(N - M, N - 1):
                y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7FFFFFFF)
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7FFFFFFF)
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            self.mti = 0
        y = mt[self.mti]
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return float(y & _M32) * 2.3283064365386963e-10

    def unif_rand(self) -> float:
        x = self._genrand()
        i2_32m1 = 2.328306437080797e-10
        if x <= 0.0:
            return 0.5 * i2_32m1
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * i2_32m1
        return x

    def _rbits(self, bits: int) -> float:
        v = 0
        for _ in range(0, bits + 1, 16):
            v = 65536 * v + int(math.floor(self.unif_rand() * 65536))
        return float(v & ((1 << bits) - 1))

    def unif_index(self, dn: float) -> float:
        if dn <= 0:
            return 0.0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample1(self, k: int) -> int:
        """sample(1:k, 1)"""
        return 1 + int(self.unif_index(float(k)))


# ---------------------------------------------------------------- long double (x87) arithmetic, exact emulation
def _ld(f: Fraction) -> Fraction:
    """Round an exact value to the nearest x87 long double (64-bit significand, ties to even)."""
    if f == 0:
        return Fraction(0)
    a = abs(f)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    scale = Fraction(2) ** (63 - e)
    m = a * scale
    q, r = divmod(m.numerator, m.denominator)
    if 2 * r > m.denominator or (2 * r == m.denominator and q % 2 == 1):
        q += 1
    out = Fraction(q) / scale
    return out if f > 0 else -out


def _ld_sum(v) -> Fraction:
    s = Fraction(0)
    for x in v:
        s = _ld(s + Fraction(float(x)))
    return s


def r_mean(v) -> float:
    """mean(): long double sum / n, then s += sum(x - s) / n (in long double), cast to double."""
    n = len(v)
    s = _ld(_ld_sum(v) / n)
    if math.isfinite(float(s)):
        t = Fraction(0)
        for x in v:
            t = _ld(t + _ld(Fraction(float(x)) - s))
        s = _ld(s + _ld(t / n))
    return float(s)


def r_col_mean(v) -> float:
    """colMeans(): one long double pass."""
    return float(_ld(_ld_sum(v) / len(v)))


def r_sd(v) -> float:
    n = len(v)
    if n < 2:
        return float("nan")
    m = Fraction(r_mean(v))                                 # cov.c: xm[] double, xxm long double
    s = Fraction(0)
    for x in v:
        d = _ld(Fraction(float(x)) - m)                     # (xx[k] - xxm) * (xx[k] - xxm) in long double
        s = _ld(s + _ld(d * d))
    return math.sqrt(float(_ld(s / (n - 1))))


def r_row_mean(vals) -> float:
    """rowMeans(): long double sum / n."""
    return float(_ld(_ld_sum(vals) / len(vals)))


# ---------------------------------------------------------------- optimize(), Box-Cox
def brent_fmin(ax: float, bx: float, f, tol: float = 1.220703125e-04) -> float:
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(2.220446049250313e-16)
    a, b = ax, bx
    v = a + c * (b - a)
    w = x = v
    d = e = 0.0
    fx = f(x)
    fv = fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def box_cox(x: float, lam: float) -> float:
    if lam < 0 and x < 0:
        return float("nan")
    if lam == 0:
        return math.log(x)
    return (_sign(x) * math.pow(abs(x), lam) - 1) / lam


def inv_box_cox(x: float, lam: float) -> float:
    if lam < 0 and x > -1 / lam:
        return float("nan")
    if lam == 0:
        return math.exp(x)
    xx = x * lam + 1
    return _sign(xx) * math.pow(abs(xx), 1 / lam)


def box_cox_lambda(x, lower: float, upper: float, freq: float) -> float:
    """forecast::BoxCox.lambda(x, "guerrero", lower, upper) for a ts of frequency freq."""
    x = [float(v) for v in x]
    if any(v <= 0 for v in x):
        lower = max(lower, 0.0)
    n = len(x)
    if n <= 2 * freq:
        return 1.0
    period = int(round(max(2.0, freq)))
    nyr = n // period
    nobst = nyr * period
    cols = [x[(n - nobst) + j * period:(n - nobst) + (j + 1) * period] for j in range(nyr)]
    mu = [r_col_mean(c) for c in cols]
    sdv = [r_sd(c) for c in cols]

    def cv(lam: float) -> float:
        rat = [sdv[j] / math.pow(mu[j], 1 - lam) for j in range(nyr)]
        return r_sd(rat) / r_mean(rat)

    return brent_fmin(lower, upper, cv)


# ---------------------------------------------------------------- bld.mbb.bootstrap
def _nextodd(v: float) -> int:
    r = int(round(v))
    return r + 1 if r % 2 == 0 else r


def stl_periodic(x, period: int):
    """stl(ts(x, frequency = period), "periodic")$time.series: (seasonal, trend, remainder)."""
    from .r_stl import stl_core
    n = len(x)
    if period < 2 or n <= 2 * period:
        raise ValueError("series is not periodic or has less than two periods")
    s_window = 10.0 * n + 1
    win = lambda k: max(3, _nextodd(k))                        # noqa: E731
    tw = win(math.ceil(1.5 * period / (1.0 - 1.5 / s_window)))
    sw, lw = win(s_window), win(_nextodd(period))
    jump = lambda w: max(1, math.ceil(w / 10.0))               # noqa: E731
    seas, trend = stl_core(x, period, sw, tw, lw, 0, 1, 1, jump(sw), jump(tw), jump(lw), 2)
    cyc = [r_mean(list(seas[c::period])) for c in range(period)]
    seasonal = np.array([cyc[t % period] for t in range(n)])
    remainder = np.array([float(x[t]) - seasonal[t] - trend[t] for t in range(n)])
    return seasonal, trend, remainder


def _mbb(x: np.ndarray, w: int, rng: RRng) -> np.ndarray:
    n = len(x)
    nb = n // w + 2
    bx = np.zeros(nb * w)
    for i in range(nb):
        c = rng.sample1(n - w + 1)
        bx[i * w:(i + 1) * w] = x[c - 1:c - 1 + w]
    start = rng.sample1(w)
    return bx[start - 1:start - 1 + n]


def bld_mbb_bootstrap(x, freq_in: int, num: int, seed: int) -> list[np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n <= 1:
        return [x.copy() for _ in range(num)]
    rng = RRng(seed)
    freq = 1 if n <= 2 * freq_in else freq_in
    block = 2 * freq if freq > 1 else min(8, n // 2)
    xs = [x.copy()]
    if num <= 1:
        return xs
    lam = box_cox_lambda(x, 0.0, 1.0, freq_in) if float(x.min()) > 1e-6 else 1.0
    xbc = np.array([box_cox(float(v), lam) for v in x])
    if freq > 1:
        seasonal, trend, remainder = stl_periodic(xbc, freq)
    else:                                                      # loess(x.bc ~ seq_along(x), span = 6 / n, degree = 1)
        from .r_loess import loess_fit_1d
        trend = np.array(loess_fit_1d([float(i + 1) for i in range(n)], list(xbc), 6.0 / n))
        seasonal = np.zeros(n)
        remainder = xbc - trend
    base = trend + seasonal
    for _ in range(1, num):
        r = _mbb(remainder, block, rng)
        xs.append(np.array([inv_box_cox(float(base[t] + r[t]), lam) for t in range(n)]))
    return xs
