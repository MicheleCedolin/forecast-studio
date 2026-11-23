"""R `loess(y ~ x, span, degree = 1)` for one predictor (gaussian, surface = "interpolate", cell = 0.2) in plain
Python: a line-by-line port of R's loessc.c / loessf.f with LINPACK dqrdc / dqrsl / dsvdc and the reference BLAS
(LAPACK 3.10+ dnrm2 and drotg), the same port as cpp/src/loess.cpp. Used by bagged ETS for non-seasonal series.
Fortran 1-based indexing is kept (element 0 of each list is unused).
"""
from __future__ import annotations

import math

_TSML, _TBIG = math.ldexp(1.0, -511), math.ldexp(1.0, 486)
_SSML, _SBIG = math.ldexp(1.0, 537), math.ldexp(1.0, -538)
_MAXN = 1.7976931348623157e308


def _ddot(n, x, ix, incx, y, iy, incy):
    """x[ix + k*incx] . y[iy + k*incy], k = 0..n-1 (sequential, as the unrolled reference loop)."""
    t = 0.0
    for k in range(n):
        t = t + x[ix + k * incx] * y[iy + k * incy]
    return t


def _daxpy(n, da, x, ix, y, iy):
    if n <= 0 or da == 0.0:
        return
    for k in range(n):
        y[iy + k] = y[iy + k] + da * x[ix + k]


def _dscal(n, da, x, ix):
    for k in range(n):
        x[ix + k] = da * x[ix + k]


def _drot(n, x, ix, y, iy, c, s):
    for k in range(n):
        t = c * x[ix + k] + s * y[iy + k]
        y[iy + k] = c * y[iy + k] - s * x[ix + k]
        x[ix + k] = t


def _dswap(n, x, ix, y, iy):
    for k in range(n):
        x[ix + k], y[iy + k] = y[iy + k], x[ix + k]


def _dnrm2(n, x, ix):
    if n <= 0:
        return 0.0
    asml = amed = abig = 0.0
    notbig = True
    for k in range(n):
        ax = abs(x[ix + k])
        if ax > _TBIG:
            t = ax * _SBIG
            abig = abig + t * t
            notbig = False
        elif ax < _TSML:
            if notbig:
                t = ax * _SSML
                asml = asml + t * t
        else:
            amed = amed + ax * ax
    if abig > 0.0:
        if amed > 0.0 or amed > _MAXN or amed != amed:
            abig = abig + (amed * _SBIG) * _SBIG
        scl, sumsq = 1.0 / _SBIG, abig
    elif asml > 0.0:
        if amed > 0.0 or amed > _MAXN or amed != amed:
            amed = math.sqrt(amed)
            asml = math.sqrt(asml) / _SSML
            ymin, ymax = (amed, asml) if asml > amed else (asml, amed)
            r = ymin / ymax
            scl, sumsq = 1.0, (ymax * ymax) * (1.0 + r * r)
        else:
            scl, sumsq = 1.0 / _SSML, asml
    else:
        scl, sumsq = 1.0, amed
    return scl * math.sqrt(sumsq)


def _drotg(a, b):
    """Returns (a, b, c, s) as the reference DROTG leaves them."""
    safmin, safmax = math.ldexp(1.0, -1022), math.ldexp(1.0, 1023)
    anorm, bnorm = abs(a), abs(b)
    if bnorm == 0.0:
        return a, 0.0, 1.0, 0.0
    if anorm == 0.0:
        return b, 1.0, 0.0, 1.0
    scl = min(safmax, max(safmin, max(anorm, bnorm)))
    sigma = math.copysign(1.0, a) if anorm > bnorm else math.copysign(1.0, b)
    as_, bs = a / scl, b / scl
    r = sigma * (scl * math.sqrt(as_ * as_ + bs * bs))
    c, s = a / r, b / r
    if anorm > bnorm:
        z = s
    elif c != 0.0:
        z = 1.0 / c
    else:
        z = 1.0
    return r, z, c, s


def _dsign(a, b):
    return abs(a) if b >= 0.0 else -abs(a)


class _Mat:
    """Column-major matrix with 1-based (i, j); idx(i, j) is the flat position."""

    def __init__(self, ld, ncol):
        self.ld = ld
        self.a = [0.0] * (ld * (ncol + 1) + 1)

    def idx(self, i, j):
        return (j - 1) * self.ld + i


def _dqrdc0(x: _Mat, n, p, qraux):
    a, I = x.a, x.idx
    for l in range(1, min(n, p) + 1):
        qraux[l] = 0.0
        if l == n:
            continue
        nrmxl = _dnrm2(n - l + 1, a, I(l, l))
        if nrmxl == 0.0:
            continue
        if a[I(l, l)] != 0.0:
            nrmxl = _dsign(nrmxl, a[I(l, l)])
        _dscal(n - l + 1, 1.0 / nrmxl, a, I(l, l))
        a[I(l, l)] = 1.0 + a[I(l, l)]
        for j in range(l + 1, p + 1):
            t = -_ddot(n - l + 1, a, I(l, l), 1, a, I(l, j), 1) / a[I(l, l)]
            _daxpy(n - l + 1, t, a, I(l, l), a, I(l, j))
        qraux[l] = a[I(l, l)]
        a[I(l, l)] = -nrmxl


def _dqrsl_qty(x: _Mat, n, k, qraux, y):
    a, I = x.a, x.idx
    ju = min(k, n - 1)
    for j in range(1, ju + 1):
        if qraux[j] == 0.0:
            continue
        temp = a[I(j, j)]
        a[I(j, j)] = qraux[j]
        t = -_ddot(n - j + 1, a, I(j, j), 1, y, j, 1) / a[I(j, j)]
        _daxpy(n - j + 1, t, a, I(j, j), y, j)
        a[I(j, j)] = temp


def _dsvdc21(x: _Mat, n, p, s, e, v: _Mat, work):
    """dsvdc(job = 21) with u aliasing x, as ehg127 calls it."""
    a, I = x.a, x.idx
    va, V = v.a, v.idx
    maxit = 30
    ncu = min(n, p)
    nct, nrt = min(n - 1, p), max(0, min(p - 2, n))
    lu = max(nct, nrt)
    for l in range(1, lu + 1):
        lp1 = l + 1
        if l <= nct:
            s[l] = _dnrm2(n - l + 1, a, I(l, l))
            if s[l] != 0.0:
                if a[I(l, l)] != 0.0:
                    s[l] = _dsign(s[l], a[I(l, l)])
                _dscal(n - l + 1, 1.0 / s[l], a, I(l, l))
                a[I(l, l)] = 1.0 + a[I(l, l)]
            s[l] = -s[l]
        for j in range(lp1, p + 1):
            if l <= nct and s[l] != 0.0:
                t = -_ddot(n - l + 1, a, I(l, l), 1, a, I(l, j), 1) / a[I(l, l)]
                _daxpy(n - l + 1, t, a, I(l, l), a, I(l, j))
            e[j] = a[I(l, j)]
        # wantu: u(i, l) = x(i, l) (same storage)
        if l <= nrt:
            e[l] = _dnrm2(p - l, e, lp1)
            if e[l] != 0.0:
                if e[lp1] != 0.0:
                    e[l] = _dsign(e[l], e[lp1])
                _dscal(p - l, 1.0 / e[l], e, lp1)
                e[lp1] = 1.0 + e[lp1]
            e[l] = -e[l]
            if not (lp1 > n or e[l] == 0.0):
                for i in range(lp1, n + 1):
                    work[i] = 0.0
                for j in range(lp1, p + 1):
                    _daxpy(n - l, e[j], a, I(lp1, j), work, lp1)
                for j in range(lp1, p + 1):
                    _daxpy(n - l, -e[j] / e[lp1], work, lp1, a, I(lp1, j))
            for i in range(lp1, p + 1):
                va[V(i, l)] = e[i]
    m = min(p, n + 1)
    nctp1, nrtp1 = nct + 1, nrt + 1
    if nct < p:
        s[nctp1] = a[I(nctp1, nctp1)]
    if n < m:
        s[m] = 0.0
    if nrtp1 < m:
        e[nrtp1] = a[I(nrtp1, m)]
    e[m] = 0.0
    for j in range(nctp1, ncu + 1):
        for i in range(1, n + 1):
            a[I(i, j)] = 0.0
        a[I(j, j)] = 1.0
    for ll in range(1, nct + 1):
        l = nct - ll + 1
        if s[l] != 0.0:
            for j in range(l + 1, ncu + 1):
                t = -_ddot(n - l + 1, a, I(l, l), 1, a, I(l, j), 1) / a[I(l, l)]
                _daxpy(n - l + 1, t, a, I(l, l), a, I(l, j))
            _dscal(n - l + 1, -1.0, a, I(l, l))
            a[I(l, l)] = 1.0 + a[I(l, l)]
            for i in range(1, l):
                a[I(i, l)] = 0.0
        else:
            for i in range(1, n + 1):
                a[I(i, l)] = 0.0
            a[I(l, l)] = 1.0
    for ll in range(1, p + 1):
        l = p - ll + 1
        lp1 = l + 1
        if l <= nrt and e[l] != 0.0:
            for j in range(lp1, p + 1):
                t = -_ddot(p - l, va, V(lp1, l), 1, va, V(lp1, j), 1) / va[V(lp1, l)]
                _daxpy(p - l, t, va, V(lp1, l), va, V(lp1, j))
        for i in range(1, p + 1):
            va[V(i, l)] = 0.0
        va[V(l, l)] = 1.0
    mm = m
    it = 0
    info = 0
    while True:
        if m == 0:
            break
        if it >= maxit:
            info = m
            break
        l = 0
        for ll in range(1, m + 1):
            l = m - ll
            if l == 0:
                break
            test = abs(s[l]) + abs(s[l + 1])
            ztest = test + abs(e[l])
            acc = abs(test - ztest) / (1.0e-100 + test)
            if not (acc > 1.e-15):
                e[l] = 0.0
                break
        if l == m - 1:
            kase = 4
        else:
            lp1, mp1 = l + 1, m + 1
            ls = l
            for lls in range(lp1, mp1 + 1):
                ls = m - lls + lp1
                if ls == l:
                    break
                test = 0.0
                if ls != m:
                    test = test + abs(e[ls])
                if ls != l + 1:
                    test = test + abs(e[ls - 1])
                ztest = test + abs(s[ls])
                acc = abs(test - ztest) / (1.0e-100 + test)
                if not (acc > 1.e-15):
                    s[ls] = 0.0
                    break
            if ls == l:
                kase = 3
            elif ls == m:
                kase = 1
            else:
                kase = 2
                l = ls
        l = l + 1
        if kase == 1:
            mm1 = m - 1
            f = e[m - 1]
            e[m - 1] = 0.0
            for kk in range(l, mm1 + 1):
                k = mm1 - kk + l
                t1, f, cs, sn = _drotg(s[k], f)
                s[k] = t1
                if k != l:
                    f = -sn * e[k - 1]
                    e[k - 1] = cs * e[k - 1]
                _drot(p, va, V(1, k), va, V(1, m), cs, sn)
        elif kase == 2:
            f = e[l - 1]
            e[l - 1] = 0.0
            for k in range(l, m + 1):
                t1, f, cs, sn = _drotg(s[k], f)
                s[k] = t1
                f = -sn * e[k]
                e[k] = cs * e[k]
                _drot(n, a, I(1, k), a, I(1, l - 1), cs, sn)
        elif kase == 3:
            scale = max(abs(s[m]), abs(s[m - 1]), abs(e[m - 1]), abs(s[l]), abs(e[l]))
            sm, smm1, emm1 = s[m] / scale, s[m - 1] / scale, e[m - 1] / scale
            sl, el = s[l] / scale, e[l] / scale
            b = ((smm1 + sm) * (smm1 - sm) + emm1 * emm1) / 2.0
            se = sm * emm1
            c = se * se
            shift = 0.0
            if not (b == 0.0 and c == 0.0):
                shift = math.sqrt(b * b + c)
                if b < 0.0:
                    shift = -shift
                shift = c / (b + shift)
            f = (sl + sm) * (sl - sm) + shift
            g = sl * el
            for k in range(l, m):
                f, g, cs, sn = _drotg(f, g)
                if k != l:
                    e[k - 1] = f
                f = cs * s[k] + sn * e[k]
                e[k] = cs * e[k] - sn * s[k]
                g = sn * s[k + 1]
                s[k + 1] = cs * s[k + 1]
                _drot(p, va, V(1, k), va, V(1, k + 1), cs, sn)
                f, g, cs, sn = _drotg(f, g)
                s[k] = f
                f = cs * e[k] + sn * s[k + 1]
                s[k + 1] = -sn * e[k] + cs * s[k + 1]
                g = sn * e[k + 1]
                e[k + 1] = cs * e[k + 1]
                if k < n:
                    _drot(n, a, I(1, k), a, I(1, k + 1), cs, sn)
            e[m - 1] = f
            it += 1
        else:
            if s[l] < 0.0:
                s[l] = -s[l]
                _dscal(p, -1.0, va, V(1, l))
            while l != mm and not (s[l] >= s[l + 1]):
                s[l], s[l + 1] = s[l + 1], s[l]
                if l < p:
                    _dswap(p, va, V(1, l), va, V(1, l + 1))
                if l < n:
                    _dswap(n, a, I(1, l), a, I(1, l + 1))
                l += 1
            it = 0
            m -= 1
    return info


def _ehg106(il, ir, k, p, pi):
    l, r = il, ir
    while l < r:
        t = p[pi[k]]
        i, j = l, r
        pi[l], pi[k] = pi[k], pi[l]
        if t < p[pi[r]]:
            pi[l], pi[r] = pi[r], pi[l]
        while i < j:
            pi[i], pi[j] = pi[j], pi[i]
            i += 1
            j -= 1
            while p[pi[i]] < t:
                i += 1
            while t < p[pi[j]]:
                j -= 1
        if p[pi[l]] == t:
            pi[l], pi[j] = pi[j], pi[l]
        else:
            j += 1
            pi[r], pi[j] = pi[j], pi[r]
        if j <= k:
            l = j + 1
        if k <= j:
            r = j - 1


def loess_fit_1d(xin, yin, span: float, cell_in: float = 0.2) -> list[float]:
    n = len(xin)
    if n < 2 or len(yin) != n:
        raise ValueError("loess: need at least two points")
    nvmax = max(200, n)
    ncmax = nvmax
    nf = min(n, int(math.floor(n * span + 1e-5)))
    if nf <= 0:
        raise ValueError("span is too small")
    k = 2
    if k > nf - 1:
        raise ValueError("loess: span too small for degree 1")
    cell = span * cell_in
    fc = int(math.floor(n * cell))
    fd = 0.0
    x = [0.0] + [float(v) for v in xin]
    yy = [0.0] + [float(v) for v in yin]
    v = [0.0] * (nvmax + 2)
    a = [0] * (ncmax + 2)
    c1 = [0] * (ncmax + 2)
    c2 = [0] * (ncmax + 2)
    lo = [0] * (ncmax + 2)
    hi = [0] * (ncmax + 2)
    xi = [0.0] * (ncmax + 2)
    vhit = [0] * (nvmax + 2)
    alpha, beta = _MAXN, -_MAXN
    for i in range(1, n + 1):
        alpha = min(alpha, x[i])
        beta = max(beta, x[i])
    mu = 0.005 * max(beta - alpha, 1.e-10 * max(abs(alpha), abs(beta)) + 1.e-30)
    v[1], v[2] = alpha - mu, beta + mu
    nv, nc = 2, 1
    c1[1], c2[1] = 1, 2
    pi = list(range(n + 1))
    p, l, u = 1, 1, n
    lo[p], hi[p] = l, u
    while p <= nc:
        diag = v[c2[p]] - v[c1[p]]
        diam = 0.0
        diam = diam + diag * diag
        diam = math.sqrt(diam)
        if (u - l) + 1 <= fc or diam <= fd:
            leaf = True
        else:
            leaf = (ncmax < nc + 2) or (nvmax < nv + 2.0 / 2.0)
        m = 0
        if not leaf:
            m = int((l + u) / 2.0)
            _ehg106(l, u, m, x, pi)
            offset = 0
            while not ((m + offset) >= u or (m + offset) < l):
                if offset < 0:
                    lower, check = l, m + offset
                    upper = check
                else:
                    lower = check = m + offset + 1
                    upper = u
                _ehg106(lower, upper, check, x, pi)
                if x[pi[m + offset]] == x[pi[m + offset + 1]]:
                    offset = -offset
                    if offset >= 0:
                        offset += 1
                else:
                    m = m + offset
                    break
            leaf = (v[c1[p]] == x[pi[m]]) or (v[c2[p]] == x[pi[m]])
        if leaf:
            a[p] = 0
        else:
            a[p] = 1
            xi[p] = x[pi[m]]
            nc += 1
            lo[p] = nc
            lo[nc], hi[nc] = l, m
            nc += 1
            hi[p] = nc
            lo[nc], hi[nc] = m + 1, u
            h = nv + 1
            v[h] = xi[p]
            match, mv = False, 1
            while not match and mv <= nv:
                match = v[mv] == v[h]
                mv += 1
            mv -= 1
            if match:
                h -= 1
            else:
                mv = h
                if vhit[1] >= 0:
                    vhit[mv] = p
            c1[lo[p]], c2[lo[p]] = c1[p], mv
            c1[hi[p]], c2[hi[p]] = mv, c2[p]
            nv = h
            if nv > nvmax:
                raise ValueError("loess: k-d tree vertex overflow")
        p += 1
        l, u = lo[p], hi[p]
    # vertex fits (ehg139 -> ehg127), psi carried over between vertices
    vval0, vval1 = [0.0] * (nv + 1), [0.0] * (nv + 1)
    psi = list(range(n + 1))
    dist = [0.0] * (n + 1)
    machep = 2.220446049250313e-16
    for lv in range(1, nv + 1):
        q = v[lv]
        for i in range(1, n + 1):
            dd = x[i] - q
            dist[i] = 0.0 + dd * dd
        _ehg106(1, n, nf, dist, psi)
        rho = dist[psi[nf]] * max(1.0, span)
        if rho <= 0:
            raise ValueError("loess: span too small (zero neighbourhood radius)")
        w = [0.0] * (nf + 1)
        for i in range(1, nf + 1):
            w[i] = math.sqrt(dist[psi[i]] / rho)
        for i in range(1, nf + 1):
            w3 = w[i] * w[i] * w[i]
            t = 1 - w3
            w[i] = math.sqrt(1.0 * (t * t * t))
        b = _Mat(nf, k)
        for i in range(1, nf + 1):
            b.a[b.idx(i, 1)] = w[i]
            b.a[b.idx(i, 2)] = w[i] * (x[psi[i]] - q)
        eta = [0.0] + [w[i] * yy[psi[i]] for i in range(1, nf + 1)]
        colnor = [0.0, 1.0, 1.0]
        for j in range(1, k + 1):
            scal = 0.0
            for i in range(1, nf + 1):
                bij = b.a[b.idx(i, j)]
                scal = scal + bij * bij
            scal = math.sqrt(scal)
            if 0 < scal:
                for i in range(1, nf + 1):
                    b.a[b.idx(i, j)] = b.a[b.idx(i, j)] / scal
                colnor[j] = scal
            else:
                colnor[j] = 1.0
        qraux = [0.0] * 16
        _dqrdc0(b, nf, k, qraux)
        _dqrsl_qty(b, nf, k, qraux, eta)
        uu, ee = _Mat(15, 15), _Mat(15, 15)
        for i in range(1, k + 1):
            for j in range(i, k + 1):
                uu.a[uu.idx(i, j)] = b.a[b.idx(i, j)]
        sigma, g, work = [0.0] * 17, [0.0] * 17, [0.0] * 17
        if _dsvdc21(uu, k, k, sigma, g, ee, work) != 0:
            raise ValueError("loess: dsvdc failed")
        tol = sigma[1] * (100 * machep)
        for j in range(1, k + 1):
            for i in range(1, k + 1):
                ee.a[ee.idx(j, i)] = ee.a[ee.idx(j, i)] / colnor[j]
        dgamma = [0.0] * (k + 1)
        for j in range(1, k + 1):
            dgamma[j] = (_ddot(k, uu.a, uu.idx(1, j), 1, eta, 1, 1) / sigma[j]) if tol < sigma[j] else 0.0
        vval0[lv] = _ddot(k, ee.a, ee.idx(1, 1), 15, dgamma, 1, 1)
        vval1[lv] = _ddot(k, ee.a, ee.idx(2, 1), 15, dgamma, 1, 1)
    out = []
    for i in range(1, n + 1):
        z = x[i]
        j = 1
        while a[j] != 0:
            j = lo[j] if z <= xi[j] else hi[j]
        ll, ur = c1[j], c2[j]
        h = (z - v[ll]) / (v[ur] - v[ll])
        omh = 1 - h
        phi0 = (omh * omh) * (1 + 2 * h)
        phi1 = (h * h) * (3 - 2 * h)
        psi0 = h * (omh * omh)
        psi1 = (h * h) * (h - 1)
        out.append(phi0 * vval0[ll] + phi1 * vval0[ur] + (psi0 * vval1[ll] + psi1 * vval1[ur]) * (v[ur] - v[ll]))
    return out
