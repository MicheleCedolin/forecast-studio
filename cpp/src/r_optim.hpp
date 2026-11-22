// R's optim(method = "Nelder-Mead") minimiser, shared by the R ports (ets, tbats).
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "util.hpp"

namespace tsf {

// ------------------------------------------------------------------------------------------- R optim Nelder-Mead
// Port of nmmin() (R src/appl/optim.c) with ets's call: abstol = -Inf, intol = sqrt(eps), alpha 1, beta 0.5, gamma 2.
template <class F>
inline Vec nmmin(F& fminfn, Vec Bvec, double& Fmin, int maxit, int& fail) {
    const double big = 1.0e+35, abstol = -INF, intol = std::sqrt(2.220446049250313e-16);
    const double alpha = 1.0, bet = 0.5, gamm = 2.0;
    const int n = static_cast<int>(Bvec.size());
    fail = 0;
    double f = fminfn(Bvec);
    if (!std::isfinite(f)) throw std::runtime_error("function cannot be evaluated at initial parameters");
    int funcount = 1;
    const double convtol = intol * (std::fabs(f) + intol);
    const int n1 = n + 1, C = n + 2;
    std::vector<Vec> P(n1, Vec(n1 + 1));                  // P[i][j]: i = 0..n (row n = function value), j = 0..n+1
    P[n1 - 1][0] = f;
    for (int i = 0; i < n; ++i) P[i][0] = Bvec[i];
    int L = 1;
    double size = 0.0, step = 0.0;
    for (int i = 0; i < n; ++i) if (0.1 * std::fabs(Bvec[i]) > step) step = 0.1 * std::fabs(Bvec[i]);
    if (step == 0.0) step = 0.1;
    for (int j = 2; j <= n1; ++j) {
        for (int i = 0; i < n; ++i) P[i][j - 1] = Bvec[i];
        double trystep = step;
        while (P[j - 2][j - 1] == Bvec[j - 2]) {
            P[j - 2][j - 1] = Bvec[j - 2] + trystep;
            trystep *= 10;
        }
        size += trystep;
    }
    double oldsize = size;
    bool calcvert = true;
    int H = 1;
    double VH, VL, VR;
    do {
        if (calcvert) {
            for (int j = 0; j < n1; ++j) {
                if (j + 1 != L) {
                    for (int i = 0; i < n; ++i) Bvec[i] = P[i][j];
                    f = fminfn(Bvec);
                    if (!std::isfinite(f)) f = big;
                    ++funcount;
                    P[n1 - 1][j] = f;
                }
            }
            calcvert = false;
        }
        VL = P[n1 - 1][L - 1];
        VH = VL;
        H = L;
        for (int j = 1; j <= n1; ++j) {
            if (j != L) {
                f = P[n1 - 1][j - 1];
                if (f < VL) { L = j; VL = f; }
                if (f > VH) { H = j; VH = f; }
            }
        }
        if (VH <= VL + convtol || VL <= abstol) break;
        for (int i = 0; i < n; ++i) {
            double temp = -P[i][H - 1];
            for (int j = 0; j < n1; ++j) temp += P[i][j];
            P[i][C - 1] = temp / n;
        }
        for (int i = 0; i < n; ++i) Bvec[i] = (1.0 + alpha) * P[i][C - 1] - alpha * P[i][H - 1];
        f = fminfn(Bvec);
        if (!std::isfinite(f)) f = big;
        ++funcount;
        VR = f;
        if (VR < VL) {
            P[n1 - 1][C - 1] = f;
            for (int i = 0; i < n; ++i) {
                f = gamm * Bvec[i] + (1 - gamm) * P[i][C - 1];
                P[i][C - 1] = Bvec[i];
                Bvec[i] = f;
            }
            f = fminfn(Bvec);
            if (!std::isfinite(f)) f = big;
            ++funcount;
            if (f < VR) {
                for (int i = 0; i < n; ++i) P[i][H - 1] = Bvec[i];
                P[n1 - 1][H - 1] = f;
            } else {
                for (int i = 0; i < n; ++i) P[i][H - 1] = P[i][C - 1];
                P[n1 - 1][H - 1] = VR;
            }
        } else {
            if (VR < VH) {
                for (int i = 0; i < n; ++i) P[i][H - 1] = Bvec[i];
                P[n1 - 1][H - 1] = VR;
            }
            for (int i = 0; i < n; ++i) Bvec[i] = (1 - bet) * P[i][H - 1] + bet * P[i][C - 1];
            f = fminfn(Bvec);
            if (!std::isfinite(f)) f = big;
            ++funcount;
            if (f < P[n1 - 1][H - 1]) {
                for (int i = 0; i < n; ++i) P[i][H - 1] = Bvec[i];
                P[n1 - 1][H - 1] = f;
            } else if (VR >= VH) {
                calcvert = true;
                size = 0.0;
                for (int j = 0; j < n1; ++j) {
                    if (j + 1 != L) {
                        for (int i = 0; i < n; ++i) {
                            P[i][j] = bet * (P[i][j] - P[i][L - 1]) + P[i][L - 1];
                            size += std::fabs(P[i][j] - P[i][L - 1]);
                        }
                    }
                }
                if (size < oldsize) {
                    oldsize = size;
                } else {
                    fail = 10;
                    break;
                }
            }
        }
    } while (funcount <= maxit);
    Fmin = P[n1 - 1][L - 1];
    Vec X(n);
    for (int i = 0; i < n; ++i) X[i] = P[i][L - 1];
    if (funcount > maxit) fail = 1;
    return X;
}

}  // namespace tsf
