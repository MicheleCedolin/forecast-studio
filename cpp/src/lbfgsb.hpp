// R's optim(par, fn, method = "L-BFGS-B", lower, upper) without a gradient function: a port of R 4.6's
// src/appl/lbfgsb.c (L-BFGS-B 2.3 by Zhu, Byrd, Lu & Nocedal, f2c'd and modified for R), the lbfgsb() driver of
// src/appl/optim.c and the bounded finite-difference gradient of stats/src/optim.c (ndeps = 1e-3, parscale and
// fnscale 1), with LINPACK dpofa / dtrsl and the reference BLAS.
#pragma once

#include <functional>
#include <string>

#include "util.hpp"

namespace tsf {

struct LbfgsbResult {
    Vec par;
    double value = 0;
    int fail = 0;                   // optim()'s convergence: 0, 1 (maxit), 51 (warning), 52 (error)
    std::string msg;
};

LbfgsbResult r_optim_lbfgsb(const std::function<double(const Vec&)>& fn, const Vec& par, const Vec& lower,
                            const Vec& upper, int maxit = 100, int lmm = 5, double factr = 1e7, double pgtol = 0.0,
                            double ndeps = 1e-3);

}  // namespace tsf
