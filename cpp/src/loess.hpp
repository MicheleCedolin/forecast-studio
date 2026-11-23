// R's loess() for one predictor, as bld.mbb.bootstrap() calls it: loess(y ~ x, span, degree = 1), gaussian family,
// surface = "interpolate" (k-d tree, local fits at the vertices, cubic Hermite interpolation), cell = 0.2.
// A line-by-line port of R's loessc.c / loessf.f (W. S. Cleveland, E. Grosse, M. J. Shyu; netlib dloess) with the
// LINPACK dqrdc / dqrsl / dsvdc of src/appl and R's reference BLAS (ddot, daxpy, dscal, drot, the LAPACK 3.10+
// dnrm2 and drotg), so the fitted values equal R's to rounding.
#pragma once

#include "util.hpp"

namespace tsf {

// Fitted values of loess(y ~ x, span = span, degree = 1) at the data points (d = 1, weights 1).
Vec loess_fit_1d(const Vec& x, const Vec& y, double span, double cell = 0.2);

}  // namespace tsf
