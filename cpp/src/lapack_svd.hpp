// R's svd(X)$v, as stats::arima uses it to rotate its regressors: La.svd -> LAPACK 3.12.1 dgesdd(JOBZ = 'S') with the
// reference BLAS that R ships. Ported routine by routine in the same floating-point operation order, so the right
// singular vectors equal R's bit for bit, including the basis LAPACK picks inside a repeated singular value (Fourier
// regressors: the sin and cos columns of a harmonic have nearly equal norms), where any other SVD returns another,
// equally valid basis and the arima optimiser then takes another path.
#pragma once

#include <vector>

#include "util.hpp"

namespace tsf {

// cols: the n x k matrix by columns (n >= k). On success V[i][j] = svd(X)$v[i + 1, j + 1] and *d = svd(X)$d.
// Returns false outside the ported paths (k > 25: LAPACK's divide and conquer; n < k; entries scaled by dgesdd).
bool r_svd_v(const std::vector<Vec>& cols, std::vector<Vec>& V, Vec* d = nullptr);

// Which reference dnrm2 the BLAS uses: true = LAPACK >= 3.10 (Anderson's algorithm, R >= 4.4), false = the classic
// scaled sum of squares. Exposed for the comparison with R's own svd() in the tests.
void r_svd_set_new_dnrm2(bool on);

}  // namespace tsf
