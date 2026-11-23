// R_cpolyroot: the complex Jenkins-Traub polynomial root finder (CACM algorithm 419) as R and the forecast package
// (src/etspolyroot.c) ship it, ported line by line. The C sources keep state in file- and function-level statics;
// they are thread_local here, so a thread sees the same carry-over between calls as an R process does.
#pragma once

namespace tsf {

// opr / opi: coefficients from the highest degree down (degree + 1 of each); zeror / zeroi receive degree roots.
void r_cpolyroot(const double* opr, const double* opi, int degree, double* zeror, double* zeroi, bool* fail);

}  // namespace tsf
