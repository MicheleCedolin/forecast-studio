// catch22, from Rcatch22 v0.1.13 src/ (GPL-3). Comments removed; GCC branch kept, code unchanged except the
// c22_unit_twiddles switch: the Rcatch22 build behind the ranking study's stored catch22 values computed every
// twiddle factor as 1 (its FFT-based autocorrelations and Welch spectra reproduce exactly that way), so this
// mode reproduces those values; the default is the correct FFT.
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "helper_functions.h"
_Thread_local int c22_unit_twiddles = 0;
void twiddles(double _Complex a[], int size)
{
    if (c22_unit_twiddles) {
        for (int i = 0; i < size; i++) a[i] = 1.0;
        return;
    }
    double PI = 3.14159265359;
    for (int i = 0; i < size; i++) {
        double _Complex tmp = 0.0  - PI * i / size * I;
        a[i] = cexp(tmp);
    }
}
static void _fft(double _Complex a[], double _Complex out[], int size, int step, double _Complex tw[])
{
    if (step < size) {
        _fft(out, a, size, step * 2, tw);
        _fft(out + step, a + step, size, step * 2, tw);
        for (int i = 0; i < size; i += 2 * step) {
            double _Complex t = _Cmulcc(tw[i], out[i + step]);
            a[i / 2] = _Caddcc(out[i], t);
            a[(i + size) / 2] = _Cminuscc(out[i], t);
        }
    }
}
void fft(double _Complex a[], int size, double _Complex tw[])
{
    double _Complex * out = malloc(size * sizeof(double _Complex));
    memcpy(out, a, size * sizeof(double _Complex));
    _fft(a, out, size, 1, tw);
    free(out);
}
