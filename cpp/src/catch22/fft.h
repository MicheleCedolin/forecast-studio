#ifndef C22_FFT_H
#define C22_FFT_H
#include <complex.h>
extern _Thread_local int c22_unit_twiddles;
void twiddles(double _Complex a[], int size);
void fft(double _Complex a[], int size, double _Complex tw[]);
#endif
