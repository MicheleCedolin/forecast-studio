// The 22 catch22 features as computed by Rcatch22 v0.1.13 (each on the z-scored series), in the order of
// CATCH22_NAMES. GPL-3 code; see cpp/src/catch22/*.c.
#ifndef C22_CATCH22_H
#define C22_CATCH22_H
#ifdef __cplusplus
extern "C" {
#endif
#define CATCH22_N 22
extern const char * const CATCH22_NAMES[CATCH22_N];
// study_fft = 1 reproduces the ranking study's Rcatch22 values (FFT with unit twiddle factors); 0 is correct catch22.
void catch22_all(const double y[], int size, double out[], int study_fft);
#ifdef __cplusplus
}
#endif
#endif
