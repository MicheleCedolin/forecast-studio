#ifndef C22_CO_AUTOCORR_H
#define C22_CO_AUTOCORR_H
int nextpow2(int n);
double * CO_AutoCorr(const double y[], const int size, const int tau[], const int tau_size);
double * co_autocorrs(const double y[], const int size);
int co_firstzero(const double y[], const int size, const int maxtau);
#endif
