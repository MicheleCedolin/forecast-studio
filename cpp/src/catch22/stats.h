#ifndef C22_STATS_H
#define C22_STATS_H
double min_(const double a[], const int size);
double max_(const double a[], const int size);
double mean(const double a[], const int size);
double sum(const double a[], const int size);
void cumsum(const double a[], const int size, double b[]);
void icumsum(const int a[], const int size, int b[]);
double isum(const int a[], const int size);
double median(const double a[], const int size);
double stddev(const double a[], const int size);
double cov(const double x[], const double y[], const int size);
double cov_mean(const double x[], const double y[], const int size);
double corr(const double x[], const double y[], const int size);
double autocorr_lag(const double x[], const int size, const int lag);
double autocov_lag(const double x[], const int size, const int lag);
void zscore_norm(double a[], int size);
void zscore_norm2(const double a[], const int size, double b[]);
double moment(const double a[], const int size, const int start, const int end, const int r);
void diff(const double a[], const int size, double b[]);
int linreg(const int n, const double x[], const double y[], double* m, double* b);
double norm_(const double a[], const int size);
#endif
