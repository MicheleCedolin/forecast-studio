#ifndef C22_HELPER_FUNCTIONS_H
#define C22_HELPER_FUNCTIONS_H
#include <complex.h>
void sort(double y[], int size);
void linspace(double start, double end, int num_groups, double out[]);
double quantile(const double y[], const int size, const double quant);
void binarize(const double a[], const int size, int b[], const char how[]);
double f_entropy(const double a[], const int size);
void subset(const int a[], int b[], const int start, const int end);
double _Complex _Cmulcc(const double _Complex x, const double _Complex y);
double _Complex _Cminuscc(const double _Complex x, const double _Complex y);
double _Complex _Caddcc(const double _Complex x, const double _Complex y);
double _Complex _Cdivcc(const double _Complex x, const double _Complex y);
#endif
