// catch22, from Rcatch22 v0.1.13 src/ (GPL-3). Comments removed; GCC branch kept, code unchanged (CMPLX falls back
// to __builtin_complex where complex.h lacks it).
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>
#include "stats.h"
#include "fft.h"
#include "histcounts.h"
#include "helper_functions.h"
#ifndef CMPLX
#define CMPLX(x, y) __builtin_complex((double)(x), (double)(y))
#endif
#define pow2(x) (1 << x)
int nextpow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}
void dot_multiply(double _Complex a[], double _Complex b[], int size)
{
    for (int i = 0; i < size; i++) {
        a[i] = _Cmulcc(a[i], conj(b[i]));
    }
}
double * CO_AutoCorr(const double y[], const int size, const int tau[], const int tau_size)
{
    double m, nFFT;
    m = mean(y, size);
    nFFT = nextpow2(size) << 1;
    double _Complex * F = malloc(nFFT * sizeof *F);
    double _Complex * tw = malloc(nFFT * sizeof *tw);
    for (int i = 0; i < size; i++) {
        F[i] = CMPLX(y[i] - m, 0.0);
    }
    for (int i = size; i < nFFT; i++) {
        F[i] = CMPLX(0.0, 0.0);
    }
    twiddles(tw, nFFT);
    fft(F, nFFT, tw);
    dot_multiply(F, F, nFFT);
    fft(F, nFFT, tw);
    double _Complex divisor = F[0];
    for (int i = 0; i < nFFT; i++) {
        F[i] = _Cdivcc(F[i], divisor);
    }
    double * out = malloc(tau_size * sizeof(out));
    for (int i = 0; i < tau_size; i++) {
        out[i] = creal(F[tau[i]]);
    }
    free(F);
    free(tw);
    return out;
}
double * co_autocorrs(const double y[], const int size)
{
    double m, nFFT;
    m = mean(y, size);
    nFFT = nextpow2(size) << 1;
    double _Complex * F = malloc(nFFT * 2 * sizeof *F);
    double _Complex * tw = malloc(nFFT * 2 * sizeof *tw);
    for (int i = 0; i < size; i++) {
        F[i] = CMPLX(y[i] - m, 0.0);
    }
    for (int i = size; i < nFFT; i++) {
        F[i] = CMPLX(0.0, 0.0);
    }
    twiddles(tw, nFFT);
    fft(F, nFFT, tw);
    dot_multiply(F, F, nFFT);
    fft(F, nFFT, tw);
    double _Complex divisor = F[0];
    for (int i = 0; i < nFFT; i++) {
        F[i] = _Cdivcc(F[i], divisor);
    }
    double * out = malloc(nFFT * 2 * sizeof(out));
    for (int i = 0; i < nFFT; i++) {
        out[i] = creal(F[i]);
    }
    free(F);
    free(tw);
    return out;
}
int co_firstzero(const double y[], const int size, const int maxtau)
{
    double * autocorrs = co_autocorrs(y, size);
    int zerocrossind = 0;
    while(autocorrs[zerocrossind] > 0 && zerocrossind < maxtau)
    {
        zerocrossind += 1;
    }
    free(autocorrs);
    return zerocrossind;
}
int CO_f1ecac(const double y[], const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return 0;
        }
    }
    double * autocorrs = co_autocorrs(y, size);
    double thresh = 1.0/exp(1);
    int out = size;
    for(int i = 0; i < size-1; i++){
        if ((autocorrs[i] - thresh)*(autocorrs[i+1] - thresh) < 0){
            out = i + 1;
            free(autocorrs);
            return out;
        }
    }
    free(autocorrs);
    return out;
}
double CO_Embed2_Basic_tau_incircle(const double y[], const int size, const double radius, const int tau)
{
    int tauIntern = 0;
    if(tau < 0)
    {
        tauIntern = co_firstzero(y, size, size);
    }
    else{
        tauIntern = tau;
    }
    double insidecount = 0;
    for(int i = 0; i < size-tauIntern; i++)
    {
        if(y[i]*y[i] + y[i+tauIntern]*y[i+tauIntern] < radius)
        {
            insidecount += 1;
        }
    }
    return insidecount/(size-tauIntern);
}
double CO_Embed2_Dist_tau_d_expfit_meandiff(const double y[], const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    int tau = co_firstzero(y, size, size);
    if (tau > (double)size/10){
        tau = floor((double)size/10);
    }
    double * d = malloc((size-tau) * sizeof(double));
    for(int i = 0; i < size-tau-1; i++)
    {
        d[i] = sqrt((y[i+1]-y[i])*(y[i+1]-y[i]) + (y[i+tau]-y[i+tau+1])*(y[i+tau]-y[i+tau+1]));
        if (isnan(d[i])){
            free(d);
            return NAN;
        }
    }
    double l = mean(d, size-tau-1);
    int nBins = num_bins_auto(d, size-tau-1);
    if (nBins == 0){
        return 0;
    }
    int * histCounts = malloc(nBins * sizeof(double));
    double * binEdges = malloc((nBins + 1) * sizeof(double));
    histcounts_preallocated(d, size-tau-1, nBins, histCounts, binEdges);
    double * histCountsNorm = malloc(nBins * sizeof(double));
    for(int i = 0; i < nBins; i++){
        histCountsNorm[i] = (double)histCounts[i]/(double)(size-tau-1);
    }
    double * d_expfit_diff = malloc(nBins * sizeof(double));
    for(int i = 0; i < nBins; i++){
        double expf = exp(-(binEdges[i] + binEdges[i+1])*0.5/l)/l;
        if (expf < 0){
            expf = 0;
        }
        d_expfit_diff[i] = fabs(histCountsNorm[i]-expf);
    }
    double out = mean(d_expfit_diff, nBins);
    free(d);
    free(d_expfit_diff);
    free(binEdges);
    free(histCountsNorm);
    free(histCounts);
    return out;
}
int CO_FirstMin_ac(const double y[], const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return 0;
        }
    }
    double * autocorrs = co_autocorrs(y, size);
    int minInd = size;
    for(int i = 1; i < size-1; i++)
    {
        if(autocorrs[i] < autocorrs[i-1] && autocorrs[i] < autocorrs[i+1])
        {
            minInd = i;
            break;
        }
    }
    free(autocorrs);
    return minInd;
}
double CO_trev_1_num(const double y[], const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    int tau = 1;
    double * diffTemp = malloc((size-1) * sizeof * diffTemp);
    for(int i = 0; i < size-tau; i++)
    {
        diffTemp[i] = pow(y[i+1] - y[i],3);
    }
    double out;
    out = mean(diffTemp, size-tau);
    free(diffTemp);
    return out;
}
#define tau 2
#define numBins 5
double CO_HistogramAMI_even_2_5(const double y[], const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    double * y1 = malloc((size-tau) * sizeof(double));
    double * y2 = malloc((size-tau) * sizeof(double));
    for(int i = 0; i < size-tau; i++){
        y1[i] = y[i];
        y2[i] = y[i+tau];
    }
    const double maxValue = max_(y, size);
    const double minValue = min_(y, size);
    double binStep = (maxValue - minValue + 0.2)/5;
    double binEdges[5+1] = {0};
    for(int i = 0; i < numBins+1; i++){
        binEdges[i] = minValue + binStep*i - 0.1;
    }
    int * bins1;
    bins1 = histbinassign(y1, size-tau, binEdges, numBins+1);
    int * bins2;
    bins2 = histbinassign(y2, size-tau, binEdges, numBins+1);
    double * bins12 = malloc((size-tau) * sizeof(double));
    double binEdges12[(5 + 1) * (5 + 1)] = {0};
    for(int i = 0; i < size-tau; i++){
        bins12[i] = (bins1[i]-1)*(numBins+1) + bins2[i];
    }
    for(int i = 0; i < (numBins+1)*(numBins+1); i++){
        binEdges12[i] = i+1;
    }
    int * jointHistLinear;
    jointHistLinear = histcount_edges(bins12, size-tau, binEdges12, (numBins + 1) * (numBins + 1));
    double pij[numBins][numBins];
    int sumBins = 0;
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            pij[j][i] = jointHistLinear[i*(numBins+1)+j];
            sumBins += pij[j][i];
        }
    }
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            pij[j][i] /= sumBins;
        }
    }
    double pi[5] = {0};
    double pj[5] = {0};
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            pi[i] += pij[i][j];
            pj[j] += pij[i][j];
        }
    }
    double ami = 0;
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            if(pij[i][j] > 0){
                ami += pij[i][j] * log(pij[i][j]/(pj[j]*pi[i]));
            }
        }
    }
    free(bins1);
    free(bins2);
    free(jointHistLinear);
    free(y1);
    free(y2);
    free(bins12);
    return ami;
}
