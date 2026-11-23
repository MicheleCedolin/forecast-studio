// catch22, from Rcatch22 v0.1.13 src/FC_LocalSimple.c, IN_AutoMutualInfoStats.c, MD_hrv.c and PD_PeriodicityWang.c
// (GPL-3). Comments removed, code unchanged (stdlib.h included for malloc/free; the unused FC_LocalSimple helpers
// fc_local_simple, FC_LocalSimple_mean_taures and FC_LocalSimple_lfit_taures are left out).
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "stats.h"
#include "CO_AutoCorr.h"
#include "splinefit.h"
double FC_LocalSimple_mean_tauresrat(const double y[], const int size, const int train_length)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    double * res = malloc((size - train_length) * sizeof *res);
    for (int i = 0; i < size - train_length; i++)
    {
        double yest = 0;
        for (int j = 0; j < train_length; j++)
        {
            yest += y[i+j];
        }
        yest /= train_length;
        res[i] = y[i+train_length] - yest;
    }
    double resAC1stZ = co_firstzero(res, size - train_length, size - train_length);
    double yAC1stZ = co_firstzero(y, size, size);
    double output = resAC1stZ/yAC1stZ;
    free(res);
    return output;
}
double FC_LocalSimple_mean_stderr(const double y[], const int size, const int train_length)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    double * res = malloc((size - train_length) * sizeof *res);
    for (int i = 0; i < size - train_length; i++)
    {
        double yest = 0;
        for (int j = 0; j < train_length; j++)
        {
            yest += y[i+j];
        }
        yest /= train_length;
        res[i] = y[i+train_length] - yest;
    }
    double output = stddev(res, size - train_length);
    free(res);
    return output;
}
double FC_LocalSimple_mean3_stderr(const double y[], const int size)
{
    return FC_LocalSimple_mean_stderr(y, size, 3);
}
double FC_LocalSimple_mean1_tauresrat(const double y[], const int size){
    return FC_LocalSimple_mean_tauresrat(y, size, 1);
}
double IN_AutoMutualInfoStats_40_gaussian_fmmi(const double y[], const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    int tau = 40;
    if(tau > ceil((double)size/2)){
        tau = ceil((double)size/2);
    }
    double * ami = malloc(size * sizeof(double));
    for(int i = 0; i < tau; i++){
        double ac = autocorr_lag(y,size, i+1);
        ami[i] = -0.5 * log(1 - ac*ac);
    }
    double fmmi = tau;
    for(int i = 1; i < tau-1; i++){
        if((ami[i] < ami[i-1]) & (ami[i] < ami[i+1])){
            fmmi = i;
            break;
        }
    }
    free(ami);
    return fmmi;
}
double MD_hrv_classic_pnn40(const double y[], const int size){
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    const int pNNx = 40;
    double * Dy = malloc((size-1) * sizeof(double));
    diff(y, size, Dy);
    double pnn40 = 0;
    for(int i = 0; i < size-1; i++){
        if(fabs(Dy[i])*1000 > pNNx){
            pnn40 += 1;
        }
    }
    free(Dy);
    return pnn40/(size-1);
}
int PD_PeriodicityWang_th0_01(const double * y, const int size){
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return 0;
        }
    }
    const double th = 0.01;
    double * ySpline = malloc(size * sizeof(double));
    splinefit(y, size, ySpline);
    double * ySub = malloc(size * sizeof(double));
    for(int i = 0; i < size; i++){
        ySub[i] = y[i] - ySpline[i];
    }
    int acmax = (int)ceil((double)size/3);
    double * acf = malloc(acmax*sizeof(double));
    for(int tau = 1; tau <= acmax; tau++){
        acf[tau-1] = autocov_lag(ySub, size, tau);
    }
    double * troughs = malloc(acmax * sizeof(double));
    double * peaks = malloc(acmax * sizeof(double));
    int nTroughs = 0;
    int nPeaks = 0;
    double slopeIn = 0;
    double slopeOut = 0;
    for(int i = 1; i < acmax-1; i ++){
        slopeIn = acf[i] - acf[i-1];
        slopeOut = acf[i+1] - acf[i];
        if((slopeIn < 0) & (slopeOut > 0))
        {
            troughs[nTroughs] = i;
            nTroughs += 1;
        }
        else if((slopeIn > 0) & (slopeOut < 0))
        {
            peaks[nPeaks] = i;
            nPeaks += 1;
        }
    }
    int iPeak = 0;
    double thePeak = 0;
    int iTrough = 0;
    double theTrough = 0;
    int out = 0;
    for(int i = 0; i < nPeaks; i++){
        iPeak = peaks[i];
        thePeak = acf[iPeak];
        int j = -1;
        while(j+1 < nTroughs && troughs[j+1] < iPeak){
            j++;
        }
        if(j == -1)
            continue;
        iTrough = troughs[j];
        theTrough = acf[iTrough];
        if(thePeak - theTrough < th)
            continue;
        if(thePeak < 0)
            continue;
        out = iPeak;
        break;
    }
    free(ySpline);
    free(ySub);
    free(acf);
    free(troughs);
    free(peaks);
    return out;
}
