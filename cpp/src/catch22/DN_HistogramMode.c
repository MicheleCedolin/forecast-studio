// catch22, from Rcatch22 v0.1.13 src/DN_HistogramMode_5.c and DN_HistogramMode_10.c (GPL-3). Comments removed,
// code unchanged (stdlib.h included for free()).
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "stats.h"
#include "histcounts.h"
double DN_HistogramMode_5(const double y[], const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    const int nBins = 5;
    int * histCounts;
    double * binEdges;
    histcounts(y, size, nBins, &histCounts, &binEdges);
    double maxCount = 0;
    int numMaxs = 1;
    double out = 0;;
    for(int i = 0; i < nBins; i++)
    {
        if (histCounts[i] > maxCount)
        {
            maxCount = histCounts[i];
            numMaxs = 1;
            out = (binEdges[i] + binEdges[i+1])*0.5;
        }
        else if (histCounts[i] == maxCount){
            numMaxs += 1;
            out += (binEdges[i] + binEdges[i+1])*0.5;
        }
    }
    out = out/numMaxs;
    free(histCounts);
    free(binEdges);
    return out;
}
double DN_HistogramMode_10(const double y[], const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    const int nBins = 10;
    int * histCounts;
    double * binEdges;
    histcounts(y, size, nBins, &histCounts, &binEdges);
    double maxCount = 0;
    int numMaxs = 1;
    double out = 0;;
    for(int i = 0; i < nBins; i++)
    {
        if (histCounts[i] > maxCount)
        {
            maxCount = histCounts[i];
            numMaxs = 1;
            out = (binEdges[i] + binEdges[i+1])*0.5;
        }
        else if (histCounts[i] == maxCount){
            numMaxs += 1;
            out += (binEdges[i] + binEdges[i+1])*0.5;
        }
    }
    out = out/numMaxs;
    free(histCounts);
    free(binEdges);
    return out;
}
