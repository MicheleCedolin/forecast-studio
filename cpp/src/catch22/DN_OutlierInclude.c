// catch22, from Rcatch22 v0.1.13 src/ (GPL-3). Comments removed, code unchanged.
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "stats.h"
double DN_OutlierInclude_np_001_mdrmd(const double y[], const int size, const int sign)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    double inc = 0.01;
    int tot = 0;
    double * yWork = malloc(size * sizeof(double));
    int constantFlag = 1;
    for(int i = 0; i < size; i++)
    {
        if(y[i] != y[0])
        {
            constantFlag = 0;
        }
        yWork[i] = sign*y[i];
        if(yWork[i] >= 0){
            tot += 1;
        }
    }
    if(constantFlag) return 0;
    double maxVal = max_(yWork, size);
    if(maxVal < inc){
        return 0;
    }
    int nThresh = maxVal/inc + 1;
    double * r = malloc(size * sizeof * r);
    double * msDti1 = malloc(nThresh * sizeof(double));
    double * msDti3 = malloc(nThresh * sizeof(double));
    double * msDti4 = malloc(nThresh * sizeof(double));
    for(int j = 0; j < nThresh; j++)
    {
        int highSize = 0;
        for(int i = 0; i < size; i++)
        {
            if(yWork[i] >= j*inc)
            {
                r[highSize] = i+1;
                highSize += 1;
            }
        }
        double * Dt_exc = malloc(highSize * sizeof(double));
        for(int i = 0; i < highSize-1; i++)
        {
            Dt_exc[i] = r[i+1] - r[i];
        }
        msDti1[j] = mean(Dt_exc, highSize-1);
        msDti3[j] = (highSize-1)*100.0/tot;
        msDti4[j] = median(r, highSize) / ((double)size/2) - 1;
        free(Dt_exc);
    }
    int trimthr = 2;
    int mj = 0;
    int fbi = nThresh-1;
    for(int i = 0; i < nThresh; i ++)
    {
        if (msDti3[i] > trimthr)
        {
            mj = i;
        }
        if (isnan(msDti1[nThresh-1-i]))
        {
            fbi = nThresh-1-i;
        }
    }
    double outputScalar;
    int trimLimit = mj < fbi ? mj : fbi;
    outputScalar = median(msDti4, trimLimit+1);
    free(r);
    free(yWork);
    free(msDti1);
    free(msDti3);
    free(msDti4);
    return outputScalar;
}
double DN_OutlierInclude_p_001_mdrmd(const double y[], const int size)
{
    return DN_OutlierInclude_np_001_mdrmd(y, size, 1.0);
}
double DN_OutlierInclude_n_001_mdrmd(const double y[], const int size)
{
    return DN_OutlierInclude_np_001_mdrmd(y, size, -1.0);
}
