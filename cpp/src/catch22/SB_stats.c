// catch22, from Rcatch22 v0.1.13 src/SB_BinaryStats.c, SB_CoarseGrain.c and SB_TransitionMatrix.c (GPL-3). Comments
// removed, code unchanged except R's error() call (unreachable here: "quantile" is always passed).
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "stats.h"
#include "helper_functions.h"
#include "CO_AutoCorr.h"
double SB_BinaryStats_diff_longstretch0(const double y[], const int size){
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    int * yBin = malloc((size-1) * sizeof(int));
    for(int i = 0; i < size-1; i++){
        double diffTemp = y[i+1] - y[i];
        yBin[i] = diffTemp < 0 ? 0 : 1;
    }
    int maxstretch0 = 0;
    int last1 = 0;
    for(int i = 0; i < size-1; i++){
        if((yBin[i] == 1) | (i == size-2)){
            double stretch0 = i - last1;
            if(stretch0 > maxstretch0){
                maxstretch0 = stretch0;
            }
            last1 = i;
        }
    }
    free(yBin);
    return maxstretch0;
}
double SB_BinaryStats_mean_longstretch1(const double y[], const int size){
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    int * yBin = malloc((size-1) * sizeof(int));
    double yMean = mean(y, size);
    for(int i = 0; i < size-1; i++){
        yBin[i] = (y[i] - yMean <= 0) ? 0 : 1;
    }
    int maxstretch1 = 0;
    int last1 = 0;
    for(int i = 0; i < size-1; i++){
        if((yBin[i] == 0) | (i == size-2)){
            double stretch1 = i - last1;
            if(stretch1 > maxstretch1){
                maxstretch1 = stretch1;
            }
            last1 = i;
        }
    }
    free(yBin);
    return maxstretch1;
}
void sb_coarsegrain(const double y[], const int size, const char how[], const int num_groups, int labels[])
{
    int i, j;
    double * th = malloc((num_groups + 1) * 2 * sizeof(th));
    double * ls = malloc((num_groups + 1) * 2 * sizeof(th));
    linspace(0, 1, num_groups + 1, ls);
    for (i = 0; i < num_groups + 1; i++) {
        th[i] = quantile(y, size, ls[i]);
    }
    th[0] -= 1;
    for (i = 0; i < num_groups; i++) {
        for (j = 0; j < size; j++) {
            if (y[j] > th[i] && y[j] <= th[i + 1]) {
                labels[j] = i + 1;
            }
        }
    }
    free(th);
    free(ls);
}
double SB_TransitionMatrix_3ac_sumdiagcov(const double y[], const int size)
{
    int constant = 1;
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
        if(y[i] != y[0]){
            constant = 0;
        }
    }
    if (constant){
        return NAN;
    }
    const int numGroups = 3;
    int tau = co_firstzero(y, size, size);
    double * yFilt = malloc(size * sizeof(double));
    for(int i = 0; i < size; i++){
        yFilt[i] = y[i];
    }
    int nDown = (size-1)/tau+1;
    double * yDown = malloc(nDown * sizeof(double));
    for(int i = 0; i < nDown; i++){
        yDown[i] = yFilt[i*tau];
    }
    int * yCG = malloc(nDown * sizeof(double));
    sb_coarsegrain(yDown, nDown, "quantile", numGroups, yCG);
    double T[3][3];
    for(int i = 0; i < numGroups; i++){
        for(int j = 0; j < numGroups; j++){
            T[i][j] = 0;
        }
    }
    for(int j = 0; j < nDown-1; j++){
        T[yCG[j]-1][yCG[j+1]-1] += 1;
    }
    for(int i = 0; i < numGroups; i++){
        for(int j = 0; j < numGroups; j++){
            T[i][j] /= (nDown-1);
        }
    }
    double column1[3] = {0};
    double column2[3] = {0};
    double column3[3] = {0};
    for(int i = 0; i < numGroups; i++){
        column1[i] = T[i][0];
        column2[i] = T[i][1];
        column3[i] = T[i][2];
    }
    double *columns[3];
    columns[0] = &(column1[0]);
    columns[1] = &(column2[0]);
    columns[2] = &(column3[0]);
    double COV[3][3];
    double covTemp = 0;
    for(int i = 0; i < numGroups; i++){
        for(int j = i; j < numGroups; j++){
            covTemp = cov(columns[i], columns[j], 3);
            COV[i][j] = covTemp;
            COV[j][i] = covTemp;
        }
    }
    double sumdiagcov = 0;
    for(int i = 0; i < numGroups; i++){
        sumdiagcov += COV[i][i];
    }
    free(yFilt);
    free(yDown);
    free(yCG);
    return sumdiagcov;
}
