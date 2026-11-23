// catch22, from Rcatch22 v0.1.13 src/SB_MotifThree.c (GPL-3): SB_MotifThree_quantile_hh only (the unused
// sb_motifthree is left out). Comments removed, code unchanged.
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "SB_CoarseGrain.h"
#include "helper_functions.h"
double SB_MotifThree_quantile_hh(const double y[], const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(isnan(y[i]))
        {
            return NAN;
        }
    }
    int tmp_idx, r_idx;
    int dynamic_idx;
    int alphabet_size = 3;
    int array_size;
    int * yt = malloc(size * sizeof(yt));
    double hh;
    double * out = malloc(124 * sizeof(out));
    sb_coarsegrain(y, size, "quantile", 3, yt);
    array_size = alphabet_size;
    int ** r1 = malloc(array_size * sizeof(*r1));
    int * sizes_r1 = malloc(array_size * sizeof(sizes_r1));
    double * out1 = malloc(array_size * sizeof(out1));
    for (int i = 0; i < alphabet_size; i++) {
        r1[i] = malloc(size * sizeof(r1[i]));
        r_idx = 0;
        sizes_r1[i] = 0;
        for (int j = 0; j < size; j++) {
            if (yt[j] == i + 1) {
                r1[i][r_idx++] = j;
                sizes_r1[i]++;
            }
        }
    }
    array_size *= alphabet_size;
    for (int i = 0; i < alphabet_size; i++) {
        if (sizes_r1[i] != 0 && r1[i][sizes_r1[i] - 1] == size - 1) {
            int* tmp_ar = malloc(sizes_r1[i] * sizeof(tmp_ar));
            subset(r1[i], tmp_ar, 0, sizes_r1[i]);
            memcpy(r1[i], tmp_ar, (sizes_r1[i] - 1) * sizeof(tmp_ar));
            sizes_r1[i]--;
            free(tmp_ar);
        }
    }
    int*** r2 = malloc(alphabet_size * sizeof(**r2));
    int** sizes_r2 = malloc(alphabet_size * sizeof(*sizes_r2));
    double** out2 = malloc(alphabet_size * sizeof(*out2));
    for (int i = 0; i < alphabet_size; i++) {
        r2[i] = malloc(alphabet_size * sizeof(*r2[i]));
        sizes_r2[i] = malloc(alphabet_size * sizeof(*sizes_r2[i]));
        out2[i] = malloc(alphabet_size * sizeof(**out2));
        for (int j = 0; j < alphabet_size; j++) {
            r2[i][j] = malloc(size * sizeof(*r2[i][j]));
        }
    }
    for (int i = 0; i < alphabet_size; i++) {
        for (int j = 0; j < alphabet_size; j++) {
            sizes_r2[i][j] = 0;
            dynamic_idx = 0;
            for (int k = 0; k < sizes_r1[i]; k++) {
                tmp_idx = yt[r1[i][k] + 1];
                if (tmp_idx == (j + 1)) {
                    r2[i][j][dynamic_idx++] = r1[i][k];
                    sizes_r2[i][j]++;
                }
            }
            double tmp = (double)sizes_r2[i][j] / ((double)(size) - (double)(1.0));
            out2[i][j] =  tmp;
        }
    }
    hh = 0.0;
    for (int i = 0; i < alphabet_size; i++) {
        hh += f_entropy(out2[i], alphabet_size);
    }
    free(yt);
    free(out);
    free(out1);
    free(sizes_r1);
    for (int i = 0; i < alphabet_size; i++) {
        free(r1[i]);
    }
    free(r1);
    for (int i = 0; i < alphabet_size; i++) {
        free(sizes_r2[i]);
        free(out2[i]);
    }
    for(int i = 0; i < alphabet_size; i++) {
        for (int j = 0; j < alphabet_size; j++) {
            free(r2[i][j]);
        }
        free(r2[i]);
    }
    free(r2);
    free(sizes_r2);
    free(out2);
    return hh;
}
