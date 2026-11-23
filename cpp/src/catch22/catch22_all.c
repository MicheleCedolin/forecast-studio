// Wrapper reproducing Rcatch22 v0.1.13's catch22_all(): every feature is computed on zscore_norm2(y), int-valued
// features converted to double.
#include <stdlib.h>
#include "stats.h"
#include "catch22.h"
#include "fft.h"
double DN_HistogramMode_5(const double y[], const int size);
double DN_HistogramMode_10(const double y[], const int size);
int CO_f1ecac(const double y[], const int size);
int CO_FirstMin_ac(const double y[], const int size);
double CO_HistogramAMI_even_2_5(const double y[], const int size);
double CO_trev_1_num(const double y[], const int size);
double MD_hrv_classic_pnn40(const double y[], const int size);
double SB_BinaryStats_mean_longstretch1(const double y[], const int size);
double SB_TransitionMatrix_3ac_sumdiagcov(const double y[], const int size);
int PD_PeriodicityWang_th0_01(const double * y, const int size);
double CO_Embed2_Dist_tau_d_expfit_meandiff(const double y[], const int size);
double IN_AutoMutualInfoStats_40_gaussian_fmmi(const double y[], const int size);
double FC_LocalSimple_mean1_tauresrat(const double y[], const int size);
double DN_OutlierInclude_p_001_mdrmd(const double y[], const int size);
double DN_OutlierInclude_n_001_mdrmd(const double y[], const int size);
double SP_Summaries_welch_rect_area_5_1(const double y[], const int size);
double SB_BinaryStats_diff_longstretch0(const double y[], const int size);
double SB_MotifThree_quantile_hh(const double y[], const int size);
double SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(const double y[], const int size);
double SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(const double y[], const int size);
double SP_Summaries_welch_rect_centroid(const double y[], const int size);
double FC_LocalSimple_mean3_stderr(const double y[], const int size);
const char * const CATCH22_NAMES[CATCH22_N] = {
    "DN_HistogramMode_5", "DN_HistogramMode_10", "CO_f1ecac", "CO_FirstMin_ac", "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num", "MD_hrv_classic_pnn40", "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov", "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi", "FC_LocalSimple_mean1_tauresrat", "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd", "SP_Summaries_welch_rect_area_5_1", "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SP_Summaries_welch_rect_centroid", "FC_LocalSimple_mean3_stderr"};
void catch22_all(const double y[], int size, double out[], int study_fft)
{
    c22_unit_twiddles = study_fft;
    double * z = malloc(size * sizeof(double));
    zscore_norm2(y, size, z);
    out[0] = DN_HistogramMode_5(z, size);
    out[1] = DN_HistogramMode_10(z, size);
    out[2] = CO_f1ecac(z, size);
    out[3] = CO_FirstMin_ac(z, size);
    out[4] = CO_HistogramAMI_even_2_5(z, size);
    out[5] = CO_trev_1_num(z, size);
    out[6] = MD_hrv_classic_pnn40(z, size);
    out[7] = SB_BinaryStats_mean_longstretch1(z, size);
    out[8] = SB_TransitionMatrix_3ac_sumdiagcov(z, size);
    out[9] = PD_PeriodicityWang_th0_01(z, size);
    out[10] = CO_Embed2_Dist_tau_d_expfit_meandiff(z, size);
    out[11] = IN_AutoMutualInfoStats_40_gaussian_fmmi(z, size);
    out[12] = FC_LocalSimple_mean1_tauresrat(z, size);
    out[13] = DN_OutlierInclude_p_001_mdrmd(z, size);
    out[14] = DN_OutlierInclude_n_001_mdrmd(z, size);
    out[15] = SP_Summaries_welch_rect_area_5_1(z, size);
    out[16] = SB_BinaryStats_diff_longstretch0(z, size);
    out[17] = SB_MotifThree_quantile_hh(z, size);
    out[18] = SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(z, size);
    out[19] = SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(z, size);
    out[20] = SP_Summaries_welch_rect_centroid(z, size);
    out[21] = FC_LocalSimple_mean3_stderr(z, size);
    free(z);
    c22_unit_twiddles = 0;
}
