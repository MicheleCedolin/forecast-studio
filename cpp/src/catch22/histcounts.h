#ifndef C22_HISTCOUNTS_H
#define C22_HISTCOUNTS_H
int num_bins_auto(const double y[], const int size);
int histcounts_preallocated(const double y[], const int size, int nBins, int * binCounts, double * binEdges);
int histcounts(const double y[], const int size, int nBins, int ** binCounts, double ** binEdges);
int * histbinassign(const double y[], const int size, const double binEdges[], const int nEdges);
int * histcount_edges(const double y[], const int size, const double binEdges[], const int nEdges);
#endif
