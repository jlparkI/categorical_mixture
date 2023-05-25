#ifndef PROB_CALCS_H
#define PROB_CALCS_H

#include <stdint.h>
#include <stdlib.h>

struct ThreadCExt_Args {
    uint8_t *x;
    double *mu;
    double *resp;
    int startRow;
    int endRow;
    int mu_dim1;
    int mu_dim2;
    int x_dim0;
    int x_dim1;
};

struct ThreadWeightedCount_Args {
    uint8_t *x;
    double *wcount;
    double *resp;
    int startRow;
    int endRow;
    int wc_dim1;
    int wc_dim2;
    int x_dim0;
    int x_dim1;
};



int getProbsCExt_main(uint8_t *x,
                   double *mu, double *resp,
                   int mu_dim0, int mu_dim1,
                   int mu_dim2, int x_dim0,
                   int x_dim1, int n_threads); 

void *getProbsCExt_worker(void *threadArgsIn); 

int getWeightedCountCExt_main(uint8_t *x,
                   double *wcount, double *resp,
                   int wcount_dim0, int wcount_dim1,
                   int wcount_dim2, int x_dim0,
                   int x_dim1, int n_threads); 

void *getWeightedCountCExt_worker(void *threadArgs); 

int getProbsCExt_single_thread(uint8_t *x, double *mu, double *resp,
                   int mu_dim0, int mu_dim1, int mu_dim2,
                   int x_dim0, int x_dim1);

int getWeightedCountCExt_single_thread(uint8_t *x, double *wcount, double *resp,
                   int wc_dim0, int wc_dim1, int wc_dim2,
                   int x_dim0, int x_dim1);


void getHardCountCExt_single_thread(uint8_t *x,
                   uint32_t *cluster_stats,
                   uint32_t *assignments,
                   int stats_dim0, int stats_dim1,
                   int stats_dim2, int x_dim0,
                   int x_dim1); 

#endif
