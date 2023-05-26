#ifndef RESPONSIBILITY_CALCS_H
#define RESPONSIBILITY_CALCS_H

#include <stdint.h>
#include <stdlib.h>


int getProbsCExt_main(uint8_t *x,
                   double *mu, double *resp,
                   int mu_dim0, int mu_dim1,
                   int mu_dim2, int x_dim0,
                   int x_dim1, int n_threads); 

void *getProbsCExt_worker(uint8_t *x, double *resp,
        double *mu, int mu_dim1, int mu_dim2,
        int x_dim0, int x_dim1, int startRow,
        int endRow);


int getProbsCExt_single_thread(uint8_t *x, double *mu, double *resp,
                   int mu_dim0, int mu_dim1, int mu_dim2,
                   int x_dim0, int x_dim1);


void getHardCountCExt_single_thread(uint8_t *x,
                   uint32_t *cluster_stats,
                   uint32_t *assignments,
                   int stats_dim0, int stats_dim1,
                   int stats_dim2, int x_dim0,
                   int x_dim1); 

#endif
