#include "prob_calcs.h"
#include <math.h>
#include <pthread.h>

#define MEMORY_ERROR 1
#define NO_ERROR 0
#define THREAD_ERROR 2

int getProbsCExt_main(uint8_t *x, double *mu, double *resp,
                   int mu_dim0, int mu_dim1, int mu_dim2,
                   int x_dim0, int x_dim1,
                   int n_threads){
    struct ThreadCExt_Args *threadArgs = malloc(n_threads *
            sizeof(struct ThreadCExt_Args));
    if (threadArgs == NULL){
        return MEMORY_ERROR;
    }

    //Note the variable length array; may be problematic on older compilers.
    int i, threadFlags[n_threads];
    void *retval[n_threads];
    int iret[n_threads];
    pthread_t thread_id[n_threads];

    int chunkSize = (mu_dim0 + n_threads - 1) / n_threads;

    for (i=0; i < n_threads; i++){
        threadArgs[i].startRow = i * chunkSize;
        threadArgs[i].endRow = (i + 1) * chunkSize;
        if (threadArgs[i].endRow > mu_dim0)
            threadArgs[i].endRow = mu_dim0;
        threadArgs[i].x = x;
        threadArgs[i].mu = mu;
        threadArgs[i].resp = resp;
        threadArgs[i].mu_dim1 = mu_dim1;
        threadArgs[i].mu_dim2 = mu_dim2;
        threadArgs[i].x_dim0 = x_dim0;
        threadArgs[i].x_dim1 = x_dim1;
    }

    for (i=0; i < n_threads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL,
                getProbsCExt_worker, &threadArgs[i]);
        if (iret[i]){
            return THREAD_ERROR;
        }
    }

    for (i=0; i < n_threads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);

    free(threadArgs);
    return NO_ERROR;
}


void *getProbsCExt_worker(void *threadArgsIn){
    struct ThreadCExt_Args *threadArgs = (struct ThreadCExt_Args *)threadArgsIn;
    int i, j, k, mu_row, mu_dim2;
    uint8_t *x_current;
    double *resp_current, *mu_current, *mu_marker;
    double resp_value;
    int mu_row_size = threadArgs->mu_dim1 * threadArgs->mu_dim2;

    mu_dim2 = threadArgs->mu_dim2;

    for (k=threadArgs->startRow; k < threadArgs->endRow; k++){
        resp_current = threadArgs->resp + k * threadArgs->x_dim0;
        x_current = threadArgs->x;
        mu_row = k * mu_row_size;
        for (i=0; i < threadArgs->x_dim0; i++){
            resp_value = 0;
            mu_marker = threadArgs->mu + mu_row;
            for (j=0; j < threadArgs->x_dim1; j++){
                mu_current = mu_marker + *x_current;
                resp_value += *mu_current;
                x_current++;
                mu_marker += mu_dim2;
            }
            *resp_current = resp_value;
            resp_current++;
        }
    }
    return NULL;
}




int getWeightedCountCExt_main(uint8_t *x, double *wcount, double *resp,
                   int wcount_dim0, int wcount_dim1, int wcount_dim2,
                   int x_dim0, int x_dim1, int n_threads){
    struct ThreadWeightedCount_Args *threadArgs = malloc(n_threads *
            sizeof(struct ThreadWeightedCount_Args));
    if (threadArgs == NULL){
        return MEMORY_ERROR;
    }

    //Note the variable length array; may be problematic on older compilers.
    int i, threadFlags[n_threads];
    void *retval[n_threads];
    int iret[n_threads];
    pthread_t thread_id[n_threads];

    int chunkSize = (wcount_dim0 + n_threads - 1) / n_threads;

    for (i=0; i < n_threads; i++){
        threadArgs[i].startRow = i * chunkSize;
        threadArgs[i].endRow = (i + 1) * chunkSize;
        if (threadArgs[i].endRow > wcount_dim0)
            threadArgs[i].endRow = wcount_dim0;
        threadArgs[i].x = x;
        threadArgs[i].wcount = wcount;
        threadArgs[i].resp = resp;
        threadArgs[i].wc_dim1 = wcount_dim1;
        threadArgs[i].wc_dim2 = wcount_dim2;
        threadArgs[i].x_dim0 = x_dim0;
        threadArgs[i].x_dim1 = x_dim1;
    }

    for (i=0; i < n_threads; i++){
        iret[i] = pthread_create(&thread_id[i], NULL,
                getWeightedCountCExt_worker, &threadArgs[i]);
        if (iret[i]){
            return THREAD_ERROR;
        }
    }

    for (i=0; i < n_threads; i++)
        threadFlags[i] = pthread_join(thread_id[i], &retval[i]);

    free(threadArgs);
    return NO_ERROR;
}



void *getWeightedCountCExt_worker(void *threadArgsIn){
    struct ThreadWeightedCount_Args *threadArgs = (struct
            ThreadWeightedCount_Args *)threadArgsIn;
    int i, j, k, wcount_row, wc_dim2;
    uint8_t *x_current;
    double *resp_current, *wcount_current, *wcount_marker;
    double resp_value;
    int wcount_row_size = threadArgs->wc_dim1 * threadArgs->wc_dim2;

    wc_dim2 = threadArgs->wc_dim2;

    for (k=threadArgs->startRow; k < threadArgs->endRow; k++){
        x_current = threadArgs->x;
        resp_current = threadArgs->resp + k * threadArgs->x_dim0;
        wcount_row = k * wcount_row_size;
        for (i=0; i < threadArgs->x_dim0; i++){
            resp_value = *resp_current;
            wcount_marker = threadArgs->wcount + wcount_row;
            for (j=0; j < threadArgs->x_dim1; j++){
                wcount_current = wcount_marker + *x_current;
                *wcount_current += resp_value;
                x_current++;
                wcount_marker += wc_dim2;
            }
            resp_current++;
        }
    }
    return NULL;
}



int getProbsCExt_single_thread(uint8_t *x, double *mu, double *resp,
                   int mu_dim0, int mu_dim1, int mu_dim2,
                   int x_dim0, int x_dim1){
    int i, j, k, mu_row;
    uint8_t *x_current;
    double *resp_current, *mu_current, *mu_marker;
    double resp_value;
    int mu_row_size = mu_dim1 * mu_dim2;

    for (k=0; k < mu_dim0; k++){
        resp_current = resp + k * x_dim0;
        x_current = x;
        mu_row = k * mu_row_size;
        for (i=0; i < x_dim0; i++){
            resp_value = 0;
            mu_marker = mu + mu_row;
            for (j=0; j < x_dim1; j++){
                mu_current = mu_marker + *x_current;
                resp_value += *mu_current;
                x_current++;
                mu_marker += mu_dim2;
            }
            *resp_current = resp_value;
            resp_current++;
        }
    }
    return NO_ERROR;
}


int getWeightedCountCExt_single_thread(uint8_t *x, double *wcount, double *resp,
                   int wc_dim0, int wc_dim1, int wc_dim2,
                   int x_dim0, int x_dim1){
    int i, j, k, wcount_row;
    uint8_t *x_current;
    double *resp_current, *wcount_current, *wcount_marker;
    double resp_value;
    int wcount_row_size = wc_dim1 * wc_dim2;

    for (k=0; k < wc_dim0; k++){
        x_current = x;
        resp_current = resp + k * x_dim0;
        wcount_row = k * wcount_row_size;
        for (i=0; i < x_dim0; i++){
            resp_value = *resp_current;
            wcount_marker = wcount + wcount_row;
            for (j=0; j < x_dim1; j++){
                wcount_current = wcount_marker + *x_current;
                *wcount_current += resp_value;
                x_current++;
                wcount_marker += wc_dim2;
            }
            resp_current++;
        }
    }
    return NO_ERROR;
}




void getHardCountCExt_single_thread(uint8_t *x,
                   uint32_t *cluster_stats,
                   uint32_t *assignments,
                   int stats_dim0, int stats_dim1,
                   int stats_dim2, int x_dim0,
                   int x_dim1){
    int i, j;
    uint8_t *x_current;
    uint32_t *cluster_stat;
    int stats_row_size = stats_dim1 * stats_dim2;

    x_current = x;
    for (i=0; i < x_dim0; i++){
        cluster_stat = cluster_stats + assignments[i] * stats_row_size;
        for (j=0; j < x_dim1; j++){
            cluster_stat[*x_current] += 1;
            x_current++;
            cluster_stat += stats_dim2;
        }
    }
}
