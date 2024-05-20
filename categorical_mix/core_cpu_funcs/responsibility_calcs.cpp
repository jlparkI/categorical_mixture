/*!
 * # responsibility_calcs.cpp
 *
 * Perform the key steps in generating responsibilities
 * (the E-step in the EM algorithm). Also calculates
 * statistics of all the sequences assigned to each cluster
 * for the "hard assignment statistics" routine.
 *
 * + getProbsCExt_main
 * Updates the responsibilities array using multithreading.
 *
 * + getProbsCExt_worker
 * A single thread of the updates launched by
 * getProbsCExt_main
 *
 * + getProbsCExt_single_thread
 * A single thread version of getProbsCExt_main.
 *
 * + getProbsCExt_terminal_masked_main
 * Updates the responsibilities array using multithreading,
 * but with n- and c-terminal masking supplied by caller.
 *
 * + getProbsCExt_terminal_masked_worker
 * A single thread of the updates launched by
 * getProbsCExt_terminal_masked_main
 *
 * + getProbsCExt_gapped_main
 * Updates the responsibilities array using multithreading,
 * but ignoring any element of x where x[i]==20 (corresponding
 * to gaps in amino acid sequences).
 *
 * + getProbsCExt_gapped_worker
 * A single thread of the updates launched by
 * getProbsCExt_masked_main
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <thread>
#include "responsibility_calcs.h"

#define MEMORY_ERROR 1
#define NO_ERROR 0
#define THREAD_ERROR 2


/*!
 * # getProbsCExt_main
 *
 * Calculates the updated responsibilities (the E-step
 * in the EM algorithm) for a batch of input data.
 * This function does not do any bounds checking, so it
 * is important for caller to do so. This function
 * is multithreaded and divides the work up into groups
 * of clusters each thread will handle.
 *
 * ## Args
 *
 * + `x` Pointer to the first element of the array x containing
 * input data. Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `mu` The current set of parameters of the model, in a (K x C x D)
 * array for K clusters, C sequence length, D options per sequence
 * element.
 * `resp` The (K x N) array of cluster responsibilities, for K clusters
 * and N datapoints.
 * `mu_dim0` shape[0] of mu
 * `mu_dim1` shape[1] of mu
 * `mu_dim2` shape[2] of mu
 * `x_dim0` shape 0 of x
 * `x_dim1` shape 1 of x
 * `n_threads` Number of threads to launch.
 *
 * All operations are in place, nothing is returned.
 */
int getProbsCExt_main(uint8_t *x, double *mu, double *resp,
                   int mu_dim0, int mu_dim1, int mu_dim2,
                   int x_dim0, int x_dim1,
                   int n_threads){
    int startRow, endRow;
    int chunkSize = (mu_dim0 + n_threads - 1) / n_threads;
    std::vector<std::thread> threads(n_threads);

    if (n_threads > x_dim0)
        n_threads = x_dim0;

    for (int i=0; i < n_threads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > mu_dim0)
            endRow = mu_dim0;
        threads[i] = std::thread(&getProbsCExt_worker,
                x, resp, mu, mu_dim1, mu_dim2,
                x_dim0, x_dim1, startRow, endRow);
    }

    for (auto& th : threads)
        th.join();

    return NO_ERROR;
}


/*!
 * # getProbsCExt_worker
 *
 * Performs the E-step responsibility calculations for a subset
 * of the K available clusters.
 *
 * ## Args
 *
 * + `x` Pointer to the first element of the array x containing
 * input data. Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `resp` The (K x N) array of cluster responsibilities, for K clusters
 * and N datapoints.
 * `mu` The current set of parameters of the model, in a (K x C x D)
 * array for K clusters, C sequence length, D options per sequence
 * element.
 * `mu_dim1` shape[1] of mu
 * `mu_dim2` shape[2] of mu
 * `x_dim0` shape 0 of x
 * `x_dim1` shape 1 of x
 * `startRow` The first row of the resp and mu arrays to use for this
 * thread since this thread will only update for some clusters.
 * `endRow` The last row of the resp and mu arrays to use for this
 * thread.
 *
 * All operations are in place, nothing is returned.
 */
void *getProbsCExt_worker(uint8_t *x, double *resp,
        double *mu, int mu_dim1, int mu_dim2,
        int x_dim0, int x_dim1, int startRow,
        int endRow){
    int i, j, k, mu_row;
    uint8_t *x_current;
    double *resp_current, *mu_current, *mu_marker;
    double resp_value;
    int mu_row_size = mu_dim1 * mu_dim2;

    for (k=startRow; k < endRow; k++){
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
    return NULL;
}


/*!
 * # getProbsCExt_terminal_masked_main
 *
 * Calculates the updated responsibilities (the E-step
 * in the EM algorithm) for a batch of input data,
 * but with masking so that some n- and c-terminal columns
 * are excluded.
 * This function does not do any bounds checking, so it
 * is important for caller to do so. This function
 * is multithreaded and divides the work up into groups
 * of clusters each thread will handle.
 *
 * ## Args
 *
 * + `x` Pointer to the first element of the array x containing
 * input data. Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `mu` The current set of parameters of the model, in a (K x C x D)
 * array for K clusters, C sequence length, D options per sequence
 * element.
 * `resp` The (K x N) array of cluster responsibilities, for K clusters
 * and N datapoints.
 * `mu_dim0` shape[0] of mu
 * `mu_dim1` shape[1] of mu
 * `mu_dim2` shape[2] of mu
 * `x_dim0` shape 0 of x
 * `x_dim1` shape 1 of x
 * `n_threads` Number of threads to launch.
 * `startCol` the first column (all previous are masked).
 * `endCol` the last column (all previous are masked).
 *
 * All operations are in place, nothing is returned.
 */
int getProbsCExt_terminal_masked_main(uint8_t *x, double *mu, double *resp,
                   int mu_dim0, int mu_dim1, int mu_dim2,
                   int x_dim0, int x_dim1,
                   int n_threads, int startCol, int endCol){
    int startRow, endRow;
    int chunkSize = (mu_dim0 + n_threads - 1) / n_threads;
    std::vector<std::thread> threads(n_threads);

    if (n_threads > x_dim0)
        n_threads = x_dim0;

    for (int i=0; i < n_threads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > mu_dim0)
            endRow = mu_dim0;
        threads[i] = std::thread(&getProbsCExt_terminal_masked_worker,
                x, resp, mu, mu_dim1, mu_dim2,
                x_dim0, x_dim1, startRow, endRow,
                startCol, endCol);
    }

    for (auto& th : threads)
        th.join();

    return NO_ERROR;
}


/*!
 * # getProbsCExt_terminal_masked_worker
 *
 * Performs the E-step responsibility calculations for a subset
 * of the K available clusters using a "mask" to exclude some
 * columns at the start and end of the sequence.
 *
 * ## Args
 *
 * + `x` Pointer to the first element of the array x containing
 * input data. Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `resp` The (K x N) array of cluster responsibilities, for K clusters
 * and N datapoints.
 * `mu` The current set of parameters of the model, in a (K x C x D)
 * array for K clusters, C sequence length, D options per sequence
 * element.
 * `mu_dim1` shape[1] of mu
 * `mu_dim2` shape[2] of mu
 * `x_dim0` shape 0 of x
 * `x_dim1` shape 1 of x
 * `startRow` The first row of the resp and mu arrays to use for this
 * thread since this thread will only update for some clusters.
 * `endRow` The last row of the resp and mu arrays to use for this
 * thread.
 * `startCol` the first column (all previous are masked).
 * `endCol` the last column (all previous are masked).
 *
 * All operations are in place, nothing is returned.
 */
void *getProbsCExt_terminal_masked_worker(uint8_t *x, double *resp,
        double *mu, int mu_dim1, int mu_dim2,
        int x_dim0, int x_dim1, int startRow,
        int endRow, int startCol, int endCol){
    int i, j, k, mu_row;
    uint8_t *x_current;
    double *resp_current, *mu_current, *mu_marker;
    double resp_value;
    int mu_row_size = mu_dim1 * mu_dim2;

    for (k=startRow; k < endRow; k++){
        resp_current = resp + k * x_dim0;
        mu_row = k * mu_row_size;
        for (i=0; i < x_dim0; i++){
            resp_value = 0;
            mu_marker = mu + mu_row + startCol * mu_dim2;
            x_current = x + i * x_dim1 + startCol;
            for (j=startCol; j < endCol; j++){
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



/*!
 * # getProbsCExt_gapped_main
 *
 * Calculates the updated responsibilities (the E-step
 * in the EM algorithm) for a batch of input data,
 * but ignoring gaps (cases where x[i]==20).
 * This function does not do any bounds checking, so it
 * is important for caller to do so. This function
 * is multithreaded and divides the work up into groups
 * of clusters each thread will handle.
 *
 * ## Args
 *
 * + `x` Pointer to the first element of the array x containing
 * input data. Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `mu` The current set of parameters of the model, in a (K x C x D)
 * array for K clusters, C sequence length, D options per sequence
 * element.
 * `resp` The (K x N) array of cluster responsibilities, for K clusters
 * and N datapoints.
 * `mu_dim0` shape[0] of mu
 * `mu_dim1` shape[1] of mu
 * `mu_dim2` shape[2] of mu
 * `x_dim0` shape 0 of x
 * `x_dim1` shape 1 of x
 * `n_threads` Number of threads to launch.
 * `startCol` the first column (all previous are masked).
 * `endCol` the last column (all previous are masked).
 *
 * All operations are in place, nothing is returned.
 */
int getProbsCExt_gapped_main(uint8_t *x, double *mu, double *resp,
                   int mu_dim0, int mu_dim1, int mu_dim2,
                   int x_dim0, int x_dim1,
                   int n_threads){
    int startRow, endRow;
    int chunkSize = (mu_dim0 + n_threads - 1) / n_threads;
    std::vector<std::thread> threads(n_threads);

    if (n_threads > x_dim0)
        n_threads = x_dim0;

    for (int i=0; i < n_threads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > mu_dim0)
            endRow = mu_dim0;
        threads[i] = std::thread(&getProbsCExt_gapped_worker,
                x, resp, mu, mu_dim1, mu_dim2,
                x_dim0, x_dim1, startRow, endRow);
    }

    for (auto& th : threads)
        th.join();

    return NO_ERROR;
}


/*!
 * # getProbsCExt_gapped_worker
 *
 * Performs the E-step responsibility calculations for a subset
 * of the K available clusters excluding all gaps.
 *
 * ## Args
 *
 * + `x` Pointer to the first element of the array x containing
 * input data. Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `resp` The (K x N) array of cluster responsibilities, for K clusters
 * and N datapoints.
 * `mu` The current set of parameters of the model, in a (K x C x D)
 * array for K clusters, C sequence length, D options per sequence
 * element.
 * `mu_dim1` shape[1] of mu
 * `mu_dim2` shape[2] of mu
 * `x_dim0` shape 0 of x
 * `x_dim1` shape 1 of x
 * `startRow` The first row of the resp and mu arrays to use for this
 * thread since this thread will only update for some clusters.
 * `endRow` The last row of the resp and mu arrays to use for this
 * thread.
 * `startCol` the first column (all previous are masked).
 * `endCol` the last column (all previous are masked).
 *
 * All operations are in place, nothing is returned.
 */
void *getProbsCExt_gapped_worker(uint8_t *x, double *resp,
        double *mu, int mu_dim1, int mu_dim2,
        int x_dim0, int x_dim1, int startRow,
        int endRow){
    int i, j, k, mu_row;
    uint8_t *x_current;
    double *resp_current, *mu_current, *mu_marker;
    double resp_value;
    int mu_row_size = mu_dim1 * mu_dim2;

    for (k=startRow; k < endRow; k++){
        resp_current = resp + k * x_dim0;
        x_current = x;
        mu_row = k * mu_row_size;
        for (i=0; i < x_dim0; i++){
            resp_value = 0;
            mu_marker = mu + mu_row;
            for (j=0; j < x_dim1; j++){
                if (*x_current == 20){
                    x_current++;
                    mu_marker += mu_dim2;
                    continue;
                }
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


/*!
 * # getProbsCExt_single_thread
 *
 * Calculates the updated responsibilities (the E-step
 * in the EM algorithm) for a batch of input data.
 * This function does not do any bounds checking, so it
 * is important for caller to do so. This function
 * is a single-threaded version; calling the multi-threaded
 * version if only a single thread is needed would incur
 * extra overhead.
 *
 * ## Args
 *
 * + `x` Pointer to the first element of the array x containing
 * input data. Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `mu` The current set of parameters of the model, in a (K x C x D)
 * array for K clusters, C sequence length, D options per sequence
 * element.
 * `resp` The (K x N) array of cluster responsibilities, for K clusters
 * and N datapoints.
 * `mu_dim0` shape[0] of mu
 * `mu_dim1` shape[1] of mu
 * `mu_dim2` shape[2] of mu
 * `x_dim0` shape 0 of x
 * `x_dim1` shape 1 of x
 *
 * All operations are in place, nothing is returned.
 */
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




/*!
 * # getHardCountCExt_single_thread
 *
 * Calculates the updated responsibilities (the E-step
 * in the EM algorithm) for a batch of input data.
 * This function does not do any bounds checking, so it
 * is important for caller to do so. This function
 * is a single-threaded version; calling the multi-threaded
 * version if only a single thread is needed would incur
 * extra overhead.
 *
 * ## Args
 *
 * + `x` Pointer to the first element of the array x containing
 * input data. Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `cluster_stats` Pointer to first element of the cluster stats
 * array that will be updated with the statistics of datapoints assigned
 * to each cluster.
 * `assignments` Pointer to first element of (N)-shaped array indicating
 * the cluster to which each datapoint has been assigned.
 * `stats_dim0` shape[0] of cluster_stats
 * `stats_dim1` shape[1] of cluster_stats
 * `stats_dim2` shape[2] of cluster_stats
 * `x_dim0` shape 0 of x
 * `x_dim1` shape 1 of x
 *
 * All operations are in place, nothing is returned.
 */
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
