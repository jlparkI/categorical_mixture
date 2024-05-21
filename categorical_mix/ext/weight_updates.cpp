/*!
 * # weighted_counts.cpp
 *
 * Perform the key steps involved in generating weighted counts
 * for the M-step in the EM algorithm.
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <thread>
#include "weight_updates.h"

#define MEMORY_ERROR 1
#define NO_ERROR 0
#define THREAD_ERROR 2
#define ARRAY_SIZING_ERROR 3



/*!
 * # getWeightedCountCExt
 *
 * Updates the wcount array containing the responsibility-
 * weighted counts. In the M step of EM optimization, this
 * array will become the new mu values.
 *
 * ## Args
 *
 * + `x` Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `wcount` Array is of shape (K x C x D) for K clusters,
 * C sequence length, D options per sequence element.
 * `resp` The (K x N) array of cluster responsibilities, for K clusters
 * and N datapoints.
 * `n_threads` Number of threads to launch.
 *
 * All operations are in place, nothing is returned.
 */
void getWeightedCountCExt(py::array_t<uint8_t, py::array::c_style> x,
        py::array_t<double, py::array::c_style> wcount,
        py::array_t<double, py::array::c_style> resp,
        int n_threads){
    // Error checking. For now, not very sophisticated -- if
    // array inputs are unacceptable, raise an exception that
    // (since this function is python-wrapped) will be handled by
    // the wrapper.
    int errcode = wcountSafetyChecks(x, wcount, resp, n_threads);
    if (errcode != NO_ERROR){
        throw std::runtime_error(std::string("Incompatible array shapes / data types "
                    "passed to a weight update function in the cpp ext."));
    }

    int wcount_dim0 = wcount.shape(0), wcount_dim1 = wcount.shape(1), wcount_dim2 = wcount.shape(2);
    int ndatapoints = x.shape(0), x_dim1 = x.shape(1);
    uint8_t *xref = (uint8_t*) x.data();
    double *wcref = (double*) wcount.data(), *respref = (double*) resp.data();

    int startRow, endRow;
    int chunkSize = (wcount_dim0 + n_threads - 1) / n_threads;
    std::vector<std::thread> threads(n_threads);

    if (n_threads > ndatapoints)
        n_threads = ndatapoints;

    for (int i=0; i < n_threads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > wcount_dim0)
            endRow = wcount_dim0;
        threads[i] = std::thread(&getWeightedCountCExt_worker,
                xref, respref, wcref, wcount_dim1, wcount_dim2,
                ndatapoints, x_dim1, startRow, endRow);
    }

    for (auto& th : threads)
        th.join();
}



/*!
 * # getWeightedCountCExt_worker
 *
 * Updates the wcount array containing the responsibility-
 * weighted counts. It only performs this operation on
 * clusters in (startRow, endRow); each thread is assigned
 * to complete some subset of the total clusters.
 *
 * ## Args
 *
 * + `x` Pointer to the first element of the array x containing
 * input data. Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `resp` The (K x N) array of cluster responsibilities, for K clusters
 * and N datapoints.
 * `wcount` Pointer to first element of weighted count array; will
 * be updated in place.
 * `wcount_dim1` shape[1] of wcount
 * `wcount_dim2` shape[2] of wcount
 * `x_dim0` shape 0 of x
 * `x_dim1` shape 1 of x
 * `startRow` The first row of wcount to update for this thread.
 * `endRow` The last row of wcount to update for this thread.
 *
 * All operations are in place, nothing is returned.
 */
void *getWeightedCountCExt_worker(uint8_t *x, double *resp,
        double *wcount, int wcount_dim1, int wcount_dim2,
        int x_dim0, int x_dim1, int startRow, int endRow){

    int i, j, k, wcount_row;
    uint8_t *x_current;
    double *resp_current, *wcount_current, *wcount_marker;
    double resp_value;
    int wcount_row_size = wcount_dim1 * wcount_dim2;


    for (k=startRow; k < endRow; k++){
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
                wcount_marker += wcount_dim2;
            }
            resp_current++;
        }
    }
    return NULL;
}



/*!
 * # wcountSafetyChecks
 *
 * Checks input arrays for correctness and compatibility.
 *
 * ## Args
 *
 * + `x` Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 * `wcount` Array is of shape (K x C x D) for K clusters,
 * C sequence length, D options per sequence element.
 * `resp` The (K x N) array of cluster responsibilities, for K clusters
 * and N datapoints.
 * `n_threads` Number of threads to launch.
 *
 * ## Returns
 *
 * + `errcode` An integer indicating the type of error if any.
 */
int wcountSafetyChecks(py::array_t<uint8_t, py::array::c_style> x,
        py::array_t<double, py::array::c_style> wcount,
        py::array_t<double, py::array::c_style> resp,
        int n_threads){
    if (wcount.ndim() != 3 || x.ndim() != 2 || resp.ndim() != 2)
        return ARRAY_SIZING_ERROR;

    if (x.shape(0) != resp.shape(1))
        return ARRAY_SIZING_ERROR;

    if (resp.shape(0) != wcount.shape(0))
        return ARRAY_SIZING_ERROR;

    if (x.shape(1) != wcount.shape(1))
        return ARRAY_SIZING_ERROR;

    return NO_ERROR;
}
