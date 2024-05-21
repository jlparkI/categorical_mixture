/*!
 * # responsibility_calcs.cpp
 *
 * Perform the key steps in generating responsibilities
 * (the E-step in the EM algorithm) and scoring.
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <thread>
#include <iostream>
#include "responsibility_calcs.h"


#define NO_ERROR 0
#define ARRAY_SIZING_ERROR 1
#define ARRAY_TYPE_ERROR 2




/*!
 * # getProbsCExt
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
 * `n_threads` Number of threads to launch.
 *
 * All operations are in place, nothing is returned.
 */
void getProbsCExt(py::array_t<uint8_t, py::array::c_style> x, 
        py::array_t<double, py::array::c_style> mu,
        py::array_t<double, py::array::c_style> resp,
        py::ssize_t n_threads){

    // Error checking. For now, not very sophisticated -- if
    // array inputs are unacceptable, raise an exception that
    // (since this function is python-wrapped) will be handled by
    // the wrapper.
    int errcode = respSafetyChecks(x, mu, resp, n_threads);
    if (errcode != NO_ERROR){
        throw std::runtime_error(std::string("Incompatible array shapes / data types "
                    "passed to a responsibility calculation function in the cpp ext."));
    }

    if (n_threads > mu.shape(0))
        n_threads = mu.shape(0);
    if (n_threads > x.shape(0))
        n_threads = x.shape(0);

    int nClusters = mu.shape(0), seqLen = mu.shape(1), muDim2 = mu.shape(2);
    int ndatapoints = x.shape(0);
    int startRow, endRow;
    uint8_t *xref = (uint8_t*) x.data();
    double *muref = (double*) mu.data(), *respref = (double*) resp.data();

    int chunkSize = (mu.shape(0) + n_threads - 1) / n_threads;
    std::vector<std::thread> threads(n_threads);

    for (int i=0; i < n_threads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > mu.shape(0))
            endRow = mu.shape(0);

        threads[i] = std::thread(&getProbsCExt_worker,
                xref, respref, muref,
                startRow, endRow, nClusters, seqLen,
                muDim2, ndatapoints);
    }

    for (auto& th : threads)
        th.join();
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
 * `startRow` The first row of the resp and mu arrays to use for this
 * thread since this thread will only update for some clusters.
 * `endRow` The last row of the resp and mu arrays to use for this
 * thread.
 * `nClusters` dim0 of mu.
 * `seqLen` dim1 of mu.
 * `muDim2` dim2 of mu.
 * `ndatapoints` number of datapoints, dim0 of x.
 *
 * All operations are in place, nothing is returned.
 */
void *getProbsCExt_worker(uint8_t *x, double *resp,
            double *mu, int startRow, int endRow,
            int nClusters, int seqLen, int muDim2,
            int ndatapoints){        

    double *resp_current, *mu_current, *mu_marker;
    uint8_t *x_current;
    int mu_row, mu_row_size = seqLen * muDim2;
    double resp_value;

    for (int k=startRow; k < endRow; k++){
        resp_current = resp + k * ndatapoints;
        x_current = x;
        mu_row = k * mu_row_size;

        for (int i=0; i < ndatapoints; i++){
            resp_value = 0;
            mu_marker = mu + mu_row;

            for (int j=0; j < seqLen; j++){
                mu_current = mu_marker + *x_current;
                resp_value += *mu_current;
                x_current++;
                mu_marker += muDim2;
            }
            *resp_current = resp_value;
            resp_current++;
        }
    }
    return NULL;
}


/*!
 * # mask_terminal_deletions
 *
 * Converts all gaps at the N- and C-terminal ends of each sequence
 * into value 255. This can then be passed to a masked scoring
 * function. The operation is performed in place. Once this
 * conversion has been performed, the result should not under
 * any circumstances be passed to a non-masked scoring function
 * since this may lead to erroneous scoring.
 *
 * ## Args
 *
 * + `x` Pointer to the first element of the array x containing
 * input data. Should be an (N x C) array for N datapoints, sequence
 * length C. Each element indicates the item chosen at that position
 * in the raw data.
 *
 * All operations are in place, nothing is returned.
 */
void mask_terminal_deletions(py::array_t<uint8_t, py::array::c_style> x){
    int nDatapoints = x.shape(0);
    int nCols = x.shape(1);
    uint8_t *xref = (uint8_t*) x.data();

    for (int i=0; i < nDatapoints; i++){
        for (int k=0; k < nCols; k++){
            if (xref[k] != 20)
                break;
            xref[k] = 255;
        }
        for (int k=nCols - 1; k > 0; k--){
            if (xref[k] != 20)
                break;
            xref[k] = 255;
        }
        xref += nCols;
    }
}





/*!
 * # getProbsCExt_masked
 *
 * Calculates the updated responsibilities (the E-step
 * in the EM algorithm) for a batch of input data,
 * but ignoring masked positions (cases where x[i] = 255).
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
 * `n_threads` Number of threads to launch.
 * `startCol` the first column (all previous are masked).
 * `endCol` the last column (all previous are masked).
 *
 * All operations are in place, nothing is returned.
 */
void getProbsCExt_masked(py::array_t<uint8_t, py::array::c_style> x, 
        py::array_t<double, py::array::c_style> mu,
        py::array_t<double, py::array::c_style> resp,
        int n_threads){
    // Error checking. For now, not very sophisticated -- if
    // array inputs are unacceptable, raise an exception that
    // (since this function is python-wrapped) will be handled by
    // the wrapper.
    int errcode = respSafetyChecks(x, mu, resp, n_threads);
    if (errcode != NO_ERROR){
        throw std::runtime_error(std::string("Incompatible array shapes / data types "
                    "passed to a responsibility calculation function in the cpp ext."));
    }

    if (n_threads > x.shape(0))
        n_threads = x.shape(0);
    if (n_threads > x.shape(0))
        n_threads = x.shape(0);

    int nClusters = mu.shape(0), seqLen = mu.shape(1), muDim2 = mu.shape(2);
    int ndatapoints = x.shape(0);
    int startRow, endRow;
    uint8_t *xref = (uint8_t*) x.data();
    double *muref = (double*) mu.data(), *respref = (double*) resp.data();

    int chunkSize = (mu.shape(0) + n_threads - 1) / n_threads;
    std::vector<std::thread> threads(n_threads);


    for (int i=0; i < n_threads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > mu.shape(0))
            endRow = mu.shape(0);
        threads[i] = std::thread(&getProbsCExt_masked_worker,
                xref, respref, muref, startRow, endRow,
                nClusters, seqLen, muDim2, ndatapoints);
    }

    for (auto& th : threads)
        th.join();
}


/*!
 * # getProbsCExt_masked_worker
 *
 * Performs the E-step responsibility calculations for a subset
 * of the K available clusters excluding masked positions.
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
 * `startRow` The first row of the resp and mu arrays to use for this
 * thread since this thread will only update for some clusters.
 * `endRow` The last row of the resp and mu arrays to use for this
 * thread.
 * `nClusters` dim0 of mu.
 * `seqLen` dim1 of mu.
 * `muDim2` dim2 of mu.
 * `ndatapoints` dim0 of x.
 *
 * All operations are in place, nothing is returned.
 */
void *getProbsCExt_masked_worker(uint8_t *x, double *resp,
        double *mu, int startRow, int endRow,
        int nClusters, int seqLen, int muDim2,
        int ndatapoints){
    int i, j, k, mu_row;
    uint8_t *x_current;
    double *resp_current, *mu_current, *mu_marker;
    double resp_value;
    int mu_row_size = seqLen * muDim2;

    for (k=startRow; k < endRow; k++){
        resp_current = resp + k * ndatapoints;
        x_current = x;
        mu_row = k * mu_row_size;

        for (i=0; i < ndatapoints; i++){
            resp_value = 0;
            mu_marker = mu + mu_row;

            for (j=0; j < seqLen; j++){
                if (*x_current == 255){
                    x_current++;
                    mu_marker += muDim2;
                    continue;
                }
                mu_current = mu_marker + *x_current;
                resp_value += *mu_current;
                x_current++;
                mu_marker += muDim2;
            }
            *resp_current = resp_value;
            resp_current++;
        }
    }
    return NULL;
}





/*!
 * # respSafetyChecks
 *
 * Checks input arrays for correctness and compatibility.
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
 * `n_threads` Number of threads to launch.
 *
 * ## Returns
 *
 * + `errcode` An integer indicating the type of error if any.
 */
int respSafetyChecks(py::array_t<uint8_t, py::array::c_style> x,
        py::array_t<double, py::array::c_style> mu,
        py::array_t<double, py::array::c_style> resp,
        int n_threads){
    if (mu.ndim() != 3 || x.ndim() != 2 || resp.ndim() != 2)
        return ARRAY_SIZING_ERROR;

    if (x.shape(0) != resp.shape(1))
        return ARRAY_SIZING_ERROR;

    if (resp.shape(0) != mu.shape(0))
        return ARRAY_SIZING_ERROR;

    if (x.shape(1) != mu.shape(1))
        return ARRAY_SIZING_ERROR;

    return NO_ERROR;
}
