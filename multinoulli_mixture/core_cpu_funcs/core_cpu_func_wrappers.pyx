"""Contains the wrapper code for the C extension which performs
the responsibility and weighted count calculations used
in the EM algorithm and in making predictions. The wrapper
performs all of the major steps in EM for fitting with the
help of the C extensions.

Read the docstrings before using the functions in this file
outside of the Python wrapper class."""
import os
import numpy as np
cimport cython
cimport numpy as np
from scipy.special import logsumexp
from libc.stdint cimport uint8_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uintptr_t

cdef float MINIMUM_PROB_VAL = 1e-14


cdef extern from "responsibility_calcs.h" nogil:
    int getProbsCExt_main(uint8_t *x,
                   double *mu, double *resp,
                   int mu_dim0, int mu_dim1,
                   int mu_dim2, int x_dim0,
                   int x_dim1, int n_threads)
    int getProbsCExt_single_thread(uint8_t *x,
                    double *mu, double *resp,
                    int mu_dim0, int mu_dim1, int mu_dim2,
                    int x_dim0, int x_dim1);
    void getHardCountCExt_single_thread(uint8_t *x,
                   uint32_t *cluster_stats,
                   uint32_t *assignments,
                   int stats_dim0, int stats_dim1,
                   int stats_dim2, int x_dim0,
                   int x_dim1); 

cdef extern from "weighted_count_calcs.h" nogil:
    int getWeightedCountCExt_main(uint8_t *x,
                   double *wcount, double *resp,
                   int wcount_dim0, int wcount_dim1,
                   int wcount_dim2, int x_dim0,
                   int x_dim1, int n_threads) 
    int getWeightedCountCExt_single_thread(uint8_t *x,
                   double *wcount, double *resp,
                   int wc_dim0, int wc_dim1, int wc_dim2,
                   int x_dim0, int x_dim1);

    
@cython.boundscheck(False)
@cython.wraparound(False)
def em_offline(list xfiles, np.ndarray[np.float64_t, ndim=1] mix_weights, 
                np.ndarray[np.float64_t, ndim=3] mu,
                int n_threads):
    """Runs a single iteration of the EM algorithm for a specified list
    of .npy files on disk. The files should all be of type np.uint8 and
    should not contain any values larger than the number of possible
    items per position (mu.shape[2]). If these conditions are violated,
    a segfault may occur. The Python wrapper checks all input data to
    ensure it meets these conditions. If you decide to use this function
    OUTSIDE the Python wrapper, you must implement these checks yourself.
    For performance reasons, it is not desirable to perform these checks
    on every iteration over the dataset.

    Args:
        xfiles (list): A list of .npy files of type np.uint8 with input
            data. Should already have been checked for acceptability.
        mix_weights (np.ndarray): The mixture weights for each component.
        mu (np.ndarray): The model parameters (probability of each choice
            at each position). Shape is (n_components, sequence_length,
            num_possible_items).
        n_threads (int): The number of threads to use.

    Returns:
        new_weights (np.ndarray): The updated mixture weights.
        lb (float): The lower bound.
        rik_counts (np.ndarray): An array of same shape as mu;
            the updated mu array.
        ndpoints (int): The number of datapoints in this file list.
    """
    cdef np.ndarray[np.float64_t, ndim=3] rik_counts = \
            np.zeros((mu.shape[0], mu.shape[1], mu.shape[2]))
    cdef np.ndarray[np.float64_t, ndim=2] log_mixweights = \
            np.log(mix_weights.clip(min=MINIMUM_PROB_VAL))[:,None]
    cdef np.ndarray[np.float64_t, ndim=1] new_weights = np.zeros((mix_weights.shape[0]))
    cdef np.ndarray[np.uint8_t, ndim=2] x
    cdef np.ndarray[np.float64_t, ndim=1] lnorm
    cdef np.ndarray[np.float64_t, ndim=2] resp
    cdef int errcode
    cdef float lb = 0
    cdef int ndpoints = 0
    cdef float net_resp = 0

    if mu.shape[0] != mix_weights.shape[0]:
        raise ValueError("Inputs to wrapped C++ function have incorrect shapes.")

    mu[mu<MINIMUM_PROB_VAL] = MINIMUM_PROB_VAL
    mu = np.log(mu)

    for xfile in xfiles:
        x = np.load(xfile)
        resp = np.zeros((mu.shape[0], x.shape[0]))
        lnorm = np.zeros((x.shape[0]))
            
        errcode = getProbsCExt_main(&x[0,0], &mu[0,0,0], &resp[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1],
                        n_threads)
        resp += log_mixweights
        lnorm[:] = logsumexp(resp, axis=0)
        with np.errstate(under="ignore"):
            resp[:] = np.exp(resp - lnorm[None,:])
        lb += lnorm.sum()
        ndpoints += x.shape[0]

        rsum = resp.sum(axis=1)
        new_weights += rsum
        net_resp += rsum.sum()

        errcode = getWeightedCountCExt_main(&x[0,0],
                    &rik_counts[0,0,0], &resp[0,0],
                    rik_counts.shape[0], rik_counts.shape[1], 
                    rik_counts.shape[2], x.shape[0], x.shape[1],
                    n_threads)
    return new_weights, lb, rik_counts, net_resp, ndpoints



@cython.boundscheck(False)
@cython.wraparound(False)
def em_online(np.ndarray[np.uint8, ndim=2] x,
            np.ndarray[np.float64_t, ndim=1] mix_weights, 
                np.ndarray[np.float64_t, ndim=3] mu,
                int n_threads):
    """Runs a single iteration of the EM algorithm for a single array of
    input data. The data should be of type np.uint8 and
    should not contain any values larger than the number of possible
    items per position (mu.shape[2]). If these conditions are violated,
    a segfault may occur. The Python wrapper checks all input data to
    ensure it meets these conditions. If you decide to use this function
    OUTSIDE the Python wrapper, you must implement these checks yourself.

    Args:
        x (np.ndarray): An array of type np.uint8 with input
            data. Should already have been checked for acceptability.
        mix_weights (np.ndarray): The mixture weights for each component.
        mu (np.ndarray): The model parameters (probability of each choice
            at each position). Shape is (n_components, sequence_length,
            num_possible_items).
        n_threads (int): The number of threads to use.

    Returns:
        new_weights (np.ndarray): The updated mixture weights.
        lb (float): The lower bound.
        rik_counts (np.ndarray): An array of same shape as mu;
            the updated mu array.
        ndpoints (int): The number of datapoints in this file list.
    """
    cdef np.ndarray[np.float64_t, ndim=3] rik_counts = \
            np.zeros((mu.shape[0], mu.shape[1], mu.shape[2]))
    cdef np.ndarray[np.float64_t, ndim=2] log_mixweights = \
            np.log(mix_weights.clip(min=MINIMUM_PROB_VAL))[:,None]
    cdef np.ndarray[np.float64_t, ndim=1] new_weights = np.zeros((mix_weights.shape[0]))
    cdef np.ndarray[np.uint8_t, ndim=2] x
    cdef np.ndarray[np.float64_t, ndim=1] lnorm
    cdef np.ndarray[np.float64_t, ndim=2] resp
    cdef int errcode
    cdef float lb = 0
    cdef int ndpoints = 0
    cdef float net_resp = 0
        
    mu[mu<MINIMUM_PROB_VAL] = MINIMUM_PROB_VAL
    mu = np.log(mu)

    if mu.shape[0] != mix_weights.shape[0]:
        raise ValueError("Inputs to wrapped C++ function have incorrect shapes.")

    x = np.load(xfile)
    resp = np.zeros((mu.shape[0], x.shape[0]))
    lnorm = np.zeros((x.shape[0]))
            
    errcode = getProbsCExt_main(&x[0,0], &mu[0,0,0], &resp[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1],
                        n_threads)
    resp += log_mixweights
    lnorm[:] = logsumexp(resp, axis=0)
    with np.errstate(under="ignore"):
        resp[:] = np.exp(resp - lnorm[None,:])
    lb += lnorm.sum()
    ndpoints += x.shape[0]

    rsum = resp.sum(axis=1)
    new_weights += rsum
    net_resp += rsum.sum()

    errcode = getWeightedCountCExt_main(&x[0,0],
                    &rik_counts[0,0,0], &resp[0,0],
                    rik_counts.shape[0], rik_counts.shape[1], 
                    rik_counts.shape[2], x.shape[0], x.shape[1],
                    n_threads)
    return new_weights, lb, rik_counts, net_resp, ndpoints





@cython.boundscheck(False)
@cython.wraparound(False)
def hard_cluster_assign(list xfiles, 
                    np.ndarray[np.float64_t, ndim=3] mu,
                    np.ndarray[np.float64_t, ndim=1] mix_weights):
    """Loops over a list of files using mu and mix_weights for
    a fitted model, assigns each datapoint to one of the clusters,
    and compiles statistics on what the sequences assigned to each
    cluster look like. The data should be of type np.uint8 and
    should not contain any values larger than the number of possible
    items per position (mu.shape[2]). If these conditions are violated,
    a segfault may occur. The Python wrapper checks all input data to
    ensure it meets these conditions. If you decide to use this function
    OUTSIDE the Python wrapper, you must implement these checks yourself.

    Args:
        xfiles (list): A list of .npy files for which statistics are desired.
            Should already have been checked for acceptability.
        mu (np.ndarray): The model parameters (probability of each choice
            at each position). Shape is (n_components, sequence_length,
            num_possible_items).
        mix_weights (np.ndarray): The mixture weights for each component.

    Returns:
        cluster_stats (np.ndarray): The number of occurrences of each
            possibility. Same shape as mu. Dtype is uint32.
    """
    cdef np.ndarray[np.uint32_t, ndim=3] cluster_stats = \
            np.zeros((mu.shape[0], mu.shape[1], mu.shape[2]), dtype=np.uint32)
    cdef np.ndarray[np.float64_t, ndim=2] log_mixweights = \
            np.log(mix_weights.clip(min=MINIMUM_PROB_VAL))[:,None]
    cdef np.ndarray[np.uint8_t, ndim=2] x
    cdef np.ndarray[np.uint32_t, ndim=1] assignments
    cdef np.ndarray[np.float64_t, ndim=1] lnorm
    cdef np.ndarray[np.float64_t, ndim=2] resp
    cdef int errcode
        
    mu[mu<MINIMUM_PROB_VAL] = MINIMUM_PROB_VAL
    mu = np.log(mu)

    if mu.shape[0] != mix_weights.shape[0]:
        raise ValueError("Inputs to wrapped C++ function have incorrect shapes.")

    for xfile in xfiles:
        x = np.load(xfile)
        resp = np.zeros((mu.shape[0], x.shape[0]))
        assignments = np.zeros((x.shape[0]), dtype=np.uint32)
        lnorm = np.zeros((x.shape[0]))
            
        errcode = getProbsCExt_main(&x[0,0], &mu[0,0,0], &resp[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1],
                        n_threads)
        resp += log_mixweights
        lnorm[:] = logsumexp(resp, axis=0)
        with np.errstate(under="ignore"):
            resp[:] = np.exp(resp - lnorm[None,:])
        assignments[:] = resp.argmax(axis=0)

        getHardCountCExt_single_thread(&x[0,0],
                    &cluster_stats[0,0,0], &assignments[0],
                    cluster_stats.shape[0], cluster_stats.shape[1], 
                    cluster_stats.shape[2], x.shape[0], x.shape[1])
    return cluster_stats



def multimix_predict(np.ndarray[np.uint8_t, ndim=2] x,
                np.ndarray[np.float64_t, ndim=3] mu,
                np.ndarray[np.float64_t, ndim=1] mix_weights,
                int n_threads = 1):
    """Assigns each input datapoint in an array to one of the
    available clusters for a fitted model.

    The data should be of type np.uint8 and
    should not contain any values larger than the number of possible
    items per position (mu.shape[2]). If these conditions are violated,
    a segfault may occur. The Python wrapper checks all input data to
    ensure it meets these conditions. If you decide to use this function
    OUTSIDE the Python wrapper, you must implement these checks yourself.

    Args:
        x (np.ndarray): An array of type np.uint8 with input
            data. Should already have been checked for acceptability.
        mu (np.ndarray): The model parameters (probability of each choice
            at each position). Shape is (n_components, sequence_length,
            num_possible_items).
        mix_weights (np.ndarray): The mixture weights for each component.
        n_threads (int): The number of threads to use.

    Returns:
        cluster_assignments (np.ndarray): Cluster assignments for the
            inputs as an array of dtype uint32, shape (x.shape[0]).
    """
    cdef np.ndarray[np.float64_t, ndim=2] probs
    cdef np.ndarray[np.float64_t, ndim=1] log_mixweights
    cdef np.ndarray[np.uint32_t, ndim=1] cluster_assignments

    if mu.shape[0] != mix_weights.shape[0]:
        raise ValueError("Inputs to wrapped C++ function have incorrect shapes.")

    mu[mu<MINIMUM_PROB_VAL] = MINIMUM_PROB_VAL
    mu[:] = np.log(mu)
    log_mixweights = np.log(mix_weights.clip(min=MINIMUM_PROB_VAL))

    probs = np.zeros((mu.shape[0], x.shape[0]))
    errcode = getProbsCExt_main(&x[0,0], &mu[0,0,0], &probs[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1], n_threads)
    probs += log_mixweights[:,None]
    cluster_assignments = probs.argmax(axis=0)
    return cluster_assignments


def multimix_cluster_probs(np.ndarray[np.uint8_t, ndim=2] x,
                np.ndarray[np.float64_t, ndim=3] mu,
                np.ndarray[np.float64_t, ndim=1] mix_weights,
                int n_threads = 1,
                bint use_mixweights = True):
    """Determines the probability of each datapoint given each
    cluster.

    The data should be of type np.uint8 and
    should not contain any values larger than the number of possible
    items per position (mu.shape[2]). If these conditions are violated,
    a segfault may occur. The Python wrapper checks all input data to
    ensure it meets these conditions. If you decide to use this function
    OUTSIDE the Python wrapper, you must implement these checks yourself.

    Args:
        x (np.ndarray): An array of type np.uint8 with input
            data. Should already have been checked for acceptability.
        mu (np.ndarray): The model parameters (probability of each choice
            at each position). Shape is (n_components, sequence_length,
            num_possible_items).
        mix_weights (np.ndarray): The mixture weights for each component.
        n_threads (int): The number of threads to use.
        use_mixweights (bint): If True, take mixture weights into account.
            If False, then do not.

    Returns:
        probs (np.ndarray): The probability of each datapoint for
            each cluster as a float64 array of shape (n_components,
            x.shape[0]).
    """
    cdef np.ndarray[np.float64_t, ndim=2] probs
    cdef np.ndarray[np.float64_t, ndim=1] log_mixweights

    if mu.shape[0] != mix_weights.shape[0]:
        raise ValueError("Inputs to wrapped C++ function have incorrect shapes.")

    mu[mu<MINIMUM_PROB_VAL] = MINIMUM_PROB_VAL
    mu[:] = np.log(mu)
    log_mixweights = np.log(mix_weights.clip(min=MINIMUM_PROB_VAL))

    probs = np.zeros((mu.shape[0], x.shape[0]))
    errcode = getProbsCExt_main(&x[0,0], &mu[0,0,0], &probs[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1], n_threads)
    if use_mixweights:
        probs += log_mixweights[:,None]
    return probs

    
def multimix_loglik_offline(list xfiles,
        np.ndarray[np.float64_t, ndim=3] mu,
        np.ndarray[np.float64_t, ndim=1] mix_weights,
        int n_threads):
    """Determines the log likelihood of a dataset represented
    by an input list of .npy files.

    The data should be of type np.uint8 and
    should not contain any values larger than the number of possible
    items per position (mu.shape[2]). If these conditions are violated,
    a segfault may occur. The Python wrapper checks all input data to
    ensure it meets these conditions. If you decide to use this function
    OUTSIDE the Python wrapper, you must implement these checks yourself.

    Args:
        xfiles (list): A list of .npy files with input data.
            Should already have been checked for acceptability.
        mu (np.ndarray): The model parameters (probability of each choice
            at each position). Shape is (n_components, sequence_length,
            num_possible_items).
        mix_weights (np.ndarray): The mixture weights for each component.
        n_threads (int): The number of threads to use.

    Returns:
        loglik (float): The log-likelihood of the input dataset.
    """
    cdef np.ndarray[np.float64_t, ndim=2] resp
    cdef np.ndarray[np.float64_t, ndim=1] log_mixweights
    cdef np.ndarray[np.float64_t, ndim=1] probs
    cdef np.ndarray[np.uint8_t, ndim=2] x

    if mu.shape[0] != mix_weights.shape[0]:
        raise ValueError("Inputs to wrapped C++ function have incorrect shapes.")

    mu[mu<MINIMUM_PROB_VAL] = MINIMUM_PROB_VAL
    mu[:] = np.log(mu)
    log_mixweights = np.log(mix_weights.clip(min=MINIMUM_PROB_VAL))

    loglik = 0

    for xfile in xfiles:
        x = np.load(xfile)
        resp = np.zeros((mu.shape[0], x.shape[0]))
        probs = np.zeros((x.shape[0]))
        errcode = getProbsCExt_main(&x[0,0], &mu[0,0,0], &resp[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1], n_threads)
        resp += log_mixweights[:,None]
        probs[:] = logsumexp(resp, axis=0)
        loglik += probs.sum()
    return loglik


def multimix_score(np.ndarray[np.uint8_t, ndim=2] x,
        np.ndarray[np.float64_t, ndim=3] mu,
        np.ndarray[np.float64_t, ndim=1] mix_weights,
        int n_threads = 1):
    """Determines the log likelihood of each input datapoint
    in an array of input data.

    The data should be of type np.uint8 and
    should not contain any values larger than the number of possible
    items per position (mu.shape[2]). If these conditions are violated,
    a segfault may occur. The Python wrapper checks all input data to
    ensure it meets these conditions. If you decide to use this function
    OUTSIDE the Python wrapper, you must implement these checks yourself.

    Args:
        x (np.ndarray): An input data array.
        mu (np.ndarray): The model parameters (probability of each choice
            at each position). Shape is (n_components, sequence_length,
            num_possible_items).
        mix_weights (np.ndarray): The mixture weights for each component.
        n_threads (int): The number of threads to use.

    Returns:
        probs (np.ndarray): A float64 array of shape (x.shape[0])
            containing the log-likelihood of each datapoint.
    """
    cdef np.ndarray[np.float64_t, ndim=2] resp
    cdef np.ndarray[np.float64_t, ndim=1] log_mixweights
    cdef np.ndarray[np.float64_t, ndim=1] probs

    if mu.shape[0] != mix_weights.shape[0]:
        raise ValueError("Inputs to wrapped C++ function have incorrect shapes.")

    mu[mu<MINIMUM_PROB_VAL] = MINIMUM_PROB_VAL
    mu[:] = np.log(mu)
    log_mixweights = np.log(mix_weights.clip(min=MINIMUM_PROB_VAL))

    resp = np.zeros((mu.shape[0], x.shape[0]))
    probs = np.zeros((x.shape[0]))

    errcode = getProbsCExt_main(&x[0,0], &mu[0,0,0], &resp[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1], n_threads)
    resp += log_mixweights[:,None]
    probs[:] = logsumexp(resp, axis=0)
    return probs
