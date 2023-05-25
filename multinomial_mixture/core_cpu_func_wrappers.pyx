import os
import numpy as np
cimport cython
cimport numpy as np
from scipy.special import logsumexp
from libc.stdint cimport uint8_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uintptr_t

cdef extern from "prob_calcs.h" nogil:
    int getProbsCExt_main(uint8_t *x,
                   double *mu, double *resp,
                   int mu_dim0, int mu_dim1,
                   int mu_dim2, int x_dim0,
                   int x_dim1, int n_threads)
    int getWeightedCountCExt_main(uint8_t *x,
                   double *wcount, double *resp,
                   int wcount_dim0, int wcount_dim1,
                   int wcount_dim2, int x_dim0,
                   int x_dim1, int n_threads) 
    int getProbsCExt_single_thread(uint8_t *x,
                    double *mu, double *resp,
                    int mu_dim0, int mu_dim1, int mu_dim2,
                    int x_dim0, int x_dim1);
    int getWeightedCountCExt_single_thread(uint8_t *x,
                   double *wcount, double *resp,
                   int wc_dim0, int wc_dim1, int wc_dim2,
                   int x_dim0, int x_dim1);
    void getHardCountCExt_single_thread(uint8_t *x,
                   uint32_t *cluster_stats,
                   uint32_t *assignments,
                   int stats_dim0, int stats_dim1,
                   int stats_dim2, int x_dim0,
                   int x_dim1); 

    
@cython.boundscheck(False)
@cython.wraparound(False)
def em_short_single_call(list xfiles, 
                  np.ndarray[np.float64_t, ndim=1] mix_weights, 
                np.ndarray[np.float64_t, ndim=3] mu,
                int n_threads):
    cdef np.ndarray[np.float64_t, ndim=3] rik_counts = \
            np.zeros((mu.shape[0], mu.shape[1], mu.shape[2]))
    cdef np.ndarray[np.float64_t, ndim=2] log_mixweights = \
            np.log(mix_weights.clip(min=1e-14))[:,None]
    cdef np.ndarray[np.float64_t, ndim=1] new_weights = np.zeros((mix_weights.shape[0]))
    cdef np.ndarray[np.uint8_t, ndim=2] x
    cdef np.ndarray[np.float64_t, ndim=1] lnorm
    cdef np.ndarray[np.float64_t, ndim=2] resp
    cdef int errcode
    cdef float lb = 0
    cdef int ndpoints = 0
    cdef float net_resp = 0
        
    mu[mu<1e-14] = 1e-14
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
def em_ray_call(list xfiles, 
                  np.ndarray[np.float64_t, ndim=1] mix_weights, 
                np.ndarray[np.float64_t, ndim=3] mu):
    cdef np.ndarray[np.float64_t, ndim=3] rik_counts = \
            np.zeros((mu.shape[0], mu.shape[1], mu.shape[2]))
    cdef np.ndarray[np.float64_t, ndim=2] log_mixweights = \
            np.log(mix_weights.clip(min=1e-14))[:,None]
    cdef np.ndarray[np.float64_t, ndim=1] new_weights = np.zeros((mix_weights.shape[0]))
    cdef np.ndarray[np.uint8_t, ndim=2] x
    cdef np.ndarray[np.float64_t, ndim=1] lnorm
    cdef np.ndarray[np.float64_t, ndim=2] resp
    cdef int errcode
    cdef float lb = 0
    cdef int ndpoints = 0
    cdef float net_resp = 0
        
    mu[mu<1e-14] = 1e-14
    mu = np.log(mu)

    for xfile in xfiles:
        x = np.load(xfile)
        resp = np.zeros((mu.shape[0], x.shape[0]))
        lnorm = np.zeros((x.shape[0]))
            
        errcode = getProbsCExt_single_thread(&x[0,0], &mu[0,0,0], &resp[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1])
        resp += log_mixweights
        lnorm[:] = logsumexp(resp, axis=0)
        with np.errstate(under="ignore"):
            resp[:] = np.exp(resp - lnorm[None,:])
        lb += lnorm.sum()
        ndpoints += x.shape[0]

        rsum = resp.sum(axis=1)
        new_weights += rsum
        net_resp += rsum.sum()

        errcode = getWeightedCountCExt_single_thread(&x[0,0],
                    &rik_counts[0,0,0], &resp[0,0],
                    rik_counts.shape[0], rik_counts.shape[1], 
                    rik_counts.shape[2], x.shape[0], x.shape[1])
    return new_weights, lb, rik_counts, net_resp, ndpoints


@cython.boundscheck(False)
@cython.wraparound(False)
def hard_cluster_assign(list xfiles, 
                    np.ndarray[np.float64_t, ndim=3] mu,
                    np.ndarray[np.float64_t, ndim=1] mix_weights, 
                    int n_threads):
    cdef np.ndarray[np.uint32_t, ndim=3] cluster_stats = \
            np.zeros((mu.shape[0], mu.shape[1], mu.shape[2]), dtype=np.uint32)
    cdef np.ndarray[np.float64_t, ndim=2] log_mixweights = \
            np.log(mix_weights.clip(min=1e-14))[:,None]
    cdef np.ndarray[np.uint8_t, ndim=2] x
    cdef np.ndarray[np.uint32_t, ndim=1] assignments
    cdef np.ndarray[np.float64_t, ndim=1] lnorm
    cdef np.ndarray[np.float64_t, ndim=2] resp
    cdef int errcode
        
    mu[mu<1e-14] = 1e-14
    mu = np.log(mu)

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
    cdef np.ndarray[np.float64_t, ndim=2] probs
    cdef np.ndarray[np.float64_t, ndim=1] log_mixweights

    mu[mu<1e-14] = 1e-14
    mu[:] = np.log(mu)
    log_mixweights = np.log(mix_weights.clip(min=1e-14))

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
                int n_threads = 1):
    cdef np.ndarray[np.float64_t, ndim=2] probs
    cdef np.ndarray[np.float64_t, ndim=1] log_mixweights

    mu[mu<1e-14] = 1e-14
    mu[:] = np.log(mu)
    log_mixweights = np.log(mix_weights.clip(min=1e-14))

    probs = np.zeros((mu.shape[0], x.shape[0]))
    errcode = getProbsCExt_main(&x[0,0], &mu[0,0,0], &probs[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1], n_threads)
    probs += log_mixweights[:,None]
    return probs


def multimix_cluster_probs_no_mixweight(np.ndarray[np.uint8_t, ndim=2] x,
                np.ndarray[np.float64_t, ndim=3] mu,
                np.ndarray[np.float64_t, ndim=1] mix_weights,
                int n_threads = 1):
    cdef np.ndarray[np.float64_t, ndim=2] probs
    cdef np.ndarray[np.float64_t, ndim=1] log_mixweights

    mu[mu<1e-14] = 1e-14
    mu[:] = np.log(mu)
    log_mixweights = np.log(mix_weights.clip(min=1e-14))

    probs = np.zeros((mu.shape[0], x.shape[0]))
    errcode = getProbsCExt_main(&x[0,0], &mu[0,0,0], &probs[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1], n_threads)
    return probs

    
def multimix_loglik_offline(list xfiles,
        np.ndarray[np.float64_t, ndim=3] mu,
        np.ndarray[np.float64_t, ndim=1] mix_weights,
        int n_threads):
    cdef np.ndarray[np.float64_t, ndim=2] resp
    cdef np.ndarray[np.float64_t, ndim=1] log_mixweights
    cdef np.ndarray[np.float64_t, ndim=1] probs
    cdef np.ndarray[np.uint8_t, ndim=2] x

    mu[mu<1e-14] = 1e-14
    mu[:] = np.log(mu)
    log_mixweights = np.log(mix_weights.clip(min=1e-14))

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
    cdef np.ndarray[np.float64_t, ndim=2] resp
    cdef np.ndarray[np.float64_t, ndim=1] log_mixweights
    cdef np.ndarray[np.float64_t, ndim=1] probs

    mu[mu<1e-14] = 1e-14
    mu[:] = np.log(mu)
    log_mixweights = np.log(mix_weights.clip(min=1e-14))

    resp = np.zeros((mu.shape[0], x.shape[0]))
    probs = np.zeros((x.shape[0]))

    errcode = getProbsCExt_main(&x[0,0], &mu[0,0,0], &resp[0,0],
                        mu.shape[0], mu.shape[1], mu.shape[2],
                        x.shape[0], x.shape[1], n_threads)
    resp += log_mixweights[:,None]
    probs[:] = logsumexp(resp, axis=0)
    return probs
