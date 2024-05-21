"""Useful tools and utilities for scoring input data and fitting models
for the categorical mixture model tool specifically."""
import numpy as np
from .special_functions import logsumexp
from ..constants import constants
from categorical_mix_cpp_ext import getProbsCExt, getWeightedCountCExt


def em_offline(xfiles, mixweights, mu, n_threads):
    """Runs a single iteration of the EM algorithm for a specified list
    of .npy files on disk. The files should all be of type np.uint8 and
    should not contain any values larger than the number of possible
    items per position (mu.shape[2]).

    Args:
        xfiles (list): A list of .npy files of type np.uint8 with input
            data. Should already have been checked for acceptability.
        mixweights (np.ndarray): The mixture weights for each component.
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
    rik_counts = np.zeros(mu.shape)
    new_weights = np.zeros((mixweights.shape[0]))

    mixweights[mixweights < constants.CATMIX_MINIMUM_PROB_VAL] = constants.CATMIX_MINIMUM_PROB_VAL
    mu[mu < constants.CATMIX_MINIMUM_PROB_VAL] = constants.CATMIX_MINIMUM_PROB_VAL
    log_mixweights = np.log(mixweights)
    log_mu = np.log(mu)

    lb = 0
    ndpoints = 0
    net_resp = 0

    for xfile in xfiles:
        x = np.load(xfile)
        resp = np.zeros((log_mu.shape[0], x.shape[0]))
        lnorm = np.zeros((x.shape[0]))

        getProbsCExt(x, log_mu, resp, n_threads)
        resp += log_mixweights[:,None]
        lnorm[:] = logsumexp(resp, axis=0)
        with np.errstate(under="ignore"):
            resp[:] = np.exp(resp - lnorm[None,:])
        lb += lnorm.sum()
        ndpoints += x.shape[0]

        rsum = resp.sum(axis=1)
        new_weights += rsum
        net_resp += rsum.sum()
        getWeightedCountCExt(x, rik_counts, resp, n_threads)
    return new_weights, lb, rik_counts, net_resp, ndpoints



def em_online(x, mixweights, mu, n_threads):
    """Runs a single iteration of the EM algorithm for a single array of
    input data. The data should be of type np.uint8 and
    should not contain any values larger than the number of possible
    items per position (mu.shape[2]).

    Args:
        x (np.ndarray): An array of type np.uint8 with input
            data. Should already have been checked for acceptability.
        mixweights (np.ndarray): The mixture weights for each component.
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
    rik_counts = np.zeros(mu.shape)
    new_weights = np.zeros((mixweights.shape[0]))
    lnorm = np.zeros((x.shape[0]))
    resp = np.zeros((mu.shape[0], x.shape[0]))

    mixweights[mixweights < constants.CATMIX_MINIMUM_PROB_VAL] = constants.CATMIX_MINIMUM_PROB_VAL
    mu[mu < constants.CATMIX_MINIMUM_PROB_VAL] = constants.CATMIX_MINIMUM_PROB_VAL
    log_mixweights = np.log(mixweights)
    log_mu = np.log(mu)

    lb = 0
    ndpoints = 0
    net_resp = 0

    getProbsCExt(x, log_mu, resp, n_threads)
    resp += log_mixweights[:,None]
    lnorm[:] = logsumexp(resp, axis=0)
    with np.errstate(under="ignore"):
        resp[:] = np.exp(resp - lnorm[None,:])
    lb += lnorm.sum()
    ndpoints += x.shape[0]

    rsum = resp.sum(axis=1)
    new_weights += rsum
    net_resp += rsum.sum()

    getWeightedCountCExt(x, rik_counts, resp, n_threads)
    return new_weights, lb, rik_counts, net_resp, ndpoints



def multimix_loglik(x, log_mu, log_mixweights, n_threads = 1):
    """Determines the log likelihood of each input datapoint.

    Args:
        x (np.ndarray): An input data array.
        log_mu (np.ndarray): The model parameters (log probability of each choice
            at each position).
        log_mixweights (np.ndarray): The log mixture weights for each component.
        n_threads (int): The number of threads to use.

    Returns:
        probs (np.ndarray): A float64 array of shape (x.shape[0])
            containing the log-likelihood of each datapoint.
    """
    resp = np.zeros((log_mu.shape[0], x.shape[0]))
    probs = np.zeros((x.shape[0]))

    getProbsCExt(x, log_mu, resp, n_threads)
    resp += log_mixweights[:,None]
    probs[:] = logsumexp(resp, axis=0)
    return probs


def multimix_loglik_offline(xfiles, log_mu, log_mixweights, n_threads):
    """Determines the log likelihood of a dataset represented
    by an input list of .npy files.

    Args:
        xfiles (list): A list of .npy files with input data.
        log_mu (np.ndarray): The model parameters (log probability of each choice
            at each position).
        log_mixweights (np.ndarray): The log mixture weights for each component.
        n_threads (int): The number of threads to use.

    Returns:
        loglik (float): The log-likelihood of the input dataset.
    """
    loglik = 0

    for xfile in xfiles:
        x = np.load(xfile)
        resp = np.zeros((log_mu.shape[0], x.shape[0]))
        probs = np.zeros((x.shape[0]))
        getProbsCExt(x, log_mu, resp, n_threads)
        resp += log_mixweights[:,None]
        probs[:] = logsumexp(resp, axis=0)
        loglik += probs.sum()
    return loglik



def get_ndatapoints(xdata):
    """If the input is a list of files, this function
    quickly count the numbers of datapoints without
    reading any of them into memory. If it is an array,
    it returns the dim[0] of the array.

    Args:
        xdata: Either a list of file paths to numpy arrays
            saved on disk or a numpy array.

    Raises:
        ValueError: A ValueError is raised if an unexpected
            data type is supplied.
    """
    if isinstance(xdata, np.ndarray):
        return xdata.shape[0]

    if not isinstance(xdata, list):
        raise ValueError("Tried to supply data that was neither a "
                    "file list nor an array.")

    ndatapoints = 0
    for xfile in xdata:
        with open(xfile, 'rb') as fhandle:
            _, _ = np.lib.format.read_magic(fhandle)
            xshape, _, _ = np.lib.format.read_array_header_1_0(fhandle)
            ndatapoints += xshape[0]
    return ndatapoints
