"""Runs basic tests on the c-extension to ensure that
calculations are correct, bypassing the Python
wrapper (ordinarily a bad idea, since these check to
ensure the array is C-contiguous etc., but fine here
since we can enforce these conditions in the test."""
import os
import unittest
import numpy as np
from scipy.special import logsumexp
from core_cpu_func_wrappers import em_online, em_offline
from categorical_mixture.categorical_mix import CategoricalMixture


def get_current_dir():
    """Finds the current directory."""
    return os.path.abspath(os.path.dirname(__file__))

def get_initial_params(seq_length = 408, num_elements = 21):
    """Gets initial parameters specialized for the test set."""
    cat_mix = CategoricalMixture(n_components = 10, num_possible_items = num_elements,
            sequence_length = seq_length)
    return cat_mix._get_init_params(123)


def ground_truth_calcs(test_data, mix_weights, mu_in):
    """Generates a 'ground-truth' to compare
    against the em calculation routines."""
    log_mixweights = np.log(mix_weights.clip(min=1e-14))[:,None]
    mu_params = mu_in.copy()
    mu_params[mu_params < 1e-14] = 1e-14
    mu_params = np.log(mu_params)

    resp = np.zeros((mu_in.shape[0], test_data.shape[0]))
    lnorm = np.zeros((test_data.shape[0]))
    rik_counts = np.zeros(mu_in.shape)

    for k in range(mu_in.shape[0]):
        for i in range(test_data.shape[0]):
            resp_value = 0
            for j in range(test_data.shape[1]):
                resp_value += mu_params[k,j,test_data[i,j]]
            resp[k,i] = resp_value

    resp += log_mixweights
    lnorm[:] = logsumexp(resp, axis=0)
    with np.errstate(under="ignore"):
        resp[:] = np.exp(resp - lnorm[None,:])
    lower_bound = lnorm.sum()

    new_weights = resp.sum(axis=1)
    net_resp = new_weights.sum()

    for k in range(mu_in.shape[0]):
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[1]):
                rik_counts[k,j,test_data[i,j]] += resp[k,i]

    return new_weights, lower_bound, rik_counts, net_resp, test_data.shape[0]



class TestBasicCPPCalcs(unittest.TestCase):
    """Runs tests for basic functionality i.e. responsibility
    and weighted count calculations) for the cpp extension."""

    def test_em_online(self):
        """Tests responsibility calculations conducted by the cpp
        extension for data in memory."""
        start_dir = get_current_dir()
        data_path = os.path.join(start_dir, "..", "test_data", "encoded_test_data.npy")
        xdata = np.load(data_path)

        mix_weights, mu_init = get_initial_params()
        ground_truth = ground_truth_calcs(xdata, mix_weights, mu_init)

        test_results = em_online(xdata, mix_weights, mu_init.copy(), 1)
        for test_res, gt_res in zip(test_results, ground_truth):
            self.assertTrue(np.allclose(test_res, gt_res))

    def test_em_offline(self):
        """Tests responsibility calculations conducted by the cpp
        extension for data on disk."""
        start_dir = get_current_dir()
        data_path = os.path.join(start_dir, "..", "test_data", "encoded_test_data.npy")
        xdata = np.load(data_path)
        xfiles = [os.path.abspath(data_path)]

        mix_weights, mu_init = get_initial_params()
        ground_truth = ground_truth_calcs(xdata, mix_weights, mu_init)

        test_results = em_offline(xfiles, mix_weights, mu_init.copy(), 1)
        for test_res, gt_res in zip(test_results, ground_truth):
            self.assertTrue(np.allclose(test_res, gt_res))


if __name__ == "__main__":
    unittest.main()
