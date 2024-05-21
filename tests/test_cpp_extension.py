"""Runs basic tests on the c-extension to ensure that
calculations are correct."""
import os
import unittest
import numpy as np
from categorical_mix import CategoricalMixture
from categorical_mix.utilities.special_functions import logsumexp
from categorical_mix.utilities.catmix_utilities import em_online, em_offline
from categorical_mix_cpp_ext import getProbsCExt, getProbsCExt_masked, mask_terminal_deletions




class TestBasicCPPCalcs(unittest.TestCase):
    """Runs tests for basic functionality i.e. responsibility
    and weighted count calculations) for the cpp extension."""

    def test_em_online(self):
        """Tests responsibility calculations conducted by the cpp
        extension for data in memory."""
        xdata, _ = load_test_data()
        mix_weights, mu_init = get_initial_params()
        ground_truth = ground_truth_em_calcs(xdata, mix_weights, mu_init)

        test_results = em_online(xdata, mix_weights, mu_init, 1)
        for test_res, gt_res in zip(test_results, ground_truth):
            self.assertTrue(np.allclose(test_res, gt_res))


    def test_em_offline(self):
        """Tests responsibility calculations conducted by the cpp
        extension for data on disk."""
        xdata, xfiles = load_test_data()

        mix_weights, mu_init = get_initial_params()
        ground_truth = ground_truth_em_calcs(xdata, mix_weights, mu_init)

        test_results = em_offline(xfiles, mix_weights, mu_init, 1)
        for test_res, gt_res in zip(test_results, ground_truth):
            self.assertTrue(np.allclose(test_res, gt_res))


    def test_cext_scoring(self):
        """Check the workhorse function that provides scores
        for consistency with a simple Python version."""
        xdata, _ = load_test_data()

        for nclusters, nthreads in zip([1,51,12], [4,2,1]):
            base_model = build_default_model(nclusters)

            raw_score = base_model.score(xdata, n_threads = nthreads)
            comp_score = np.zeros((base_model.n_components, xdata.shape[0]))

            for i in range(xdata.shape[0]):
                for k in range(base_model.log_mu_mix.shape[0]):
                    for j in range(base_model.log_mu_mix.shape[1]):
                        comp_score[k,i] += base_model.log_mu_mix[k,j,xdata[i,j]]

            comp_score += base_model.log_mix_weights[:,None]
            comp_score = logsumexp(comp_score, axis=0)
            self.assertTrue(np.allclose(comp_score, raw_score))



    def test_terminal_deletions(self):
        """Tests the categorical mixture masked scoring function by
        comparing it with a simple python implementation."""
        xdata, _ = load_test_data()

        dummy_arr = xdata.copy()
        mask_terminal_deletions(dummy_arr)

        for i in range(xdata.shape[0]):
            for k in range(xdata.shape[1]):
                if xdata[i,k] != 20:
                    break
                xdata[i,k] = 255
            for k in range(xdata.shape[1]-1, 0, -1):
                if xdata[i,k] != 20:
                    break
                xdata[i,k] = 255

        self.assertTrue(np.allclose(dummy_arr, xdata))



    def test_mask_scoring(self):
        """Ensure that masking yields expected results."""
        xdata, _ = load_test_data()
        masks = [(i < 10 or i > 20) for i in range(xdata.shape[1])]

        for nclusters, nthreads in zip([1,51,12], [4,2,1]):
            base_model = build_default_model(nclusters)

            raw_score = base_model.score(xdata, np.array(masks), n_threads = nthreads)
            comp_score = np.zeros((base_model.n_components, xdata.shape[0]))

            for i in range(xdata.shape[0]):
                for k in range(base_model.log_mu_mix.shape[0]):
                    for j in range(base_model.log_mu_mix.shape[1]):
                        if not masks[j]:
                            continue
                        comp_score[k,i] += base_model.log_mu_mix[k,j,xdata[i,j]]

            comp_score += base_model.log_mix_weights[:,None]
            comp_score = logsumexp(comp_score, axis=0)
            self.assertTrue(np.allclose(comp_score, raw_score))




def load_test_data():
    """Loads some saved test data."""
    current_dir = os.getcwd()
    start_dir = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(start_dir, "test_data", "encoded_test_data.npy")
    xdata = np.load(data_path)
    xfiles = [os.path.abspath(data_path)]
    os.chdir(current_dir)
    return xdata, xfiles


def get_initial_params(seq_length = 408, num_elements = 21):
    """Gets initial parameters specialized for the test set."""
    cat_mix = CategoricalMixture(n_components = 10, num_possible_items = num_elements,
            sequence_length = seq_length)
    return cat_mix._get_init_params(123)


def build_default_model(nclusters = 51, seq_length = 408, num_elements = 21):
    """Builds an initial default model for the test set."""
    cat_mix = CategoricalMixture(n_components = nclusters, num_possible_items = num_elements,
            sequence_length = seq_length)
    mix_weights, mu_mix = cat_mix._get_init_params(123)
    cat_mix.load_params(mu_mix, mix_weights)
    return cat_mix


def ground_truth_em_calcs(test_data, mix_weights, mu_in):
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






if __name__ == "__main__":
    unittest.main()
