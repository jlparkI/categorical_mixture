"""Checks to make sure that exceptions are thrown as expected
for invalid input."""
import os
import unittest
import numpy as np
from categorical_mix import CategoricalMixture
from categorical_mix.utilities.special_functions import logsumexp
from categorical_mix.utilities.catmix_utilities import em_online, em_offline
from categorical_mix_cpp_ext import getProbsCExt, getProbsCExt_masked, mask_terminal_deletions




class TestExceptionHandling(unittest.TestCase):
    """Checks that exceptions are raised for invalid input."""

    def test_model_construction(self):
        """Tests exceptions for model construction."""
        base_model = CategoricalMixture(n_components=10, num_possible_items=21,
                sequence_length=408)
        xdata, xfiles = load_test_data()

        with self.assertRaises(RuntimeError):
            base_model.score(xdata)
        with self.assertRaises(RuntimeError):
            base_model.predict(xdata)
        with self.assertRaises(RuntimeError):
            base_model.BIC(xdata)
        with self.assertRaises(RuntimeError):
            base_model.AIC(xdata)
        with self.assertRaises(RuntimeError):
            base_model.BIC_offline(xfiles)
        with self.assertRaises(RuntimeError):
            base_model.AIC_offline(xfiles)

        mu = np.ones((9, 400, 21))
        mix_weights = np.ones((9))
        with self.assertRaises(RuntimeError):
            base_model.load_params(mu, mix_weights)
        mu = np.ones((10, 408, 22))
        mix_weights = np.ones((10))
        with self.assertRaises(RuntimeError):
            base_model.load_params(mu, mix_weights)



    def test_data_acceptance(self):
        """Tests exceptions for accepting input data."""
        base_model = build_default_model()
        xdata, xfiles = load_test_data()
        xdummy = xdata.copy()
        xdummy[xdummy > 10] = 45

        with self.assertRaises(RuntimeError):
            base_model.score(xdata[:,:-2])
        with self.assertRaises(RuntimeError):
            base_model.score(xdummy)

        with self.assertRaises(RuntimeError):
            base_model.predict(xdata[:,:-2])
        with self.assertRaises(RuntimeError):
            base_model.predict(xdummy)

        with self.assertRaises(RuntimeError):
            base_model.BIC(xdata[:,:-2])
        with self.assertRaises(RuntimeError):
            base_model.BIC(xdummy)

        with self.assertRaises(RuntimeError):
            base_model.AIC(xdata[:,:-2])
        with self.assertRaises(RuntimeError):
            base_model.AIC(xdummy)

        with self.assertRaises(RuntimeError):
            base_model.AIC(xfiles)
        with self.assertRaises(RuntimeError):
            base_model.AIC(xfiles)




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




if __name__ == "__main__":
    unittest.main()
