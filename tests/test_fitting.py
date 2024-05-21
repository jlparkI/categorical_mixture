"""Fits the model to the example dataset."""
import os
import unittest
import numpy as np
from categorical_mix import CategoricalMixture




class TestFitting(unittest.TestCase):
    """Fits the example dataset."""

    def test_online_fit(self):
        """Runs some quick and dirty sanity checks on the
        fitting procedure for in-memory data."""
        xdata, _ = load_test_data()
        base_model = build_default_model()

        init_bic = base_model.BIC(xdata)
        init_aic = base_model.BIC(xdata)

        base_model.fit(xdata, max_iter = 150, n_restarts=3)
        final_bic = base_model.BIC(xdata)
        final_aic = base_model.AIC(xdata)

        self.assertTrue(final_bic < init_bic)
        self.assertTrue(final_aic < init_aic)



    def test_offline_fit(self):
        """Runs some quick and dirty sanity checks on the
        fitting procedure for on-disk data."""
        _, xfiles = load_test_data()
        base_model = build_default_model()

        init_bic = base_model.BIC_offline(xfiles)
        init_aic = base_model.BIC_offline(xfiles)

        base_model.fit(xfiles, max_iter = 150, n_restarts=3)
        final_bic = base_model.BIC_offline(xfiles)
        final_aic = base_model.AIC_offline(xfiles)

        self.assertTrue(final_bic < init_bic)
        self.assertTrue(final_aic < init_aic)




def load_test_data():
    """Loads some saved test data."""
    current_dir = os.getcwd()
    start_dir = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(start_dir, "test_data", "encoded_test_data.npy")
    xdata = np.load(data_path)
    xfiles = [os.path.abspath(data_path)]
    os.chdir(current_dir)
    return xdata, xfiles


def build_default_model(nclusters = 51, seq_length = 408, num_elements = 21):
    """Builds an initial default model for the test set."""
    cat_mix = CategoricalMixture(n_components = nclusters, num_possible_items = num_elements,
            sequence_length = seq_length)
    mix_weights, mu_mix = cat_mix._get_init_params(123)
    cat_mix.load_params(mu_mix, mix_weights)
    return cat_mix




if __name__ == "__main__":
    unittest.main()
