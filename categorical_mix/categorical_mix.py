"""Implements the CategoricalMixture class, for all operations involved
in generating mixture model predictions for new datapoints. The tools
necessary to fit the model (via EM) are in a separate repo.
Fitting right now is strictly via EM although we may evaluate
variational methods in future."""
import copy
from multiprocessing import Pool
import numpy as np
from .utilities.special_functions import logsumexp
from .utilities.catmix_utilities import get_ndatapoints, em_online, em_offline
from .utilities.catmix_utilities import multimix_loglik, multimix_loglik_offline
from .constants import constants
from categorical_mix_cpp_ext import getProbsCExt, getProbsCExt_masked, mask_terminal_deletions



class CategoricalMixture:
    """A CategoricalMixture model, with all the methods necessary
    to score and do inference.

    Attributes:
        mu_mix (np.ndarray): Array of type np.float64, shape (self.n_components,
            self.sequence_length, self.num_possible_items). The probability of
            each possible item for each point in sequence length for each cluster.
            Initialized to None, only converted to array once model is fitted.
        log_mu_mix (np.ndarray): Array of type np.float64, shape (self.n_components,
            self.sequence_length, self.num_possible_items). The log-probability of
            each possible item for each point in sequence length for each cluster.
            Initialized to None, only converted to array once model is fitted.
        mix_weights (np.ndarray): Array of type np.float64, shape (self.n_components).
            The weight for each distribution in the mixture.
        log_mix_weights (np.ndarray): Array of type np.float64, shape (self.n_components).
            The log of the mixture weight for each distribution in the mixture.
        n_components (int): The number of components.
        num_possible_items (int): The number of possible choices
            at each position in the sequence.
        sequence_length (int): The length of the sequences that the
            model will be fitted to / can analyze.
            
    """

    def __init__(self, n_components, num_possible_items = 21,
                sequence_length = 158):
        """Class constructor.

        Args:
            n_components (int): The number of mixture components
                (i.e. number of clusters).
            num_possible_items (int): The number of possible choices
                at each position in the sequence. For a protein sequence,
                for example, this might be the number of amino acid symbols
                that are possible; Currently limited to the range from 1 - 254.
            sequence_length (int): The length of the sequences that the
                model will be fitted to / can analyze.

        Raises:
            RuntimeError: A RuntimeError is raised if unacceptable arguments are
                supplied.
        """
        if n_components <= 0:
            raise RuntimeError("n_components must be > 0.")
        #This class reserves 255 for internal use, so num_possible_items must
        #be < 255 (because input is expected to be uint8).
        if num_possible_items >= 255 or num_possible_items <= 0:
            raise RuntimeError("Currently num_possible_items is limited to "
                    "values from 1 to 254, inclusive.")
        if sequence_length <= 0:
            raise RuntimeError("Sequence length must be positive.")

        self.mix_weights = None
        self.log_mix_weights = None
        self.mu_mix = None
        self.log_mu_mix = None
        self.n_components = n_components
        self.num_possible_items = num_possible_items
        self.sequence_length = sequence_length


    def load_params(self, mu_mix, mix_weights):
        """Checks input params to ensure they are compatible with selected
        settings, and if so, sets the model parameters to the inputs.

        Args:
            mu_mix (np.ndarray): Array of type np.float64, shape (self.n_components,
                self.sequence_length, self.num_possible_items). The probability of
                each possible item for each point in sequence length for each cluster.
            mix_weights (np.ndarray): Array of type np.float64, shape (self.n_components).
                The weight for each distribution in the mixture.

        Raises:
            RuntimeError: A RuntimeError is raised if the inputs are inappropriate.
        """
        if not isinstance(mu_mix, np.ndarray) or not isinstance(mix_weights, np.ndarray):
            raise RuntimeError("mu_mix and mix_weights should both be numpy arrays.")
        if len(mu_mix.shape) != 3 or len(mix_weights.shape) != 1:
            raise RuntimeError("mu_mix and mix_weights must be 3d and 1d arrays respectively.")
        if mu_mix.shape[0] != self.n_components or mu_mix.shape[1] != self.sequence_length \
                or mu_mix.shape[2] != self.num_possible_items:
            raise RuntimeError("mu_mix has an inappropriate shape.")
        if mix_weights.shape[0] != self.n_components:
            raise RuntimeError("mix_weights has an inappropriate shape.")

        if not mu_mix.flags["C_CONTIGUOUS"] or not mix_weights.flags["C_CONTIGUOUS"]:
            raise RuntimeError("Non-C-contiguous arrays supplied.")

        self.mix_weights = mix_weights
        self.log_mix_weights = mix_weights.copy()
        self.log_mix_weights[self.log_mix_weights < constants.CATMIX_MINIMUM_PROB_VAL] = \
                constants.CATMIX_MINIMUM_PROB_VAL
        self.log_mix_weights = np.log(self.log_mix_weights)

        self.mu_mix = mu_mix
        self.log_mu_mix = mu_mix.copy()
        self.log_mu_mix[self.log_mu_mix < constants.CATMIX_MINIMUM_PROB_VAL] = \
                constants.CATMIX_MINIMUM_PROB_VAL
        self.log_mu_mix[:] = np.log(self.log_mu_mix)



    def _check_input_array(self, xdata, n_threads):
        """Checks an input array to make sure that all have
        the correct format to ensure that no problems will be encountered.

        Args:
            x (np.ndarray): A numpy array with the input data.
            n_threads (int): The number of threads requested.

        Raises:
            RuntimeError: A RuntimeError is raised if unacceptable
                input data is supplied.
        """
        if not isinstance(xdata, np.ndarray):
            raise RuntimeError("Unexpected input supplied.")
        if np.max(xdata) >= self.num_possible_items or np.min(xdata) < 0:
            raise RuntimeError("Values in input data are out of range.")
        if xdata.dtype != "uint8":
            raise RuntimeError("Unexpected datatype for input data.")
        if len(xdata.shape) != 2:
            raise RuntimeError("Unexpected shape for input data.")
        if xdata.shape[1] != self.sequence_length or xdata.shape[0] < 1:
            raise RuntimeError("Unexpected shape for input data.")
        if not xdata.flags["C_CONTIGUOUS"]:
            raise RuntimeError("Input data is not C-contiguous.")

        if n_threads <= 0:
            raise RuntimeError("n_threads should be a positive integer.")
        if not isinstance(n_threads, int):
            raise RuntimeError("n_threads should be a positive integer.")


    def _prep_masked_array(self, xdata, mask = None, mask_terminal_dels = False,
            mask_gaps = False):
        """Prepares a masked version of an input array if any masking options are
        supplied.

        Args:
            xdata (np.ndarray): A 2d numpy array of type np.uint8.
            mask (np.ndarray): Either None or a numpy array of type bool
                and shape (xdata.shape[1]). If not None, indicated
                positions are masked, i.e. are not taken into account
                when calculating the score.
            mask_gaps (bool): If True, all gaps in the sequence
                are ignored when calculating the score. This is useful when your
                sequence has unusual deletions and you would like to ignore these.
            mask_terminal_dels (bool): If True, ignore N- and C-terminal
                deletions when assigning to a cluster.

        Returns:
            xmasked (np.ndarray): A 2d numpy array of type np.uint8.

        Raises:
            RuntimeError: A RuntimeError is raised if bad inputs are supplied.
        """
        #This option is only available if the number of possible items is < 255.
        if self.num_possible_items >= 255:
            raise RuntimeError("Currently masking is only available if the num_possible_items "
                    "setting is < 255.")

        xmasked = xdata.copy()

        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise RuntimeError("Mask must be a numpy array.")
            if mask.shape[0] != xdata.shape[1]:
                raise RuntimeError("Mask shape must be consistent with xdata shape.")
            if len(mask.shape) != 1:
                raise RuntimeError("Mask must be a 1d array.")

            #We use 255 as a convenient "please ignore" indicator.
            xmasked[:,~mask] = 255

        if mask_terminal_dels:
            mask_terminal_deletions(xmasked)

        if mask_gaps:
            xmasked[xmasked==20] = 255

        return xmasked



    def predict(self, xdata, mask = None, mask_terminal_dels = False,
            mask_gaps = False, use_mixweights = True, return_raw_probs = False,
            n_threads = 1):
        """Determine the most probable cluster for each datapoint
        in a numpy array. Note that you should also check the
        overall probability of each datapoint. If a datapoint is
        very different from your training set, it will have
        very low overall probability, but this function will
        still assign it to the most likely cluster -- whichever
        that is -- by default.

        Args:
            xdata (np.ndarray): A 2d numpy array of type np.uint8.
            n_threads (int): The number of threads to use.
            mask (np.ndarray): Either None or a numpy array of type bool
                and shape (xdata.shape[1]). If not None, indicated
                positions are masked, i.e. are not taken into account
                when calculating the score.
            mask_gaps (bool): If True, all non-filled IMGT positions in the sequence
                are ignored when calculating the score. This is useful when your
                sequence has unusual deletions and you would like to ignore these.
            mask_terminal_dels (bool): If True, ignore N- and C-terminal
                deletions when assigning to a cluster.
            use_mixweights (bool): If True, take mixture weights into
                account; otherwise, find the closest cluster (even if
                it is a low-probability cluster).
            return_raw_probs (bool): If True, return the raw probabilities
                instead of cluster assignments.

        Returns:
            preds (np.ndarray): EITHER an array of shape (xdata.shape[0])
                containing a number from 0 to self.n_components - 1
                indicating the predicted cluster for each datapoint if
                return_raw_probs is False, OR an array of shape (ncomponents,
                xdata.shape[0]) if return_raw_probs is True, indicating
                the probability of each cluster for each datapoint.

        Raises:
            RuntimeError: Raised if unexpected inputs are supplied.
        """
        self._check_input_array(xdata, n_threads)
        if self.log_mu_mix is None or self.log_mix_weights is None:
            raise RuntimeError("Model not fitted yet.")

        resp = np.zeros((self.log_mu_mix.shape[0], xdata.shape[0]))

        if mask is not None or mask_terminal_dels or mask_gaps:
            xmasked = self._prep_masked_array(xdata, mask, mask_terminal_dels, mask_gaps)
            getProbsCExt_masked(xmasked, self.log_mu_mix, resp, n_threads)

        else:
            getProbsCExt(xdata, self.log_mu_mix, resp, n_threads)

        if use_mixweights:
            resp += self.log_mix_weights[:,None]
        if return_raw_probs:
            return resp
        cluster_assignments = resp.argmax(axis=0).astype(np.uint32)
        return cluster_assignments



    def score(self, xdata, mask = None, mask_terminal_dels = False,
            mask_gaps = False, normalize_scores = False, n_threads = 1):
        """Generate the overall log-likelihood of individual datapoints.
        This is very useful to determine if a new datapoint is very
        different from the training set. If the log-likelihood of
        the new datapoints is much lower than the training set
        distribution of log-likelihoods, it is fairly unlikely
        that the new datapoint is a sample from the distribution
        represented by the model, and you should not try to
        assign it to a cluster.

        Args:
            xdata (np.ndarray): An array with the input data,
                of type np.uint8.
            mask (np.ndarray): Either None or a numpy array of type bool
                and shape (xdata.shape[1]). If not None, indicated
                positions are masked, i.e. are not taken into account
                when calculating the score.
            mask_terminal_dels (bool): If True, ignore N- and C-terminal
                deletions when calculating a score.
            mask_gaps (bool): If True, all gaps in the sequence
                are ignored when calculating the score. This is useful when your
                sequence has unusual deletions and you would like to ignore these.
            normalize_scores (bool): If True, normalize the score by dividing by
                the number of non-masked residues in the input.
            n_threads (int): the number of threads to use.

        Returns:
            loglik (np.ndarray): A float64 array of shape (x.shape[0])
                where each element is the log-likelihood of that
                datapoint given the model.

        Raises:
            RuntimeError: Raised if unexpected inputs are supplied.
        """
        self._check_input_array(xdata, n_threads)
        if self.mu_mix is None or self.mix_weights is None:
            raise RuntimeError("Model not fitted yet.")

        resp = np.zeros((self.log_mu_mix.shape[0], xdata.shape[0]))
        if mask is not None or mask_terminal_dels or mask_gaps:
            xmasked = self._prep_masked_array(xdata, mask, mask_terminal_dels, mask_gaps)
            getProbsCExt_masked(xmasked, self.log_mu_mix, resp, n_threads)
            resp += self.log_mix_weights[:,None]
            resp = logsumexp(resp, axis=0)
            if normalize_scores:
                resp /= (xmasked < 255).sum(axis=1)

        else:
            getProbsCExt(xdata, self.log_mu_mix, resp, n_threads)
            resp += self.log_mix_weights[:,None]
            resp = logsumexp(resp, axis=0)
            if normalize_scores:
                resp /= xdata.shape[1]

        return resp




    # The remaining functions are used when fitting, preparing to fit,
    # or evaluating the quality of a fit.


    def _check_input_files(self, xlist):
        """Checks an input list of files to make sure that all have
        the correct format to ensure that no problems will be encountered
        during fitting.

        Args:
            xfiles (list): A list of file paths to numpy arrays saved
                on disk as .npy files.
        Raises:
            RuntimeError: A RuntimeError is raised if one or more files has
                unacceptable issues.
        """
        if not isinstance(xlist, list):
            raise RuntimeError("Unexpected input supplied.")
        for xfile in xlist:
            x_in = np.load(xfile)
            if np.max(x_in) > self.num_possible_items or np.min(x_in) < 0:
                raise RuntimeError(f"Values in {xfile} are out of range.")
            if x_in.dtype != "uint8":
                raise RuntimeError(f"Unexpected datatype for {xfile}.")
            if len(x_in.shape) != 2:
                raise RuntimeError(f"Unexpected shape for {xfile}.")
            if x_in.shape[1] != self.sequence_length or x_in.shape[0] < 1:
                raise RuntimeError(f"Unexpected shape for {xfile}.")
            if not x_in.flags["C_CONTIGUOUS"]:
                raise RuntimeError("Input data is not C-contiguous.")


    def _get_init_params(self, random_state):
        """Gets initial parameters for the catmix model.

        Args:
            random_state (int): A seed to the random number generator.
        """
        rng = np.random.default_rng(random_state)
        mix_weights = rng.uniform(size = self.n_components)
        mix_weights /= mix_weights.sum()
        mu_mix = rng.uniform(size = (self.n_components, self.sequence_length, \
                self.num_possible_items))
        mu_mix /= mu_mix.sum(axis=2)[:,:,None]
        return mix_weights, mu_mix



    def fit(self, xdata, max_iter = 150, tol = 1e-3,
                n_restarts = 1, random_state = 123,
                n_processes = 1, n_threads = 1):
        """Fits the model to either an input array or a list
        of input files, potentially using multiple restarts if so
        specified.

        Args:
            xdata: Either a list of file paths to numpy arrays
                saved on disk as .npy files, or a numpy array.
                Either way, must be of type np.uint8 (this restriction
                will be lifted in future).
            max_iter (int): The maximum number of iterations for
                one restart.
            tol (float): Once the iteration to iteration change in loss
                falls below this value, fitting is assumed to have
                converged.
            n_restarts (int): The number of times to restart the
                fitting process with a new random seed.
            random_state (int): The random number generator seed for
                the first restart (the second will use this value + 1,
                the third will use this value + 2 etc.)
            n_processes (int): The number of processes to use if running
                in parallel using multiprocessing. Only used
                if xdata is a list of files, in which case it is split up
                into sub-lists and each process is given a sublist. If xdata
                is an array this argument is ignored. Note that
                multiprocessing can greatly increase speed but also
                increases memory consumption, because doubling
                n_processes doubles the copies of the parameters in memory.
            n_threads (int): The number of threads to use on each array.
                Unlike n_processes, increasing this value does not increase
                memory consumption at all, but it does not give quite as
                large of a speed gain. If you set n_threads to > 1 and
                n_processes to > 1, each process in n_processes will
                spawn n_threads, so total_threads is n_threads *
                n_processes.

        Raises:
            RuntimeError: Raised if unexpected inputs are supplied.
        """
        if isinstance(xdata, np.ndarray):
            self._check_input_array(xdata, n_threads)
            iter_runner = self._single_iter_online
        elif isinstance(xdata, list):
            self._check_input_files(xdata)
            if n_processes == 1 or len(xdata) < 3:
                iter_runner = self._single_iter_offline
            else:
                iter_runner = self._single_iter_mp
        else:
            raise RuntimeError("xdata must be either a list or a numpy array.")


        best_loss = -np.inf
        for restart in range(n_restarts):
            mix_weights, mu_mix, loss = iter_runner(xdata, tol,
                    max_iter, random_state + restart,
                    n_processes = n_processes, n_threads = n_threads)
            if loss > best_loss:
                self.load_params(mu_mix.clip(1e-16), mix_weights)
                best_loss = copy.deepcopy(loss)

        if self.mu_mix is None:
            raise RuntimeError("No restarts converged!")


    def _single_iter_online(self, xdata, tol,
                    max_iter, random_state,
                    n_processes = 1,
                    n_threads = 1):
        """Fit the input numpy array with one random state,
        no multiprocessing.

        Args:
            xdata (np.ndarray): A numpy array of type uint8.
            max_iter (int): The maximum number of iterations for
                this restart.
            tol (float): Once the iteration to iteration change in loss
                falls below this value, fitting is assumed to have
                converged.
            random_state (int): The random number generator seed.
            n_processes (int): This function accepts but does not use
                n_processes (to remaing consistent with the other single_iter
                functions).
            n_threads (int): The number of threads to use on each array.

        Raises:
            RuntimeError: Raised if unexpected inputs are supplied.
        """
        loss = -np.inf
        mix_weights, mu_mix = self._get_init_params(random_state)

        for i in range(max_iter):
            prev_loss = copy.copy(loss)
            mix_weights, loss, mu_mix, net_resp, ndpoints = em_online(xdata,
                                mix_weights, mu_mix, n_threads)

            mix_weights /= net_resp
            loss /= ndpoints
            mu_mix /= mu_mix.sum(axis=2)[:,:,None].clip(min=1)

            print(f"Loss: {loss}", flush=True)
            if np.abs(loss - prev_loss) < tol:
                break

        print(f"Iterations: {i}****************\n")
        return mix_weights, mu_mix, loss



    def _single_iter_offline(self, xfiles, tol,
                    max_iter, random_state,
                    n_processes = 1,
                    n_threads = 1):
        """Fit the input list of on-disk numpy .npy files
        with one random state, no multiprocessing.

        Args:
            xfiles (list): A list of .npy files, each a numpy
                array of type np.uint8 saved on disk.
            max_iter (int): The maximum number of iterations for
                this restart.
            tol (float): Once the iteration to iteration change in loss
                falls below this value, fitting is assumed to have
                converged.
            random_state (int): The random number generator seed.
            n_processes (int): This function accepts but does not use
                n_processes (to remaing consistent with the other single_iter
                functions).
            n_threads (int): The number of threads to use on each array.

        Raises:
            RuntimeError: Raised if unexpected inputs are supplied.
        """
        loss = -np.inf
        mix_weights, mu_mix = self._get_init_params(random_state)

        for i in range(max_iter):
            prev_loss = copy.copy(loss)
            mix_weights, loss, mu_mix, net_resp, ndpoints = em_offline(xfiles,
                                mix_weights, mu_mix, n_threads)
            mix_weights /= net_resp
            loss /= ndpoints
            mu_mix /= mu_mix.sum(axis=2)[:,:,None].clip(min=1)

            print(f"Loss: {loss}", flush=True)
            if np.abs(loss - prev_loss) < tol:
                break

        print(f"Iterations: {i}****************\n")
        return mix_weights, mu_mix, loss



    def _single_iter_mp(self, xfiles, tol,
                    max_iter, random_state,
                    n_processes = 1,
                    n_threads = 1):
        """Fit the input list of on-disk numpy .npy files
        with one random state, using multiprocessing with
        n_processes and multithreading on each process.

        Args:
            xfiles (list): A list of .npy files, each a numpy
                array of type np.uint8 saved on disk.
            max_iter (int): The maximum number of iterations for
                this restart.
            tol (float): Once the iteration to iteration change in loss
                falls below this value, fitting is assumed to have
                converged.
            random_state (int): The random number generator seed.
            n_processes (int): The number of processes to use.
            n_threads (int): The number of threads to use on each array
                in each process.

        Raises:
            RuntimeError: Raised if unexpected inputs are supplied.
        """
        n_processes = min(len(xfiles), n_processes)
        chunk_size = int((len(xfiles) + n_processes - 1) / n_processes)
        print(f"Using MP. Chunk size: {chunk_size} files")
        xchunks = [xfiles[i:i + chunk_size] for i in
                range(0, len(xfiles), chunk_size)]

        loss = -np.inf
        mix_weights, mu_mix = self._get_init_params(random_state)

        for i in range(max_iter):
            prev_loss = copy.copy(loss)
            caller_args = [(xchunk, mix_weights.copy(), mu_mix.copy(),
                                n_threads) for xchunk in xchunks]
            with Pool(n_processes) as mp_pool:
                mp_res = list(mp_pool.starmap(em_offline, caller_args))

            mix_weights = np.zeros(mp_res[0][0].shape)
            mu_mix = np.zeros(mp_res[0][2].shape)
            loss = 0
            for res in mp_res:
                mix_weights += res[0]
                mu_mix += res[2]
                loss += res[1]

            mix_weights /= np.sum([res[3] for res in mp_res])
            loss /= np.sum([res[4] for res in mp_res])
            mu_mix /= mu_mix.sum(axis=2)[:,:,None].clip(min=1)

            print(f"Loss: {loss}", flush=True)
            if np.abs(loss - prev_loss) < tol:
                break
        print(f"Iterations: {i}****************\n")
        return mix_weights, mu_mix, loss



    def _get_nparams(self):
        """Returns the number of parameters. Does not check if model
        has been fitted yet, caller must check."""
        return self.n_components - 1 + self.n_components * \
                ((self.num_possible_items - 1) * self.sequence_length)


    def BIC(self, xdata, n_threads = 1):
        """Calculate the Bayes information criterion for an input array.

        Args:
            xdata (np.ndarray): A numpy array of type np.uint8, shape 2.
            n_threads (int): The number of threads to use.

        Raises:
            RuntimeError: Raised if unexpected inputs are supplied.
        """
        self._check_input_array(xdata, n_threads)

        if self.mu_mix is None:
            raise RuntimeError("Model not fitted yet.")

        ndatapoints = get_ndatapoints(xdata)
        loglik = multimix_loglik(xdata, self.log_mu_mix, self.mix_weights,
                n_threads).sum()

        nparams = self._get_nparams()
        return nparams * np.log(ndatapoints) - 2 * loglik



    def BIC_offline(self, xfiles, n_processes = 1, n_threads = 1):
        """Calculate the Bayes information criterion for a list
        of numpy arrays saved as .npy files, each of type np.uint8.

        Args:
            xdata (list): A list of file paths to .npy files.
            n_processes (int): The number of processes. If > 1,
                multiprocessing is used. This will increase
                memory consumption since n_processes copies must
                be made of the model parameters.
            n_threads (int): the number of threads to use per
                process. Unlike multiprocessing, does not increase
                memory consumption.

        Raises:
            RuntimeError: Raised if unexpected inputs are supplied.
        """
        self._check_input_files(xfiles)

        if self.mu_mix is None:
            raise RuntimeError("Model not fitted yet.")
        ndatapoints = get_ndatapoints(xfiles)

        if len(xfiles) < 3 or n_processes == 1:
            loglik = multimix_loglik_offline(xfiles, self.log_mu_mix,
                    self.log_mix_weights, n_threads)
        else:
            n_processes = min(len(xfiles), n_processes)
            chunk_size = int((len(xfiles) + n_processes - 1) / n_processes)
            print(f"Using MP. Chunk size: {chunk_size} files")
            xchunks = [xfiles[i:i + chunk_size] for i in
                    range(0, len(xfiles), chunk_size)]
            caller_args = [(xchunk, self.log_mu_mix.copy(), self.log_mix_weights.copy(),
                                n_threads) for xchunk in xchunks]
            with Pool(n_processes) as mp_pool:
                mp_res = list(mp_pool.starmap(multimix_loglik_offline, caller_args))
            loglik = np.sum(mp_res)

        nparams = self._get_nparams()
        return nparams * np.log(ndatapoints) - 2 * loglik


    def AIC(self, xdata, n_threads = 1):
        """Calculate the Akaike information criterion for an input array.

        Args:
            xdata (np.ndarray): a numpy array of type np.uint8, shape 2.
            n_threads (int): the number of threads to use.

        Raises:
            RuntimeError: raised if unexpected inputs are supplied.
        """
        self._check_input_array(xdata, n_threads)
        if self.mu_mix is None:
            raise RuntimeError("Model not fitted yet.")
        loglik = multimix_loglik(xdata, self.log_mu_mix, self.mix_weights.copy(),
                n_threads).sum()
        return 2 * self._get_nparams() - 2 * loglik


    def AIC_offline(self, xfiles, n_processes = 1, n_threads = 1):
        """Calculate the Akaike information criterion for a list
        of numpy arrays saved as .npy files, each of type np.uint8.

        Args:
            xdata (list): A list of file paths to .npy files.
            n_processes (int): The number of processes. If > 1,
                multiprocessing is used. This will increase
                memory consumption since n_processes copies must
                be made of the model parameters.
            n_threads (int): the number of threads to use per
                process. Unlike multiprocessing, does not increase
                memory consumption.

        Raises:
            RuntimeError: Raised if unexpected inputs are supplied.
        """
        self._check_input_files(xfiles)
        if self.mu_mix is None:
            raise RuntimeError("Model not fitted yet.")

        if len(xfiles) < 3 or n_processes == 1:
            loglik = multimix_loglik_offline(xfiles, self.log_mu_mix,
                    self.log_mix_weights, n_threads)
        else:
            n_processes = min(len(xfiles), n_processes)
            chunk_size = int((len(xfiles) + n_processes - 1) / n_processes)
            print(f"Using MP. Chunk size: {chunk_size} files")
            xchunks = [xfiles[i:i + chunk_size] for i in
                    range(0, len(xfiles), chunk_size)]
            caller_args = [(xchunk, self.log_mu_mix.copy(), self.log_mix_weights.copy(),
                                n_threads) for xchunk in xchunks]
            with Pool(n_processes) as mp_pool:
                mp_res = list(mp_pool.starmap(multimix_loglik_offline, caller_args))
            loglik = np.sum(mp_res)

        return 2 * self._get_nparams() - 2 * loglik
