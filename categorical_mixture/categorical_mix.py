"""Implements the CategoricalMixture class, for all operations involved
in fitting the mixture model and in generating predictions for new
datapoints."""
import copy
from multiprocessing import Pool
import numpy as np
from core_cpu_func_wrappers import em_online, em_offline, multimix_predict
from core_cpu_func_wrappers import multimix_loglik_offline, multimix_score, multimix_cluster_probs
from core_cpu_func_wrappers import hard_cluster_assign, multimix_score_masked




class CategoricalMixture:
    """A CategoricalMixture model, with all the methods necessary
    to fit the model, score it and do inference. This
    CategoricalMixture is fitted using EM -- if it ever proves
    useful, we can add an alternative based on variational
    inference.

    Attributes:
        mu_mix (np.ndarray): Array of type np.float64, shape (self.n_components,
            self.sequence_length, self.num_possible_items). The probability of
            each possible item for each point in sequence length for each cluster.
            Initialized to None, only converted to array once model is fitted.
        mix_weights (np.ndarray): Array of type np.float64, shape (self.n_components).
            The weight for each distribution in the mixture.
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
                that are possible; for a sequence of letters, this might
                be the number of letters in the alphabet; for shopping
                data, this might be the number of unique items the customer
                might purchase. Currently limited to the range from 1 - 255,
                this restriction will be lifted in a future version.
            sequence_length (int): The length of the sequences that the
                model will be fitted to / can analyze.

        Raises:
            ValueError: A ValueError is raised if unacceptable arguments are
                supplied.
        """
        if n_components <= 0:
            raise ValueError("n_components must be > 0.")
        if num_possible_items > 255 or num_possible_items <= 0:
            raise ValueError("Currently num_possible_items is limited to "
                    "values from 1 to 255, inclusive.")
        if sequence_length <= 0:
            raise ValueError("Sequence length must be positive.")

        self.mix_weights = None
        self.mu_mix = None
        self.n_components = n_components
        self.num_possible_items = num_possible_items
        self.sequence_length = sequence_length


    def _get_ndatapoints(self, xdata):
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


    def _get_init_params(self, random_state):
        """Initializes the model using the random seed to generate
        starting parameters.

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


    def _check_input_files(self, xlist):
        """Checks an input list of files to make sure that all have
        the correct format to ensure that no problems will be encountered
        during fitting.

        Args:
            xfiles (list): A list of file paths to numpy arrays saved
                on disk as .npy files.
        Raises:
            ValueError: A ValueError is raised if one or more files has
                unacceptable issues.
        """
        if not isinstance(xlist, list):
            raise ValueError("Unexpected input supplied.")
        for xfile in xlist:
            x_in = np.load(xfile)
            if np.max(x_in) > self.num_possible_items or np.min(x_in) < 0:
                raise ValueError(f"Values in {xfile} are out of range.")
            if x_in.dtype != "uint8":
                raise ValueError(f"Unexpected datatype for {xfile}.")
            if len(x_in.shape) != 2:
                raise ValueError(f"Unexpected shape for {xfile}.")
            if x_in.shape[1] != self.sequence_length or x_in.shape[0] < 1:
                raise ValueError(f"Unexpected shape for {xfile}.")
            if not x_in.flags["C_CONTIGUOUS"]:
                raise ValueError("Input data is not C-contiguous.")


    def _check_input_array(self, xdata):
        """Checks an input array to make sure that all have
        the correct format to ensure that no problems will be encountered
        during fitting.

        Args:
            x (np.ndarray): A numpy array with the input data.

        Raises:
            ValueError: A ValueError is raised if unacceptable
                input data is supplied.
        """
        if not isinstance(xdata, np.ndarray):
            raise ValueError("Unexpected input supplied.")
        if np.max(xdata) >= self.num_possible_items or np.min(xdata) < 0:
            raise ValueError("Values in input data are out of range.")
        if xdata.dtype != "uint8":
            raise ValueError("Unexpected datatype for input data.")
        if len(xdata.shape) != 2:
            raise ValueError("Unexpected shape for input data.")
        if xdata.shape[1] != self.sequence_length or xdata.shape[0] < 1:
            raise ValueError("Unexpected shape for input data.")
        if not xdata.flags["C_CONTIGUOUS"]:
            raise ValueError("Input data is not C-contiguous.")


    def fit(self, xdata, max_iter = 150, tol = 1e-3,
                n_restarts = 1, random_state = 123,
                enable_input_checking = True,
                model_polishing = False,
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
            enable_input_checking (bool): Defaults to True.
                If False, the inputs are not
                checked to ensure they are valid. WARNING: DO NOT SET
                THIS TO FALSE without careful consideration. If you
                set this to False, the validity of your input data will
                not be checked, which may result in serious fatal errors
                if your input data is not what the model expects. The speed
                gain from setting this to False is quite modest, so this
                is really only worthwhile if (1) you've already checked your
                input data and (2) your training set is very large (> 100
                million datapoints).
            model_polishing (bool): Defaults to False. If True, it
                is assumed the model has already been fitted and you
                merely want to refine the fit further by doing more
                iterations with a smaller tol. If you set this to True
                and the model has not yet been fitted a ValueError is
                raised.
            n_processes (int): The number of processes to use if running
                in parallel using multiprocessing or Ray. Only used
                if xdata is a list of files, in which case it is split up
                into sub-lists and each process is given a sublist. If xdata
                is an array this argument is ignored. Note that
                multiprocessing can greatly increase speed but also
                increases memory consumption, because doubling
                n_processes doubles the number of copies of the parameters
                that are held in memory.
            n_threads (int): The number of threads to use on each array.
                Unlike n_processes, increasing this value does not increase
                memory consumption at all, but it does not give quite as
                large of a speed gain. If you set n_threads to > 1 and
                n_processes to > 1, each process in n_processes will
                spawn n_threads, so total_threads is n_threads *
                n_processes.

        Raises:
            ValueError: Raised if unexpected inputs are supplied.
        """
        if isinstance(xdata, np.ndarray):
            if enable_input_checking:
                self._check_input_array(xdata)
            iter_runner = self._single_iter_online
        elif isinstance(xdata, list):
            if enable_input_checking:
                self._check_input_files(xdata)
            if n_processes == 1 or len(xdata) < 3:
                iter_runner = self._single_iter_offline
            else:
                iter_runner = self._single_iter_mp
        else:
            raise ValueError("xdata must be either a list or a numpy array.")


        best_loss = -np.inf
        for restart in range(n_restarts):
            mix_weights, mu_mix, loss = iter_runner(xdata, tol,
                            max_iter, random_state + restart,
                            model_polishing, n_processes = n_processes,
                            n_threads = n_threads)
            if loss > best_loss:
                best_loss = copy.deepcopy(loss)
                self.mix_weights = mix_weights
                self.mu_mix = mu_mix.clip(min=1e-16)

        if self.mu_mix is None:
            raise ValueError("No restarts converged!")


    def _single_iter_online(self, xdata, tol,
                    max_iter, random_state,
                    model_polishing,
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
            model_polishing (bool): Defaults to False. If True,
                ignore random_state and use existing weights instead
                of reinitializing.
            n_processes (int): This function accepts but does not use
                n_processes (to remaing consistent with the other single_iter
                functions).
            n_threads (int): The number of threads to use on each array.

        Raises:
            ValueError: Raised if unexpected inputs are supplied.
        """
        loss = -np.inf
        if not model_polishing:
            mix_weights, mu_mix = self._get_init_params(random_state)
        else:
            if self.mix_weights is None or self.mu_mix is None:
                raise ValueError("Model polishing specified, but model has not "
                        "been fitted yet.")
            mix_weights, mu_mix = self.mix_weights, self.mu_mix

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
                    model_polishing,
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
            model_polishing (bool): Defaults to False. If True,
                ignore random_state and use existing weights instead
                of reinitializing.
            n_processes (int): This function accepts but does not use
                n_processes (to remaing consistent with the other single_iter
                functions).
            n_threads (int): The number of threads to use on each array.

        Raises:
            ValueError: Raised if unexpected inputs are supplied.
        """
        loss = -np.inf
        if not model_polishing:
            mix_weights, mu_mix = self._get_init_params(random_state)
        else:
            if self.mix_weights is None or self.mu_mix is None:
                raise ValueError("Model polishing specified, but model has not "
                        "been fitted yet.")
            mix_weights, mu_mix = self.mix_weights, self.mu_mix

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
                    model_polishing,
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
            model_polishing (bool): Defaults to False. If True,
                ignore random_state and use existing weights instead
                of reinitializing.
            n_processes (int): The number of processes to use.
            n_threads (int): The number of threads to use on each array
                in each process.

        Raises:
            ValueError: Raised if unexpected inputs are supplied.
        """
        n_processes = min(len(xfiles), n_processes)
        chunk_size = int((len(xfiles) + n_processes - 1) / n_processes)
        print(f"Using MP. Chunk size: {chunk_size} files")
        xchunks = [xfiles[i:i + chunk_size] for i in
                range(0, len(xfiles), chunk_size)]

        loss = -np.inf
        if not model_polishing:
            mix_weights, mu_mix = self._get_init_params(random_state)
        else:
            if self.mix_weights is None or self.mu_mix is None:
                raise ValueError("Model polishing specified, but model has not "
                        "been fitted yet.")
            mix_weights, mu_mix = self.mix_weights, self.mu_mix

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
            ValueError: Raised if unexpected inputs are supplied.
        """
        self._check_input_array(xdata)

        if self.mu_mix is None:
            raise ValueError("Model not fitted yet.")

        ndatapoints = self._get_ndatapoints(xdata)
        #The mu parameters must be copied since multimix_score modifies
        #the mu input in place (to avoid creating an extra copy when
        #multiprocessing is used).
        loglik = multimix_score(xdata, self.mu_mix.copy(), self.mix_weights,
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
            ValueError: Raised if unexpected inputs are supplied.
        """
        self._check_input_files(xfiles)

        if self.mu_mix is None:
            raise ValueError("Model not fitted yet.")
        ndatapoints = self._get_ndatapoints(xfiles)

        if len(xfiles) < 3 or n_processes == 1:
            loglik = self._loglik_offline(xfiles)
        else:
            n_processes = min(len(xfiles), n_processes)
            chunk_size = int((len(xfiles) + n_processes - 1) / n_processes)
            print(f"Using MP. Chunk size: {chunk_size} files")
            xchunks = [xfiles[i:i + chunk_size] for i in
                    range(0, len(xfiles), chunk_size)]
            caller_args = [(xchunk, self.mu_mix.copy(), self.mix_weights.copy(),
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
            ValueError: raised if unexpected inputs are supplied.
        """
        self._check_input_array(xdata)
        if self.mu_mix is None:
            raise ValueError("Model not fitted yet.")
        #The mu parameters must be copied since multimix_score modifies
        #the mu input in place (to avoid creating an extra copy when
        #multiprocessing is used).
        loglik = multimix_score(xdata, self.mu_mix.copy(), self.mix_weights.copy(),
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
            ValueError: Raised if unexpected inputs are supplied.
        """
        self._check_input_files(xfiles)
        if self.mu_mix is None:
            raise ValueError("Model not fitted yet.")

        if len(xfiles) < 3 or n_processes == 1:
            loglik = self._loglik_offline(xfiles)
        else:
            n_processes = min(len(xfiles), n_processes)
            chunk_size = int((len(xfiles) + n_processes - 1) / n_processes)
            print(f"Using MP. Chunk size: {chunk_size} files")
            xchunks = [xfiles[i:i + chunk_size] for i in
                    range(0, len(xfiles), chunk_size)]
            caller_args = [(xchunk, self.mu_mix.copy(), self.mix_weights.copy(),
                                n_threads) for xchunk in xchunks]
            with Pool(n_processes) as mp_pool:
                mp_res = list(mp_pool.starmap(multimix_loglik_offline, caller_args))
            loglik = np.sum(mp_res)

        return 2 * self._get_nparams() - 2 * loglik



    def generate_samples(self, nsamples, random_state = 123):
        """Samples from the distribution to generate new sequences.

        Args:
            nsamples (int): The number of samples to generate.
            random_state (int): The random seed.

        Returns:
            preds (np.ndarray): An array of shape (nsamples,
                self.sequence_length) where each element is a uint8
                ranging from 0 to self.num_possible_items - 1.

        Raises:
            ValueError: Raised if model not fitted yet.
        """
        if self.mu_mix is None or self.mix_weights is None:
            raise ValueError("Model not fitted yet.")
        rng = np.random.default_rng(random_state)
        sample_mat = np.zeros((nsamples, self.sequence_length), dtype = np.uint8)
        for i in range(nsamples):
            cnum = rng.multinomial(1, pvals = self.mix_weights).argmax()
            for j in range(self.sequence_length):
                sample_mat[i,j] = rng.multinomial(1, pvals=self.mu_mix[cnum,j,:]).argmax()
        return sample_mat


    def predict(self, xdata, n_threads = 1):
        """Determine the most probable cluster for each datapoint
        in a numpy array. Note that you should also check the
        overall probability of each datapoint. If a datapoint is
        very different from your training set, it will have
        very low overall probability, but this function will
        still assign it to the most likely cluster -- whichever
        that is -- by default.

        Args:
            xdata (np.ndarray): A numpy array of type np.uint8, shape 2.
            n_threads (int): The number of threads to use.

        Returns:
            preds (np.ndarray): An array of shape (xdata.shape[0])
                containing a number from 0 to self.n_components - 1
                indicating the predicted cluster for each datapoint.

        Raises:
            ValueError: Raised if unexpected inputs are supplied.
        """
        self._check_input_array(xdata)
        if self.mu_mix is None or self.mix_weights is None:
            raise ValueError("Model not fitted yet.")
        #The mu parameters must be copied since multimix_score modifies
        #the mu input in place (to avoid creating an extra copy when
        #multiprocessing is used).
        return multimix_predict(xdata, self.mu_mix.copy(),
                self.mix_weights, n_threads)



    def predict_proba(self, xdata, n_threads = 1, use_mixweights = True):
        """Predict the probability of each datapoint in the input
            array.

        Args:
            xdata (np.ndarray): A numpy array of type np.uint8, shape 2.
            n_threads (int): The number of threads to use.
            use_mixweights (bool): If True, take mixture weights into
                account; otherwise, generate p(x | cluster = a) ignoring
                mixture weights.

        Returns:
            probs (np.ndarray): An array of shape (xdata.shape[0])
                with the calculated probs.

        Raises:
            ValueError: Raised if unexpected inputs are supplied.
        """
        self._check_input_array(xdata)
        if self.mu_mix is None or self.mix_weights is None:
            raise ValueError("Model not fitted yet.")
        #The mu parameters must be copied since multimix_score modifies
        #the mu input in place (to avoid creating an extra copy when
        #multiprocessing is used).
        return multimix_cluster_probs(xdata, self.mu_mix.copy(),
                self.mix_weights, n_threads, use_mixweights)





    def hard_assignment_mp(self, xfiles, n_processes = 1):
        """Hard-assign each sequence to a single cluster, and compile
        statistics on the distribution of amino acids for sequences
        assigned to each cluster. Not currently set up to use
        multithreading but can use multiprocessing. Currently
        accepts only a list of .npy files as input.

        Args:
            xdata (list): A list of file paths to .npy files.
            n_processes (int): The number of processes. If > 1,
                multiprocessing is used. This will increase
                memory consumption since n_processes copies must
                be made of the model parameters.

        Returns:
            cluster_stats (np.ndarray): An array of size (self.n_components,
                self.sequence_length, self.num_possible_items) where each
                element is the number of matching observations in the
                input data. Type uint32.

        Raises:
            ValueError: An error is raised if unexpected inputs are supplied.
        """
        self._check_input_files(xfiles)

        n_proc = min(len(xfiles), n_processes)
        chunk_size = int((len(xfiles) + n_proc - 1) / n_proc)
        print(f"Using MP. Chunk size: {chunk_size} files")
        xchunks = [xfiles[i:i + chunk_size] for i in
                range(0, len(xfiles), chunk_size)]

        cluster_stats = np.zeros_like((self.mu_mix), dtype = np.uint32)

        caller_args = [(xchunk, self.mu_mix.copy(), self.mix_weights.copy())
                for xchunk in xchunks]
        with Pool(n_proc) as mp_pool:
            mp_res = list(mp_pool.starmap(hard_cluster_assign,
                                caller_args))

        for res in mp_res:
            cluster_stats += res

        return cluster_stats


    def _loglik_offline(self, xfiles, n_threads = 1):
        """Generate the log-likelihood of the whole dataset
        for a list of files as input.

        Args:
            xdata (list): A list of file paths to .npy files.
            n_threads (int): the number of threads to use.

        Returns:
            loglik (float): The log-likelihood of the whole
                input dataset.

        Raises:
            ValueError: Raised if unexpected inputs are supplied.
        """
        self._check_input_files(xfiles)
        if self.mu_mix is None:
            raise ValueError("Model not fitted yet.")
        #The mu parameters must be copied since multimix_score modifies
        #the mu input in place (to avoid creating an extra copy when
        #multiprocessing is used).
        return multimix_loglik_offline(xfiles, self.mu_mix.copy(),
                self.mix_weights, n_threads)


    def score(self, xdata, n_threads = 1):
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
            n_threads (int): the number of threads to use.

        Returns:
            loglik (np.ndarray): A float64 array of shape (x.shape[0])
                where each element is the log-likelihood of that
                datapoint given the model.

        Raises:
            ValueError: Raised if unexpected inputs are supplied.
        """
        self._check_input_array(xdata)
        if self.mu_mix is None or self.mix_weights is None:
            raise ValueError("Model not fitted yet.")
        #The mu parameters must be copied since multimix_score modifies
        #the mu input in place (to avoid creating an extra copy when
        #multiprocessing is used).
        return multimix_score(xdata, self.mu_mix.copy(), self.mix_weights, n_threads)


    def masked_score(self, xdata, start_col = 0, end_col = 1, n_threads = 1):
        """Generate the overall log-likelihood of individual datapoints,
        but with a "mask" such that amino acids at the c-terminal or
        n-terminal are masked out and ignored. This is useful primarily
        when there is a large n- or c-terminal deletion and we would
        like to assess the humanness of the remaining sequence ignoring this
        region.

        Args:
            xdata (np.ndarray): An array with the input data,
                of type np.uint8.
            n_threads (int): the number of threads to use.
            start_col (int): The first column of the input to use;
                previous columns are masked.
            end_col (int): The last column of the input to use;
                remaining columns are masked.

        Returns:
            loglik (np.ndarray): A float64 array of shape (x.shape[0])
                where each element is the log-likelihood of that
                datapoint given the model.

        Raises:
            ValueError: Raised if unexpected inputs are supplied.
        """
        self._check_input_array(xdata)
        if self.mu_mix is None or self.mix_weights is None:
            raise ValueError("Model not fitted yet.")
        if start_col < 0 or start_col >= end_col or end_col > self.mu_mix.shape[1]:
            raise ValueError("Inappropriate col start / col end passed.")
        #The mu parameters must be copied since multimix_score modifies
        #the mu input in place (to avoid creating an extra copy when
        #multiprocessing is used).
        return multimix_score_masked(xdata, self.mu_mix.copy(), self.mix_weights, n_threads,
                start_col, end_col)
