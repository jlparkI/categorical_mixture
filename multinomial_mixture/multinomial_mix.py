"""Implements the AntibodyMultiMixture class, for all operations involved
in fitting the mixture model and in generating predictions for new
datapoints."""
import time
import copy
from multiprocessing import Pool
import numpy as np
import ray

from core_cpu_func_wrappers import em_short_single_call, multimix_predict
from core_cpu_func_wrappers import multimix_loglik_offline, multimix_score, multimix_cluster_probs
from core_cpu_func_wrappers import em_ray_call, multimix_cluster_probs_no_mixweight
from core_cpu_func_wrappers import hard_cluster_assign

@ray.remote
def ray_caller(xchunk, mix_weights, mu):
    """A simple wrapper for the C extension that ray can use."""
    res = em_ray_call(xchunk, mix_weights.copy(), mu.copy())
    return res


class AntibodyMultiMixture:

    def __init__(self, n_components, aa_dim = 21, num_aas = 158, n_threads = 1,
            n_processes = 1, use_ray = False):
        self.mix_weights = None
        self.mu = None
        self.K = n_components
        self.aa_dim = aa_dim
        self.num_aas = num_aas
        self.n_threads = n_threads
        self.n_processes = n_processes
        self.use_ray = use_ray


    def get_ndatapoints(self, xfiles):
        """Quickly count the number of datapoints in a list
        of files without reading any of them into memory."""
        ndatapoints = 0
        for xfile in xfiles:
            with open(xfile, 'rb') as fhandle:
                _, _ = np.lib.format.read_magic(fhandle)
                xshape, _, _ = np.lib.format.read_array_header_1_0(fhandle)
                ndatapoints += xshape[0]
        return ndatapoints


    def get_init_params(self, random_state):
        """Initializes the model using the random seed to generate
        starting parameters."""
        rng = np.random.default_rng(random_state)
        mix_weights = rng.uniform(size = self.K)
        mix_weights /= mix_weights.sum()
        mu = rng.uniform(size = (self.K, self.num_aas, self.aa_dim))
        mu /= mu.sum(axis=2)[:,:,None]
        return mix_weights, mu


    def check_input_data(self, xlist):
        """Checks an input list of files to make sure that all have
        the correct format to ensure that no problems will be encountered
        during fitting."""
        for xfile in xlist:
            x = np.load(xfile)
            if np.max(x) > self.aa_dim or np.min(x) < 0:
                raise ValueError(f"Values in {xfile} are out of range.")
            if x.dtype != "uint8":
                raise ValueError(f"Unexpected datatype for {xfile}.")
            if len(x.shape) != 2:
                raise ValueError(f"Unexpected shape for {xfile}.")
            if x.shape[1] != self.num_aas or x.shape[0] < 1:
                raise ValueError(f"Unexpected shape for {xfile}.")


    def fit(self, xfiles, max_iter, tol,
                n_restarts, random_state = 123,
                enable_input_checking = True,
                model_polishing = False):
        """Fits the model to a list of input files, potentially using
        multiple restarts if so specified. (In practice given our
        cluster configuration we have found it more efficient to run
        fit with one restart per task and run multiple tasks so that
        multiple restarts were run in parallel, but you can also use
        this function to run multiple restarts directly.)"""
        if self.use_ray:
            iter_runner = self.single_iter_ray
        elif self.n_processes == 1 or len(xfiles) < 3:
            iter_runner = self.single_iter
        else:
            iter_runner = self.single_iter_mp

        best_loss = -np.inf
        if enable_input_checking:
            self.check_input_data(xfiles)

        for restart in range(n_restarts):
            mix_weights, mu, loss = iter_runner(xfiles, tol,
                                        max_iter,
                                        random_state + restart,
                                        model_polishing)
            if loss > best_loss:
                best_loss = loss
                self.mix_weights = mix_weights
                self.mu = mu.clip(min=1e-16)

        if self.mu is None:
            raise ValueError("No restarts converged!")



    def single_iter(self, xfiles, tol,
                    max_iter, random_state,
                    model_polishing):
        """Fit the input list of xfiles once (a single restart),
        without using any kind of multiprocessing (files are processed
        serially, although multithreading is used in the wrapped C
        code to process each file)."""
        em_caller = em_short_single_call

        loss = -np.inf
        if not model_polishing:
            mix_weights, mu = self.get_init_params(random_state)
        else:
            if self.mix_weights is None or self.mu is None:
                raise ValueError("Model polishing specified, but model has not "
                        "been fitted yet.")
            mix_weights, mu = self.mix_weights, self.mu
        for i in range(max_iter):
            prev_loss = copy.copy(loss)
            mix_weights, loss, mu, net_resp, ndpoints = em_caller(xfiles,
                                mix_weights, mu, self.n_threads)

            mix_weights /= net_resp
            loss /= ndpoints
            mu /= mu.sum(axis=2)[:,:,None].clip(min=1)

            print("Loss: %s"%loss, flush=True)
            if np.abs(loss - prev_loss) < tol:
                break

        print(f"Iterations: {i}****************\n")
        return mix_weights, mu, loss



    def single_iter_mp(self, xfiles, tol,
                    max_iter, random_state,
                    model_polishing):
        """Fit the input list of xfiles once (a single restart),
        using multiprocessing to divide the list of files up
        into separate jobs (each job runs one sub-list of files)."""
        em_caller = em_short_single_call
        n_processes = min(len(xfiles), self.n_processes)
        chunk_size = int((len(xfiles) + n_processes - 1) / n_processes)
        print(f"Using MP. Chunk size: {chunk_size} files")
        xchunks = [xfiles[i:i + chunk_size] for i in
                range(0, len(xfiles), chunk_size)]

        loss = -np.inf
        if not model_polishing:
            mix_weights, mu = self.get_init_params(random_state)
        else:
            if self.mix_weights is None or self.mu is None:
                raise ValueError("Model polishing specified, but model has not "
                        "been fitted yet.")
            mix_weights, mu = self.mix_weights, self.mu

        
        for i in range(max_iter):
            prev_loss = copy.copy(loss)
            caller_args = [(xchunk, mix_weights.copy(), mu.copy(),
                                self.n_threads) for xchunk in xchunks]
            with Pool(n_processes) as mp_pool:
                mp_res = [result for result in mp_pool.starmap(em_caller, caller_args)]

            mix_weights = np.zeros(mp_res[0][0].shape)
            mu = np.zeros(mp_res[0][2].shape)
            loss = 0
            for res in mp_res:
                mix_weights += res[0]
                mu += res[2]
                loss += res[1]

            mix_weights /= np.sum([res[3] for res in mp_res])
            loss /= np.sum([res[4] for res in mp_res])
            mu /= mu.sum(axis=2)[:,:,None].clip(min=1)

            print("Loss: %s"%loss, flush=True)
            if np.abs(loss - prev_loss) < tol:
                break
        print(f"Iterations: {i}****************\n")
        return mix_weights, mu, loss



    def single_iter_ray(self, xfiles, tol,
                    max_iter, random_state,
                    model_polishing):
        """Fit the input list of xfiles once (a single restart),
        using the ray library to divide the list of files up
        into separate jobs (each job runs one sub-list of files)."""
        ray.init()
        print(f"Using ray.", flush=True)
        n_processes = min(len(xfiles), self.n_processes)
        chunk_size = int((len(xfiles) + n_processes - 1) / n_processes)
        xchunks = [xfiles[i:i + chunk_size] for i in
                range(0, len(xfiles), chunk_size)]

        loss = -np.inf
        if not model_polishing:
            mix_weights, mu = self.get_init_params(random_state)
        else:
            if self.mix_weights is None or self.mu is None:
                raise ValueError("Model polishing specified, but model has not "
                        "been fitted yet.")
            mix_weights, mu = self.mix_weights, self.mu

        for i in range(max_iter):
            prev_loss = copy.copy(loss)
            futures = [ray_caller.remote(xchunk, mix_weights, mu) for xchunk in xchunks]
            mp_res = ray.get(futures)

            mix_weights = np.zeros(mp_res[0][0].shape)
            mu = np.zeros(mp_res[0][2].shape)
            loss = 0
            for res in mp_res:
                mix_weights += res[0]
                mu += res[2]
                loss += res[1]

            mix_weights /= np.sum([res[3] for res in mp_res])
            loss /= np.sum([res[4] for res in mp_res])
            mu /= mu.sum(axis=2)[:,:,None].clip(min=1)

            print(f"Loss: {loss}", flush=True)
            if np.abs(loss - prev_loss) < tol:
                break
        print(f"Iterations: {i}****************\n")
        return mix_weights, mu, loss




    def BIC_offline(self, xfiles):
        """Calculate the Bayes information criterion for a list
        of files as input. Multiprocessing is not used, although
        multithreading can be used in the wrapped c code."""
        if self.mu is None:
            raise ValueError("Model not fitted yet.")
        ndatapoints = self.get_ndatapoints(xfiles)

        if len(xfiles) < 3 or self.n_processes == 1:
            loglik = self.loglik_offline(xfiles)
        else:
            n_processes = min(len(xfiles), self.n_processes)
            chunk_size = int((len(xfiles) + n_processes - 1) / n_processes)
            print(f"Using MP. Chunk size: {chunk_size} files")
            xchunks = [xfiles[i:i + chunk_size] for i in
                    range(0, len(xfiles), chunk_size)]
            caller_args = [(xchunk, self.mu.copy(), self.mix_weights.copy(),
                                self.n_threads) for xchunk in xchunks]
            with Pool(n_processes) as mp_pool:
                mp_res = [result for result in
                        mp_pool.starmap(multimix_loglik_offline, caller_args)]
            loglik = np.sum(mp_res)

        nparams = self.K - 1 + self.K * ((self.aa_dim - 1) * self.num_aas)
        bic = nparams * np.log(ndatapoints) - 2 * loglik
        return bic


    def AIC_offline(self, xfiles):
        ndatapoints = self.get_ndatapoints(xfiles)
        loglik = self.loglik_offline(xfiles)
        nparams = self.K - 1 + self.K * ((self.aa_dim - 1) * self.num_aas)
        aic = 2 * nparams - 2 * loglik
        return aic


    def generate_samples(self, nsamples, random_state = 123):
        """Samples from the distribution to generate new sequences."""
        rng = np.random.default_rng(random_state)
        sample_mat = np.zeros((nsamples, self.num_aas, self.aa_dim))
        for i in range(nsamples):
            cnum = rng.multinomial(1, pvals = self.mix_weights).argmax()
            for j in range(self.num_aas):
                amino_acid = rng.multinomial(1, pvals=self.mu[cnum,j,:]).argmax()
                sample_mat[i,j,amino_acid] = 1
        return sample_mat


    def predict(self, x, n_threads = 1):
        if self.mu is None:
            raise ValueError("Model not fitted yet.")
        return multimix_predict(x, self.mu.copy(),
                self.mix_weights, n_threads)

    def predict_proba(self, x, n_threads = 1):
        """Predict the probability of each datapoint for
        each cluster."""
        if self.mu is None:
            raise ValueError("Model not fitted yet.")
        return multimix_cluster_probs(x, self.mu.copy(),
                self.mix_weights, n_threads)

    def predict_proba_no_mixweight(self, x, n_threads = 1):
        """Predict the probability of each datapoint for each
        cluster, ignoring the mixture weights."""
        if self.mu is None:
            raise ValueError("Model not fitted yet.")
        return multimix_cluster_probs_no_mixweight(x, self.mu.copy(),
                self.mix_weights, n_threads)


    def hard_assignment_mp(self, xfiles, n_processes):
        """Hard-assign each sequence to a single cluster, and compile
        statistics on the distribution of amino acids for sequences
        assigned to each cluster."""
        n_proc = min(len(xfiles), n_processes)
        chunk_size = int((len(xfiles) + n_proc - 1) / n_proc)
        print(f"Using MP. Chunk size: {chunk_size} files")
        xchunks = [xfiles[i:i + chunk_size] for i in
                range(0, len(xfiles), chunk_size)]

        cluster_stats = np.zeros_like((self.mu), dtype = np.uint32)

        caller_args = [(xchunk, self.mu.copy(), self.mix_weights.copy(),
                        self.n_threads) for xchunk in xchunks]
        with Pool(n_proc) as mp_pool:
            mp_res = list(mp_pool.starmap(hard_cluster_assign,
                                caller_args))

        for res in mp_res:
            cluster_stats += res

        return cluster_stats


    def loglik_offline(self, xfiles):
        """Generate the log-likelihood of the whole dataset
        for a list of files as input."""
        if self.mu is None:
            raise ValueError("Model not fitted yet.")
        return multimix_loglik_offline(xfiles, self.mu.copy(),
                self.mix_weights, self.n_threads)


    def score(self, x, n_threads = 1):
        """Generate the log-likelihood of individual datapoints."""
        if self.mu is None:
            raise ValueError("Model not fitted yet.")
        return multimix_score(x, self.mu.copy(),
                self.mix_weights, n_threads)
