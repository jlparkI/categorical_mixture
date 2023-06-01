Fitting a model
================

Fitting a model to a small dataset is easy and fast. For larger
datasets (e.g. > 1 million datapoints), the package provides a
few different options to make the calculations easier to run
in parallel depending on your configuration.

To start, your data must be in one of two formats. It must be *either*

* (1) a Numpy array in memory of type uint8 of shape N x C, where N is the
  number of datapoints and C is the number of elements in each sequence
  (all sequences should be of the same length). If you're working with
  data that is *not* sequence data, C can be 1; *or*
* (2) A list of filepaths to .npy files saved on disk, where each .npy file is
  a numpy array of the same format as the in-memory dataset described
  above.

Either way, each element of the array should be an integer indicating which
of the possible selections for that sequence element is actually present. For
example, if your data is a list of sequences of amino acids of length 5,
and if there are 20 possible amino acids and amino acids S and T are numbered
5 and 6, then the sequence:

STTT

would be stored in the numpy array as:

5666

If your dataset is small, (1) is preferable. If it's large, you can split
it up into chunks of say 10,000 datapoints, save each chunk on disk as a
separate .npy file and use the list of filepaths to those files as your
dataset.

Once you have your data, you can create a model:::

 my_model = MultinoulliMixture(n_components, num_possible_items = 21,
                          sequence_length = 158)

Here `n_components` is the number of clusters. `num_possible_items` is the
number of possible selections for each element in each sequence. (For amino
acids in proteins, for example, this might be 20, while for DNA sequences it
might be 4.)

To fit the model:::

  my_model.fit(xdata, max_iter = 150, tol = 1e-3,
                    n_restarts = 1, random_state = 123,
                    enable_input_checking = True,
                    model_polishing = False,
                    n_processes = 1, n_threads = 1,
                    use_ray = False)

`xdata` here is either a numpy array or a list of filepaths as described above.
`max_iter` is the maximum # of iterations for which fitting is allowed to run.
If the loss function increases by less than `tol` at any point before `max_iter`
is reached, fitting is terminated. `model_polishing` if True indicates that the
model has already been fitted and you just want to "fine-tune" it, e.g. by using
a smaller value for `tol`.

`n_restarts` dictates the number of times the model is refitted to the data;
the best result is saved. Each restart increments the `random_state` by 1.
This is important because fitting uses the EM algorithm, which finds a local
maximum, so to find the best possible outcome usually requires more than 1
restart (typically 3 - 5 is pretty good).

`enable_input_checking` should always be kept as True -- it tells the model to
check your input data before fitting to make sure it's valid. Don't set this
to False unless you have a really good reason to do it; disabling input checking
can result in unanticipated errors if your input data is invalid during fitting.

The remaining options enable parallel computation during fitting, which isn't
important if your dataset is small but is very important if your dataset is
large.

Evaluating a model
------------------

Frequently we need to evaluate a number of different possible cluster numbers
to determine which is most suitable. We can use Bayes Information Criterion
(BIC) and Akaike Information Criterion (AIC):::

  my_model.AIC(xdata, n_threads = 1)
  my_model.AIC_offline(xdata, n_processes = 1, n_threads = 1)
  my_model.BIC(xdata, n_threads = 1)
  my_model.BIC_offline(xdata, n_processes = 1, n_threads = 1)

Use the `offline` functions if your data is a list of .npy files rather
than a numpy array. For more about `n_processes` and `n_threads`, see
below.

A model with a smaller AIC / BIC is better. Note that BIC is more conservative,
it tends to favor models with a smaller number of parameters. BIC will typically show
a global minimum, where all possible number of cluster settings yield a higher
BIC than this minimum. AIC will *sometimes* exhibit a global minimum but more
frequently exhibits an "elbow" with an obvious kink in the plot, and then the
location of the "elbow" is the best number of clusters. See the example notebook
for an illustration of how to use these to choose a good number of clusters for
your training dataset.


Multiprocessing and multithreading
------------------------------------

For data in memory, the model can use multithreading as specified by
`n_threads`. For data saved on disk, the model can use either:

* multiprocessing, which divides the list of files into chunks and runs each
  simultaneously using a separate process;
* multithreading, which runs multiple threads on each chunk of data as it's
  loaded into memory;
* multiprocessing + multithreading, which divides the list of files into chunks
  and runs each simultaneously using a separate process; each process uses
  `n_threads`; or
* the Ray Python package, which behaves similarly to multiprocessing but may
  be more useful in some cluster environments. If Ray is used multithreading
  is disabled.

Multiprocessing & Ray have higher memory consumption than multithreading, because
they have to make multiple copies of the model parameters. Multithreading has
no extra memory consumption. On the other hand, multithreading doesn't yield
quite as large of a speed gain as multiprocessing or Ray.

To use multithreading (with or without multiprocessing), set `n_threads` to
a value > 1. To use multiprocessing, set `n_processes` to a value > 1. To
use Ray, set `n_processes` to a value > 1 and set `use_ray` to True.
