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

 my_model = CategoricalMixture(n_components, num_possible_items = 21,
                          sequence_length = 158)

Here `n_components` is the number of clusters. `num_possible_items` is the
number of possible selections for each element in each sequence. (For amino
acids in proteins, for example, this might be 20, while for DNA sequences it
might be 4.) Computational expense scales linearly with number of clusters, so
that 2x the number of clusters roughly doubles fitting time.

There are several kinds of predictions we can
make using a fitted model.

1) The *overall* probability that a
new datapoint was generated by this model.
2) The probability that a new
datapoint was generated by a *specific cluster*
in this model.
3) The *most probable* cluster in the model.

We can use (3) to assign a new datapoint to a
cluster. Bear in mind however: if the probability
of a new datapoint given the model is very low,
you should not assign it to a cluster, because
it's sufficiently different from the training set
that model predictions for this new datapoint are
likely not useful. We'll see how to determine
when we should not make a cluster assignment shortly.

The class methods for fitting the model, scoring or assigning clusters for
new data, and calculating AIC / BIC are described below. (`BIC<https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_
/ `AIC<https://en.wikipedia.org/wiki/Akaike_information_criterion>`_ can be
used to choose the number of clusters; if you're unfamiliar with BIC / AIC,
see the links for an intro). Notice that for
fitting and BIC / AIC scoring, the model has options to use either a list
of files if your data is too large to fit in memory or a single numpy
array as input otherwise. Also note that for fitting and BIC / AIC
calculation there are options to use multiple processes (this is
handled through ``multiprocessing`` in Python) and multiple threads
(this is handled by the C++ code). If you specify multiple threads,
each process will use the number of threads indicated.

For large datasets, it's better to increase the number of processes
rather than the number of threads, because the threads all cooperate
on each block of data as it's loaded, while the processes all work
on separate blocks of data. So increasing the number of processes
provides a much better speed benefit and is preferable. Using
more processes does however increase memory consumption, whereas
using multiple threads does not (because each process is loading
its own chunk of data). So increasing the number of processes
2x will slightly more than double memory consumption.

If your dataset is small (consider anything under one or two million
datapoints to be small for these purposes), none of this is
likely to be a concern.

Once the model is trained, if needed, you can retrieve the per-position
per-element probabilities as ``my_model.mix_weights`` and ``my_model.mu_mix``
respectively for easy plotting / analysis; these are both numpy arrays.

Also note that currently to save the model (aside from pickling it,
which is not always ideal from a security standpoint) you can
call the ``load_params`` function to load saved parameters for
a trained model (the ``mu_mix`` and ``mix_weights``).


.. autoclass:: categorical_mix.CategoricalMixture
   :special-members: __init__
   :members: load_params, fit, predict, score, BIC, BIC_offline, AIC, AIC_offline
