# categorical_mixture
Fast, scalable clustering for up to 1 billion fixed length sequences with a simple generative model.


The categorical_mixture models fixed-length sequences where each element may be up to one of
*x* different choices or items (e.g. DNA or protein sequence alignments). It can generate
new sequences, calculate the probability that a test datapoint could have come from the
distribution it describes, and assign to clusters. It's fairly specialized, not very general
purpose, but there are some types of tasks where it's "just right" for what you're trying to do.

For installation and usage, see the docs.
