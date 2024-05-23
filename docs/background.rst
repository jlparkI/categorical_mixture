Background -- What this package does
============================================

The categorical mixture is a simple generative
model for aligned sequences / sequences that are
all of the same length / certain kinds of fixed
length-vector data. It has three main advantages:

1) It's simple yet flexible;
2) It's highly scalable. We've fitted this model to
a dataset of over 900 million sequences (>150 billion tokens)
using > 2000 clusters. (This isn't how we usually use this
model, but it's possible if desired.)
3) It's a probability distribution over the sequence
space, so it's both highly interpretable (easy to plot) and
easily used to generate new sequences.

It's a very special-purpose model that only really makes
sense for certain specific problems. On the other hand,
when you do run into one of those problems, it's nice
to have. This package is used
heavily by `AntPack <https://github.com/jlparkI/AntPack>`_
but is also free for independent use if you can find
another use for it.

If you encounter any errors, have questions or would like
to see a new feature, please use the project github page:

https://github.com/jlparki/categorical_mixture

Please use Discussions to suggest new features or ask
questions, and use Issues to report a bug.
