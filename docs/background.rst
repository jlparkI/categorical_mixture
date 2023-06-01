Background -- the categorical mixture model
============================================

The categorical mixture is a simple generative
model for aligned sequences / sequences that are
all of the same length / certain kinds of fixed
length-vector data. Despite its simplicity, it
has several advantages:

1) It's simple yet flexible;
2) It's highly scalable. We've fitted this model to
a dataset of over 900 million sequences (>150 billion tokens)
using > 2000 clusters.
3) It's a probability distribution over the sequence
space, so it's both highly interpretable and easily
used to generate new sequences.

So it's definitely not a general-purpose model that you can
use to solve a wide range of problems. At the same time,
however, there are certain kinds
of problems where it's "just right" for what you're
trying to do.

If you encounter any errors, have questions or would like
to see a new feature, please use the project github page:

https://github.com/jlparki/categorical_mixture

Please use Discussions to suggest new features or ask
questions, and use Issues to report a bug.
