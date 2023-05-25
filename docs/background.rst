Background -- the multinomial mixture model
============================================

The multinomial mixture is a simple generative
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

So, despite its limitations, there are certain kinds
of problems where it's "just right" for what you're
trying to do.


