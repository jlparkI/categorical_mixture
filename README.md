# categorical_mix
Fast, scalable clustering for fixed length sequences with a simple generative model.

This package is a fairly special-purpose tool designed for fitting multiple sequence alignments
of protein or DNA sequences to a categorical mixture model. (It's possible you could use
this for other tasks, although that's a possibility we've never investigated.) This is a *very*
simple model but for precisely this reason it can sometimes be quite useful -- it's fully
human-interpretable, easy to visualize and can fit a few million sequences very quickly. It's
designed to fit datasets too large to fit in memory.

This package is primarily used by [AntPack](https://github.com/jlparkI/AntPack), which uses it to score
antibody sequences for human-likeness and for other tasks. If you are interested in using it for
some other task, for installation and usage, see [the docs](https://categorical-mix.readthedocs.io/en/latest/).
