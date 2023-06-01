Installation
================

To install, categorical_mixture must be compiled from source. Because it is
a little specialized, it is not currently distributed on PyPi
(although it can easily be if there is sufficient interest). To install it,
clone the git repo then install with a virtual environment or conda
environment already active:::

  git clone https://github.com/jlparkI/categorical_mixture
  cd categorical_mixture
  python setup.py install

Requirements are Python 3, scipy, numpy and Cython (Ray is an optional
dependency that's useful on some cluster environments).

If you encounter any errors, please report at the project github page.
