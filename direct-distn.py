"""
Try to estimate a distribution over a simplex using numerical methods.

This approximation tries to adjust weights of each piece
of a discretized simplex so that certain integrals over the simplex have 
the correct values.
Symmetry constraints are also imposed.
A sum of squares of errors with respect to the constraints is
the objective function which will be minimized.
"""

import numpy as np
import scipy.optimize
import scipy.stats
import algopy

import multinomstate

def get_beta_approx(npoints, alpha):
    """
    These are normalized likelihoods from a beta function pdf.
    @param N: evaluate this many equally spaced points on [0, 1]
    @param alpha: beta distribution concentration parameter
    @return: a discrete distribution
    """
    x = np.linspace(0, 1, npoints)
    scipy.stats.beta.pdf(x, alpha, alpha)
    return x / np.sum(x)

def eval_f(
        M, T,
        X,
        ):
    """
    This algopy-enabled function should be callable from leastsq.
    The first group of args should be partially evaluated.
    The last arg is a vector of logs of parameter values.
    @param X: parameter values
    @return: vector of errors whose squares are to be minimized
    """
    nstates = len(M)

    # unpack the parameter values into a distribution
    v = algopy.zeros(nstates, dtype=X)
    v[:-1] = algopy.exp(X)
    v[-1] = 1.0
    v = v / algopy.sum(v)

    # compute the symmetry errors

    # compute the distribution errors

def main():

    N = 10

    result = scipy.optimize.leastsq(
            f,
            x0,
            Dfun=g,
            )
    print result

if __name__ == '__main__':
    main()

