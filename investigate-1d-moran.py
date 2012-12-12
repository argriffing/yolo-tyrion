"""
Look in detail at finite popsize 2-allele Moran equilibrium.

The equilibrium approximates a beta distribution.
But I want to understand the nature of the approximation error
so that I could eventually determine the correct analog of a Dirichlet
distribution for an infinite popsize 4-allele parent-dependent mutation process.
"""

import argparse
import math

import numpy as np
import scipy.linalg
import scipy.stats

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

def get_mathematica_matrix_string(M):
    elements = []
    for row in M:
        s = '{' + ','.join(str(x) for x in row) + '}'
        elements.append(s)
    return '{' + ','.join(s for s in elements) + '}'

def exact_alpha_3_N_7(x):
    # see http://oeis.org/A098358/table
    return (x-10)*(x-9)*(x+0)*(x+1) / math.factorial(4)

def exact_scaled_distn_alpha_3(alpha, N):
    """
    This was found by abusing wolfram alpha and oeis.
    http://oeis.org/A098358/table
    @param alpha: this should be positive integer
    @param N: this should be positive
    """
    if alpha != 3:
        raise Exception
    arr = np.zeros(N+3, dtype=int)
    for i in range(N+3):
        j = N + 2 - i
        tri_i = (i * (i + 1)) / 2
        tri_j = (j * (j + 1)) / 2
        arr[i] = tri_i * tri_j
    return arr[1:-1]

def exact_scaled_distn_alpha_4(alpha, N):
    """
    This uses the tetrahedral generalization of triangular numbers.
    """
    if alpha != 4:
        raise Exception
    arr = np.zeros(N+3, dtype=int)
    for i in range(N+3):
        j = N + 2 - i
        tet_i = choose(i+2, 3)
        tet_j = choose(j+2, 3)
        arr[i] = tet_i * tet_j
    return arr[1:-1]

def exact_scaled_distn(alpha, N):
    """
    Generalize to any positive integer alpha.
    Note that this is a convolution.
    The trick is apparently to treat the beta density
    as the product of two generating functions.
    Then I can write this product as the
    http://en.wikipedia.org/wiki/Cauchy_product
    of the two series associated with the generating functions.
    """
    arr = np.zeros(N+3, dtype=float)
    for i in range(N+3):
        j = N + 2 - i
        #a = choose(i+alpha-2, alpha-1)
        #b = choose(j+alpha-2, alpha-1)
        a = scipy.special.binom(i+alpha-2, alpha-1)
        b = scipy.special.binom(j+alpha-2, alpha-1)
        """
        a = scipy.special.gamma(i+alpha-1) / (
                scipy.special.gamma(alpha) *
                scipy.special.gamma(i + alpha - 2 - (alpha - 1)))
        b = scipy.special.gamma(j+alpha-1) / (
                scipy.special.gamma(alpha) *
                #scipy.special.gamma(j-1))
                scipy.special.gamma(j + alpha - 2 - (alpha - 1)))
        """
        # use the generalization of
        # triangular, tetrahedral, etc. numbers
        # implied by the generating function 1/(1-x)^n
        #a = -((-1)**i)*scipy.special.binom(-alpha, i-1)
        #b = -((-1)**j)*scipy.special.binom(-alpha, j-1)
        arr[i] = a * b
    return arr[1:-1]

def main(args):
    alpha = args.alpha
    if alpha == math.floor(alpha):
        alpha = int(alpha)
    N = args.N
    print 'alpha:', alpha
    print 'N:', N
    print
    pre_Q = np.zeros((N+1, N+1), dtype=type(alpha))
    print 'dtype of pre-rate-matrix:'
    print pre_Q.dtype
    print
    # Construct a tridiagonal rate matrix.
    # This rate matrix is scaled by pop size so its entries become huge,
    # but this does not affect the equilibrium distribution.
    for i in range(N+1):
        if i > 0:
            # add the drift rate
            pre_Q[i, i-1] += (i*(N-i))
            # add the mutation rate
            pre_Q[i, i-1] += alpha * i
        if i < N:
            # add the drift rate
            pre_Q[i, i+1] += ((N-i)*i)
            # add the mutation rate
            pre_Q[i, i+1] += alpha * (N-i)
    Q = pre_Q - np.diag(np.sum(pre_Q, axis=1))
    # define the domain
    #x = np.linspace(0, 1, N+2)
    #x = 0.5 * (x[:-1] + x[1:])
    #x = np.linspace(0, 1, N+1)
    x = np.linspace(0, 1, N+3)[1:-1]
    y = scipy.stats.beta.pdf(x, alpha, alpha)
    y /= scipy.linalg.norm(y)
    # pick out the correct eigenvector
    W, V = scipy.linalg.eig(Q.T)
    w, v = min(zip(np.abs(W), V.T))
    #
    print 'rate matrix:'
    print Q
    print
    print 'transpose of rate matrix:'
    print Q.T
    print
    print 'transpose of rate matrix in mathematica notation:'
    print get_mathematica_matrix_string(Q.T)
    print
    #print 'eigendecomposition of transpose of rate matrix:'
    print 'abs eigenvector corresponding to smallest abs eigenvalue:'
    print np.abs(v)
    print
    #print 'domain:'
    #print x
    print
    print 'beta distribution samples normalized to unit norm:'
    print y
    print
    #print 'an exact unnormalized solution for alpha=3, N=7:'
    #print exact_alpha_3_N_7(np.arange(1, 8))
    #print
    if False:
    #if alpha == 3:
        v_exact = exact_scaled_distn_alpha_3(alpha, N)
        v_exact_normalized = v_exact / scipy.linalg.norm(v_exact)
    elif False:
    #elif alpha == 4:
        v_exact = exact_scaled_distn_alpha_4(alpha, N)
        v_exact_normalized = v_exact / scipy.linalg.norm(v_exact)
    else:
        v_exact = exact_scaled_distn(alpha, N)
        v_exact_normalized = v_exact / scipy.linalg.norm(v_exact)
    print 'exact unnormalized solution:'
    print v_exact
    print
    print 'exact normalized solution:'
    print v_exact_normalized
    print

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=2.0, type=float,
            help='concentration parameter of beta distribution')
    parser.add_argument('--N', default=5, type=int,
            help='population size'),
    main(parser.parse_args())
