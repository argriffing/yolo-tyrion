"""
Bernstein approximation theory is probably somehow applicable to this puzzle.

But I am not sure how.
The top of page 103 of
A Generalized Bernstein Approximation Theorem
by Miloslav Duchon
may give a relevant generalization to approximations on a simplex.
"""

import argparse
import math

import numpy as np
import scipy.linalg
import scipy.stats

import MatrixUtil

def x_get_bernstein_approximation(alpha, beta, N):
    """
    In this example we approximate the beta distribution.
    B_n(f)(x) = sum_{v=0}^n ( f(v/n) b_{v,n}(x) )
    """
    n = N
    X = np.linspace(0, 1, n+3)[1:-1]
    Y = np.zeros(n+1)
    for index in range(n+1):
        x = X[index]
        y = 0
        # compute the bernstein polynomial approximation for x
        for k in range(n+1):
            b1 = x**k
            b2 = (1-x)**(n-k)
            b3 = scipy.special.binom(n, k)
            bnk = b1 * b2 * b3
            b = scipy.stats.beta.pdf(k / float(n), alpha, beta)
            y += bnk * b
        Y[index] = y
    return Y

def get_bernstein_approximation(alpha, beta, N):
    """
    In this example we approximate the beta distribution.
    B_n(f)(x) = sum_{v=0}^n ( f(v/n) b_{v,n}(x) )
    """
    n = N+2
    X = np.linspace(0, 1, n+1)
    Y = np.zeros(n+1)
    for index in range(n+1):
        x = X[index]
        y = 0
        # compute the bernstein polynomial approximation for x
        for k in range(n+1):
            #
            b1 = x**k
            b2 = (1-x)**(n-k)
            b3 = scipy.special.binom(n, k)
            bnk = b1 * b2 * b3
            b = scipy.stats.beta.pdf(k / float(n), alpha, beta)
            y += bnk * b
            #
            #y += (x**(k+alpha-1) * (1-x)**(n-k+beta-1)) / scipy.special.beta(
                    #alpha, beta)
        Y[index] = y
    return Y[1:-1]

def exact_scaled_distn_asymmetric(alpha, beta, N):
    arr = np.zeros(N+1, dtype=float)
    for i in range(N+1):
        a = scipy.special.binom(i+alpha-1, alpha-1)
        b = scipy.special.binom(N-i+beta-1, beta-1)
        arr[i] = a * b
    return arr

def main(args):
    alpha = args.alpha
    beta = args.beta
    if alpha is None:
        alpha = beta
    if beta is None:
        beta = alpha
    if alpha == math.floor(alpha) and beta == math.floor(beta):
        alpha = int(alpha)
        beta = int(beta)
    N = args.N
    print 'alpha:', alpha
    print 'beta:', alpha
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
            pre_Q[i, i-1] += beta * i
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
    #
    #x = np.linspace(0, 1, N+3)[1:-1]
    #y = scipy.stats.beta.pdf(x, alpha, alpha)
    #y /= scipy.linalg.norm(y)
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
    print MatrixUtil.m_to_mathematica_string(Q.T)
    print
    print 'abs eigenvector corresponding to smallest abs eigenvalue:'
    print np.abs(v)
    print
    v_exact = exact_scaled_distn_asymmetric(alpha, beta, N)
    v_exact_normalized = v_exact / scipy.linalg.norm(v_exact)
    print 'exact unnormalized solution:'
    print v_exact
    print
    print 'exact normalized solution:'
    print v_exact_normalized
    print
    v_bernstein = get_bernstein_approximation(alpha, beta, N)
    v_bernstein_normalized = v_bernstein / scipy.linalg.norm(v_bernstein)
    print 'bernstein unnormalized solution:'
    print v_bernstein
    print
    print 'bernstein normalized solution:'
    print v_bernstein_normalized
    print
    v_bernstein = x_get_bernstein_approximation(alpha, beta, N)
    v_bernstein_normalized = v_bernstein / scipy.linalg.norm(v_bernstein)
    print 'bernstein unnormalized solution:'
    print v_bernstein
    print
    print 'bernstein normalized solution:'
    print v_bernstein_normalized
    print


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float,
            help='alpha parameter of beta distribution')
    parser.add_argument('--beta', type=float,
            help='beta parameter of beta distribution')
    parser.add_argument('--N', default=5, type=int,
            help='population size'),
    main(parser.parse_args())
