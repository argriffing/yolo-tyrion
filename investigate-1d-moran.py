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

def get_mathematica_matrix_string(M):
    elements = []
    for row in M:
        s = '{' + ','.join(str(x) for x in row) + '}'
        elements.append(s)
    return '{' + ','.join(s for s in elements) + '}'

def main(args):
    alpha = args.alpha
    if alpha == math.floor(alpha):
        alpha = int(alpha)
    N = args.N
    pre_Q = np.zeros((N+1, N+1), dtype=int)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=2.0, type=float,
            help='concentration parameter of beta distribution')
    parser.add_argument('--N', default=5, type=int,
            help='population size'),
    main(parser.parse_args())

