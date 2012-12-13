"""
Look in detail at finite popsize 4-allele continuous time Moran process.

The longer term goal is to find a slick way to write the limit of the
joint distribution over allele distributions
as the population size approaches infinity.
A short step towards this goal is to investigate the distributions
for small population sizes.
"""

import argparse
import math

import numpy as np
import scipy.linalg
import scipy.stats

import wrightcore
import multinomstate

def get_mathematica_matrix_string(M):
    elements = []
    for row in M:
        s = '{' + ','.join(str(x) for x in row) + '}'
        elements.append(s)
    return '{' + ','.join(s for s in elements) + '}'

def main(args):
    alpha = args.alpha
    N = args.N
    k = 4
    print 'alpha:', alpha
    print 'N:', N
    print 'k:', k
    print
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    R_mut = wrightcore.create_mutation(M, T)
    R_drift = wrightcore.create_moran_drift_rate_k4(M, T)
    Q = alpha * R_mut + R_drift
    # pick out the correct eigenvector
    W, V = scipy.linalg.eig(Q.T)
    w, v = min(zip(np.abs(W), V.T))
    print 'rate matrix:'
    print Q
    print
    print 'transpose of rate matrix:'
    print Q.T
    print
    print 'eigendecomposition of transpose of rate matrix as integers:'
    print scipy.linalg.eig(Q.T)
    print
    print 'transpose of rate matrix in mathematica notation:'
    print get_mathematica_matrix_string(Q.T.astype(int))
    print
    print 'abs eigenvector corresponding to smallest abs eigenvalue:'
    print np.abs(v)
    print


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=2.0, type=float,
            help='concentration parameter of beta distribution')
    parser.add_argument('--N', default=5, type=int,
            help='population size'),
    main(parser.parse_args())
