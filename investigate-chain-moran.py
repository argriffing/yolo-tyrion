"""
Look in detail at finite-N A--B--C mutation continuous time Moran process.

This is the simplest parent-dependent mutational process that I can imagine.
"""

import argparse
import math

import numpy as np
import scipy.linalg
import scipy.stats

import wrightcore
import multinomstate
import MatrixUtil

def main(args):
    alpha = args.alpha
    N = args.N
    k = 3
    print 'alpha:', alpha
    print 'N:', N
    print 'k:', k
    print
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    R_mut = wrightcore.create_mutation_abc(M, T)
    R_drift = wrightcore.create_moran_drift_rate_k3(M, T)
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
    print MatrixUtil.m_to_mathematica_string(Q.T.astype(int))
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
