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
import scipy.sparse
import scipy.sparse.linalg

import wrightcore
import multinomstate
import MatrixUtil

def _coo_helper(coo_i, coo_j, coo_r, i, j, r):
    """
    Update the lists which will be used to construct a scipy.sparse.coo_matrix.
    @param coo_i: list of source state indices
    @param coo_j: list of sink state indices
    @param coo_r: list of rates
    @param i: source state index
    @param j: sink state index
    @param r: rate from source to sink
    """
    # add to rate from source to sink
    coo_i.append(i)
    coo_j.append(j)
    coo_r.append(r)
    # add to rate out of the source
    coo_i.append(i)
    coo_j.append(i)
    coo_r.append(-r)


def create_coo_moran(M, T, alpha):
    """
    Construct the sparse Moran rate matrix.
    Initially build it as a coo_matrix,
    which presumably is efficient to transpose and to convert to a csr_matrix.
    @param M: index to state description
    @param T: state description to index
    @return: scipy.sparse.coo_matrix
    """
    nstates = M.shape[0]
    ci = []
    cj = []
    cr = []
    # add the mutation component
    for i, (AB, Ab, aB, ab) in enumerate(M):
        if AB > 0:
            _coo_helper(ci, cj, cr, i, T[AB-1, Ab+1, aB,   ab  ], alpha*AB)
            _coo_helper(ci, cj, cr, i, T[AB-1, Ab,   aB+1, ab  ], alpha*AB)
        if Ab > 0:
            _coo_helper(ci, cj, cr, i, T[AB+1, Ab-1, aB,   ab  ], alpha*Ab)
            _coo_helper(ci, cj, cr, i, T[AB,   Ab-1, aB,   ab+1], alpha*Ab)
        if aB > 0:
            _coo_helper(ci, cj, cr, i, T[AB+1, Ab,   aB-1, ab  ], alpha*aB)
            _coo_helper(ci, cj, cr, i, T[AB,   Ab,   aB-1, ab+1], alpha*aB)
        if ab > 0:
            _coo_helper(ci, cj, cr, i, T[AB,   Ab+1, aB,   ab-1], alpha*ab)
            _coo_helper(ci, cj, cr, i, T[AB,   Ab,   aB+1, ab-1], alpha*ab)
    # add the drift component
    for i, (X, Y, Z, W) in enumerate(M):
        if X > 0:
            _coo_helper(ci, cj, cr, i, T[X-1, Y+1, Z,   W  ], X*Y)
            _coo_helper(ci, cj, cr, i, T[X-1, Y,   Z+1, W  ], X*Z)
            _coo_helper(ci, cj, cr, i, T[X-1, Y,   Z,   W+1], X*W)
        if Y > 0:
            _coo_helper(ci, cj, cr, i, T[X+1, Y-1, Z,   W  ], Y*X)
            _coo_helper(ci, cj, cr, i, T[X,   Y-1, Z+1, W  ], Y*Z)
            _coo_helper(ci, cj, cr, i, T[X,   Y-1, Z,   W+1], Y*W)
        if Z > 0:
            _coo_helper(ci, cj, cr, i, T[X+1, Y,   Z-1, W  ], Z*X)
            _coo_helper(ci, cj, cr, i, T[X,   Y+1, Z-1, W  ], Z*Y)
            _coo_helper(ci, cj, cr, i, T[X,   Y,   Z-1, W+1], Z*W)
        if W > 0:
            _coo_helper(ci, cj, cr, i, T[X+1, Y,   Z,   W-1], W*X)
            _coo_helper(ci, cj, cr, i, T[X,   Y+1, Z,   W-1], W*Y)
            _coo_helper(ci, cj, cr, i, T[X,   Y,   Z+1, W-1], W*Z)
    # return the coo_matrix
    return scipy.sparse.coo_matrix(
            (cr, (ci, cj)), (nstates, nstates), dtype=float)

def main(args):
    alpha = args.alpha
    N = args.N
    k = 4
    print 'alpha:', alpha
    print 'N:', N
    print 'k:', k
    print
    print 'defining the state vectors...'
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    if args.dense:
        print 'constructing dense mutation rate matrix...'
        R_mut = wrightcore.create_mutation(M, T)
        print 'constructing dense drift rate matrix...'
        R_drift = wrightcore.create_moran_drift_rate_k4(M, T)
        Q = alpha * R_mut + R_drift
        # pick out the correct eigenvector
        print 'taking eigendecomposition of dense rate matrix...'
        W, V = scipy.linalg.eig(Q.T)
        w, v = min(zip(np.abs(W), V.T))
        print 'rate matrix:'
        print Q
        print
        print 'transpose of rate matrix:'
        print Q.T
        print
        print 'eigendecomposition of transpose of rate matrix as integers:'
        print (W, V)
        print
        #print 'transpose of rate matrix in mathematica notation:'
        #print MatrixUtil.m_to_mathematica_string(Q.T.astype(int))
        #print
        print 'abs eigenvector corresponding to smallest abs eigenvalue:'
        print np.abs(v)
        print
    if args.sparse:
        print 'constructing sparse coo mutation+drift rate matrix...'
        R_coo = create_coo_moran(M, T, alpha)
        print 'converting to sparse csr transpose rate matrix...'
        RT_csr = scipy.sparse.csr_matrix(R_coo.T)
        print 'computing an eigenpair of the sparse matrix using shift-invert...'
        W, V = scipy.sparse.linalg.eigs(RT_csr, k=1, sigma=1)
        #print 'dense form of sparsely constructed matrix:'
        #print RT_csr.todense()
        #print
        print 'sparse eigenvalues:'
        print W
        print
        print 'sparse stationary distribution eigenvector:'
        print V[:, 0]
        print


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=2.0, type=float,
            help='concentration parameter of beta distribution')
    parser.add_argument('--N', default=5, type=int,
            help='population size'),
    parser.add_argument('--dense', action='store_true',
            help='use dense matrix linear algebra')
    parser.add_argument('--sparse', action='store_true',
            help='use sparse matrix linear algebra')
    main(parser.parse_args())

