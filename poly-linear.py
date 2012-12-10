"""
Attempt to show that only low order moments can be solved.

Assume that we know all moments of a marginal distribution
and that we have a dihedral symmetry D4
that gives us some equalities among various moments.
"""

import argparse
import itertools

import numpy as np
import scipy.linalg

def gen_equivalent_tuples(x):
    """
    @param x: a tuple of state indices
    """
    AB, Ab, aB, ab = (0, 1, 2, 3)
    equivalent_states = (
            (AB, Ab, aB, ab), # original
            (Ab, AB, ab, aB), # B <--> b
            (aB, ab, AB, Ab), # A <--> a
            (ab, aB, Ab, AB), # B <--> b and A <--> a
            (AB, aB, Ab, ab), # flip middle two
            (Ab, ab, AB, aB), # flip and B <--> b
            (aB, AB, ab, Ab), # flip and A <--> a
            (ab, Ab, aB, AB), # flip and B <--> b and A <--> a
            )
    for s in equivalent_states:
        yield tuple(s.index(i) for i in x)

def N_to_tuples(N):
    """
    @param N: homogeneous polynomial order
    @return: a sequence of 4**N tuples each of length N
    """
    return list(itertools.product((0, 1, 2, 3), repeat=N))

def get_canonical_tuples(N):
    canonical_tuples = set()
    for x in N_to_tuples(N):
        canonical_tuples.add(get_canonical_tuple(x))
    return sorted(canonical_tuples)

def get_canonical_tuple(x):
    return min(tuple(sorted(y)) for y in gen_equivalent_tuples(x))

def get_indicator_matrix(N):
    """
    Construct an indicator matrix reflecting the d4 mutational symmetry.
    There are 4**N rows.
    The number of columns depends on a symmetry and function of N;
    it is probably in oeis but I do not know the number.
    """
    canonical_tuples = get_canonical_tuples(N)
    T = dict((t, i) for i, t in enumerate(canonical_tuples))
    M = np.zeros((4**N, len(canonical_tuples)), dtype=int)
    for i, t in enumerate(N_to_tuples(N)):
        M[i, T[get_canonical_tuple(t)]] = 1
    return M

def get_contrast_moment_matrix(N, is_minimal=False):
    """
    The seven columns of the returned ndarray correspond to seven contrasts.
    Well one of the contrasts is not actually a contrast.
    And some of the columns of the returned ndarray may be redundant.
    """
    if is_minimal:
        contrasts = np.array([
            [+1, +1, +1, +1],
            [+1, +1, -1, -1],
            #[+1, -1, +1, -1],
            [+1, -1, -1, +1],
            #[-1, +1, +1, -1],
            #[-1, +1, -1, +1],
            #[-1, -1, +1, +1],
            ], dtype=int)
    else:
        contrasts = np.array([
            [+1, +1, +1, +1],
            [+1, +1, -1, -1],
            [+1, -1, +1, -1],
            [+1, -1, -1, +1],
            [-1, +1, +1, -1],
            [-1, +1, -1, +1],
            [-1, -1, +1, +1],
            ], dtype=int)
    M = np.zeros((4**N, len(contrasts)), dtype=int)
    for i, t in enumerate(N_to_tuples(N)):
        for j, c in enumerate(contrasts):
            M[i, j] = np.prod([c[k] for k in t])
    return M


def main(args):
    N = args.N

    # Construct two arrays each with 4**N rows.
    # One of the arrays will have entries in {+1, -1}
    # corresponding to coefficients in 7 moment-of-contrast constraint.
    # The other array will have entries in {0, 1}
    # corresponding to symmetries among moments.
    print get_canonical_tuples(N)
    X = get_indicator_matrix(N)
    M = get_contrast_moment_matrix(N, args.minimal_contrasts)
    print X
    print M
    R = np.dot(M.T, X)
    print 'R:'
    print R
    print
    print 'singular values of R:'
    print scipy.linalg.svd(R, full_matrices=False, compute_uv=False)
    print
    print 'pseudoinverse of R:'
    print scipy.linalg.pinv(R)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=3, type=int,
            help='moments of this order')
    parser.add_argument('--minimal-contrasts', action='store_true',
            help='use only three moment equations')
    main(parser.parse_args())

