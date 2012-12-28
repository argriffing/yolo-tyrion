"""
Compute the equilibrium distribution of alleles in a Moran-like process.

The death/birth replacement events and mutation events can happen at any time,
and these events occur according to poisson-like processes.
The ratio of the rates of these two processes is defined
by a user-specified parameter.
Furthermore, we assume a particular genetic process with four states
and with a restricted mutational structure.
This mutational structure can be interpreted as disallowing transversions
if the four states are interpreted as A,C,G,T nucleotides,
or alternatively the mutational structure can be interpreted as
disallowing simultaneous mutations at paired sites
if the four states are interpreted as pairs of binary alleles.
Note that the mutational structure is not "parent-independent."
This script is meant to be self-contained in the sense that it does
not require any imports except scipy, numpy, and standard library modules.
"""

import argparse
import logging

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


##############################################################################
# These functions define an ordering on vertices in a lattice in a tetrahedron.

def gen_states(N, k):
    """
    Yield multinomial states.
    Each state is a list of length k and with sum N.
    The state ordering is one which simple to generate recursively.
    @param N: population size
    @param k: number of bins
    """
    if k == 1:
        yield [N]
    elif k == 2:
        for i in range(N+1):
            yield [i, N-i]
    else:
        for i in range(N+1):
            for suffix in gen_states(N-i, k-1):
                yield [i] + suffix

def get_inverse_dict(M):
    """
    The input M[i,j] is count of allele j for pop state index i.
    The output T[(i,j,...)] maps allele count tuple to pop state index
    @param M: multinomial state map
    @return: T
    """
    return dict((tuple(state), i) for i, state in enumerate(M))


##############################################################################
# These functions construct a sparse rate matrix.

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
    which presumably is efficient to transpose and to convert to a csc_matrix.
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


##############################################################################
# These functions give informative messages
# in response to errors in the command line usage of this script.

def pos_float(s):
    x = float(s)
    if x <= 0:
        raise argparse.ArgumentTypeError('Value must be positive')
    return x

def pos_int(s):
    x = int(s)
    if x <= 0:
        raise argparse.ArgumentTypeError('Value must be positive')
    return x

def nonneg_float(s):
    x = float(s)
    if x < 0:
        raise argparse.ArgumentTypeError('Value must be nonnegative')
    return x


##############################################################################
# Run the script from the command line with reasonable defaults
# and an attempt to give informative help and error messages.

def main(args):

    # set up logging
    if args.quiet:
        logging_level = logging.ERROR
    elif args.verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging.basicConfig(
            level=logging_level,
            #format='%(asctime)s - %(levelname)s - %(message)s',
            format='%(levelname)s - %(message)s',
            )

    # extract some more settings from the command line
    alpha = args.alpha
    N = args.popsize

    # do everything else
    k = 4
    logging.info('alpha: %s' % alpha)
    logging.info('popsize: %s' % N)
    logging.info('k: %s' % k)
    logging.info('defining the state vectors...')
    M = np.array(list(gen_states(N, k)), dtype=int)
    nstates = M.shape[0]
    logging.debug('M.shape: %s' % str(M.shape))
    logging.info('number of population genetic states: %s' % nstates)
    logging.info('defining the state vector inverse dict...')
    T = get_inverse_dict(M)
    logging.info('constructing sparse coo mutation+drift rate matrix...')
    R_coo = create_coo_moran(M, T, alpha)
    logging.info('converting to sparse csc transpose rate matrix...')
    RT_csc = scipy.sparse.csc_matrix(R_coo.T)
    logging.info('computing an eigenpair using "small magnitude" mode...')
    W, V = scipy.sparse.linalg.eigs(RT_csc, k=1, which='SM')
    logging.debug('sparse eigenvalues:')
    logging.debug(str(W))
    logging.debug('sparse stationary distribution eigenvector:')
    logging.debug(str(V[:, 0]))
    v = abs(V[:, 0])
    v /= np.sum(v)
    logging.info('opening %s for writing...' % args.output)
    with open(args.output, 'w') as fout:
        logging.info('writing the header...')
        print >> fout, '\t'.join(str(x) for x in ('AB', 'Ab', 'aB', 'ab', 'p'))
        logging.info('writing the equilibrium distribution...')
        for i in range(nstates):
            p = v[i]
            X, Y, Z, W = M[i]
            print >> fout, '\t'.join(str(x) for x in (i+1, X, Y, Z, W, p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument('-v', '--verbose', action='store_true',
        help='print lots of annoying messages')
    verbosity_group.add_argument('-q', '--quiet', action='store_true',
        help='print fewer annoying messages')
    parser.add_argument('-a', '--alpha', default=1.0, type=pos_float,
        help='concentration parameter of beta distribution of P({AB, Ab})')
    parser.add_argument('-N', '--popsize', default=2, type=pos_int,
        help='haploid population size'),
    parser.add_argument('-o', '--output',
        default='consolidated-moran-autosave.table',
        help='write this output file')
    main(parser.parse_args())

