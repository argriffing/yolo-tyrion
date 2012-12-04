"""
Quickly create large mutation and recombination rate matrices.

This is for studying compensatory nucleotide changes in evolution.
The four haplotypes are (AB, Ab, aB, ab).
The AB and ab haplotypes have high fitness,
while the Ab and aB haplotypes have low fitness.
.
The implementation is in Cython for speed
and uses python numpy arrays for speed and convenience.
For compilation instructions see
http://docs.cython.org/src/reference/compilation.html
For example:
$ cython -a wrightcore.pyx
$ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
      -I/usr/include/python2.7 -o wrightcore.so wrightcore.c
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log

np.import_array()


#############################################################################
# These are two functions to create mutation rate matrices.
# They are specific to the models being studied.


@cython.boundscheck(False)
@cython.wraparound(False)
def create_mutation(
        np.ndarray[np.int_t, ndim=2] M,
        np.ndarray[np.int_t, ndim=4] T,
        ):
    """
    The scaling of the resulting rate matrix is strange.
    Every rate is an integer, but in double precision float format.
    @param M: M[i, j] is the count of allele j in state index i
    @param T: T[AB, Ab, aB, ab] is the state index for the given allele counts
    @return: mutation rate matrix
    """
    cdef int i
    cdef int AB, Ab, aB, ab
    cdef int nstates = M.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] R = np.zeros((nstates, nstates))
    for i in range(nstates):
        # meticulously unpack the allele counts from the corresponding state
        AB = M[i, 0]
        Ab = M[i, 1]
        aB = M[i, 2]
        ab = M[i, 3]
        #
        if AB > 0:
            R[i, T[AB-1, Ab+1, aB,   ab  ]] = AB
            R[i, T[AB-1, Ab,   aB+1, ab  ]] = AB
        if Ab > 0:
            R[i, T[AB+1, Ab-1, aB,   ab  ]] = Ab
            R[i, T[AB,   Ab-1, aB,   ab+1]] = Ab
        if aB > 0:
            R[i, T[AB+1, Ab,   aB-1, ab  ]] = aB
            R[i, T[AB,   Ab,   aB-1, ab+1]] = aB
        if ab > 0:
            R[i, T[AB,   Ab+1, aB,   ab-1]] = ab
            R[i, T[AB,   Ab,   aB+1, ab-1]] = ab
        R[i, i] = -2*(AB + Ab + aB + ab)
    return R

@cython.boundscheck(False)
@cython.wraparound(False)
def create_mutation_collapsed(
        np.ndarray[np.int_t, ndim=2] M,
        np.ndarray[np.int_t, ndim=3] T,
        ):
    """
    Two states have been collapsed into a single state.
    The scaling of the resulting rate matrix is strange.
    Every rate is an integer, but in double precision float format.
    @param M: M[i, j] is the count of allele j in state index i
    @param T: T[AB, Ab_aB, ab] is the state index for the given allele counts
    @return: mutation rate matrix
    """
    cdef int i
    cdef int AB, Ab_aB, ab
    cdef int nstates = M.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] R = np.zeros((nstates, nstates))
    for i in range(nstates):
        # meticulously unpack the allele counts from the corresponding state
        AB    = M[i, 0]
        Ab_aB = M[i, 1]
        ab    = M[i, 2]
        #
        if AB > 0:
            R[i, T[AB-1, Ab_aB+1, ab  ]] = 2*AB
        if Ab_aB > 0:
            R[i, T[AB+1, Ab_aB-1, ab  ]] = Ab_aB
            R[i, T[AB,   Ab_aB-1, ab+1]] = Ab_aB
        if ab > 0:
            R[i, T[AB,   Ab_aB+1, ab-1]] = 2*ab
        R[i, i] = -2*(AB + Ab_aB + ab)
    return R

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def create_moran_drift_rate(
        np.ndarray[np.int_t, ndim=2] M,
        np.ndarray[np.int_t, ndim=3] T,
        ):
    """
    The scaling of the resulting rate matrix is strange.
    @param M: M[i, j] is the count of allele j in state index i
    @param T: T[X, Y, Z] is the state index for the given allele counts
    @return: mutation rate matrix
    """
    #FIXME: how to generalize to a rate matrix with more than three alleles
    #FIXME: without changing the function signature?
    #FIXME: is cython flexible enough to do this?
    cdef int i
    cdef int X, Y, Z
    cdef int nstates = M.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] R = np.zeros((nstates, nstates))
    for i in range(nstates):
        X = M[i, 0]
        Y = M[i, 1]
        Z = M[i, 2]
        if X > 0:
            R[i, T[X-1, Y+1, Z  ]] = X*Y
            R[i, T[X-1, Y,   Z+1]] = X*Z
        if Y > 0:
            R[i, T[X+1, Y-1, Z  ]] = Y*X
            R[i, T[X,   Y-1, Z+1]] = Y*Z
        if Z > 0:
            R[i, T[X+1, Y,   Z-1]] = Z*X
            R[i, T[X,   Y+1, Z-1]] = Z*Y
        R[i, i] = -(X*Y + X*Z + Y*X + Y*Z + Z*X + Z*Y)
    return R


#############################################################################
# These are helper functions for the Wright-Fisher drift component.


@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_log_fact_array(int N):
    """
    Precompute some logarithms of factorials up to N.
    @param N: max integer whose log of factorial to compute
    @return: a numpy array of length N+1
    """
    cdef np.ndarray[np.float64_t, ndim=1] log_fact = np.zeros(N+1)
    cdef double accum = 0
    for i in range(2, N+1):
        accum += log(i)
        log_fact[i] = accum
    return log_fact

@cython.boundscheck(False)
@cython.wraparound(False)
def get_lmcs(
        np.ndarray[np.int_t, ndim=2] M,
        ):
    """
    This computes the lmcs ndarray with logs of multinomial coefficients.
    This is separate from selection or recombination or mutation.
    It is used for the drift component of wright fisher processes.
    @param M: M[i,j] is count of bin j in state i
    @return: a one dimensional array of log multinomial coefficients
    """
    cdef int nstates = M.shape[0]
    cdef int k = M.shape[1]
    cdef int N = np.sum(M[0])
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=1] log_fact = get_log_fact_array(N)
    cdef np.ndarray[np.float64_t, ndim=1] v = np.empty(nstates)
    for i in range(nstates):
        v[i] = log_fact[N]
        for index in range(k):
            v[i] -= log_fact[M[i, index]]
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
def create_selection_neutral(
        np.ndarray[np.int_t, ndim=2] M,
        ):
    """
    This makes the lps array used by create_neutral_drift.
    """
    cdef int nstates = M.shape[0]
    cdef int k = M.shape[1]
    cdef int i, j
    cdef double neg_logp
    cdef double popsize
    cdef np.ndarray[np.float64_t, ndim=2] L = np.empty((nstates, k))
    for i in range(nstates):
        popsize = 0
        for j in range(k):
            popsize += M[i, j]
        neg_logp = -log(popsize)
        for j in range(k):
            L[i, j] = neg_logp + log(M[i, j])
    return L


#############################################################################
# Use the output of the helper functions to make the drift transition.
# Note that this function is general enough to deal with genic selection,
# but for the first application we will feed it a neutral lps.


@cython.boundscheck(False)
@cython.wraparound(False)
def create_neutral_drift(
        np.ndarray[np.float64_t, ndim=1] lmcs,
        np.ndarray[np.float64_t, ndim=2] lps,
        np.ndarray[np.int_t, ndim=2] M,
        ):
    """
    This is a more flexible way to create a multinomial transition matrix.
    It is also fast.
    The output may have -inf but it should not have nan.
    @param lmcs: log multinomial count per state
    @param lps: log probability per haplotype per state
    @param M: allele count per haplotype per state
    @return: entrywise log of transition matrix
    """
    cdef int nstates = M.shape[0]
    cdef int k = M.shape[1]
    cdef int i, j, index
    cdef np.ndarray[np.float64_t, ndim=2] L = np.zeros((nstates, nstates))
    for i in range(nstates):
        for j in range(nstates):
            L[i, j] = lmcs[j]
            for index in range(k):
                if M[j, index]:
                    L[i, j] += M[j, index] * lps[i, index]
    return L

