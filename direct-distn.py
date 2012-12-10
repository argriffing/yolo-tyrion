"""
Try to estimate a distribution over a simplex using numerical methods.

This approximation tries to adjust weights of each piece
of a discretized simplex so that certain integrals over the simplex have 
the correct values.
Symmetry constraints are also imposed.
A sum of squares of errors with respect to the constraints is
the objective function which will be minimized.
"""

import functools

import numpy as np
import scipy.optimize
import scipy.stats
import algopy

import multinomstate

def get_d4_reduction(M, T):
    """
    @param M: index to state vector
    @param T: state vector to index
    @return: map from full to reduced state index, number of reduced states
    """
    nstates = len(M)
    nstates_reduced = 0
    full_index_to_reduced_index = -1 * np.ones(nstates, dtype=int)
    for i, full_state in enumerate(M):
        AB, Ab, aB, ab = full_state.tolist()
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
        canonical_state = min(equivalent_states)
        canonical_full_index = T[canonical_state]
        if full_index_to_reduced_index[canonical_full_index] == -1:
            full_index_to_reduced_index[canonical_full_index] = nstates_reduced
            nstates_reduced += 1
        reduced_index = full_index_to_reduced_index[canonical_full_index]
        full_index_to_reduced_index[i] = reduced_index
    return full_index_to_reduced_index, nstates_reduced

#FIXME: unused
def get_simplex_d4_symmetry_constraints(M, T):
    """
    The input defines the discretization of a simplex.
    The output defines the pairwise equalities implied by d4 symmetry.
    This function is slow and should be used only for testing.
    @param M: index to state vector
    @param T: state vector to index
    @return: a huge sparse binary array of pairwise equality constraints
    """
    k = 4
    if not M.ndim == 2:
        raise Exception
    if not T.ndim == k:
        raise Exception
    if not M.shape[1] == k:
        raise Exception
    nstates = M.shape[0]
    C = np.zeros((nstates, nstates), dtype=int)
    for i, state_i in enumerate(M):
        state = state_i.tolist()
        for j, state_j in enumerate(M):
            AB, Ab, aB, ab = state_j.tolist()
            d4_elements = [
                    [AB, Ab, aB, ab], # original
                    [Ab, AB, ab, aB], # B <--> b
                    [aB, ab, AB, Ab], # A <--> a
                    [ab, aB, Ab, AB], # B <--> b and A <--> a
                    [AB, aB, Ab, ab], # flip middle two
                    [Ab, ab, AB, aB], # flip and B <--> b
                    [aB, AB, ab, Ab], # flip and A <--> a
                    [ab, Ab, aB, AB], # flip and B <--> b and A <--> a
                    ]
            if any(state == element for element in d4_elements):
                C[i, j] = 1
    return C

def get_beta_approx(npoints, alpha):
    """
    These are normalized likelihoods from a beta function pdf.
    Speed does not matter.
    @param N: evaluate this many equally spaced points on [0, 1]
    @param alpha: beta distribution concentration parameter
    @return: a discrete distribution
    """
    x = np.linspace(0, 1, npoints)
    y = scipy.stats.beta.pdf(x, alpha, alpha)
    return y / np.sum(y)

def get_design_matrix_side(M):
    """
    Precompute a matrix to help compute a marginal distribution.
    Speed does not matter.
    """
    nstates = len(M)
    N = np.sum(M[0])
    X = np.zeros((nstates, N+1), dtype=int)
    for i, state in enumerate(M):
        AB, Ab, aB, ab = M[i].tolist()
        X[i, AB + Ab] = 1
    return X

def get_design_matrix_diag(M):
    """
    Precompute a matrix to help compute a marginal distribution.
    Speed does not matter.
    """
    nstates = len(M)
    N = np.sum(M[0])
    X = np.zeros((nstates, N+1), dtype=int)
    for i, state in enumerate(M):
        AB, Ab, aB, ab = M[i].tolist()
        X[i, AB + ab] = 1
    return X

def get_errors(M, v, approx_1a, approx_2a, X_side, X_diag):
    nstates = len(M)
    N = np.sum(M[0])
    observed_1a_1 = algopy.zeros(N+1, dtype=v)
    #observed_1a_2 = algopy.zeros(N+1, dtype=v)
    observed_2a = algopy.zeros(N+1, dtype=v)
    """
    for i in range(nstates):
        p = v[i]
        AB, Ab, aB, ab = M[i].tolist()
        observed_1a_1[AB + Ab] += p
        #observed_1a_2[AB + aB] += p
        #observed_2a[AB + ab] += p
        observed_2a[Ab + aB] += p
    """
    observed_1a_1 = algopy.dot(v, X_side)
    observed_2a = algopy.dot(v, X_diag)
    #
    errors_1a_1 = observed_1a_1 - approx_1a
    #errors_1a_2 = observed_1a_2 - approx_1a
    errors_2a = observed_2a - approx_2a
    #FIXME: Use algopy.hstack when it becomes available.
    #FIXME: using the workaround http://projects.scipy.org/scipy/ticket/1454
    #FIXME: but this padding of the errors with zeros should not be necessary
    #nonunif_penalty = 0.01
    #nonunif = v - np.ones(nstates) / float(nstates)
    #nonunif = np.zeros(nstates)
    errors = algopy.zeros(
            #len(nonunif) +
            errors_1a_1.shape[0] +
            #len(errors_1a_2) +
            errors_2a.shape[0],
            dtype=v,
            )
    index = 0
    errors[index:index+errors_1a_1.shape[0]] = errors_1a_1
    index += errors_1a_1.shape[0]
    #errors[index:index+len(errors_1a_2)] = errors_1a_2
    #index += len(errors_1a_2)
    errors[index:index+errors_2a.shape[0]] = errors_2a
    index += errors_2a.shape[0]
    #errors[index:index+len(nonunif)] = nonunif_penalty * nonunif
    #index += len(nonunif)
    return errors



def eval_f(
        M, T, d4_reduction, d4_nstates, approx_1a, approx_2a,
        X_side, X_diag,
        X,
        ):
    """
    This algopy-enabled function should be callable from leastsq.
    The first two groups of args should be partially evaluated.
    The M and T args are related to the discretization of a simplex volume.
    The d4 args are related to symmetries of a function on the simplex.
    The X_side and X_diag args are precomputed design matrices.
    The last arg is a vector of logs of parameter values.
    @param M: full index to state vector
    @param T: state vector to full index
    @param d4_reduction: map from full index to symmetry-reduced index
    @param d4_nstates: number of symmetry-reduced states
    @param approx_1a: one of the expected marginal distributions
    @param approx_2a: another of the expected marginal distributions
    @param X: parameter values
    @return: vector of errors whose squares are to be minimized
    """
    if X.shape[0] != d4_nstates - 1:
        raise Exception

    # get the number of states and the population size
    nstates = len(M)
    N = np.sum(M[0])

    # unpack the parameter values into a d4 symmetric joint distribution
    v = unpack_distribution(nstates, d4_reduction, d4_nstates, X)
    errors = get_errors(M, v, approx_1a, approx_2a, X_side, X_diag)
    return errors


def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))

def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))

def eval_grad_reverse_mode(f, x):
    cg = algopy.CGraph()
    fx = algopy.Function(x)
    fy = f(x)
    cg.trace_off()
    cg.indepndentFunctionList = [fx]
    cg.dependentFunctionList = [fy]
    #cg.plot('omg.png')
    #result = cg.gradient([x])
    #print 'reverse mode gradient:'
    result = cg.jacobian(fx)
    print 'reverse mode jacobian:'
    print result
    return result

def unpack_distribution(nstates, d4_reduction, d4_nstates, X):
    log_v = algopy.zeros(nstates, dtype=X)
    for i_full, i_reduced in enumerate(d4_reduction):
        if i_reduced == d4_nstates - 1:
            log_v[i_full] = 0.0
        else:
            log_v[i_full] = X[i_reduced]
    v = algopy.exp(log_v)
    v = v / algopy.sum(v)
    return v

def apply_sum_of_squares(f, X):
    errors = f(X)
    sse = algopy.dot(errors, errors)
    print sse
    return sse

def main():

    # use standard notation
    Nmu = 1.0
    N = 20
    mu = Nmu / float(N)

    print 'N*mu:', Nmu
    print 'N:', N
    print

    k = 4
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    nstates = len(M)
    #R_mut = m_factor * wrightcore.create_mutation_collapsed(M, T)
    #v = distn_helper(M, T, R_mut)

    # get the approximations
    alpha = 2*N*mu
    approx_1a = get_beta_approx(N+1, alpha)
    approx_2a = get_beta_approx(N+1, 2*alpha)

    d4_reduction, d4_nstates = get_d4_reduction(M, T)
    # for the initial guess all logs of ratios of probs are zero
    x0 = np.zeros(d4_nstates - 1)

    # precompute some design matrices
    X_side = get_design_matrix_side(M)
    X_diag = get_design_matrix_diag(M)

    print 'number of variables:', d4_nstates - 1
    print

    f_errors = functools.partial(
            eval_f,
            M, T, d4_reduction, d4_nstates, approx_1a, approx_2a,
            X_side, X_diag,
            )

    g_errors = functools.partial(eval_grad, f_errors)

    f = functools.partial(apply_sum_of_squares, f_errors)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

    g_reverse = functools.partial(eval_grad_reverse_mode, f)

    """
    result = scipy.optimize.leastsq(
            f_errors,
            x0,
            Dfun=g_errors,
            full_output=1,
            )
    """

    """
    result = scipy.optimize.fmin_ncg(
            f,
            x0,
            fprime=g,
            fhess=h,
            avextol=1e-6,
            full_output=True,
            )
    """

    result = scipy.optimize.fmin_bfgs(
            f,
            x0,
            #fprime=g,
            fprime=g_reverse,
            full_output=True,
            )

    print result

    xopt = result[0]

    v = unpack_distribution(nstates, d4_reduction, d4_nstates, xopt)

    # print some variances
    check_variance(M, T, v)


#FIXME: mostly copypasted from check-folds.py
def collapse_diamond(N, M, v):
    """
    Collapse the middle two states.
    @param N: population size
    @param M: index to state vector
    @param v: a distribution over a 3-simplex
    @return: a distribution over a 2-simplex
    """
    k = 4
    nstates_collapsed = multinomstate.get_nstates(N, k-1)
    M_collapsed = np.array(list(multinomstate.gen_states(N, k-1)), dtype=int)
    T_collapsed = multinomstate.get_inverse_map(M_collapsed)
    v_collapsed = np.zeros(nstates_collapsed)
    for i, bigstate in enumerate(M):
        AB, Ab, aB, ab = bigstate.tolist()
        Ab_aB = Ab + aB
        j = T_collapsed[AB, Ab_aB, ab]
        v_collapsed[j] += v[i]
    return M_collapsed, T_collapsed, v_collapsed

#FIXME: mostly copypasted from check-diamond.py
def check_variance(M, T, v):
    N = np.sum(M[0])
    M_collapsed, T_collapsed, v_collapsed = collapse_diamond(N, M, v)
    for Ab_aB in range(N+1):
        nremaining = N - Ab_aB
        # compute the volume for normalization
        volume = 0.0
        for AB in range(nremaining+1):
            ab = nremaining - AB
            volume += v_collapsed[T_collapsed[AB, Ab_aB, ab]]
        # print some info
        print 'X_1 + X_4 =', Ab_aB, '/', N
        print 'probability =', volume
        print 'Y = X_2 / (1 - (X_1 + X_4)) = X_2 / (X_2 + X_3)'
        if not nremaining:
            print 'conditional distribution of Y is undefined'
        else:
            # compute the conditional moments
            m1 = 0.0
            m2 = 0.0
            for AB in range(nremaining+1):
                ab = nremaining - AB
                p = v_collapsed[T_collapsed[AB, Ab_aB, ab]] / volume
                x = AB / float(nremaining)
                m1 += x*p
                m2 += x*x*p
            # print some info
            print 'conditional E(Y) =', m1
            print 'conditional E(Y^2) =', m2
            print 'conditional V(Y) =', m2 - m1*m1
        print

if __name__ == '__main__':
    main()

