"""
Check some more properties of the equilibrium distribution.

For various ways of folding the states,
check that the equilibrium of the folded process
is equal to the folded equilibrium of the original process.
"""


import numpy as np
import numpy.testing
import scipy.linalg

import MatrixUtil
import multinomstate
import wrightcore

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
    return v_collapsed

def collapse_side(N, M, v):
    """
    Collapse two pairs of states.
    @param N: population size
    @param M: index to state vector
    @param v: a distribution over a 3-simplex
    @return: a distribution over a 1-simplex
    """
    k = 4
    nstates_collapsed = multinomstate.get_nstates(N, k-2)
    M_collapsed = np.array(list(multinomstate.gen_states(N, k-2)), dtype=int)
    T_collapsed = multinomstate.get_inverse_map(M_collapsed)
    v_collapsed = np.zeros(nstates_collapsed)
    for i, bigstate in enumerate(M):
        AB, Ab, aB, ab = bigstate.tolist()
        AB_Ab = AB + Ab
        aB_ab = aB + ab
        j = T_collapsed[AB_Ab, aB_ab]
        v_collapsed[j] += v[i]
    return v_collapsed

def collapse_diag(N, M, v):
    """
    Collapse two pairs of states.
    @param N: population size
    @param M: index to state vector
    @param v: a distribution over a 3-simplex
    @return: a distribution over a 1-simplex
    """
    k = 4
    nstates_collapsed = multinomstate.get_nstates(N, k-2)
    M_collapsed = np.array(list(multinomstate.gen_states(N, k-2)), dtype=int)
    T_collapsed = multinomstate.get_inverse_map(M_collapsed)
    v_collapsed = np.zeros(nstates_collapsed)
    for i, bigstate in enumerate(M):
        AB, Ab, aB, ab = bigstate.tolist()
        AB_ab = AB + ab
        Ab_aB = Ab + aB
        j = T_collapsed[AB_ab, Ab_aB]
        v_collapsed[j] += v[i]
    return v_collapsed

def wright_distn_helper(M, T, R_mut):
    """
    @param M: index to states
    @param T: states to index
    @param R_mut: scaled mutation rate matrix
    @return: stationary distribution of the process
    """
    #FIXME: remove dependence on T
    lmcs = wrightcore.get_lmcs(M)
    lps = wrightcore.create_selection_neutral(M)
    log_drift = wrightcore.create_neutral_drift(lmcs, lps, M)
    P_drift = np.exp(log_drift)
    P_mut = scipy.linalg.expm(R_mut)
    P = np.dot(P_mut, P_drift)
    v = MatrixUtil.get_stationary_distribution(P)
    return v

def get_moran_drift(M, T):
    #FIXME: this is dumb,
    #FIXME: but I'm not sure how to use variable-dimension ndarrays in cython
    k = M.shape[1]
    if k == 2:
        return wrightcore.create_moran_drift_rate_k2(M, T)
    elif k == 3:
        return wrightcore.create_moran_drift_rate_k3(M, T)
    elif k == 4:
        return wrightcore.create_moran_drift_rate_k4(M, T)
    else:
        raise NotImplementedError

def moran_distn_helper(M, T, R_mut):
    """
    @param M: index to states
    @param T: states to index
    @param R_mut: scaled mutation rate matrix
    @return: stationary distribution of the process
    """
    #FIXME: this is dumb,
    #FIXME: you should get the stationary distn directly from the rate matrix
    #FIXME: remove dependence on T
    N = np.sum(M[0])
    R_drift = 0.5 * get_moran_drift(M, T) / float(N)
    R = R_mut + R_drift
    P = scipy.linalg.expm(R)
    v = MatrixUtil.get_stationary_distribution(P)
    return v

def get_full_simplex(m_factor, N, distn_helper):
    """
    Note that this uses the non-moran formulation of drift.
    The distn_helper function taken as an argument is expected
    to be either moran_distn_helper or wright_distn_helper.
    @param m_factor: the mutation rate matrix is multiplied by this number
    @param N: population size
    @param distn_helper: a function (M, T, R_mut) -> v
    @return: M, T, v
    """
    k = 4
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    R_mut = m_factor * wrightcore.create_mutation(M, T)
    v = distn_helper(M, T, R_mut)
    return M, T, v

def get_collapsed_diamond_process_distn(m_factor, N, distn_helper):
    k = 3
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    R_mut = m_factor * wrightcore.create_mutation_collapsed(M, T)
    return distn_helper(M, T, R_mut)

def get_collapsed_side_process_distn(m_factor, N, distn_helper):
    k = 2
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    R_mut = m_factor * wrightcore.create_mutation_collapsed_side(M, T)
    return distn_helper(M, T, R_mut)

def get_collapsed_diag_process_distn(m_factor, N, distn_helper):
    k = 2
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    R_mut = m_factor * wrightcore.create_mutation_collapsed_diag(M, T)
    return distn_helper(M, T, R_mut)


class Test_EquilibriumDistributions(numpy.testing.TestCase):

    def test_distns(self):

        # use standard notation
        Nmu = 1.0
        N = 20
        mu = Nmu / float(N)

        # multiply the rate matrix by this scaling factor
        m_factor = mu

        for drift_name, distn_helper in (
                ('wright', wright_distn_helper),
                ('moran', moran_distn_helper),
                ):

            print 'testing', drift_name, 'drift...'

            # get properties of the full process
            M_full, T_full, v_full = get_full_simplex(m_factor, N, distn_helper)

            # compute collapsed distributions explicitly from the full distn
            v_full_diamond = collapse_diamond(N, M_full, v_full)
            v_full_diag = collapse_diag(N, M_full, v_full)
            v_full_side = collapse_side(N, M_full, v_full)

            # compute distributions of collapsed processes
            v_diamond = get_collapsed_diamond_process_distn(
                    m_factor, N, distn_helper)
            v_diag = get_collapsed_diag_process_distn(
                    m_factor, N, distn_helper)
            v_side = get_collapsed_side_process_distn(
                    m_factor, N, distn_helper)

            # compare the distributions
            for pair in (
                    (v_full_diamond, v_diamond),
                    (v_full_diag, v_diag),
                    (v_full_side, v_side),
                    ):
                numpy.testing.assert_allclose(*pair)

            print 'finished testing', drift_name, 'drift.'


    def test_collapsed_variance(self):

        # use standard notation
        Nmu = 1.0
        N = 100
        mu = Nmu / float(N)

        # multiply the rate matrix by this scaling factor
        m_factor = mu

        print 'N*mu:', Nmu
        print 'N:', N
        print

        for drift_name, distn_helper in (
                ('wright', wright_distn_helper),
                ('moran', moran_distn_helper),
                ):

            print 'checking properties of', drift_name, 'drift...'

            # get the distribution of this collapsed process
            v_diag = get_collapsed_diag_process_distn(
                    m_factor, N, distn_helper)

            # compute some moments of the distribution
            m1 = 0.0
            m2 = 0.0
            for i, p in enumerate(v_diag):
                x = i / float(N)
                m1 += x*p
                m2 += x*x*p

            # show some expectations
            print 'E(X) =', m1
            print 'E(X^2) =', m2
            print 'V(X) =', m2 - m1*m1
            print

            """
            # compute some moments of the distribution using midpoint rule
            m1_mid = 0.0
            m2_mid = 0.0
            for i in range(N-1):
                x = (i + 0.5) / float(N)
                p = (v_diag[i] + v_diag[i+1]) / 2
                m1_mid += x*p
                m2_mid += x*x*p

            # show some expectations with midpoint
            print 'using midpoint rule:'
            print 'E(X) =', m1_mid
            print 'E(X^2) =', m2_mid
            print 'V(X) =', m2_mid - m1_mid*m1_mid
            print
            """

        u = 4*N*mu
        v_expected = 1 / (4 * (2*u + 1))
        print 'u = 4*N*mu'
        print 'expected V(X) =', v_expected
        print
        u = 2*N*mu
        v_expected = 1 / (4 * (2*u + 1))
        print 'u = 2*N*mu'
        print 'expected V(X) =', v_expected
        print

        raise Exception


if __name__ == '__main__':
    numpy.testing.run_module_suite()

