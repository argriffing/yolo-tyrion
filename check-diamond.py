"""
Check variances of conditional distributions.

If the variances are all the same under various conditions,
then perhaps the distributions whose variances are taken
are independent of the conditions being varied.
This is related to the "diamond" collapse of the simplex of AB, Ab, aB, ab,
where Ab and aB are treated as the same state.
Use the Moran model which has been calibrated in check-fold.py.
"""

import numpy as np
import scipy.linalg

import wrightcore
import multinomstate
import MatrixUtil


##############################################################################
#FIXME: this code has been copypasted from check-fold.py
#FIXME: and it should be reorganized into a separate module.


def get_moran_drift(M, T):
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
    N = np.sum(M[0])
    R_drift = 0.5 * get_moran_drift(M, T) / float(N)
    R = R_mut + R_drift
    P = scipy.linalg.expm(R)
    v = MatrixUtil.get_stationary_distribution(P)
    return v

def get_collapsed_diamond_process_distn(m_factor, N, distn_helper):
    k = 3
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    R_mut = m_factor * wrightcore.create_mutation_collapsed(M, T)
    return distn_helper(M, T, R_mut)


##############################################################################
# The non-boilerplate part of this code has not been copypasted
# from anywhere else.


def main():

    # use standard notation
    Nmu = 1.0
    N = 120
    mu = Nmu / float(N)

    print 'N*mu:', Nmu
    print 'N:', N
    print

    # multiply the rate matrix by this scaling factor
    m_factor = mu

    # use the moran drift
    distn_helper = moran_distn_helper

    # get properties of the collapsed diamond process
    k = 3
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    R_mut = m_factor * wrightcore.create_mutation_collapsed(M, T)
    v = distn_helper(M, T, R_mut)

    for Ab_aB in range(N+1):
        nremaining = N - Ab_aB
        # compute the volume for normalization
        volume = 0.0
        for AB in range(nremaining+1):
            ab = nremaining - AB
            volume += v[T[AB, Ab_aB, ab]]
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
                p = v[T[AB, Ab_aB, ab]] / volume
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

