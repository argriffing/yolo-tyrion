"""
Check some properties of the equilibrium distribution.
"""

import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from mayavi.mlab import *

import MatrixUtil
import multinomstate
import wrightcore


def drawtri(M, T, v):
    nstates = len(M)
    angles = np.array([0, 1, 2], dtype=float) * (2.0*np.pi / 3.0)
    points = np.array([[np.cos(a), np.sin(a)] for a in angles])
    x = np.empty(nstates)
    y = np.empty(nstates)
    z = np.empty(nstates)
    for i, state in enumerate(M):
        popsize = np.sum(state)
        a, b, c = M[i].tolist()
        xy = (a*points[0] + b*points[1] + c*points[2]) / float(popsize)
        x[i] = xy[0]
        y[i] = xy[1]
        z[i] = v[i]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

    plt.show()


def do_full_simplex_then_collapse(mutrate):
    #mutrate = 0.01
    #mutrate = 0.2
    #mutrate = 10
    #mutrate = 100
    #mutrate = 1
    N = 20
    k = 4
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    # Create the joint site pair mutation rate matrix.
    R = mutrate * wrightcore.create_mutation(M, T)
    # Create the joint site pair drift transition matrix.
    lmcs = wrightcore.get_lmcs(M)
    lps = wrightcore.create_selection_neutral(M)
    log_drift = wrightcore.create_neutral_drift(lmcs, lps, M)
    # Define the drift and mutation transition matrices.
    P_drift = np.exp(log_drift)
    P_mut = scipy.linalg.expm(R)
    # Define the composite per-generation transition matrix.
    P = np.dot(P_mut, P_drift)
    # Solve a system of equations to find the stationary distribution.
    v = MatrixUtil.get_stationary_distribution(P)
    for state, value in zip(M, v):
        print state, value
    # collapse the two middle states
    nstates_collapsed = multinomstate.get_nstates(N, k-1)
    M_collapsed = np.array(list(multinomstate.gen_states(N, k-1)), dtype=int)
    T_collapsed = multinomstate.get_inverse_map(M_collapsed)
    v_collapsed = np.zeros(nstates_collapsed)
    for i, bigstate in enumerate(M):
        AB, Ab, aB, ab = bigstate.tolist()
        Ab_aB = Ab + aB
        j = T_collapsed[AB, Ab_aB, ab]
        v_collapsed[j] += v[i]
    for state, value in zip(M_collapsed, v_collapsed):
        print state, value
    # draw an equilateral triangle
    #drawtri(M_collapsed, T_collapsed, v_collapsed)
    #test_mesh()
    return M_collapsed, T_collapsed, v_collapsed

def do_collapsed_simplex(mutrate):
    #mutrate = 100
    N = 20
    k = 3
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    # Create the joint site pair mutation rate matrix.
    R = mutrate * wrightcore.create_mutation_collapsed(M, T)
    # Create the joint site pair drift transition matrix.
    lmcs = wrightcore.get_lmcs(M)
    lps = wrightcore.create_selection_neutral(M)
    log_drift = wrightcore.create_neutral_drift(lmcs, lps, M)
    # Define the drift and mutation transition matrices.
    P_drift = np.exp(log_drift)
    P_mut = scipy.linalg.expm(R)
    # Define the composite per-generation transition matrix.
    P = np.dot(P_mut, P_drift)
    # Solve a system of equations to find the stationary distribution.
    v = MatrixUtil.get_stationary_distribution(P)
    for state, value in zip(M, v):
        print state, value
    # draw an equilateral triangle
    #drawtri(M, T, v)
    return M, T, v


if __name__ == '__main__':
    mutrate = 0.01
    Ma, Ta, va = do_full_simplex_then_collapse(mutrate)
    Mb, Tb, vb = do_collapsed_simplex(mutrate)
    print Ma - Mb
    print Ta - Tb
    print va - vb

