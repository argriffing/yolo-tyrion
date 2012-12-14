"""
This is based on a pyqtgraph volumetric example for graphics.

It uses a tetrahedral mesh,
where the simplex basis
((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
is mapped to the hamming basis
((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1))
.
"""


import argparse
import math

import numpy as np
import scipy.linalg
import scipy.stats

import wrightcore
import multinomstate
import MatrixUtil

import sys

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl


def psi(i, j, k, offset=(50,50,100)):
    """
    Hydrogen electron probability density.
    This is from the pyqtgraph volumetric example.
    """
    x = i-offset[0]
    y = j-offset[1]
    z = k-offset[2]
    th = np.arctan2(z, (x**2+y**2)**0.5)
    phi = np.arctan2(y, x)
    r = (x**2 + y**2 + z **2)**0.5
    a0 = 2
    ps = (
            (1. / 81.) *
            (1. / np.sqrt(6. * np.pi)) *
            ((1./a0)**(3./2.)) *
            ((r/a0)**2) *
            (np.exp(-r/(3*a0))) *
            (3 * np.cos(th)**2 - 1) )
    return ps


def main(args):

    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 200
    w.show()


    #b = gl.GLBoxItem()
    #w.addItem(b)
    g = gl.GLGridItem()
    g.scale(10, 10, 1)
    w.addItem(g)

    data = np.fromfunction(psi, (100,100,200))
    positive = np.log(np.clip(data, 0, data.max())**2)
    negative = np.log(np.clip(-data, 0, -data.min())**2)

    d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
    d2[..., 0] = positive * (255./positive.max())
    d2[..., 1] = negative * (255./negative.max())
    d2[..., 2] = d2[...,1]
    d2[..., 3] = d2[..., 0]*0.3 + d2[..., 1]*0.3
    d2[..., 3] = (d2[..., 3].astype(float) / 255.) **2 * 255

    d2[:, 0, 0] = [255,0,0,100]
    d2[0, :, 0] = [0,255,0,100]
    d2[0, 0, :] = [0,0,255,100]


    v = gl.GLVolumeItem(d2)
    v.translate(-50,-50,-100)
    w.addItem(v)

    ax = gl.GLAxisItem()
    w.addItem(ax)

    ## Start Qt event loop unless running in interactive mode.
    if sys.flags.interactive != 1:
        app.exec_()


def old_main(args):
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
    print MatrixUtil.m_to_mathematica_string(Q.T.astype(int))
    print
    print 'abs eigenvector corresponding to smallest abs eigenvalue:'
    print np.abs(v)
    print

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
    P = scipy.linalg.expm(Q)
    v = MatrixUtil.get_stationary_distribution(P)
    #
    # Define the volumetric data using the stationary distribution.
    max_prob = np.max(v)
    d2 = np.zeros((N+1, N+1, N+1, 4), dtype=float)
    U = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        ], dtype=int)
    for p, state in zip(v, M):
        x, y, z = np.dot(state, U).tolist()
        # r, g, b, alpha
        d2[x, y, z] = np.array([
            255 * (p / max_prob),
            0,
            0,
            100,
            ], dtype=float)
        #d2[x, y, z, 0] = 255 * (p / max_prob)
        #d2[x, y, z, 1] = 0
        #d2[x, y, z, 2] = 0
        #d2[x, y, z, 3] = 100
    # fill the empty states
    for x in range(N+1):
        for y in range(N+1):
            for z in range(N+1):
                if (x + y + z) % 2 == 1:
                    p_accum = np.zeros(4, dtype=float)
                    n_accum = 0
                    for dx in (-1, 1):
                        if 0 <= x+dx <= N:
                            p_accum += d2[x+dx, y, z]
                            n_accum += 1
                    for dy in (-1, 1):
                        if 0 <= y+dy <= N:
                            p_accum += d2[x, y+dy, z]
                            n_accum += 1
                    for dz in (-1, 1):
                        if 0 <= z+dz <= N:
                            p_accum += d2[x, y, z+dz]
                            n_accum += 1
                    d2[x, y, z] = p_accum / n_accum
    #
    # Do things that the example application does.
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 2*N
    w.show()
    #
    # a visual grid or something
    #g = gl.GLGridItem()
    #g.scale(10, 10, 1)
    #w.addItem(g)
    #
    # Do some more things that the example application does.
    vol = gl.GLVolumeItem(d2, sliceDensity=1, smooth=True)
    #vol.translate(-5,-5,-10)
    vol.translate(-0.5*N, -0.5*N, -0.5*N)
    w.addItem(vol)
    #
    # add an axis thingy
    #ax = gl.GLAxisItem()
    #w.addItem(ax)
    if sys.flags.interactive != 1:
        app.exec_()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=2.0, type=float,
            help='concentration parameter of beta distribution')
    parser.add_argument('--N', default=5, type=int,
            help='population size'),
    main(parser.parse_args())

