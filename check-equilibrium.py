"""
Check some properties of the equilibrium distribution.
"""


import numpy as np
import scipy.linalg
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#import matplotlib.pyplot as plt
#from mayavi.mlab import *

import matplotlib.delaunay
import matplotlib.tri

import MatrixUtil
import multinomstate
import wrightcore

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import scipy.ndimage as ndi

def get_xyz(M, T, v):
    """
    This is a helper function for visualization.
    """
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
    return x, y, z

def show_tri_qtgraph_mesh(M, T, v):

    x, y, z = get_xyz(M, T, v)
    #z *= 20
    z *= 100

    # Try to construct a mesh from the vertices.
    #results = matplotlib.delaunay.delaunay(x, y)
    #circumcenters, edges, tri_points, tri_neighbors = results
    #faces = tri_points

    t = matplotlib.tri.triangulation.Triangulation(x, y)
    faces = np.array([list(reversed(face)) for face in t.triangles])

    verts = np.vstack((x, y, z)).T
    r, g, b, alpha = 1.0, 0.0, 0.0, 0.5
    colors = np.array([[r, g, b, alpha] for face in faces])
    print len(v)
    print verts.shape
    print faces.shape
    print colors.shape

    ## Create a GL View widget to display data
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    #w.setCameraPosition(distance=50)
    w.setCameraPosition(distance=2)

    ## Mesh item will automatically compute face normals.
    m1 = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            shader='shaded',
            #faceColors=colors,
            #shader='balloon',
            #smooth=False,
            smooth=True,
            )
    #m1.translate(5, 5, 0)
    #m1.setGLOptions('additive')
    w.addItem(m1)

    app.exec_()


def show_tri_qtgraph(M, T, v):

    x, y, z = get_xyz(M, T, v)

    ## Create a GL View widget to display data
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    w.setCameraPosition(distance=50)

    ## Add a grid to the view
    g = gl.GLGridItem()
    g.scale(2,2,1)
    # draw grid after surfaces since they may be translucent
    g.setDepthValue(10)
    w.addItem(g)

    ## Saddle example with x and y specified
    print x.shape
    print y.shape
    print z.shape
    x = np.linspace(-8, 8, 50)
    y = np.linspace(-8, 8, 50)
    z = 0.1 * ((x.reshape(50,1) ** 2) - (y.reshape(1,50) ** 2))
    print x.shape
    print y.shape
    print z.shape
    #p2 = gl.GLSurfacePlotItem(x=x, y=y, z=z, shader='normalColor')
    p2 = gl.GLSurfacePlotItem(x=x, y=y, z=z, shader='shaded')
    #p2.translate(-10,-10,0)
    w.addItem(p2)

    app.exec_()


def drawtri(M, T, v):
    x, y, z = get_xyz(M, T, v)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    plt.show()


def do_full_simplex_then_collapse(mutrate, popsize):
    #mutrate = 0.01
    #mutrate = 0.2
    #mutrate = 10
    #mutrate = 100
    #mutrate = 1
    N = popsize
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

def do_collapsed_simplex(scaled_mut, N):
    """
    @param N: population size
    """
    k = 3
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    # Create the joint site pair mutation rate matrix.
    # This is scaled so that there are about popsize mutations per generation.
    R_mut_raw = wrightcore.create_mutation_collapsed(M, T)
    R_mut = (scaled_mut / float(N)) * R_mut_raw
    # Create the joint site pair drift transition matrix.
    lmcs = wrightcore.get_lmcs(M)
    lps = wrightcore.create_selection_neutral(M)
    #log_drift = wrightcore.create_neutral_drift(lmcs, lps, M)
    # Define the drift and mutation transition matrices.
    #P_drift = np.exp(log_drift)
    #P_mut = scipy.linalg.expm(R)
    # Define the composite per-generation transition matrix.
    #P = np.dot(P_mut, P_drift)
    # Solve a system of equations to find the stationary distribution.
    #v = MatrixUtil.get_stationary_distribution(P)
    # Try a new thing.
    # The raw drift matrix is scaled so that there are about N*N
    # replacements per generation.
    generation_rate = 1.0
    R_drift_raw = wrightcore.create_moran_drift_rate(M, T)
    R_drift = (generation_rate / float(N)) * R_drift_raw
    #FIXME: you should get the stationary distn directly from the rate matrix
    P = scipy.linalg.expm(R_mut + R_drift)
    v = MatrixUtil.get_stationary_distribution(P)
    """
    for state, value in zip(M, v):
        print state, value
    """
    # draw an equilateral triangle
    #drawtri(M, T, v)
    return M, T, v

def check_collapsed_equilibrium_equivalence():
    mutrate = 0.01
    popsize = 20
    Ma, Ta, va = do_full_simplex_then_collapse(mutrate, popsize)
    Mb, Tb, vb = do_collapsed_simplex(mutrate, popsize)
    print Ma - Mb
    print Ta - Tb
    print va - vb

def main():
    #scaled_mu = 0.01
    #scaled_mu = 0.1
    #scaled_mu = 1.0
    #scaled_mu = 0.5
    scaled_mu = 2.0
    #scaled_mu = 10
    N = 50
    mu = scaled_mu / float(N)
    M, T, v = do_collapsed_simplex(scaled_mu, N)
    # check the moments of the distribution
    m1 = 0.0
    m2 = 0.0
    ex1x4 = 0.0
    for i, p in enumerate(v):
        x = M[i, 0] / float(N)
        y = M[i, 2] / float(N)
        m1 += x*p
        m2 += x*x*p
        ex1x4 += x*y*p
    # compute the second moment according to the formula from jeff
    a = 8*(N*mu)*(N*mu) + 8*N*mu + 1
    b = 4*(4*N*mu + 1)*(8*N*mu + 1)
    em2_j = a/b
    # compute the covariance of a product of opposite corner state frequencies
    a = 2*(N*mu)*(N*mu)
    b = (4*N*mu + 1)*(8*N*mu + 1)
    ex1x4_j = a/b
    cx1x2 = -1/(16*(8*N*mu + 1))
    # compute the second moment as a mixture
    u = N * mu
    em2_mix = (u + 1) / (4 * (2*u + 1) )
    print 'observed E[X_1] =', m1
    print 'expected E[X_1] = 0.25'
    print
    print 'this uses the formula from jeff:'
    print 'observed E[X_1 ^ 2] =', m2
    print 'expected E[X_1 ^ 2] =', a/b
    print
    print 'this uses the formula from jeff:'
    print 'observed E[X_1 * X_4] =', ex1x4
    print 'expected E[X_1 * X_4] =', ex1x4_j
    print
    print 'should be better when popsize-scaled mutation is small:'
    print 'observed E[X_1 ^ 2] =', m2
    print 'expected E[X_1 ^ 2] =', em2_mix
    #drawtri(M, T, v)
    #show_tri_qtgraph(M, T, v)
    show_tri_qtgraph_mesh(M, T, v)

if __name__ == '__main__':
    main()

