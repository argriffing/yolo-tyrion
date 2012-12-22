"""
Do things with autosaved stationary distributions.
"""

import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import multinomstate

def plot_reciprocal_variance_AB_given_AB_plus_Ab(rows):
    filename = 'reciprocal_variance_AB_given_AB_plus_Ab.png'
    N = sum(rows[0][:-1])
    variances = []
    for AB_plus_Ab in range(1, N+1):
        m = np.zeros(AB_plus_Ab+1)
        for AB, Ab, aB, ab, p in rows:
            if AB + Ab == AB_plus_Ab:
                m[AB] += p
        m /= np.sum(m)
        x_values = np.linspace(0, 1, AB_plus_Ab+1)
        mu = np.dot(m, x_values)
        if not np.allclose(mu, 0.5):
            raise Exception
        v = np.dot(m, (x_values - mu)**2)
        variances.append(np.reciprocal(v))
    xs = np.arange(1, N+1)
    ys = variances
    plt.plot(xs, ys)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 2*y2))
    plt.savefig(filename)
    plt.close()

def plot_aB_given_half_AB_plus_Ab(rows):
    N = sum(rows[0][:-1])
    Nd2 = N // 2
    filename = 'AB_given_%d_AB_plus_Ab.png' % Nd2
    m = np.zeros(Nd2 + 1)
    for AB, Ab, aB, ab, p in rows:
        if AB + Ab == Nd2:
            m[aB] += p
    m /= np.sum(m)
    xs = np.arange(Nd2 + 1)
    ys = m
    plt.plot(xs, ys)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 2*y2))
    plt.savefig(filename)
    plt.close()

def plot_aB_given_zero_AB_zero_Ab(rows):
    N = sum(rows[0][:-1])
    filename = 'AB_given_%d_AB_plus_Ab.png' % N
    m = np.zeros(N+1)
    for AB, Ab, aB, ab, p in rows:
        if AB + Ab == N:
            m[AB] += p
    m /= np.sum(m)
    xs = np.arange(N+1)
    ys = m
    plt.plot(xs, ys)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 2*y2))
    plt.savefig(filename)
    plt.close()

def plot_AB_plus_Ab_distn(rows):
    filename = 'AB_plus_Ab.png'
    N = sum(rows[0][:-1])
    m = np.zeros(N+1)
    for AB, Ab, aB, ab, p in rows:
        m[AB+Ab] += p
    m /= np.sum(m)
    xs = np.arange(N+1)
    ys = m
    plt.plot(xs, ys)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 2*y2))
    plt.savefig(filename)
    plt.close()


def main():
    k = 4
    print 'reading state distribution from stdin...'
    rows = []
    for line in sys.stdin.readlines():
        AB, Ab, aB, ab, p = line.split()
        row = [int(AB), int(Ab), int(aB), int(ab), float(p)]
        rows.append(row)
    N = sum(rows[0][:-1])
    print 'defining the state vectors...'
    M = np.array(list(multinomstate.gen_states(N, k)), dtype=int)
    T = multinomstate.get_inverse_map(M)
    m_AB_ab = np.zeros(N+1)
    m_AB_Ab = np.zeros(N+1)
    for AB, Ab, aB, ab, p in rows:
        m_AB_ab[AB+ab] += p
        m_AB_Ab[AB+Ab] += p
    print 'marginal distribution of AB+ab:'
    print m_AB_ab
    print
    print 'marginal distribution of AB+Ab:'
    print m_AB_Ab
    print
    plot_AB_plus_Ab_distn(rows)
    plot_aB_given_zero_AB_zero_Ab(rows)
    plot_aB_given_half_AB_plus_Ab(rows)
    plot_reciprocal_variance_AB_given_AB_plus_Ab(rows)


if __name__ == '__main__':
    main()

