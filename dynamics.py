# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:06:22 2018

@author: garrettsmith

Defining the equations for the dynamics
"""


import numpy as np
from scipy.spatial.distance import cdist
from numba import jit


@jit(nopython=True)
def _phi(x, center, gamma):
    """Radial basis function \phi. Returns a scalar.
    """
    assert x.shape == center.shape, "Shape mismatch in phi fn."
    # Using Einstein summation notation for speed. Multiplies repeated indices,
    # sums if across an index if it doesn't appear on the RHS of ->
#    return np.exp(-np.einsum('i,i->', x - center, x - center) / gamma)
    return np.exp(-np.dot(x - center, x - center) / gamma)


@jit(nopython=True)
def calc_harmony(x, centers, local_harmonies, gamma):
    """Calculate the global harmony at a given position.
    Assumes centers is an array with the coordinates of one center on each
    row.
    """
    harmony = 0
    for c in range(centers.shape[0]):
        harmony += local_harmonies[c] * _phi(x, centers[c], gamma)
    return harmony


@jit(nopython=True)
def calc_harmony_bc(x, centers, local_harmonies, gamma, alpha):
    """Calculate the global harmony at a given position with baseline
    constraint. Assumes centers is an array with the coordinates of one center
    on each row.
    """
    harmony = 0
    for c in range(centers.shape[0]):
        harmony += local_harmonies[c] * _phi(x, centers[c], gamma)
        # Baseline constraint: pulling state towards the middle
    harmony -= alpha/2 * np.dot(x-0.5, x-0.5)
    return harmony


@jit(nopython=True)
def iterate(x, centers, harmonies, gamma):
    """Iterate the discretized dynamics.
    """
    dx = np.zeros(x.shape)
    mult = -2./gamma
    for c in range(centers.shape[0]):
        dx += (harmonies[c] * (x - centers[c]) * _phi(x, centers[c], gamma))
    return mult * dx


@jit(nopython=True)
def iterate_bc(x, centers, harmonies, gamma, alpha):
    """Iterate the discretized dynamics.
    """
    dx = np.zeros(x.shape)
    mult = -2./gamma
    for c in range(centers.shape[0]):
        dx += (harmonies[c] * (x - centers[c]) * _phi(x, centers[c], gamma))
    dx *= mult
    return dx - alpha * (x - 0.5)


def euclid_stop(x, attrs, tol):
    """Returns True if not within a certain radius of an attractor.
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    dists = cdist(attrs, x, metric='euclidean')
    if np.any(dists < tol):
        return False
    else:
        return True


def vel_stop(x, centers, tol):
#    vel = np.sqrt(x @ x)
    vel = np.linalg.norm(x)
    if vel < tol:
        return False
    else:
        return True


def cheb_stop(x, attrs, tol):
    """Returns True if the maximum distance of any dimension from the attr. is
    greater than tol.
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    dists = cdist(attrs, x, metric='chebyshev')
    if np.any(dists < tol):
        return False
    else:
        return True
