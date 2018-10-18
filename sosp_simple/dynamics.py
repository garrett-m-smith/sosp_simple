# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:06:22 2018

@author: garrettsmith

Defining the equations for the dynamics
"""


import numpy as np
from scipy.spatial.distance import cdist
#from numba import jit


#@jit(nopython=True)
def _phi(x, center, gamma):
    """Radial basis function (RBF) \phi.

    Parameters
    ----------
    x : array
        A point in the state space.
    center : array
        The center of the RBF.
    gamma : scalar
        The width parameter for the RBF.

    Returns
    -------
    \phi
        The value of the RBF evaluated at point x 
    """
    assert x.shape == center.shape, "Shape mismatch in phi fn."
    return np.exp(-np.dot(x - center, x - center) / gamma)


#@jit(nopython=True)
def calc_harmony(x, centers, harmonies, gamma):
    """Calculate the global harmony at a given position.

    Parameters
    ----------
    x : array
        A point in the state space.
    centers : array
        The centers of all of the RBFs, one per row.
    harmonies : vector
        The local harmonies of each of the RBFs.
    gamma : scalar
        The width parameter for the RBF.

    Returns
    -------
    harmony
        The global harmony of a point in the state space.
    """
    harmony = 0
    for c in range(centers.shape[0]):
        harmony += harmonies[c] * _phi(x, centers[c], gamma)
    return harmony


#@jit(nopython=True)
#def calc_harmony_bc(x, centers, local_harmonies, gamma, alpha):
#    """Calculate the global harmony at a given position with baseline
#    constraint. Assumes centers is an array with the coordinates of one center
#    on each row.
#    """
#    harmony = 0
#    for c in range(centers.shape[0]):
#        harmony += local_harmonies[c] * _phi(x, centers[c], gamma)
#        # Baseline constraint: pulling state towards the middle
#    harmony -= alpha/2 * np.dot(x-0.5, x-0.5)
#    return harmony


#@jit(nopython=True)
def iterate(x, centers, harmonies, gamma):
    """Iterate the discretized dynamics by evaluating the dynamical equations dx/dt = f(x).

    Parameters
    ----------
    x : array
        A point in the state space.
    centers : array
        The centers of all of the RBFs, one per row.
    harmonies : vector
        The local harmonies of each of the RBFs.
    gamma : scalar
        The width parameter for the RBF.

    Returns
    -------
    dx
        The value of dx/dt = f(x).
    """
    dx = np.zeros(x.shape)
    mult = -2./gamma
    for c in range(centers.shape[0]):
        dx += (harmonies[c] * (x - centers[c]) * _phi(x, centers[c], gamma))
    return mult * dx


#@jit(nopython=True)
#def iterate_bc(x, centers, harmonies, gamma, alpha):
#    """Iterate the discretized dynamics.
#    """
#    dx = np.zeros(x.shape)
#    mult = -2./gamma
#    for c in range(centers.shape[0]):
#        dx += (harmonies[c] * (x - centers[c]) * _phi(x, centers[c], gamma))
#    dx *= mult
#    return dx - alpha * (x - 0.5)


def euclid_stop(x, attrs, tol):
    """Returns True if not within a certain radius of an attractor using Euclidean distance.

    Parameters
    ----------
    x : array
        A point in the state space.
    attrs : array
        The locations of the attractors of the dynamics, one per row.
    tol :
        The tolerance; how close the sytem has to get before stopping.

    Returns
    -------
    Boolean
        Returns true as long as the system is not within tol of any attractor.
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    dists = cdist(attrs, x, metric='euclidean')
    if np.any(dists < tol):
        return False
    else:
        return True


def vel_stop(x, attrs, tol):
    """Returns True if the velocity of the system is not less than tol.

    Parameters
    ----------
    x : array
        A point in the state space.
    attrs : array
        The locations of the attractors of the dynamics, one per row. Not used.
    tol :
        The tolerance; how close the sytem has to get before stopping.

    Returns
    -------
    Boolean
        Returns true as long as the system has not slowed to less than tol.
    """
    vel = np.linalg.norm(x)
    if vel < tol:
        return False
    else:
        return True


def cheb_stop(x, attrs, tol):
    """Returns True if the maximum distance of any dimension from the attr. is
    greater than tol, i.e., Chebyshev distance.

    Parameters
    ----------
    x : array
        A point in the state space.
    attrs : array
        The locations of the attractors of the dynamics, one per row.
    tol :
        The tolerance; how close the sytem has to get before stopping.

    Returns
    -------
    Boolean
        Returns true as long as the system is not within tol of an attractor on any dimension.
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    dists = cdist(attrs, x, metric='chebyshev')
    if np.any(dists < tol):
        return False
    else:
        return True
