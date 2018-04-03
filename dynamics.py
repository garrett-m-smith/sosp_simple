# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:06:22 2018

@author: garrettsmith

Defining the equations for the dynamics
"""


import numpy as np
from scipy.spatial.distance import cdist


def _phi(x, center, gamma):
    """Radial basis function \phi. Returns a scalar.
    """
    assert x.shape == center.shape, "Shape mismatch in phi fn."
    l2norm = np.linalg.norm(x - center)
    return np.exp(-l2norm**2 / gamma)


def calc_harmony(x, centers, local_harmonies, gamma):
    """Calculate the global harmony at a given position.
    Assumes centers is an array with the coordinates of one center on each
    row.
    """
    harmony = 0
    for c in range(centers.shape[0]):
        harmony += local_harmonies[c] * _phi(x, centers[c], gamma)
    return harmony


def iterate(x, centers, harmonies, gamma):
    """Iterate the discretized dynamics.
    """
    dx = np.zeros(x.shape)
    mult = -2./gamma
    for c in range(centers.shape[0]):
        dx += (mult * harmonies[c]
               * (x - centers[c]) * _phi(x, centers[c], gamma))
    return dx


def euclid_stop(x, centers, tol):
    """Returns True if not within a certain radius of a center.
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    dists = cdist(centers, x, metric='euclidean')
    if np.any(dists < tol):
        return False
    else:
        return True


def vel_stop(x, tol):
    vel = np.sqrt(x @ x)
    if vel < tol:
        return False
    else:
        return True
