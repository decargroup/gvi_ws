import numpy as np
import scipy as sp
import navlie as nav
import matplotlib.pyplot as plt
import itertools

from math import factorial
from numpy.polynomial.hermite_e import hermeroots
from scipy.stats.distributions import chi2
from scipy.special import eval_hermitenorm

def _compute_weight(order_p, sigma_points:np.ndarray):
    H_p = _compute_hermite(order_p-1)
    weights = []
    for sigma_i in sigma_points:
        num = factorial(order_p)
        den = order_p*_eval_hermite(H_p, sigma_i)
        w_i = num / den**2
        weights.append(w_i)
    return weights

def _compute_hermite_coeffs(order_p) -> np.ndarray:
    if order_p == 0:
        return np.array([1])
    elif order_p == 1:
        return np.array([1, 0])
    else:
        h_p = np.append(_compute_hermite_coeffs(order_p-1),0)
        h_p_2 = -1*(order_p-1)*_compute_hermite_coeffs(order_p-2)
        h_p_1 = np.concatenate([[0, 0], h_p_2])
        return h_p + h_p_1

def _compute_hermite(order_p):
    poly = _compute_hermite_coeffs(order_p)
    return poly

def _eval_hermite(polynomial, point):
    weight = 0
    poly = polynomial[::-1]
    for i in range(len(poly)):
        w_i = poly[i]*(point**i)
        weight+=w_i
    return weight

def gh_cubature(order_p, state_dof):
    hermite_poly = _compute_hermite(order_p)
    unit_sigma_points = np.roots(hermite_poly)
    unit_sigma_points_stack = np.zeros((order_p**state_dof, state_dof))
    # Form cartesian product of sigma points
    count = 0
    for x in itertools.product(unit_sigma_points, repeat=state_dof):
            unit_sigma_points_stack[count, :] = np.array(x)
            count +=1
    # Form weights from multiplying cartesian product of sigma point weights together
    weights = _compute_weight(order_p, unit_sigma_points)
    weights_stack = np.zeros(order_p**state_dof)
    count = 0
    for ws in itertools.product(weights, repeat=state_dof):
        weights_stack[count] = np.prod(ws)
        count += 1
    return unit_sigma_points_stack, weights_stack

def spherical_cubature(state_dof:int, order_p=None):
    sigma_points = np.sqrt(state_dof) * np.block([[np.eye(state_dof), -np.eye(state_dof)]])
    w = 1 / (2 * state_dof) * np.ones((2 * state_dof))
    return sigma_points.T, w