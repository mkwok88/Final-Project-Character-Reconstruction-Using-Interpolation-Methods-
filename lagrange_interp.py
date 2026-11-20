"""
Lagrange polynomial interpolation.

Given n+1 nodes t_0, ..., t_n with values f(t_0), ..., f(t_n), the unique
polynomial of degree <= n passing through all of them is

    P(t) = sum_{k=0}^{n} L_{n,k}(t) f(t_k),
    L_{n,k}(t) = prod_{i=0, i != k}^{n} (t - t_i) / (t_k - t_i).

A single high-degree polynomial through equally spaced nodes oscillates
badly near the ends of the interval (Runge's phenomenon), so keeping MORE
points can make the reconstruction WORSE.
"""

import numpy as np


def _to_unit_interval(t, a, b):
    """Map [a, b] -> [-1, 1]. Keeps the products in L_k(t) well scaled."""
    return 2.0 * (np.asarray(t, dtype=float) - a) / (b - a) - 1.0


def lagrange_interpolate(t_nodes, f_nodes, t_eval):
    """
    Evaluate the Lagrange interpolating polynomial through (t_nodes, f_nodes).

    Parameters
    ----------
    t_nodes : 1-D array of distinct parameter values
    f_nodes : 1-D array of values at t_nodes
    t_eval  : 1-D array of parameter values to evaluate at

    Returns
    -------
    1-D array of P(t_eval)
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    f_nodes = np.asarray(f_nodes, dtype=float)
    a, b = t_nodes.min(), t_nodes.max()
    s_nodes = _to_unit_interval(t_nodes, a, b)
    s_eval = _to_unit_interval(t_eval, a, b)

    n = len(s_nodes) - 1                     # polynomial degree
    result = np.zeros_like(s_eval)
    for k in range(n + 1):
        L_nk = np.ones_like(s_eval)          # basis function L_{n,k}(t)
        for i in range(n + 1):
            if i != k:
                L_nk *= (s_eval - s_nodes[i]) / (s_nodes[k] - s_nodes[i])
        result += L_nk * f_nodes[k]
    return result


def reconstruct_lagrange(t_nodes, x_nodes, y_nodes, t_eval):
    """Reconstruct a 2-D stroke by interpolating x(t) and y(t) separately."""
    return (lagrange_interpolate(t_nodes, x_nodes, t_eval),
            lagrange_interpolate(t_nodes, y_nodes, t_eval))