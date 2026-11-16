"""
Lagrange polynomial interpolation.

Given n+1 nodes (t_0, f_0), ..., (t_n, f_n), the unique polynomial of
degree <= n passing through all of them is

    P(t) = sum_{k=0}^{n} f_k * L_k(t),
    L_k(t) = prod_{i != k} (t - t_i) / (t_k - t_i).

A single high-degree polynomial through equally spaced nodes oscillates
badly near the ends of the interval (Runge's phenomenon). Choosing nodes
near the Chebyshev points clusters them toward the ends and reduces this.
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

    n = len(s_nodes)
    result = np.zeros_like(s_eval)
    for k in range(n):
        L_k = np.ones_like(s_eval)
        for i in range(n):
            if i != k:
                L_k *= (s_eval - s_nodes[i]) / (s_nodes[k] - s_nodes[i])
        result += f_nodes[k] * L_k
    return result


def chebyshev_indices(num_points, num_nodes):
    """
    Pick `num_nodes` indices out of range(num_points) that lie closest to the
    Chebyshev points cos((2k+1)pi / (2n)), mapped onto [0, num_points-1].
    The first and last points are always included so the whole stroke is covered.
    """
    n = num_nodes
    k = np.arange(n)
    cheb = np.cos((2 * k + 1) * np.pi / (2 * n))          # in (-1, 1)
    positions = (cheb + 1) / 2 * (num_points - 1)          # in (0, N-1)
    idx = np.unique(np.round(positions).astype(int))
    idx = np.union1d(idx, [0, num_points - 1])
    return idx


def reconstruct_lagrange(t_nodes, x_nodes, y_nodes, t_eval):
    """Reconstruct a 2-D stroke by interpolating x(t) and y(t) separately."""
    return (lagrange_interpolate(t_nodes, x_nodes, t_eval),
            lagrange_interpolate(t_nodes, y_nodes, t_eval))
