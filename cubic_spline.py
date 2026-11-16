"""
Natural cubic spline interpolation (Burden & Faires, Algorithm 3.4).

On each interval [t_j, t_{j+1}] the spline is a cubic

    S_j(t) = a_j + b_j (t - t_j) + c_j (t - t_j)^2 + d_j (t - t_j)^3

chosen so that S, S' and S'' are continuous at every node, with the
"natural" boundary conditions S''(t_0) = S''(t_n) = 0. The coefficients c_j
come from a tridiagonal linear system, solved here with the Thomas
(Crout) algorithm in O(n) operations.
"""

import numpy as np


def natural_spline_coefficients(t_nodes, f_nodes):
    """
    Compute coefficients (a, b, c, d) of the natural cubic spline.

    Returns arrays a, b, c, d of length n (one cubic per interval),
    where n = len(t_nodes) - 1.
    """
    t = np.asarray(t_nodes, dtype=float)
    a = np.asarray(f_nodes, dtype=float).copy()
    n = len(t) - 1
    h = np.diff(t)                                   # h_j = t_{j+1} - t_j

    # Right-hand side alpha_j, j = 1..n-1
    alpha = np.zeros(n + 1)
    for j in range(1, n):
        alpha[j] = (3.0 / h[j]) * (a[j + 1] - a[j]) - (3.0 / h[j - 1]) * (a[j] - a[j - 1])

    # Solve the tridiagonal system for c (Crout factorization)
    l = np.ones(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    for j in range(1, n):
        l[j] = 2.0 * (t[j + 1] - t[j - 1]) - h[j - 1] * mu[j - 1]
        mu[j] = h[j] / l[j]
        z[j] = (alpha[j] - h[j - 1] * z[j - 1]) / l[j]

    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    return a[:-1], b, c[:-1], d


def cubic_spline_interpolate(t_nodes, f_nodes, t_eval):
    """
    Evaluate the natural cubic spline through (t_nodes, f_nodes) at t_eval.
    Needs at least 2 nodes (with 2 nodes it reduces to a straight line).
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)
    a, b, c, d = natural_spline_coefficients(t_nodes, f_nodes)

    j = np.searchsorted(t_nodes, t_eval, side="right") - 1
    j = np.clip(j, 0, len(t_nodes) - 2)
    dt = t_eval - t_nodes[j]
    return a[j] + b[j] * dt + c[j] * dt**2 + d[j] * dt**3


def reconstruct_spline(t_nodes, x_nodes, y_nodes, t_eval):
    """Reconstruct a 2-D stroke by interpolating x(t) and y(t) separately."""
    return (cubic_spline_interpolate(t_nodes, x_nodes, t_eval),
            cubic_spline_interpolate(t_nodes, y_nodes, t_eval))
