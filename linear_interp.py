"""
Piecewise linear interpolation (baseline method).

Between two consecutive known points (t_i, f_i) and (t_{i+1}, f_{i+1}),
the value at t is

    P(t) = f_i + (f_{i+1} - f_i) * (t - t_i) / (t_{i+1} - t_i)

i.e. the known points are connected by straight lines.
"""

import numpy as np


def linear_interpolate(t_nodes, f_nodes, t_eval):
    """
    Evaluate the piecewise linear interpolant through (t_nodes, f_nodes).

    Parameters
    ----------
    t_nodes : 1-D array, strictly increasing parameter values of known points
    f_nodes : 1-D array, function values (x or y coordinate) at t_nodes
    t_eval  : 1-D array, parameter values where the interpolant is evaluated
              (must lie inside [t_nodes[0], t_nodes[-1]])

    Returns
    -------
    1-D array of interpolated values at t_eval
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    f_nodes = np.asarray(f_nodes, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    # For each evaluation point, find the interval [t_i, t_{i+1}] containing it.
    i = np.searchsorted(t_nodes, t_eval, side="right") - 1
    i = np.clip(i, 0, len(t_nodes) - 2)

    t0, t1 = t_nodes[i], t_nodes[i + 1]
    f0, f1 = f_nodes[i], f_nodes[i + 1]
    return f0 + (f1 - f0) * (t_eval - t0) / (t1 - t0)


def reconstruct_linear(t_nodes, x_nodes, y_nodes, t_eval):
    """Reconstruct a 2-D stroke by interpolating x(t) and y(t) separately."""
    return (linear_interpolate(t_nodes, x_nodes, t_eval),
            linear_interpolate(t_nodes, y_nodes, t_eval))
