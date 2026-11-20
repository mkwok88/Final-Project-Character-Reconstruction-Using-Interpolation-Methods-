"""
Subsampling a character and measuring reconstruction error.

Procedure for one character:
  1. The full, densely sampled stroke is the ground truth.
  2. Keep only some points (every k-th point).
  3. Rebuild x(t) and y(t) from the kept points with an interpolation method.
  4. Measure the distance between the rebuilt curve and the TRUE points that
     were removed (held-out points).

Errors are divided by the size of the character (diagonal of its bounding
box) so characters of different sizes can be compared; an error of 0.05
means 5% of the character's size.
"""

import numpy as np

from linear_interp import reconstruct_linear
from lagrange_interp import reconstruct_lagrange
from cubic_spline import reconstruct_spline

METHODS = {
    "Piecewise linear": reconstruct_linear,
    "Lagrange": reconstruct_lagrange,
    "Cubic spline": reconstruct_spline,
}


def every_kth_indices(num_points, k):
    """Indices 0, k, 2k, ... and always the last point."""
    idx = np.arange(0, num_points, k)
    if idx[-1] != num_points - 1:
        idx = np.append(idx, num_points - 1)
    return idx


def character_size(x, y):
    """Diagonal of the bounding box, used to normalize errors."""
    return np.hypot(x.max() - x.min(), y.max() - y.min())


def reconstruct(char, kept_idx, method):
    """
    Rebuild a character from the points at kept_idx.
    The parameter t is the sample index (the data is sampled at a constant
    rate, so this is proportional to time).
    Returns (x_rec, y_rec) evaluated at every original sample.
    """
    x, y = char["x"], char["y"]
    t = np.arange(len(x), dtype=float)
    return METHODS[method](t[kept_idx], x[kept_idx], y[kept_idx], t)


def reconstruction_error(char, kept_idx, method):
    """
    Normalized error at held-out points.
    Returns dict with RMSE and maximum error (both as fraction of char size).
    """
    x, y = char["x"], char["y"]
    x_rec, y_rec = reconstruct(char, kept_idx, method)

    held_out = np.setdiff1d(np.arange(len(x)), kept_idx)
    dist = np.hypot(x_rec[held_out] - x[held_out], y_rec[held_out] - y[held_out])
    size = character_size(x, y)
    return {
        "rmse": np.sqrt(np.mean(dist**2)) / size,
        "max": dist.max() / size,
    }


def kept_indices(num_points, k):
    """Choose which points to keep: every k-th point plus the last point."""
    return every_kth_indices(num_points, k)