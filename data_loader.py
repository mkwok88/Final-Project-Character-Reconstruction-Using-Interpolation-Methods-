"""
Loading handwritten character data.

Real data: UCI "Character Trajectories" dataset
    https://archive.ics.uci.edu/dataset/175/character+trajectories
Download the zip, extract it, and pass the path of `mixoutALL_shifted.mat`.
The file stores, for each character, the pen-tip VELOCITY in x and y
(plus pen force) sampled at 200 Hz. We recover positions by numerically
integrating the velocities with the composite trapezoidal rule.

Synthetic data: a few parametric letter-like curves so the code can be
tested without downloading anything.
"""

import numpy as np


def cumulative_trapezoid(v, dt=1.0):
    """
    Integrate sampled velocity v(t) to position using the composite
    trapezoidal rule:  x_k = x_{k-1} + dt * (v_{k-1} + v_k) / 2,  x_0 = 0.
    """
    v = np.asarray(v, dtype=float)
    x = np.zeros_like(v)
    x[1:] = np.cumsum(dt * (v[:-1] + v[1:]) / 2.0)
    return x


def load_uci_characters(mat_path, max_per_letter=None):
    """
    Load the UCI Character Trajectories .mat file.

    Returns a list of dicts: {"label": str, "x": array, "y": array}
    with positions obtained by integrating the stored velocities.
    """
    from scipy.io import loadmat

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    mixout = mat["mixout"]                     # one 3 x T array per character
    consts = mat["consts"]
    labels = np.atleast_1d(consts.charlabels)  # 1-based index into consts.key
    keys = np.atleast_1d(consts.key)

    characters = []
    counts = {}
    for traj, lab in zip(mixout, labels):
        letter = str(keys[int(lab) - 1])
        if max_per_letter is not None:
            if counts.get(letter, 0) >= max_per_letter:
                continue
            counts[letter] = counts.get(letter, 0) + 1
        traj = np.asarray(traj, dtype=float)
        vx, vy = traj[0], traj[1]
        # Drop trailing zero padding, if any
        nonzero = np.nonzero((vx != 0) | (vy != 0))[0]
        if len(nonzero) < 10:
            continue
        end = nonzero[-1] + 1
        x = cumulative_trapezoid(vx[:end])
        y = cumulative_trapezoid(vy[:end])
        characters.append({"label": letter, "x": x, "y": y})
    return characters


def synthetic_characters(num_points=150, jitter=0.004, seed=0):
    """
    A few letter-like strokes (parametric curves), sampled densely.
    Used for testing the pipeline without the real dataset.

    Real pen data is never perfectly smooth, so a small random hand jitter
    (a fraction `jitter` of the letter size, smoothed so it looks like
    tremor rather than noise) is added. Without it, these ideal curves
    would favor high-degree polynomials unrealistically.
    """
    rng = np.random.default_rng(seed)
    s = np.linspace(0, 1, num_points)
    chars = []

    # 'o' : a slightly tilted ellipse
    th = 2 * np.pi * s
    chars.append({"label": "o", "x": np.cos(th) + 0.1 * np.sin(th), "y": 1.3 * np.sin(th)})

    # 'e' : loop that opens to the right
    th = 2 * np.pi * (0.15 + 0.95 * s)
    chars.append({"label": "e", "x": np.cos(th) * (1 - 0.2 * s), "y": np.sin(th) * (1 - 0.1 * s) + 0.3 * s})

    # 'l' : tall loop
    th = 2 * np.pi * s
    chars.append({"label": "l", "x": 2 * s + 0.4 * np.sin(th), "y": 3 * np.sin(np.pi * s) ** 2})

    # 'w' : zig-zag with rounded corners
    chars.append({"label": "w", "x": 4 * s, "y": np.abs(np.sin(2 * np.pi * s)) ** 0.8})

    # 's' : S-curve
    th = np.pi * (2 * s - 1)
    chars.append({"label": "s", "x": 0.6 * np.sin(2 * th), "y": -2 * s + 0.05 * np.cos(th)})

    kernel = np.ones(5) / 5.0
    for ch in chars:
        size = np.hypot(np.ptp(ch["x"]), np.ptp(ch["y"]))
        for coord in ("x", "y"):
            tremor = np.convolve(rng.normal(size=num_points), kernel, mode="same")
            ch[coord] = ch[coord] + jitter * size * tremor / tremor.std()
    return chars
