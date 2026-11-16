# Character Reconstruction Using Interpolation Methods

CSCE 440/840 Final Project — Lucas Estevao Moreira De Souza Coelho, Matthew Kwok, Sami Chowdhury, Yared Kinde

We keep only a few points from a handwritten character, rebuild the full
character with three interpolation methods, and measure how close each
reconstruction is to the original.

## Files

| File | Purpose |
|---|---|
| `linear_interp.py` | Piecewise linear interpolation (baseline) |
| `lagrange_interp.py` | Lagrange polynomial interpolation + Chebyshev node selection |
| `cubic_spline.py` | Natural cubic spline (Burden & Faires Algorithm 3.4) |
| `data_loader.py` | Loads the UCI dataset (velocities → positions with the trapezoidal rule) and synthetic test letters |
| `evaluation.py` | Choosing which points to keep, reconstructing, measuring error |
| `main.py` | Runs the full experiment, writes `results.csv` and `figures/*.pdf` |

All three interpolation methods are implemented from scratch with NumPy.

## Setup

```
pip install -r requirements.txt
```

## Data

Download the UCI **Character Trajectories** dataset:
https://archive.ics.uci.edu/dataset/175/character+trajectories

Unzip it and put `mixoutALL_shifted.mat` in the `data/` folder.

## Running

```
python main.py                                       # synthetic test letters (no download needed)
python main.py --data data/mixoutALL_shifted.mat     # real dataset (2858 characters)
python main.py --data data/mixoutALL_shifted.mat --max-per-letter 30   # faster
```

## Method

1. Each character is a densely sampled pen trajectory (x(t), y(t)); this is the ground truth.
2. We keep every k-th point (k = 3, 5, 8, 10, 15, 20, i.e. 33% down to 5% of the points).
   For Lagrange we also test keeping the same number of points at Chebyshev-spaced positions.
3. x(t) and y(t) are interpolated separately, because letters such as "o" or "e"
   loop back on themselves and y is not a function of x. The parameter t is the
   sample index, which is proportional to time since the data is recorded at 200 Hz.
4. Error is the distance between the reconstruction and the true points that were
   removed, divided by the character's size (bounding-box diagonal). We report
   RMSE and maximum error; the median over all characters is used because
   Lagrange errors can be extremely large for a few characters.

## Output

- `results.csv` — error for each method and sampling level
- `figures/error_vs_points.pdf` — error vs. percentage of points kept
- `figures/example_reconstruction.pdf` — one letter reconstructed by each method
- `figures/runge_phenomenon.pdf` — Lagrange with uniform vs. Chebyshev nodes
