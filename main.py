"""
Run the full experiment: reconstruct every character with each method at
several sampling levels, save a results table and the figures.

Usage
-----
    python main.py                                  # synthetic test letters
    python main.py --data data/mixoutALL_shifted.mat   # real UCI dataset
    python main.py --data data/mixoutALL_shifted.mat --max-per-letter 30
"""

import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_uci_characters, synthetic_characters
from evaluation import METHODS, kept_indices, reconstruct, reconstruction_error

# Keep every k-th point. k = 5 keeps 20% of the points, k = 20 keeps 5%.
K_VALUES = [3, 5, 8, 10, 15, 20]
FIG_DIR = "figures"


def run_experiment(characters):
    """Return a list of result rows (one per method and k)."""
    rows = []
    for method in METHODS:
        for k in K_VALUES:
            rmse, maxe = [], []
            for ch in characters:
                idx = kept_indices(len(ch["x"]), k)
                err = reconstruction_error(ch, idx, method)
                rmse.append(err["rmse"])
                maxe.append(err["max"])
            rows.append({
                "method": method,
                "k": k,
                "percent_kept": round(100.0 / k, 1),
                "median_rmse": np.median(rmse),
                "mean_rmse": np.mean(rmse),
                "median_max_error": np.median(maxe),
            })
    return rows


def save_table(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({key: (f"{v:.6g}" if isinstance(v, float) else v) for key, v in r.items()})


def plot_error_vs_points(rows, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in METHODS:
        sub = [r for r in rows if r["method"] == method]
        ax.plot([r["percent_kept"] for r in sub], [r["median_rmse"] for r in sub],
                marker="o", label=method)
    ax.set_xlabel("Points kept (% of original)")
    ax.set_ylabel("Median RMSE (fraction of character size)")
    ax.set_yscale("log")
    ax.set_title("Reconstruction error vs. number of points kept")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_reconstruction(ax, char, idx, x_rec, y_rec, title):
    """Original stroke, reconstruction and kept points on one set of axes."""
    ax.plot(char["x"], char["y"], color="0.7", lw=3, label="Original")
    ax.plot(x_rec, y_rec, lw=1.5, label="Reconstruction")
    ax.plot(char["x"][idx], char["y"][idx], "k.", ms=6, label="Kept points")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    # Lagrange can blow up far outside the letter; keep the view on the letter
    pad = 0.3 * max(np.ptp(char["x"]), np.ptp(char["y"]))
    ax.set_xlim(char["x"].min() - pad, char["x"].max() + pad)
    ax.set_ylim(char["y"].min() - pad, char["y"].max() + pad)


def plot_example(char, k, path):
    """Original character vs. the three reconstructions from the same points."""
    idx = kept_indices(len(char["x"]), k)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, method in zip(axes, METHODS):
        x_rec, y_rec = reconstruct(char, idx, method)
        _draw_reconstruction(ax, char, idx, x_rec, y_rec, method)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"Letter '{char['label']}', keeping every {k}th point ({100 // k}% of points)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_runge(char, k_few, k_many, path):
    """
    Runge's phenomenon: Lagrange interpolation of the same letter with FEW
    kept points (low degree) vs. MANY kept points (high degree).
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, k in zip(axes, [k_few, k_many]):
        idx = kept_indices(len(char["x"]), k)
        x_rec, y_rec = reconstruct(char, idx, "Lagrange")
        title = f"{len(idx)} points (degree {len(idx) - 1})"
        _draw_reconstruction(ax, char, idx, x_rec, y_rec, title)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"Runge's phenomenon: Lagrange interpolation of letter '{char['label']}'")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="path to mixoutALL_shifted.mat (omit for synthetic letters)")
    parser.add_argument("--max-per-letter", type=int, default=None,
                        help="limit characters per letter (faster runs)")
    args = parser.parse_args()

    if args.data:
        characters = load_uci_characters(args.data, args.max_per_letter)
        print(f"Loaded {len(characters)} characters from {args.data}")
    else:
        characters = synthetic_characters()
        print(f"No --data given: using {len(characters)} synthetic test letters")

    os.makedirs(FIG_DIR, exist_ok=True)
    rows = run_experiment(characters)
    save_table(rows, "results.csv")

    print(f"\n{'Method':20s} {'% kept':>7s} {'median RMSE':>12s}")
    for r in rows:
        print(f"{r['method']:20s} {r['percent_kept']:7.1f} {r['median_rmse']:12.4g}")

    plot_error_vs_points(rows, os.path.join(FIG_DIR, "error_vs_points.pdf"))
    plot_example(characters[0], 10, os.path.join(FIG_DIR, "example_reconstruction.pdf"))
    plot_runge(characters[0], 20, 5, os.path.join(FIG_DIR, "runge_phenomenon.pdf"))
    print("\nSaved results.csv and figures/*.pdf")


if __name__ == "__main__":
    main()