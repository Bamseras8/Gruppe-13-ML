"""Make three graphs from results/summary.csv. Requires matplotlib."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save images without opening extra windows.
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ["baseline", "hires", "multiscale"]
COLORS = ["#3478A5", "#E49A35", "#31856B"]


def read_summary():
    path = RESULTS / "summary.csv"
    if not path.is_file():
        raise FileNotFoundError("Run evaluate.py first: results/summary.csv is missing.")
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 3 or {row["model"] for row in rows} != set(MODELS):
        raise ValueError("Expected one row each for baseline, hires and multiscale.")
    rows = sorted(rows, key=lambda row: MODELS.index(row["model"]))
    for row in rows:
        for key in ("images", "under", "exact", "over"):
            row[key] = int(row[key])
        for key in ("threshold", "precision", "recall", "f1", "mae", "rmse"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"Invalid {key} for {row['model']}.")
        if row["images"] <= 0 or min(row[k] for k in ("under", "exact", "over")) < 0:
            raise ValueError("Image counts must be positive and outcomes non-negative.")
        if sum(row[k] for k in ("under", "exact", "over")) != row["images"]:
            raise ValueError("Under + exact + over must equal the image count.")
        if any(not 0 <= row[k] <= 1 for k in ("threshold", "precision", "recall", "f1")):
            raise ValueError("Threshold and detection scores must be between 0 and 1.")
        if min(row["mae"], row["rmse"]) < 0:
            raise ValueError("Counting errors cannot be negative.")
    if len({row["images"] for row in rows}) != 1:
        raise ValueError("Models must have the same number of evaluated images.")
    return rows


def new_graph(rows, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.set_title(title, fontsize=15, pad=18)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(rows)), MODELS)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    thresholds = ", ".join(f"{r['model']} {r['threshold']:.2f}" for r in rows)
    fig.text(0.5, 0.025,
             f"{rows[0]['images']} test images | Confidence thresholds: {thresholds}",
             ha="center", fontsize=9, color="#444444")
    return fig, ax


def save_graph(fig, filename):
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = RESULTS / filename
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def grouped_graph(rows, keys, title, ylabel, filename, score=False):
    fig, ax = new_graph(rows, title, ylabel)
    width = 0.22
    for i, key in enumerate(keys):
        positions = [x + (i - (len(keys) - 1) / 2) * width for x in range(len(rows))]
        bars = ax.bar(positions, [r[key] for r in rows], width,
                      label=key.upper() if key in ("mae", "rmse", "f1") else key.title(),
                      color=COLORS[i])
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    maximum = max(r[key] for r in rows for key in keys)
    ax.set_ylim(0, 1 if score else max(maximum * 1.3, 0.1))
    ax.legend(loc="upper right", ncol=len(keys), frameon=False)
    save_graph(fig, filename)


def outcome_graph(rows):
    fig, ax = new_graph(rows, "Vehicle counting outcomes", "Number of images")
    bottom = [0] * len(rows)
    for key, label, color in zip(
        ("under", "exact", "over"),
        ("Undercount", "Exact count", "Overcount"),
        ("#E49A35", "#31856B", "#3478A5"),
    ):
        values = [r[key] for r in rows]
        bars = ax.bar(range(len(rows)), values, width=0.55,
                      bottom=bottom, label=label, color=color)
        ax.bar_label(bars, labels=[str(v) if v else "" for v in values],
                     label_type="center", color="white", fontsize=12, fontweight="bold")
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_ylim(0, rows[0]["images"] * 1.18)
    ax.legend(loc="upper center", ncol=3, frameon=False)
    save_graph(fig, "counting_outcomes.png")


def main():
    rows = read_summary()
    grouped_graph(rows, ["mae", "rmse"], "Counting error (lower is better)",
                  "Count error (vehicles per image)", "counting_errors.png")
    grouped_graph(rows, ["precision", "recall", "f1"],
                  "Detection scores (higher is better)", "Score",
                  "detection_scores.png", score=True)
    outcome_graph(rows)
    print("Done: three graphs saved in the results folder.")


if __name__ == "__main__":
    main()
