from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "expanded_dataset"
FIGURES = EXPERIMENT / "figures"


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(EXPERIMENT / "page_level_results.csv")
    cer = data.dropna(subset=["cer"])

    counts = data.groupby("notebook_id").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=ax, color="#4472C4")
    ax.set(title="Pages sampled by notebook", xlabel="Notebook", ylabel="Pages")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "pages_by_notebook.png")

    metrics = ["baseline_consistency", "spatial_organization", "text_fragmentation"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, metric in zip(axes, metrics):
        ax.hist(data[metric], bins=8, color="#70AD47", edgecolor="white")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Score")
        ax.set_ylabel("Pages")
    save(fig, "metric_distributions.png")

    for target, label, name in [
        ("cer", "CER", "score_vs_cer.png"),
        ("ocr_confidence", "OCR confidence", "score_vs_ocr_confidence.png"),
    ]:
        subset = data.dropna(subset=[target])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(subset["combined_score"], subset[target], color="#C00000")
        ax.set(
            title=f"Combined readability score vs. {label}",
            xlabel="Combined readability score",
            ylabel=label,
        )
        save(fig, name)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, metric in zip(axes, metrics):
        ax.scatter(cer[metric], cer["cer"], color="#ED7D31")
        ax.set(
            title=metric.replace("_", " ").title(),
            xlabel="Score",
            ylabel="CER",
        )
    save(fig, "individual_metrics_vs_cer.png")

    grouped = [
        group["combined_score"].to_numpy()
        for _, group in data.groupby("notebook_id")
    ]
    labels = list(data.groupby("notebook_id").groups)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(grouped, tick_labels=labels)
    ax.set(title="Combined score by notebook", xlabel="Notebook", ylabel="Score")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "score_by_notebook.png")

    sensitivity = pd.read_csv(EXPERIMENT / "weight_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        sensitivity["weighting"],
        sensitivity["pearson_score_vs_cer"].fillna(0),
        color="#5B9BD5",
    )
    ax.set(
        title="Weight sensitivity: Pearson score vs. CER",
        xlabel="Weighting",
        ylabel="Pearson correlation",
    )
    ax.tick_params(axis="x", rotation=30)
    save(fig, "weight_sensitivity.png")


if __name__ == "__main__":
    main()
