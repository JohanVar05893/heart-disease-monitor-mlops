"""Exploratory data analysis: writes professional-quality figures to a directory."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data import CLEAN_PATH  # noqa: E402
from src.features import CATEGORICAL, NUMERIC  # noqa: E402

FIG_DIR = Path("screenshots/eda")
PALETTE = {0: "#4C78A8", 1: "#E45756"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, out: Path, name: str) -> Path:
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def class_balance(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="target", hue="target", palette=PALETTE, legend=False, ax=ax)
    for bar in ax.patches:
        ax.annotate(int(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_title("Class balance")
    return _save(fig, out, "class_balance.png")


def histograms(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), NUMERIC):
        sns.histplot(data=df, x=col, hue="target", palette=PALETTE, kde=True,
                     element="step", ax=ax)
        ax.set_title(f"{col} by target")
    axes.ravel()[-1].axis("off")
    return _save(fig, out, "histograms.png")


def boxplots(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, len(NUMERIC), figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC):
        sns.boxplot(data=df, x="target", y=col, hue="target", palette=PALETTE,
                    legend=False, ax=ax)
    return _save(fig, out, "boxplots.png")


def categorical_vs_target(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL):
        sns.countplot(data=df, x=col, hue="target", palette=PALETTE, ax=ax)
        ax.set_title(col)
        ax.legend(title="target", fontsize=8)
    return _save(fig, out, "categorical_vs_target.png")


def correlation_heatmap(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature correlation (Pearson)")
    return _save(fig, out, "correlation_heatmap.png")


def missing_values(df: pd.DataFrame, out: Path) -> Path:
    miss = df.isna().sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    miss.plot.bar(color="#72B7B2", ax=ax)
    ax.set_title("Missing values per column")
    ax.set_ylabel("count")
    return _save(fig, out, "missing_values.png")


def run(path=CLEAN_PATH, out: Path = FIG_DIR) -> list:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    figs = [fn(df, out) for fn in (class_balance, histograms, boxplots,
                                   categorical_vs_target, correlation_heatmap,
                                   missing_values)]
    return figs


if __name__ == "__main__":
    for p in run():
        print("wrote", p)
