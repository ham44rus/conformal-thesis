"""results/ の csv から figures/ の図を再生成する。

対応箇所: 卒論 第5章の図

実験を回し直さずに図だけ直せるようにするため、実験の実行と作図は分離している。
締切前の修正で効いてくるので、この分離は崩さないこと。

出力形式は PDF（卒論の TeX に貼る用。提出要領で eps もしくは画像の pdf が指定されている）
と PNG（画面確認用）の両方。

実行:
    python experiments/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# 検証済みの2色（colorblind-safe, 印刷でも判別可）
BLUE = "#0B6E99"   # 実測（経験分布）
RED = "#C0492F"    # 理論（Beta 密度・境界）
INK = "#2A2E35"
MUTED = "#6B7280"

# 日本語フォント。環境にあるものを順に試す（Windows なら "Meiryo" / "Yu Gothic" が入る）
_JP_FONTS = ["Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "Meiryo", "Yu Gothic", "Hiragino Sans"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": _JP_FONTS + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.edgecolor": "#C9CFD6",
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#E5E9ED",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}.pdf, figures/{name}.png")


def fig_coverage_beta(df: pd.DataFrame, alpha: float = 0.10) -> None:
    """図5.1: 被覆率の経験分布と理論 Beta 密度の重ね描き。

    キャリブレーションサイズ n ごとに1枚。実測のヒストグラムに理論密度が
    そのまま乗ることを示す、本論文の中心的な図。
    """
    ns = sorted(df["n_cal"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(2.5 * len(ns), 2.4), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, n_cal in zip(axes, ns):
        cov = df[(df.n_cal == n_cal) & (df.alpha == alpha)]["coverage"].to_numpy()
        if cov.size == 0:
            continue
        l = int(np.floor((n_cal + 1) * alpha))
        a_par, b_par = n_cal + 1 - l, l

        ax.hist(cov, bins=30, density=True, color=BLUE, alpha=0.75,
                edgecolor="white", linewidth=0.4, label="実測（経験分布）")
        if b_par > 0:
            xs = np.linspace(cov.min(), cov.max(), 400)
            ax.plot(xs, stats.beta.pdf(xs, a_par, b_par), color=RED, linewidth=2,
                    label="理論 Beta 密度")
        ax.axvline(1 - alpha, color=INK, linewidth=1, linestyle=(0, (4, 3)),
                   label=r"名目値 $1-\alpha$")
        ax.set_title(f"$n$ = {n_cal}  ·  Beta({a_par}, {b_par})", fontsize=9)
        ax.set_xlabel("被覆率")
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("密度")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.30))
    fig.suptitle(rf"被覆率の分布（$\alpha$ = {alpha}）", y=1.04, fontsize=11, color=INK)
    save(fig, f"fig_e1_coverage_beta_alpha{int(alpha * 100):02d}")


def fig_bounds(summary: pd.DataFrame) -> None:
    """図5.2: 平均被覆率が理論の上下界に挟まれることの確認。"""
    fig, axes = plt.subplots(1, summary["alpha"].nunique(), figsize=(3.1 * summary["alpha"].nunique(), 2.6),
                             sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (alpha, g) in zip(axes, summary.groupby("alpha")):
        g = g.sort_values("n_cal")
        x = np.arange(len(g))
        ax.fill_between(x, g["lower_bound"], g["upper_bound"], color=RED, alpha=0.16,
                        label="理論の上下界")
        ax.plot(x, g["lower_bound"], color=RED, linewidth=1.4)
        ax.plot(x, g["upper_bound"], color=RED, linewidth=1.4)
        ax.plot(x, g["coverage_mean"], "o-", color=BLUE, linewidth=2, markersize=5,
                label="実測の平均被覆率")
        ax.set_xticks(x, [str(v) for v in g["n_cal"]])
        ax.set_xlabel("キャリブレーションサイズ $n$")
        ax.set_title(rf"$\alpha$ = {alpha}")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("被覆率")
    axes[-1].legend(loc="upper right", fontsize=7.5)
    save(fig, "fig_e1_bounds")


def main() -> None:
    cov_path, sum_path = RESULTS / "e1_coverage.csv", RESULTS / "e1_summary.csv"
    if not cov_path.exists():
        raise SystemExit("results/e1_coverage.csv がありません。先に `make e1` を実行してください。")
    df = pd.read_csv(cov_path)
    summary = pd.read_csv(sum_path)
    print("作図:")
    for a in sorted(df["alpha"].unique()):
        fig_coverage_beta(df, alpha=float(a))
    fig_bounds(summary)


if __name__ == "__main__":
    main()
