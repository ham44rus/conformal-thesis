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


# --- E2: 下敷きモデルへの非依存性（仕様書 specs/E2_model_agnostic.md「図」）---

# n_cal と alpha は csv に列として持っていないので、仕様書「手続き」の値を定数で置く。
# experiments/e2_model_agnostic.py の N_CAL, ALPHA と同じ値。片方だけ変えないこと。
E2_N_CAL, E2_ALPHA = 500, 0.10

# 横軸のモデル順。e2_model_agnostic.py の MODEL_KEYS と同じ順に固定する
# （幅が単調に下がる順なので、この順でないと「階段状」が見えない）。
E2_MODEL_ORDER = ("constant", "ridge", "rf", "gbm", "mlp")

# 表示用の和名（csv には英字キーが入っている）
E2_LABELS = {
    "constant": "定数予測器",
    "ridge": "リッジ回帰",
    "rf": "ランダム\nフォレスト",
    "gbm": "勾配\nブースティング",
    "mlp": "ニューラル\nネット",
}


def fig_e2_model_agnostic(summary: pd.DataFrame) -> None:
    """図5.3: 被覆はモデルによらず、区間幅だけがモデルで変わることを示す2段組。

    上段が平ら・下段が階段状に見えることがこの図の主張（仕様書「図」の節）。

    上段の縦軸は理論の上下界の幅 1/(n+1) を単位にして取る。データの範囲に
    合わせて自動で詰めるとモデル間のわずかな差が画面いっぱいに拡大され、
    「被覆はモデルによらない」という主張と逆の印象を与えるため。

    上段の誤差棒は `coverage_se`（= 試行間成分とテスト集合間成分の合成 ±1.96 SE）。
    E2 は全試行・全モデルで同じテスト集合を使い回すので、その集合固有のズレは
    試行数では薄まらない。どの成分を含めたかは凡例に明記する
    （CLAUDE.md「報告の作法」）。

    下段の誤差棒は試行間のみ（`width_se`）。学習集合を引き直したときのばらつきは
    含まれていないので、層内の順序をこの誤差棒で論じてはいけない（C4 の問題）。
    """
    g = summary.set_index("model").loc[list(E2_MODEL_ORDER)]
    x = np.arange(len(g))

    lower = 1 - E2_ALPHA                 # 名目値
    band = 1 / (E2_N_CAL + 1)            # 理論の上下界の幅（第2章 定理2.1）
    upper = lower + band

    fig, (ax_cov, ax_w) = plt.subplots(
        2, 1, figsize=(6.0, 4.8), sharex=True, gridspec_kw={"hspace": 0.16}
    )

    # 上段：平均被覆率 -------------------------------------------------
    ax_cov.fill_between([-0.5, len(g) - 0.5], lower, upper, color=RED, alpha=0.16,
                        label="理論の上下界")
    ax_cov.axhline(lower, color=INK, linewidth=1, linestyle=(0, (4, 3)),
                   label=r"名目値 $1-\alpha$")
    ax_cov.axhline(upper, color=RED, linewidth=1.4,
                   label=r"上界 $1-\alpha+1/(n+1)$")
    ax_cov.errorbar(x, g["coverage_mean"], yerr=1.96 * g["coverage_se"],
                    fmt="o-", color=BLUE, markersize=5, linewidth=1.6, capsize=3,
                    label="実測（試行間＋テスト集合間 95%）")

    # 縦軸は理論帯の上下に 1.5 帯分ずつ余白を取る（帯が軸の 1/4 を占める）
    margin = 1.5 * band
    ax_cov.set_ylim(lower - margin, upper + margin)
    ax_cov.set_xlim(-0.5, len(g) - 0.5)
    ax_cov.set_ylabel("平均被覆率")
    ax_cov.grid(axis="y")
    ax_cov.set_axisbelow(True)
    ax_cov.legend(loc="upper left", fontsize=7, ncol=2)

    # 下段：平均区間幅 -------------------------------------------------
    w, w_err = g["width_mean"].to_numpy(), 1.96 * g["width_se"].to_numpy()
    ax_w.errorbar(x, w, yerr=w_err, fmt="o-", color=BLUE, markersize=5,
                  linewidth=1.6, capsize=3, label="実測（試行間 95%）")  # 学習集合間は含まない
    span = w.max() - w.min()
    ax_w.set_ylim(w.min() - 0.18 * span, w.max() + 0.18 * span)
    ax_w.set_ylabel(r"平均区間幅 $2\hat{q}$")
    ax_w.set_xticks(x, [E2_LABELS[k] for k in E2_MODEL_ORDER], fontsize=8)
    ax_w.grid(axis="y")
    ax_w.set_axisbelow(True)

    fig.suptitle(
        rf"被覆はモデルによらず、幅だけが変わる（$\alpha$ = {E2_ALPHA}, $n$ = {E2_N_CAL}）",
        y=0.96, fontsize=11, color=INK,
    )
    save(fig, "fig_e2_model_agnostic")


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

    # E2 は独立に回すので、csv が無ければ飛ばす（E1 の図だけは常に出せるように）
    e2_path = RESULTS / "e2_summary.csv"
    if e2_path.exists():
        fig_e2_model_agnostic(pd.read_csv(e2_path))
    else:
        print("  results/e2_summary.csv が無いため E2 の図は省略（先に `make e2`）")


if __name__ == "__main__":
    main()
