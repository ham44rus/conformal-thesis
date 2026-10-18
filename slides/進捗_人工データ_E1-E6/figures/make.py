# スライド専用の図（論文の図ではない）。results/ の csv から作る。
import sys, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

root, out = sys.argv[1], sys.argv[2]
INK, MUTED, GRID = "#1F2A44", "#6B7486", "#E3E7ED"
COL = {"abs": "#6B7486", "norm": "#2F6DB5", "cqr": "#C46F1A"}
NAME = {"abs": "絶対残差", "norm": "正規化残差", "cqr": "CQR"}
MK = {"abs": "o", "norm": "s", "cqr": "^"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Hiragino Sans", "Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 13, "axes.edgecolor": "#C9CFD6", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

def clean(ax):
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)

# 実験1：被覆率のばらつきと理論（Beta）。alpha=0.10、較正用 20 / 100 / 2000 点
e1 = pd.read_csv(f"{root}/results/e1_coverage.csv")
f, axes = plt.subplots(1, 3, figsize=(8, 3.1))
for ax, n in zip(axes, [20, 100, 2000]):
    c = e1[(e1.n_cal == n) & (np.isclose(e1.alpha, 0.10))].coverage.to_numpy()
    k = math.ceil((n + 1) * 0.9)
    ax.hist(c, bins=30, density=True, color="#9DB8DA", edgecolor="white", linewidth=0.6)
    x = np.linspace(c.min(), c.max(), 400)
    ax.plot(x, stats.beta.pdf(x, k, n + 1 - k), color="#2F6DB5", lw=2.4)
    ax.axvline(0.9, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.set_title(f"較正用 {n:,} 点", fontsize=14, color=INK, pad=6)
    ax.set_yticks([]); clean(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.locator_params(axis="x", nbins=4)
    ax.set_xlabel("1 回ごとの当たる割合", fontsize=11, color=MUTED)
f.tight_layout(); f.savefig(f"{out}/e1_beta.png", dpi=220, facecolor="white")

# 実験3：組ごとの当たる割合（不均一分散データ、X1 の 5 分位の組）
e3 = pd.read_csv(f"{root}/results/e3_summary.csv")
f, axes = plt.subplots(1, 3, figsize=(8, 3.3), sharey=True)
for ax, methods, t in zip(axes, [["abs"], ["abs", "norm"], ["abs", "cqr"]], ["絶対残差", "正規化残差", "CQR"]):
    ax.axhline(0.9, color=INK, lw=1.2, ls=(0, (4, 3)))
    for m in methods:
        y = e3[e3.method == m].sort_values("bin").coverage_mean.to_numpy()
        faded = (m == "abs" and len(methods) > 1)
        ax.plot(range(1, 6), y, color=COL[m], lw=1.6 if faded else 2.4, marker=MK[m], ms=7,
                alpha=0.65 if faded else 1, markeredgecolor="white", markeredgewidth=1.2)
    ax.set_title(t, fontsize=14, color=INK, pad=6)
    ax.set_ylim(0.6, 1.03); ax.set_xlim(0.6, 5.4)
    ax.set_xticks(range(1, 6), ["1\n小", "2", "3", "4", "5\n大"])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0], ["60%", "70%", "80%", "90%", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_xlabel("X₁ の値で分けた組", fontsize=11, color=MUTED)
axes[0].set_ylabel("各組の当たる割合")
f.tight_layout(); f.savefig(f"{out}/e3_strata.png", dpi=220, facecolor="white")
