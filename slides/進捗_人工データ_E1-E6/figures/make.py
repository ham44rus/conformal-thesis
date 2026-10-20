# スライド専用の図（論文の図ではない）。results/ の csv から作る。
# 投影で読めるよう、図の中の文字は表示時に約 36px 以上になる大きさにしている。
import sys, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

root, out = sys.argv[1], sys.argv[2]
INK, MUTED, GRID = "#1F2A44", "#4A5568", "#DDE2E9"
COL = {"abs": "#4A5568", "norm": "#1F5A9D", "cqr": "#A34F00"}
REF = "#8A93A3"  # 比較用（絶対残差の再掲）
MK = {"abs": "o", "norm": "s", "cqr": "^"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Hiragino Sans", "Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 20, "axes.edgecolor": "#8A93A3", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 20, "ytick.labelsize": 20})

def pct(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

# 実験1：被覆率のばらつきと理論（Beta）。alpha=0.10、較正用 20 / 2000 点
e1 = pd.read_csv(f"{root}/results/e1_coverage.csv")
f, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, n, ticks in zip(axes, [20, 2000], [[0.7, 0.8, 0.9, 1.0], [0.88, 0.90, 0.92]]):
    c = e1[(e1.n_cal == n) & (np.isclose(e1.alpha, 0.10))].coverage.to_numpy()
    k = math.ceil((n + 1) * 0.9)
    ax.hist(c, bins=28, density=True, color="#A9C1E0", edgecolor="white", linewidth=0.6)
    x = np.linspace(c.min(), c.max(), 400)
    ax.plot(x, stats.beta.pdf(x, k, n + 1 - k), color="#1F5A9D", lw=3)
    ax.axvline(0.9, color=INK, lw=1.6, ls=(0, (4, 3)))
    ax.set_title(f"較正用 {n:,} 点", fontsize=22, color=INK, pad=8)
    ax.set_yticks([]); ax.set_xticks(ticks); pct(ax)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
f.tight_layout(w_pad=2.5); f.savefig(f"{out}/e1_beta.png", dpi=220, facecolor="white")

# 実験3：組ごとの当たる割合（不均一分散データ、X1 で 5 組）
e3 = pd.read_csv(f"{root}/results/e3_summary.csv")
f, axes = plt.subplots(1, 3, figsize=(8, 3.6), sharey=True)
for ax, methods, t in zip(axes, [["abs"], ["abs", "norm"], ["abs", "cqr"]], ["絶対残差", "正規化残差", "CQR"]):
    ax.axhline(0.9, color=INK, lw=1.6, ls=(0, (4, 3)))
    for m in methods:
        y = e3[e3.method == m].sort_values("bin").coverage_mean.to_numpy()
        ref = (m == "abs" and len(methods) > 1)
        ax.plot(range(1, 6), y, color=REF if ref else COL[m], lw=2 if ref else 3.2, marker=MK[m], ms=7 if ref else 9,
                markeredgecolor="white", markeredgewidth=1.2)
    ax.set_title(t, fontsize=22, color=INK, pad=8)
    ax.set_ylim(0.6, 1.03); ax.set_xlim(0.6, 5.4)
    ax.set_xticks([1, 3, 5], ["1", "3", "5"])
    ax.set_yticks([0.6, 0.8, 0.9, 1.0], ["60%", "80%", "90%", "100%"])
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
f.tight_layout(rect=(0, 0.09, 1, 1))
f.text(0.55, 0.03, "X1 の値で分けた組（右ほど誤差が大きい）", ha="center", fontsize=20, color=INK); f.savefig(f"{out}/e3_strata.png", dpi=220, facecolor="white")

# 実験4：入力のずれのイメージ（データ②の残差。左＝手元の較正用、右＝ずれ「大」の予測したいデータ）
# 帯は較正用で決めた半幅（区間の幅の半分）、題の「帯の外」は実験の値（results/e4_summary.csv）。点は説明用の乱数 300 個
rng = np.random.default_rng(0)
e4 = pd.read_csv(f"{root}/results/e4_summary.csv").query("method == 'unweighted'").set_index("gamma")
q = e4.loc[0.0, "mean_width"] / 2
f, axes = plt.subplots(1, 2, figsize=(8, 3.9), sharey=True)
for ax, gamma, col, t in zip(axes, [0, 4], [COL["norm"], COL["cqr"]], ["手元（較正用）", "予測したいデータ（ずれ大）"]):
    x1 = rng.uniform(size=300) ** (1 / (1 + gamma))  # 密度 (1+γ) x^γ からの乱数
    e = (0.1 + x1) * rng.normal(size=300)
    out_ = np.abs(e) > q
    ax.axhspan(-q, q, color="#DDE2E9", zorder=0)
    ax.scatter(x1[~out_], e[~out_], s=14, color=col, alpha=0.8, linewidths=0)
    ax.scatter(x1[out_], e[out_], s=30, color=INK, marker="x", linewidths=1.6)
    ax.set_title(f"{t}\n帯の外：約 {(1 - e4.loc[float(gamma), 'coverage_mean'])*100:.0f}%", fontsize=20, color=INK, pad=8, linespacing=1.3)
    ax.set_xlim(0, 1); ax.set_ylim(-3.2, 3.2); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([-2, 0, 2])
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].set_ylabel("残差", fontsize=20, color=INK)
f.tight_layout(rect=(0, 0.08, 1, 1))
f.text(0.55, 0.025, "横軸 X1（右ほどぶれ大）。灰色の帯＝手元で決めた幅", ha="center", fontsize=18, color=INK)
f.savefig(f"{out}/e4_shift.png", dpi=220, facecolor="white")
