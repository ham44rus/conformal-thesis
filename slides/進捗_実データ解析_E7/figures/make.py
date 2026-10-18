# スライド専用の図（論文の図ではない）。results/e7_summary.csv から作る。
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root, out = sys.argv[1], sys.argv[2]
s = pd.read_csv(f"{root}/results/e7_summary.csv")
INK, MUTED, GRID = "#1F2A44", "#6B7486", "#E3E7ED"
COL = {"abs": "#6B7486", "norm": "#2F6DB5", "cqr": "#C46F1A"}
NAME = {"abs": "絶対残差", "norm": "正規化残差", "cqr": "CQR"}
MK = {"abs": "o", "norm": "s", "cqr": "^"}
DS = [("superconductivity", "Superconductivity"), ("bike", "Bike Sharing"), ("airfoil", "Airfoil")]
plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Hiragino Sans", "Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 13, "axes.edgecolor": "#C9CFD6", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "svg.hashsalt": "x"})

def panel(ax, key, title, methods):
    ax.axhline(0.9, color=INK, lw=1.2, ls=(0, (4, 3)))
    for m in methods:
        r = s[(s.dataset == key) & (s.method == m)].iloc[0]
        y = [r[f"cov_bin{i}"] for i in range(1, 6)]
        faded = (m == "abs" and len(methods) > 1)
        ax.plot(range(1, 6), y, color=COL[m], lw=1.6 if faded else 2.4, marker=MK[m], ms=7,
                alpha=0.45 if faded else 1, markeredgecolor="white", markeredgewidth=1.2)
    ax.set_title(title, fontsize=14, color=INK, pad=8)
    ax.set_ylim(0.6, 1.03); ax.set_xlim(0.6, 5.4)
    ax.set_xticks(range(1, 6), ["1\n小", "2", "3", "4", "5\n大"])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0], ["60%", "70%", "80%", "90%", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

def fig(methods, path, legend):
    f, axes = plt.subplots(1, 3, figsize=(8, 3.3), sharey=True)
    for ax, (k, t) in zip(axes, DS):
        panel(ax, k, t, methods)
    axes[0].set_ylabel("各組の当たる割合")
    for ax in axes: ax.set_xlabel("予測値の大きさで分けた組", fontsize=11, color=MUTED)
    if legend:
        hs = [plt.Line2D([], [], color=COL[m], marker=MK[m], lw=2.4, ms=7, alpha=0.45 if m == "abs" else 1) for m in methods]
        f.legend(hs, [NAME[m] for m in methods], loc="upper center", ncol=len(methods), frameon=False, fontsize=13,
                 bbox_to_anchor=(0.5, 1.04))
    f.tight_layout(rect=(0, 0, 1, 0.93 if legend else 1))
    f.savefig(path, dpi=220, facecolor="white")

fig(["abs"], f"{out}/strata_abs.png", False)
fig(["abs", "norm"], f"{out}/strata_norm.png", True)
fig(["abs", "cqr"], f"{out}/strata_cqr.png", True)
