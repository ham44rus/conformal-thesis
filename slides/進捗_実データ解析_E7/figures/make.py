# スライド専用の図（論文の図ではない）。results/e7_summary.csv から作る。
# 投影で読めるよう、図の中の文字は表示時に約 36px 以上になる大きさにしている。
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root, out = sys.argv[1], sys.argv[2]
s = pd.read_csv(f"{root}/results/e7_summary.csv")
INK, GRID = "#1F2A44", "#DDE2E9"
COL = {"abs": "#4A5568", "norm": "#1F5A9D", "cqr": "#A34F00"}
REF = "#8A93A3"  # 比較用（絶対残差の再掲）
NAME = {"abs": "絶対残差", "norm": "正規化残差", "cqr": "CQR"}
MK = {"abs": "o", "norm": "s", "cqr": "^"}
DS = [("superconductivity", "Supercon."), ("bike", "Bike"), ("airfoil", "Airfoil")]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Hiragino Sans", "Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 20, "axes.edgecolor": "#8A93A3", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 20, "ytick.labelsize": 20})

def fig(methods, path):
    f, axes = plt.subplots(1, 3, figsize=(8, 3.6), sharey=True)
    for ax, (k, t) in zip(axes, DS):
        ax.axhline(0.9, color=INK, lw=1.6, ls=(0, (4, 3)))
        for m in methods:
            r = s[(s.dataset == k) & (s.method == m)].iloc[0]
            y = [r[f"cov_bin{i}"] for i in range(1, 6)]
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
    f.text(0.55, 0.03, "予測値の大きさで分けた組（右ほど大きい）", ha="center", fontsize=20, color=INK)
    f.savefig(path, dpi=220, facecolor="white")

fig(["abs"], f"{out}/strata_abs.png")
fig(["abs", "norm"], f"{out}/strata_norm.png")
fig(["abs", "cqr"], f"{out}/strata_cqr.png")
