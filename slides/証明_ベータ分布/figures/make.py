# 補足スライド「被覆率がベータ分布に従う証明」（論文 3.5 節 sec:cond-coverage）専用の図と数式画像。
# 実行例：.venv/bin/python slides/証明_ベータ分布/figures/make.py slides/証明_ベータ分布/figures
# 数式は matplotlib の mathtext で PNG にする（スライド形式に数式の仕組みがないため）。
# 画像は 2 倍の解像度で作り、スライドでは半分の大きさで表示する（表示サイズを最後に出力する）。
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

out = sys.argv[1]
INK, BLUE, ORANGE, LIGHT = "#1F2A44", "#1F5A9D", "#A34F00", "#A9C1E0"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Hiragino Sans", "Noto Sans CJK JP", "DejaVu Sans"],
    "mathtext.fontset": "cm", "axes.edgecolor": "#8A93A3", "xtick.color": INK, "ytick.color": INK,
    "svg.hashsalt": "0", "pdf.compression": 9})
META = {"Software": None}  # PNG に作成ソフトの情報を埋め込まない（同じ入力なら同じバイト列にする）

# 数式（名前 → mathtext）。本文 40px の文字より少し大きめに表示される
FORMULAS = {
    "f_cov": r"$\mathrm{Cov}\;=\;P\left(\,|Y-\hat{f}(X)|\leq\hat{q}\,\right)\;=\;F(\hat{q})$",
    "f_thm": r"$\mathrm{Cov}\;\sim\;\mathrm{Beta}(k,\;n+1-k),\qquad k=\lceil (n+1)(1-\alpha)\rceil$",
    "f_pit": r"$P\left(F(S)\leq u\right)\;=\;u\qquad(0\leq u\leq 1)$",
    "f_order": r"$\mathrm{Cov}\;=\;F(S_{(k)})\;=\;V_{(k)}$",
    "f_equiv": r"$V_{(k)}\leq u\quad\Longleftrightarrow\quad\#\{\,i:\ V_i\leq u\,\}\;\geq\;k$",
    "f_binom": r"$P\left(V_{(k)}\leq u\right)\;=\;\sum_{j=k}^{n}\binom{n}{j}\,u^{j}(1-u)^{n-j}$",
    "f_density": r"$\frac{d}{du}P\left(V_{(k)}\leq u\right)\;=\;\frac{n!}{(k-1)!\,(n-k)!}\,u^{k-1}(1-u)^{n-k}$",
    "f_moments": r"$E[\mathrm{Cov}]=\frac{k}{n+1},\qquad \mathrm{Var}[\mathrm{Cov}]=\frac{k(n+1-k)}{(n+1)^{2}(n+2)}$",
    "f_ex": r"$n=20,\ k=19:\quad u^{18}(1-u)^{1}$",
}
for name, tex in FORMULAS.items():
    f = plt.figure()
    f.text(0, 0, tex, fontsize=28, color=INK)
    f.savefig(f"{out}/{name}.png", dpi=240, transparent=True, bbox_inches="tight", pad_inches=0.06, metadata=META)
    plt.close(f)


def clean(ax):
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_yticks([])


# ステップ 1：残差 S の分布（偏っている）と、F(S) の分布（0〜1 に一様）
rng = np.random.default_rng(0)
s = np.abs(rng.normal(size=20000))  # 残差の例：|標準正規|（半正規分布）
from scipy import stats
u = 2 * stats.norm.cdf(s) - 1      # 半正規分布の分布関数 F に通す
f, axes = plt.subplots(1, 2, figsize=(6, 2.9))
axes[0].hist(s, bins=40, density=True, color=LIGHT, edgecolor="white", linewidth=0.5)
axes[0].set_title("残差 S：偏っている", fontsize=20, color=INK, pad=8)
axes[0].set_xticks([0, 1, 2, 3]); axes[0].tick_params(labelsize=20)
axes[1].hist(u, bins=20, density=True, color=LIGHT, edgecolor="white", linewidth=0.5)
axes[1].axhline(1, color=BLUE, lw=3)
axes[1].set_title("F(S)：平ら", fontsize=20, color=INK, pad=8)
axes[1].set_xticks([0, 0.5, 1]); axes[1].set_xticklabels(["0%", "50%", "100%"]); axes[1].tick_params(labelsize=20)
for ax in axes:
    clean(ax)
f.tight_layout(w_pad=3)
f.savefig(f"{out}/step1_uniform.png", dpi=220, facecolor="white", metadata=META)
plt.close(f)

# ステップ 2・3：0〜1 に投げた 20 本のダーツ（較正用 20 点）。左から 19 本目が Cov
v = np.sort(np.random.default_rng(15).uniform(size=20))
f, ax = plt.subplots(figsize=(6, 1.8))
ax.hlines(0, 0, 1, color="#8A93A3", lw=2)
ax.scatter(v[:18], np.zeros(18), s=110, color=BLUE, zorder=3)
ax.scatter(v[19:], np.zeros(1), s=110, color=BLUE, zorder=3)
ax.scatter(v[18], 0, s=260, color=ORANGE, marker="D", zorder=4)
ax.annotate(f"19 本目 = Cov（{v[18]*100:.0f}%）", (v[18], 0), xytext=(v[18] + 0.03, 0.62), ha="right", fontsize=20, color=ORANGE,
    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.35, 1.0)
ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0%", "50%", "100%"]); ax.tick_params(labelsize=20, length=0)
for sp in ("top", "right", "left", "bottom"):
    ax.spines[sp].set_visible(False)
ax.set_yticks([])
f.tight_layout()
f.savefig(f"{out}/darts.png", dpi=220, facecolor="white", metadata=META)
plt.close(f)

for name in list(FORMULAS) + ["step1_uniform", "darts"]:
    w, h = Image.open(f"{out}/{name}.png").size
    print(f"{name}: 表示 {w // 2}x{h // 2}px（画像 {w}x{h}）")
