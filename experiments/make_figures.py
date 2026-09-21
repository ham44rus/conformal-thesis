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

from conformal import conformal_index

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
        # PDF は既定で生成時刻を埋め込むので、同じ csv から作り直しても毎回
        # git の差分に出てしまう（実際に差分は CreationDate の2バイトだけだった）。
        # 図は提出物として git 管理するため時刻を落とし、内容が同じなら
        # バイト単位で一致するようにする。PNG には元から時刻が入らない。
        meta = {"metadata": {"CreationDate": None}} if ext == "pdf" else {}
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight", **meta)
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
        l = n_cal + 1 - conformal_index(n_cal, alpha)  # floor((n+1)*alpha) を厳密に
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

# 学習集合間の区間に使う t(4) の両側5%点。rep が5通りなので自由度は 4
# （experiments/e2_model_agnostic.py の T_CRIT_4 と同じ値。片方だけ変えないこと）
T_CRIT_4 = 2.776

# 表示用の和名（csv には英字キーが入っている）
E2_LABELS = {
    "constant": "定数予測器",
    "ridge": "リッジ回帰",
    "rf": "ランダム\nフォレスト",
    "gbm": "勾配\nブースティング",
    "mlp": "ニューラル\nネット",
}


def fig_e1b_ecdf(df: pd.DataFrame) -> None:
    """E1b: 3 ケースの被覆率の経験分布関数と理論 Beta(901, 100) の分布関数の重ね描き。

    仕様書 specs/E1b_detection_levels.md「図」に対応する。
    誤実装は曲線全体が左にずれ（位置のずれ）、テスト集合が小さいケースは中央で交差して
    両裾が外に出る（形のずれ）。正しい実装は理論に重なる。
    乱数は使わず csv から決定的に描く。
    """
    n_cal, alpha = 1000, 0.10  # experiments/e1b_detection_levels.py と同じ
    l = n_cal + 1 - conformal_index(n_cal, alpha)
    a_par, b_par = n_cal + 1 - l, l
    lower, upper = 1 - alpha, 1 - alpha + 1 / (n_cal + 1)

    cases = [
        ("correct", "正しい実装", BLUE, "-"),
        ("offbyone", r"誤実装（$n+1$ を $n$ に）", "#7A4E9E", "-"),
        ("small_test", "テスト集合が小さい（2,000 点）", "#D28C1A", "-"),
    ]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    xs = np.linspace(0.86, 0.94, 800)
    ax.plot(xs, stats.beta.cdf(xs, a_par, b_par), color=RED, linewidth=2.4,
            label=f"理論 Beta({a_par}, {b_par})", zorder=1)
    for key, label, color, ls in cases:
        cov = np.sort(df[df.case == key]["coverage"].to_numpy())
        if cov.size == 0:
            continue
        ecdf = np.arange(1, cov.size + 1) / cov.size
        ax.step(cov, ecdf, where="post", color=color, linewidth=1.1, linestyle=ls,
                label=label, zorder=2)

    ax.axvline(lower, color=INK, linewidth=1, linestyle=(0, (4, 3)), label=r"下界 $1-\alpha$")
    ax.axvline(upper, color=INK, linewidth=1, linestyle=(0, (1, 2)), label=r"上界 $1-\alpha+1/(n+1)$")
    ax.set_xlim(0.865, 0.935)
    ax.set_ylim(0, 1)
    ax.set_xlabel("被覆率")
    ax.set_ylabel("累積確率")
    ax.grid(axis="both")
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title(rf"E1b：被覆率の経験分布関数（$n$ = {n_cal}, $\alpha$ = {alpha}, 2000 試行）",
                 fontsize=9.5, color=INK)
    save(fig, "fig_e1b_ecdf")


def fig_e2_model_agnostic(summary: pd.DataFrame,
                          robust: pd.DataFrame | None = None) -> None:
    """図5.3: 被覆はモデルによらず、区間幅だけがモデルで変わることを示す2段組。

    上段が平ら・下段が階段状に見えることがこの図の主張（仕様書「図」の節）。

    上段の縦軸は理論の上下界の幅 1/(n+1) を単位にして取る。データの範囲に
    合わせて自動で詰めるとモデル間のわずかな差が画面いっぱいに拡大され、
    「被覆はモデルによらない」という主張と逆の印象を与えるため。

    上段の誤差棒は `coverage_se`（= 試行間成分とテスト集合間成分の合成 ±1.96 SE）。
    E2 は全試行・全モデルで同じテスト集合を使い回すので、その集合固有のズレは
    試行数では薄まらない。どの成分を含めたかは凡例に明記する
    （CLAUDE.md「報告の作法」）。

    下段の誤差棒は**二重**にする。内側が試行間（`width_se`、較正集合の引き直し）、
    外側が学習集合間（`e2_robustness.csv` の rep 間 SD、t(4) の 2.776 による95%）。

    同じ「平均区間幅」という量に対して、何に条件づけるかで不確実性の大きさが
    変わることを一目で見せるのがこの図の狙い。内側だけを見ると非線形3モデルは
    綺麗に分離しているように見えるが、外側まで含めると重なる。C2 と C4 が
    別の問いに答えているのはこのためである（仕様書 C2 の注意書き）。
    """
    g = summary.set_index("model").loc[list(E2_MODEL_ORDER)]
    x = np.arange(len(g))

    lower = 1 - E2_ALPHA                 # 名目値
    band = 1 / (E2_N_CAL + 1)            # 理論の上下界の幅（卒論 系 cor:upper、式 eq:two-sided）
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
    w = g["width_mean"].to_numpy()
    w_trial = 1.96 * g["width_se"].to_numpy()

    # 外側：学習集合間。rep が5通りなので t(4) の両側5%点を使う
    w_train = None
    if robust is not None:
        r = robust.groupby("model")["width_mean"].agg(["std", "count"])
        r = r.loc[list(E2_MODEL_ORDER)]
        w_train = (T_CRIT_4 * r["std"] / np.sqrt(r["count"])).to_numpy()
        # 色相は増やさず、同じ BLUE の淡い太線で外側を描く
        ax_w.errorbar(x, w, yerr=w_train, fmt="none", ecolor=BLUE, alpha=0.30,
                      elinewidth=6, capsize=0,
                      label=f"学習集合間 95%（rep 間 SD, t(4)）")

    ax_w.errorbar(x, w, yerr=w_trial, fmt="o-", color=BLUE, markersize=5,
                  linewidth=1.6, capsize=3, elinewidth=1.4,
                  label="試行間 95%（較正集合の引き直し）")
    ax_w.legend(loc="upper right", fontsize=7)

    # 縦軸は外側の誤差棒まで入るように取る
    lo = (w - w_train).min() if w_train is not None else w.min()
    hi = (w + w_train).max() if w_train is not None else w.max()
    span = hi - lo
    ax_w.set_ylim(lo - 0.12 * span, hi + 0.22 * span)
    ax_w.set_ylabel(r"平均区間幅 $2\hat{q}$")
    ax_w.set_xticks(x, [E2_LABELS[k] for k in E2_MODEL_ORDER], fontsize=8)
    ax_w.grid(axis="y")
    ax_w.set_axisbelow(True)

    fig.suptitle(
        rf"被覆はモデルによらず、幅だけが変わる（$\alpha$ = {E2_ALPHA}, $n$ = {E2_N_CAL}）",
        y=0.96, fontsize=11, color=INK,
    )
    save(fig, "fig_e2_model_agnostic")


def fig_e2_tier1_zoom(summary: pd.DataFrame, tiers: pd.DataFrame) -> None:
    """図5.4: 第1層（非線形3モデル）だけを拡大し、内側と外側の誤差棒を比べる。

    fig_e2_model_agnostic の下段は縦軸が 1.7〜3.0 あるため、第1層内の 0.03 程度の
    差と、そこに付く誤差棒がほとんど見えない。この図は第1層だけを専用スケールで
    描き、**内側（試行間）だけ見ると3モデルは分離しているが、外側（学習集合間）
    まで含めると重なる**ことを示す。

    C2 は内側で「この学習集合のもとでは幅が違う」と言い、C4 は外側で
    「学習集合を超えては言えない」と言う。両者が別の問いに答えていることの図示。

    誤差棒の中心は主分析（学習集合1つ・1000試行）の平均。外側の半幅だけを
    頑健性の確認（学習集合5通り）の rep 間 SD から作っている点に注意
    （fig_e2_model_agnostic の下段と同じ作法）。
    """
    keys = [k for k in E2_MODEL_ORDER if k in set(tiers[tiers.tier == 1]["model"])]
    s = summary.set_index("model").loc[keys]
    t = tiers.set_index("model").loc[keys]
    x = np.arange(len(keys))

    m = s["width_mean"].to_numpy()
    inner = 1.96 * s["width_se"].to_numpy()
    outer = T_CRIT_4 * t["width_rep_se"].to_numpy()

    fig, ax = plt.subplots(figsize=(3.2, 2.6))

    ax.errorbar(x, m, yerr=outer, fmt="none", ecolor=BLUE, alpha=0.30,
                elinewidth=9, capsize=0, label="学習集合間 95%")
    ax.errorbar(x, m, yerr=inner, fmt="o", color=BLUE, markersize=5,
                capsize=3, elinewidth=1.4, label="試行間 95%")

    lo, hi = (m - outer).min(), (m + outer).max()
    span = hi - lo
    ax.set_ylim(lo - 0.10 * span, hi + 0.26 * span)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_xticks(x, [E2_LABELS[k] for k in keys], fontsize=8)  # 和名は2行のまま使う
    ax.set_ylabel(r"平均区間幅 $2\hat{q}$")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("第1層の拡大：内側は分離、外側は重なる", fontsize=9, color=INK)
    save(fig, "fig_e2_tier1_zoom")


# --- E3: 異分散と条件付き被覆の破綻（仕様書 specs/E3_conditional_coverage.md「図」）---

# alpha は csv に列として持っていないので、仕様書「規模のパラメータ」の値を定数で置く。
# experiments/e3_conditional.py の ALPHA と同じ値。片方だけ変えないこと。
E3_ALPHA = 0.10

# 手法の順。experiments/e3_conditional.py の METHOD_KEYS と同じ順に固定する
E3_METHOD_ORDER = ("abs", "norm", "cqr")
E3_LABELS = {"abs": "絶対残差", "norm": "正規化残差", "cqr": "CQR"}

# 3手法を塗り分けるための3色目。BLUE / RED に PURPLE を足した3色は
# colorblind-safe の検証（明度帯・彩度下限・CVD分離・通常色覚分離・地との
# コントラスト）をすべて通る。BLUE と PURPLE の2色だけでは通常色覚での分離が
# 足りず（ΔE 12.1 < 15）、RED を含む3色が唯一の通過解だった。
PURPLE = "#7A5C9E"
E3_COLORS = {"abs": BLUE, "norm": RED, "cqr": PURPLE}


def fig_e3_band(band: pd.DataFrame) -> None:
    """図5.5: 予測区間の帯。定数幅（絶対残差）と適応幅（CQR）の対比。

    本論文の顔になる図（仕様書「図1」）。絶対残差は X1 によらず幅が一定、
    CQR は X1 とともに広がる。この対比だけを狙い、正規化残差は載せない
    （図5.6 に出る）。

    **縦軸は Y そのものではなく Y から区間の中心を引いた値にしている。**
    仕様書は縦軸 Y と書いているが、本 DGP は Y = X1 + sin(2*pi*X2) + sigma(X1)*eps
    であり、区間の中心は X2 にも依存する。実測でも中心の標準偏差は
    どの X1 帯でも 0.6〜0.8 あり、(X1, Y) 平面では区間が帯にならず波形に潰れる。
    中心を引くと X2 由来の変動が落ち、幅の X1 依存だけが残る。

    この変換で**被覆の読み取りは保たれる**。点が帯の内側にあることと
    lo <= y <= hi は同値なので、帯からはみ出す点の割合はそのまま非被覆率になる。

    残った幅のばらつき（CQR で X1 帯内 SD 約 0.25）は、分位点回帰が X1 以外の
    共変量も拾っているため。全体レンジ 2.34 の約10%なので帯の傾向は読める。

    キャプション（TeX にそのまま貼る。**この2文は省略しないこと**）:

        縦軸は Y から予測区間の中心を引いた値である。区間の中心を引いて
        X_2 由来の変動を除いた。点が帯の内側にあることと被覆は同値なので、
        はみ出す点の割合はそのまま非被覆率である。
    """
    keys = ("abs", "cqr")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True,
                             gridspec_kw={"wspace": 0.08})

    work = {}
    for key in keys:
        g = band[band["method"] == key].sort_values("x1").copy()
        center = (g["lo"] + g["hi"]) / 2.0
        work[key] = (g["x1"].to_numpy(),
                     (g["lo"] - center).to_numpy(),
                     (g["hi"] - center).to_numpy(),
                     (g["y"] - center).to_numpy())

    # 2枚で共通の縦軸レンジ（揃えないと広がり方の違いが消える）
    y_lo = min(min(lo.min(), yy.min()) for _, lo, _, yy in work.values())
    y_hi = max(max(hi.max(), yy.max()) for _, _, hi, yy in work.values())
    pad = 0.05 * (y_hi - y_lo)

    for ax, key in zip(axes, keys):
        x, lo, hi, yy = work[key]
        ax.fill_between(x, lo, hi, color=E3_COLORS[key], alpha=0.30, linewidth=0,
                        label="予測区間")
        ax.plot(x, lo, color=E3_COLORS[key], linewidth=0.9)
        ax.plot(x, hi, color=E3_COLORS[key], linewidth=0.9)
        ax.scatter(x, yy, s=2.5, color=INK, alpha=0.30, linewidths=0,
                   label="テスト点", zorder=3)

        w = hi - lo
        note = f"幅 {w.min():.2f}（一定）" if w.ptp() < 1e-9 else f"幅 {w.min():.2f} → {w.max():.2f}"
        ax.set_title(f"{E3_LABELS[key]}　{note}", fontsize=9.5, color=INK)
        ax.set_xlabel("$X_1$")
        ax.set_xlim(0, 1)
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    axes[0].set_ylim(y_lo - pad, y_hi + pad)
    axes[0].set_ylabel("$Y$ − 区間の中心")
    axes[0].legend(loc="lower left", fontsize=7.5, markerscale=3)
    fig.suptitle(r"予測区間の形（$\alpha$ = %.2f, 1試行分）" % E3_ALPHA,
                 y=1.02, fontsize=11, color=INK)
    save(fig, "fig_e3_band")


def fig_e3_stratified(summary: pd.DataFrame) -> None:
    """図5.6: X1 の5分位で層別した被覆率。周辺は満たすが条件付きは割れる。

    誤差棒は**試行間＋テスト集合間の2成分合成**（CLAUDE.md「報告の作法」）。
    層別ではビン内の点数が 1/5 になるので se_test が √5 倍効き、
    この図ではテスト集合成分の方が支配的である。

    絶対残差は右肩下がりに大きく割れ、正規化残差と CQR は名目線の近くに寄る。
    ただし周辺較正なので完全には平らにならない（正規化が不完全なぶん残る）。そこが第5章の論点になる。
    """
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    n_bins = int(summary["bin"].max()) + 1
    x = np.arange(n_bins)
    width = 0.26

    for i, key in enumerate(E3_METHOD_ORDER):
        g = summary[summary["method"] == key].sort_values("bin")
        se = np.sqrt(g["se_trial"] ** 2 + g["se_test"] ** 2)
        gap = float(g["coverage_mean"].max() - g["coverage_mean"].min())
        ax.bar(x + (i - 1) * width, g["coverage_mean"], width * 0.92,
               color=E3_COLORS[key], alpha=0.88, linewidth=0,
               label=f"{E3_LABELS[key]}（gap {gap:.3f}）")
        ax.errorbar(x + (i - 1) * width, g["coverage_mean"], yerr=1.96 * se,
                    fmt="none", ecolor=INK, elinewidth=0.9, capsize=2, alpha=0.75)

    ax.axhline(1 - E3_ALPHA, color=INK, linewidth=1.1, linestyle=(0, (4, 3)),
               zorder=4, label=r"名目値 $1-\alpha$")

    # 縦軸は名目値をはさんで、割れの大きさがそのまま高さに出る範囲にする
    ax.set_ylim(0.70, 1.01)
    ax.set_xticks(x, [f"層{b}\n$X_1$ {0.2 * b:.1f}–{0.2 * (b + 1):.1f}" for b in x],
                  fontsize=8)
    ax.set_ylabel("被覆率")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    # 層3・層4 の棒は 0.90 を超えないので、右上が空く
    ax.legend(loc="upper right", fontsize=7.5, ncol=1)
    fig.suptitle(r"$X_1$ で層別した被覆率（誤差棒は試行間＋テスト集合間 95%）",
                 y=0.99, fontsize=11, color=INK)
    save(fig, "fig_e3_stratified")


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

    # E1b は独立に回すので、csv が無ければ飛ばす
    e1b_path = RESULTS / "e1b_detection_levels.csv"
    if e1b_path.exists():
        fig_e1b_ecdf(pd.read_csv(e1b_path))
    else:
        print("  results/e1b_detection_levels.csv が無いため E1b の図は省略（先に `make e1b`）")

    # E2 は独立に回すので、csv が無ければ飛ばす（E1 の図だけは常に出せるように）
    e2_path, e2_rob = RESULTS / "e2_summary.csv", RESULTS / "e2_robustness.csv"
    if e2_path.exists():
        rob = pd.read_csv(e2_rob) if e2_rob.exists() else None
        e2_sum = pd.read_csv(e2_path)
        fig_e2_model_agnostic(e2_sum, rob)

        # 第1層の拡大図。C4 の層構造の解析が済んでいるときだけ作る
        e2_tiers = RESULTS / "e2_tiers.csv"
        if e2_tiers.exists():
            fig_e2_tier1_zoom(e2_sum, pd.read_csv(e2_tiers))
    else:
        print("  results/e2_summary.csv が無いため E2 の図は省略（先に `make e2`）")

    # E3 も独立に回すので、csv が無ければ飛ばす
    e3_band, e3_sum = RESULTS / "e3_band_trial0.csv", RESULTS / "e3_summary.csv"
    if e3_band.exists() and e3_sum.exists():
        fig_e3_band(pd.read_csv(e3_band))
        fig_e3_stratified(pd.read_csv(e3_sum))
    else:
        print("  results/e3_*.csv が無いため E3 の図は省略（先に `make e3`）")


if __name__ == "__main__":
    main()
