"""実験E3: 異分散データと条件付き被覆の破綻。

対応箇所:
    卒論 第5章 E3 の節（sec:e3。本論文の見せ場）
    仕様書 specs/E3_conditional_coverage.md

周辺被覆が 1-alpha を満たしていても、部分集団ごとの被覆は大きく割れる。
その割れを正規化残差スコアと CQR がどこまで均せるかを測る。

Barber et al. (2021) の不可能性定理により、分布に仮定を置かない限り条件付き被覆は
原理的に達成できない。したがって (b)(c) でも層別被覆は平らにならない。
この実験が測るのは「ゼロになるか」ではなく「どこまで縮むか」である。

実行:
    python experiments/e3_conditional.py --seed 20260901
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

from conformal import (
    absolute_residual,
    conformal_quantile,
    cqr_score,
    feature_stratified_coverage,
    fit_quantile_pair,
    fit_sigma,
    normalized_residual,
    size_stratified_coverage,
)
from conformal.datasets import heteroscedastic

# --- 規模のパラメータ（仕様書「規模のパラメータ」）---
N_TRAIN, N_CAL, N_TEST = 2_000, 1_000, 200_000
N_TRIAL, ALPHA = 500, 0.10
N_BINS = 5

# 層別のビン境界。X1 ~ Uniform(0,1) なので真の5分位が既知（仕様書「層別のビン境界」）。
# 固定境界にすると試行間でビンが動かず、層別被覆の試行間ばらつきが
# 「被覆のばらつき」だけになる。実データでは使えない手であることを第6章に書く。
BIN_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, np.inf])

# 正規化残差の分母に足す eps。src/conformal/scores.py: normalized_residual の
# 既定値と**必ず一致させる**こと（仕様書「正規化残差の区間を組むときの規約」）。
# スコアが (sigma + eps) で割っているので、区間も (sigma + eps) 倍で組む。
EPS_SCALE = 1e-8

# 手法。この順序は変えないこと（表示順と、乱数の導出順に対応する）
METHOD_KEYS = ("abs", "norm", "cqr")
LABELS = {
    "abs": "絶対残差",
    "norm": "正規化残差",
    "cqr": "CQR",
}

# 内部乱数を持つモデル。**この順序は変えないこと。**
# random_state をこの順で rng から引くため、順序を変えると同じ seed でも結果が変わる
# （仕様書「乱数の導出」・CLAUDE.md 再現性ルール）。
MODEL_SEED_KEYS = ("f_hat", "sigma", "cqr_lo", "cqr_hi")

RESULTS = Path(__file__).resolve().parents[1] / "results"


def build_quantile_factory():
    """CQR の分位点回帰を作る関数を返す（仕様書「モデルの設定」）。

    ハイパーパラメータは scikit-learn の既定値
    （n_estimators=100, max_depth=3, learning_rate=0.1）。
    チューニングの巧拙が結論に混ざるのを避けるため、あえて既定値のままにする。
    """

    def factory(q: float, seed: int | None) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(loss="quantile", alpha=q, random_state=seed)

    return factory


def fit_all(
    X_tr: np.ndarray, y_tr: np.ndarray, model_seeds: dict[str, int]
) -> tuple[object, object, object]:
    """3手法の下敷きを学習集合から作る（仕様書「下敷きモデルの共有関係」）。

    (a) と (b) は**同じ f_hat を共有する**。差が sigma の有無だけに帰属するので、
    「異分散への適応が効いたのか、モデルが違うだけなのか」という反論を封じられる。

    すべての手法が同じ学習集合全体を使う（(b) だけ多くのデータを使う不公平を作らない）。

    out-of-fold 分割の乱数には model_seeds["sigma"] を使う。fold の切り方は
    sigma 推定の一部なので、sigma 用の seed から取るのが自然であり、
    仕様書「乱数の導出」の引き順（学習集合より前に model_seeds を引く）も崩さない。

    Returns
    -------
    (f_hat, sigma_est, cqr_pair)
    """
    # (a)(b): 平均回帰 f_hat と sigma(x)。out-of-fold 残差の絶対値を回帰する
    f_hat, sigma_est = fit_sigma(
        GradientBoostingRegressor(random_state=model_seeds["f_hat"]),
        GradientBoostingRegressor(random_state=model_seeds["sigma"]),
        X_tr,
        y_tr,
        n_splits=5,
        random_state=model_seeds["sigma"],
        floor=None,           # out-of-fold 残差の5%分位点を下限に使う
        target="abs",         # 目的変数は |残差|
    )

    # (c): 分位点回帰を alpha/2 と 1-alpha/2 で別々に学習する
    cqr_pair = fit_quantile_pair(
        build_quantile_factory(),
        X_tr,
        y_tr,
        q_lo=ALPHA / 2,
        q_hi=1 - ALPHA / 2,
        random_state_lo=model_seeds["cqr_lo"],
        random_state_hi=model_seeds["cqr_hi"],
    )
    return f_hat, sigma_est, cqr_pair


def intervals(
    method: str,
    q_hat: float,
    pred: np.ndarray,
    scale: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """共形分位点から予測区間を組む。

    正規化残差の区間は f(x) ± q_hat * (sigma(x) + eps) とする。
    スコア側が (sigma + eps) で割っているので、分母の規約を揃えないと
    スコアと区間の対応が厳密には成り立たない（仕様書「実装の契約」）。
    """
    if method == "abs":
        return pred - q_hat, pred + q_hat
    if method == "norm":
        w = q_hat * (scale + EPS_SCALE)
        return pred - w, pred + w
    if method == "cqr":
        return q_lo - q_hat, q_hi + q_hat
    raise ValueError(f"未知の手法: {method}")


def run_trials(
    rng: np.random.Generator,
    f_hat,
    sigma_est,
    cqr_pair,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """試行ループ（仕様書「手続き」）。

    **試行ごとに引き直すのは較正集合のみ。** 学習集合とモデルは全試行で固定する。
    **3手法は同じ較正集合を共有する**（手法間の差からサンプリング変動を除くため）。

    モデルが固定なのでテスト集合側の予測は1回だけ計算して使い回す。

    Returns
    -------
    (層別の長い表, 試行ごとの周辺量, trial=0 の帯グラフ用の生データ)
    """
    # テスト集合側の予測はモデルが固定なので試行ループの外で一度だけ作る
    pred_te = f_hat.predict(X_te)
    sigma_te = sigma_est.predict(X_te)
    qlo_te, qhi_te = cqr_pair.predict(X_te)

    # 報告義務の2項目。モデルが固定なので試行によらず一定
    crossing_rate = cqr_pair.crossing_rate_on(X_te)
    sigma_clip_rate = sigma_est.clip_rate_on(X_te)

    strat_rows, marg_rows, band_rows = [], [], []

    for trial in range(N_TRIAL):
        # 較正集合だけを引き直す。3手法で共有する
        X_c, y_c = heteroscedastic(N_CAL, rng)
        pred_c = f_hat.predict(X_c)
        sigma_c = sigma_est.predict(X_c)
        qlo_c, qhi_c = cqr_pair.predict(X_c)

        q_hats = {
            "abs": conformal_quantile(absolute_residual(y_c, pred_c), ALPHA),
            "norm": conformal_quantile(normalized_residual(y_c, pred_c, sigma_c), ALPHA),
            "cqr": conformal_quantile(cqr_score(y_c, qlo_c, qhi_c), ALPHA),
        }

        for key in METHOD_KEYS:
            lo, hi = intervals(key, q_hats[key], pred_te, sigma_te, qlo_te, qhi_te)
            width = hi - lo

            # 層別被覆。境界は固定なので毎試行同じ層で切られる
            strat = feature_stratified_coverage(
                X_te[:, 0], y_te, lo, hi, edges=BIN_EDGES
            )
            for r in strat:
                m = (X_te[:, 0] >= BIN_EDGES[r["bin"]]) & (
                    X_te[:, 0] < BIN_EDGES[r["bin"] + 1]
                )
                strat_rows.append(
                    {
                        "method": key,
                        "trial": trial,
                        "bin": r["bin"],
                        "n_bin": r["n"],
                        "coverage": r["coverage"],
                        "mean_width": float(width[m].mean()),
                    }
                )

            # SSC の最悪層。abs は幅が定数なので層が1つに潰れる（仕様書「報告義務」）。
            # 潰れたことが分かるよう NaN を入れる。値 0 や周辺被覆を入れると
            # 「層別した結果」と読めてしまうため
            ssc = size_stratified_coverage(y_te, lo, hi, n_bins=N_BINS)
            ssc_worst = float(min(r["coverage"] for r in ssc)) if len(ssc) > 1 else np.nan

            marg_rows.append(
                {
                    "method": key,
                    "trial": trial,
                    "coverage": float(np.mean((y_te >= lo) & (y_te <= hi))),
                    "mean_width": float(width.mean()),
                    "q_hat": float(q_hats[key]),
                    "ssc_worst": ssc_worst,
                    # 該当しない手法には NaN。0.0 だと「起きなかった」と読めてしまう
                    "crossing_rate": crossing_rate if key == "cqr" else np.nan,
                    "sigma_clip_rate": sigma_clip_rate if key == "norm" else np.nan,
                }
            )

            if trial == 0:
                band_rows.append((key, lo, hi))

        if (trial + 1) % 100 == 0:
            print(f"  {trial + 1} / {N_TRIAL} 試行 完了")

    # 帯グラフ用に trial=0 から2,000点を無作為抽出する。
    # 仕様書「乱数の導出」のとおり、試行ループの後に同じ rng から引く
    idx = rng.choice(N_TEST, size=2_000, replace=False)
    band = pd.DataFrame(
        {
            "method": np.repeat([k for k, _, _ in band_rows], idx.size),
            "x1": np.tile(X_te[idx, 0], len(band_rows)),
            "y": np.tile(y_te[idx], len(band_rows)),
            "lo": np.concatenate([lo[idx] for _, lo, _ in band_rows]),
            "hi": np.concatenate([hi[idx] for _, _, hi in band_rows]),
        }
    )
    return pd.DataFrame(strat_rows), pd.DataFrame(marg_rows), band


def summarize(strat: pd.DataFrame, marg: pd.DataFrame) -> pd.DataFrame:
    """手法×ビンの集計（仕様書「出力」の e3_summary.csv）。

    被覆率の標準誤差は試行間とテスト集合間の2成分の合成
    （CLAUDE.md「報告の作法」）。**層別ではビン内の点数が 1/5 になるので
    se_test が √5 倍効く。** se_test は全試行に共通のバイアスなので
    sqrt(R) では薄まらない。

    手法ごとの指標（gap, worst_bin_coverage, width_ratio, spearman_bin_width）は
    粒度が違うが、同じ csv に収めるため各行に同じ値を繰り返して入れている。
    """
    rows = []
    for key in METHOD_KEYS:
        g = strat[strat["method"] == key]
        per_bin = []
        for b in range(N_BINS):
            gb = g[g["bin"] == b]
            cov = gb["coverage"].to_numpy()
            m = float(cov.mean())
            n_bin = int(gb["n_bin"].iloc[0])
            se_trial = float(cov.std(ddof=1) / np.sqrt(cov.size))
            # ビンごとの二項ノイズ。実際のビン点数を使う（仕様書の N_test/5 の意図どおり）
            se_test = float(np.sqrt(m * (1 - m) / n_bin))
            se = float(np.sqrt(se_trial**2 + se_test**2))
            per_bin.append(
                {
                    "method": key,
                    "bin": b,
                    "n_bin": n_bin,
                    "coverage_mean": m,
                    "se_trial": se_trial,
                    "se_test": se_test,
                    "ci_lo": m - 1.96 * se,
                    "ci_hi": m + 1.96 * se,
                    "width_mean": float(gb["mean_width"].mean()),
                }
            )

        cov_means = np.array([r["coverage_mean"] for r in per_bin])
        w_means = np.array([r["width_mean"] for r in per_bin])
        gap = float(cov_means.max() - cov_means.min())
        worst = float(cov_means.min())
        ratio = float(w_means[-1] / w_means[0])
        rho = float(stats.spearmanr(np.arange(N_BINS), w_means).correlation)

        for r in per_bin:
            r.update(
                {
                    "gap": gap,
                    "worst_bin_coverage": worst,
                    "width_ratio": ratio,
                    "spearman_bin_width": rho,
                }
            )
        rows.extend(per_bin)
    return pd.DataFrame(rows)


def judge(summary: pd.DataFrame, marg: pd.DataFrame) -> None:
    """検証基準 C1〜C4 の自動判定と、報告義務3項目の出力（仕様書「検証」）。"""
    ok, ng = "○", "×"
    lower, band = 1 - ALPHA, 1 / (N_CAL + 1)
    upper = lower + band

    print("\n" + "=" * 78)
    print(f"実験E3 結果  n_cal={N_CAL}, alpha={ALPHA}, 試行={N_TRIAL}, "
          f"テスト集合={N_TEST:,}点, ビン={N_BINS}")
    print("=" * 78)

    # 手法ごとの周辺量。被覆の SE は試行間＋テスト集合間の2成分
    marg_stats = {}
    for key in METHOD_KEYS:
        cov = marg[marg["method"] == key]["coverage"].to_numpy()
        m = float(cov.mean())
        se_trial = float(cov.std(ddof=1) / np.sqrt(cov.size))
        se_test = float(np.sqrt(m * (1 - m) / N_TEST))
        marg_stats[key] = {
            "mean": m,
            "se": float(np.sqrt(se_trial**2 + se_test**2)),
            "se_trial": se_trial,
            "se_test": se_test,
            "width": float(marg[marg["method"] == key]["mean_width"].mean()),
        }

    print("\n【周辺被覆】（SE は試行間＋テスト集合間の2成分合成）")
    print(f"  {'手法':12}{'被覆率':>10}{'2SE':>10}{'平均幅':>10}"
          f"{'se_trial':>11}{'se_test':>10}")
    for key in METHOD_KEYS:
        s = marg_stats[key]
        print(f"  {LABELS[key]:12}{s['mean']:>10.5f}{2 * s['se']:>10.5f}"
              f"{s['width']:>10.4f}{s['se_trial']:>11.6f}{s['se_test']:>10.6f}")

    # --- C1: 3手法とも周辺被覆を満たす ---
    in_band = {
        key: bool(lower - 2 * marg_stats[key]["se"] <= marg_stats[key]["mean"]
                  <= upper + 2 * marg_stats[key]["se"])
        for key in METHOD_KEYS
    }
    c1 = all(in_band.values())
    print(f"\n【C1】3手法とも周辺被覆を満たす ... {ok if c1 else ng}")
    print(f"  理論の上下界 [{lower:.5f}, {upper:.5f}]（±2SE の許容つきで判定）")
    for key in METHOD_KEYS:
        print(f"    {LABELS[key]:12}{marg_stats[key]['mean']:.5f}  "
              f"{ok if in_band[key] else ng}")
    if not c1:
        print("  → 崩れている。異分散でも周辺被覆は保証されるはずなので実装ミスを疑うこと")

    # --- 層別被覆の表 ---
    print("\n【層別被覆】X1 の固定境界 0.2/0.4/0.6/0.8 で5層に分割")
    print(f"  {'手法':12}" + "".join(f"{f'層{b}':>11}" for b in range(N_BINS))
          + f"{'gap':>9}{'最悪層':>9}")
    for key in METHOD_KEYS:
        g = summary[summary["method"] == key].sort_values("bin")
        cells = "".join(f"{v:>11.4f}" for v in g["coverage_mean"])
        print(f"  {LABELS[key]:12}{cells}{g['gap'].iloc[0]:>9.4f}"
              f"{g['worst_bin_coverage'].iloc[0]:>9.4f}")

    print(f"\n  {'手法':12}" + "".join(f"{f'層{b}の幅':>11}" for b in range(N_BINS))
          + f"{'幅の比':>9}")
    for key in METHOD_KEYS:
        g = summary[summary["method"] == key].sort_values("bin")
        cells = "".join(f"{v:>11.4f}" for v in g["width_mean"])
        print(f"  {LABELS[key]:12}{cells}{g['width_ratio'].iloc[0]:>9.4f}")

    gaps = {k: float(summary[summary["method"] == k]["gap"].iloc[0]) for k in METHOD_KEYS}
    worst = {k: float(summary[summary["method"] == k]["worst_bin_coverage"].iloc[0])
             for k in METHOD_KEYS}

    # --- C2: (a) で層別被覆が割れる ---
    c2 = gaps["abs"] >= 0.10
    print(f"\n【C2】絶対残差で層別被覆が割れる ... {ok if c2 else ng}")
    print(f"  gap = {gaps['abs']:.4f} {'>=' if c2 else '<'} 0.10")
    print(f"  最悪層の被覆率 = {worst['abs']:.4f}（名目 {lower:.2f}）")
    if not c2:
        print("  → 失敗ではなく「異分散の強さが足りない」という結果。"
              "sigma(x) の傾きを上げた追加実験を検討すること")

    # --- C3: (b)(c) で割れが縮む ---
    shrink = {k: gaps[k] <= gaps["abs"] / 2 for k in ("norm", "cqr")}
    # 「改善」は名目 1-alpha からの乖離が小さくなること
    improved = {k: abs(worst[k] - lower) < abs(worst["abs"] - lower)
                for k in ("norm", "cqr")}
    c3 = all(shrink.values()) and all(improved.values())
    print(f"\n【C3】正規化残差と CQR で割れが縮む ... {ok if c3 else ng}")
    print(f"  基準: gap が abs の 1/2 = {gaps['abs'] / 2:.4f} 以下")
    for key in ("norm", "cqr"):
        print(f"    {LABELS[key]:12} gap {gaps[key]:.4f} "
              f"({gaps[key] / gaps['abs']:.0%} に縮小)  {ok if shrink[key] else ng}"
              f"   最悪層 {worst[key]:.4f}  {ok if improved[key] else ng}")
    print("  ※ 不可能性定理により平らにはならない。どこまで縮むかを測る実験である")

    # --- C4: (c) の区間が適応的である ---
    rho_cqr = float(summary[summary["method"] == "cqr"]["spearman_bin_width"].iloc[0])
    ratio_cqr = float(summary[summary["method"] == "cqr"]["width_ratio"].iloc[0])
    ratio_abs = float(summary[summary["method"] == "abs"]["width_ratio"].iloc[0])
    c4_rho = rho_cqr >= 1.0 - 1e-12
    c4_ratio = ratio_cqr >= 3.0
    c4_abs = abs(ratio_abs - 1.0) < 1e-9
    c4 = c4_rho and c4_ratio and c4_abs
    print(f"\n【C4】CQR の区間が適応的である ... {ok if c4 else ng}")
    print(f"  Spearman(ビン番号, 平均幅) = {rho_cqr:+.4f}  "
          f"{ok if c4_rho else ng}（基準: 1.0）")
    print(f"  最上位ビン / 最下位ビンの幅の比 = {ratio_cqr:.4f}  "
          f"{ok if c4_ratio else ng}（基準: 3.0 以上、層の sigma 比は 5.0）")
    print(f"  絶対残差の同じ比 = {ratio_abs:.6f}  "
          f"{ok if c4_abs else ng}（定数幅なので 1.0 になるはず）")

    # --- 報告義務（判定ではないが必ず出す）---
    print("\n【報告義務】")
    cr = marg[marg["method"] == "cqr"]["crossing_rate"].iloc[0]
    sc = marg[marg["method"] == "norm"]["sigma_clip_rate"].iloc[0]
    print(f"  CQR の分位点交差率        {cr:.5f}"
          f"（テスト集合 {N_TEST:,} 点中 約{int(round(cr * N_TEST)):,} 点）")
    print(f"  sigma のクリップ発動率    {sc:.5f}"
          f"（同 約{int(round(sc * N_TEST)):,} 点）")
    n_ssc = marg["ssc_worst"].notna().groupby(marg["method"]).sum()
    print("  SSC の最悪層:")
    for key in METHOD_KEYS:
        v = marg[marg["method"] == key]["ssc_worst"]
        if v.notna().any():
            print(f"    {LABELS[key]:12}{v.mean():.4f}")
        else:
            print(f"    {LABELS[key]:12}—（幅が定数なので層が1つに潰れる。"
                  "欠陥ではなく、定数幅の手法にサイズ層別という概念がないだけ）")

    print("\n" + "-" * 78)
    print(f"判定: C1 {ok if c1 else ng} / C2 {ok if c2 else ng} / "
          f"C3 {ok if c3 else ng} / C4 {ok if c4 else ng}"
          f"  → {'全基準を満たす' if all((c1, c2, c3, c4)) else '未達の基準あり（上記を参照）'}")
    print("-" * 78)


def main(seed: int) -> None:
    rng = np.random.default_rng(seed)

    # 乱数の導出（仕様書「乱数の導出」）。この引き順は変えないこと
    model_seeds = {k: int(rng.integers(0, 2**31 - 1)) for k in MODEL_SEED_KEYS}
    X_tr, y_tr = heteroscedastic(N_TRAIN, rng)
    X_te, y_te = heteroscedastic(N_TEST, rng)

    print(f"学習集合 {N_TRAIN:,} 点で3手法の下敷きを学習中...")
    f_hat, sigma_est, cqr_pair = fit_all(X_tr, y_tr, model_seeds)

    print(f"試行ループ: {N_TRIAL} 試行 × {len(METHOD_KEYS)} 手法")
    strat, marg, band = run_trials(rng, f_hat, sigma_est, cqr_pair, X_te, y_te)
    summary = summarize(strat, marg)

    RESULTS.mkdir(exist_ok=True)
    strat.to_csv(RESULTS / "e3_conditional.csv", index=False)
    marg.to_csv(RESULTS / "e3_marginal.csv", index=False)
    summary.to_csv(RESULTS / "e3_summary.csv", index=False)
    band.to_csv(RESULTS / "e3_band_trial0.csv", index=False)

    judge(summary, marg)
    for name in ("e3_conditional.csv", "e3_marginal.csv", "e3_summary.csv",
                 "e3_band_trial0.csv"):
        print(f"出力: {RESULTS / name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="実験E3: 異分散データと条件付き被覆の破綻")
    ap.add_argument("--seed", type=int, default=20260901, help="乱数シード（再現性のため必須）")
    main(ap.parse_args().seed)
