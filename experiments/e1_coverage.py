"""実験E1: 有限標本被覆保証の検証。

対応箇所: 卒論 第5章 E1 の節（sec:e1）/ 仕様書 specs/E1_coverage.md

目的:
    分割等角予測の被覆率が理論の上下界 [1-alpha, 1-alpha+1/(n+1)] に入ることを確認し、
    被覆率の分布が Beta(n+1-l, l), l = floor((n+1)*alpha) と一致することを示す。

実行:
    python experiments/e1_coverage.py --seed 20260901
出力:
    results/e1_coverage.csv        試行ごとの被覆率と区間幅
    results/e1_summary.csv         (n, alpha) ごとの要約と理論値
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from conformal import absolute_residual, conformal_index, conformal_quantile
from conformal.datasets import homoscedastic

N_CALS = (20, 50, 100, 500, 2000)
ALPHAS = (0.05, 0.10, 0.20)
N_TRIAL = 1000
N_TEST = 100_000  # 二項ノイズを抑えるため大きめの固定テスト集合を使う
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main(seed: int) -> None:
    rng = np.random.default_rng(seed)

    # 学習データとモデルは全試行で固定する（キャリブレーション集合の効果だけを見るため）
    X_tr, y_tr = homoscedastic(500, rng)
    model = Ridge(alpha=1.0).fit(X_tr, y_tr)

    # テスト集合の絶対残差を先に計算しておく（被覆率 = mean(残差 <= q) で高速に評価できる）
    X_te, y_te = homoscedastic(N_TEST, rng)
    test_res = absolute_residual(y_te, model.predict(X_te))

    rows = []
    for n_cal in N_CALS:
        for alpha in ALPHAS:
            for trial in range(N_TRIAL):
                X_c, y_c = homoscedastic(n_cal, rng)
                q = conformal_quantile(absolute_residual(y_c, model.predict(X_c)), alpha)
                rows.append(
                    {
                        "n_cal": n_cal,
                        "alpha": alpha,
                        "trial": trial,
                        "coverage": float(np.mean(test_res <= q)),
                        "width": 2.0 * q,
                    }
                )
            print(f"  n_cal={n_cal:>5}, alpha={alpha:.2f} 完了")

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "e1_coverage.csv", index=False)

    # --- 要約: 理論の上下界と Beta 分布への適合 ---
    summary = []
    for (n_cal, alpha), g in df.groupby(["n_cal", "alpha"]):
        # l = floor((n+1)*alpha) は k = ceil((n+1)(1-alpha)) から l = n+1-k で求める。
        # float の floor だと alpha によっては 1 ずれる（例: n=89, alpha=0.7）。
        l = n_cal + 1 - conformal_index(n_cal, alpha)
        a_par, b_par = n_cal + 1 - l, l
        cov = g["coverage"].to_numpy()
        # 被覆率の平均に付ける区間は2成分の合成（CLAUDE.md「報告の作法」）。
        #   se_trial : 較正集合の引き直しによる試行間のばらつき。試行数 R で薄まる
        #   se_test  : 固定テスト集合1つ分の二項ノイズ。全試行に共通のバイアスとして
        #              乗るので **sqrt(R) で割ってはいけない**
        # n_cal が大きいほど se_trial は縮み、se_test が支配的になる。
        m = float(cov.mean())
        se_trial = float(cov.std(ddof=1) / np.sqrt(cov.size))
        se_test = float(np.sqrt(m * (1 - m) / N_TEST))
        se = float(np.sqrt(se_trial**2 + se_test**2))
        ci_lo, ci_hi = m - 1.96 * se, m + 1.96 * se
        summary.append(
            {
                "n_cal": n_cal,
                "alpha": alpha,
                "lower_bound": 1 - alpha,
                "coverage_mean": cov.mean(),
                "se_trial": se_trial,
                "se_test": se_test,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "upper_bound": 1 - alpha + 1 / (n_cal + 1),
                "in_bounds": bool(1 - alpha - 0.005 <= cov.mean() <= 1 - alpha + 1 / (n_cal + 1) + 0.005),
                "beta_a": a_par,
                "beta_b": b_par,
                "beta_mean": a_par / (a_par + b_par) if b_par > 0 else np.nan,
                "ks_pvalue": stats.kstest(cov, "beta", args=(a_par, b_par)).pvalue if b_par > 0 else np.nan,
                "width_mean": g["width"].mean(),
            }
        )
    pd.DataFrame(summary).to_csv(RESULTS / "e1_summary.csv", index=False)

    print(f"\n出力: {RESULTS / 'e1_coverage.csv'}")
    print(f"出力: {RESULTS / 'e1_summary.csv'}")
    print(pd.DataFrame(summary)[
        ["n_cal", "alpha", "lower_bound", "coverage_mean", "upper_bound", "in_bounds", "ks_pvalue"]
    ].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="実験E1: 有限標本被覆保証の検証")
    ap.add_argument("--seed", type=int, default=20260901, help="乱数シード（再現性のため必須）")
    main(ap.parse_args().seed)
