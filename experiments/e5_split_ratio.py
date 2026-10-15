"""実験E5: 学習／較正の分割比。

対応箇所:
    卒論 第5章 E5 の節（sec:e5）
    仕様書 specs/E5_split_ratio.md

データの総数 1,000 点を学習用と較正用に分ける比を 5 通りに変え、被覆率のばらつき・
区間の幅・予測モデルの精度がどう変わるかを測る。E1〜E4 と違い、試行ごとに学習用データも
引き直す（比べたいのは学習用の量なので、1 回の学習の出来に結果が左右されないようにする）。

理論（卒論 thm:cond-coverage-beta）：学習用データを固定したときの被覆率は Beta(k, n+1-k)
に従い、この分布は学習用データによらない。よって学習用データを引き直しても被覆率は
Beta(k, n+1-k) に従う。平均は k/(n+1)（cor:upper）。n は較正用の点数。

乱数の引き順（同じ seed で同じ csv を出すため、この順は変えないこと）:
    1. テスト集合 100,000 点（全試行・全割合で共有）
    2. 試行 0, 1, ..., 499 の順に、試行ごとに
       a. データ 1,000 点（学習用＋較正用）
       b. 0..999 のランダムな並べ替え 1 つ
       c. 予測モデルの random_state（整数を 1 つ。5 つの割合で共有）

実行:
    python experiments/e5_split_ratio.py --seed 20260901
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from conformal import absolute_residual, conformal_index, conformal_quantile
from conformal.datasets import homoscedastic

# --- 規模のパラメータ（仕様書「規模のパラメータ」）---
N_TOTAL, N_TEST = 1_000, 100_000
N_TRIAL, ALPHA = 500, 0.10

# 較正用の割合。較正用 n_cal = 100..500、学習用 n_train = 900..500
CAL_FRACS = (0.1, 0.2, 0.3, 0.4, 0.5)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def split_sizes(cal_frac: float) -> tuple[int, int]:
    """較正用の割合から (n_train, n_cal) を返す。round で float の丸め（0.3*1000 など）を避ける。"""
    n_cal = round(N_TOTAL * cal_frac)
    return N_TOTAL - n_cal, n_cal


def run_trials(rng: np.random.Generator, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """試行ループ（仕様書「手続き」）。

    試行ごとにデータ 1,000 点と並べ替えを引き直し、5 つの割合で同じデータ・同じ並べ替えを使う
    （割合どうしの比較を対にする）。並べ替えた先頭 n_train 点を学習用、残り n_cal 点を較正用にする。
    **較正用データは学習に使わない**（下の assert で重なりがないことを確かめる）。
    """
    rows = []
    for trial in range(N_TRIAL):
        X, y = homoscedastic(N_TOTAL, rng)
        perm = rng.permutation(N_TOTAL)
        model_seed = int(rng.integers(0, 2**31 - 1))

        for cal_frac in CAL_FRACS:
            n_train, n_cal = split_sizes(cal_frac)
            idx_train, idx_cal = perm[:n_train], perm[n_train:]
            # データリークの禁止（CLAUDE.md）：学習用と較正用は重ならず、合わせて全データ
            assert np.intersect1d(idx_train, idx_cal).size == 0
            assert idx_train.size + idx_cal.size == N_TOTAL

            model = GradientBoostingRegressor(random_state=model_seed)
            model.fit(X[idx_train], y[idx_train])

            # 絶対残差（卒論 eq:score）と共形分位点（卒論 eq:khat。k = ceil((n_cal+1)(1-alpha))）
            cal_scores = absolute_residual(y[idx_cal], model.predict(X[idx_cal]))
            q_hat = conformal_quantile(cal_scores, ALPHA)

            # 区間は [f(x) - q_hat, f(x) + q_hat] なので、被覆は「テスト点の残差 <= q_hat」と同値
            pred_test = model.predict(X_test)
            test_scores = absolute_residual(y_test, pred_test)
            rows.append(
                {
                    "trial": trial,
                    "cal_frac": cal_frac,
                    "n_train": n_train,
                    "n_cal": n_cal,
                    "k": conformal_index(n_cal, ALPHA),
                    "q_hat": q_hat,
                    "coverage": float(np.mean(test_scores <= q_hat)),
                    "width": 2.0 * q_hat,
                    "test_mse": float(np.mean((y_test - pred_test) ** 2)),
                }
            )

        if (trial + 1) % 50 == 0:
            print(f"  {trial + 1} / {N_TRIAL} 試行 完了")
    return pd.DataFrame(rows)


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    """割合ごとの集計（仕様書「出力」の e5_summary.csv）。

    理論の Beta(k, n+1-k)（卒論 thm:cond-coverage-beta）の平均と SD:
        beta_mean = k / (n+1)
        beta_sd   = sqrt( k (n+1-k) / ((n+1)^2 (n+2)) )
    被覆率の標準誤差は 2 成分の合成（CLAUDE.md「報告の作法」）:
        se_trial = 試行間 SD / sqrt(R)          データ（学習用・較正用）の引き直し
        se_test  = sqrt(p(1-p) / N_test)        固定のテスト集合を全試行で共有するぶん
        se_total = sqrt(se_trial^2 + se_test^2)
    """
    rows = []
    for cal_frac in CAL_FRACS:
        g = trials[trials["cal_frac"] == cal_frac]
        n_train, n_cal = split_sizes(cal_frac)
        k = conformal_index(n_cal, ALPHA)
        cov = g["coverage"].to_numpy()
        mean = float(cov.mean())
        sd = float(cov.std(ddof=1))
        beta_mean = k / (n_cal + 1)
        beta_sd = float(np.sqrt(k * (n_cal + 1 - k) / ((n_cal + 1) ** 2 * (n_cal + 2))))
        se_trial = sd / np.sqrt(cov.size)
        se_test = float(np.sqrt(mean * (1 - mean) / N_TEST))
        se_total = float(np.sqrt(se_trial**2 + se_test**2))
        rows.append(
            {
                "cal_frac": cal_frac,
                "n_train": n_train,
                "n_cal": n_cal,
                "k": k,
                "alpha": ALPHA,
                "n_trial": cov.size,
                "n_test": N_TEST,
                "coverage_mean": mean,
                "coverage_sd": sd,
                "beta_mean": beta_mean,
                "beta_sd": beta_sd,
                "sd_ratio": sd / beta_sd,
                "se_trial": se_trial,
                "se_test": se_test,
                "se_total": se_total,
                "ci_lo": mean - 1.96 * se_total,
                "ci_hi": mean + 1.96 * se_total,
                "width_mean": float(g["width"].mean()),
                "width_sd": float(g["width"].std(ddof=1)),
                "test_mse_mean": float(g["test_mse"].mean()),
                "test_mse_sd": float(g["test_mse"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def judge(summary: pd.DataFrame) -> None:
    """検証基準 V1〜V4 の判定を画面に出す（仕様書「検証」。基準は実行前に決めたもの）。"""
    ok, ng = "○", "×"
    s = summary.sort_values("cal_frac")

    print("\n" + "=" * 92)
    print(f"実験E5 結果  N_total={N_TOTAL}, alpha={ALPHA}, 試行={N_TRIAL}, テスト集合={N_TEST:,}点")
    print("=" * 92)
    print(f"  {'割合':>4}{'n_cal':>6}{'k':>5}{'被覆率':>10}{'理論平均':>10}{'se_total':>10}"
          f"{'SD':>9}{'理論SD':>9}{'SD比':>7}{'平均幅':>9}{'test_mse':>10}")
    for _, r in s.iterrows():
        print(f"  {r['cal_frac']:>5.1f}{int(r['n_cal']):>6}{int(r['k']):>5}{r['coverage_mean']:>11.5f}"
              f"{r['beta_mean']:>11.5f}{r['se_total']:>10.6f}{r['coverage_sd']:>10.5f}"
              f"{r['beta_sd']:>10.5f}{r['sd_ratio']:>8.3f}{r['width_mean']:>10.4f}{r['test_mse_mean']:>10.5f}")

    # --- V1: 平均が理論平均と 3*se_total 以内 ---
    dev = (s["coverage_mean"] - s["beta_mean"]).abs()
    v1 = bool((dev <= 3 * s["se_total"]).all())
    print(f"\n【V1】すべての割合で |被覆率 - k/(n+1)| <= 3*se_total ... {ok if v1 else ng}")
    for (_, r), d in zip(s.iterrows(), dev):
        print(f"    {r['cal_frac']:.1f}: |差| = {d:.5f}, 3*se_total = {3 * r['se_total']:.5f}"
              f"  {ok if d <= 3 * r['se_total'] else ng}")

    # --- V2: SD 比が 0.90〜1.10 ---
    v2 = bool(s["sd_ratio"].between(0.90, 1.10).all())
    print(f"【V2】すべての割合で 0.90 <= SD比 <= 1.10 ... {ok if v2 else ng}"
          f"（割合の順に {', '.join(f'{v:.3f}' for v in s['sd_ratio'])}）")

    # --- V3: coverage_sd が n_cal について単調に減少 ---
    sds = s["coverage_sd"].to_list()
    v3 = all(a > b for a, b in zip(sds, sds[1:]))
    print(f"【V3】被覆率の SD が n_cal について単調に減少 ... {ok if v3 else ng}"
          f"（{', '.join(f'{v:.5f}' for v in sds)}）")

    # --- V4: test_mse_mean が cal_frac について単調に増加 ---
    mses = s["test_mse_mean"].to_list()
    v4 = all(a < b for a, b in zip(mses, mses[1:]))
    print(f"【V4】test_mse の平均が n_train の減少（割合の増加）について単調に増加 ... {ok if v4 else ng}"
          f"（{', '.join(f'{v:.5f}' for v in mses)}）")

    results = (v1, v2, v3, v4)
    print("\n" + "-" * 92)
    print("判定: " + " / ".join(f"V{i} {ok if r else ng}" for i, r in enumerate(results, 1))
          + f"  → {'全基準を満たす' if all(results) else '未達の基準あり（上記を参照）'}")
    print("-" * 92)


def main(seed: int) -> None:
    rng = np.random.default_rng(seed)

    # 乱数の引き順はモジュール冒頭のコメントのとおり。変えないこと
    X_test, y_test = homoscedastic(N_TEST, rng)

    print(f"試行ループ: {N_TRIAL} 試行 × 割合 {len(CAL_FRACS)} 通り（勾配ブースティングの学習 "
          f"{N_TRIAL * len(CAL_FRACS):,} 回）")
    trials = run_trials(rng, X_test, y_test)
    summary = summarize(trials)

    RESULTS.mkdir(exist_ok=True)
    trials.to_csv(RESULTS / "e5_trials.csv", index=False)
    summary.to_csv(RESULTS / "e5_summary.csv", index=False)

    judge(summary)
    for name in ("e5_trials.csv", "e5_summary.csv"):
        print(f"出力: {RESULTS / name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="実験E5: 学習／較正の分割比")
    ap.add_argument("--seed", type=int, default=20260901, help="乱数シード（再現性のため必須）")
    main(ap.parse_args().seed)
