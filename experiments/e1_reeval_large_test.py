"""実験E1 の再評価: 大きな独立テスト集合で被覆率を測り直し、KS 検定をやり直す。

対応箇所: 卒論 第5章 E1 の節（sec:e1）/ 仕様書 specs/E1_coverage.md「再評価」

目的:
    E1（results/e1_summary.csv）では n=2000 の 2 セルで KS 検定が棄却された。
    仮説は「固定した 100,000 点のテスト集合による共通のずれ（Beta の SD の
    約 sqrt((n+2)/N) 倍）が、R=1000 の KS 検定で検出できる大きさだった」である。
    仮説が正しければ、テスト集合を 4,000,000 点にすれば（比は約 0.022）通るはずである。

やること:
    - 試行は再実行しない。results/e1_coverage.csv の width = 2 q_hat から各試行の
      共形分位点 q_hat を復元する
    - 学習集合とモデルは E1 と同じ seed・同じ引き順で再構成する（E1 と一致する）
    - E1 のテスト集合も同じ引き順で再生成し、csv の coverage が再現できることを確かめる
      （モデルの再構成が正しいことの検査。ずれたら止める）
    - 別の seed で 4,000,000 点の独立なテスト集合を作り、各試行の被覆率を測り直す
    - セルごとに KS p 値を出し、results/e1_reeval_large_test.csv に保存する
      （既存の csv は変更しない）

実行:
    python experiments/e1_reeval_large_test.py --seed 20260901 --test-seed 20260922
出力:
    results/e1_reeval_large_test.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from conformal import absolute_residual, conformal_index
from conformal.datasets import homoscedastic

# E1 と同じ値でなければならない（experiments/e1_coverage.py）
N_TRAIN, N_TEST_ORIGINAL = 500, 100_000
N_TEST_LARGE = 4_000_000  # E1b の「真値」と同じ規模
RESULTS = Path(__file__).resolve().parents[1] / "results"


def coverage_from_sorted(sorted_res: np.ndarray, q: float) -> float:
    """被覆率 mean(残差 <= q) をソート済み残差から O(log N) で求める。"""
    return float(np.searchsorted(sorted_res, q, side="right") / sorted_res.size)


def main(seed: int, test_seed: int) -> None:
    src = RESULTS / "e1_coverage.csv"
    if not src.exists():
        raise SystemExit("results/e1_coverage.csv がありません。先に `make e1` を実行してください。")
    df = pd.read_csv(src)
    if not np.all(np.isfinite(df["width"])):
        raise SystemExit("width に inf があり q_hat を復元できないセルがある")
    df["q_hat"] = df["width"] / 2.0

    # --- E1 と同じ引き順で学習集合・モデル・元のテスト集合を再構成する ---
    rng = np.random.default_rng(seed)
    X_tr, y_tr = homoscedastic(N_TRAIN, rng)
    model = Ridge(alpha=1.0).fit(X_tr, y_tr)
    X_te, y_te = homoscedastic(N_TEST_ORIGINAL, rng)
    res_orig = np.sort(absolute_residual(y_te, model.predict(X_te)))

    # 検査: 元のテスト集合で csv の coverage が再現できること（再構成が正しい根拠）
    recomputed = np.array([coverage_from_sorted(res_orig, q) for q in df["q_hat"]])
    max_dev = float(np.max(np.abs(recomputed - df["coverage"].to_numpy())))
    if max_dev > 1e-12:
        raise SystemExit(f"元のテスト集合で coverage が再現できない（最大差 {max_dev:.3e}）。"
                         "seed か引き順が E1 と違う")
    print(f"元のテスト集合での coverage を再現（最大差 {max_dev:.1e}）")

    # --- 独立な大きいテスト集合（別の seed）---
    rng_test = np.random.default_rng(test_seed)
    X_big, y_big = homoscedastic(N_TEST_LARGE, rng_test)
    res_big = np.sort(absolute_residual(y_big, model.predict(X_big)))
    df["coverage_reeval"] = [coverage_from_sorted(res_big, q) for q in df["q_hat"]]

    rows = []
    for (n_cal, alpha), g in df.groupby(["n_cal", "alpha"]):
        l = n_cal + 1 - conformal_index(n_cal, alpha)
        a_par, b_par = n_cal + 1 - l, l
        beta_sd = float(np.sqrt(a_par * b_par / ((a_par + b_par) ** 2 * (a_par + b_par + 1))))
        cov_o = g["coverage"].to_numpy()
        cov_r = g["coverage_reeval"].to_numpy()
        m_r = float(cov_r.mean())
        rows.append(
            {
                "n_cal": n_cal,
                "alpha": alpha,
                "n_test": N_TEST_LARGE,
                "ks_pvalue_original": stats.kstest(cov_o, "beta", args=(a_par, b_par)).pvalue,
                "ks_pvalue_reeval": stats.kstest(cov_r, "beta", args=(a_par, b_par)).pvalue,
                "coverage_mean_original": float(cov_o.mean()),
                "coverage_mean_reeval": m_r,
                # 固定テスト集合 1 つ分の二項ノイズ。sqrt(R) では割らない（CLAUDE.md「報告の作法」）
                "se_test_reeval": float(np.sqrt(m_r * (1 - m_r) / N_TEST_LARGE)),
                # 経験 SD / 理論 Beta の SD。1 に近いほど分布の広がりが理論と合う
                "sd_ratio_original": float(cov_o.std(ddof=1) / beta_sd),
                "sd_ratio_reeval": float(cov_r.std(ddof=1) / beta_sd),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "e1_reeval_large_test.csv", index=False)
    print(f"出力: {RESULTS / 'e1_reeval_large_test.csv'}")
    print(out[["n_cal", "alpha", "ks_pvalue_original", "ks_pvalue_reeval",
               "sd_ratio_original", "sd_ratio_reeval"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="実験E1 の再評価: 大きな独立テスト集合で KS 検定をやり直す")
    ap.add_argument("--seed", type=int, default=20260901, help="E1 と同じ seed（学習集合とモデルの再構成に使う）")
    ap.add_argument("--test-seed", type=int, default=20260922, help="独立なテスト集合の seed（E1 とは別にする）")
    main(ap.parse_args().seed, ap.parse_args().test_seed)
