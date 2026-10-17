"""実験E6: jackknife+・CV+ とのデータ効率比較。

対応箇所:
    卒論 第5章 E6 の節（sec:e6）
    仕様書 specs/E6_jackknife_cv.md
出典: Barber, Candès, Ramdas, Tibshirani (2021)（refs.bib の barber2021jackknife）

データが少ないとき（総数 50, 100, 200 点）に、データを分けずに使う jackknife+ と CV+（K=10）を
分割等角予測（学習用：較正用 = 50:50）と比べる。比べるのは被覆率、区間の幅、学習の回数。
3 つの方法には、試行ごとに同じデータを使う（方法どうしの比較を対にする）。

理論の下限（出典の定理による。split は卒論 thm:coverage）:
    split        : 1 - alpha
    jackknife+   : 1 - 2*alpha
    CV+          : 1 - 2*alpha - sqrt(2/n_total)

乱数の引き順（同じ seed で同じ csv を出すため、この順は変えないこと）:
    1. テスト集合 10,000 点（全試行・全総数で共有）
    2. 総数 50, 100, 200 の順に、試行 0, 1, ..., 199 の順に、試行ごとに
       a. データ n_total 点
       b. split 用の 0..n_total-1 の並べ替え 1 つ
       c. 予測モデルの random_state（整数を 1 つ。3 つの方法のすべての学習で共有）
       d. CV+ の組分け用の 0..n_total-1 の並べ替え 1 つ（cv_plus_interval の中で引く）

実行時間は csv に入れない（同じ seed で csv が一致しなくなるため）。代わりに学習の回数 n_fits を記録する。

実行:
    python experiments/e6_jackknife_cv.py --seed 20260901
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from conformal import (
    absolute_residual,
    conformal_index,
    conformal_quantile,
    cv_plus_interval,
    interval_from_quantile,
    jackknife_plus_interval,
)
from conformal.datasets import homoscedastic

# --- 規模のパラメータ（仕様書「規模のパラメータ」）---
N_TOTALS = (50, 100, 200)
N_TEST, N_TRIAL, ALPHA = 10_000, 200, 0.10
N_FOLDS = 10  # CV+ の組の数 K

# csv に出す方法の順
METHODS = ("split", "jackknife_plus", "cv_plus")

RESULTS = Path(__file__).resolve().parents[1] / "results"


def make_model(seed: int) -> GradientBoostingRegressor:
    """予測モデル（勾配ブースティング、scikit-learn の既定値）。"""
    return GradientBoostingRegressor(random_state=seed)


def split_interval(X, y, X_test, perm, model_seed):
    """分割等角予測の区間（卒論 第3章、eq:khat）。並べ替えた前半を学習用、後半を較正用にする。

    **較正用データは学習に使わない**（assert で重なりがないことを確かめる）。
    """
    n_train = X.shape[0] // 2
    idx_train, idx_cal = perm[:n_train], perm[n_train:]
    assert np.intersect1d(idx_train, idx_cal).size == 0
    model = make_model(model_seed)
    model.fit(X[idx_train], y[idx_train])
    q_hat = conformal_quantile(absolute_residual(y[idx_cal], model.predict(X[idx_cal])), ALPHA)
    lo, hi = interval_from_quantile(model.predict(X_test), q_hat)
    return lo, hi, 1


def interval_stats(lo, hi, y_test) -> dict:
    """被覆率、有限な区間の平均の幅（全区間が無限なら NaN）、無限区間の割合。"""
    width = hi - lo
    finite = np.isfinite(width)
    return {
        "coverage": float(np.mean((lo <= y_test) & (y_test <= hi))),
        "mean_width": float(width[finite].mean()) if finite.any() else float("nan"),
        "frac_infinite": float(np.mean(~finite)),
    }


def run_trials(rng: np.random.Generator, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """試行ループ（仕様書「手続き」）。乱数の引き順はモジュール冒頭のとおり。"""
    rows = []
    for n_total in N_TOTALS:
        start = time.perf_counter()
        for trial in range(N_TRIAL):
            X, y = homoscedastic(n_total, rng)
            perm = rng.permutation(n_total)
            model_seed = int(rng.integers(0, 2**31 - 1))

            results = {
                "split": split_interval(X, y, X_test, perm, model_seed),
                "jackknife_plus": jackknife_plus_interval(
                    make_model, X, y, X_test, ALPHA, [model_seed] * n_total
                ),
                "cv_plus": cv_plus_interval(
                    make_model, X, y, X_test, ALPHA, N_FOLDS, rng, [model_seed] * N_FOLDS
                ),
            }
            for method in METHODS:
                lo, hi, n_fits = results[method]
                rows.append({"n_total": n_total, "method": method, "trial": trial,
                             **interval_stats(lo, hi, y_test), "n_fits": n_fits})

            if (trial + 1) % 20 == 0:
                print(f"  n_total={n_total}: {trial + 1} / {N_TRIAL} 試行 完了"
                      f"（{time.perf_counter() - start:.0f} 秒）", flush=True)

    trials = pd.DataFrame(rows)
    # csv の並びを n_total, method（METHODS の順）, trial にそろえる
    trials["method"] = pd.Categorical(trials["method"], categories=METHODS, ordered=True)
    trials = trials.sort_values(["n_total", "method", "trial"], kind="stable").reset_index(drop=True)
    trials["method"] = trials["method"].astype(str)
    return trials


def theory_bound(method: str, n_total: int) -> float:
    """理論の被覆率の下限（モジュール冒頭）。"""
    if method == "split":
        return 1 - ALPHA
    if method == "jackknife_plus":
        return 1 - 2 * ALPHA
    return 1 - 2 * ALPHA - float(np.sqrt(2 / n_total))


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    """総数×方法ごとの集計（仕様書「出力」の e6_summary.csv）。

    被覆率の標準誤差は 2 成分の合成（CLAUDE.md「報告の作法」）:
        se_trial = 試行間 SD / sqrt(R)          データの引き直し
        se_test  = sqrt(p(1-p) / N_test)        固定のテスト集合を全試行で共有するぶん
        se_total = sqrt(se_trial^2 + se_test^2)
    幅は試行ごとの mean_width の平均と SD（NaN の試行は除く）。
    """
    rows = []
    for n_total in N_TOTALS:
        for method in METHODS:
            g = trials[(trials["n_total"] == n_total) & (trials["method"] == method)]
            cov = g["coverage"].to_numpy()
            mean = float(cov.mean())
            sd = float(cov.std(ddof=1))
            se_trial = sd / np.sqrt(cov.size)
            se_test = float(np.sqrt(mean * (1 - mean) / N_TEST))
            se_total = float(np.sqrt(se_trial**2 + se_test**2))
            n_fits = g["n_fits"].unique()
            assert n_fits.size == 1
            rows.append(
                {
                    "n_total": n_total,
                    "method": method,
                    "alpha": ALPHA,
                    "n_trial": cov.size,
                    "n_test": N_TEST,
                    "coverage_mean": mean,
                    "coverage_sd": sd,
                    "se_trial": se_trial,
                    "se_test": se_test,
                    "se_total": se_total,
                    "ci_lo": mean - 1.96 * se_total,
                    "ci_hi": mean + 1.96 * se_total,
                    "width_mean": float(g["mean_width"].mean()),
                    "width_sd": float(g["mean_width"].std(ddof=1)),
                    "frac_infinite": float(g["frac_infinite"].mean()),
                    "n_fits": int(n_fits[0]),
                    "bound": theory_bound(method, n_total),
                }
            )
    return pd.DataFrame(rows)


def judge(summary: pd.DataFrame) -> None:
    """検証基準 V1〜V3 の判定を画面に出す（仕様書「検証」。基準は実行前に決めたもの）。"""
    ok, ng = "○", "×"
    print("\n" + "=" * 100)
    print(f"実験E6 結果  alpha={ALPHA}, 試行={N_TRIAL}, テスト集合={N_TEST:,}点, CV+ の K={N_FOLDS}")
    print("=" * 100)
    print(f"  {'n_total':>7} {'方法':<15}{'被覆率':>9}{'se_total':>10}{'下限':>9}"
          f"{'平均幅':>9}{'幅SD':>8}{'無限':>7}{'学習回数':>8}")
    for _, r in summary.iterrows():
        print(f"  {int(r['n_total']):>7} {r['method']:<15}{r['coverage_mean']:>10.5f}{r['se_total']:>10.5f}"
              f"{r['bound']:>10.5f}{r['width_mean']:>10.4f}{r['width_sd']:>9.4f}"
              f"{r['frac_infinite']:>8.3f}{int(r['n_fits']):>8}")

    # --- V1: jackknife+ と CV+ が理論の下限 - 3*se_total 以上 ---
    s1 = summary[summary["method"] != "split"]
    v1_each = s1["coverage_mean"] >= s1["bound"] - 3 * s1["se_total"]
    v1 = bool(v1_each.all())
    print(f"\n【V1】jackknife+・CV+ で coverage_mean >= bound - 3*se_total ... {ok if v1 else ng}")
    for (_, r), good in zip(s1.iterrows(), v1_each):
        print(f"    n={int(r['n_total'])} {r['method']}: {r['coverage_mean']:.5f} >= "
              f"{r['bound'] - 3 * r['se_total']:.5f}  {ok if good else ng}")

    # --- V2: split が [1-alpha - 3se, k/(n_cal+1) + 3se] に入る ---
    s2 = summary[summary["method"] == "split"]
    v2 = True
    print("【V2】split の coverage_mean が [1-alpha - 3*se_total, k/(n_cal+1) + 3*se_total] に入る")
    for _, r in s2.iterrows():
        n_cal = int(r["n_total"]) - int(r["n_total"]) // 2
        k = conformal_index(n_cal, ALPHA)
        lo, hi = 1 - ALPHA - 3 * r["se_total"], k / (n_cal + 1) + 3 * r["se_total"]
        good = lo <= r["coverage_mean"] <= hi
        v2 = v2 and good
        print(f"    n={int(r['n_total'])} (n_cal={n_cal}, k={k}): {r['coverage_mean']:.5f} in "
              f"[{lo:.5f}, {hi:.5f}]  {ok if good else ng}")
    print(f"    → {ok if v2 else ng}")

    # --- V3: 学習の回数 ---
    expected = {"split": lambda n: 1, "jackknife_plus": lambda n: n, "cv_plus": lambda n: N_FOLDS}
    v3 = all(int(r["n_fits"]) == expected[r["method"]](int(r["n_total"])) for _, r in summary.iterrows())
    print(f"【V3】n_fits が split 1、CV+ {N_FOLDS}、jackknife+ n_total ... {ok if v3 else ng}")

    results = (v1, v2, v3)
    print("\n" + "-" * 100)
    print("判定: " + " / ".join(f"V{i} {ok if r else ng}" for i, r in enumerate(results, 1))
          + f"  → {'全基準を満たす' if all(results) else '未達の基準あり（上記を参照）'}")
    print("-" * 100)


def main(seed: int) -> None:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)

    # 乱数の引き順はモジュール冒頭のコメントのとおり。変えないこと
    X_test, y_test = homoscedastic(N_TEST, rng)

    n_fit_total = N_TRIAL * sum(1 + n + N_FOLDS for n in N_TOTALS)
    print(f"試行ループ: 総数 {N_TOTALS} × {N_TRIAL} 試行（勾配ブースティングの学習 {n_fit_total:,} 回）")
    trials = run_trials(rng, X_test, y_test)
    summary = summarize(trials)

    RESULTS.mkdir(exist_ok=True)
    trials.to_csv(RESULTS / "e6_trials.csv", index=False)
    summary.to_csv(RESULTS / "e6_summary.csv", index=False)

    judge(summary)
    for name in ("e6_trials.csv", "e6_summary.csv"):
        print(f"出力: {RESULTS / name}")
    print(f"総実行時間: {time.perf_counter() - start:.0f} 秒")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="実験E6: jackknife+・CV+ とのデータ効率比較")
    ap.add_argument("--seed", type=int, default=20260901, help="乱数シード（再現性のため必須）")
    main(ap.parse_args().seed)
