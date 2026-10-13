"""実験E4: 交換可能性が破れる場合（共変量シフト）と重み付き等角予測。

対応箇所:
    卒論 第5章 E4 の節（sec:e4）, 第4章 共変量シフトの節（sec:covariate-shift）
    仕様書 specs/E4_covariate_shift.md

較正用データとテスト点で入力 X1 の分布がずれると、通常の分割等角予測の被覆率は
1-alpha を下回る。ずれの尤度比（正しい重み）で較正スコアを重み付けすると被覆が戻る。
この2つを、ずれの強さ gamma を変えながら測る。

乱数の引き順（同じ seed で同じ csv を出すため、この順は変えないこと）:
    1. 予測モデルの random_state（整数を1つ）
    2. 学習用データ 2,000 点（シフトなし）
    3. テスト集合 200,000 点を gamma = 0, 1, 2, 4 の順に 1 組ずつ
    4. 試行 0, 1, ..., 499 の順に、較正用データ 1,000 点（シフトなし）を 1 組ずつ
       （1 つの較正用データを、全部の gamma と 2 つの手法で共有する）

実行:
    python experiments/e4_covariate_shift.py --seed 20260901
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from conformal import absolute_residual, conformal_quantile, weighted_conformal_quantile
from conformal.datasets import heteroscedastic, heteroscedastic_shifted, shift_weight

# --- 規模のパラメータ（仕様書「規模のパラメータ」）---
N_TRAIN, N_CAL, N_TEST = 2_000, 1_000, 200_000
N_TRIAL, ALPHA = 500, 0.10

# シフトの強さ。テスト点の X1 の密度が (1+gamma) x^gamma。0 はシフトなし。
# この順序は変えないこと（テスト集合をこの順に rng から引く）
GAMMAS = (0.0, 1.0, 2.0, 4.0)

# 手法。unweighted は通常の分割等角予測、weighted は正しい重みを使う重み付き等角予測
METHOD_KEYS = ("unweighted", "weighted")
LABELS = {"unweighted": "重みなし", "weighted": "重み付き"}

RESULTS = Path(__file__).resolve().parents[1] / "results"


def evaluate(q_hat: np.ndarray, test_scores: np.ndarray) -> dict[str, float]:
    """テスト点ごとの共形分位点から、1 試行ぶんの被覆率・平均幅・無限区間の割合を出す。

    区間は [f(x) - q_hat, f(x) + q_hat] なので、被覆は「テスト点のスコア <= q_hat」と同値で、
    幅は 2 * q_hat。平均幅は有限な区間だけで取る（1 つでも +inf があると平均が +inf になり、
    幅の比較ができなくなるため。無限区間がどれだけあったかは frac_infinite で別に報告する）。
    """
    finite = np.isfinite(q_hat)
    return {
        "coverage": float(np.mean(test_scores <= q_hat)),
        "mean_width": float(np.mean(2.0 * q_hat[finite])) if finite.any() else np.nan,
        "frac_infinite": float(np.mean(~finite)),
    }


def run_trials(rng: np.random.Generator, model, test_sets: dict[float, dict]) -> pd.DataFrame:
    """試行ループ（仕様書「手続き」）。

    試行ごとに引き直すのは較正用データ（シフトなし）だけ。学習用データ・モデル・
    テスト集合は全試行で固定する。**較正用データは学習に使わない**（モデルは main で
    学習済みのものを受け取るだけ）。同じ較正用データを全部の gamma と 2 手法で共有する。
    """
    rows = []
    for trial in range(N_TRIAL):
        X_cal, y_cal = heteroscedastic(N_CAL, rng)
        cal_scores = absolute_residual(y_cal, model.predict(X_cal))

        # 重みなし：通常の共形分位点（卒論 eq:khat）。gamma によらず 1 つの値
        q_unweighted = conformal_quantile(cal_scores, ALPHA)

        for gamma in GAMMAS:
            test = test_sets[gamma]
            # 重み付き：テスト点ごとに分位点が変わる（卒論 sec:covariate-shift）
            q_weighted = weighted_conformal_quantile(
                cal_scores, shift_weight(X_cal, gamma), test["weights"], ALPHA
            )
            q_hats = {
                "unweighted": np.full(N_TEST, q_unweighted),
                "weighted": q_weighted,
            }
            for key in METHOD_KEYS:
                rows.append(
                    {"gamma": gamma, "method": key, "trial": trial}
                    | evaluate(q_hats[key], test["scores"])
                )

        if (trial + 1) % 100 == 0:
            print(f"  {trial + 1} / {N_TRIAL} 試行 完了")
    return pd.DataFrame(rows)


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    """gamma × 手法の集計（仕様書「出力」の e4_summary.csv）。

    被覆率の標準誤差は 2 成分の合成（CLAUDE.md「報告の作法」）:
        se_trial = 試行間 SD / sqrt(R)          較正用データの引き直し
        se_test  = sqrt(p(1-p) / N_test)        固定のテスト集合を全試行で共有するぶん
        se_total = sqrt(se_trial^2 + se_test^2)
    se_test は全試行に共通のずれなので、試行数を増やしても小さくならない。
    """
    rows = []
    for gamma in GAMMAS:
        for key in METHOD_KEYS:
            g = trials[(trials["gamma"] == gamma) & (trials["method"] == key)]
            cov = g["coverage"].to_numpy()
            mean = float(cov.mean())
            sd = float(cov.std(ddof=1))
            se_trial = sd / np.sqrt(cov.size)
            se_test = float(np.sqrt(mean * (1 - mean) / N_TEST))
            se_total = float(np.sqrt(se_trial**2 + se_test**2))
            rows.append(
                {
                    "gamma": gamma,
                    "method": key,
                    "n_cal": N_CAL,
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
                    "mean_width": float(g["mean_width"].mean()),
                    "frac_infinite": float(g["frac_infinite"].mean()),
                    "target": 1 - ALPHA,
                }
            )
    return pd.DataFrame(rows)


def judge(trials: pd.DataFrame, summary: pd.DataFrame) -> None:
    """検証基準 V0〜V4 の判定を画面に出す（仕様書「検証」。基準は実行前に決めたもの）。"""
    ok, ng = "○", "×"
    target = 1 - ALPHA
    by_method = {key: summary[summary["method"] == key].set_index("gamma") for key in METHOD_KEYS}

    print("\n" + "=" * 78)
    print(f"実験E4 結果  n_cal={N_CAL}, alpha={ALPHA}, 試行={N_TRIAL}, テスト集合={N_TEST:,}点")
    print("=" * 78)
    print("\n【被覆率】（SE は試行間＋テスト集合間の2成分合成 se_total）")
    print(f"  {'gamma':>5}  {'手法':8}{'被覆率':>9}{'se_total':>10}{'se_trial':>10}"
          f"{'se_test':>10}{'平均幅':>9}{'無限区間':>9}")
    for gamma in GAMMAS:
        for key in METHOD_KEYS:
            s = by_method[key].loc[gamma]
            print(f"  {gamma:>5.0f}  {LABELS[key]:8}{s['coverage_mean']:>10.5f}{s['se_total']:>10.6f}"
                  f"{s['se_trial']:>10.6f}{s['se_test']:>10.6f}{s['mean_width']:>10.4f}"
                  f"{s['frac_infinite']:>10.4f}")

    # --- V0: gamma = 0 では 2 手法の被覆率が試行ごとに完全に一致する ---
    g0 = trials[trials["gamma"] == 0.0].pivot(index="trial", columns="method", values="coverage")
    n_differ = int((g0["unweighted"] != g0["weighted"]).sum())
    v0 = n_differ == 0
    print(f"\n【V0】gamma=0 で 2 手法の被覆率が試行ごとに完全に一致 ... {ok if v0 else ng}"
          f"（一致しない試行 {n_differ} / {len(g0)}）")

    # --- V1: 重みなしは gamma = 1, 2, 4 で 1-alpha をはっきり下回る ---
    shifted = [gamma for gamma in GAMMAS if gamma > 0]
    unweighted, weighted = by_method["unweighted"], by_method["weighted"]
    v1_each = {gamma: bool(unweighted.loc[gamma, "coverage_mean"]
                           + 3 * unweighted.loc[gamma, "se_total"] < target) for gamma in shifted}
    v1 = all(v1_each.values())
    print(f"【V1】重みなしは gamma=1,2,4 で 被覆率 + 3*se_total < {target:.2f} ... {ok if v1 else ng}")
    for gamma in shifted:
        upper = unweighted.loc[gamma, "coverage_mean"] + 3 * unweighted.loc[gamma, "se_total"]
        print(f"    gamma={gamma:.0f}: {unweighted.loc[gamma, 'coverage_mean']:.5f} + 3SE = {upper:.5f}"
              f"  {ok if v1_each[gamma] else ng}")

    # --- V2: 重みなしの被覆率は gamma について単調に減少する ---
    unweighted_means = [float(unweighted.loc[gamma, "coverage_mean"]) for gamma in GAMMAS]
    v2 = all(a > b for a, b in zip(unweighted_means, unweighted_means[1:]))
    print(f"【V2】重みなしの被覆率が gamma について単調に減少 ... {ok if v2 else ng}"
          f"（gamma の順に {', '.join(f'{m:.5f}' for m in unweighted_means)}）")

    # --- V3: 重み付きはすべての gamma で 1-alpha - 3*se_total 以上 ---
    v3_each = {gamma: bool(weighted.loc[gamma, "coverage_mean"]
                           >= target - 3 * weighted.loc[gamma, "se_total"]) for gamma in GAMMAS}
    v3 = all(v3_each.values())
    print(f"【V3】重み付きはすべての gamma で 被覆率 >= {target:.2f} - 3*se_total ... {ok if v3 else ng}")
    for gamma in GAMMAS:
        lower = target - 3 * weighted.loc[gamma, "se_total"]
        print(f"    gamma={gamma:.0f}: {weighted.loc[gamma, 'coverage_mean']:.5f} >= {lower:.5f}"
              f"  {ok if v3_each[gamma] else ng}")

    # --- V4: 無限区間は起こらない ---
    max_infinite = float(trials["frac_infinite"].max())
    v4 = max_infinite == 0.0
    print(f"【V4】無限区間の割合がすべて 0 ... {ok if v4 else ng}（最大 {max_infinite:.6f}）")

    results = (v0, v1, v2, v3, v4)
    print("\n" + "-" * 78)
    print("判定: " + " / ".join(f"V{i} {ok if r else ng}" for i, r in enumerate(results))
          + f"  → {'全基準を満たす' if all(results) else '未達の基準あり（上記を参照）'}")
    print("-" * 78)


def main(seed: int) -> None:
    rng = np.random.default_rng(seed)

    # 乱数の引き順はモジュール冒頭のコメントのとおり。変えないこと
    model_seed = int(rng.integers(0, 2**31 - 1))
    X_train, y_train = heteroscedastic(N_TRAIN, rng)
    raw_test_sets = {gamma: heteroscedastic_shifted(N_TEST, rng, gamma) for gamma in GAMMAS}

    # 予測モデルは学習用データだけで 1 回学習し、全試行で固定する（既定値のまま。E3 と同じ方針）
    print(f"学習用データ {N_TRAIN:,} 点で予測モデルを学習中...")
    model = GradientBoostingRegressor(random_state=model_seed).fit(X_train, y_train)

    # テスト集合のスコアと重みはモデルが固定なので最初に 1 回だけ計算する
    test_sets = {
        gamma: {
            "scores": absolute_residual(y_test, model.predict(X_test)),
            "weights": shift_weight(X_test, gamma),
        }
        for gamma, (X_test, y_test) in raw_test_sets.items()
    }

    print(f"試行ループ: {N_TRIAL} 試行 × gamma {len(GAMMAS)} 水準 × {len(METHOD_KEYS)} 手法")
    trials = run_trials(rng, model, test_sets)
    summary = summarize(trials)

    RESULTS.mkdir(exist_ok=True)
    trials.to_csv(RESULTS / "e4_trials.csv", index=False)
    summary.to_csv(RESULTS / "e4_summary.csv", index=False)

    judge(trials, summary)
    for name in ("e4_trials.csv", "e4_summary.csv"):
        print(f"出力: {RESULTS / name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="実験E4: 共変量シフトと重み付き等角予測")
    ap.add_argument("--seed", type=int, default=20260901, help="乱数シード（再現性のため必須）")
    main(ap.parse_args().seed)
