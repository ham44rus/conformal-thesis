"""実験E1b: 検証の水準と、誤りの検出力。

対応箇所: 卒論 第5章 5.5節 E1b / 仕様書 specs/E1b_detection_levels.md

目的:
    被覆保証の検証には水準がある。
      L0 点推定を眺めるだけ
      L1 下界 1-alpha を満たすか
      L2 上界 1-alpha+1/(n+1) も満たすか
      L3 被覆率の分布が Beta(n+1-l, l) と一致するか
    どの水準ならどの誤りを検出できるのかを、2種類の誤りで実演する。

    (A) off-by-one: k = ceil(n(1-alpha)) にする典型的な実装バグ  -> 位置がずれる
    (B) 分散のずれ: 実装は正しいがテスト集合が小さい            -> 形がずれる

    (A) は平均の検定でも捉えられるが、(B) は平均では捉えられず分布でしか捉えられない。
    「位置のずれは平均が、形のずれは分布が捕まえる」ことを示すのがこの実験の主張。

実行:
    python experiments/e1b_detection_levels.py --seed 7
出力:
    results/e1b_detection_levels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from conformal import absolute_residual
from conformal.datasets import homoscedastic

N_CAL, ALPHA, N_TRIAL = 1000, 0.10, 2000
N_POOL_CHUNKS, CHUNK = 8, 500_000   # 被覆率の真値を出すための巨大テスト集合（400万点）
N_SMALL_TEST = 2_000                # ケース(B)で使う小さいテスト集合
RESULTS = Path(__file__).resolve().parents[1] / "results"


def q_correct(scores: np.ndarray, alpha: float) -> float:
    """正しい共形分位点: k = ceil((n+1)(1-alpha))。"""
    return float(np.sort(scores)[int(np.ceil((len(scores) + 1) * (1 - alpha))) - 1])


def q_offbyone(scores: np.ndarray, alpha: float) -> float:
    """誤実装: k = ceil(n(1-alpha))。n+1 を n にしてしまう典型的なバグ。"""
    return float(np.sort(scores)[int(np.ceil(len(scores) * (1 - alpha))) - 1])


def main(seed: int) -> None:
    rng = np.random.default_rng(seed)
    X, y = homoscedastic(500, rng)
    model = Ridge(alpha=1.0).fit(X, y)

    # 被覆率の「真値」を評価するための巨大テスト集合。
    # 小さいテスト集合だと、その集合固有のズレが全試行に共通のバイアスとして乗るため、
    # 正しい実装まで棄却されてしまう（この現象自体がケース(B)の主題でもある）。
    pool = np.sort(np.concatenate([
        absolute_residual(*(lambda Xt, yt: (yt, model.predict(Xt)))(*homoscedastic(CHUNK, rng)))
        for _ in range(N_POOL_CHUNKS)
    ]))
    n_pool = pool.size

    def true_coverage(q: float) -> float:
        """F(q) を巨大テスト集合の経験分布で評価する（searchsorted で高速）。"""
        return float(np.searchsorted(pool, q, side="right") / n_pool)

    rows = []
    for trial in range(N_TRIAL):
        Xc, yc = homoscedastic(N_CAL, rng)
        s = absolute_residual(yc, model.predict(Xc))

        # (A) 正しい実装 と off-by-one。いずれも被覆率は真値で評価する
        rows.append({"case": "correct", "trial": trial, "coverage": true_coverage(q_correct(s, ALPHA))})
        rows.append({"case": "offbyone", "trial": trial, "coverage": true_coverage(q_offbyone(s, ALPHA))})

        # (B) 実装は正しいが、被覆率を小さいテスト集合で推定した場合
        Xs, ys = homoscedastic(N_SMALL_TEST, rng)
        cov_small = float(np.mean(absolute_residual(ys, model.predict(Xs)) <= q_correct(s, ALPHA)))
        rows.append({"case": "small_test", "trial": trial, "coverage": cov_small})

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "e1b_detection_levels.csv", index=False)

    # --- 4つの水準で判定する ---
    l = int(np.floor((N_CAL + 1) * ALPHA))
    a_par, b_par = N_CAL + 1 - l, l
    theo_mean = a_par / (a_par + b_par)
    theo_sd = np.sqrt(a_par * b_par / ((a_par + b_par) ** 2 * (a_par + b_par + 1)))
    se = theo_sd / np.sqrt(N_TRIAL)
    lower, upper = 1 - ALPHA, 1 - ALPHA + 1 / (N_CAL + 1)

    label = {"correct": "正しい実装", "offbyone": "誤実装 (n+1 を n に)", "small_test": "テスト集合が小さい"}
    print(f"n={N_CAL}, alpha={ALPHA}, 試行={N_TRIAL}, テスト集合={n_pool:,}点")
    print(f"理論分布 Beta({a_par},{b_par})  平均 {theo_mean:.5f}  標準偏差 {theo_sd:.5f}")
    print(f"理論の上下界 [{lower:.5f}, {upper:.5f}]（幅 {upper-lower:.5f}）\n")
    print(f"{'':22}{'平均':>9}{'標準偏差':>10}{'L1下界':>8}{'L2上下界':>10}{'平均のz':>9}{'L3 KS p':>12}")
    for case in ("correct", "offbyone", "small_test"):
        v = df[df.case == case]["coverage"].to_numpy()
        z = (v.mean() - theo_mean) / se
        ks = stats.kstest(v, "beta", args=(a_par, b_par)).pvalue
        print(f"{label[case]:22}{v.mean():>9.5f}{v.std():>10.5f}"
              f"{'○' if v.mean() >= lower else '×':>8}"
              f"{'○' if lower <= v.mean() <= upper else '×':>10}"
              f"{z:>+9.2f}{ks:>12.2e}")
    print("\n※ z は理論平均との比較。|z|<2 なら平均の検定では異常を検出できない。")
    print(f"出力: {RESULTS / 'e1b_detection_levels.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="実験E1b: 検証の水準と誤りの検出力")
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args().seed)
