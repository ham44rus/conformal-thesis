"""実験E3 の補足: CQR の区間が空になった点があるかを確定する。

対応箇所: 卒論 第4章 注意 rem:general-score の 3・注意 rem:cqr-negative（空区間）、
          第5章 E3 の節（sec:e3）/ 仕様書 specs/E3_conditional_coverage.md「空区間の確認」

CQR の区間 [q_lo(x) - q_hat, q_hi(x) + q_hat]（eq:cqr-interval）は q_hat < 0 のとき
q_hi(x) - q_lo(x) < -2 q_hat となる x で空になる。results/e3_marginal.csv では
CQR の q_hat が 500 試行中 2 試行で負（最小 -0.00554）なので、空区間が起きたかどうかは
テスト集合 200,000 点での min(q_hi - q_lo) を見ないと確定できない。

やること:
    - E3 と同じ seed・同じ引き順で分位点回帰モデルを再構成する（学習集合・テスト集合・
      モデルの乱数がすべて一致する）
    - 再構成が正しいことを、最初の数試行の較正集合を同じ引き順で引き直し、
      3 手法の q_hat が csv と一致することで確かめる（ずれたら止める）
    - テスト集合での min(q_hi - q_lo) を求め、2 * |min q_hat| と比べる
    - 結果を results/e3_empty_interval_check.csv に保存する（既存の csv は変更しない）

実行:
    python experiments/e3_empty_interval_check.py --seed 20260901
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_conditional import (  # noqa: E402  E3 本体と同じ定数・手順を使う
    ALPHA,
    MODEL_SEED_KEYS,
    N_CAL,
    N_TEST,
    N_TRAIN,
    fit_all,
)

from conformal import (  # noqa: E402
    absolute_residual,
    conformal_quantile,
    cqr_score,
    normalized_residual,
)
from conformal.datasets import heteroscedastic  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_CHECK_TRIALS = 3  # q_hat の再現を確かめる試行数


def main(seed: int) -> None:
    marg_path = RESULTS / "e3_marginal.csv"
    if not marg_path.exists():
        raise SystemExit("results/e3_marginal.csv がありません。先に `make e3` を実行してください。")
    marg = pd.read_csv(marg_path)

    # --- E3 と同じ引き順で再構成する（experiments/e3_conditional.py: main と同じ）---
    rng = np.random.default_rng(seed)
    model_seeds = {k: int(rng.integers(0, 2**31 - 1)) for k in MODEL_SEED_KEYS}
    X_tr, y_tr = heteroscedastic(N_TRAIN, rng)
    X_te, y_te = heteroscedastic(N_TEST, rng)
    f_hat, sigma_est, cqr_pair = fit_all(X_tr, y_tr, model_seeds)

    # --- 検査: 最初の数試行の q_hat が csv と一致すること ---
    max_dev = 0.0
    for trial in range(N_CHECK_TRIALS):
        X_c, y_c = heteroscedastic(N_CAL, rng)
        pred_c = f_hat.predict(X_c)
        qlo_c, qhi_c = cqr_pair.predict(X_c)
        q_hats = {
            "abs": conformal_quantile(absolute_residual(y_c, pred_c), ALPHA),
            "norm": conformal_quantile(normalized_residual(y_c, pred_c, sigma_est.predict(X_c)), ALPHA),
            "cqr": conformal_quantile(cqr_score(y_c, qlo_c, qhi_c), ALPHA),
        }
        for key, q in q_hats.items():
            q_csv = float(marg[(marg.method == key) & (marg.trial == trial)]["q_hat"].iloc[0])
            max_dev = max(max_dev, abs(q - q_csv))
    if max_dev > 1e-9:
        raise SystemExit(f"q_hat が csv と一致しない（最大差 {max_dev:.3e}）。seed か引き順が E3 と違う")
    print(f"最初の {N_CHECK_TRIALS} 試行の q_hat（3 手法）を再現（最大差 {max_dev:.1e}）")

    # --- テスト集合での分位点回帰の幅 ---
    qlo_te, qhi_te = cqr_pair.predict(X_te)   # 交差は入れ替え済み（eq:cqr-pair）
    width_raw = qhi_te - qlo_te
    i_min = int(np.argmin(width_raw))
    q_hat_min = float(marg[marg.method == "cqr"]["q_hat"].min())
    threshold = 2.0 * abs(min(q_hat_min, 0.0))   # 空になりうる幅の上限 -2 q_hat
    n_below = int(np.sum(width_raw < threshold))

    out = pd.DataFrame(
        [
            {
                "n_test": N_TEST,
                "min_width_raw": float(width_raw[i_min]),
                "x1_at_min": float(X_te[i_min, 0]),
                "q_hat_min_cqr": q_hat_min,
                "n_trials_q_hat_negative": int((marg[marg.method == "cqr"]["q_hat"] < 0).sum()),
                "threshold_2abs_q_hat_min": threshold,
                "n_test_points_width_below_threshold": n_below,
                "empty_interval_occurred": bool(n_below > 0),
                "crossing_rate_test": cqr_pair.crossing_rate_on(X_te),
                "q_hat_reproduction_max_dev": max_dev,
            }
        ]
    )
    out.to_csv(RESULTS / "e3_empty_interval_check.csv", index=False)
    print(f"出力: {RESULTS / 'e3_empty_interval_check.csv'}")
    print(out.T.to_string(header=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="実験E3 の補足: CQR の空区間の有無を確定する")
    ap.add_argument("--seed", type=int, default=20260901, help="E3 と同じ seed")
    main(ap.parse_args().seed)
