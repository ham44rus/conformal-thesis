"""実験E2: 下敷きモデルへの非依存性。

対応箇所: 卒論 第5章 5.6節 E2 / 仕様書 specs/E2_model_agnostic.md

目的:
    被覆率は下敷きモデルによらず一定であり、区間の幅だけがモデルの精度に応じて変わる。
    この対比そのものを示す（等角予測の中心的な主張の実証）。

    実験の作りとして重要なのは以下の2点。いずれもモデル間の差だけを取り出すための工夫。
      - 学習集合とモデルは全試行で固定する（学習の変動を混ぜない）
      - 較正集合は5モデルで共有する（モデル間の差から較正のサンプリング変動を除く）

    主分析は学習集合1つに条件づいた結果なので、幅の順位が学習の運で決まっていないことを
    学習集合5通りの頑健性確認（C4）で別途調べる。

実行:
    python experiments/e2_model_agnostic.py --seed 20260901
出力:
    results/e2_model_agnostic.csv   試行ごとの被覆率と区間幅（モデル別）
    results/e2_summary.csv          モデルごとの要約と精度指標
    results/e2_robustness.csv       学習集合5通りでの幅の順位
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from conformal import absolute_residual, conformal_quantile
from conformal.datasets import homoscedastic

# --- 規模のパラメータ（仕様書「手続き」より）---
N_TRAIN, N_CAL, N_TEST = 500, 500, 200_000
N_TRIAL, ALPHA = 1000, 0.10

# 頑健性の確認（仕様書「頑健性の確認」より）
N_TRAIN_REP, N_TRIAL_ROBUST = 5, 200

# 下敷きモデル。**この順序は変えないこと。** random_state をこの順で rng から引くため、
# 順序を変えると同じ seed でも結果が変わり、再現性ルールが壊れる。
MODEL_KEYS = ("constant", "ridge", "rf", "gbm", "mlp")

# 表示用の和名（csv には英字キーの方を書く）
LABELS = {
    "constant": "定数予測器",
    "ridge": "リッジ回帰",
    "rf": "ランダムフォレスト",
    "gbm": "勾配ブースティング",
    "mlp": "ニューラルネット",
}

RESULTS = Path(__file__).resolve().parents[1] / "results"


def build_models(model_seeds: dict[str, int]) -> dict[str, object]:
    """未学習の下敷きモデル5種を、MODEL_KEYS の順に並べて返す（仕様書「下敷きモデル」）。

    `constant` は学習データの y の平均を常に返す最悪のモデル。
    これを含めるのは、「最悪のモデルでも被覆する」というモデル非依存性の主張が
    最も強く出るため（検証基準 C3）。

    Parameters
    ----------
    model_seeds : モデルキー -> random_state。内部乱数を持たない
        `constant` と `ridge` の分も引いてあるが、そちらでは使わない
    """
    return {
        "constant": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),  # E1 と同一のモデル
        "rf": RandomForestRegressor(n_estimators=200, random_state=model_seeds["rf"]),
        "gbm": GradientBoostingRegressor(random_state=model_seeds["gbm"]),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 64), max_iter=2000, random_state=model_seeds["mlp"]
        ),
    }


def fit_models(X_tr: np.ndarray, y_tr: np.ndarray, model_seeds: dict[str, int]) -> dict[str, object]:
    """5モデルすべてを**同じ学習集合**で学習する（仕様書「手続き」1）。"""
    return {key: model.fit(X_tr, y_tr) for key, model in build_models(model_seeds).items()}


def evaluate_on_test(
    models: dict[str, object], X_te: np.ndarray, y_te: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    r"""テスト集合での絶対残差（ソート済み）と、モデルの精度指標を返す。

    残差を先にソートしておくのは、被覆率を searchsorted で評価するため（仕様書「手続き」2）。
    テスト点は全モデルで同じものを使い、残差だけモデルごとに持つ。

    Returns
    -------
    sorted_res : モデルキー -> 昇順の絶対残差 |y - \hat{f}(x)|
    perf : モデルキー -> {test_rmse, test_q90_abs_resid}
    """
    sorted_res: dict[str, np.ndarray] = {}
    perf: dict[str, dict[str, float]] = {}
    for key, model in models.items():
        pred = model.predict(X_te)
        res = absolute_residual(y_te, pred)
        sorted_res[key] = np.sort(res)
        perf[key] = {
            "test_rmse": float(np.sqrt(np.mean((y_te - pred) ** 2))),
            # 絶対残差の 0.90 分位点。区間幅 2q の理論的な対応物にあたる。
            # これは共形分位点ではなくテスト集合の記述統計なので np.quantile を使ってよい
            # （CLAUDE.md「絶対に守ること1」が禁じているのは共形分位点の計算のこと）。
            "test_q90_abs_resid": float(np.quantile(res, 1 - ALPHA)),
        }
    return sorted_res, perf


def coverage_from_sorted(sorted_res: np.ndarray, q: float) -> float:
    r"""ソート済みテスト残差から被覆率 P(|Y - \hat{f}(X)| \le \hat{q}) を求める（卒論 式(5.1)）。

    `mean(res <= q)` と同値。テスト集合が20万点あるので searchsorted で数える。
    `side="right"` は「q 以下」を数えるため。`"left"` にすると「q 未満」になり、
    同点のテスト点を取りこぼす。
    """
    return float(np.searchsorted(sorted_res, q, side="right") / sorted_res.size)


def run_trials(
    models: dict[str, object],
    sorted_res: dict[str, np.ndarray],
    rng: np.random.Generator,
    n_trial: int,
) -> list[dict]:
    r"""試行ごとに較正集合を引き直し、モデル別の被覆率と区間幅を返す（仕様書「手続き」3）。

    較正集合は**全モデルで共有する**。モデルごとに引き直すと、モデル間の差に
    較正集合のサンプリング変動が上乗せされ、C1 の「モデル間の差」が見えなくなる。

    区間は C(x) = \hat{f}(x) \pm \hat{q} なので幅は 2\hat{q}（卒論 式(3.5)）。
    """
    rows = []
    for trial in range(n_trial):
        X_c, y_c = homoscedastic(N_CAL, rng)
        for key, model in models.items():
            scores = absolute_residual(y_c, model.predict(X_c))
            q = conformal_quantile(scores, ALPHA)  # k = ceil((n+1)(1-alpha)) の順序統計量
            rows.append(
                {
                    "model": key,
                    "trial": trial,
                    "coverage": coverage_from_sorted(sorted_res[key], q),
                    "width": 2.0 * q,
                }
            )
    return rows


def summarize(df: pd.DataFrame, perf: dict[str, dict[str, float]]) -> pd.DataFrame:
    """モデルごとの要約を作る（仕様書「出力」の e2_summary.csv）。

    標準誤差は**試行間**の標準偏差 / sqrt(R) で作る。
    ここで Clopper–Pearson を使わないのは、あれが1試行内でテスト点を二項標本とみなす区間で
    あるのに対し、E2 で問題になるのは較正集合の引き直しによる試行間のばらつきだから
    （仕様書「出力」の注意書き）。
    """
    rows = []
    for key in MODEL_KEYS:
        g = df[df["model"] == key]
        cov, width = g["coverage"].to_numpy(), g["width"].to_numpy()
        cov_se = float(cov.std(ddof=1) / np.sqrt(cov.size))
        rows.append(
            {
                "model": key,
                "coverage_mean": float(cov.mean()),
                "coverage_se": cov_se,
                "coverage_ci_lo": float(cov.mean() - 1.96 * cov_se),
                "coverage_ci_hi": float(cov.mean() + 1.96 * cov_se),
                "width_mean": float(width.mean()),
                "width_se": float(width.std(ddof=1) / np.sqrt(width.size)),
                "test_rmse": perf[key]["test_rmse"],
                "test_q90_abs_resid": perf[key]["test_q90_abs_resid"],
            }
        )
    return pd.DataFrame(rows)


def run_robustness(rng: np.random.Generator, X_te: np.ndarray, y_te: np.ndarray) -> pd.DataFrame:
    """学習集合と random_state を5通り振り、幅のモデル間順位が変わらないかを見る。

    主分析は学習集合1つに条件づいた結果なので、幅の順位が学習の運で決まっていないことを
    ここで確認する（検証基準 C4）。テスト点は主分析と同じものを使い回す
    （学習集合が変われば残差は変わるので、テスト点まで引き直す必要はない）。

    `width_rank` は幅の昇順で 1 = 最も狭い。
    """
    rows = []
    for train_rep in range(N_TRAIN_REP):
        # 主分析と同じ順序（先に random_state、次に学習集合）で rng から引く。
        # rng は1本の流れなので、同じ seed なら毎回同じ列が出る。
        rep_seeds = {key: int(rng.integers(0, 2**31 - 1)) for key in MODEL_KEYS}
        X_tr, y_tr = homoscedastic(N_TRAIN, rng)
        models = fit_models(X_tr, y_tr, rep_seeds)
        sorted_res, _ = evaluate_on_test(models, X_te, y_te)

        g = pd.DataFrame(run_trials(models, sorted_res, rng, N_TRIAL_ROBUST))
        agg = g.groupby("model", sort=False).agg(
            coverage_mean=("coverage", "mean"), width_mean=("width", "mean")
        )
        agg["width_rank"] = agg["width_mean"].rank(method="min").astype(int)
        for key in MODEL_KEYS:
            rows.append(
                {
                    "train_rep": train_rep,
                    "model": key,
                    "coverage_mean": float(agg.loc[key, "coverage_mean"]),
                    "width_mean": float(agg.loc[key, "width_mean"]),
                    "width_rank": int(agg.loc[key, "width_rank"]),
                }
            )
        print(f"  train_rep={train_rep} 完了")
    return pd.DataFrame(rows)


def judge(summary: pd.DataFrame, robust: pd.DataFrame) -> None:
    """検証基準 C1〜C4 を自動判定して表示する（仕様書「検証」）。"""
    ok = "○"
    ng = "×"
    lower, upper = 1 - ALPHA, 1 - ALPHA + 1 / (N_CAL + 1)
    band = 1 / (N_CAL + 1)  # 理論の上下界の幅。C1 のモデル間差の基準にもこれを使う

    s = summary.set_index("model").loc[list(MODEL_KEYS)]

    print("\n" + "=" * 78)
    print(f"実験E2 結果  n_cal={N_CAL}, alpha={ALPHA}, 試行={N_TRIAL}, テスト集合={N_TEST:,}点")
    print("=" * 78)
    print(f"{'model':10}{'被覆率':>10}{'±2SE':>9}{'幅':>9}{'幅の95%CI':>19}{'RMSE':>9}{'q90':>8}")
    for key in MODEL_KEYS:
        r = s.loc[key]
        w_lo, w_hi = r.width_mean - 1.96 * r.width_se, r.width_mean + 1.96 * r.width_se
        print(
            f"{key:10}{r.coverage_mean:>10.5f}{2 * r.coverage_se:>9.5f}{r.width_mean:>9.4f}"
            f"  [{w_lo:.4f}, {w_hi:.4f}]{r.test_rmse:>9.4f}{r.test_q90_abs_resid:>8.4f}"
        )
    print("  " + " / ".join(f"{k}={LABELS[k]}" for k in MODEL_KEYS))

    # --- C1: 被覆はモデルによらない ---
    in_band = {
        key: bool(lower - 2 * s.loc[key].coverage_se <= s.loc[key].coverage_mean
                  <= upper + 2 * s.loc[key].coverage_se)
        for key in MODEL_KEYS
    }
    max_gap = float(s["coverage_mean"].max() - s["coverage_mean"].min())
    c1 = all(in_band.values()) and max_gap < band
    print(f"\n【C1】被覆はモデルによらない ... {ok if c1 else ng}")
    print(f"  理論の上下界 [{lower:.5f}, {upper:.5f}]（±2SE の許容つきで判定）")
    for key in MODEL_KEYS:
        print(f"    {key:10} {s.loc[key].coverage_mean:.5f}  {ok if in_band[key] else ng}")
    print(f"  モデル間の平均被覆率の最大差 {max_gap:.5f} "
          f"{'<' if max_gap < band else '>='} 1/(n_cal+1) = {band:.5f}  "
          f"{ok if max_gap < band else ng}")

    # --- C2: 幅はモデルの精度に依存する ---
    rho_rmse = float(stats.spearmanr(s["test_rmse"], s["width_mean"]).correlation)
    rho_q90 = float(stats.spearmanr(s["test_q90_abs_resid"], s["width_mean"]).correlation)
    c2 = rho_rmse >= 1.0 - 1e-12
    print(f"\n【C2】幅はモデルの精度に依存する ... {ok if c2 else ng}")
    print(f"  Spearman(test_rmse, width_mean) = {rho_rmse:+.4f}  {ok if c2 else ng}（基準: 1.0）")
    print(f"  Spearman(test_q90_abs_resid, width_mean) = {rho_q90:+.4f}  "
          f"{ok if rho_q90 >= 1.0 - 1e-12 else ng}（参考。1.0 でなければ実装を疑う）")

    # 隣接順位のペアで幅の95%CIが重なるか。重なっても失敗ではなく、
    # 「そのペアは幅で区別できない」と報告する（仕様書 C2 の2点目）。
    order = s.sort_values("width_mean").index.tolist()
    print("  幅の昇順: " + " < ".join(order))
    overlaps = []
    for a, b in zip(order, order[1:]):
        hi_a = s.loc[a].width_mean + 1.96 * s.loc[a].width_se
        lo_b = s.loc[b].width_mean - 1.96 * s.loc[b].width_se
        sep = hi_a < lo_b
        print(f"    {a:10} vs {b:10} 95%CI {'分離' if sep else '重なり'}  {ok if sep else '△'}")
        if not sep:
            overlaps.append((a, b))
    if overlaps:
        print("  → 次のペアは幅で区別できない（第5章にそう書くこと）: "
              + ", ".join(f"{a}/{b}" for a, b in overlaps))

    # --- C3: 定数予測器でも被覆する ---
    cov_const = float(s.loc["constant"].coverage_mean)
    c3 = cov_const >= lower
    print(f"\n【C3】定数予測器でも被覆する ... {ok if c3 else ng}")
    print(f"  constant の平均被覆率 {cov_const:.5f} "
          f"{'>=' if c3 else '<'} 1-alpha = {lower:.5f}"
          f"{'' if c3 else '  → データリークを疑うこと'}")

    # --- C4: 頑健性 ---
    ranks = {
        rep: tuple(g.set_index("model").loc[list(MODEL_KEYS), "width_rank"])
        for rep, g in robust.groupby("train_rep")
    }
    c4 = len(set(ranks.values())) == 1
    print(f"\n【C4】幅の順位が学習集合によらない ... {ok if c4 else ng}")
    print(f"  {'train_rep':11}" + "".join(f"{k:>12}" for k in MODEL_KEYS))
    for rep in sorted(ranks):
        print(f"  {rep:<11}" + "".join(f"{r:>12}" for r in ranks[rep]))
    print("  ※ width_rank は幅の昇順で 1 = 最も狭い")
    if not c4:
        print("  → 幅の順位が学習集合に依存する。第5章に明記すること")

    print("\n" + "-" * 78)
    print(f"判定: C1 {ok if c1 else ng} / C2 {ok if c2 else ng} / "
          f"C3 {ok if c3 else ng} / C4 {ok if c4 else ng}"
          f"  → {'全基準を満たす' if all((c1, c2, c3, c4)) else '未達の基準あり（上記を参照）'}")
    print("-" * 78)


def main(seed: int) -> None:
    rng = np.random.default_rng(seed)

    # 内部乱数を持つモデルの random_state を seed から決定的に導出する（仕様書「乱数の扱い」）。
    # MODEL_KEYS の順に引くので、順序を変えると同じ seed でも結果が変わる。
    model_seeds = {key: int(rng.integers(0, 2**31 - 1)) for key in MODEL_KEYS}

    # 学習集合は1回だけ生成し、5モデルすべてで共有する（仕様書「手続き」1）
    X_tr, y_tr = homoscedastic(N_TRAIN, rng)
    models = fit_models(X_tr, y_tr, model_seeds)

    # テスト集合の絶対残差を先に計算・ソートしておく（仕様書「手続き」2）
    X_te, y_te = homoscedastic(N_TEST, rng)
    sorted_res, perf = evaluate_on_test(models, X_te, y_te)

    print(f"主分析: {N_TRIAL} 試行 × {len(MODEL_KEYS)} モデル")
    df = pd.DataFrame(run_trials(models, sorted_res, rng, N_TRIAL))
    summary = summarize(df, perf)

    print(f"頑健性の確認: 学習集合 {N_TRAIN_REP} 通り × {N_TRIAL_ROBUST} 試行")
    robust = run_robustness(rng, X_te, y_te)

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "e2_model_agnostic.csv", index=False)
    summary.to_csv(RESULTS / "e2_summary.csv", index=False)
    robust.to_csv(RESULTS / "e2_robustness.csv", index=False)

    judge(summary, robust)
    for name in ("e2_model_agnostic.csv", "e2_summary.csv", "e2_robustness.csv"):
        print(f"出力: {RESULTS / name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="実験E2: 下敷きモデルへの非依存性")
    ap.add_argument("--seed", type=int, default=20260901, help="乱数シード（再現性のため必須）")
    main(ap.parse_args().seed)
