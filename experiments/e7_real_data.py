"""実験E7: 実データ解析（UCI の公開データ 3 件）。

対応箇所:
    卒論 第6章 実データ解析（sec:real-data）
    仕様書 specs/E7_real_data.md
    データの入手方法 docs/データの入手方法.md

UCI の公開データ 3 件（Airfoil Self-Noise、Superconductivity、Bike Sharing）で、
分割等角予測の 3 手法（(a) 絶対残差 eq:score、(b) 正規化残差 eq:score-normalized、
(c) CQR eq:cqr-score）の周辺被覆率・層別被覆率・平均の幅を測る。
人工データ（E2・E3、sec:e3）で見た性質が実データでも再現するかを見る。

3 手法の作り方は E3 と同じにするため、experiments/e3_conditional.py の
fit_all（下敷きモデルの学習）と intervals（区間の組み方）をそのまま使う。

主分析：データ集合を固定し、行をランダムに並べ替えて学習用 50%・較正用 25%・テスト用 25% に分ける。
これを R=100 回くり返す。並べ替えで分けた較正用とテスト点は交換可能なので、
定理 thm:coverage の周辺被覆保証が（分割の乱数について）成り立つ。

補足分析：Bike Sharing を時間の順（instant）に分ける（1 回だけ）。交換可能性が成り立たないかもしれない例
（rem:exch-violation）として報告する。合否の基準にはしない。

チェリーピッキングをしない：データ・前処理・分割・層・検証基準は結果を見る前に仕様書で固定した。

実行:
    python experiments/e7_real_data.py --seed 20260901
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_conditional import (  # noqa: E402  E3 本体と同じ手法の作り方を使う
    ALPHA,
    LABELS,
    METHOD_KEYS,
    MODEL_SEED_KEYS,
    fit_all,
    intervals,
)

from conformal import (  # noqa: E402
    absolute_residual,
    bin_edges_from_quantiles,
    clopper_pearson,
    conformal_index,
    conformal_quantile,
    cqr_score,
    feature_stratified_coverage,
    normalized_residual,
)

# --- 規模のパラメータ（仕様書「規模のパラメータ」）---
N_TRIAL = 100
N_BINS = 5
assert ALPHA == 0.10  # 仕様書の alpha。E3 の定数を共有しているので念のため確かめる

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = ROOT / "data" / "raw"
GUIDE = "docs/データの入手方法.md"

# データの順。この順序は変えないこと（データごとの乱数列の割り当てに対応する）
DATASETS = ("airfoil", "superconductivity", "bike")
DATA_LABELS = {
    "airfoil": "Airfoil Self-Noise",
    "superconductivity": "Superconductivity",
    "bike": "Bike Sharing",
}
ZIP_PATHS = {
    "airfoil": RAW / "airfoil" / "airfoil.zip",
    "superconductivity": RAW / "superconductivity" / "superconductivity.zip",
    "bike": RAW / "bike" / "bike.zip",
}
# 入手時（2026-10-17）の zip の sha256（仕様書「データ」・docs/データの入手方法.md）
SHA256 = {
    "airfoil": "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c",
    "superconductivity": "87f4490d73390ff94ee01dbf0d7d32abc80b22f2c803d471765cfc46a9f6371e",
    "bike": "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401",
}
# Bike Sharing の説明変数（12 列、数値のまま）。
# instant, dteday, casual, registered は除く。casual + registered = cnt なので入れるとリークになる
BIKE_FEATURES = ("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
                 "weathersit", "temp", "atemp", "hum", "windspeed")

BIN_COLS = [f"cov_bin{b + 1}" for b in range(N_BINS)]


def check_sha256(name: str) -> None:
    """zip の sha256 を入手時の値と照合する（検証基準 V2）。違えば入手方法を示して止まる。"""
    path = ZIP_PATHS[name]
    if not path.exists():
        raise SystemExit(f"{path} がありません。{GUIDE} の手順でダウンロードしてください。")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        raise SystemExit(
            f"{path} の sha256 が一致しません（{digest}）。期待値は {SHA256[name]}。\n"
            f"{GUIDE} の手順で入手し直してください。"
        )


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """zip から直接読み、(説明変数, 目的変数, 元の表) を返す（仕様書「データの読み込みと前処理」）。

    展開はしない。欠損値があれば止まる（仕様書では PM にエスカレーションする扱い）。
    """
    check_sha256(name)
    with zipfile.ZipFile(ZIP_PATHS[name]) as zf:
        if name == "airfoil":
            # タブ区切り・ヘッダーなし。1〜5 列目が説明変数、6 列目が音圧レベル（dB）
            df = pd.read_csv(zf.open("airfoil_self_noise.dat"), sep="\t", header=None)
            X, y = df.iloc[:, :5], df.iloc[:, 5]
        elif name == "superconductivity":
            # train.csv の critical_temp が目的変数、残り 81 列が説明変数。重複行も除かない
            df = pd.read_csv(zf.open("train.csv"))
            X, y = df.drop(columns="critical_temp"), df["critical_temp"]
        elif name == "bike":
            df = pd.read_csv(zf.open("hour.csv"))
            X, y = df[list(BIKE_FEATURES)], df["cnt"]
        else:
            raise ValueError(f"未知のデータ: {name}")
    if df.isna().any().any():
        raise SystemExit(f"{name} に欠損値があります。仕様書の想定外なので PM に確認すること。")
    return X.to_numpy(dtype=float), y.to_numpy(dtype=float), df


def split_indices(order: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """行の並び order を学習用 50%・較正用 25%・テスト用 25% に分ける。

    np.array_split で 4 つに分け、最初の 2 つを学習用、3 つ目を較正用、4 つ目をテスト用にする
    （端数は前の塊に 1 点ずつ寄る）。3 つが重ならないことを assert で確かめる（検証基準 V2）。
    """
    parts = np.array_split(order, 4)
    tr, cal, te = np.concatenate(parts[:2]), parts[2], parts[3]
    assert len(np.intersect1d(tr, cal)) == 0
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.intersect1d(cal, te)) == 0
    assert len(tr) + len(cal) + len(te) == len(order)
    return tr, cal, te


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    cal: np.ndarray,
    te: np.ndarray,
    model_seeds: dict[str, int],
) -> list[dict]:
    """1 回の分割で 3 手法を学習・較正・評価し、手法ごとの記録を返す。

    共形分位点は conformal_quantile（eq:khat、k = ceil((n_cal+1)(1-alpha))）。
    層は予測値 f_hat(x) の 5 分位。境界は**較正用データ**の予測値から作り、テスト点を層に分ける。
    層は f_hat だけで決まるので 3 手法で共通である。
    """
    f_hat, sigma_est, cqr_pair = fit_all(X[tr], y[tr], model_seeds)

    X_c, y_c, X_te, y_te = X[cal], y[cal], X[te], y[te]
    pred_c, pred_te = f_hat.predict(X_c), f_hat.predict(X_te)
    sigma_c, sigma_te = sigma_est.predict(X_c), sigma_est.predict(X_te)
    (qlo_c, qhi_c), (qlo_te, qhi_te) = cqr_pair.predict(X_c), cqr_pair.predict(X_te)

    q_hats = {
        "abs": conformal_quantile(absolute_residual(y_c, pred_c), ALPHA),
        "norm": conformal_quantile(normalized_residual(y_c, pred_c, sigma_c), ALPHA),
        "cqr": conformal_quantile(cqr_score(y_c, qlo_c, qhi_c), ALPHA),
    }
    k = conformal_index(len(cal), ALPHA)

    # 層の境界：較正用の予測値の 5 分位（記述統計としての分位点。共形分位点ではない。
    # bin_edges_from_quantiles の中で np.quantile を使う）。最上端は +inf になっている。
    # 最下端も -inf にし、較正用の最小値より小さい予測値のテスト点も層 1 に入れる
    edges = bin_edges_from_quantiles(pred_c, N_BINS)
    edges[0] = -np.inf

    rows = []
    for key in METHOD_KEYS:
        lo, hi = intervals(key, q_hats[key], pred_te, sigma_te, qlo_te, qhi_te)
        covered = (y_te >= lo) & (y_te <= hi)
        strat = {r["bin"]: r["coverage"]
                 for r in feature_stratified_coverage(pred_te, y_te, lo, hi, edges=edges)}
        # テスト点が 1 つも入らない層は NaN（0 にすると「被覆しなかった」と読めてしまう）
        cov_bins = [strat.get(b, np.nan) for b in range(N_BINS)]
        rows.append(
            {
                "method": key,
                "n_train": len(tr),
                "n_cal": len(cal),
                "n_test": len(te),
                "k": k,
                "q_hat": float(q_hats[key]),
                "n_covered": int(covered.sum()),
                "coverage": float(covered.mean()),
                "mean_width": float(np.mean(hi - lo)),
                # 層別被覆率の幅：層の被覆率の最大 − 最小（空の層は除く）
                "strata_range": float(np.nanmax(cov_bins) - np.nanmin(cov_bins)),
                **dict(zip(BIN_COLS, cov_bins)),
            }
        )
    return rows


def draw_model_seeds(rng: np.random.Generator) -> dict[str, int]:
    """モデルの random_state を E3 と同じ順（MODEL_SEED_KEYS）で rng から引く。"""
    return {key: int(rng.integers(0, 2**31 - 1)) for key in MODEL_SEED_KEYS}


def run_random_splits(name: str, rng: np.random.Generator) -> pd.DataFrame:
    """主分析：ランダム分割を N_TRIAL 回くり返す（仕様書「主分析」）。

    試行ごとに、(1) モデルの random_state 4 つ、(2) 行の並べ替え、の順に rng から引く
    （E3 と同じく、データより先にモデルの seed を引く）。
    """
    X, y, _ = load_dataset(name)
    rows = []
    start = time.perf_counter()
    for trial in range(1, N_TRIAL + 1):
        model_seeds = draw_model_seeds(rng)
        tr, cal, te = split_indices(rng.permutation(len(y)))
        for r in evaluate(X, y, tr, cal, te, model_seeds):
            rows.append({"dataset": name, "trial": trial, **r})
        if trial == 1 or trial % 10 == 0:
            print(f"  {DATA_LABELS[name]}: {trial} / {N_TRIAL} 試行（{time.perf_counter() - start:.0f} 秒）",
                  flush=True)
    return pd.DataFrame(rows)


def run_time_split(rng: np.random.Generator) -> pd.DataFrame:
    """補足分析：Bike Sharing の時間順の分割（1 回だけ。仕様書「補足分析」）。

    instant（時間の順番）で並べ、最初の 50% を学習用、次の 25% を較正用、最後の 25% をテスト用にする。
    分割には乱数を使わない。モデルの random_state だけを rng から引く。
    被覆率の区間は 1 回の適用なので Clopper–Pearson（CLAUDE.md「報告の作法」の 1 試行内の行）。
    """
    X, y, df = load_dataset("bike")
    order = np.argsort(df["instant"].to_numpy(), kind="stable")
    tr, cal, te = split_indices(order)
    assert df["instant"].to_numpy()[tr].max() < df["instant"].to_numpy()[cal].min()
    assert df["instant"].to_numpy()[cal].max() < df["instant"].to_numpy()[te].min()
    rows = []
    for r in evaluate(X, y, tr, cal, te, draw_model_seeds(rng)):
        cp_lo, cp_hi = clopper_pearson(r["n_covered"], r["n_test"])
        rows.append({"method": r["method"], "n_train": r["n_train"], "n_cal": r["n_cal"],
                     "n_test": r["n_test"], "k": r["k"], "coverage": r["coverage"],
                     "cp_lo": cp_lo, "cp_hi": cp_hi, "mean_width": r["mean_width"],
                     **{c: r[c] for c in BIN_COLS}})
    return pd.DataFrame(rows)


def summarize(trials: pd.DataFrame, n_totals: dict[str, int]) -> pd.DataFrame:
    """データ×手法ごとの要約（仕様書「要約」）。

    テスト集合は試行ごとに引き直すので、全試行に共通のテスト集合バイアスは無い。
    SE は試行間 SD だけから作る：se = SD / sqrt(R)、95% 区間は ±1.96 se
    （CLAUDE.md「報告の作法」の試行間の行）。
    理論の範囲は [1-alpha, k/(n_cal+1)]（thm:coverage と上界 cor:upper）。
    """
    rows = []
    for name in DATASETS:
        for key in METHOD_KEYS:
            g = trials[(trials["dataset"] == name) & (trials["method"] == key)]
            cov = g["coverage"].to_numpy()
            n_cal, k = int(g["n_cal"].iloc[0]), int(g["k"].iloc[0])
            m, sd = float(cov.mean()), float(cov.std(ddof=1))
            se = sd / np.sqrt(cov.size)
            rows.append(
                {
                    "dataset": name,
                    "method": key,
                    "n_total": n_totals[name],
                    "n_train": int(g["n_train"].iloc[0]),
                    "n_cal": n_cal,
                    "n_test": int(g["n_test"].iloc[0]),
                    "k": k,
                    "alpha": ALPHA,
                    "n_trial": int(cov.size),
                    "coverage_mean": m,
                    "coverage_sd": sd,
                    "se": se,
                    "ci_lo": m - 1.96 * se,
                    "ci_hi": m + 1.96 * se,
                    "theory_lo": 1 - ALPHA,
                    "theory_hi": k / (n_cal + 1),
                    "width_mean": float(g["mean_width"].mean()),
                    "width_sd": float(g["mean_width"].std(ddof=1)),
                    "strata_range_mean": float(g["strata_range"].mean()),
                    "strata_range_sd": float(g["strata_range"].std(ddof=1)),
                    # 空の層（NaN）は平均から除く。空の層の数は画面に出す
                    **{c: float(g[c].mean()) for c in BIN_COLS},
                }
            )
    return pd.DataFrame(rows)


def report(summary: pd.DataFrame, trials: pd.DataFrame, timesplit: pd.DataFrame) -> None:
    """要約と検証基準の合否を画面に出す（仕様書「検証基準」）。基準は結果を見る前に決めたもの。"""
    ok, ng = "○", "×"
    print("\n" + "=" * 78)
    print(f"実験E7 結果  alpha={ALPHA}, 試行={N_TRIAL}（ランダム分割 50:25:25）")
    print("=" * 78)
    print("SE は試行間 SD / sqrt(R)（テスト集合は試行ごとに引き直すので共通バイアスなし）")

    v1_all = True
    for name in DATASETS:
        s = summary[summary["dataset"] == name]
        r0 = s.iloc[0]
        print(f"\n【{DATA_LABELS[name]}】n={r0['n_total']:,}（学習 {r0['n_train']:,}・較正 {r0['n_cal']:,}"
              f"・テスト {r0['n_test']:,}）、k={r0['k']}、理論の範囲 [{r0['theory_lo']:.5f}, {r0['theory_hi']:.5f}]")
        print(f"  {'手法':10}{'被覆率':>9}{'se':>9}{'下限-3se':>10}{'V1':>4}{'上限比較':>9}"
              f"{'平均幅':>10}{'層の幅':>9}  層別被覆率（層1〜5）")
        for _, r in s.iterrows():
            v1 = bool(r["coverage_mean"] >= 1 - ALPHA - 3 * r["se"])
            v1_all &= v1
            above = "上回る" if r["coverage_mean"] > r["theory_hi"] else "以下"
            bins = " ".join(f"{r[c]:.4f}" for c in BIN_COLS)
            print(f"  {LABELS[r['method']]:10}{r['coverage_mean']:>9.5f}{r['se']:>9.5f}"
                  f"{1 - ALPHA - 3 * r['se']:>10.5f}{ok if v1 else ng:>4}{above:>9}"
                  f"{r['width_mean']:>10.4f}{r['strata_range_mean']:>9.4f}  {bins}")

    n_empty = int(trials[BIN_COLS].isna().sum().sum())
    print(f"\n空の層（テスト点が 0）の数：{n_empty}（全 {len(trials) * N_BINS} 層中）")

    print("\n【補足】Bike Sharing の時間順分割（1 回。区間は Clopper–Pearson 95%。合否の基準にしない）")
    for _, r in timesplit.iterrows():
        bins = " ".join(f"{r[c]:.4f}" for c in BIN_COLS)
        print(f"  {LABELS[r['method']]:10}被覆率 {r['coverage']:.5f} [{r['cp_lo']:.5f}, {r['cp_hi']:.5f}]"
              f"  平均幅 {r['mean_width']:.4f}  層別 {bins}")

    print("\n" + "-" * 78)
    print(f"V1（周辺被覆の下限 coverage_mean >= 1-alpha-3se）: {ok if v1_all else ng}")
    print(f"V2（分割が重ならない・sha256 一致）: {ok}（assert と照合を通過）")
    print("-" * 78)


def main(seed: int) -> None:
    start = time.perf_counter()
    # 乱数の導出（仕様書「乱数の導出」）：seed から 4 本の独立な乱数列を作り、
    # Airfoil・Superconductivity・Bike のランダム分割と、Bike の時間順分割に 1 本ずつ割り当てる。
    # データごとに乱数列を分けるので、あるデータの処理が他のデータの乱数に影響しない
    rngs = np.random.default_rng(seed).spawn(len(DATASETS) + 1)

    frames, n_totals = [], {}
    for name, rng in zip(DATASETS, rngs):
        t0 = time.perf_counter()
        df = run_random_splits(name, rng)
        n_totals[name] = int(df["n_train"].iloc[0] + df["n_cal"].iloc[0] + df["n_test"].iloc[0])
        frames.append(df)
        print(f"  {DATA_LABELS[name]} 完了（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    trials = pd.concat(frames, ignore_index=True)
    summary = summarize(trials, n_totals)
    timesplit = run_time_split(rngs[-1])

    RESULTS.mkdir(exist_ok=True)
    out_cols = ["dataset", "trial", "method", "n_train", "n_cal", "n_test", "k", "q_hat",
                "coverage", "mean_width", "strata_range", *BIN_COLS]
    trials[out_cols].to_csv(RESULTS / "e7_trials.csv", index=False)
    summary.to_csv(RESULTS / "e7_summary.csv", index=False)
    timesplit.to_csv(RESULTS / "e7_timesplit.csv", index=False)

    report(summary, trials, timesplit)
    for fname in ("e7_trials.csv", "e7_summary.csv", "e7_timesplit.csv"):
        print(f"出力: {RESULTS / fname}")
    # 実行時間は csv に入れない（同じ seed で csv が一致しなくなるため）
    print(f"総実行時間: {time.perf_counter() - start:.0f} 秒")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="実験E7: 実データ解析（UCI の公開データ 3 件）")
    ap.add_argument("--seed", type=int, default=20260901, help="乱数シード（再現性のため必須）")
    main(ap.parse_args().seed)
