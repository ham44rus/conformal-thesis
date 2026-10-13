"""重み付き等角予測（共変量シフトの下での分割等角予測）の共形分位点。

対応箇所:
    卒論 第4章 共変量シフトの節（sec:covariate-shift）
    実験 E4（sec:e4。交換可能性が破れる場合）
    仕様書 specs/E4_covariate_shift.md
出典: Tibshirani, Barber, Candès, Ramdas (2019)（refs.bib の tibshirani2019covariate）

較正用データとテスト点で入力 X の分布がずれると、スコアは交換可能でなくなり、
通常の共形分位点（split.py: conformal_quantile、卒論 eq:khat）の保証は成り立たない。
ずれが尤度比 w(x) = (テスト側の密度) / (較正側の密度) で分かっているときは、
較正スコアを w で重み付けした分位点を使うと 1-alpha の被覆が戻る。
"""

from __future__ import annotations

import numpy as np

__all__ = ["weighted_conformal_quantile"]

# 累積の重みと閾値を比べるときの相対許容幅（理由は下の docstring「丸めについて」）
RELATIVE_TOLERANCE = 1e-12


def weighted_conformal_quantile(
    cal_scores: np.ndarray,
    cal_weights: np.ndarray,
    test_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    r"""テスト点ごとの重み付き共形分位点 \hat{q}(x) を返す（卒論 sec:covariate-shift）。

    定義。較正スコア S_1..S_n、較正点の重み w_i = w(X_i)、テスト点の重み w(x) に対し、

        p_i(x)    = w_i  / (\sum_j w_j + w(x))        (i = 1..n)
        p_inf(x)  = w(x) / (\sum_j w_j + w(x))        （+inf に置く重み）
        \hat{q}(x) = \min\{ s : \sum_{i: S_i \le s} p_i(x) \ge 1-\alpha \}

    そのような s が無ければ（較正点の重みを全部足しても 1-\alpha に届かなければ）
    \hat{q}(x) = +\infty とする。テスト点自身の重み p_inf を +inf に置くのは、
    テスト点のスコアがまだ分からないので最悪の場合を取るためである。
    区間は [\hat{f}(x) - \hat{q}(x), \hat{f}(x) + \hat{q}(x)]。

    重みがすべて 1 のときは p_i = 1/(n+1) で、条件は「S_i \le s の個数 \ge (n+1)(1-\alpha)」
    になるので、通常の共形分位点 S_{(k)}, k = \lceil (n+1)(1-\alpha) \rceil（卒論 eq:khat）に
    一致する。重みは比だけが効く（全体を定数倍しても p_i は変わらない）ので、
    正規化は p_i の分母で済んでおり、呼び出し側で正規化する必要はない。

    計算。両辺に (\sum_j w_j + w(x)) を掛けると、条件は

        （スコアの小さい順に重みを足した累積和） \ge (1-\alpha)(\sum_j w_j + w(x))

    なので、累積和が右辺に初めて届く位置のスコアを返せばよい。

    丸めについて。重みは実数なので split.py: conformal_index のような有理数の厳密な
    計算はできない。float では、たとえば \alpha=1/3, n=8 で右辺 9*(1-1/3) が
    6.000000000000001 になり、累積和 6 が「届いていない」と判定されて順位が 1 つずれる。
    これを避けるため、右辺を相対幅 1e-12 だけ小さくして比べる。この簡略化で
    被覆確率が動くのは高々この相対幅の程度である。

    Parameters
    ----------
    cal_scores : 形状 (n,) の較正スコア
    cal_weights : 形状 (n,) の較正点の重み。非負で有限
    test_weights : 形状 (n_test,) のテスト点の重み（スカラーでもよい）。非負で有限
    alpha : 有意水準。0 < alpha < 1

    Returns
    -------
    形状 (n_test,) の配列。テスト点ごとの \hat{q}(x)。np.inf を含みうる

    Notes
    -----
    np.quantile は使わない（CLAUDE.md「絶対に守ること 1」）。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha は (0,1) でなければならない: {alpha}")
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_weights = np.asarray(cal_weights, dtype=float)
    test_weights = np.atleast_1d(np.asarray(test_weights, dtype=float))
    if cal_scores.ndim != 1 or cal_weights.shape != cal_scores.shape:
        raise ValueError(
            "cal_scores と cal_weights は同じ長さの1次元配列でなければならない: "
            f"shape={cal_scores.shape}, {cal_weights.shape}"
        )
    if test_weights.ndim != 1:
        raise ValueError(f"test_weights は1次元配列でなければならない: shape={test_weights.shape}")
    for name, weights in (("cal_weights", cal_weights), ("test_weights", test_weights)):
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError(f"{name} は非負で有限でなければならない")

    # 重みは比だけが効くので、最大の重みで割ってから足す。個々の重みが有限でも、
    # 非常に大きいと合計が float の上限を超えて inf になり、分位点を誤るため
    largest_weight = max(cal_weights.max(initial=0.0), test_weights.max(initial=0.0))
    if largest_weight > 0:
        cal_weights = cal_weights / largest_weight
        test_weights = test_weights / largest_weight

    total_cal_weight = float(cal_weights.sum())  # W = sum_j w_j
    if np.any(total_cal_weight + test_weights <= 0):
        raise ValueError("重みの合計 sum_j w_j + w(x) は正でなければならない")

    # スコアの小さい順に並べ、重みの累積和を作る
    order = np.argsort(cal_scores, kind="stable")
    sorted_scores = cal_scores[order]
    cumulative_weights = np.cumsum(cal_weights[order])

    # テスト点ごとの閾値 (1-alpha)(W + w(x))
    # ponytail: 許容幅 1e-12。被覆への影響はこの相対幅以下（厳密にするなら重みを有理数で持つ）
    thresholds = (1.0 - alpha) * (total_cal_weight + test_weights) * (1.0 - RELATIVE_TOLERANCE)

    # 累積和が閾値に初めて届く位置（side="left" は cum_j >= 閾値 を満たす最小の j）。
    # どこでも届かなければ位置は n になるので、末尾に +inf を足した配列から引く
    positions = np.searchsorted(cumulative_weights, thresholds, side="left")
    return np.append(sorted_scores, np.inf)[positions]
