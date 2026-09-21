"""分割等角予測（split conformal prediction）の中核実装。

対応箇所:
    卒論 第3章「分割等角予測の理論」
    実験 E1（被覆保証の検証）, E2（モデル非依存性）, E5（分割比）

このモジュールは論文の心臓部にあたる。読んで理解できる長さを保つこと。
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

__all__ = [
    "conformal_index",
    "conformal_quantile",
    "conformal_pvalue",
    "interval_from_quantile",
]


def conformal_index(n: int, alpha: float) -> int:
    r"""共形分位点の順位 k = \lceil (n+1)(1-\alpha) \rceil を厳密に返す（卒論 eq:khat）。

    (n+1)*(1-alpha) を float で計算すると丸め誤差で k が 1 ずれることがある。
    例: alpha=1/3, n=8 では 9*(1-1/3) が 6.000000000000001 になり ceil で 7 になる
    （正しくは 6）。そこで alpha を「分母が 10^9 以下の有理数」として復元し、
    整数・有理数の演算で天井をとる。float(1/3) は 1/3 に戻り、小数点以下 9 桁までの
    alpha は正確に戻る。

    Parameters
    ----------
    n : キャリブレーション集合のサイズ
    alpha : 有意水準。0 < alpha < 1

    Returns
    -------
    int : k。1 <= k <= n+1（k = n+1 は保証できない場合で、呼び出し側で +inf にする）
    """
    alpha_exact = Fraction(alpha).limit_denominator(10**9)
    return math.ceil((n + 1) * (1 - alpha_exact))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    r"""非適合度スコアから共形分位点 \hat{q} を返す。

    定義（卒論 eq:khat。順序統計量は定義 def:order-stat）:
        k = \lceil (n+1)(1-\alpha) \rceil
        \hat{q} = S_{(k)}          （S_{(k)} は昇順第 k 順序統計量）
        k > n のときは \hat{q} = +\infty

    +\infty になるのは、キャリブレーション集合が小さすぎて有限標本で
    1-\alpha の被覆を保証できない場合である（例: n=8, \alpha=0.1）。
    これはバグではなく理論どおりの挙動なので、握りつぶさずそのまま返す。

    Parameters
    ----------
    scores : 形状 (n,) のキャリブレーション集合の非適合度スコア
    alpha : 有意水準。0 < alpha < 1

    Returns
    -------
    float : 共形分位点。保証できない場合は np.inf

    Notes
    -----
    np.quantile の既定の線形補間を使ってはならない。有限標本被覆保証が壊れる。
    詳細は CLAUDE.md「絶対に守ること 1」を参照。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha は (0,1) でなければならない: {alpha}")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"scores は1次元配列でなければならない: shape={scores.shape}")

    n = scores.size
    k = conformal_index(n, alpha)  # ← (n+1) であって n ではない。float の丸めも避ける
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])  # 第 k 順序統計量（0-indexed で k-1）


def conformal_pvalue(cal_scores: np.ndarray, test_score: float) -> float:
    r"""共形 p 値を返す（卒論 第3章。式のラベルは未定）。

        p = \frac{1 + \#\{ i : S_i \ge S_{n+1} \}}{n+1}

    交換可能性の下で p は \{1/(n+1), 2/(n+1), ..., 1\} 上の離散一様分布に従う。
    この性質は tests/test_pvalue_uniformity.py で実装全体の統合テストに使う。
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    n = cal_scores.size
    return float((1.0 + np.sum(cal_scores >= test_score)) / (n + 1))


def interval_from_quantile(
    pred: np.ndarray, q_hat: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""絶対残差スコア S = |y - \hat{f}(x)|（卒論 eq:score）に対する予測区間を返す。

        C(x) = [\hat{f}(x) - \hat{q},  \hat{f}(x) + \hat{q}]      （卒論 eq:interval）

    正規化残差スコアや CQR の場合は区間の作り方が異なるため、
    src/conformal/scores.py 側の対応する関数を使うこと。
    """
    pred = np.asarray(pred, dtype=float)
    return pred - q_hat, pred + q_hat
