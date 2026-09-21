"""非適合度スコア（nonconformity score）。

対応箇所:
    卒論 第3章（絶対残差）, 第4章（正規化残差, CQR）
    実験 E2（スコアの比較）, E3（異分散データと条件付き被覆）

スコアを差し替えるだけで手法が変わる、というのが等角予測の設計上の要点。
"""

from __future__ import annotations

import numpy as np

__all__ = ["absolute_residual", "normalized_residual", "cqr_score", "cqr_interval"]


def absolute_residual(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    r"""絶対残差スコア  S = |y - \hat{f}(x)|（卒論 eq:score）。

    最も単純。区間幅が x によらず一定になるため、異分散データでは
    条件付き被覆が大きく崩れる（実験E3で示す）。
    """
    return np.abs(np.asarray(y) - np.asarray(pred))


def normalized_residual(
    y: np.ndarray, pred: np.ndarray, scale: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    r"""正規化残差スコア  S = |y - \hat{f}(x)| / \hat{\sigma}(x)（卒論 4.1 節。ラベルは仮に eq:score-normalized）。

    \hat{\sigma}(x) は残差の大きさを別に回帰して推定したもの。
    異分散への簡易な対処。区間は \hat{f}(x) \pm \hat{q}\,\hat{\sigma}(x)。

    Parameters
    ----------
    scale : \hat{\sigma}(x) の推定値。正でなければならない
    """
    scale = np.asarray(scale, dtype=float)
    if np.any(scale <= 0):
        raise ValueError("scale はすべて正でなければならない")
    return np.abs(np.asarray(y) - np.asarray(pred)) / (scale + eps)


def cqr_score(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    r"""CQR の非適合度スコア（卒論 4.2 節。ラベルは仮に eq:cqr-score。Romano et al. 2019）。

        S = \max\{ \hat{q}_{lo}(x) - y,\;  y - \hat{q}_{hi}(x) \}

    区間の内側にあれば負、外側にあれば正になる。符号を逆にするのが
    典型的なバグなので、tests/test_cqr_score.py で手計算例と照合すること。
    （CLAUDE.md「よくあるバグ」#4）
    """
    y, q_lo, q_hi = map(np.asarray, (y, q_lo, q_hi))
    return np.maximum(q_lo - y, y - q_hi)


def cqr_interval(
    q_lo: np.ndarray, q_hi: np.ndarray, q_hat: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""CQR の予測区間（卒論 4.2 節。ラベルは仮に eq:cqr-interval）。

        C(x) = [\hat{q}_{lo}(x) - \hat{q},\;  \hat{q}_{hi}(x) + \hat{q}]

    分位点回帰が x に応じた幅を出すので、区間が適応的になる。
    """
    return np.asarray(q_lo) - q_hat, np.asarray(q_hi) + q_hat
