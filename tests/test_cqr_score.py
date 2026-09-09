"""CQR のスコアを手計算例と照合する（CLAUDE.md「よくあるバグ」#4）。

対応箇所: 卒論 式(4.3), 式(4.4), 実験E3

符号を逆にするのが典型的なバグ。手で計算できる数例を直接書いておく。
"""

import numpy as np

from conformal import cqr_interval, cqr_score
from conformal.split import conformal_quantile


def test_区間の内側なら負_外側なら正():
    q_lo = np.array([0.0, 0.0, 0.0])
    q_hi = np.array([1.0, 1.0, 1.0])
    y = np.array([0.5, -0.3, 1.4])
    #  y=0.5 : max(0-0.5, 0.5-1) = max(-0.5,-0.5) = -0.5   （内側 -> 負）
    #  y=-0.3: max(0+0.3, -0.3-1) = 0.3                    （下に外れ -> 正）
    #  y=1.4 : max(0-1.4, 1.4-1) = 0.4                     （上に外れ -> 正）
    np.testing.assert_allclose(cqr_score(y, q_lo, q_hi), [-0.5, 0.3, 0.4])


def test_はみ出し量そのものになる():
    """外れている場合、スコアは区間からのはみ出し距離に一致する。"""
    q_lo, q_hi = np.array([2.0]), np.array([5.0])
    assert cqr_score(np.array([7.5]), q_lo, q_hi)[0] == 2.5
    assert cqr_score(np.array([0.5]), q_lo, q_hi)[0] == 1.5


def test_区間は両側に同じ量だけ広がる():
    q_lo, q_hi = np.array([1.0, 3.0]), np.array([2.0, 6.0])
    lo, hi = cqr_interval(q_lo, q_hi, 0.5)
    np.testing.assert_allclose(lo, [0.5, 2.5])
    np.testing.assert_allclose(hi, [2.5, 6.5])


def test_分位点回帰が完璧ならqはほぼ0():
    """真の条件付き分位点を与えた場合、共形補正量は 0 の近くに来るはず。"""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.normal(0.0, 1.0, size=n)
    # 分位点回帰の出力を模した定数。標本分位点であって共形分位点ではないので
    # np.quantile を使ってよい（共形分位点は下の conformal_quantile が計算する）。
    q_lo = np.full(n, float(np.quantile(y, 0.05)))
    q_hi = np.full(n, float(np.quantile(y, 0.95)))
    q_hat = conformal_quantile(cqr_score(y, q_lo, q_hi), 0.1)
    assert abs(q_hat) < 0.15, f"補正量が大きすぎる: {q_hat}"
