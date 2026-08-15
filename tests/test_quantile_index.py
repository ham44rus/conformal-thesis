"""共形分位点のインデックスが理論どおりか（CLAUDE.md「よくあるバグ」#1, #2）。

対応箇所: 卒論 式(3.4)

このテストは実装を見ずに、理論だけから書ける。だからこそ実装の正しさの根拠になる。
"""

import numpy as np
import pytest

from conformal import conformal_quantile


def test_境界ケース_n9_alpha01():
    """n=9, alpha=0.1 -> k = ceil(10*0.9) = 9 -> 第9順序統計量（最大値）。"""
    s = np.arange(1.0, 10.0)  # 1..9
    assert conformal_quantile(s, 0.10) == 9.0


def test_境界ケース_n9_alpha02():
    """n=9, alpha=0.2 -> k = ceil(10*0.8) = 8 -> 8.0。"""
    s = np.arange(1.0, 10.0)
    assert conformal_quantile(s, 0.20) == 8.0


def test_保証できない場合は無限大():
    """n=8, alpha=0.1 -> k = ceil(9*0.9) = 9 > 8 なので保証不能 -> +inf。"""
    s = np.arange(1.0, 9.0)  # n = 8
    assert np.isinf(conformal_quantile(s, 0.10))


def test_np_quantile_とは異なる():
    """線形補間版と一致してはいけない（一致したら誤実装）。"""
    s = np.arange(1.0, 10.0)
    assert conformal_quantile(s, 0.10) != np.quantile(s, 0.9)


def test_n倍版のoff_by_oneと異なる():
    """ceil(n*(1-alpha)) 版と食い違う設定で、正しい方を返すこと。"""
    s = np.arange(1.0, 21.0)  # n = 20
    alpha = 0.05
    correct_k = int(np.ceil((20 + 1) * (1 - alpha)))  # = 20
    wrong_k = int(np.ceil(20 * (1 - alpha)))          # = 19
    assert correct_k != wrong_k
    assert conformal_quantile(s, alpha) == float(correct_k)


def test_単調性():
    """alpha を小さくすると分位点は広がる（狭くならない）。"""
    rng = np.random.default_rng(0)
    s = rng.normal(size=200)
    assert conformal_quantile(s, 0.01) >= conformal_quantile(s, 0.10)
    assert conformal_quantile(s, 0.10) >= conformal_quantile(s, 0.30)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_不正なalphaを弾く(alpha):
    with pytest.raises(ValueError):
        conformal_quantile(np.arange(10.0), alpha)
