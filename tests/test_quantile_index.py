"""共形分位点のインデックスが理論どおりか（CLAUDE.md「よくあるバグ」#1, #2）。

対応箇所: 卒論 eq:khat（k の定義）と定義 def:order-stat（順序統計量）

このテストは実装を見ずに、理論だけから書ける。だからこそ実装の正しさの根拠になる。
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from conformal import conformal_index, conformal_quantile


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


# ---------------------------------------------------------------------------
# k の計算の浮動小数点誤差（thesis/drafts/chapter2_spec.md「実装との対応メモ」落とし穴 4）
# (n+1)*(1-alpha) を float で計算すると ceil が 1 ずれることがある。
# ---------------------------------------------------------------------------


def test_n564_alpha01_は第509順序統計量():
    """n=564, alpha=0.1 -> k = ceil(565*0.9) = ceil(508.5) = 509 -> S_(509)。"""
    s = np.arange(1.0, 565.0)  # 1..564
    assert conformal_quantile(s, 0.10) == 509.0


def test_alpha_3分の1_n8_は第6順序統計量():
    """n=8, alpha=1/3 -> (n+1)(1-alpha) = 6 ちょうど -> k = 6 -> S_(6)。

    float では 9*(1-1/3) = 6.000000000000001 となり、素朴な ceil は 7 を返す。
    """
    s = np.arange(1.0, 9.0)  # 1..8
    assert conformal_index(8, 1 / 3) == 6
    assert conformal_quantile(s, 1 / 3) == 6.0


@pytest.mark.parametrize(
    "alpha", [1 / 3, 2 / 3, 0.45, 0.7, 0.95, 0.99, 0.05, 0.10, 0.20]
)
def test_kは有理数演算の厳密値と一致する(alpha):
    """n=1..3000 で k が Fraction による厳密計算と一致し、q_hat が S_(k) になること。

    alpha は 1/3, 2/3 のように 10 進で書けないものを含む。
    """
    alpha_exact = Fraction(alpha).limit_denominator(10**9)
    for n in range(1, 3001):
        k_exact = math.ceil((n + 1) * (1 - alpha_exact))
        assert conformal_index(n, alpha) == k_exact
        # スコアを 1..n にしておくと S_(k) = k なので、返り値がそのまま k の検査になる
    for n in (8, 14, 17, 20, 564, 2077, 2119, 2999, 3000):
        k_exact = math.ceil((n + 1) * (1 - alpha_exact))
        q = conformal_quantile(np.arange(1.0, n + 1.0), alpha)
        assert q == (np.inf if k_exact > n else float(k_exact))


# ---------------------------------------------------------------------------
# ベータ分布のパラメータ l = floor((n+1)*alpha) は k から l = n+1-k で求める
# （卒論 定理 thm:cond-coverage-beta。E1 と make_figures がこの式を使う）。
# float の floor だと alpha によっては 1 ずれる（chapter3_sections_spec.md 実装メモ 5）。
# ---------------------------------------------------------------------------


def test_n89_alpha07_のlは63():
    """n=89, alpha=0.7 -> (n+1)*alpha = 63 ちょうど -> l = 63。

    float では 90*0.7 = 62.99999999999999 となり、素朴な floor は 62 を返す。
    """
    n, alpha = 89, 0.7
    assert n + 1 - conformal_index(n, alpha) == 63


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20, 1 / 3, 0.7, 0.45, 0.95])
def test_lは有理数演算の厳密値と一致する(alpha):
    """n=1..3000 で n+1-k が Fraction で厳密に計算した floor((n+1)*alpha) と一致すること。"""
    alpha_exact = Fraction(alpha).limit_denominator(10**9)
    for n in range(1, 3001):
        l_exact = math.floor((n + 1) * alpha_exact)
        assert n + 1 - conformal_index(n, alpha) == l_exact
