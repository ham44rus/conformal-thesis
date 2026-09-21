"""外部実装 MAPIE との数値照合（CLAUDE.md「テストは実装と独立に」の 4 つ目の型）。

対応箇所: 卒論 第3章 sec:split-algorithm（eq:score, eq:khat, eq:interval）、
          付録「生成AIの利用について」の検証体制 (3)

絶対残差の split 法について、同じ学習済みモデル・同じ較正データ・同じ alpha で、
自作実装と MAPIE 0.9.2（``MapieRegressor(method="base", cv="prefit")``）の
予測区間の端点が一致することを検査する。MAPIE は検証用にのみ使う（本体には使わない）。

MAPIE 0.9.2 の共形分位点は

    alpha_cor = ceil((1 - alpha) * (n + 1)) / n
    q = np.quantile(scores, alpha_cor, method="lower")     # 0 始まりの添字 floor((n-1) * alpha_cor)

で、k = ceil((1-alpha)(n+1)) <= n のとき floor((n-1) k/n) = k-1 なので S_(k) に一致する。
一致しないのは次の 2 つの場合で、いずれも MAPIE 側の計算法によるものである（xfail で記録する。
テストを MAPIE に合わせにはいかない）。

1. (n+1)(1-alpha) が整数になる alpha（例 n=8, alpha=1/3）。MAPIE は浮動小数点で
   ceil を取るので 9*(1-1/3) = 6.000000000000001 -> k=7 となり、S_(7) を返す。
   正しくは k=6（chapter2_spec.md「実装との対応メモ」落とし穴 4。自作実装は
   Fraction で厳密に計算する）。
2. k > n（例 n=8, alpha=0.1）。理論では q_hat = +inf、区間は実数全体。MAPIE は既定では
   ValueError を投げ、``allow_infinite_bounds=True`` でも「無限にする」判定を
   ``1 - alpha >= 1`` で行っているため常に偽になり、最大スコア S_(n) を返す。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from conformal import absolute_residual, conformal_index, conformal_quantile, interval_from_quantile
from conformal.datasets import homoscedastic

mapie = pytest.importorskip("mapie")
from mapie.regression import MapieRegressor  # noqa: E402


@pytest.fixture(scope="module")
def fitted_model():
    rng = np.random.default_rng(20260921)
    X, y = homoscedastic(300, rng)
    return Ridge(alpha=1.0).fit(X, y)


def _ours_and_mapie(model, n_cal: int, alpha: float, seed: int):
    """同じ較正集合・テスト点で (自作の lo, hi), (MAPIE の lo, hi) を返す。"""
    rng = np.random.default_rng(seed)
    Xc, yc = homoscedastic(n_cal, rng)
    Xt, _ = homoscedastic(50, rng)

    q = conformal_quantile(absolute_residual(yc, model.predict(Xc)), alpha)
    lo, hi = interval_from_quantile(model.predict(Xt), q)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = MapieRegressor(model, method="base", cv="prefit").fit(Xc, yc)
        _, pis = mp.predict(Xt, alpha=alpha, allow_infinite_bounds=True)
    return (lo, hi), (pis[:, 0, 0], pis[:, 1, 0])


@pytest.mark.parametrize(
    "n_cal, alpha",
    [(20, 0.10), (50, 0.05), (100, 0.20), (19, 0.10), (9, 0.10), (564, 0.10), (2000, 0.05)],
)
def test_区間の端点がMAPIEと一致する(fitted_model, n_cal, alpha):
    """k <= n の通常の設定では両者の区間は一致する。

    n=19, alpha=0.1 は (n+1)(1-alpha) = 18 ちょうど（10 進で表せる alpha なので float でも
    ずれない）、n=9 は k = n（最大値）、n=564 は k/n の丸めが問題になりうる例
    （chapter2_spec.md 落とし穴 1 は inverted_cdf の話で、MAPIE の lower には効かない）。
    """
    (lo, hi), (mlo, mhi) = _ours_and_mapie(fitted_model, n_cal, alpha, seed=n_cal)
    np.testing.assert_allclose(lo, mlo, rtol=0, atol=1e-12)
    np.testing.assert_allclose(hi, mhi, rtol=0, atol=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="MAPIE は ceil((1-alpha)(n+1)) を float で計算するため 9*(2/3) -> 7 となり S_(7) を返す。"
    "正しい k は 6（Fraction による厳密計算）。MAPIE 側の丸めの問題であり、自作実装は変えない",
)
def test_境界例_n8_alpha3分の1(fitted_model):
    """(n+1)(1-alpha) = 6 ちょうどの例。自作実装は k=6、MAPIE は k=7 になる。"""
    n_cal, alpha = 8, 1 / 3
    assert conformal_index(n_cal, alpha) == 6
    (lo, hi), (mlo, mhi) = _ours_and_mapie(fitted_model, n_cal, alpha, seed=8)
    np.testing.assert_allclose(lo, mlo, rtol=0, atol=1e-12)
    np.testing.assert_allclose(hi, mhi, rtol=0, atol=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="k > n では理論は q_hat = +inf（eq:khat の規約 S_(n+1) = +inf）。MAPIE 0.9.2 は "
    "allow_infinite_bounds=True でも無限判定を 1-alpha >= 1 で行うため常に偽になり、S_(n) を返す",
)
def test_保証できない場合_kがnを超える(fitted_model):
    """n=8, alpha=0.1 -> k=9 > n。自作実装は (-inf, inf)、MAPIE は最大スコアの有限区間。"""
    n_cal, alpha = 8, 0.10
    assert conformal_index(n_cal, alpha) == 9
    (lo, hi), (mlo, mhi) = _ours_and_mapie(fitted_model, n_cal, alpha, seed=80)
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))
    assert np.all(np.isneginf(mlo)) and np.all(np.isposinf(mhi))


def test_MAPIEの添字の計算法を記録する():
    """MAPIE の lower 分位点が k <= n で S_(k) に一致する理由（docstring の式）の確認。

    alpha_cor = k/n のとき np.quantile(..., method="lower") の添字は floor((n-1) k/n) = k-1。
    n=1..3000、alpha の代表値で、この添字が k-1 に一致することを直接確かめる。
    """
    for alpha in (0.05, 0.10, 0.20):
        for n in range(1, 3001):
            k = conformal_index(n, alpha)
            if k > n:
                continue
            idx = int(np.floor((n - 1) * (k / n)))
            assert idx == k - 1, (n, alpha, k, idx)
