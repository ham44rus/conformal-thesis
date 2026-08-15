"""理論から導かれる性質による統合テスト。

対応箇所: 卒論 第3章（定理3.1, 系3.2）, 実験E1

ここで検査するのは「実装がこう動く」ではなく「理論がこう言う」性質だけ。
実装と独立に正しさを判定できるので、生成AIが書いたコードの検証に使える。
"""

import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import Ridge

from conformal import (
    absolute_residual,
    conformal_pvalue,
    conformal_quantile,
    interval_from_quantile,
)
from conformal.datasets import homoscedastic


@pytest.fixture(scope="module")
def fitted_model():
    """全テストで共通の学習済みモデル（学習データは固定）。"""
    rng = np.random.default_rng(20260901)
    X, y = homoscedastic(500, rng)
    return Ridge(alpha=1.0).fit(X, y)


def test_共形p値が離散一様分布に従う(fitted_model):
    """交換可能性の下で p は {1/(n+1),...,1} 上の離散一様分布（卒論 補題3.1）。

    これが通れば実装はほぼ正しい、という最も強力な統合テスト。
    """
    n_cal, n_trial = 50, 3000
    rng = np.random.default_rng(1)
    pvals = np.empty(n_trial)
    for r in range(n_trial):
        Xc, yc = homoscedastic(n_cal, rng)
        Xt, yt = homoscedastic(1, rng)
        cal_s = absolute_residual(yc, fitted_model.predict(Xc))
        test_s = absolute_residual(yt, fitted_model.predict(Xt))[0]
        pvals[r] = conformal_pvalue(cal_s, test_s)

    counts = np.bincount(np.round(pvals * (n_cal + 1)).astype(int), minlength=n_cal + 2)[1:]
    expected = np.full(n_cal + 1, n_trial / (n_cal + 1))
    p = stats.chisquare(counts, expected).pvalue
    assert p > 0.01, f"共形p値が一様でない (chi2 p={p:.4f})"


def test_被覆率がBeta分布に従う(fitted_model):
    """キャリブレーション集合を固定したときの被覆率は
    Beta(n+1-l, l), l = floor((n+1)*alpha) に従う（Vovk 2012, 卒論 定理3.3）。

    注意: テスト集合が小さいと二項ノイズが上乗せされて分散が理論より大きくなり、
    実装が正しくても棄却される。大きな固定テスト集合の残差を先に計算しておく。
    """
    n_cal, alpha, n_trial = 100, 0.1, 2000
    rng = np.random.default_rng(2)

    Xt, yt = homoscedastic(200_000, rng)
    test_res = absolute_residual(yt, fitted_model.predict(Xt))

    cov = np.empty(n_trial)
    for r in range(n_trial):
        Xc, yc = homoscedastic(n_cal, rng)
        q = conformal_quantile(absolute_residual(yc, fitted_model.predict(Xc)), alpha)
        cov[r] = np.mean(test_res <= q)

    l = int(np.floor((n_cal + 1) * alpha))
    a_par, b_par = n_cal + 1 - l, l
    ks_p = stats.kstest(cov, "beta", args=(a_par, b_par)).pvalue
    assert ks_p > 0.01, f"被覆率が Beta({a_par},{b_par}) に従わない (KS p={ks_p:.4f})"


@pytest.mark.parametrize("n_cal", [20, 50, 200])
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_有限標本の上下界(fitted_model, n_cal, alpha):
    """1-alpha <= P(Y in C(X)) <= 1-alpha + 1/(n+1)（卒論 定理3.2）。

    モンテカルロ誤差を見込んで両側に 0.005 の余裕を与える。
    """
    rng = np.random.default_rng(3 + n_cal)
    Xt, yt = homoscedastic(50_000, rng)
    test_res = absolute_residual(yt, fitted_model.predict(Xt))

    cov = []
    for _ in range(800):
        Xc, yc = homoscedastic(n_cal, rng)
        q = conformal_quantile(absolute_residual(yc, fitted_model.predict(Xc)), alpha)
        cov.append(np.mean(test_res <= q))
    m = float(np.mean(cov))

    lower, upper = 1 - alpha, 1 - alpha + 1 / (n_cal + 1)
    assert lower - 0.005 <= m <= upper + 0.005, (
        f"n={n_cal}, alpha={alpha}: 平均被覆率 {m:.4f} が "
        f"[{lower:.4f}, {upper:.4f}] を外れた"
    )


def test_最悪のモデルでも被覆する():
    """定数予測器（常に0）でも被覆率は 1-alpha を満たす（モデル非依存性）。

    満たさない場合はキャリブレーション集合が学習に混入している。
    （CLAUDE.md「よくあるバグ」#3 の検出テスト）
    """
    n_cal, alpha = 200, 0.1
    rng = np.random.default_rng(4)
    cov = []
    for _ in range(300):
        _, yc = homoscedastic(n_cal, rng)
        q = conformal_quantile(np.abs(yc - 0.0), alpha)
        _, yt = homoscedastic(1000, rng)
        lo, hi = interval_from_quantile(np.zeros(1000), q)
        cov.append(np.mean((yt >= lo) & (yt <= hi)))
    assert np.mean(cov) >= 1 - alpha, "定数予測器で被覆が保証を下回った（リークの疑い）"


def test_alphaが小さいと区間が発散する():
    """保証できないほど alpha が小さいと区間は (-inf, inf) になる。"""
    rng = np.random.default_rng(5)
    _, y = homoscedastic(10, rng)
    q = conformal_quantile(np.abs(y), 0.001)
    lo, hi = interval_from_quantile(np.zeros(3), q)
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))
