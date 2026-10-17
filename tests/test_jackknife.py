"""jackknife+ と CV+（src/conformal/jackknife.py）の検証テスト。

対応箇所: 実験E6（sec:e6）、仕様書 specs/E6_jackknife_cv.md
出典: Barber, Candès, Ramdas, Tibshirani (2021)（refs.bib の barber2021jackknife）

ここで検査するのは「実装がこう動く」ではなく「定義と理論がこう言う」性質だけ。
1. 学習用データによらないモデル（mu_{-i} = mu）なら、区間は分割型の
   [mu(x) - S_(k), mu(x) + S_(k)] に一致する（下端の l = n+1-k の恒等式、卒論 prop:excess）
2. CV+ で K = n なら、各組が 1 点になり jackknife+ と一致する
3. 定義を手で計算した例と一致する
4. 交換可能なデータでは jackknife+ の被覆率が 1 - 2*alpha 以上（出典の定理による）
"""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from conformal import (
    conformal_quantile,
    cv_plus_interval,
    interval_from_quantile,
    jackknife_plus_interval,
)
from conformal.datasets import homoscedastic
from conformal.jackknife import loo_interval


class FixedModel:
    """学習用データを見ない予測器。fit は何もせず、常に同じ関数 mu(x) = 2*x1 - x2 を返す。"""

    def __init__(self, seed: int = 0):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return 2.0 * X[:, 0] - X[:, 1]


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20, 1 / 3, 0.50])
@pytest.mark.parametrize("n", [2, 8, 9, 19, 20, 50])
def test_学習用データによらないモデルなら分割型と一致する(n, alpha):
    """mu_{-i} = mu なので R_i = S_i = |Y_i - mu(X_i)|。上端は mu(x) + S_(k)、
    下端は mu(x) - S_(n+1-l) = mu(x) - S_(k)。k > n のときは両端とも無限大。

    格子には k > n（例：n=8, alpha=0.10）と float の丸めで k がずれうる例（alpha=1/3, n=8）を含む。
    許容幅なしの完全一致で比べる。
    """
    rng = np.random.default_rng(100 * n + int(1000 * alpha))
    X, y = homoscedastic(n, rng)
    X_test, _ = homoscedastic(30, rng)
    mu = FixedModel()
    q_hat = conformal_quantile(np.abs(y - mu.predict(X)), alpha)
    exp_lo, exp_hi = interval_from_quantile(mu.predict(X_test), q_hat)

    lo, hi, n_fits = jackknife_plus_interval(FixedModel, X, y, X_test, alpha, seeds=[0] * n)
    assert n_fits == n
    assert np.array_equal(lo, exp_lo) and np.array_equal(hi, exp_hi)

    for k_folds in sorted({2, n}):
        lo, hi, n_fits = cv_plus_interval(FixedModel, X, y, X_test, alpha, k_folds, rng, [0] * k_folds)
        assert n_fits == k_folds
        assert np.array_equal(lo, exp_lo) and np.array_equal(hi, exp_hi)


def test_一致の格子に無限大の場合が含まれている():
    """上のテストが k > n（両端が無限大）を本当に通っていることの確認（n=8, alpha=0.10 は k=9）。"""
    rng = np.random.default_rng(0)
    X, y = homoscedastic(8, rng)
    lo, hi, _ = jackknife_plus_interval(FixedModel, X, y, X[:3], 0.10, [0] * 8)
    assert np.all(lo == -np.inf) and np.all(hi == np.inf)


def test_CVプラスでKがnならjackknifeプラスと一致する():
    """K = n なら各組は 1 点で、mu_{-S(i)} = mu_{-i}。同じ seed の同じ学習アルゴリズムを
    同じ並び順のデータで学習するので、区間は完全に一致する。"""
    rng = np.random.default_rng(3)
    n, alpha = 15, 0.2
    X, y = homoscedastic(n, rng)
    X_test, _ = homoscedastic(50, rng)

    def factory(seed):
        return GradientBoostingRegressor(n_estimators=20, random_state=seed)

    jk = jackknife_plus_interval(factory, X, y, X_test, alpha, [7] * n)
    cv = cv_plus_interval(factory, X, y, X_test, alpha, n, rng, [7] * n)
    assert np.array_equal(jk[0], cv[0]) and np.array_equal(jk[1], cv[1])
    assert jk[2] == cv[2] == n


def test_手計算の例():
    """n = 4, alpha = 0.2：k = ceil(5*0.8) = 4、l = floor(5*0.2) = 1。

    テスト点 1 つ。抜いた予測 mu_{-i}(x) = 1, 2, 0.5, 1.5、残差 R_i = 0.25, 0.125, 0.75, 0.5
        上側の値 mu + R = 1.25, 2.125, 1.25, 2.0   → 小さい方から 4 番目 = 2.125
        下側の値 mu - R = 0.75, 1.875, -0.25, 1.0  → 小さい方から 1 番目 = -0.25
    alpha = 0.1 なら k = ceil(4.5) = 5 > 4、l = 0 なので (-inf, +inf)。
    （値は 2 進で正確に表せるものを選んだので、完全一致で比べる）
    """
    loo_pred = np.array([[1.0], [2.0], [0.5], [1.5]])
    residuals = np.array([0.25, 0.125, 0.75, 0.5])
    lo, hi = loo_interval(loo_pred, residuals, 0.2)
    assert lo.tolist() == [-0.25] and hi.tolist() == [2.125]

    lo, hi = loo_interval(loo_pred, residuals, 0.1)
    assert lo.tolist() == [-np.inf] and hi.tolist() == [np.inf]


def test_jackknifeプラスの被覆率は1マイナス2アルファ以上():
    """交換可能なデータでは P(Y in C(X)) >= 1 - 2*alpha（出典の定理による）。

    n = 30、試行ごとにデータとテスト点を引き直す。被覆率の平均がこの下限から
    3 SE（SE = 試行間 SD / sqrt(R)。テスト点も試行ごとに新しいので、これで全変動を含む）
    以上下回らないことを確かめる。
    """
    rng = np.random.default_rng(20261015)
    n, alpha, n_trial, n_test = 30, 0.10, 300, 200
    covs = []
    for _ in range(n_trial):
        X, y = homoscedastic(n, rng)
        X_test, y_test = homoscedastic(n_test, rng)
        lo, hi, _ = jackknife_plus_interval(lambda s: LinearRegression(), X, y, X_test, alpha, [0] * n)
        covs.append(np.mean((lo <= y_test) & (y_test <= hi)))
    covs = np.array(covs)
    se = covs.std(ddof=1) / np.sqrt(n_trial)
    assert covs.mean() >= 1 - 2 * alpha - 3 * se, f"被覆率 {covs.mean():.4f}, SE {se:.4f}"


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alphaの範囲外はエラー(alpha):
    X, y = homoscedastic(10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        jackknife_plus_interval(FixedModel, X, y, X, alpha, [0] * 10)
    with pytest.raises(ValueError):
        cv_plus_interval(FixedModel, X, y, X, alpha, 5, np.random.default_rng(0), [0] * 5)
    with pytest.raises(ValueError):
        loo_interval(np.zeros((10, 1)), np.zeros(10), alpha)


def test_形状の不一致はエラー():
    X, y = homoscedastic(10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        jackknife_plus_interval(FixedModel, X, y[:9], X, 0.1, [0] * 10)
    with pytest.raises(ValueError):
        jackknife_plus_interval(FixedModel, X, y, X[:, :3], 0.1, [0] * 10)
    with pytest.raises(ValueError):
        jackknife_plus_interval(FixedModel, X[:, 0], y, X, 0.1, [0] * 10)
    with pytest.raises(ValueError):
        jackknife_plus_interval(FixedModel, X, y, X, 0.1, [0] * 9)  # seeds の長さ
    with pytest.raises(ValueError):
        loo_interval(np.zeros((10, 1)), np.zeros(9), 0.1)


@pytest.mark.parametrize("n_folds", [0, 1, 11])
def test_組の数は2以上n以下(n_folds):
    X, y = homoscedastic(10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        cv_plus_interval(FixedModel, X, y, X, 0.1, n_folds, np.random.default_rng(0), [0] * n_folds)
