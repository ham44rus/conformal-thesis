"""理論から導かれる性質による統合テスト。

対応箇所: 卒論 第3章（定理 thm:coverage, 系 cor:upper, 定理 thm:cond-coverage-beta）, 実験E1
          第4章（sec:normalized-score, sec:cqr：一般のスコアでも同じ Beta 分布に従うこと）

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
    cqr_score,
    interval_from_quantile,
    normalized_residual,
)
from conformal.datasets import homoscedastic


@pytest.fixture(scope="module")
def fitted_model():
    """全テストで共通の学習済みモデル（学習データは固定）。"""
    rng = np.random.default_rng(20260901)
    X, y = homoscedastic(500, rng)
    return Ridge(alpha=1.0).fit(X, y)


def test_共形p値が離散一様分布に従う(fitted_model):
    """交換可能性の下で p は {1/(n+1),...,1} 上の離散一様分布（卒論 第3章。共形 p 値の式は本文に未掲載）。

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
    Beta(n+1-l, l), l = floor((n+1)*alpha) に従う（Vovk 2012, 卒論 定理 thm:cond-coverage-beta）。

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
    """1-alpha <= P(Y in C(X)) <= 1-alpha + 1/(n+1)（卒論 系 cor:upper、式 eq:two-sided）。

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


# ---------------------------------------------------------------------------
# 一般のスコアへの拡張（卒論 第4章 sec:normalized-score・sec:cqr）
#
# 定理 thm:coverage の証明でスコアの形 |y - f(x)| を使うのは、補題 lem:score-exch の
# 写像 G と、被覆と S_{n+1} <= q_hat の同値（eq:cover-iff）だけである。学習用データ
# だけで決まる固定のスコア関数 s(x, y) なら、G を s に置き換えるだけで同じ証明が通り、
# C(x) = {y : s(x, y) <= q_hat} の被覆率は同じ Beta 分布に従う。
#
# ここでは sigma(x) や分位点関数に**わざと外した手書きの固定関数**を使う。
# 「真の sigma や真の分位点に近いから被覆する」のではなく「学習用データだけで決まる
# 固定関数なら何でも被覆する」という理論の主張を、そのまま検査するためである。
# ---------------------------------------------------------------------------


def _sigma_wrong(X: np.ndarray) -> np.ndarray:
    """わざと外した sigma(x)。データは等分散（真の sigma は定数 0.5）なので無関係な関数。"""
    return 1.0 + X[:, 0] ** 2


def _quantile_pair_wrong(X: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """わざと外した分位点関数の組。x1 > 0.8 の領域で lo > hi と交差させてある。

    交差した点は adaptive.QuantilePair.predict と同じく入れ替えて lo <= hi にする。
    入れ替えも x だけで決まる固定の操作なので、スコアは固定関数のままである。
    """
    lo = pred - 0.6 + X[:, 0]
    hi = pred + 0.2
    return np.minimum(lo, hi), np.maximum(lo, hi)


def _coverage_over_trials(score_fn, rng, n_cal: int, alpha: float, n_trial: int) -> np.ndarray:
    """学習用データ（score_fn に固定済み）を固定し、較正集合を引き直したときの被覆率。

    テスト集合は大きく固定して二項ノイズを抑える（test_被覆率がBeta分布に従う と同じ理由）。
    """
    Xt, yt = homoscedastic(200_000, rng)
    test_s = score_fn(Xt, yt)

    cov = np.empty(n_trial)
    for r in range(n_trial):
        Xc, yc = homoscedastic(n_cal, rng)
        q = conformal_quantile(score_fn(Xc, yc), alpha)
        cov[r] = np.mean(test_s <= q)
    return cov


def _assert_beta(cov: np.ndarray, n_cal: int, alpha: float, name: str) -> None:
    l = int(np.floor((n_cal + 1) * alpha))
    a_par, b_par = n_cal + 1 - l, l
    ks_p = stats.kstest(cov, "beta", args=(a_par, b_par)).pvalue
    assert ks_p > 0.01, f"{name}: 被覆率が Beta({a_par},{b_par}) に従わない (KS p={ks_p:.4f})"


def test_正規化残差でも被覆率はBeta分布に従う(fitted_model):
    """sigma_hat がどんな固定関数でも、正規化残差スコアの被覆率は Beta(n+1-l, l) に従う。

    sigma_hat(x) = 1 + x1^2 はわざと外した関数（真の sigma は定数）。
    それでも被覆率の分布が絶対残差のときと同じなのは、保証がスコアの形ではなく
    「学習用データだけで決まる固定写像であること」から出るため（卒論 sec:normalized-score）。
    """
    n_cal, alpha, n_trial = 100, 0.1, 2000
    rng = np.random.default_rng(6)

    def score(X, y):
        return normalized_residual(y, fitted_model.predict(X), _sigma_wrong(X))

    cov = _coverage_over_trials(score, rng, n_cal, alpha, n_trial)
    _assert_beta(cov, n_cal, alpha, "正規化残差")


def test_CQRでも被覆率はBeta分布に従う(fitted_model):
    """分位点関数がどんな固定関数でも、CQR スコアの被覆率は Beta(n+1-l, l) に従う。

    q_lo, q_hi はわざと外した関数で、x1 > 0.8 では交差する（入れ替えて使う）。
    スコアは負にもなるが、共形分位点と被覆の同値関係は変わらない（卒論 sec:cqr）。
    """
    n_cal, alpha, n_trial = 100, 0.1, 2000
    rng = np.random.default_rng(7)

    def score(X, y):
        lo, hi = _quantile_pair_wrong(X, fitted_model.predict(X))
        return cqr_score(y, lo, hi)

    # 交差する領域が実際に含まれていることを確認しておく（含まれないなら検査の意味が薄い）
    Xchk, _ = homoscedastic(1000, rng)
    pred = fitted_model.predict(Xchk)
    assert np.mean((pred - 0.6 + Xchk[:, 0]) > (pred + 0.2)) > 0.1, "交差する点が少なすぎる"

    cov = _coverage_over_trials(score, rng, n_cal, alpha, n_trial)
    _assert_beta(cov, n_cal, alpha, "CQR")
