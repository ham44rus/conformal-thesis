"""較正用データが学習に混入していないことの検査（CLAUDE.md「2. データリークの禁止」）。

対応箇所:
    卒論 第3章 仮定 (A2)（sec:coverage-guarantee：学習用データは較正用データ・テスト点と独立）
    卒論 第4章 sec:normalized-score（sigma_hat の学習）・sec:cqr（分位点回帰の学習）
    実装 src/conformal/adaptive.py の fit_sigma, fit_quantile_pair

(A2) を実装で守るとは、「sigma_hat も分位点回帰も学習用データだけで学習し、較正用データを
渡さない」ことである。検査は 2 段に分ける。

1. 契約：fit_sigma / fit_quantile_pair の出力は学習用データと seed だけで決まる。
   較正用データをどう引いても（引く前でも後でも）同じ sigma_hat・分位点関数が返る。
2. 検出力：わざと較正用データで sigma_hat（分位点回帰）を学習すると、被覆は 1-alpha を
   大きく割る。同じモデルを学習用データで学習すれば 1-alpha 以上を保つ。
   モデルには過学習の極端な 1-NN を使う。リークの影響が最も強く出るので、
   「この検査はリークを見つけられる」ことも同時に確かめられる。

定数予測器によるリーク検出（tests/test_coverage_theory.py の test_最悪のモデルでも被覆する）は
絶対残差スコアの検査であり、ここはその第4章版にあたる。
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

from conformal import (
    conformal_quantile,
    cqr_score,
    fit_quantile_pair,
    fit_sigma,
    normalized_residual,
)
from conformal.datasets import heteroscedastic

N_TRAIN, N_CAL, N_TEST, N_TRIAL, ALPHA = 500, 100, 10_000, 300, 0.1


def _knn1_factory(q: float, seed: int | None) -> KNeighborsRegressor:
    """分位点回帰の代わりに使う 1-NN。目標分位点 q は無視する（lo = hi になる）。

    CQR の枠組みでは lo = hi でも構わない（区間 [lo - q_hat, hi + q_hat] は q_hat で広がる）。
    学習点では自分の y をそのまま返すので、較正用データで学習すると較正スコアが
    すべて 0 になり、リークの影響が最大になる。
    """
    return KNeighborsRegressor(n_neighbors=1)


# --------------------------------------------------------------------------
# 1. 契約：出力は学習用データと seed だけで決まる
# --------------------------------------------------------------------------

def test_fit_sigmaの出力は学習用データとseedだけで決まる():
    """較正用データを引く前と後、別の較正用データを引いた後で sigma_hat が一致する。"""
    rng = np.random.default_rng(0)
    X_tr, y_tr = heteroscedastic(N_TRAIN, rng)
    X_grid = np.random.default_rng(99).uniform(size=(300, 5))

    def fit():
        _, sig = fit_sigma(Ridge(alpha=1.0), LinearRegression(), X_tr, y_tr,
                           n_splits=5, random_state=1)
        return sig.predict(X_grid), sig.floor_

    s0, f0 = fit()
    heteroscedastic(N_CAL, rng)          # 較正用データを 1 組引く（fit には渡さない）
    s1, f1 = fit()
    heteroscedastic(N_CAL, rng)          # さらに別の較正用データ
    s2, f2 = fit()
    np.testing.assert_array_equal(s0, s1)
    np.testing.assert_array_equal(s0, s2)
    assert f0 == f1 == f2


def test_fit_quantile_pairの出力は学習用データとseedだけで決まる():
    rng = np.random.default_rng(0)
    X_tr, y_tr = heteroscedastic(N_TRAIN, rng)
    X_grid = np.random.default_rng(99).uniform(size=(300, 5))

    def fit():
        pair = fit_quantile_pair(_knn1_factory, X_tr, y_tr, q_lo=ALPHA / 2, q_hi=1 - ALPHA / 2)
        lo, hi = pair.predict(X_grid)
        return lo, hi

    lo0, hi0 = fit()
    heteroscedastic(N_CAL, rng)
    lo1, hi1 = fit()
    np.testing.assert_array_equal(lo0, lo1)
    np.testing.assert_array_equal(hi0, hi1)


# --------------------------------------------------------------------------
# 2. 検出力：較正用データで学習すると被覆が壊れる
# --------------------------------------------------------------------------

def _mean_coverage(rng, score_cal_and_test) -> float:
    """較正集合を引き直して平均被覆率を返す。score_cal_and_test(Xc, yc) は
    (較正スコア, テストスコア) を返す。リーク版はここで較正用データを学習に使う。"""
    cov = np.empty(N_TRIAL)
    for r in range(N_TRIAL):
        Xc, yc = heteroscedastic(N_CAL, rng)
        cal_s, test_s = score_cal_and_test(Xc, yc)
        q = conformal_quantile(cal_s, ALPHA)
        cov[r] = np.mean(test_s <= q)
    return float(cov.mean())


def test_sigmaを較正用データで学習すると被覆が壊れる():
    """学習用データで学習した sigma_hat なら被覆 >= 1-alpha、較正用データで学習すると大きく割る。

    リーク版：sigma_hat を較正用データの残差 |y - f(x)| に 1-NN で当てはめる。
    較正点では sigma_hat(x_i) = |y_i - f(x_i)| となり正規化残差がすべて約 1 に潰れるので、
    q_hat は約 1 になる。テスト点の sigma_hat は無関係な較正点の残差なので、被覆は
    おおよそ P(|e| <= |e'|) = 1/2 まで落ちる。
    """
    rng = np.random.default_rng(10)
    X_tr, y_tr = heteroscedastic(N_TRAIN, rng)
    X_te, y_te = heteroscedastic(N_TEST, rng)
    f_hat, sigma_ok = fit_sigma(Ridge(alpha=1.0), KNeighborsRegressor(n_neighbors=1),
                                X_tr, y_tr, n_splits=5, random_state=1)
    pred_te = f_hat.predict(X_te)
    floor = sigma_ok.floor_
    # 正しい手順ではモデルが固定なのでテストスコアは 1 回だけ計算すればよい
    test_s_ok = normalized_residual(y_te, pred_te, sigma_ok.predict(X_te))

    def proper(Xc, yc):
        return normalized_residual(yc, f_hat.predict(Xc), sigma_ok.predict(Xc)), test_s_ok

    def leaked(Xc, yc):
        # 較正用データで sigma_hat を学習する（やってはいけない例）
        resid_c = np.abs(yc - f_hat.predict(Xc))
        sig = KNeighborsRegressor(n_neighbors=1).fit(Xc, resid_c)
        s_c = np.maximum(sig.predict(Xc), floor)
        s_t = np.maximum(sig.predict(X_te), floor)
        return (normalized_residual(yc, f_hat.predict(Xc), s_c),
                normalized_residual(y_te, pred_te, s_t))

    cov_ok = _mean_coverage(np.random.default_rng(11), proper)
    cov_ng = _mean_coverage(np.random.default_rng(11), leaked)
    assert cov_ok >= 1 - ALPHA - 0.01, f"学習用データで学習した sigma で被覆が割れた: {cov_ok:.4f}"
    assert cov_ng < 1 - ALPHA - 0.10, (
        f"較正用データで学習しても被覆が落ちない ({cov_ng:.4f})。この検査ではリークを検出できていない"
    )


def test_分位点回帰を較正用データで学習すると被覆が壊れる():
    """学習用データで学習した分位点関数なら被覆 >= 1-alpha、較正用データで学習すると大きく割る。

    リーク版：1-NN を較正用データで学習すると較正点で lo = hi = y_i となり、
    CQR スコアがすべて 0、q_hat = 0 になる。テスト点の区間は 1 点に潰れ、被覆はほぼ 0 になる。
    """
    rng = np.random.default_rng(20)
    X_tr, y_tr = heteroscedastic(N_TRAIN, rng)
    X_te, y_te = heteroscedastic(N_TEST, rng)
    pair_ok = fit_quantile_pair(_knn1_factory, X_tr, y_tr, q_lo=ALPHA / 2, q_hi=1 - ALPHA / 2)
    lo_te, hi_te = pair_ok.predict(X_te)
    test_s_ok = cqr_score(y_te, lo_te, hi_te)

    def proper(Xc, yc):
        lo_c, hi_c = pair_ok.predict(Xc)
        return cqr_score(yc, lo_c, hi_c), test_s_ok

    def leaked(Xc, yc):
        # 較正用データで分位点回帰を学習する（やってはいけない例）
        pair = fit_quantile_pair(_knn1_factory, Xc, yc, q_lo=ALPHA / 2, q_hi=1 - ALPHA / 2)
        lo_c, hi_c = pair.predict(Xc)
        lo_t, hi_t = pair.predict(X_te)
        return cqr_score(yc, lo_c, hi_c), cqr_score(y_te, lo_t, hi_t)

    cov_ok = _mean_coverage(np.random.default_rng(21), proper)
    cov_ng = _mean_coverage(np.random.default_rng(21), leaked)
    assert cov_ok >= 1 - ALPHA - 0.01, f"学習用データで学習した分位点関数で被覆が割れた: {cov_ok:.4f}"
    assert cov_ng < 1 - ALPHA - 0.10, (
        f"較正用データで学習しても被覆が落ちない ({cov_ng:.4f})。この検査ではリークを検出できていない"
    )
