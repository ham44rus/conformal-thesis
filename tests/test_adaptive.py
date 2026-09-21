"""sigma(x) 推定と CQR 分位点回帰の検証（実験E3 の部品）。

対応箇所: 卒論 第4章 sec:normalized-score・sec:cqr、実装 src/conformal/adaptive.py

理論から導かれる性質を検査する：

- sigma が全点で同じ定数なら、正規化残差スコアは絶対残差スコアと
  **同じ予測区間**を与える（重みなしへの帰着。CLAUDE.md「よくあるバグ」#6 と同じ型）
- out-of-fold 予測には自分自身の目的変数が使われていない
- sigma は必ず正（クリップの下限を下回らない）
- 分位点回帰が交差したとき predict は lo <= hi を保証し、交差率を正しく返す
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from conformal import (
    absolute_residual,
    conformal_quantile,
    fit_quantile_pair,
    fit_sigma,
    interval_from_quantile,
    normalized_residual,
    out_of_fold_predict,
)
from conformal.datasets import heteroscedastic


# --------------------------------------------------------------------------
# 正規化残差 → 絶対残差への帰着
# --------------------------------------------------------------------------

def test_sigmaが定数なら正規化残差は絶対残差と同じ区間を与える():
    r"""重みなしへの帰着。

    S = |y - f(x)| / c は絶対残差スコアの定数倍なので、共形分位点も同じ倍率で
    スケールする。区間は f(x) ± q_hat * c となり、c が消えて絶対残差の区間
    f(x) ± q_hat_abs と一致する。**c の値によらず一致する**ことを確かめる。

    厳密には normalized_residual は 0 除算を避けるため `scale + eps`
    （eps=1e-8）で割るので、一致は eps/c の相対誤差までである。
    sigma の下限が eps より十分大きければ実害はないが、区間を組み立てる側は
    同じ規約（分母に eps を足す）で揃えないとここがずれる。
    """
    rng = np.random.default_rng(0)
    n_cal, alpha = 500, 0.1
    y = rng.normal(0.0, 1.0, size=n_cal)
    pred = rng.normal(0.0, 0.3, size=n_cal)

    # 絶対残差スコアでの区間
    q_abs = conformal_quantile(absolute_residual(y, pred), alpha)
    lo_abs, hi_abs = interval_from_quantile(pred, q_abs)

    for c in (0.5, 1.0, 2.0, 17.3):
        scale = np.full(n_cal, c)
        q_norm = conformal_quantile(normalized_residual(y, pred, scale), alpha)
        # 正規化残差の区間は f(x) ± q_hat * sigma(x)
        lo_norm, hi_norm = pred - q_norm * c, pred + q_norm * c

        assert q_norm == pytest.approx(q_abs / c, rel=1e-6), f"c={c} で分位点がずれた"
        np.testing.assert_allclose(lo_norm, lo_abs, rtol=1e-6)
        np.testing.assert_allclose(hi_norm, hi_abs, rtol=1e-6)

        # 分母の規約を揃えれば厳密に一致する
        assert q_norm == pytest.approx(q_abs / (c + 1e-8), rel=1e-12)


def test_sigmaが定数なら被覆率も一致する():
    """区間が一致するので被覆率も一致するはず（上の系）。"""
    rng = np.random.default_rng(1)
    n, alpha = 400, 0.2
    y_cal = rng.normal(size=n)
    pred_cal = np.zeros(n)
    y_te = rng.normal(size=5000)
    pred_te = np.zeros(5000)

    q_abs = conformal_quantile(absolute_residual(y_cal, pred_cal), alpha)
    c = 3.0
    q_norm = conformal_quantile(normalized_residual(y_cal, pred_cal, np.full(n, c)), alpha)

    cov_abs = np.mean(np.abs(y_te - pred_te) <= q_abs)
    cov_norm = np.mean(np.abs(y_te - pred_te) <= q_norm * c)
    assert cov_abs == pytest.approx(cov_norm)


# --------------------------------------------------------------------------
# out-of-fold 予測
# --------------------------------------------------------------------------

def test_out_of_fold予測に自分自身のyは使われない():
    """各点の予測が、その点を含まない fold から作られていることを検査する。

    y をランダムな定数ラベルにして DummyRegressor(mean) を使うと、
    fold 内平均が返る。自分自身を含めた全体平均とは一致しないはず。
    """
    rng = np.random.default_rng(2)
    n = 50
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)

    oof = out_of_fold_predict(DummyRegressor(strategy="mean"), X, y,
                              n_splits=5, random_state=0)

    assert oof.shape == (n,)
    # 全体平均（自分自身を含む）とは一致しない
    assert not np.allclose(oof, y.mean())
    # DummyRegressor は fold の平均を返すので、oof の値は高々 n_splits 種類
    assert len(np.unique(np.round(oof, 12))) <= 5


def test_out_of_fold予測は同じrandom_stateで再現する():
    """再現性ルール（CLAUDE.md 3）。分割の乱数を固定すれば同じ結果になる。"""
    rng = np.random.default_rng(3)
    X, y = heteroscedastic(200, rng)
    a = out_of_fold_predict(LinearRegression(), X, y, n_splits=5, random_state=7)
    b = out_of_fold_predict(LinearRegression(), X, y, n_splits=5, random_state=7)
    np.testing.assert_array_equal(a, b)


def test_out_of_fold予測は呼び出し元のモデルを学習済みにしない():
    """clone して使っているので、渡した推定器は未学習のままのはず。"""
    rng = np.random.default_rng(4)
    X, y = heteroscedastic(100, rng)
    model = LinearRegression()
    out_of_fold_predict(model, X, y, n_splits=5, random_state=0)
    with pytest.raises(Exception):
        model.predict(X)


# --------------------------------------------------------------------------
# sigma(x) の推定
# --------------------------------------------------------------------------

def test_sigmaは必ず正でクリップ下限を下回らない():
    """normalized_residual は scale<=0 で ValueError を投げる。その前提を守る。"""
    rng = np.random.default_rng(5)
    X, y = heteroscedastic(300, rng)

    _, sigma = fit_sigma(LinearRegression(), LinearRegression(), X, y,
                         n_splits=5, random_state=0)

    X_new, _ = heteroscedastic(1000, rng)
    s = sigma.predict(X_new)
    assert np.all(s > 0.0)
    assert np.all(s >= sigma.floor_)
    # 下限が正であることも確認（0 だと割り算で発散する）
    assert sigma.floor_ > 0.0


def test_sigmaは異分散の向きを捉える():
    r"""DGP は sigma(x) = 0.1 + 1.0 * X1 なので、X1 が大きいほど sigma も大きい。

    推定が向きすら合っていなければ、正規化残差スコアを使う意味がない。
    """
    rng = np.random.default_rng(6)
    X, y = heteroscedastic(2000, rng)

    _, sigma = fit_sigma(GradientBoostingRegressor(random_state=0),
                         GradientBoostingRegressor(random_state=0),
                         X, y, n_splits=5, random_state=0)

    X_new, _ = heteroscedastic(4000, rng)
    s = sigma.predict(X_new)
    lo_group = s[X_new[:, 0] < 0.2].mean()
    hi_group = s[X_new[:, 0] > 0.8].mean()
    assert hi_group > 2.0 * lo_group, f"X1 の上下で sigma が動いていない: {lo_group:.3f} -> {hi_group:.3f}"


def test_sigmaはtarget_sqでも正の尺度を返す():
    """残差^2 を回帰する経路でも、平方根を取って正の尺度が返ること。"""
    rng = np.random.default_rng(7)
    X, y = heteroscedastic(300, rng)
    _, sigma = fit_sigma(LinearRegression(), LinearRegression(), X, y,
                         n_splits=5, random_state=0, target="sq")
    s = sigma.predict(X)
    assert np.all(s > 0.0)


def test_クリップ発動率が記録される():
    """報告義務のための値が入っていること。"""
    rng = np.random.default_rng(8)
    X, y = heteroscedastic(300, rng)
    _, sigma = fit_sigma(LinearRegression(), LinearRegression(), X, y,
                         n_splits=5, random_state=0)
    assert 0.0 <= sigma.clip_rate_ <= 1.0
    assert 0.0 <= sigma.clip_rate_on(X) <= 1.0


def test_fit_sigmaは同じseedで再現する():
    rng1 = np.random.default_rng(9)
    X, y = heteroscedastic(300, rng1)
    _, s1 = fit_sigma(LinearRegression(), LinearRegression(), X, y,
                      n_splits=5, random_state=11)
    _, s2 = fit_sigma(LinearRegression(), LinearRegression(), X, y,
                      n_splits=5, random_state=11)
    np.testing.assert_allclose(s1.predict(X), s2.predict(X))


# --------------------------------------------------------------------------
# CQR の分位点回帰
# --------------------------------------------------------------------------

def _gbm_quantile(q: float, seed: int | None):
    return GradientBoostingRegressor(loss="quantile", alpha=q, random_state=seed)


def test_分位点回帰はlo以下hi以上を返す():
    """predict は交差を入れ替えるので、常に lo <= hi が成り立つ。"""
    rng = np.random.default_rng(10)
    X, y = heteroscedastic(500, rng)

    pair = fit_quantile_pair(_gbm_quantile, X, y, q_lo=0.05, q_hi=0.95,
                             random_state_lo=1, random_state_hi=2)

    X_new, _ = heteroscedastic(1000, rng)
    lo, hi = pair.predict(X_new)
    assert np.all(lo <= hi)


def test_交差した点は入れ替えられ交差率が返る():
    """交差を人工的に起こして、入れ替えと交差率の両方を検査する。

    q_lo に大きい分位点、q_hi に小さい分位点を学習させれば、ほぼ全点で交差する。
    （fit_quantile_pair は q_lo < q_hi を要求するので、ここでは
    QuantilePair を直接組み立てて predict の振る舞いだけを見る）
    """
    from conformal.adaptive import QuantilePair

    rng = np.random.default_rng(11)
    X, y = heteroscedastic(400, rng)
    m_hi_like = _gbm_quantile(0.9, 0).fit(X, y)     # 大きい方を lo 側に置く
    m_lo_like = _gbm_quantile(0.1, 0).fit(X, y)     # 小さい方を hi 側に置く
    pair = QuantilePair(m_hi_like, m_lo_like, 0.9, 0.1)

    raw_lo, raw_hi = pair.predict_raw(X)
    lo, hi = pair.predict(X)

    assert np.all(lo <= hi), "入れ替えが効いていない"
    assert pair.crossing_rate_on(X) == pytest.approx(np.mean(raw_lo > raw_hi))
    assert pair.crossing_rate_on(X) > 0.9, "この構成ならほぼ全点で交差するはず"
    # 入れ替え後の集合は元の2値の集合と一致する（値を捨てていない）
    np.testing.assert_allclose(np.sort(np.c_[lo, hi], axis=1),
                               np.sort(np.c_[raw_lo, raw_hi], axis=1))


def test_交差がなければ交差率はゼロ():
    """線形の分位点回帰なら交差はまず起きない。対照として置く。"""
    from sklearn.linear_model import QuantileRegressor

    rng = np.random.default_rng(12)
    X, y = heteroscedastic(300, rng)

    def factory(q, seed):
        return QuantileRegressor(quantile=q, alpha=0.0, solver="highs")

    pair = fit_quantile_pair(factory, X, y, q_lo=0.05, q_hi=0.95)
    assert pair.crossing_rate_on(X) == 0.0


def test_分位点回帰は目標分位点の順序を検査する():
    rng = np.random.default_rng(13)
    X, y = heteroscedastic(100, rng)
    with pytest.raises(ValueError):
        fit_quantile_pair(_gbm_quantile, X, y, q_lo=0.95, q_hi=0.05)


def test_分位点回帰は同じseedで再現する():
    """再現性ルール（CLAUDE.md 3）。"""
    rng = np.random.default_rng(14)
    X, y = heteroscedastic(300, rng)
    a = fit_quantile_pair(_gbm_quantile, X, y, 0.05, 0.95,
                          random_state_lo=5, random_state_hi=6).predict(X)
    b = fit_quantile_pair(_gbm_quantile, X, y, 0.05, 0.95,
                          random_state_lo=5, random_state_hi=6).predict(X)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_分位点回帰は名目のカバー率を概ね満たす():
    r"""学習データ上で、[q_lo(x), q_hi(x)] に入る点の割合が q_hi - q_lo に近いこと。

    分位点回帰そのものの妥当性の確認。ここが大きく外れていれば、
    CQR の共形補正量 q_hat が異常に大きくなって適応性の意味が薄れる。
    （分位点回帰は有限標本保証を持たないので、緩い許容で見る）
    """
    rng = np.random.default_rng(15)
    X, y = heteroscedastic(3000, rng)
    pair = fit_quantile_pair(_gbm_quantile, X, y, q_lo=0.05, q_hi=0.95,
                             random_state_lo=1, random_state_hi=2)
    lo, hi = pair.predict(X)
    inside = float(np.mean((y >= lo) & (y <= hi)))
    assert 0.82 < inside < 0.98, f"学習データ上の被覆が名目 0.90 から離れすぎ: {inside:.3f}"
