"""適応的な区間幅を作るための部品。sigma(x) の推定と CQR の分位点回帰。

対応箇所:
    卒論 第4章「拡張」正規化残差スコアの節（sec:normalized-score）・CQR の節（sec:cqr）
    実験E3（異分散データと条件付き被覆の破綻）で使う

ここに置くのは**スコアの材料を作る**部分だけである。スコアそのものの定義は
scores.py、共形分位点の計算は split.py にある。役割を混ぜないこと。

この module は「どのモデルを使うか」「どこで打ち切るか」を決めない。
すべて引数で受け取る。実験ごとの選択は仕様書と実験スクリプト側に置き、
部品は選択を持たないようにしてある。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

__all__ = [
    "out_of_fold_predict",
    "fit_sigma",
    "SigmaEstimator",
    "fit_quantile_pair",
    "QuantilePair",
]


def out_of_fold_predict(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int | None = None,
) -> np.ndarray:
    """K分割の out-of-fold 予測を返す（卒論 sec:normalized-score）。

    各点の予測を、**その点を含まない** fold で学習したモデルから作る。
    自分自身の情報が予測に入らないので、残差が過学習で過小評価されない。

    sigma(x) の推定にこれが要る理由：学習集合にそのまま当てた残差は過学習のぶん
    小さく出る。縮み方が x によらず一様なら、sigma の定数倍は共形分位点 q_hat に
    吸収されて区間は変わらない（sigma を定数倍しても区間が同じことは
    tests/test_adaptive.py で検査している）。問題になるのは、縮み方が x によって
    一様でない場合である。柔軟なモデルほど学習点の残差はほぼ 0 になり、
    sigma の形が歪んで区間幅の x への適応が効かなくなる。被覆保証自体は
    較正集合が守るので壊れないが、区間の有用性が落ちる。

    Parameters
    ----------
    model : 未学習の推定器。fold ごとに clone して使う（呼び出し元は汚さない）
    n_splits : 分割数。大きいほど残差の偏りは減るが学習回数が増える
    random_state : 分割の乱数。再現性のため呼び出し側が seed から導出して渡すこと

    Returns
    -------
    y と同じ長さの out-of-fold 予測
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X と y の長さが一致しない")
    if n_splits < 2:
        raise ValueError("n_splits は2以上でなければならない")

    oof = np.empty(y.shape[0], dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, te in kf.split(X):
        m = clone(model)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


class SigmaEstimator:
    """sigma(x) の推定器。predict(X) が正の尺度を返す（卒論 sec:normalized-score。ラベルは仮に eq:sigma-hat）。

    正規化残差スコア S = |y - f(x)| / sigma(x) の分母を作る。
    区間は f(x) ± q_hat * sigma(x) になるので、sigma(x) が x に応じて動けば
    区間幅も動く。これが「適応的」の中身である。

    クリップの下限を持つ理由：残差の大きさを回帰した値は容易に 0 や負になり、
    そのまま割ると発散するか scores.normalized_residual が ValueError を投げる。
    下限で打ち切った点の割合は `clip_rate` に残るので、実験側で報告できる。

    Attributes
    ----------
    model_ : 学習済みの尺度回帰モデル
    floor_ : クリップの下限値（実際に使われた値）
    clip_rate_ : 学習データ上でクリップが発動した点の割合
    """

    def __init__(self, model_: BaseEstimator, floor_: float, clip_rate_: float) -> None:
        self.model_ = model_
        self.floor_ = float(floor_)
        self.clip_rate_ = float(clip_rate_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """sigma(x) を返す。下限でクリップ済みなので必ず正。"""
        return np.maximum(self.model_.predict(np.asarray(X)), self.floor_)

    def clip_rate_on(self, X: np.ndarray) -> float:
        """与えた X 上でクリップが発動する点の割合。報告用。"""
        raw = self.model_.predict(np.asarray(X))
        return float(np.mean(raw < self.floor_))


def fit_sigma(
    base_model: BaseEstimator,
    sigma_model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int | None = None,
    floor: float | None = None,
    floor_quantile: float = 0.05,
    target: str = "abs",
) -> tuple[BaseEstimator, SigmaEstimator]:
    """下敷きモデルと sigma(x) を学習集合から作る（卒論 sec:normalized-score）。

    手順：

    1. `base_model` を学習集合全体で学習する（これが f(x)）
    2. 同じ学習集合の **out-of-fold 残差** を作る（自分自身の情報を使わない）
    3. その残差の大きさを `sigma_model` で回帰する
    4. 下限でクリップして正にする

    **較正集合を渡してはいけない。** 較正集合が sigma の推定に入るとデータリークに
    なり、有限標本被覆保証が壊れる（CLAUDE.md「2. データリークの禁止」）。
    ここで受け取る X, y は学習集合だけである。

    Parameters
    ----------
    target : 尺度回帰の目的変数。
        "abs"  -> |残差| をそのまま回帰する（sigma の推定値がそのまま出る）
        "sq"   -> 残差^2 を回帰し、predict の平方根を取る（分散の回帰にあたる）
        どちらを使うかは実験の設計判断なので既定を "abs" にしてあるだけである。
    floor : クリップの下限。None なら out-of-fold 残差の `floor_quantile` 分位点を使う。
        データの尺度に自動で追随するので、DGP を変えても効く。
    floor_quantile : floor=None のときに使う分位点。

    Returns
    -------
    (学習済みの f, SigmaEstimator)
    """
    X, y = np.asarray(X), np.asarray(y)
    if target not in ("abs", "sq"):
        raise ValueError('target は "abs" か "sq" でなければならない')

    # 1. 下敷きモデルは学習集合全体で学習する
    base = clone(base_model)
    base.fit(X, y)

    # 2. out-of-fold 残差。ここが過学習の抑制点
    oof_pred = out_of_fold_predict(base_model, X, y, n_splits=n_splits,
                                   random_state=random_state)
    resid = np.abs(y - oof_pred)

    # 3. 残差の大きさを回帰する
    sm = clone(sigma_model)
    sm.fit(X, resid if target == "abs" else resid ** 2)

    # 4. 下限を決めてクリップする。
    # 分位点で決めるのは残差の記述統計であって共形分位点ではない
    # （CLAUDE.md「絶対に守ること 1」が禁じているのは共形分位点の計算のこと）。
    if floor is None:
        floor = float(np.quantile(resid, floor_quantile))
        if floor <= 0.0:
            floor = float(np.mean(resid)) * 1e-3
    if floor <= 0.0:
        raise ValueError("floor は正でなければならない")

    inner = _SqrtWrapper(sm) if target == "sq" else sm
    raw = inner.predict(X)
    clip_rate = float(np.mean(raw < floor))
    return base, SigmaEstimator(inner, floor, clip_rate)


class _SqrtWrapper:
    """残差^2 を回帰したモデルの出力を sigma の尺度に戻す薄いラッパ。"""

    def __init__(self, model: BaseEstimator) -> None:
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(self.model.predict(X), 0.0))


class QuantilePair:
    """CQR の下側・上側の分位点回帰の組（卒論 sec:cqr, Romano et al. 2019）。

    下側と上側は**別々のモデルとして学習する**。sklearn の分位点損失は片側ずつしか
    最適化しないため、1本のモデルで両側を出すことはできない。

    別々に学習した結果、点によっては q_lo(x) > q_hi(x) と**分位点が交差する**。
    分位点回帰の既知の性質であって実装のバグではない。predict は交差した点で
    2値を入れ替えて lo <= hi を保証し、交差した点の割合を `crossing_rate_on` で
    返す。この割合は実験側で必ず報告すること（黙って入れ替えると、
    分位点回帰がどれだけ効いていないかが見えなくなる）。

    Attributes
    ----------
    model_lo_, model_hi_ : 学習済みの下側・上側モデル
    q_lo_, q_hi_ : 目標分位点（例 0.05 と 0.95）
    """

    def __init__(self, model_lo_, model_hi_, q_lo_: float, q_hi_: float) -> None:
        self.model_lo_ = model_lo_
        self.model_hi_ = model_hi_
        self.q_lo_ = float(q_lo_)
        self.q_hi_ = float(q_hi_)

    def predict_raw(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """入れ替えをせず、学習したままの (lo, hi) を返す。交差率の計算に使う。"""
        X = np.asarray(X)
        return self.model_lo_.predict(X), self.model_hi_.predict(X)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) を返す。交差した点は入れ替えて lo <= hi を保証する。"""
        lo, hi = self.predict_raw(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def crossing_rate_on(self, X: np.ndarray) -> float:
        """与えた X で分位点が交差した点の割合（報告義務）。"""
        lo, hi = self.predict_raw(X)
        return float(np.mean(lo > hi))


def fit_quantile_pair(
    model_factory: Callable[[float, int | None], BaseEstimator],
    X: np.ndarray,
    y: np.ndarray,
    q_lo: float,
    q_hi: float,
    random_state_lo: int | None = None,
    random_state_hi: int | None = None,
) -> QuantilePair:
    """下側・上側の分位点回帰を別々に学習する（卒論 sec:cqr）。

    **較正集合を渡してはいけない。** 分位点回帰の学習に較正集合が入ると
    データリークになる（CLAUDE.md「2. データリークの禁止」）。

    Parameters
    ----------
    model_factory : (目標分位点, random_state) を受け取り未学習の推定器を返す関数。
        どのモデル・どのハイパーパラメータを使うかは実験側の判断なので、
        ここでは決めずに呼び出し元から受け取る。
    q_lo, q_hi : 目標分位点。CQR では通常 alpha/2 と 1-alpha/2
    random_state_lo, random_state_hi : 2本のモデルの乱数。再現性のため
        呼び出し側が seed から決定的に導出して渡すこと（CLAUDE.md 再現性ルール）
    """
    if not 0.0 < q_lo < q_hi < 1.0:
        raise ValueError("0 < q_lo < q_hi < 1 でなければならない")
    X, y = np.asarray(X), np.asarray(y)

    m_lo = model_factory(q_lo, random_state_lo)
    m_lo.fit(X, y)
    m_hi = model_factory(q_hi, random_state_hi)
    m_hi.fit(X, y)
    return QuantilePair(m_lo, m_hi, q_lo, q_hi)
