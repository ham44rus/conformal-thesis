"""jackknife+ と CV+ の予測区間。

対応箇所:
    卒論 第4章 jackknife+ と CV+ の節（次の run で書く）
    実験 E6（sec:e6。データが少ないときの分割型との比較）
    仕様書 specs/E6_jackknife_cv.md
出典: Barber, Candès, Ramdas, Tibshirani (2021)（refs.bib の barber2021jackknife）

分割等角予測（split.py）はデータを学習用と較正用に分けるので、データが少ないと
どちらも小さくなる。jackknife+ と CV+ はデータを分けず、1 点（CV+ では 1 組）を
抜いて学習したモデルで「抜いた点の残差」を作り、全点を較正にも学習にも使う。
その代わり、学習の回数が増える（jackknife+ は n 回、CV+ は K 回）。

保証（出典の定理による。交換可能性と、学習アルゴリズムが点の順序によらないことを仮定）:
    jackknife+ : 被覆確率 >= 1 - 2*alpha
    CV+        : 被覆確率 >= 1 - 2*alpha - sqrt(2/n)   （n が組の数 K で割り切れるとき）
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .split import conformal_index

__all__ = ["loo_interval", "jackknife_plus_interval", "cv_plus_interval"]


def loo_interval(
    loo_pred: np.ndarray, residuals: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""抜いた点の予測と残差から jackknife+ / CV+ の区間を作る（卒論 sec:e6。両者で共通）。

    loo_pred[i, j] = \mu_{-i}(x_j)（点 i を含まないモデルによるテスト点 j の予測）、
    residuals[i]   = R_i = |Y_i - \mu_{-i}(X_i)| として、テスト点 x ごとに

        k = \lceil (n+1)(1-\alpha) \rceil        （卒論 eq:khat。conformal_index で求める）
        l = \lfloor (n+1)\alpha \rfloor = n+1-k  （卒論 prop:excess の証明の恒等式）
        hi(x) = \{ \mu_{-i}(x) + R_i \}_{i=1..n} の小さい方から k 番目   （k > n なら +inf）
        lo(x) = \{ \mu_{-i}(x) - R_i \}_{i=1..n} の小さい方から l 番目   （l = 0 なら -inf）

    出典（barber2021jackknife）の定義どおり。線形補間の入る分位点関数は使わず、並べ替えで順序統計量を取る
    （CLAUDE.md「絶対に守ること 1」）。

    学習用データによらないモデル（\mu_{-i} = \mu）では、hi = \mu(x) + S_{(k)}、
    lo = \mu(x) - S_{(n+1-l)} = \mu(x) - S_{(k)} となり、分割型の区間と一致する。

    Parameters
    ----------
    loo_pred : 形状 (n, m)。m はテスト点の数
    residuals : 形状 (n,)
    alpha : 有意水準。0 < alpha < 1

    Returns
    -------
    (lo, hi) : どちらも形状 (m,)
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha は (0,1) でなければならない: {alpha}")
    loo_pred = np.asarray(loo_pred, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if loo_pred.ndim != 2 or residuals.shape != (loo_pred.shape[0],):
        raise ValueError(
            f"loo_pred は (n, m)、residuals は (n,) でなければならない: "
            f"{loo_pred.shape}, {residuals.shape}"
        )

    n, m = loo_pred.shape
    k = conformal_index(n, alpha)
    l = n + 1 - k

    if k > n:
        hi = np.full(m, np.inf)
    else:
        # 列（テスト点）ごとに昇順に並べ、第 k 順序統計量（0-indexed で k-1）を取る
        hi = np.sort(loo_pred + residuals[:, None], axis=0)[k - 1]
    if l == 0:
        lo = np.full(m, -np.inf)
    else:
        lo = np.sort(loo_pred - residuals[:, None], axis=0)[l - 1]
    return lo, hi


def _check_inputs(X, y, X_test, alpha) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """引数の形と alpha を確かめて配列にして返す。"""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha は (0,1) でなければならない: {alpha}")
    X, y, X_test = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(X_test, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X は (n, d)、y は (n,) でなければならない: {X.shape}, {y.shape}")
    if X.shape[0] < 2:
        raise ValueError(f"データは 2 点以上必要: n={X.shape[0]}")
    if X_test.ndim != 2 or X_test.shape[1] != X.shape[1]:
        raise ValueError(f"X_test は (m, {X.shape[1]}) でなければならない: {X_test.shape}")
    return X, y, X_test


def _check_seeds(seeds: Sequence[int], n_fits: int) -> list[int]:
    seeds = [int(s) for s in seeds]
    if len(seeds) != n_fits:
        raise ValueError(f"seeds の長さは学習の回数 {n_fits} でなければならない: {len(seeds)}")
    return seeds


def jackknife_plus_interval(
    model_factory: Callable[[int], object],
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    r"""jackknife+ の予測区間（卒論 sec:e6、出典 barber2021jackknife）。

    各 i = 1..n について、点 i を除いた n-1 点で \mu_{-i} を学習し、
        R_i = |Y_i - \mu_{-i}(X_i)|
    とする（点 i は \mu_{-i} の学習に入らないので、R_i はデータリークのない残差）。
    区間は loo_interval の式。被覆確率は 1 - 2\alpha 以上（出典の定理による）。

    Parameters
    ----------
    model_factory : seed を受け取り、学習前のモデル（fit・predict を持つ）を返す関数
    X, y : 形状 (n, d), (n,) のデータ
    X_test : 形状 (m, d) のテスト点
    alpha : 有意水準
    seeds : 長さ n。i 番目のモデルに seeds[i] を渡す。保証は学習アルゴリズムが
            すべての i で同じであることを仮定するので、実験では同じ値を並べて渡す

    Returns
    -------
    (lo, hi, n_fits) : 形状 (m,) の下端・上端と、学習の回数 n
    """
    X, y, X_test = _check_inputs(X, y, X_test, alpha)
    n = X.shape[0]
    seeds = _check_seeds(seeds, n)

    loo_pred = np.empty((n, X_test.shape[0]))
    residuals = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i  # 点 i を抜く（残りの並び順は変えない）
        model = model_factory(seeds[i])
        model.fit(X[keep], y[keep])
        residuals[i] = abs(y[i] - model.predict(X[i : i + 1])[0])
        loo_pred[i] = model.predict(X_test)

    lo, hi = loo_interval(loo_pred, residuals, alpha)
    return lo, hi, n


def cv_plus_interval(
    model_factory: Callable[[int], object],
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    n_folds: int,
    rng: np.random.Generator,
    seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    r"""CV+ の予測区間（卒論 sec:e6、出典 barber2021jackknife）。

    データをランダムに K 組 S_1..S_K に分け（rng で 0..n-1 を並べ替え、ほぼ等分に切る）、
    組 S_j を除いた点で \mu_{-S_j} を学習する。点 i の属する組を S(i) として
        R_i = |Y_i - \mu_{-S(i)}(X_i)|,   \mu_{-i} := \mu_{-S(i)}
    とおき、区間は loo_interval の式。被覆確率は 1 - 2\alpha - \sqrt{2/n} 以上（出典の定理による）。
    ただし出典の定理は各組が同じ大きさ（n が K で割り切れる）ことを仮定している。割り切れない n も
    受け付けるが（組の大きさの差は 1 以下）、そのときこの下限は出典の定理では裏付けられない。
    K = n なら各組が 1 点になり、jackknife+ と同じになる。

    Parameters
    ----------
    n_folds : 組の数 K。2 <= K <= n
    rng : 組分けに使う乱数生成器（並べ替えを 1 回引く）
    seeds : 長さ K。j 番目の組を抜いたモデルに seeds[j] を渡す
    その他は jackknife_plus_interval と同じ

    Returns
    -------
    (lo, hi, n_fits) : 形状 (m,) の下端・上端と、学習の回数 K
    """
    X, y, X_test = _check_inputs(X, y, X_test, alpha)
    n = X.shape[0]
    if not 2 <= n_folds <= n:
        raise ValueError(f"n_folds は 2 以上 n={n} 以下でなければならない: {n_folds}")
    seeds = _check_seeds(seeds, n_folds)

    folds = np.array_split(rng.permutation(n), n_folds)
    loo_pred = np.empty((n, X_test.shape[0]))
    residuals = np.empty(n)
    for j, fold in enumerate(folds):
        keep = np.ones(n, dtype=bool)
        keep[fold] = False  # 組 S_j を抜く（残りの並び順は変えない）
        model = model_factory(seeds[j])
        model.fit(X[keep], y[keep])
        residuals[fold] = np.abs(y[fold] - model.predict(X[fold]))
        loo_pred[fold] = model.predict(X_test)  # 組 S_j のどの点にも同じ予測を使う

    lo, hi = loo_interval(loo_pred, residuals, alpha)
    return lo, hi, n_folds
