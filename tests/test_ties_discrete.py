"""同点（離散スコア）があるときの被覆保証の非対称性。

対応箇所: 卒論 第3章 補題3.1・定理3.2 / CLAUDE.md「よくあるバグ」#5

理論が言っていること:
    下界 P(S_{n+1} <= S_(k)) >= 1-alpha は、順位が一様であることと切り上げの定義
    だけから出るので、**同点があっても成り立つ**。同点は「順位が k 以下」の事象を
    増やす方向にしか効かないためである。

    一方 上界 <= 1-alpha + 1/(n+1) は、順位がちょうど k 以下である確率が
    k/(n+1) に「一致する」ことを使うので、**同点があると破れる**。

    したがって同点処理を誤っても被覆率のテストでは検出できない（区間が広がる方向に
    しか壊れないため）。上界を見て初めて落ちる。この非対称性を検査する。

同値な言い換え: 共形 p 値は同点があっても超一様（stochastically larger than uniform）
であり、P(p <= alpha) <= alpha は保たれる（Lei et al. 2018）。
"""

import numpy as np
import pytest

from conformal import conformal_pvalue, conformal_quantile

N_CAL, ALPHA, N_TRIAL = 18, 0.10, 100_000
K = int(np.ceil((N_CAL + 1) * (1 - ALPHA)))          # = 18
LOWER, UPPER = 1 - ALPHA, 1 - ALPHA + 1 / (N_CAL + 1)


def _coverage(scores: np.ndarray) -> float:
    """形状 (R, n+1) の交換可能なスコアから、被覆率 P(S_{n+1} <= q_hat) を推定する。

    各行の先頭 n 個を較正集合、最後の1個を新しい点とみなす。
    """
    cal, new = scores[:, :N_CAL], scores[:, N_CAL]
    q = np.array([conformal_quantile(row, ALPHA) for row in cal])
    return float(np.mean(new <= q))


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260910)


def test_連続スコアでは上下界の両方が成り立つ(rng):
    """同点が確率0なら、理論どおり両側で挟まれる。"""
    cov = _coverage(rng.normal(size=(N_TRIAL, N_CAL + 1)))
    assert LOWER <= cov <= UPPER, f"被覆率 {cov:.4f} が [{LOWER:.4f}, {UPPER:.4f}] の外"
    # 理論値 k/(n+1) の近傍にあること（モンテカルロ誤差 4SE を許容）
    theo = K / (N_CAL + 1)
    assert abs(cov - theo) < 4 * np.sqrt(theo * (1 - theo) / N_TRIAL)


@pytest.mark.parametrize("n_levels", [2, 3, 5])
def test_同点があっても下界は保たれる(rng, n_levels):
    """離散スコア（同点だらけ）でも被覆保証は破れない。ここが保証の本体である。"""
    scores = rng.integers(0, n_levels, size=(N_TRIAL, N_CAL + 1)).astype(float)
    cov = _coverage(scores)
    assert cov >= LOWER, f"水準数 {n_levels} で被覆率 {cov:.4f} が下界 {LOWER:.4f} を割った"


@pytest.mark.parametrize("n_levels", [2, 3])
def test_同点があると上界は破れる(rng, n_levels):
    """上界は連続性（同点確率0）に依存するので、離散スコアでは成り立たない。

    これは実装の誤りではなく理論どおりの挙動である。上界を無条件に主張してはならない、
    ということを固定するためのテスト。
    """
    scores = rng.integers(0, n_levels, size=(N_TRIAL, N_CAL + 1)).astype(float)
    cov = _coverage(scores)
    assert cov > UPPER, f"水準数 {n_levels} で被覆率 {cov:.4f} が上界 {UPPER:.4f} を超えなかった"


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_共形p値は同点があっても超一様(rng, alpha):
    """P(p <= alpha) <= alpha が離散スコアでも保たれる（Lei et al. 2018）。

    下界が同点に強いことの、p 値による言い換え。第3章ではこの形で述べる。
    """
    scores = rng.integers(0, 3, size=(20_000, N_CAL + 1)).astype(float)
    p = np.array([conformal_pvalue(row[:N_CAL], row[N_CAL]) for row in scores])
    assert np.mean(p <= alpha) <= alpha + 1e-12, "共形 p 値の超一様性が破れた"
