"""重み付き共形分位点（src/conformal/weighted.py）の検証テスト。

対応箇所: 卒論 第4章 共変量シフトの節（sec:covariate-shift）, 実験E4（sec:e4）
          仕様書 specs/E4_covariate_shift.md

ここで検査するのは「実装がこう動く」ではなく「定義と理論がこう言う」性質だけ。
1. 重みがすべて 1 なら通常の共形分位点（卒論 eq:khat）に一致する
2. 重みは比だけが効くので、全体を定数倍しても変わらない
3. 定義を手で計算した数例と一致する
4. 共変量シフトの下で、正しい重みなら最悪のモデルでも被覆率が 1-alpha 以上になる
   （Tibshirani ほか 2019）。重みなしでは 1-alpha を下回る
"""

import numpy as np
import pytest

from conformal import conformal_quantile, weighted_conformal_quantile
from conformal.datasets import heteroscedastic, heteroscedastic_shifted, shift_weight


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20, 1 / 3, 0.50])
@pytest.mark.parametrize("n_cal", [1, 2, 8, 9, 19, 20, 100, 999, 1000])
def test_重みがすべて1なら通常の共形分位点と一致する(n_cal, alpha):
    """重みがすべて 1 のとき p_i = 1/(n+1) で、定義は S_(k), k = ceil((n+1)(1-alpha)) に戻る。

    格子には、float の丸めで順位が 1 つずれうる例（alpha=1/3, n=8）と、
    k = n+1 で +inf になる例（alpha=0.10, n=8 など）を含む。許容幅なしの完全一致で比べる。
    """
    rng = np.random.default_rng(1000 * n_cal + int(1000 * alpha))
    scores = rng.exponential(size=n_cal)
    expected = conformal_quantile(scores, alpha)
    q_hat = weighted_conformal_quantile(scores, np.ones(n_cal), np.ones(3), alpha)
    assert q_hat.shape == (3,)
    assert np.all(q_hat == expected), f"n={n_cal}, alpha={alpha}: {q_hat} != {expected}"


def test_重みが1の格子に無限大になる場合が含まれている():
    """上のテストが +inf の場合を本当に通っていることの確認（n=8, alpha=0.10 は k = 9 > n）。"""
    q_hat = weighted_conformal_quantile(np.arange(8.0), np.ones(8), np.ones(1), 0.10)
    assert np.isinf(q_hat[0]) and np.isinf(conformal_quantile(np.arange(8.0), 0.10))


@pytest.mark.parametrize("factor", [1e-3, 0.5, 7.0, 1e6])
def test_重みの定数倍に不変(factor):
    """p_i = w_i / (sum_j w_j + w(x)) は、較正とテストの重みを同じ正の定数倍しても変わらない。"""
    rng = np.random.default_rng(2)
    n_cal, n_test, alpha = 200, 500, 0.10
    scores = rng.exponential(size=n_cal)
    cal_weights = rng.uniform(0.0, 5.0, size=n_cal)
    test_weights = rng.uniform(0.0, 5.0, size=n_test)
    base = weighted_conformal_quantile(scores, cal_weights, test_weights, alpha)
    scaled = weighted_conformal_quantile(scores, factor * cal_weights, factor * test_weights, alpha)
    assert np.array_equal(base, scaled)


def test_非常に大きい重みでも定数倍に不変():
    """個々の重みが有限なら、合計が float の上限を超えるほど大きくても結果は変わらない。

    スコア [1, 2]、重みはすべて等しい、alpha = 0.1。較正点の重みの割合は 2/3 < 0.9 なので、
    定義から重みの大きさによらず +inf。alpha = 0.5 なら必要な割合 1/2 にスコア 2 で届く。
    """
    for factor in (1.0, 1e308):
        weights = factor * np.ones(2)
        assert np.isinf(weighted_conformal_quantile([1.0, 2.0], weights, weights[:1], 0.1)[0])
        assert weighted_conformal_quantile([1.0, 2.0], weights, weights[:1], 0.5)[0] == 2.0


@pytest.mark.parametrize(
    "alpha, expected",
    [
        (0.4, 3.0),     # 合計 5、必要な累積 3.0。累積は 1, 2, 4 → スコア 3 で初めて届く
        (0.1, np.inf),  # 必要な累積 4.5。較正点を全部足しても 4 → 届かないので +inf
        (0.6, 2.0),     # 必要な累積 2.0。スコア 2 でちょうど届く
    ],
)
def test_手計算の例と一致する(alpha, expected):
    """スコア [1, 2, 3]、較正の重み [1, 1, 2]、テストの重み 1 を定義どおり手で計算した値。

    スコアの順序を入れ替えて渡しても同じ値になること（並べ替えを実装に任せてよいこと）も見る。
    """
    q_hat = weighted_conformal_quantile([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], [1.0], alpha)
    assert q_hat[0] == expected
    shuffled = weighted_conformal_quantile([3.0, 1.0, 2.0], [2.0, 1.0, 1.0], [1.0], alpha)
    assert shuffled[0] == expected


def test_テスト点の重みが大きいほど分位点は小さくならない():
    """w(x) が大きいほど +inf に置く重みが増えるので、\\hat{q}(x) は w(x) について単調非減少。"""
    rng = np.random.default_rng(3)
    scores = rng.exponential(size=100)
    cal_weights = rng.uniform(0.5, 2.0, size=100)
    test_weights = np.linspace(0.0, 30.0, 301)
    q_hat = weighted_conformal_quantile(scores, cal_weights, test_weights, 0.10)
    assert np.all(q_hat[1:] >= q_hat[:-1])  # diff だと inf - inf が nan になるので直接比べる
    assert np.isinf(q_hat[-1])  # p_inf = 30 / (sum + 30) > alpha なので有限では届かない


def test_シフト下の被覆_定数予測器():
    """共変量シフトの下で、正しい重みなら最悪のモデルでも被覆率は 1-alpha 以上（Tibshirani ほか 2019）。

    予測が常に 0 の定数予測器を使う（スコアは |y|）。モデルの良さに頼らず保証だけを見るため。
    較正用データはシフトなし、テスト点は gamma = 2 でシフトさせる。

    同じ設定で重みなしの被覆率が 1-alpha をはっきり下回ることも確かめる。
    これが無いと「シフトが弱くて何をしても通る」テストと区別がつかない（検出力の確認）。

    SE について（CLAUDE.md「報告の作法」）: 試行ごとに較正用データもテスト点も引き直すので、
    試行は互いに独立で、全試行に共通の固定テスト集合は無い。したがって平均被覆率の SE は
    試行間 SD / sqrt(R) だけでよい（テスト点の二項変動は試行間 SD に含まれている）。
    """
    gamma, n_cal, n_test, alpha, n_trial = 2.0, 200, 100, 0.10, 2000
    rng = np.random.default_rng(20260901)
    cov_weighted = np.empty(n_trial)
    cov_unweighted = np.empty(n_trial)
    for trial in range(n_trial):
        X_cal, y_cal = heteroscedastic(n_cal, rng)                   # シフトなし
        X_test, y_test = heteroscedastic_shifted(n_test, rng, gamma)  # シフトあり
        cal_scores, test_scores = np.abs(y_cal), np.abs(y_test)      # 予測は常に 0

        q_weighted = weighted_conformal_quantile(
            cal_scores, shift_weight(X_cal, gamma), shift_weight(X_test, gamma), alpha
        )
        cov_weighted[trial] = np.mean(test_scores <= q_weighted)
        cov_unweighted[trial] = np.mean(test_scores <= conformal_quantile(cal_scores, alpha))

    se_weighted = cov_weighted.std(ddof=1) / np.sqrt(n_trial)
    se_unweighted = cov_unweighted.std(ddof=1) / np.sqrt(n_trial)
    assert cov_weighted.mean() >= 1 - alpha - 3 * se_weighted, (
        f"重み付きの被覆率が 1-alpha を下回る: {cov_weighted.mean():.4f} ± {se_weighted:.4f}"
    )
    assert cov_unweighted.mean() + 3 * se_unweighted <= 1 - alpha, (
        f"重みなしでも被覆が落ちていない（テストに検出力がない）: "
        f"{cov_unweighted.mean():.4f} ± {se_unweighted:.4f}"
    )


def test_gamma0のシフトはシフトなしと同じデータを返す():
    """gamma = 0 は密度が 1 で、Uniform(0,1) そのもの。重みもすべて 1 になる。"""
    X_plain, y_plain = heteroscedastic(50, np.random.default_rng(4))
    X_shift, y_shift = heteroscedastic_shifted(50, np.random.default_rng(4), gamma=0.0)
    assert np.array_equal(X_plain, X_shift) and np.array_equal(y_plain, y_shift)
    assert np.array_equal(shift_weight(X_shift, 0.0), np.ones(50))


@pytest.mark.parametrize("gamma", [1.0, 2.0, 4.0])
def test_シフトしたX1の分布と重みの平均が理論と合う(gamma):
    """X1 の密度が (1+gamma) x^gamma なら E[X1] = (1+gamma)/(2+gamma)。

    尤度比はシフトなしの分布のもとで平均 1 になる（E[w(X)] = 密度の積分 = 1）。
    どちらも平均 ± 5 SE で確かめる（標本 SD から SE を作る）。
    """
    n = 200_000
    rng = np.random.default_rng(5)
    X_shift, _ = heteroscedastic_shifted(n, rng, gamma)
    x1 = X_shift[:, 0]
    assert abs(x1.mean() - (1 + gamma) / (2 + gamma)) < 5 * x1.std(ddof=1) / np.sqrt(n)

    X_plain, _ = heteroscedastic(n, rng)
    weights = shift_weight(X_plain, gamma)
    assert weights.shape == (n,)
    assert abs(weights.mean() - 1.0) < 5 * weights.std(ddof=1) / np.sqrt(n)


@pytest.mark.parametrize(
    "scores, cal_weights, test_weights, alpha",
    [
        ([1.0, 2.0], [1.0, 1.0], [1.0], 0.0),            # alpha が (0,1) の外
        ([1.0, 2.0], [1.0, 1.0], [1.0], 1.0),
        ([1.0, 2.0], [1.0], [1.0], 0.1),                 # 長さが違う
        ([[1.0, 2.0]], [[1.0, 1.0]], [1.0], 0.1),        # 1次元でない
        ([1.0, 2.0], [1.0, -1.0], [1.0], 0.1),           # 負の重み
        ([1.0, 2.0], [1.0, 1.0], [np.inf], 0.1),         # 有限でない重み
        ([1.0, 2.0], [1.0, np.nan], [1.0], 0.1),
        ([1.0, 2.0], [0.0, 0.0], [0.0], 0.1),            # 重みの合計が 0
    ],
)
def test_不正な引数はValueError(scores, cal_weights, test_weights, alpha):
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, cal_weights, test_weights, alpha)
