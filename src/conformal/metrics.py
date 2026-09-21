"""評価指標。被覆率・区間幅・層別被覆。

対応箇所:
    卒論 第5章「数値実験」5.1節（評価指標の定義）
    全実験で共通に使う
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "coverage",
    "mean_width",
    "clopper_pearson",
    "bin_edges_from_quantiles",
    "feature_stratified_coverage",
    "size_stratified_coverage",
]


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """周辺被覆率 (1/m) * sum 1[y_j in [lo_j, hi_j]]（卒論 5.1.1 項。ラベルは仮に eq:coverage-rate）。"""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    """平均区間幅。被覆が同じなら狭いほど良い（卒論 5.1.1 項。ラベルは仮に eq:mean-width）。"""
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def clopper_pearson(n_success: int, n_total: int, conf: float = 0.95) -> tuple[float, float]:
    """二項比率の Clopper–Pearson 信頼区間。

    **これは「1試行内」のばらつき専用である。** 有限個のテスト点を二項標本とみなして
    作る区間なので、母数は「その1回の適用での被覆率」になる。

    **試行間のばらつきには使えない。** 較正集合を引き直して多数回試行するシミュレーション
    （E1, E2 など）で平均被覆率に付ける区間は母数が違うため、試行間SD / sqrt(R) と
    固定テスト集合成分を合成して作ること（CLAUDE.md「報告の作法」の表を参照）。

    用途は、1回きりの適用の被覆率を報告するとき（実データ解析）と、
    層別被覆率のように1試行内で層ごとの被覆を見るとき。
    """
    a = 1.0 - conf
    lo = 0.0 if n_success == 0 else stats.beta.ppf(a / 2, n_success, n_total - n_success + 1)
    hi = 1.0 if n_success == n_total else stats.beta.ppf(1 - a / 2, n_success + 1, n_total - n_success)
    return float(lo), float(hi)


def bin_edges_from_quantiles(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """データ x の等頻度分位点から層の境界を作る（長さ n_bins+1）。

    境界を「どのデータから決めるか」は実験の設計判断なので、決める側と使う側を
    分けてある。呼び出し側が較正集合から決めることも、真の分布が既知なら
    固定境界を直接 feature_stratified_coverage に渡すこともできる。

    最上端は +inf にする。最大値をちょうど取る点を最後の層に含めるため。
    """
    # 層の境界を等頻度で切るだけの記述統計。共形分位点ではないので np.quantile を使う
    # （CLAUDE.md「絶対に守ること 1」が禁じているのは共形分位点の計算のこと）。
    edges = np.quantile(np.asarray(x), np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = np.inf
    return edges


def feature_stratified_coverage(
    x: np.ndarray,
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    n_bins: int = 5,
    edges: np.ndarray | None = None,
) -> list[dict]:
    """特徴量 x の分位点で層別した各層の被覆率（卒論 5.1節・実験E3）。

    周辺被覆が 1-alpha でも層ごとに大きく割れることがある。
    これが「条件付き被覆は保証されない」ことの実証になる。

    Parameters
    ----------
    edges : 層の境界（長さ n_bins+1、昇順）。省略すると x 自身の等頻度分位点から作る。
        **試行ごとに層の位置を動かしたくないときは外から固定境界を渡す。**
        x の分位点から毎回切ると、境界そのものが試行ごとに揺れて層別被覆の
        ばらつきに混ざる。真の分布が既知なら固定境界のほうが解釈が楽になる。
        渡した場合 n_bins は無視され、len(edges)-1 が層の数になる。

    Notes
    -----
    空の層は結果から落ちる。**区間幅が全点で一定のとき**（絶対残差スコアなど）
    size_stratified_coverage 経由で呼ぶと境界がすべて同値になり、層が1つに
    潰れて周辺被覆と同じ値になる。SSC を報告する側でこの縮退を検査すること。

    各層の信頼区間は Clopper–Pearson。これは1試行内でその層のテスト点を
    二項標本とみなす区間である（CLAUDE.md「報告の作法」の1行目）。
    試行間や固定テスト集合のばらつきは含まないので、多数回試行の結果を
    まとめるときは呼び出し側で別に合成すること。
    """
    x = np.asarray(x)
    if edges is None:
        edges = bin_edges_from_quantiles(x, n_bins)
    else:
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges は長さ2以上の1次元配列でなければならない")
        if np.any(np.diff(edges) < 0):
            raise ValueError("edges は昇順でなければならない")
    out = []
    for b in range(len(edges) - 1):
        m = (x >= edges[b]) & (x < edges[b + 1])
        if m.sum() == 0:
            continue
        cov = coverage(np.asarray(y)[m], np.asarray(lo)[m], np.asarray(hi)[m])
        ci = clopper_pearson(int(round(cov * m.sum())), int(m.sum()))
        out.append({"bin": b, "n": int(m.sum()), "coverage": cov, "ci_lo": ci[0], "ci_hi": ci[1]})
    return out


def size_stratified_coverage(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_bins: int = 5, rtol: float = 1e-12
) -> list[dict]:
    """区間幅で層別した各層の被覆率（SSC, 卒論 5.1節）。

    最悪層の被覆率が、条件付き被覆の代理指標として最も分かりやすい。

    **区間幅が全点で一定だと層が1つに潰れる。** 絶対残差スコアの区間は
    幅が定数なので、SSC は周辺被覆と同じ値しか返さない（層の数で判別できる）。
    適応的な区間（正規化残差・CQR）でなければ SSC は意味を持たない。

    定数幅の判定は**相対許容 rtol で行う**。lo = f(x) - q, hi = f(x) + q と
    組んだ区間の幅は数学的には 2q で一定だが、浮動小数では f(x) の大きさに応じて
    丸めが変わり、実際に 1e-15 程度の相対的な散らばりが出る。これをそのまま
    np.quantile に渡すと**丸め誤差で層が分かれてしまい**、意味のない「最悪層」が
    出てくる（実測で 0.55 という値が出た）。定数幅とみなせるときは層を1つに畳む。
    """
    y = np.asarray(y)
    lo, hi = np.asarray(lo), np.asarray(hi)
    width = hi - lo

    scale = float(np.abs(width).mean())
    if width.size == 0 or scale == 0.0 or float(np.ptp(width)) <= rtol * scale:
        # 定数幅。層に分けず、全点を1つの層として返す
        return feature_stratified_coverage(
            width, y, lo, hi, edges=np.array([-np.inf, np.inf])
        )
    return feature_stratified_coverage(width, y, lo, hi, n_bins=n_bins)
