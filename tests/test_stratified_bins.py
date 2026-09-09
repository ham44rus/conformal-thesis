"""層別被覆のビン分けの検証（実験E3 の部品）。

対応箇所: 卒論 第5章 5.7節・実装 src/conformal/metrics.py

理論から導かれる性質を検査する：

- 固定境界を渡したら、境界の定義どおりの区間 [e_b, e_{b+1}) に点が入る
- 層別被覆の点数の合計は、境界の外に出た点を除いた全点数に一致する
- 区間幅が定数なら SSC は層に分かれない（絶対残差スコアの構造的な性質）
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal.metrics import (
    bin_edges_from_quantiles,
    feature_stratified_coverage,
    size_stratified_coverage,
)


def _dummy_interval(n: int, half_width: float = 1.0):
    """被覆の中身は問わないので、原点まわりの固定幅区間を返す。"""
    return np.full(n, -half_width), np.full(n, half_width)


def test_固定境界を渡すと定義どおりのビンに分かれる():
    """[e_b, e_{b+1}) の半開区間に入ることを、境界上の点も含めて確認する。"""
    # 0.05, 0.10, ..., 0.95 の19点。境界 0.2/0.4/0.6/0.8 で5層に分かれる
    x = np.arange(0.05, 1.0, 0.05)
    y = np.zeros_like(x)
    lo, hi = _dummy_interval(x.size)
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, np.inf])

    out = feature_stratified_coverage(x, y, lo, hi, edges=edges)

    assert len(out) == 5, "5つの層に分かれるはず"
    assert [r["bin"] for r in out] == [0, 1, 2, 3, 4]
    # 第0層は 0.05/0.10/0.15 の3点。0.20 は半開区間の定義により第1層に入る
    assert [r["n"] for r in out] == [3, 4, 4, 4, 4]
    assert sum(r["n"] for r in out) == x.size, "全点がどれかの層に入る"


def test_境界上の点は上の層に入る():
    """半開区間 [e_b, e_{b+1}) の定義そのものを検査する。"""
    x = np.array([0.2, 0.4, 0.6])          # すべて境界のちょうど上
    y = np.zeros_like(x)
    lo, hi = _dummy_interval(x.size)
    edges = np.array([0.0, 0.2, 0.4, 0.6, np.inf])

    out = feature_stratified_coverage(x, y, lo, hi, edges=edges)
    got = {r["bin"]: r["n"] for r in out}

    # 0.2 は bin1、0.4 は bin2、0.6 は bin3。bin0 は空なので結果から落ちる
    assert got == {1: 1, 2: 1, 3: 1}


def test_固定境界なら層の位置がデータによらない():
    """同じ境界を渡せば、標本が変わっても層の切れ目は動かない。

    試行ごとに x の分位点から切ると境界自体が揺れて層別被覆のばらつきに
    混ざる。固定境界を渡せる意味はここにある。
    """
    rng = np.random.default_rng(0)
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, np.inf])
    counts = []
    for _ in range(5):
        x = rng.uniform(0.0, 1.0, size=20_000)
        y = np.zeros_like(x)
        lo, hi = _dummy_interval(x.size)
        out = feature_stratified_coverage(x, y, lo, hi, edges=edges)
        counts.append([r["n"] for r in out])

    counts = np.array(counts)
    # Uniform(0,1) の5等分なので各層は 4000 点前後。二項の3SD ≒ 54 点
    assert np.all(np.abs(counts - 4000) < 200), f"層の点数が偏りすぎ: {counts}"


def test_境界省略時は従来どおり等頻度で切る():
    """既存の呼び出しを壊していないことの確認（edges は省略可能）。"""
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, size=1000)
    y = np.zeros_like(x)
    lo, hi = _dummy_interval(x.size)

    out = feature_stratified_coverage(x, y, lo, hi, n_bins=5)

    assert len(out) == 5
    assert [r["n"] for r in out] == [200, 200, 200, 200, 200], "等頻度なので均等"


def test_昇順でない境界は拒否される():
    x = np.array([0.1, 0.5])
    y = np.zeros_like(x)
    lo, hi = _dummy_interval(x.size)
    with pytest.raises(ValueError):
        feature_stratified_coverage(x, y, lo, hi, edges=np.array([0.0, 0.5, 0.2]))


def test_bin_edges_from_quantiles_の最上端は無限大():
    """最大値をちょうど取る点が最後の層から漏れないこと。"""
    x = np.linspace(0.0, 1.0, 101)
    edges = bin_edges_from_quantiles(x, n_bins=4)

    assert edges.size == 5
    assert np.isinf(edges[-1])
    assert np.all(np.diff(edges[:-1]) > 0)


def test_定数幅の区間ではSSCが層に分かれない():
    """絶対残差スコアの構造的な性質。

    区間幅が全点で同じだと幅の分位点がすべて同値になり、層が1つに潰れて
    SSC は周辺被覆と同じ値しか返さない。E3 で abs の SSC を報告する際は
    この縮退を明示すること。
    """
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.normal(size=n)
    lo, hi = _dummy_interval(n, half_width=1.5)   # 幅が定数

    out = size_stratified_coverage(y, lo, hi, n_bins=5)

    assert len(out) == 1, "定数幅なら層は1つに潰れる"
    assert out[0]["n"] == n
    assert out[0]["coverage"] == pytest.approx(np.mean(np.abs(y) <= 1.5))


def test_可変幅の区間ではSSCが層に分かれる():
    """適応的な区間なら SSC が意味を持つことの対照。"""
    rng = np.random.default_rng(3)
    n = 2000
    y = rng.normal(size=n)
    w = rng.uniform(0.5, 3.0, size=n)

    out = size_stratified_coverage(y, -w, w, n_bins=5)

    assert len(out) == 5
    assert sum(r["n"] for r in out) == n
