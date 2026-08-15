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
    "feature_stratified_coverage",
    "size_stratified_coverage",
]


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """周辺被覆率 (1/m) * sum 1[y_j in [lo_j, hi_j]]（卒論 式(5.1)）。"""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    """平均区間幅。被覆が同じなら狭いほど良い（卒論 式(5.2)）。"""
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def clopper_pearson(n_success: int, n_total: int, conf: float = 0.95) -> tuple[float, float]:
    """二項比率の Clopper–Pearson 信頼区間。

    被覆率を報告するときは必ずこれを併記する（CLAUDE.md「報告の作法」）。
    点推定だけでは、名目値との差がモンテカルロ誤差の範囲かどうか判断できない。
    """
    a = 1.0 - conf
    lo = 0.0 if n_success == 0 else stats.beta.ppf(a / 2, n_success, n_total - n_success + 1)
    hi = 1.0 if n_success == n_total else stats.beta.ppf(1 - a / 2, n_success + 1, n_total - n_success)
    return float(lo), float(hi)


def feature_stratified_coverage(
    x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_bins: int = 5
) -> list[dict]:
    """特徴量 x の分位点で層別した各層の被覆率（卒論 5.1節・実験E3）。

    周辺被覆が 1-alpha でも層ごとに大きく割れることがある。
    これが「条件付き被覆は保証されない」ことの実証になる。
    """
    x = np.asarray(x)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    out = []
    for b in range(n_bins):
        m = (x >= edges[b]) & (x < edges[b + 1])
        if m.sum() == 0:
            continue
        cov = coverage(np.asarray(y)[m], np.asarray(lo)[m], np.asarray(hi)[m])
        ci = clopper_pearson(int(round(cov * m.sum())), int(m.sum()))
        out.append({"bin": b, "n": int(m.sum()), "coverage": cov, "ci_lo": ci[0], "ci_hi": ci[1]})
    return out


def size_stratified_coverage(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_bins: int = 5
) -> list[dict]:
    """区間幅で層別した各層の被覆率（SSC, 卒論 5.1節）。

    最悪層の被覆率が、条件付き被覆の代理指標として最も分かりやすい。
    """
    width = np.asarray(hi) - np.asarray(lo)
    return feature_stratified_coverage(width, y, lo, hi, n_bins=n_bins)
