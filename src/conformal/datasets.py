"""実験用のデータ生成過程（DGP）。

対応箇所:
    卒論 第5章 実験設定の節（sec:exp-setup）
仕様は specs/ の各実験仕様書に対応する。
"""

from __future__ import annotations

import numpy as np


def homoscedastic(n: int, rng: np.random.Generator, sigma: float = 0.5):
    """等分散のDGP（E1, E2, E5, E6 で使用）。

        X ~ Uniform(0,1)^5,  Y = X1 + sin(2*pi*X2) + N(0, sigma^2)
    """
    X = rng.uniform(0.0, 1.0, size=(n, 5))
    y = X[:, 0] + np.sin(2 * np.pi * X[:, 1]) + rng.normal(0.0, sigma, size=n)
    return X, y


def heteroscedastic(n: int, rng: np.random.Generator, base: float = 0.1, slope: float = 1.0):
    """異分散のDGP（E3 で使用。条件付き被覆の破綻を見せるための設定）。

        X ~ Uniform(0,1)^5,  Y = X1 + sin(2*pi*X2) + sigma(X1) * N(0,1)
        sigma(x) = base + slope * x        ← X1 が大きいほど誤差が大きい
    """
    X = rng.uniform(0.0, 1.0, size=(n, 5))
    scale = base + slope * X[:, 0]
    y = X[:, 0] + np.sin(2 * np.pi * X[:, 1]) + scale * rng.normal(0.0, 1.0, size=n)
    return X, y
