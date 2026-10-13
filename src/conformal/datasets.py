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


def heteroscedastic_shifted(
    n: int, rng: np.random.Generator, gamma: float, base: float = 0.1, slope: float = 1.0
):
    """共変量シフトをかけた異分散のDGP（E4 のテスト点で使用。卒論 sec:e4）。

    heteroscedastic と違うのは X1 の分布だけである。X1 を密度

        p_gamma(x) = (1 + gamma) * x^gamma        (0 < x < 1)

    から引く。分布関数は F(x) = x^(1+gamma) なので、逆関数法で
    X1 = U^(1/(1+gamma)), U ~ Uniform(0,1) とすればよい。
    X2..X5 は Uniform(0,1) のまま、Y | X の式も heteroscedastic と同じ
    （入力の分布だけがずれ、入力から出力への関係は変わらない＝共変量シフト）。

    gamma = 0 はシフトなしで、同じ rng の状態から呼べば heteroscedastic と同じ値を返す。
    gamma が大きいほど X1 が 1 の近く（誤差の大きい側）に寄る。
    """
    if gamma < 0:
        raise ValueError(f"gamma は 0 以上でなければならない: {gamma}")
    X = rng.uniform(0.0, 1.0, size=(n, 5))
    X[:, 0] = X[:, 0] ** (1.0 / (1.0 + gamma))  # 逆関数法で X1 だけをずらす
    scale = base + slope * X[:, 0]
    y = X[:, 0] + np.sin(2 * np.pi * X[:, 1]) + scale * rng.normal(0.0, 1.0, size=n)
    return X, y


def shift_weight(X: np.ndarray, gamma: float) -> np.ndarray:
    """heteroscedastic_shifted の、シフトなしの分布に対する尤度比（正しい重み）を返す。

        w(x) = p_gamma(x1) / 1 = (1 + gamma) * x1^gamma

    シフトなしの X1 の密度は Uniform(0,1) の 1 であり、X2..X5 は分布が同じなので
    比に現れない。重み付き等角予測（卒論 sec:covariate-shift）の重みに使う。
    gamma = 0 のときはすべて 1 になる。

    Returns
    -------
    形状 (n,) の重み
    """
    if gamma < 0:
        raise ValueError(f"gamma は 0 以上でなければならない: {gamma}")
    X = np.asarray(X, dtype=float)
    return (1.0 + gamma) * X[:, 0] ** gamma
