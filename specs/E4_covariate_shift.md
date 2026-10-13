# E4: 交換可能性が破れる場合（共変量シフトと重み付き等角予測）

対応：卒論 第5章 E4 の節（`sec:e4`）、第4章 共変量シフトの節（`sec:covariate-shift`）
／ 実装 `experiments/e4_covariate_shift.py`、`src/conformal/weighted.py`

## 目的

較正用データとテスト点で入力の分布がずれる（共変量シフト）と、分割等角予測の被覆率が
`1-alpha` を下回ることを示す。そのうえで、正しい重み（式で分かっている尤度比）を使った
重み付き等角予測（Tibshirani ほか 2019、`refs.bib` の `tibshirani2019covariate`）で
被覆が `1-alpha` に戻ることを確かめる。

---

## 著者の決定（2026-10-13）

1. 範囲は**共変量シフトだけ**とする。時系列の変化（当初計画の (b)）は行わない。
2. 重みは**正しい重み（式で分かっている尤度比）**を使う。重みの推定は行わない。
3. データは E3 と同じ異分散データ（`src/conformal/datasets.py: heteroscedastic`）を使い回す。

正規化残差・CQR との組み合わせも、この実験では行わない。

---

## データ生成過程

較正用データ・学習用データ（シフトなし）：`src/conformal/datasets.py: heteroscedastic`（E3 と同じ）

    X ~ Uniform(0,1)^5
    Y = X1 + sin(2*pi*X2) + sigma(X1) * N(0,1)
    sigma(x) = 0.1 + 1.0 * x

テスト点（シフトあり）：`src/conformal/datasets.py: heteroscedastic_shifted`

    X1 の密度  p_gamma(x) = (1 + gamma) * x^gamma      (0 < x < 1)
    生成       X1 = U^(1/(1+gamma)),  U ~ Uniform(0,1)   （逆関数法）
    X2..X5 は Uniform(0,1) のまま。Y | X は上と同じ式

`gamma = 0` はシフトなし。`gamma` が大きいほど X1 が 1 の近く（誤差の大きい側）に寄る。

尤度比（正しい重み）：`src/conformal/datasets.py: shift_weight`

    w(x) = (1 + gamma) * x1^gamma

シフトなしの X1 の密度は 1 で、X2..X5 は分布が同じなので、比は X1 だけで決まる。

---

## 重み付き共形分位点

`src/conformal/weighted.py: weighted_conformal_quantile`

較正スコア `S_1..S_n`、較正点の重み `w_i = w(X_i)`、テスト点の重み `w(x)` に対し、

    p_i(x)   = w_i  / (sum_j w_j + w(x))        (i = 1..n)
    p_inf(x) = w(x) / (sum_j w_j + w(x))        （+inf に置く重み）
    q_hat(x) = min{ s : sum_{i: S_i <= s} p_i(x) >= 1 - alpha }   （そのような s が無ければ +inf）

区間は `[f_hat(x) - q_hat(x), f_hat(x) + q_hat(x)]`。重みがすべて 1 のときは通常の共形分位点
（`eq:khat`）に一致する。

実数の重みでは `eq:khat` のような有理数の厳密な計算ができないので、累積の重みと閾値の比較に
相対許容幅 `1e-12` を入れている（`alpha=1/3, n=8` のような float の丸めで順位が 1 つずれるのを
避けるため）。被覆への影響はこの相対幅以下である。

---

## 規模のパラメータ

| 項目 | 値 | 備考 |
|---|---|---|
| 学習用データ | 2,000 点 | シフトなし。モデルの学習に 1 回だけ使う |
| 較正用データ `n_cal` | 1,000 点 | シフトなし。試行ごとに引き直す唯一のもの |
| `alpha` | 0.10 | |
| 試行数 `R` | 500 | |
| テスト集合 `N_test` | 200,000 点 | `gamma` ごとに 1 組。**全試行で共有** |
| `gamma` の水準 | 0, 1, 2, 4 | 0 はシフトなし |
| seed | 20260901 | `make e4` |

---

## 手続き

1. 予測モデル `GradientBoostingRegressor`（scikit-learn の既定値）を学習用データ 2,000 点で
   1 回だけ学習し、全試行で固定する。スコアは絶対残差（`eq:score`）。
2. `gamma` ごとにテスト集合 200,000 点を 1 組作り、テスト点のスコアと重みを最初に 1 回計算する。
3. 試行ごとに較正用データ 1,000 点（シフトなし）を引き直す。**較正用データは学習に使わない。**
   同じ較正用データを、全部の `gamma` と 2 つの手法で共有する。
4. `gamma` ごとに 2 つの手法を評価する。
   - `unweighted`（重みなし）：`conformal_quantile` による通常の分割等角予測
   - `weighted`（重み付き）：`weighted_conformal_quantile`。重みは `shift_weight`（正しい重み）
5. 試行ごとに被覆率・平均の幅・無限区間の割合を記録し、`gamma` × 手法で集計する。

### 乱数の導出

`np.random.default_rng(seed)` の 1 本の乱数列から、次の順に引く（順序は変えないこと）。

1. 予測モデルの `random_state`（整数を 1 つ）
2. 学習用データ 2,000 点
3. テスト集合 200,000 点を `gamma = 0, 1, 2, 4` の順に 1 組ずつ
4. 試行 0, 1, …, 499 の順に、較正用データ 1,000 点を 1 組ずつ

---

## 出力

- `results/e4_trials.csv` : 列 = `gamma, method, trial, coverage, mean_width, frac_infinite`
  （4 水準 × 2 手法 × 500 試行 = 4,000 行。`mean_width` は有限な区間の幅 `2*q_hat` の
  テスト点平均、`frac_infinite` は `q_hat = +inf` になったテスト点の割合）
- `results/e4_summary.csv` : 列 = `gamma, method, n_cal, alpha, n_trial, n_test, coverage_mean,
  coverage_sd, se_trial, se_test, se_total, ci_lo, ci_hi, mean_width, frac_infinite, target`（8 行）

被覆率の標準誤差は、試行間とテスト集合間の 2 成分を合成する（CLAUDE.md「報告の作法」）。
テスト集合を全試行で共有しているので、その集合に固有のずれは試行数を増やしても消えない。

    se_trial = coverage_sd / sqrt(R)
    se_test  = sqrt(p(1-p) / N_test)          p = coverage_mean
    se_total = sqrt(se_trial^2 + se_test^2)
    ci_lo, ci_hi = coverage_mean ± 1.96 * se_total
    target = 1 - alpha

---

## 検証（この実験が正しく走ったと判断する基準）

**基準は実行前に決めた。結果を見てから変えない。** 満たさなかった場合は、基準を変えずに
事実を「実行結果」に書く。

- **V0**：`gamma = 0` では、2 つの手法の被覆率が試行ごとに完全に一致する（重みがすべて 1 になるため）。
- **V1**：`unweighted` は `gamma = 1, 2, 4` のそれぞれで `coverage_mean + 3*se_total < 1 - alpha`
  （保証が破れることの実証）。
- **V2**：`unweighted` の `coverage_mean` は `gamma` について単調に減少する。
- **V3**：`weighted` はすべての `gamma` で `coverage_mean >= 1 - alpha - 3*se_total`
  （正しい重みなら理論上 `1-alpha` 以上）。
- **V4**：`frac_infinite` はすべて 0（この規模では `w(x) <= 5` で、無限区間は起こらないはず）。

実装そのものの検証は `tests/test_weighted.py` で行う（重みが 1 なら通常の共形分位点と一致、
重みの定数倍に不変、手計算の例、定数予測器でのシフト下の被覆）。

---

## 図

- `figures/fig_e4_coverage.pdf`・`.png` … 左右 2 枚。
  左：横軸 `gamma`、縦軸 平均被覆率。2 手法を別の印と色で描き、誤差棒は `±1.96*se_total`
  （試行間＋テスト集合間）。`1-alpha` に破線。
  右：横軸 `gamma`、縦軸 平均の幅 `mean_width`。2 手法。
- `experiments/make_figures.py: fig_e4_coverage` が `results/e4_summary.csv` から作る。

---

## 実行結果

2026-10-13 実行（`make e4`、seed 20260901）。数値と基準の合否だけを書く。解釈・考察は著者が本文に書く。

`n_cal = 1000`、`alpha = 0.10`、`R = 500`、`N_test = 200,000`。SE は試行間＋テスト集合間の合成（`se_total`）。

| gamma | 手法 | 平均被覆率 | se_trial | se_test | se_total | 95% 区間（±1.96 se_total） | 平均の幅 | 無限区間の割合 |
|---|---|---|---|---|---|---|---|---|
| 0 | 重みなし | 0.90101 | 0.00039 | 0.00067 | 0.00077 | [0.89949, 0.90252] | 2.3071 | 0 |
| 0 | 重み付き | 0.90101 | 0.00039 | 0.00067 | 0.00077 | [0.89949, 0.90252] | 2.3071 | 0 |
| 1 | 重みなし | 0.84320 | 0.00057 | 0.00081 | 0.00099 | [0.84126, 0.84514] | 2.3071 | 0 |
| 1 | 重み付き | 0.90078 | 0.00048 | 0.00067 | 0.00082 | [0.89917, 0.90239] | 2.7720 | 0 |
| 2 | 重みなし | 0.80605 | 0.00065 | 0.00088 | 0.00110 | [0.80390, 0.80820] | 2.3071 | 0 |
| 2 | 重み付き | 0.89901 | 0.00058 | 0.00067 | 0.00089 | [0.89727, 0.90074] | 3.0057 | 0 |
| 4 | 重みなし | 0.76620 | 0.00072 | 0.00095 | 0.00119 | [0.76386, 0.76853] | 2.3071 | 0 |
| 4 | 重み付き | 0.90064 | 0.00075 | 0.00067 | 0.00100 | [0.89867, 0.90261] | 3.2694 | 0 |

検証基準の合否：

| 基準 | 合否 | 数値 |
|---|---|---|
| V0 | 満たす | `gamma = 0` で被覆率が一致しない試行は 500 中 0 |
| V1 | 満たす | `coverage_mean + 3*se_total` は `gamma = 1, 2, 4` で 0.84617、0.80934、0.76977（いずれも 0.90 未満） |
| V2 | 満たす | 重みなしの平均被覆率は `gamma = 0, 1, 2, 4` の順に 0.90101、0.84320、0.80605、0.76620 |
| V3 | 満たす | 重み付きの平均被覆率 0.90101、0.90078、0.89901、0.90064 に対し、下限 `1 - alpha - 3*se_total` は 0.89768、0.89753、0.89734、0.89699 |
| V4 | 満たす | `frac_infinite` の最大値は 0 |

補足（事実のみ）：`gamma = 2` の重み付きの平均被覆率 0.89901 は `1 - alpha = 0.90` を 0.00099 下回るが、
`se_total = 0.00089` の 3 倍（0.00266）以内である。

再現性：同じ seed で 2 回実行し、`results/e4_trials.csv`・`results/e4_summary.csv` の sha256 が一致した。
`make figures` を 2 回実行し、`figures/fig_e4_coverage.pdf`・`.png` の sha256 が一致した。
