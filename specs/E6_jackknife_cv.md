# E6: jackknife+・CV+ とのデータ効率比較

対応：卒論 第5章 E6 の節（`sec:e6`）、第4章 jackknife+ と CV+ の節（次の run で書く）
／ 実装 `experiments/e6_jackknife_cv.py`、`src/conformal/jackknife.py`

## 目的

データが少ないとき（総数 50、100、200 点）に、データを分割せずに使う jackknife+ と CV+ を
分割等角予測と比べる。比べるのは被覆率、区間の幅、学習の回数。
出典：Barber, Candès, Ramdas, Tibshirani (2021)（`refs.bib` の `barber2021jackknife`）。

---

## 著者と PM の決定（2026-10-15）

著者の決定：

1. 比べる方法は **jackknife+ と CV+ の両方**（＋分割型）。
2. 予測モデルは**勾配ブースティング**（`GradientBoostingRegressor`、scikit-learn の既定値）。

PM の決定：

1. データは**等分散**（`src/conformal/datasets.py: homoscedastic`、E1・E2・E5 と同じ）。
2. `alpha = 0.10`、CV+ の組の数 `K = 10`、分割型の学習用：較正用は 50:50。
3. 試行ごとに、3 つの方法で**同じデータ**を使う（方法どうしの比較を対にする）。

---

## 理論

データ `(X_i, Y_i), i=1..n`。jackknife+ は点 `i` を除いた `n-1` 点で `mu_{-i}` を学習する。
CV+ はデータを `K` 組に分け、点 `i` の属する組 `S(i)` を除いて学習した `mu_{-S(i)}` を `mu_{-i}` とする。
抜いた点の残差 `R_i = |Y_i - mu_{-i}(X_i)|`（点 `i` は `mu_{-i}` の学習に入らない）。テスト点 `x` の区間は

    k = ceil((n+1)(1-alpha))                 （eq:khat）
    l = floor((n+1)alpha) = n+1-k            （prop:excess の証明の恒等式）
    hi(x) = { mu_{-i}(x) + R_i } の小さい方から k 番目   （k > n なら +inf）
    lo(x) = { mu_{-i}(x) - R_i } の小さい方から l 番目   （l = 0 なら -inf）

被覆確率の下限（split は `thm:coverage`、他は出典の定理による。交換可能性と、学習アルゴリズムが
点の順序によらないことを仮定。CV+ の下限は、各組が同じ大きさ（`n` が `K` で割り切れる）ことも仮定する。
E6 の総数 50、100、200 はどれも `K = 10` で割り切れる）：

| 方法 | 下限 `bound` | 学習の回数 |
|---|---|---|
| split（50:50） | `1 - alpha` | 1 |
| jackknife+ | `1 - 2*alpha` | `n` |
| CV+ | `1 - 2*alpha - sqrt(2/n)` | `K` |

学習用データによらないモデル（`mu_{-i} = mu`）では、区間は分割型の `[mu(x) - S_(k), mu(x) + S_(k)]` と一致する
（`tests/test_jackknife.py`）。

---

## データ生成過程

`src/conformal/datasets.py: homoscedastic`（E1・E2・E5 と同じ）

    X ~ Uniform(0,1)^5
    Y = X1 + sin(2*pi*X2) + N(0, 0.5^2)

---

## 規模のパラメータ

| 項目 | 値 | 備考 |
|---|---|---|
| データの総数 `N_TOTALS` | 50, 100, 200 | 試行ごとに引き直す |
| `alpha` | 0.10 | |
| 試行数 `R` | 200 | 総数ごと |
| テスト集合 `N_TEST` | 10,000 点 | **全試行・全総数で共有**する固定の集合 |
| CV+ の組の数 `K` | 10 | |
| 分割型の学習用：較正用 | 50:50 | `n_cal = n_total/2` |
| seed | 20260901 | `make e6` |

---

## 手続き

1. `np.random.default_rng(seed)` を作る。
2. テスト集合 10,000 点を `homoscedastic` で 1 回だけ生成する。
3. 総数ごと・試行ごとに、`homoscedastic` で `n_total` 点を生成し、3 つの方法で同じデータを使う。
   - split：並べ替えた前半 `n_total/2` 点を学習用、後半を較正用にする（重ならないことを `assert` で確かめる）。
     絶対残差（`eq:score`）から `conformal_quantile`（`eq:khat`）で `q_hat` を求める。
   - jackknife+：`jackknife_plus_interval`（学習 `n_total` 回）。
   - CV+：`cv_plus_interval`（`K = 10`、学習 10 回）。
   - モデルの `random_state` は試行ごとに 1 つ引き、3 つの方法のすべての学習で共有する
     （jackknife+・CV+ の保証は、すべての学習で同じアルゴリズムを使うことを仮定するため）。
4. テスト集合で、被覆率、平均の幅（有限な区間の `hi - lo` のテスト点平均）、無限区間の割合、学習の回数 `n_fits` を記録する。
   実行時間は csv に入れない（同じ seed で csv が一致しなくなるため）。総実行時間は画面に出し、下の「実行結果」に書く。

### 乱数の導出

`np.random.default_rng(seed)` の 1 本の乱数列から、次の順に引く（順序は変えないこと）。

1. テスト集合 10,000 点
2. 総数 50, 100, 200 の順に、試行 0, 1, …, 199 の順に、試行ごとに
   1. データ `n_total` 点
   2. split 用の 0..n_total-1 の並べ替え 1 つ
   3. 予測モデルの `random_state`（整数を 1 つ）
   4. CV+ の組分け用の 0..n_total-1 の並べ替え 1 つ

---

## 出力

- `results/e6_trials.csv` : 列 = `n_total, method, trial, coverage, mean_width, frac_infinite, n_fits`
  （3 総数 × 3 方法 × 200 試行 = 1,800 行）
- `results/e6_summary.csv` : 列 = `n_total, method, alpha, n_trial, n_test, coverage_mean, coverage_sd,
  se_trial, se_test, se_total, ci_lo, ci_hi, width_mean, width_sd, frac_infinite, n_fits, bound`（9 行）

    se_trial = coverage_sd / sqrt(R)
    se_test  = sqrt(p(1-p) / N_test)          p = coverage_mean
    se_total = sqrt(se_trial^2 + se_test^2)
    ci_lo, ci_hi = coverage_mean ± 1.96 * se_total

被覆率の標準誤差は、試行間とテスト集合間の 2 成分を合成する（CLAUDE.md「報告の作法」）。
`width_mean`・`width_sd` は試行ごとの `mean_width` の平均と SD。

---

## 検証（この実験が正しく走ったと判断する基準）

**基準は実行前に決めた。結果を見てから変えない。** 満たさなかった場合は、基準を変えずに
事実を「実行結果」に書く。

- **V1（理論の保証）**：すべての総数で、jackknife+ と CV+ の `coverage_mean >= bound - 3*se_total`。
- **V2（分割型）**：すべての総数で、split の `coverage_mean` が
  `[1-alpha - 3*se_total, k/(n_cal+1) + 3*se_total]` に入る（`n_cal = n_total/2`、`k = ceil((n_cal+1)(1-alpha))`）。
- **V3（学習の回数）**：`n_fits` が split 1、CV+ 10、jackknife+ `n_total` と一致する。
- 幅の大小（jackknife+・CV+ が split より狭いか）と、jackknife+・CV+ の被覆率が `1-alpha` にどれだけ近いかは、
  合否の基準にせず数値を報告する（この実験で観察する点）。

---

## 図

- `figures/fig_e6_small_sample.pdf`・`.png` … 左右 2 枚。横軸はデータの総数（50、100、200）。
  左：平均被覆率（誤差棒 `±1.96*se_total`）、3 方法。`1-alpha` と `1-2*alpha` に破線。
  右：平均の幅（誤差棒は試行間の ±1 SD）、3 方法。
- `experiments/make_figures.py: fig_e6_small_sample` が `results/e6_summary.csv` から作る。

---

## 実行結果

2026-10-16 実行（`make e6`、seed 20260901。総実行時間 1,824 秒（約 30 分。総数 50：224 秒、100：462 秒、200：1,137 秒の累計））。
数値と基準の合否だけを書く。解釈・考察は著者が本文に書く。

`alpha = 0.10`、`R = 200`、`N_test = 10,000`、CV+ の `K = 10`。SE は試行間＋テスト集合間の合成（`se_total`）。
無限区間の割合 `frac_infinite` は 9 行すべてで 0。

| 総数 | 方法 | 平均被覆率 | se_total | 95% 区間（±1.96 se_total） | 被覆率の SD | 下限 `bound` | 平均の幅 | 幅の SD | 学習の回数 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | split | 0.92825 | 0.00437 | [0.91969, 0.93681] | 0.04983 | 0.90000 | 2.7223 | 0.5589 | 1 |
| 50 | jackknife+ | 0.90682 | 0.00464 | [0.89772, 0.91592] | 0.05121 | 0.80000 | 2.2471 | 0.3657 | 50 |
| 50 | CV+ | 0.92506 | 0.00415 | [0.91694, 0.93319] | 0.04528 | 0.60000 | 2.3395 | 0.3513 | 10 |
| 100 | split | 0.90249 | 0.00404 | [0.89456, 0.91041] | 0.03884 | 0.90000 | 2.2117 | 0.3060 | 1 |
| 100 | jackknife+ | 0.90199 | 0.00374 | [0.89466, 0.90932] | 0.03211 | 0.80000 | 2.0214 | 0.2005 | 100 |
| 100 | CV+ | 0.91738 | 0.00344 | [0.91064, 0.92412] | 0.02911 | 0.65858 | 2.0889 | 0.2031 | 10 |
| 200 | split | 0.89273 | 0.00375 | [0.88538, 0.90008] | 0.02996 | 0.90000 | 1.9821 | 0.1861 | 1 |
| 200 | jackknife+ | 0.90077 | 0.00351 | [0.89390, 0.90765] | 0.02588 | 0.80000 | 1.9054 | 0.1499 | 200 |
| 200 | CV+ | 0.91437 | 0.00322 | [0.90807, 0.92067] | 0.02241 | 0.70000 | 1.9598 | 0.1409 | 10 |

検証基準の合否：

| 基準 | 合否 | 数値 |
|---|---|---|
| V1 | 満たす | `bound - 3*se_total` は jackknife+ が総数の順に 0.78607、0.78878、0.78948、CV+ が 0.58756、0.64827、0.69035。`coverage_mean` は上の表のとおりすべて上回る |
| V2 | 満たす | 総数 50（`n_cal=25, k=24`）：0.92825 ∈ [0.88690, 0.93618]。100（`n_cal=50, k=46`）：0.90249 ∈ [0.88787, 0.91409]。200（`n_cal=100, k=91`）：0.89273 ∈ [0.88875, 0.91224] |
| V3 | 満たす | `n_fits` は split 1、jackknife+ 50・100・200、CV+ 10 |

補足（事実のみ）：平均の幅は 3 つの総数すべてで jackknife+ < CV+ < split の順。
`abs(coverage_mean - 0.9)` は jackknife+ が 0.00682、0.00199、0.00077、CV+ が 0.02506、0.01738、0.01437。

再現性：同じ seed で 2 回実行し、`results/e6_trials.csv`（1,800 行）・`results/e6_summary.csv`（9 行）の sha256 が一致した。
`make figures` を 2 回実行し、`figures/fig_e6_small_sample.pdf`・`.png` の sha256 が一致し、既存の図に差分は出なかった。
