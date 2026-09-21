# 第4・5章 ラベル仕様書（v1：式ラベルの確定）

> **この文書について（TeX 本文には含めない）**
> - 第4章（拡張）と第5章（数値実験）の本文はまだ書いていない。この文書は、コードの docstring と
>   テストが先に参照している **6 つの式ラベルを確定する**ためのもの。第4・5章の仕様書を書くときは
>   ここに書いた名前で `\label` を付ける。
> - 記号は TeX 本文に合わせる（$\hat f$、$\hat q$、$\mathcal{D}_{\mathrm{tr}}$、$\mathcal{D}_{\mathrm{cal}}$）。
> - 命名規則は既存のラベルと同じ：`eq:` + 小文字・ハイフン区切りの名詞
>   （既存例：`eq:score`, `eq:khat`, `eq:interval`, `eq:two-sided`, `eq:cond-coverage`, `eq:ceil-bounds`）。
> - コードの docstring は「ラベルは仮に …」と書いている。TeX に `\label` が付いた時点で「仮に」を外す
>   （コード側の作業。この文書では触れない）。

---

## 確定ラベル

| ラベル | 予定の節 | 内容 | 参照しているコード |
|---|---|---|---|
| `eq:score-normalized` | 4.1 正規化残差スコア | $S=\dfrac{\lvert y-\hat f(x)\rvert}{\hat\sigma(x)}$。区間は $\hat f(x)\pm\hat q\,\hat\sigma(x)$ | `src/conformal/scores.py` `normalized_residual` |
| `eq:sigma-hat` | 4.1 正規化残差スコア | $\hat\sigma(x)$ の推定：学習用データの out-of-fold 残差の大きさ $\lvert y-\hat f(x)\rvert$（または $(y-\hat f(x))^2$ の平方根）を $x$ で回帰し、下限 $\sigma_{\min}>0$ でクリップしたもの。較正用データは使わない | `src/conformal/adaptive.py` `fit_sigma`, `SigmaEstimator` |
| `eq:cqr-score` | 4.2 CQR | $S=\max\{\hat q_{\mathrm{lo}}(x)-y,\ y-\hat q_{\mathrm{hi}}(x)\}$。区間の内側で負、外側で正 | `src/conformal/scores.py` `cqr_score`、`tests/test_cqr_score.py` |
| `eq:cqr-interval` | 4.2 CQR | $C(x)=[\hat q_{\mathrm{lo}}(x)-\hat q,\ \hat q_{\mathrm{hi}}(x)+\hat q]$ | `src/conformal/scores.py` `cqr_interval`、`tests/test_cqr_score.py` |
| `eq:coverage-rate` | 5.1.1 被覆率と区間幅 | テスト点 $(x_j,y_j)$、$j=1,\dots,m$ での**経験的な被覆割合** $\widehat{\mathrm{Cov}}:=\dfrac1m\sum_{j=1}^{m}\mathbf{1}\{y_j\in C(x_j)\}$ | `src/conformal/metrics.py` `coverage`、`experiments/e2_model_agnostic.py` `coverage_from_sorted` |
| `eq:mean-width` | 5.1.1 被覆率と区間幅 | 平均区間幅 $\dfrac1m\sum_{j=1}^{m}\lvert C(x_j)\rvert$（$\lvert C(x)\rvert$ は区間の長さ） | `src/conformal/metrics.py` `mean_width` |

### `eq:coverage-rate` の定義についての注記

`eq:coverage-rate` は**テスト点での経験的な被覆割合**であり、次の 2 つの理論上の量とは区別する。

- 定理 3.3 の**被覆確率** $\mathbb{P}(Y_{n+1}\in C(X_{n+1}))$：学習用・較正用・テストのすべてについて平均した確率。
- 定義 3.16 の**被覆率** $\mathrm{Cov}=F_{\mathcal{D}_{\mathrm{tr}}}(\hat q)$：学習用・較正用データを固定したときに、新しい 1 点を覆う確率。

$\widehat{\mathrm{Cov}}$ は、区間（したがって $\mathcal{D}_{\mathrm{tr}}$ と $\hat q$）を固定したときの $\mathrm{Cov}$ の推定値であり、テスト点が $P$ からの i.i.d. なら $\widehat{\mathrm{Cov}}$ の条件付き期待値は $\mathrm{Cov}$、条件付き分散は $\mathrm{Cov}(1-\mathrm{Cov})/m$ である。第5章では、記号 $\widehat{\mathrm{Cov}}$ を使い、「被覆率」と書くときはどちらを指すか文脈で明示する。

---

## 変更履歴

| 版 | 内容 |
|---|---|
| v1 | 仮ラベル 6 つを確定。`eq:coverage-rate` に経験的な被覆割合であることの定義を明記 |
