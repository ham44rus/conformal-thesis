# 第4章 仕様書（v1：導入・4.1 正規化残差スコア・4.2 CQR）

> **この文書について（TeX 本文には含めない）**
> - 第4章「拡張」のうち、導入（一般のスコアに対する保証）、4.1 正規化残差スコア（`sec:normalized-score`）、4.2 CQR（`sec:cqr`）の仕様書。論理の正典はこの文書とする。4.3 jackknife+・CV+ と 4.4 共変量シフト（`sec:covariate-shift`）は範囲外。
> - 前提：第2章の仕様書（`chapter2_spec.md` v1.2 の 2.4 分位点回帰を含む）、主定理の仕様書（`main_theorem_spec.md` v3）、第3章 3.3〜3.6 の仕様書（`chapter3_sections_spec.md` v2.4）。
> - `chapter4_5_labels_spec.md` v1 で確定した第4章分の 4 ラベル（`eq:score-normalized`, `eq:sigma-hat`, `eq:cqr-score`, `eq:cqr-interval`）はこの文書に統合した。名前は変えない。
> - 記号は TeX 本文に合わせる（$\hat f$、$\hat q$、$\mathcal{D}_{\mathrm{tr}}$、$\mathcal{D}_{\mathrm{cal}}$、$\mathbf{1}\{\cdot\}$）。
> - 番号（命題 4.1 など）は骨組みの定理環境（section ごとの通し番号）で順に置いた場合の目安。TeX では提案ラベルで参照し、番号を直接書かない。
> - **使用するモデル（GBM 等）は第5章の話であり、本章の本文はモデルによらない形で書く。** 実装の選択は末尾の「実装との対応メモ」に置き、本文には含めない。
> - **TODO(著者)** の一覧は末尾にまとめた。

---

## 導入：一般のスコアに対する保証　提案ラベル `sec:general-score`

**本文の導入.** 第3章のスコア $S_i=|Y_i-\hat f(X_i)|$（`eq:score`）は、区間の幅が $x$ によらず一定になる（例 `ex:x-conditional-hetero`）。本章では、スコアを $x$ に応じて変えることで区間の幅を $x$ に適応させる 2 つの手法を扱う。その前に、定理 `thm:coverage` の証明がスコアの形 $|y-\hat f(x)|$ に依存していないことを確かめ、一般のスコアに対する保証として述べ直す。以後 4.1・4.2 の被覆保証はすべてこの命題から従う。

### 設定

- 学習用データ $d$ を受け取り、可測関数 $\mathcal{S}(d)\colon\mathcal{X}\times\mathbb{R}\to\mathbb{R}$ を返す写像 $\mathcal{S}$ を**スコア関数の学習アルゴリズム**とよぶ。$\mathcal{A}$ と同じく、$\mathcal{S}$ が乱数を用いる場合はその乱数も $\mathcal{D}_{\mathrm{tr}}$ に含めて考える。
- $s:=\mathcal{S}(\mathcal{D}_{\mathrm{tr}})$ とおき、**スコア**を
  $$S_i:=s(X_i,Y_i)\qquad(i=1,\dots,n+1)\tag{`eq:general-score`}$$
  と定める。第3章は $\mathcal{S}(d)(x,y)=|y-\mathcal{A}(d)(x)|$ の場合である。
- $k$ と $\hat q=S_{(k)}$ は `eq:khat` のとおり（$S_{(n+1)}=+\infty$ の規約を含む。スコアは負の値をとってもよい）。
- **予測集合**を
  $$C(x):=\{\,y\in\mathbb{R}:\ s(x,y)\le\hat q\,\}\tag{`eq:general-set`}$$
  と定める。$\hat q=+\infty$ のときは $C(x)=\mathbb{R}$ である（$s$ は実数値なので）。第3章の `eq:interval` は、$s(x,y)=|y-\hat f(x)|$ のときの `eq:general-set` そのものである。
- 仮定 (A1)(A2) は `sec:coverage-guarantee` と同じ。(A3) は $S_i$ を `eq:general-score` のスコアに読み替える：
  **(A3)** $(\mathcal{D}_{\mathrm{tr}},Z_1,\dots,Z_{n+1})$ の同時分布のもとで、確率 1 で $S_1,\dots,S_{n+1}$ は相異なる。

### 命題 4.1（一般のスコアに対する被覆保証）　提案ラベル `prop:general-score`

(A1)(A2) のもとで
$$\mathbb{P}\bigl(Y_{n+1}\in C(X_{n+1})\bigr)\ \ge\ 1-\alpha .$$
さらに (A3) のもとで
$$\mathbb{P}\bigl(Y_{n+1}\in C(X_{n+1})\bigr)=\frac{k}{n+1}\ \le\ 1-\alpha+\frac{1}{n+1}.$$

**証明.** まず、`eq:general-set` の定義から直ちに
$$Y_{n+1}\in C(X_{n+1})\iff s(X_{n+1},Y_{n+1})\le\hat q\iff S_{n+1}\le\hat q$$
である（$\hat q=+\infty$ のときは両辺とも常に成り立つ）。これは `eq:cover-iff` にあたるが、第3章と違って絶対値をほどく計算は要らない。

次に $(S_1,\dots,S_{n+1})$ が交換可能であることを示す。補題 `lem:score-exch` の証明で、写像 $G$ を
$$G(d,z_1,\dots,z_{n+1}):=\bigl(\mathcal{S}(d)(x_1,y_1),\ \dots,\ \mathcal{S}(d)(x_{n+1},y_{n+1})\bigr)$$
に置き換える。$G$ はランダム性を含まない固定された写像であり、各座標に同じ写像 $(x,y)\mapsto\mathcal{S}(d)(x,y)$ を適用しているので、置換 $\pi$ に対し $(S_{\pi(1)},\dots,S_{\pi(n+1)})=G(\mathcal{D}_{\mathrm{tr}},Z_\pi)$ である。以下は補題 `lem:score-exch` の証明と一字一句同じである（(A2) と命題 `prop:indep-pair` より $(\mathcal{D}_{\mathrm{tr}},Z_\pi)\overset{d}{=}(\mathcal{D}_{\mathrm{tr}},Z)$、命題 `prop:image` より $G(\mathcal{D}_{\mathrm{tr}},Z_\pi)\overset{d}{=}G(\mathcal{D}_{\mathrm{tr}},Z)$）。

補題 `lem:quantile` は交換可能性しか使わないので、そのまま適用できる。下界は (iii)（$\beta=1-\alpha$、$\lceil(n+1)\beta\rceil=k$）、(A3) のもとでの等式は (ii)（$m=k$）から従い、上界は `eq:ceil-bounds` より系 `cor:upper` と同じ計算で得る。$\blacksquare$

**証明の後に置く文.** 用いたのは、$s$ が $\mathcal{D}_{\mathrm{tr}}$ のみから定まること、および較正用データとテスト点に同じ $s$ を適用していることだけである。$\mathcal{S}$ の中身（どんな回帰を何回行うか）には何も仮定していない。

### 注意 4.2（仮定 (A2) の実装上の意味・後処理・空集合）　提案ラベル `rem:general-score`

1. **$s$ を作るのに較正用データを使ってはならない.** (A2) は $\mathcal{D}_{\mathrm{tr}}$ が $(Z_1,\dots,Z_{n+1})$ と独立であることを要求する。4.1 の $\hat\sigma$ や 4.2 の分位点回帰を較正用データも使って学習すると、$s$ が $Z_1,\dots,Z_n$ に依存し、$Z_{n+1}$ とは対等でなくなるので、命題 `prop:general-score` の証明は通らない。較正用データで学習すると被覆が大きく崩れることは `tests/test_split_leakage.py` で確認しており、本文では 1 文で触れるにとどめる（第5章の実験にはしない）。
2. **後処理も $s$ の一部である.** 4.1 の下限クリップ $\max\{\cdot,\sigma_{\min}\}$ や 4.2 の分位点の交差の入れ替え $\min/\max$ のような操作は、学習用データと $x$ だけで決まる固定の写像なので、$\mathcal{S}(d)$ の定義に含めれば命題の仮定を満たす。したがって後処理は保証を壊さない。
3. **$C(x)$ は空になりうる.** `eq:general-set` は集合として定義しており、区間であることも空でないことも要求していない。4.1 では常に $\hat f(x)\in C(x)$ だが、4.2 では $\hat q<0$ のとき空になりうる（注意 `rem:cqr-negative`）。空になった $x$ ではその点は覆われないが、命題は平均としての被覆確率を保証しているので矛盾はない。
4. **保証されるのは周辺被覆だけである.** 適応的なスコアにしても、条件付き被覆は保証されない（注意 `rem:x-conditional-ch4` の 1）。

---

## 4.1 正規化残差スコア（`sec:normalized-score`）

**本文の導入.** 絶対残差の区間 `eq:interval` は $x$ によらず幅 $2\hat q$ で一定である。$Y\mid X=x$ の散らばりが $x$ によって変わるなら、散らばりの大きい $x$ では広く、小さい $x$ では狭くしたい。最も単純な方法は、残差を散らばりの推定値 $\hat\sigma(x)$ で割ることである。

### 定義（正規化残差スコア）

学習用データから点予測器 $\hat f=\mathcal{A}(\mathcal{D}_{\mathrm{tr}})$ と、**尺度関数** $\hat\sigma\colon\mathcal{X}\to(0,\infty)$ を作る（作り方は `eq:sigma-hat`）。スコアを
$$S_i:=\frac{|Y_i-\hat f(X_i)|}{\hat\sigma(X_i)}\qquad(i=1,\dots,n+1)\tag{`eq:score-normalized`}$$
と定める。$\hat q=S_{(k)}$ を `eq:khat` のとおりとると、$\hat\sigma(x)>0$ より $s(x,y)\le\hat q\iff|y-\hat f(x)|\le\hat q\,\hat\sigma(x)$ なので、`eq:general-set` は区間
$$C(x)=\bigl[\,\hat f(x)-\hat q\,\hat\sigma(x),\ \hat f(x)+\hat q\,\hat\sigma(x)\,\bigr]\tag{`eq:interval-normalized`}$$
になる。$S_i\ge0$ より $\hat q\ge0$ であり、常に $\hat f(x)\in C(x)$ である。区間の幅 $2\hat q\,\hat\sigma(x)$ は $\hat\sigma(x)$ に比例して $x$ とともに変わる。

**被覆保証.** $(\hat f,\hat\sigma)$ は $\mathcal{D}_{\mathrm{tr}}$ のみから決まるので、$\mathcal{S}(d)(x,y):=|y-\mathcal{A}(d)(x)|/\hat\sigma_d(x)$（$\hat\sigma_d$ は $d$ から作った尺度関数）とおけば命題 `prop:general-score` の設定に含まれる。したがって (A1)(A2) のもとで被覆確率は $1-\alpha$ 以上であり、(A3) のもとで $k/(n+1)$ に等しい。**この保証は $\hat\sigma$ がどれだけ正確かによらない.** $\hat\sigma$ の質が効くのは区間の幅とその $x$ への適応の仕方だけである。

### $\hat\sigma$ の作り方　提案ラベル `eq:sigma-hat`

学習用データ $\mathcal{D}_{\mathrm{tr}}=((x_1',y_1'),\dots,(x_N',y_N'))$ から次のように作る。

1. **out-of-fold 残差.** $\{1,\dots,N\}$ を $K$ 個の fold に分け、各 $i$ について、$i$ の属する fold を除いた学習用データで $\mathcal{A}$ を学習した予測器を $\hat f^{(-i)}$ とし、$r_i:=|y_i'-\hat f^{(-i)}(x_i')|$ とおく。
2. **残差の大きさの回帰.** $(x_i',r_i)$（$i=1,\dots,N$）に回帰法を当てはめ、$\hat g\colon\mathcal{X}\to\mathbb{R}$ を得る（回帰法は任意。$r_i$ の代わりに $r_i^2$ を回帰して平方根をとってもよい）。
3. **下限クリップ.** 学習用データから決めた定数 $\sigma_{\min}>0$ を用いて
   $$\hat\sigma(x):=\max\{\hat g(x),\ \sigma_{\min}\}.\tag{`eq:sigma-hat`}$$

fold の分け方の乱数は $\mathcal{D}_{\mathrm{tr}}$ に含めて考える（`sec:split-algorithm` の $\mathcal{A}$ と同じ扱い）。$\hat f$ 自身は学習用データ全体で学習したものを使う。

**out-of-fold にする理由.** $\hat f$ を学習用データ全体で学習し、同じデータ上の残差 $|y_i'-\hat f(x_i')|$ を回帰の目的変数にすると、過学習のぶんだけ残差が真の誤差より小さく出る。縮み方が $x$ によらず一様なら定数倍として $\hat q$ に吸収されて害はない（注意 `rem:normalized-reduction`）が、一般には一様ではなく（柔軟なモデルほど学習点の残差はほぼ 0 になる）、$\hat\sigma$ の形が歪んで幅の適応が効かなくなる。$i$ を含まないデータで学習した $\hat f^{(-i)}$ の残差なら、この問題を避けられる。被覆保証は命題 `prop:general-score` によりどちらでも壊れない。

**下限クリップの理由.** 残差の大きさを回帰した $\hat g(x)$ は 0 や負になりうる。$\hat\sigma(x)\le0$ ではスコアが定義できず、$\hat\sigma(x)$ が極端に小さいと区間が 1 点に潰れる。$\sigma_{\min}$ は学習用データだけから決めるので、注意 `rem:general-score` の 2 により保証に影響しない。

### 注意 4.3（絶対残差への帰着）　提案ラベル `rem:normalized-reduction`

$\hat\sigma\equiv c$（定数 $c>0$）ならば、`eq:interval-normalized` は `eq:interval` に一致する。実際、$S_i=|Y_i-\hat f(X_i)|/c$ は絶対残差スコア $S_i^{\mathrm{abs}}$ の $1/c$ 倍であり、$t\mapsto t/c$ は非減少なので補題 `lem:beta-tools` (ii) より $\hat q=S^{\mathrm{abs}}_{(k)}/c$、よって $\hat q\,\hat\sigma(x)=S^{\mathrm{abs}}_{(k)}$ である。より一般に、$\hat\sigma$ を定数倍しても区間は変わらない。正規化残差が第3章と違う結果を与えるのは、$\hat\sigma$ が $x$ とともに変わるときだけである。

（この性質は `tests/test_adaptive.py` の「sigma が定数なら正規化残差は絶対残差と同じ区間を与える」で検査している。）

### 注意 4.4（条件付き被覆・限界）　提案ラベル `rem:normalized-limits`

1. **条件付き被覆.** $\hat\sigma$ が真の条件付き標準偏差に一致する場合の条件付き被覆については、注意 `rem:x-conditional-ch4` の 2 を参照する（本節では繰り返さない）。
2. **対称な区間しか作れない.** `eq:interval-normalized` は常に $\hat f(x)$ を中心とする対称な区間である。$Y\mid X=x$ の分布が非対称なら、片側が余り、片側が足りない区間になる。この限界を克服するのが 4.2 の CQR である。
3. **$\hat\sigma$ の質は幅にだけ効く.** $\hat\sigma$ が散らばりを捉えられなければ、区間は絶対残差と同程度に $x$ によらないものに戻るだけで、被覆は失われない（命題 `prop:general-score`）。

---

## 4.2 CQR（`sec:cqr`）

**本文の導入.** 正規化残差は $\hat f(x)$ を中心に対称に広げる。Romano, Patterson and Candès (2019) の Conformalized Quantile Regression（CQR）は、点予測器の代わりに 2.4 節の分位点回帰で区間の下端と上端を別々に推定し、それを共形分位点で補正する。区間の位置と幅の両方が $x$ に適応し、非対称にもなれる。

### 定義（CQR）

$\alpha_{\mathrm{lo}}:=\alpha/2$、$\alpha_{\mathrm{hi}}:=1-\alpha/2$ とし、学習用データで 2.4 節の分位点回帰（`eq:quantile-regression`）を水準 $\alpha_{\mathrm{lo}}$ と $\alpha_{\mathrm{hi}}$ について**別々に**行い、
$$\hat q_{\mathrm{lo}}\colon\mathcal{X}\to\mathbb{R},\qquad \hat q_{\mathrm{hi}}\colon\mathcal{X}\to\mathbb{R}\tag{`eq:cqr-pair`}$$
を得る。以後 $\hat q_{\mathrm{lo}}(x)\le\hat q_{\mathrm{hi}}(x)$ とする（交差する場合の扱いは注意 `rem:cqr-crossing`）。スコアを
$$S_i:=\max\bigl\{\hat q_{\mathrm{lo}}(X_i)-Y_i,\ \ Y_i-\hat q_{\mathrm{hi}}(X_i)\bigr\}\qquad(i=1,\dots,n+1)\tag{`eq:cqr-score`}$$
と定める。$S_i$ は、$Y_i$ が区間 $[\hat q_{\mathrm{lo}}(X_i),\hat q_{\mathrm{hi}}(X_i)]$ の外にあるときはそこからのはみ出しの長さ（正）、内側にあるときは近い方の端までの距離の $-1$ 倍（負）である。$\hat q=S_{(k)}$ を `eq:khat` のとおりとると、$\max\{a,b\}\le t\iff a\le t$ かつ $b\le t$ より
$$C(x)=\bigl[\,\hat q_{\mathrm{lo}}(x)-\hat q,\ \hat q_{\mathrm{hi}}(x)+\hat q\,\bigr]\tag{`eq:cqr-interval`}$$
となる。分位点回帰の区間を、両側に同じ量 $\hat q$ だけ広げた（$\hat q<0$ なら縮めた）ものである。

**被覆保証.** $(\hat q_{\mathrm{lo}},\hat q_{\mathrm{hi}})$ は $\mathcal{D}_{\mathrm{tr}}$ のみから決まるので、命題 `prop:general-score` により (A1)(A2) のもとで被覆確率は $1-\alpha$ 以上、(A3) のもとで $k/(n+1)$ である。分位点回帰の良し悪し（どのモデルを使うか、水準どおりに推定できているか）は保証に影響しない。

### 注意 4.5（スコアの符号と $\hat q$ の符号）　提案ラベル `rem:cqr-negative`

1. 絶対残差・正規化残差と違い、`eq:cqr-score` は負の値をとる。較正用データの多く（$k$ 個以上）が分位点回帰の区間の内側にあれば $\hat q<0$ となり、`eq:cqr-interval` は分位点回帰の区間より狭くなる。分位点回帰が広すぎる区間を出したときに、共形分位点がそれを縮める方向に補正するということである。
2. $\hat q<0$ のときは $\hat q_{\mathrm{hi}}(x)-\hat q_{\mathrm{lo}}(x)<-2\hat q$ となる $x$ で $C(x)=\emptyset$ になりうる（注意 `rem:general-score` の 3）。
3. スコアの符号を逆にする（$\max\{y-\hat q_{\mathrm{lo}},\hat q_{\mathrm{hi}}-y\}$ とする）と、区間の内側で正・外側で負になり、共形分位点が区間を広げる方向と逆になる。実装ではこの符号ミスが典型的な誤りである（CLAUDE.md「よくあるバグ」4。`tests/test_cqr_score.py` で手計算例と照合している）。

### 注意 4.6（分位点の交差）　提案ラベル `rem:cqr-crossing`

$\hat q_{\mathrm{lo}}$ と $\hat q_{\mathrm{hi}}$ は別々に学習するので、$x$ によっては $\hat q_{\mathrm{lo}}(x)>\hat q_{\mathrm{hi}}(x)$ と**交差**することがある。これは分位点回帰の既知の性質であって誤りではない。本論文では、交差した $x$ で 2 つの値を入れ替え、$\tilde q_{\mathrm{lo}}(x):=\min\{\hat q_{\mathrm{lo}}(x),\hat q_{\mathrm{hi}}(x)\}$、$\tilde q_{\mathrm{hi}}(x):=\max\{\cdot\}$ を `eq:cqr-score`・`eq:cqr-interval` に用いる。入れ替えは学習用データと $x$ だけで決まる固定の操作なので、注意 `rem:general-score` の 2 により保証に影響しない。交差した点の割合は分位点回帰の質の指標なので、第5章では必ず報告する。

### 注意 4.7（分位点回帰が正確な場合）　提案ラベル `rem:cqr-oracle`

$\hat q_{\mathrm{lo}}$、$\hat q_{\mathrm{hi}}$ が真の条件付き分位点 $q_{\alpha/2}(x)$、$q_{1-\alpha/2}(x)$（定義 `def:cond-quantile`）に一致し、$Y\mid X=x$ の分布関数が各 $x$ で連続ならば、学習用データを固定したときのテスト点のスコア $S=\max\{q_{\alpha/2}(X)-Y,\ Y-q_{1-\alpha/2}(X)\}$ は
$$\mathbb{P}(S\le0\mid\mathcal{D}_{\mathrm{tr}})=\mathbb{P}\bigl(q_{\alpha/2}(X)\le Y\le q_{1-\alpha/2}(X)\bigr)=1-\alpha$$
を満たす（$X=x$ で条件付ければ $F_{Y\mid X}(q_{1-\alpha/2}(x)\mid x)-F_{Y\mid X}(q_{\alpha/2}(x)\mid x)=(1-\alpha/2)-\alpha/2$ であり、$x$ について平均する）。すなわちスコアの分布の $(1-\alpha)$ 分位点は 0 であり、較正用データが多ければ $\hat q\approx0$ で、`eq:cqr-interval` は分位点回帰の区間にほぼ一致する。共形分位点による補正は、分位点回帰が不正確なぶんだけ働く。

（`tests/test_cqr_score.py` の「真の条件付き分位点を与えれば $\hat q\approx0$」で検査している。）

### 注意 4.8（条件付き被覆について）　提案ラベル `rem:cqr-asymptotic`

Romano et al. (2019) は、分位点回帰が真の条件付き分位点に一致する極限では CQR の区間が条件付き被覆をもつことを述べている（`romano2019cqr`。本論文では引用にとどめ、証明しない）。有限標本で保証されるのは命題 `prop:general-score` の周辺被覆だけであり、条件付き被覆は分位点回帰の正確さに依存する。これは注意 `rem:x-conditional-ch4` の 1・2 と同じ位置づけである。第5章 E3 では、不均一分散のデータで正規化残差と CQR の層別被覆が絶対残差よりどれだけ平らになるかを測る。

---

## 第3章の証明のうち、スコアの形 $|y-\hat f(x)|$ に依存している箇所（TeX 本文には含めない）

命題 `prop:general-score` を書くにあたり、第3章の証明でスコアの形が使われている箇所を `thesis/R08_NAME.tex` の行番号で挙げる（2026-09-21、コミット `24da614` 時点）。

| 箇所 | 行 | 依存の内容 | 命題 `prop:general-score` での扱い |
|---|---|---|---|
| (A3) の主張 | 457–459 | $S_i$ が `eq:score` のスコアであることを前提 | $S_i$ を `eq:general-score` に読み替えて述べ直す |
| 補題 `lem:score-exch` の証明：$G$ の定義 | 470–474 | $G$ を $\lvert y_i-\mathcal{A}(d)(x_i)\rvert$ で明示的に定義 | $G$ を $\mathcal{S}(d)(x_i,y_i)$ に置き換える |
| 同：座標ごとの写像 | 479 | 「各座標に同じ写像 $(x,y)\mapsto\lvert y-\mathcal{A}(d)(x)\rvert$ を適用する」 | 「同じ写像 $(x,y)\mapsto\mathcal{S}(d)(x,y)$」に置き換える。以降の行（480–494）は変更不要 |
| 補題 `lem:score-exch` の後の文 | 496–497 | 「用いたのは $\hat f$ が $\mathcal{D}_{\mathrm{tr}}$ のみから定まること、および全座標に同じスコア関数を適用していることだけ」 | 一般化の根拠としてそのまま引用できる |
| 定理 `thm:coverage` の証明：`eq:cover-iff` の導出 | 604–612 | $-\hat q\le Y-\hat f(X)\le\hat q\iff\lvert Y-\hat f(X)\rvert\le\hat q$ と絶対値をほどく | `eq:general-set` の定義から直ちに従うので計算不要 |
| 系 `cor:upper` の証明 | 640–653 | 直接の依存なし（`eq:cover-iff`・補題 `lem:score-exch`・補題 `lem:quantile` (ii) を経由するのみ） | 変更不要 |
| 補題 `lem:quantile` | 499–590 | 依存なし（交換可能性のみ） | 変更不要 |

第3章の他の節でスコアの形に依存している箇所（命題 `prop:general-score` の範囲外だが、第4章で言及する可能性があるもの）：

| 箇所 | 行 | 内容 |
|---|---|---|
| 定義 `def:random-tiebreak` の $C^U(x)$ | 851, 861–862 | $\lvert y-\hat f(x)\rvert$ で定義。一般のスコアでは $\{y:(s(x,y),U_{n+1})\preceq P_{(k)}\}$ になる |
| 3.5 節の (A3$'$)・`eq:score-cdf` | 963–966 | $F_d$ を $\lvert Y-\mathcal{A}(d)(X)\rvert$ の分布関数で定義。一般のスコアなら $\mathcal{S}(d)(X,Y)$ の分布関数 |
| 定理 `thm:cond-coverage-beta` の証明 | 1053 | $S_i^d:=\lvert Y_i-\mathcal{A}(d)(X_i)\rvert$ |
| 命題 `prop:x-conditional-group` の $C_G(x)$ | 1308 | 区間の形で定義 |

---

## 実装との対応メモ（TeX 本文には含めない）

- **正規化残差の分母の $10^{-8}$.** `src/conformal/scores.py` の `normalized_residual` は 0 除算を避けるため $\hat\sigma(x)+10^{-8}$ で割り、`experiments/e3_conditional.py` は区間も $\hat q\,(\hat\sigma(x)+10^{-8})$ で組んで規約を揃えている。本文は `eq:score-normalized` のとおり $\hat\sigma(x)$ で割る形で書き、**付録A の注記の候補**として「実装では 0 除算回避のため分母に $10^{-8}$ を加える。$\sigma_{\min}$ がこれより十分大きいので結果には影響しない」を書く。$\hat\sigma+10^{-8}$ も学習用データだけで決まる固定の写像なので、保証には影響しない。
- **`eq:sigma-hat` の実装.** `src/conformal/adaptive.py` の `fit_sigma`。$K=5$、目的変数は $r_i$（`target="abs"`。`"sq"` で $r_i^2$）、$\sigma_{\min}$ は out-of-fold 残差の 5% 分位点（記述統計。共形分位点ではない）。回帰法は引数で受け取る。E3 では GBM（`specs/E3_conditional_coverage.md`「モデルの設定」）。
- **`eq:cqr-pair` の実装.** `adaptive.py` の `fit_quantile_pair`。`model_factory(q, seed)` で水準ごとに別モデルを作る。E3 では GBM の分位点損失（水準 0.05・0.95）。交差の入れ替えは `QuantilePair.predict`、交差率は `crossing_rate_on`。
- **空区間.** `scores.py` の `cqr_interval` は $\hat q<0$ で `lo > hi` となる区間をそのまま返し、`metrics.coverage` は `(y >= lo) & (y <= hi)` で判定するので空区間は「覆わない」と数えられる。本文の注意 `rem:general-score` の 3 と整合する。
- **検査.** 命題 `prop:general-score` の「固定関数なら何でも」は `tests/test_coverage_theory.py` の正規化残差・CQR の Beta 検定（わざと外した手書きの $\hat\sigma$・分位点関数）で、(A2) の実装上の意味（注意 `rem:general-score` の 1）は `tests/test_split_leakage.py` で検査している。
- **docstring の修正候補（本仕様書の範囲外）.** `adaptive.py` の `out_of_fold_predict` の docstring は「sigma が過小評価され、正規化残差スコアが大きく出て区間が不必要に広がる」と書いているが、一様な過小評価は定数倍として $\hat q$ に吸収される（注意 `rem:normalized-reduction`）。正しくは「縮み方が $x$ によって一様でないため $\hat\sigma$ の形が歪み、幅の適応が効かなくなる」である。TeX 化のあとに直す。

---

## TODO(著者) 一覧

| # | 箇所 | 内容 |
|---|---|---|
| 1 | 2.4（`chapter2_spec.md`） | 命題 `prop:pinball-quantile`（条件付き分位点が pinball 損失の期待値を最小化する）の証明を本文に載せるか、主張と引用（`koenker1978regression`）だけにするか。証明の草案は `chapter2_spec.md` v1.2 に置いた |
| 2 | 2.4 | `koenker1978regression` が支持する主張の範囲。同論文は線形モデルの回帰分位点を pinball 損失の最小化として定義した論文であり、母集団の命題 `prop:pinball-quantile` の出典としてよいか（本仕様書は指示どおり同論文を引いている） |
| 3 | 2.4 | 条件付き分位点の定義 `def:cond-quantile` で、条件付き分布関数 $F_{Y\mid X}(\cdot\mid x)$ の存在をどの程度の厳密さで扱うか（「正則条件付き分布が存在すると仮定する」の 1 文で済ませる案） |
| 4 | 導入 | 命題 `prop:general-score` の上界（(A3) のもとでの $k/(n+1)$）を本文に入れるか、下界だけにして上界は「系 `cor:upper` と同様」で済ませるか。本仕様書は入れる案で書いた |
| 5 | 導入 | 導入を「4.0」相当の独立した subsection（`sec:general-score`）にするか、`\section{拡張}` 直後の節見出しなしの本文にするか。命題と注意を含むので subsection を推奨 |
| 6 | 導入 | 3.5 節の被覆率のベータ分布（定理 `thm:cond-coverage-beta`）も (A3$'$) を一般のスコアに読み替えれば成り立つ。本文で 1 文触れるか（`tests/test_coverage_theory.py` の新しい 2 本はこれを検査している）、触れないか |
| 7 | 4.1 | `eq:sigma-hat` で out-of-fold の手続きをどこまで式で書くか（fold の記法 $\hat f^{(-i)}$ を導入する案と、文章で済ませる案） |
| 8 | 4.1 | $\sigma_{\min}$ の決め方を本文でどこまで言うか。本仕様書は「学習用データから決めた定数」とだけ書き、5% 分位点は実装メモに置いた |
| 9 | 4.2 | 交差の入れ替え（注意 `rem:cqr-crossing`）を定義に組み込む（$\tilde q$ を定義に使う）か、本仕様書のように「以後 $\hat q_{\mathrm{lo}}\le\hat q_{\mathrm{hi}}$ とする」と断って注意で扱うか |
| 10 | 4.2 | 注意 `rem:cqr-oracle` の $\hat q\approx0$ を、母集団の $(1-\alpha)$ 分位点が 0 という主張までで止めるか、$n\to\infty$ の収束まで言うか（収束を言うなら根拠が要る。本仕様書は分位点が 0 までで止めている） |
| 11 | 4.2 | 注意 `rem:cqr-asymptotic` で Romano et al. (2019) の主張をどの文言で引くか。原論文の該当箇所（Theorem 1 の周辺か、Section の議論か）を著者が確認して文言を確定する。Claude は原論文を確認していない |
| 12 | 全体 | 新しい提案ラベル（`sec:general-score`, `eq:general-score`, `eq:general-set`, `prop:general-score`, `rem:general-score`, `eq:interval-normalized`, `rem:normalized-reduction`, `rem:normalized-limits`, `eq:cqr-pair`, `rem:cqr-negative`, `rem:cqr-crossing`, `rem:cqr-oracle`, `rem:cqr-asymptotic`、2.4 の `sec:quantile-regression`, `eq:pinball`, `def:cond-quantile`, `prop:pinball-quantile`, `eq:quantile-regression`）の採否。TeX・仕様書・コードのいずれとも衝突しないことは確認済み |

---

## 変更履歴

| 版 | 内容 |
|---|---|
| v1 | 初版。`chapter4_5_labels_spec.md` v1 の第4章分 4 ラベルを統合 |
