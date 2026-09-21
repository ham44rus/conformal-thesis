# 第2章 仕様書：記法・交換可能性・順序統計量と経験分位点・分位点回帰

> **この文書について（TeX 本文には含めない）**
> - 卒論第2章のうち、2.1 記法、2.2 交換可能性、2.3 順序統計量と経験分位点、2.4 回帰モデルと分位点回帰（v1.2 で追加。第4章 CQR の前提節）の仕様書。論理の正典はこの文書とする。2.5 は範囲外。
> - 第3章の仕様書（`main_theorem_spec.md`）は v3 で、ここの定義・命題を参照する形に改めた（末尾の「第3章への反映」）。
> - 定義・命題の番号は、骨組みの定理環境（section ごとの通し番号）で順に置いた場合の目安。TeX では各項目に付けた**提案ラベル**で参照し、番号を直接書かない。
> - 骨組みには「例」の環境がないので、例 2.7 のために追加が必要（definition スタイル、定義と同じカウンタ）。
> - 末尾の「実装との対応メモ」は付録A と `src/` の確認用で、本文には含めない。

---

## 2.1 記法

### 確率の枠組み

- 確率空間 $(\Omega,\mathcal{F},\mathbb{P})$ を固定し、確率変数はすべてその上で定義されているとする。期待値を $\mathbb{E}$ と書く。
- 「確率変数」という語は、実数値のものに限らず、ベクトルやデータ点の有限列に値をとるもの（確率要素）にも用いる。
- **本論文に現れる写像はすべて可測であるとする。**
- i.i.d. は「独立かつ同分布」の略である。

### 定義 2.1（同分布）　提案ラベル `def:eqd`

同じ空間に値をとる確率変数 $A,B$ について、任意の可測集合 $E$ に対し $\mathbb{P}(A\in E)=\mathbb{P}(B\in E)$ が成り立つとき、$A$ と $B$ は**同分布**であるといい、$A\overset{d}{=}B$ と書く。これは $A=B$（値として等しい）とは異なる。

### 命題 2.2（同分布の像）　提案ラベル `prop:image`

$A\overset{d}{=}B$ であり、$g$ がランダム性を含まない写像ならば、$g(A)\overset{d}{=}g(B)$ である。

**証明.** 任意の可測集合 $E$ について
$$\mathbb{P}(g(A)\in E)=\mathbb{P}(A\in g^{-1}(E))=\mathbb{P}(B\in g^{-1}(E))=\mathbb{P}(g(B)\in E).\qquad\blacksquare$$

### 命題 2.3（独立な組の同分布）　提案ラベル `prop:indep-pair`

$A$ と $B$ が独立、$A$ と $B'$ が独立で、$B\overset{d}{=}B'$ ならば、$(A,B)\overset{d}{=}(A,B')$ である。

**証明.** 任意の可測集合 $E,F$ について
$$\mathbb{P}(A\in E,\ B\in F)=\mathbb{P}(A\in E)\,\mathbb{P}(B\in F)=\mathbb{P}(A\in E)\,\mathbb{P}(B'\in F)=\mathbb{P}(A\in E,\ B'\in F).$$
直積集合 $E\times F$ の全体は共通部分について閉じており（π 系）、組の値の空間の積 σ-加法族を生成する。よって測度の一意性定理（π-λ 定理）より、$(A,B)$ と $(A,B')$ の同時分布は一致する。$\blacksquare$

### 天井関数と床関数

$u\in\mathbb{R}$ に対し $\lceil u\rceil:=\min\{\,m\in\mathbb{Z}: m\ge u\,\}$、$\lfloor u\rfloor:=\max\{\,m\in\mathbb{Z}: m\le u\,\}$ とする。このとき
$$u\ \le\ \lceil u\rceil\ <\ u+1$$
が成り立つ。左の不等式は定義から従う。右は、$\lceil u\rceil\ge u+1$ ならば $\lceil u\rceil-1$ も $u$ 以上の整数となり、最小性に反することから従う。また整数 $c$ と実数 $x$ について $c\ge x\iff c\ge\lceil x\rceil$ である。

### その他の記法

- $\#E$ は有限集合 $E$ の要素数、$\mathbf{1}\{\cdot\}$ は指示関数（条件が成り立てば 1、成り立たなければ 0）とする。$\mathbb{E}[\mathbf{1}\{E\}]=\mathbb{P}(E)$ である。
- $\{1,\dots,N\}$ から自分自身への全単射を**置換**といい、その全体を $\mathfrak{S}_N$ と書く。2 つの元だけを入れ替えて他を動かさない置換を**互換**という。
- 列 $w=(w_1,\dots,w_N)$ と $\pi\in\mathfrak{S}_N$ に対し、$w_\pi:=(w_{\pi(1)},\dots,w_{\pi(N)})$ と書く。

---

## 2.2 交換可能性

### 定義 2.4（交換可能性）　提案ラベル `def:exch`

同じ空間に値をとる確率変数の列 $W=(W_1,\dots,W_N)$ が**交換可能**であるとは、任意の $\pi\in\mathfrak{S}_N$ について $W_\pi\overset{d}{=}W$、すなわち
$$(W_{\pi(1)},\dots,W_{\pi(N)})\overset{d}{=}(W_1,\dots,W_N)$$
が成り立つことをいう。

並び順を入れ替えても同時分布が変わらない、という性質である。

### 命題 2.5（i.i.d. ならば交換可能）　提案ラベル `prop:iid-exch`

$W_1,\dots,W_N$ が i.i.d. ならば、$(W_1,\dots,W_N)$ は交換可能である。

**証明.** 共通の分布を $\mu$ とする。$\pi\in\mathfrak{S}_N$ と可測集合 $E_1,\dots,E_N$ について、$W_{\pi(1)},\dots,W_{\pi(N)}$ も独立なので
$$\mathbb{P}(W_{\pi(1)}\in E_1,\dots,W_{\pi(N)}\in E_N)=\prod_{l=1}^{N}\mu(E_l)=\mathbb{P}(W_1\in E_1,\dots,W_N\in E_N).$$
直積集合の上で同時分布が一致するので、命題 2.3 の証明と同じく一意性定理より $W_\pi\overset{d}{=}W$ である。$\blacksquare$

逆は成り立たない（例 2.7）。

### 命題 2.6（部分列と周辺分布）　提案ラベル `prop:subvector`

$(W_1,\dots,W_N)$ が交換可能ならば、次が成り立つ。

- **(i)** 任意の $M\le N$ について、$(W_1,\dots,W_M)$ も交換可能である。
- **(ii)** 任意の $i$ について $W_i\overset{d}{=}W_1$ である。

**証明.** (i) $\sigma\in\mathfrak{S}_M$ を、$M+1,\dots,N$ を動かさない $\pi\in\mathfrak{S}_N$ に延長すると $W_\pi\overset{d}{=}W$。最初の $M$ 座標を取り出す写像に命題 2.2 を適用すれば $(W_{\sigma(1)},\dots,W_{\sigma(M)})\overset{d}{=}(W_1,\dots,W_M)$ を得る。(ii) $\pi$ を $1$ と $i$ の互換とすると $W_\pi$ の最初の座標は $W_i$ である。最初の座標を取り出す写像に命題 2.2 を適用すれば $W_i\overset{d}{=}W_1$ を得る。$\blacksquare$

### 例 2.7（非復元抽出）　提案ラベル `ex:without-replacement`

値 $a_1,\dots,a_N$（重複してもよい）が書かれた $N$ 枚のカードを、1 枚ずつ非復元で引く。これは $\mathfrak{S}_N$ 上の一様分布に従う置換 $\Pi$ を用いて、$W_i:=a_{\Pi(i)}$（$i=1,\dots,N$）と表せる。

**交換可能である.** $h(\psi):=(a_{\psi(1)},\dots,a_{\psi(N)})$ とおくと、$\sigma\in\mathfrak{S}_N$ に対し $W_\sigma=h(\Pi\circ\sigma)$ である。$\psi\mapsto\psi\circ\sigma$ は $\mathfrak{S}_N$ 上の全単射なので $\Pi\circ\sigma$ も一様分布に従い、$\Pi\circ\sigma\overset{d}{=}\Pi$。命題 2.2 より $W_\sigma=h(\Pi\circ\sigma)\overset{d}{=}h(\Pi)=W$ である。命題 2.6 (i) より、最初の $M$ 回の結果 $(W_1,\dots,W_M)$（$M\le N$）も交換可能である。

**独立ではない.** $N=2$、$(a_1,a_2)=(0,1)$ とすると常に $W_1+W_2=1$ なので
$$\mathbb{P}(W_1=1,\ W_2=1)=0\ \ne\ \frac14=\mathbb{P}(W_1=1)\,\mathbb{P}(W_2=1).$$

したがって交換可能性は i.i.d. より真に弱い。さらに $N=n+1$ とすると、$W_1,\dots,W_n$ が決まれば残りの 1 枚 $W_{n+1}$ も決まる。交換可能であっても、最初の $n$ 個を与えたときの $W_{n+1}$ の条件付き分布は、i.i.d. の場合とは大きく異なりうる。

### 注意 2.8（交換可能性が破れる典型例）　提案ラベル `rem:exch-violation`

第3章の保証は、較正用データとテスト点を合わせた列の交換可能性に依存する。次のような場合には、一般に交換可能性は成り立たない。

- **時系列.** トレンドや自己相関があると、後から来るデータほど分布がずれ、並び順に意味が生じる。
- **共変量シフト.** テスト点の入力の分布が較正用データと異なる場合（4.4 節、第5章 E4）。
- **並び順やデータに依存する選び方.** 例えば、較正用データを予測がよく当たった点から選ぶと、較正用データとテスト点は対等でなくなる。

---

## 2.3 順序統計量と経験分位点

この節では実数の組 $s_1,\dots,s_n$（$n\ge1$）について定義し、確率変数の組には各 $\omega\in\Omega$ ごとに適用する。

### 定義 2.9（順序統計量）　提案ラベル `def:order-stat`

$s_{\sigma(1)}\le\cdots\le s_{\sigma(n)}$ となる $\sigma\in\mathfrak{S}_n$（小さい順に並べる置換）をとり、
$$s_{(l)}:=s_{\sigma(l)}\quad(l=1,\dots,n),\qquad s_{(n+1)}:=+\infty$$
と定める。$s_{(l)}$ を $l$ 番目の**順序統計量**という。同点があれば同じ値が重複して並ぶ。

このような $\sigma$ は常に存在するが、同点があると一意ではない。$s_{(l)}$ が $\sigma$ の選び方によらないことは補題 2.10 で示す。規約 $s_{(n+1)}:=+\infty$ は第3章の共形分位点で用いる（命題 2.13 の注を参照）。

### 補題 2.10（数え上げによる特徴づけ）　提案ラベル `lem:order-count`

$l\in\{1,\dots,n\}$、$t\in\mathbb{R}$ に対し、次が成り立つ。

- **(i)** $s_{(l)}\le t\iff\#\{\,i: s_i\le t\,\}\ge l$
- **(ii)** $s_{(l)}<t\iff\#\{\,i: s_i<t\,\}\ge l$

特に $s_{(l)}=\min\{\,t\in\mathbb{R}:\#\{i: s_i\le t\}\ge l\,\}$ であり、$s_{(l)}$ は $\sigma$ の選び方によらない。

**証明.** $\sigma$ を定義 2.9 の置換とする。

(i) ($\Rightarrow$) $j\le l$ ならば $s_{\sigma(j)}=s_{(j)}\le s_{(l)}\le t$ なので、相異なる $l$ 個の添字 $\sigma(1),\dots,\sigma(l)$ が $s_i\le t$ を満たす。($\Leftarrow$) 対偶を示す。$s_{(l)}>t$ ならば、$j\ge l$ について $s_{\sigma(j)}\ge s_{(l)}>t$ なので、$s_i\le t$ を満たす添字は $\sigma(1),\dots,\sigma(l-1)$ の中にしかなく、高々 $l-1$ 個である。

(ii) (i) の証明で「$\le t$」を「$<t$」に、「$>t$」を「$\ge t$」に置き換えればよい。

最後の主張について、(i) より $\{\,t:\#\{i: s_i\le t\}\ge l\,\}=[\,s_{(l)},\infty)$ であり、左辺は $\sigma$ によらない。$\blacksquare$

*注.* 確率変数 $S_1,\dots,S_n$ については、(i) より
$$\{S_{(l)}\le t\}=\bigcup_{I}\ \bigcap_{i\in I}\{S_i\le t\}$$
（$I$ は $\{1,\dots,n\}$ の $l$ 元部分集合を動く）が事象となるので、$S_{(l)}$ は確率変数である。同点があっても、この補題は添字の個数を数えているので成り立つ。

### 定義 2.11（経験分布関数と経験分位点）　提案ラベル `def:emp-quantile`

$$\hat F_n(t):=\frac1n\,\#\{\,i: s_i\le t\,\}\qquad(t\in\mathbb{R})$$
を**経験分布関数**という。$\beta\in(0,1]$ に対し
$$\hat Q_n(\beta):=\min\{\,t\in\mathbb{R}:\hat F_n(t)\ge\beta\,\}$$
を**経験 $\beta$ 分位点**という（最小値の存在は命題 2.12 で示す）。

### 命題 2.12（経験分位点は順序統計量）　提案ラベル `prop:emp-quantile`

$\beta\in(0,1]$ に対し $\hat Q_n(\beta)=s_{(\lceil n\beta\rceil)}$ である。

**証明.** $0<n\beta\le n$ より $\lceil n\beta\rceil\in\{1,\dots,n\}$。$\#\{i: s_i\le t\}$ は整数なので、2.1 節の天井関数の性質と補題 2.10 (i) より
$$\hat F_n(t)\ge\beta\iff\#\{i: s_i\le t\}\ge n\beta\iff\#\{i: s_i\le t\}\ge\lceil n\beta\rceil\iff s_{(\lceil n\beta\rceil)}\le t.$$
よって $\{\,t:\hat F_n(t)\ge\beta\,\}=[\,s_{(\lceil n\beta\rceil)},\infty)$ であり、その最小値は $s_{(\lceil n\beta\rceil)}$ である。$\blacksquare$

### 命題 2.13（$+\infty$ を加えた経験分位点）　提案ラベル `prop:aug-quantile`

$s_1,\dots,s_n$ と $+\infty$ にそれぞれ確率 $1/(n+1)$ を置いた分布を考え、その分布関数を
$$\tilde F(t):=\frac{1}{n+1}\,\#\{\,i\le n: s_i\le t\,\}\quad(t\in\mathbb{R}),\qquad\tilde F(+\infty):=1$$
とする。$\beta\in(0,1)$ に対し $\tilde Q(\beta):=\min\{\,t\in\mathbb{R}\cup\{+\infty\}:\tilde F(t)\ge\beta\,\}$ とおくと、
$$\tilde Q(\beta)=s_{(\lceil(n+1)\beta\rceil)}$$
である（$s_{(n+1)}=+\infty$ の規約のもとで）。

**証明.** $m:=\lceil(n+1)\beta\rceil$ とおくと、$0<(n+1)\beta<n+1$ より $m\in\{1,\dots,n+1\}$。$t\in\mathbb{R}$ については、命題 2.12 の証明と同様に $\tilde F(t)\ge\beta\iff\#\{i\le n: s_i\le t\}\ge m$ である。$m\le n$ ならば、補題 2.10 (i) よりこれは $s_{(m)}\le t$ と同値なので、最小値は $s_{(m)}$ である。$m=n+1$ ならば、これを満たす $t\in\mathbb{R}$ は存在せず（左辺は高々 $n$）、$\tilde F(+\infty)=1\ge\beta$ なので、最小値は $+\infty=s_{(n+1)}$ である。$\blacksquare$

*注.* 3.1 節の共形分位点 $\hat q=S_{(k)}$、$k=\lceil(n+1)(1-\alpha)\rceil$ は、この命題で $\beta=1-\alpha$ としたもの、すなわち**較正用スコアに、テスト点のための席として $+\infty$ を 1 つ加えた分布の $(1-\alpha)$ 分位点**である。$k\le n$ のときは命題 2.12 より $\hat q=\hat Q_n(k/n)$ でもあり、通常の経験分位点を、$1-\alpha$ より少し大きい水準 $k/n$ で取ったものにあたる。

### 注意 2.14（分位点の定義の違い）　提案ラベル `rem:quantile-types`

標本分位点には複数の定義があり、Hyndman and Fan (1996) は 9 通りを整理している。統計ソフトウェアの既定は、隣り合う順序統計量を線形補間するものが多い（例えば numpy.quantile の既定）。本論文の定義 2.11 はその第 1 の定義（経験分布関数の逆関数）にあたり、値は常にいずれかの順序統計量になる。

第3章の保証は順序統計量 $S_{(k)}$ そのものについてのものである。補間で得た値や、補正前の水準 $1-\alpha$ で計算した値は定理 3.3 の対象外であり、$S_{(k)}$ より小さい値になれば、被覆確率は $1-\alpha$ を下回りうる。

---

## 2.4 回帰モデルと分位点回帰　提案ラベル `sec:quantile-regression`

> v1.2 で追加、v1.3 で著者の決定（`chapter4_sections_spec.md` の決定事項 1〜3）を反映。第4章の CQR（`chapter4_sections_spec.md` の 4.2）が前提とする。第3章からは参照されない。

**本文の導入.** 第3章の点予測器 $\hat f$ は、$Y\mid X=x$ の分布の「中心」を推定するものであれば何でもよい（例えば最小二乗法による条件付き期待値の推定）。第4章の CQR では、中心ではなく $Y\mid X=x$ の分布の**分位点**を推定する回帰が要る。本節では、そのための損失関数（pinball 損失）と、分位点がその期待値の最小化元であることを述べる。

### 定義 2.15（分位点・条件付き分位点）　提案ラベル `def:cond-quantile`

実数値確率変数 $Y$ の分布関数を $F(t):=\mathbb{P}(Y\le t)$ とする。$\tau\in(0,1)$ に対し
$$Q(\tau):=\min\{\,t\in\mathbb{R}: F(t)\ge\tau\,\}$$
を $Y$ の **$\tau$ 分位点**という。$F$ は右連続で $\lim_{t\to-\infty}F(t)=0<\tau<1=\lim_{t\to\infty}F(t)$ なので、この集合は空でない下に有界な区間 $[Q(\tau),\infty)$ であり、最小値が存在する。定義 2.11 の経験分位点は、経験分布関数 $\hat F_n$ に対するこの定義である。

組 $(X,Y)$ について、$X=x$ を与えたときの $Y$ の条件付き分布関数を $F(t\mid x):=\mathbb{P}(Y\le t\mid X=x)$ と書き（脚注：実数値の $Y$ には正則条件付き分布が常に存在するので、$F(\cdot\mid x)$ を各 $x$ で分布関数になるようにとれる。本論文ではこれ以上立ち入らない）、
$$q_\tau(x):=\min\{\,t\in\mathbb{R}: F(t\mid x)\ge\tau\,\}$$
を **条件付き $\tau$ 分位点**という。$\tau=1/2$ なら条件付き中央値である。

### pinball 損失　提案ラベル `eq:pinball`

$\tau\in(0,1)$ に対し
$$\rho_\tau(u):=\begin{cases}\tau\,u & (u\ge0)\\ (\tau-1)\,u & (u<0)\end{cases}\ =\ \max\{\tau u,\ (\tau-1)u\}\qquad(u\in\mathbb{R})\tag{`eq:pinball`}$$
を **pinball 損失**（check 損失）という。$\rho_\tau\ge0$、$\rho_\tau(0)=0$ で、2 つの 1 次関数の最大値なので凸であり、傾きの絶対値は $\max\{\tau,1-\tau\}\le1$ である。$\tau=1/2$ のとき $\rho_{1/2}(u)=|u|/2$ で、絶対誤差の $1/2$ である。$\tau>1/2$ なら過小予測（$u>0$）の方が過大予測より重く罰せられ、最小化元は上側に寄る。

### 命題 2.16（分位点は pinball 損失の期待値を最小化する）　提案ラベル `prop:pinball-quantile`

$\mathbb{E}|Y|<\infty$、$\tau\in(0,1)$ とし、$L(c):=\mathbb{E}[\rho_\tau(Y-c)]$（$c\in\mathbb{R}$）とおく。このとき $c=Q(\tau)$ は $L$ を最小化する。

さらに、$(X,Y)$ について $\mathbb{E}|Y|<\infty$ ならば、$\mathbb{E}|g(X)|<\infty$ を満たす可測な $g\colon\mathcal{X}\to\mathbb{R}$ の中で $g=q_\tau$ が $\mathbb{E}[\rho_\tau(Y-g(X))]$ を最小化する。

**証明（本文に載せる。決定事項 1）.**

*(1) $L$ は有限で凸である.* $0\le\rho_\tau(Y-c)\le|Y|+|c|$ より $L(c)<\infty$。$\rho_\tau$ は凸なので、各 $\omega$ で $c\mapsto\rho_\tau(Y-c)$ は凸であり、期待値をとっても凸である。

*(2) 片側微分.* 実数 $y$ を固定すると、$c\mapsto\rho_\tau(y-c)$ は区分的に 1 次で、右微分係数は $\mathbf{1}\{y\le c\}-\tau$、左微分係数は $\mathbf{1}\{y<c\}-\tau$ である（$y>c$ では傾き $-\tau$、$y<c$ では傾き $1-\tau$、$y=c$ では右が $1-\tau$、左が $-\tau$）。差分商の絶対値は $\max\{\tau,1-\tau\}\le1$ で抑えられるので、有界収束定理により期待値と極限を交換でき、
$$L'_+(c)=\mathbb{P}(Y\le c)-\tau=F(c)-\tau,\qquad L'_-(c)=\mathbb{P}(Y<c)-\tau .$$

*(3) 凸関数の最小化.* 凸関数 $L$ について、$c'>c$ ならば差分商の単調性より $\dfrac{L(c')-L(c)}{c'-c}\ge L'_+(c)$、$c'<c$ ならば $\dfrac{L(c')-L(c)}{c'-c}\le L'_-(c)$ である。よって $L'_-(c)\le0\le L'_+(c)$ ならば、どちらの場合も $L(c')\ge L(c)$ であり、$c$ は $L$ を最小化する。

*(4) $q:=Q(\tau)$ が条件を満たす.* 定義より $F(q)\ge\tau$ なので $L'_+(q)\ge0$。$t<q$ では $q$ の最小性より $F(t)<\tau$ であり、$\mathbb{P}(Y<q)=\lim_{t\uparrow q}F(t)\le\tau$ なので $L'_-(q)\le0$。(3) より $q$ は $L$ を最小化する。

*(5) 条件付き版.* $\mathbb{E}[\rho_\tau(Y-g(X))]=\mathbb{E}\bigl[\,h_g(X)\,\bigr]$、$h_g(x):=\int\rho_\tau(y-g(x))\,F(dy\mid x)$ と書ける。各 $x$ で $h_g(x)$ は分布関数 $F(\cdot\mid x)$ をもつ確率変数について $L$ を $c=g(x)$ で評価したものなので、(1)〜(4) より $h_g(x)\ge h_{q_\tau}(x)$ である。$x$ について期待値をとれば結論を得る。$\blacksquare$

### 分位点回帰　提案ラベル `eq:quantile-regression`

データ $(x_i',y_i')$（$i=1,\dots,N$）と、関数のクラス $\mathcal{G}$ が与えられたとき、
$$\hat g_\tau\in\operatorname*{arg\,min}_{g\in\mathcal{G}}\ \frac1N\sum_{i=1}^{N}\rho_\tau\bigl(y_i'-g(x_i')\bigr)\tag{`eq:quantile-regression`}$$
を水準 $\tau$ の**分位点回帰**という。命題 2.16 の経験版であり、最小二乗法が条件付き期待値の経験版であるのと同じ関係にある。分位点回帰は Koenker and Bassett (1978) が $\mathcal{G}$ を線形関数のクラスとして提案した（`koenker1978regression`。決定事項 2：提案の出典としてのみ引き、命題 2.16 の根拠にはしない）。本論文では $\mathcal{G}$ を特定せず、pinball 損失を最小化する任意の回帰法を分位点回帰とよぶ（第5章で使うモデルは実験設定の節で述べる）。

**注（第4章への接続）.** 異なる水準 $\tau_1<\tau_2$ の分位点回帰は別々の最適化問題として解くので、$x$ によっては $\hat g_{\tau_1}(x)>\hat g_{\tau_2}(x)$ と推定値が**交差**しうる（真の分位点は $\tau$ について単調なので交差しない）。交差の扱いは第4章 CQR の節（`eq:cqr-pair`・`rem:cqr-crossing`）で述べる。

---

## 第3章への反映（`main_theorem_spec.md` v3 で反映済み。TeX 本文には含めない）

| 箇所 | v2 | v3 |
|---|---|---|
| 設定と記号：順序統計量 | 定義をその場で記述 | 定義 2.9 を参照（$S_{(n+1)}:=+\infty$ の規約を含む） |
| 設定と記号：$\hat q$ の後 | なし | 命題 2.13 による解釈を 1 文追加 |
| 設定と記号：記法・可測性 | その場で記述 | 「2.1 節に従う」に置き換え（TeX では 3.1 節末の 3 行を削除） |
| (A1) | 定義を式で書き下し | 「交換可能である（定義 2.4）」 |
| 補題 3.1 の証明 | 独立な組・同分布の像を文章で使用 | 命題 2.3・命題 2.2 を引用 |
| 補題 3.2 の主張 | 「交換可能」「順序統計量」を無参照で使用 | 定義 2.4・定義 2.9 を参照 |
| 補題 3.2 (a) の $(*)$ | 置換 $\sigma$ と集合 $B$ による証明 | 補題 2.10 (ii) の適用に置き換え（論理は同じ。証明を第2章に移した） |
| 補題 3.2 (b) | 同分布の像を文章で使用 | 命題 2.2 を引用 |
| 系 3.4 の証明 | 「天井関数の定義より」 | 「2.1 節の天井関数の不等式より」 |

---

## 実装との対応メモ（TeX 本文には含めない）

定義 2.11・命題 2.12 の経験分位点は、numpy の `method='inverted_cdf'`（Hyndman–Fan の第 1 の定義）に一致する。ただし実装には次の落とし穴がある。1〜3 は numpy 2.4.4 で確認した（本プロジェクトの numpy 1.26.4 では未確認。本実装は `np.quantile` を使わないので影響しない）。4 は Python の浮動小数点演算の問題なので numpy の版に依らず、本プロジェクトの環境でも再現した。

1. **`np.quantile(scores, k/n, method='inverted_cdf')` は浮動小数点誤差で $S_{(k+1)}$ を返すことがある.** $k/n$ に $n$ を掛け戻すと $k$ をわずかに超える場合に起きる。$n\le3000$ で調べると、$\alpha=0.1$ では 12 件（最小は $n=564$、$k=509$ で $S_{(510)}$ が返る）、$\alpha=0.05$ では 2 件（$n=2077,\ 2119$）、$\alpha=0.5$ では 334 件。
2. **`method='higher'` を水準 $k/n$ で使うと、$k<n$ のとき常に $S_{(k+1)}$ を返す.** 保守的すぎて、系 3.4 の上界を破る（被覆 $(k+1)/(n+1)$）。
3. **既定の `method='linear'` を補正前の水準 $1-\alpha$ で使うと、$S_{(k)}$ より小さい値を返しうる.** 例：$n=19$、$\alpha=0.1$ で $S_{(18)}$ ではなく $S_{(17)}$ と $S_{(18)}$ の間の値。下界を破りうる。
4. **$k$ を `ceil((n+1)*(1-alpha))` で計算すると、$\alpha$ によっては 1 ずれる.** 例：$\alpha=1/3$、$n=8$ では $9\cdot(1-1/3)$ が `6.000000000000001` と計算され、$k=7$ になる（正しくは 6）。$n\le3000$ でのずれの件数は、$\alpha=1/3$ と $2/3$ で各 990、$0.7$ で 300、$0.95$ で 150、$0.45$ で 76、$0.99$ で 30。本プロジェクトの実験で使う $\alpha\in\{0.05,0.10,0.20\}$ を含め、$\alpha\in\{0.01,0.05,0.1,0.2,0.25,0.3,0.5\}$ ではずれはなかった。

**推奨.** $k$ は整数・有理数の演算で求め（例：`Fraction(alpha).limit_denominator(10**9)` で $\alpha$ を有理数に戻してから天井をとる）、$\hat q$ は分位点関数を使わず、ソートして直接 `np.sort(scores)[k-1]`（$k\le n$ のとき。$k=n+1$ なら $+\infty$）で取る。分母の上限は `10**6` では不十分で、例えば $\alpha=0.123456789$ が $10/81$ に丸められてしまう。`10**9` なら小数点以下 9 桁までの $\alpha$ を正確に戻せる。

**`src/conformal/split.py` の確認結果.** $\hat q$ はソートして直接取っており、1〜3 には該当しない。$k$ の計算（50 行目）は 4 に該当する。実験の $\alpha$ では $k$ がずれないので、既存の `results/` への影響はない。同じ式が `experiments/e1b_detection_levels.py` の 46 行目にもある。

**その他.** docstring などから卒論の式を参照するときは、式番号ではなくラベル名（`eq:score`、`eq:khat`、`eq:interval`）を使う。式番号は第2章の式の追加でずれる（実際、`eq:ceil-bounds` の追加で 1 つずれた）。付録A の対応表では、$\hat q$ の計算を `eq:khat` と定義 2.9 に対応させるとよい。

---

## 変更履歴

| 版 | 内容 |
|---|---|
| v1 | 初版 |
| v1.1 | 実装との対応メモのみ更新（確認環境の明記、Claude Code による実装確認の結果、落とし穴 4 の該当 $\alpha$ の追記、`limit_denominator` の上限を `10**9` に変更）。TeX 本文への影響なし |
| v1.2 | 2.4 回帰モデルと分位点回帰を追加（定義 `def:cond-quantile`、pinball 損失 `eq:pinball`、命題 `prop:pinball-quantile` と証明の草案、分位点回帰 `eq:quantile-regression`）。第4章の仕様書 `chapter4_sections_spec.md` v1 の前提節。証明を本文に載せるかは TODO(著者)（第4章の仕様書の TODO 一覧 1〜3） |
| v1.3 | 2.4 に著者の決定を反映：命題 `prop:pinball-quantile` の証明を本文に載せる（凸性と片側微分。条件付き版は $x$ ごとに適用）、`koenker1978regression` は分位点回帰の提案の出典としてのみ引く、条件付き分位点は $F(t\mid x)$ から定義し正則条件付き分布の存在は脚注で一言。分位点回帰の記号を $\hat g_\tau$ に変更（第4章で入れ替え後を $\hat q$ とするため） |
