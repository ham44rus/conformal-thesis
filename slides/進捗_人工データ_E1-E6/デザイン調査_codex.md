# 投影スライドの可読性監査

15枚の HTML と描画画像を確認しました。**見出しの 64px はおおむね十分です。優先して直すべきなのは、28〜32px の本文、24px の長い注、約20px相当の図中ラベル、そして一部のページの情報量です。** 特に7・10枚目は、図を表示上だけ拡大してもラベルを十分大きくできません。

## A. 調査の要点

1. **文字サイズは投影距離で決まります。** 大きな部屋の後方から読むスライドには少なくとも24pt、教室向けには本文28pt以上という大学の指針があります。CSSでは24pt＝32px、28pt≒37pxです。ただし、1920×1080という画素数だけでは客席からの読みやすさは保証できません。[ワシントン大学 DO-IT](https://doit.uw.edu/knowledge-base/how-can-you-make-your-presentation-accessible)、[同大学 HCDE](https://www.hcde.washington.edu/inclusive-teaching-toolkit/instructor-resources/accessible-course-content)、[W3CのCSS単位](https://www.w3.org/TR/css3-values/)

2. **WCAGのコントラスト比は下限として使います。** 通常の文字は4.5:1以上、大きな文字は3:1以上、意味を伝える図形や線は3:1以上が基準です。投影はモニターよりコントラストが落ち得るため、この資料の本文・注・図ラベルは**7:1程度を目標**にすると余裕があります。WCAG適合だけで後方から読めるとは判断できません。[W3C：文字](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum)、[W3C：図形](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html)、[UCSBの投影資料](https://capstone.cs.ucsb.edu/cs189b/lectures/CS189B_LectureSlides_2-13.pdf)

3. **青と橙は使いやすい組み合わせですが、色だけに意味を任せません。** 線種、点の形、直接ラベルを併用します。1枚では背景・基本文字に加え、意味を持つ強調色は青と橙の**2色程度**に絞るのが、この資料に適した設計上の目安です。色数はWCAGの規定ではありません。[W3C：色の使用](https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html)、[Wongの色覚に関する論考](https://www.nature.com/articles/nmeth.1618)

4. **BIZ UDPGothic は維持してよい書体です。** UDを意図して作られ、RegularとBoldがあります。見出し・主要数値は700、本文は400を基本にし、細い字形や淡い文字色を避けます。日本語の複雑な漢字や数式は、英数字だけの表示より実物で確認してください。[Google FontsによるBIZ UD書体の紹介](https://fonts.googleblog.com/2022/04/morisawa-biz-universal-design-ud.html)

5. **行間は本文で1.35〜1.5、表で1.25〜1.4を目安にします。** 日本語の字間はまず標準のままにし、必要な箇所だけ0〜0.02em程度を試すのが妥当です。WCAGの「1.5倍の行間・0.12emの字間」は、利用者がそのように変更しても内容が欠けないという要件であり、常時その字間にする推奨値ではありません。[W3C：テキスト間隔](https://www.w3.org/WAI/WCAG22/Understanding/text-spacing.html)

6. **1枚1主張を優先します。** この日本語発表では、本文をおおむね3〜5行、1行20〜30字、表は見出しを除き3〜5行から検討するとよいでしょう。これは固定規則ではなく編集の出発点です。詳細な条件、検定結果、文献は発表者ノートや配布資料へ移します。[Penn Stateのスライド設計指針](https://www.ems.psu.edu/accessibility/cut-clutter-designing-slides-everyone-can-follow)、[ワシントン大学 DO-IT](https://doit.uw.edu/knowledge-base/how-can-you-make-your-presentation-accessible)

7. **図は縮小後の大きさで評価します。** 軸、目盛り、凡例、直接ラベルも本文に近い大きさが必要です。図を複雑なまま小さく載せるより、比較対象を減らして要点を口頭で説明する方法が勧められています。投影する教室で、照明を通常の状態にして最後列から確認してください。[ワシントン大学 DO-IT](https://doit.uw.edu/knowledge-base/how-can-you-make-your-presentation-accessible)、[UCSBの投影資料](https://capstone.cs.ucsb.edu/cs189b/lectures/CS189B_LectureSlides_2-13.pdf)

## B. この資料への具体的な推奨

以下は**1920×1080で等倍表示するHTMLのCSS px**を前提とした、最後列向けの設計値です。部屋の広さ・スクリーンの大きさが未指定なので、最終判断は現地投影で行います。資料の並びと書体設定は :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/deck.json" purpose="source"} で確認しました。

| 要素 | 最小の目安 | 推奨 | 現状への判断 |
|---|---:|---:|---|
| 見出し | 56px | **64〜72px** | 現在の64pxは維持。長い見出しは短くする |
| 本文 | 38px | **40〜44px** | 28〜32pxから拡大し、文章を削る |
| 表の文字 | 36px | **38〜40px** | 28〜30pxから拡大。行・列を減らす |
| ページ下の注 | 32px | **34〜36px** | 24pxは小さい。説明文は注から本文・ノートへ |
| 図中の文字 | 32px | **36〜40px** | 約20px相当は作図時の文字から作り直す |
| ページ番号だけ | 26px | 28〜30px | 読む必要のある注と分けて扱う |

**配色の計算値**（WCAGの相対輝度式。実際の投影コントラストの測定値ではありません）：

| 用途 | 文字・線／背景 | コントラスト比 | 推奨 |
|---|---|---:|---|
| 基本文字 | `#1F2A44`／`#F7F6F2` | **13.18:1** | 維持 |
| 基本文字 | `#1F2A44`／`#FFFFFF` | **14.26:1** | 維持 |
| 補助文字 | `#4A5568`／`#F7F6F2` | **6.96:1** | 本文には使えるが、主要な結論は紺へ |
| 現在の薄い注 | `#6B7486`／`#F7F6F2` | **4.35:1** | `#4A5568`または紺へ |
| 青の強調文字 | `#2F6DB5`／`#FFFFFF` | **5.28:1** | 大きい文字なら可。図の線・文字は濃い青 `#1F5A9D` も検討 |
| 現在の橙 | `#D9822B`／`#FFFFFF` | **2.93:1** | 文字や重要な図の線には使わない |
| 提案する濃い橙 | `#A34F00`／`#FFFFFF` | **5.71:1** | 白地の文字・重要な線に使用 |
| 暗色面の文字 | `#F7F6F2`／`#1F2A44` | **13.18:1** | 維持 |
| 暗色面の橙 | `#F6C28B`／`#1F2A44` | **8.83:1** | 暗色面の強調に使用 |

最優先で直す色は、白系背景の**`#6B7486`の長い注**と**`#D9822B`の文字・重要線**です。後者は大きな文字に適用する3:1にも白地でわずかに届きません。`#6B7486`の24px注はWCAG上の「大きな文字」に相当し得ますが、投影用としてはサイズ・コントラストとも余裕がありません。淡い矢印 `#9AA6B8` も生成り背景に対して**2.28:1**なので、流れを示す重要な矢印なら濃くします。図の青・橙には、線種や点形状も割り当ててください。[WCAGの文字基準](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum)、[図形基準](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html)

余白は現状の外側128pxを概ね維持できます。本文を拡大して収まらない場合は、まず**補足文、重複、表の比較対象**を減らしてください。強調は太字・位置・囲みを主に使い、橙の着色だけに頼らない設計が適しています。

## C. スライドごとの問題と対処

| 枚 | 問題と具体的な対処 |
|---|---|
| **1 表紙** | 主題は十分大きい。日付・所属・氏名は32pxなので、最後列で必要なら36〜40pxへ。`[氏名]`は発表前に埋める。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/cover.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_01.png" purpose="source"} |
| **2 研究テーマ** | 右側の条件説明と下の等角予測の定義が重複する。下の定義は「条件のもとで、新しい点が範囲に入る確率は90%以上」の一文にし、交換可能性の詳説は口頭へ。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/theme.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_02.png" purpose="source"} |
| **3 しくみ** | 右下の順位の説明が長く、28〜30pxのままでも密集している。**手順図と「99点なら90番目」の計算例を分ける**。計算例は別枚または発表者ノートに置き、この枚は3段階と帯の意味に集中する。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/mechanism.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_03.png" purpose="source"} |
| **4 これまでの流れ** | 3つの箱の説明を各1行に縮められる。10〜15分の本編では、この流れを口頭で述べて枚自体を省く選択もできる。その時間を3・7枚目の説明に充てる。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/timeline.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_04.png" purpose="source"} |
| **5 人工データ** | 数式2本と説明3か所が競合している。各データの違いを「誤差が一定／場所で変わる」に絞り、実験番号と入力Xの細部はノートへ移す。数式を残すなら40px前後で読みやすくする。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/artificial.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_05.png" purpose="source"} |
| **6 実験の一覧** | 6行の問いが長く、下の説明も加わる。問いを「保証」「モデル」「組別」「分布ずれ」「分割比」「少数データ」程度に短縮し、最下段の定義は口頭へ。38pxの表にするならデータ列も簡略化する。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/overview.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_06.png" purpose="source"} |
| **7 実験1** | **最優先。** 3つのヒストグラムの軸・目盛りが小さく、24pxの注に結果と検定上の補足を詰め込んでいる。図は大きな文字で再作成し、必要なら「20点と2,000点」の2例を本編、100点と検定の詳細を補足へ。注は条件の最小限だけ残す。横軸の範囲が異なることは図の近くで明示する。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/exp1.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_07.png" purpose="source"} |
| **8 実験2** | 5モデルすべてを38〜40pxで載せるのは窮屈。主張に必要な「定数・リッジ・代表的な高性能モデル」の3行を本編にし、全5行は補足へ。結論文も「当たる割合はほぼ90%、幅は2.96から約1.8へ」と短くする。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/exp2.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_08.png" purpose="source"} |
| **9 実験3の準備** | 3方式の図は有効だが、説明文28pxが小さい。各方式を「同じ幅」「外れやすい所で広い」「上下端を予測」の一文にし、具体的な計算法は口頭へ。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/methods3.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_09.png" purpose="source"} |
| **10 実験3** | **最優先。** 3面の図で軸・系列のラベルが小さい。図中文字を作図段階で36px程度にし、灰色の比較線は濃くする。3面を保てなければ「絶対残差」と「改善2方式」に分ける。上の長い条件文は「X1で5組に分けた」程度に縮める。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/exp3.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_10.png" purpose="source"} |
| **11 実験4** | 表の核心は「ずれ大で0.766、重み付きで0.901、その代わり幅3.27」。本編ではこの比較を大きく示し、小・中の列と長い結論文は補足へ移せる。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/exp4.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_11.png" purpose="source"} |
| **12 実験5** | 5列を大きくすると幅が不足する。主題である「較正用点数」と「1回ごとのぶれ」を主表に残し、区間幅・予測誤差は短い補助数値か補足へ。注の指標定義は発表中に口頭で説明する。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/exp5.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_12.png" purpose="source"} |
| **13 実験6** | **情報量が多い。** 上の3つの方法説明と下の表を同時に40px近くへ拡大できない。方法の定義は前口上か別枚に移し、本編は50点の比較を大きく表示する。100・200点の行と理論保証の細部は補足へ。表の「割合／幅」は列見出しで明示する。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/exp6.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_13.png" purpose="source"} |
| **14 まとめ** | 4つの長い結論と2つの相談事項が同じ枚にある。結論は**3つの短文**に集約し、次回予定を1行に。相談事項は口頭で尋ねるか別枚へ。濃紺上の白文字は良好だが、相談欄の28pxは拡大する。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/summary.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_14.png" purpose="source"} |
| **15 補足** | 質疑用と明示されているため本編では飛ばせる。表示するなら表の第3列が長いので、「正しい実装／順位の誤り／テスト点不足」の3ケースと検出指標だけを大きく示し、数値の詳説は口頭へ。:codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/project/slides/appendix1b.html" purpose="source"} :codex-file-citation{path="/Users/rintaro/.orchestra/projects/conformal-thesis-cacc1599/research/slide-design/render/e16_15.png" purpose="source"} |

**作業順は、7・10枚目の図の作り直し、全ページの注の整理、3・13・14枚目の文章削減、最後に教室の最後列での確認**を勧めます。ファイルは変更していません。