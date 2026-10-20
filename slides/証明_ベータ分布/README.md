# ゼミ補足スライド：被覆率がベータ分布に従う証明（論文 3.5 節）

- 用途：人工データの実験 1 の補足。論文 3.5 節（`sec:cond-coverage`）の定理 `thm:cond-coverage-beta` の証明を、ダーツのたとえで説明する（12 枚）
- 本体：claude.ai のスライド https://claude.ai/artifact/4uHfCZ4fQHiN3BaX5zkm7V （PowerPoint・PDF で書き出せる）
- `project/`：スライドの元ファイル（deck.json と 1 枚ずつの HTML。`<aside>` が読み上げ原稿）
- `figures/`：数式の画像（matplotlib の mathtext）と図、それを作る `make.py`
  （実行例：`.venv/bin/python slides/証明_ベータ分布/figures/make.py slides/証明_ベータ分布/figures`）
- 11 枚目の図は `slides/進捗_人工データ_E1-E6/figures/e1_beta.png` の再掲
