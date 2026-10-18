# 進捗報告スライド：人工データでの検証（実験 1〜6）

- 用途：2026-10-20 のゼミの進捗報告（前回は定理の証明。今回は人工データの実験 E1〜E6、補足に E1b）
- 本体：claude.ai のスライド https://claude.ai/artifact/48CppWijZoFYKh358a2R5V （PowerPoint・PDF で書き出せる）
- `project/`：スライドの元ファイル（deck.json と 1 枚ずつの HTML。`<aside>` が読み上げ原稿）
- `figures/`：スライド専用の図（7・10 枚目）と、それを `results/` から作るスクリプト `make.py`
  （実行例：`.venv/bin/python slides/進捗_人工データ_E1-E6/figures/make.py . slides/進捗_人工データ_E1-E6/figures`）
- 内容の根拠：論文 第5章（`sec:e1`〜`sec:e6`）、`results/e1*`〜`e6*` の csv。2・3・9 枚目は実データ版（`slides/進捗_実データ解析_E7/`）と共通
