# 進捗報告スライド：実データ解析（E7）

- 用途：ゼミの進捗報告（次回以降に使う予定。2026-10-18 作成）
- 本体：claude.ai のスライド https://claude.ai/artifact/JJxBye4nQdqNUGdyT47mSW （PowerPoint・PDF で書き出せる）
- `project/`：スライドの元ファイル（deck.json と 1 枚ずつの HTML。`<aside>` が読み上げ原稿）
- `figures/`：スライド専用の図（11〜13 枚目）と、それを `results/e7_summary.csv` から作るスクリプト `make.py`
  （実行例：`.venv/bin/python slides/進捗_実データ解析_E7/figures/make.py . slides/進捗_実データ解析_E7/figures`）
- 内容の根拠：論文 第6章（`sec:real-data`）、`results/e7_*.csv`。Codex のレビューを 6 回反映
