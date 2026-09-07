# 卒業研究 タスクランナー
# Windows など python3 が無い環境では:  make PYTHON=python test
PYTHON ?= python3

.PHONY: help setup test e1 e1b e2 all-experiments figures clean check

help:
	@echo "make setup      依存パッケージをインストール"
	@echo "make test       検証テストを実行（実装を信用する唯一の根拠）"
	@echo "make e1         実験E1を実行し results/ に csv を出力"
	@echo "make e1b        実験E1b（検証の水準）を実行"
	@echo "make e2         実験E2（モデル非依存性）を実行"
	@echo "make figures    results/ の csv から figures/ の図を再生成"
	@echo "make check      test + 主要実験の通し確認"
	@echo "make clean      中間生成物を削除（results/ figures/ は消さない）"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

e1:
	$(PYTHON) experiments/e1_coverage.py --seed 20260901

e1b:
	$(PYTHON) experiments/e1b_detection_levels.py --seed 7

e2:
	$(PYTHON) experiments/e2_model_agnostic.py --seed 20260901

figures:
	$(PYTHON) experiments/make_figures.py

check: test e1 e1b e2

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
