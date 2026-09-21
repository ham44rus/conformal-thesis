"""コード・仕様書・テストが参照する卒論のラベルが、TeX と整合していることの検査。

対応箇所: 卒論全体（thesis/R08_NAME.tex の \\label）と
          thesis/drafts/chapter4_5_labels_spec.md（第4・5章の確定ラベル）

卒論の式番号・節番号は章を書き足すたびにずれるので、コードからは番号ではなく
\\label の名前で参照する約束にしている（CLAUDE.md「実装方針」の docstring の規則）。
このテストはその約束が守られているかを機械的に確かめる。

1. 「4.1 節」「5.5節」のような番号による節参照が残っていないこと
2. 参照している eq: / sec: / def: / thm: / prop: / lem: / cor: / rem: / ex: / tab: ラベルが、
   TeX の \\label か、第4・5章のラベル仕様書で
   確定したラベルのどちらかに存在すること
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("experiments", "src", "specs", "tests")
SCAN_SUFFIXES = {".py", ".md"}
THIS_FILE = Path(__file__).resolve()

TEX = ROOT / "thesis" / "R08_NAME.tex"
LABELS_SPEC = ROOT / "thesis" / "drafts" / "chapter4_5_labels_spec.md"

# 「4.1 節」「5.5節」「5.1.1 項」「第3章6節」「第3章第6節」「§5.1」。
# 「第3章」のような章単位の参照は章番号がずれないので許す
SECTION_NUMBER = re.compile(
    r"\d+\.\d+(?:\.\d+)? ?[節項]"        # 5.1節 / 5.1 節 / 5.1.1項
    r"|第 ?\d+ ?章 ?第? ?\d+ ?[節項]"        # 第3章6節 / 第3章第6節
    r"|§ ?\d+(?:\.\d+)*"                  # §5.1 / § 5.1.1
)
# 参照される可能性のあるラベルの接頭辞（式・節・定義・定理・命題・補題・系・注意・例・表）
PREFIXES = r"(?:eq|sec|def|thm|prop|lem|cor|rem|ex|tab)"
LABEL_REF = re.compile(r"\b" + PREFIXES + r":[a-z0-9][a-z0-9-]*")
TEX_LABEL = re.compile(r"\\label\{(" + PREFIXES + r":[a-z0-9-]+)\}")
SPEC_LABEL = re.compile(r"`(" + PREFIXES + r":[a-z0-9-]+)`")


def _scan_files() -> list[Path]:
    files = []
    for d in SCAN_DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.suffix in SCAN_SUFFIXES and p.resolve() != THIS_FILE and "__pycache__" not in p.parts:
                files.append(p)
    return sorted(files)


def test_番号による節参照が残っていない():
    hits = []
    for p in _scan_files():
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if SECTION_NUMBER.search(line):
                hits.append(f"{p.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not hits, "番号による節参照が残っている（ラベル名に直すこと）:\n" + "\n".join(hits)


def test_参照ラベルがTeXか確定ラベル仕様書に存在する():
    tex_labels = set(TEX_LABEL.findall(TEX.read_text(encoding="utf-8")))
    spec_labels = set(SPEC_LABEL.findall(LABELS_SPEC.read_text(encoding="utf-8")))
    known = tex_labels | spec_labels
    assert tex_labels, "TeX から \\label が 1 つも読めない（パスを確認）"

    missing = {}
    for p in _scan_files():
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in LABEL_REF.finditer(line):
                label = m.group(0)
                # ラベル仕様書の中の「sec:e<番号>」のような雛形は対象外
                if "<" in line[m.end():m.end() + 1]:
                    continue
                if label not in known:
                    missing.setdefault(label, []).append(f"{p.relative_to(ROOT)}:{lineno}")
    assert not missing, "TeX にも確定ラベル仕様書にも無いラベルを参照している:\n" + "\n".join(
        f"  {lab}: {', '.join(locs)}" for lab, locs in sorted(missing.items())
    )


@pytest.mark.parametrize(
    "text", ["5.1節", "5.1 節", "5.1.1項", "5.1.1 項", "第3章6節", "第3章第6節", "§5.1", "§ 5.1.1"]
)
def test_検出パターンは番号による節参照を拾う(text):
    assert SECTION_NUMBER.search(text), f"{text!r} を節参照として検出できない"


@pytest.mark.parametrize("text", ["第3章", "第5章に書くこと", "alpha=0.1", "Beta(18,2)", "sec:e1", "n=1..3000"])
def test_検出パターンは章参照や数値を拾わない(text):
    assert not SECTION_NUMBER.search(text), f"{text!r} を誤検出した"
