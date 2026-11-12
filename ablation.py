"""Task 5: the ablation study.

Row B0 is the naive baseline; every following row turns on one more switch of hw4.SETTINGS, and the
last row is the submitted pipeline (the defaults of hw4.SETTINGS). Each row runs the unchanged runner
of hw4.py on public_test and keeps its results.csv.

    python ablation.py                  # every row, with the generator (needs DEEPSEEK_API_KEY)
    python ablation.py --no-generate    # retrieval only (Tasks 1 to 3): no API calls

Writes ablation/<row>.csv and the table ablation/ablation.md.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path

import hw4

BASELINE = {
    "table_text": False, "section_chunks": False, "chunk_header": False, "tokenizer": "whitespace",
    "query_words": False, "metadata_filter": False, "per_company": False, "table_boost": 1.0, "abstain": False,
}
# (row, what changes relative to the row above, the switch it turns on)
ROWS = [
    ("B0", "Naive RAG: get_text(), RecursiveCharacterTextSplitter, whitespace BM25 top-5 over the corpus, never []", {}),
    ("V1", "Tables through table_to_text", {"table_text": True}),
    ("V2", "Chunks follow the headings instead of fixed-size pieces", {"section_chunks": True}),
    ("V3", "Contextual chunk header (company, document, section)", {"chunk_header": True}),
    ("V4", "Tokeniser: no punctuation, light stemming, British spelling", {"tokenizer": "stemmed"}),
    ("V5", "Query: content words, synonyms, abbreviations", {"query_words": True}),
    ("V6", "parse_query and metadata filter (company, year, document type)", {"metadata_filter": True}),
    ("V7", "Comparisons: the companies take turns", {"per_company": True}),
    ("V8", "Table chunks x1.3 when the question asks for or compares figures", {"table_boost": 1.3}),
    ("V9", "Abstention: unknown company, uncovered year, unknown word + LLM check", {"abstain": True}),
]
SUBMITTED = dict(hw4.SETTINGS)


def scores(path: Path, truth: dict) -> dict:
    """The runner's measures, recomputed from a results.csv and the ground truth."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    answerable = [r for r in rows if truth[r["id"]]["evidence"]]
    unanswerable = [r for r in rows if not truth[r["id"]]["evidence"]]
    recall = sum(r["retrieval_hit"] == "True" for r in answerable) / len(answerable)
    kept = sum(bool(r["retrieved"]) for r in answerable) / len(answerable)
    rejected = sum(not r["retrieved"] for r in unanswerable) / len(unanswerable)
    accuracy = sum(r["correct"] == "True" for r in rows) / len(rows)
    by_category = {}
    for category in ("fact", "table", "comparison", "unanswerable"):
        group = [r for r in rows if truth[r["id"]]["category"] == category]
        by_category[category] = sum(r["correct"] == "True" for r in group) / len(group)
    points = (hw4.POINTS["2.1"] * sum(r["valid"] == "True" and bool(r["retrieved"]) for r in answerable) / len(answerable)
              + hw4.POINTS["2.3"] * hw4.clip((recall - hw4.RECALL_ZERO) / (hw4.RECALL_FULL - hw4.RECALL_ZERO))
              + hw4.POINTS["3"] * hw4.clip(kept + rejected - 1)
              + hw4.POINTS["4"] * accuracy)
    return {"recall": recall, "kept": kept, "rejected": rejected, "accuracy": accuracy,
            "points": points, **by_category}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--out", default="ablation")
    parser.add_argument("--no-generate", action="store_true", help="skip the generator (Task 4 then shows 0)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    truth = hw4.load_json(Path(args.data_folder) / "ground_truth.json")
    if args.no_generate:
        hw4.generate = lambda question, chunks: "NOT_FOUND"  # the runner looks generate up at call time
        hw4.os.environ.pop("DEEPSEEK_API_KEY", None)         # and no LLM check in retrieve

    settings = dict(BASELINE)
    results = []
    for row, change, switch in ROWS:
        settings.update(switch)
        hw4.SETTINGS.clear()
        hw4.SETTINGS.update(settings)
        path = out / f"{row}.csv"
        sys.argv = ["hw4.py", "--data-folder", args.data_folder, "--output", str(path)]
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            hw4.main()
        (out / f"{row}.log").write_text(log.getvalue(), encoding="utf-8")
        result = scores(path, truth)
        results.append((row, change, result))
        print(f"{row}  recall {result['recall']:6.1%}  kept {result['kept']:6.1%}  rejected {result['rejected']:6.1%}"
              f"  accuracy {result['accuracy']:6.1%}  points(2.1-4) {result['points']:5.2f}  | {change}", flush=True)
    assert settings == SUBMITTED, "the last row must be the submitted pipeline"
    hw4.SETTINGS.update(SUBMITTED)

    lines = ["| Row | Change relative to the row above | Recall@5 | Kept | Rejected | Fact | Table | Comparison"
             " | Unanswerable | Accuracy | Points 2.1-4 (of 50) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row, change, r in results:
        lines.append(f"| {row} | {change} | {r['recall']:.1%} | {r['kept']:.1%} | {r['rejected']:.1%} | {r['fact']:.0%}"
                     f" | {r['table']:.0%} | {r['comparison']:.0%} | {r['unanswerable']:.0%} | {r['accuracy']:.1%}"
                     f" | {r['points']:.2f} |")
    note = "\n\nGenerator skipped (--no-generate): accuracy columns are not meaningful.\n" if args.no_generate else "\n"
    (out / "ablation.md").write_text("\n".join(lines) + note, encoding="utf-8")
    print(f"wrote {out / 'ablation.md'}")


if __name__ == "__main__":
    main()
