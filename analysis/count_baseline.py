"""Enumeration baseline for the counting failure class (post-v1.0 upgrade).

The Build composer answers every count question with the constant
"one". This baseline asks whether the pixels carry the count: it labels
4-connected non-background regions in each Deep document figure and
answers with their number. No learning, no encoder. It is a ceiling
check on this synthetic set (shapes are drawn at fixed, non-touching
slots), not a counting model, and it says nothing about real images.

Deterministic, NumPy only. Writes results_count_baseline.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("level-1-learn", "level-2-build", "level-3-deep"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import clip_numpy as clip  # noqa: E402
import deep_eval as deep  # noqa: E402


def count_regions(image_flat):
    """Number of 4-connected regions whose pixels differ from the background."""
    img = image_flat.reshape(clip.IMAGE_SIZE, clip.IMAGE_SIZE, 3)
    fg = np.any(np.abs(img - clip.BACKGROUND) > 1e-9, axis=2)
    seen = np.zeros_like(fg)
    regions = 0
    for y0 in range(fg.shape[0]):
        for x0 in range(fg.shape[1]):
            if not fg[y0, x0] or seen[y0, x0]:
                continue
            regions += 1
            stack = [(y0, x0)]
            seen[y0, x0] = True
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < fg.shape[0] and 0 <= nx < fg.shape[1] and fg[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return regions


def main():
    rows = []
    for doc in deep.make_documents():
        n = count_regions(doc["image"])
        answer = deep.NUM_WORDS.get(n, "unknown")
        gold = deep.gold_answer(doc, "count")
        rows.append({"doc_id": doc["doc_id"], "item_count": doc["item_count"], "regions": n,
                     "answer": answer, "labelled_answer": gold, "exact_match": 1.0 if answer == gold else 0.0})
    by_n = {str(k): round(float(np.mean([r["exact_match"] for r in rows if r["item_count"] == k])), 6)
            for k in (1, 2, 3, 4)}
    results = {
        "project": "P22 multimodal-lab",
        "analysis": "enumeration baseline (connected regions) on the Deep count questions",
        "data_is_public": False,
        "cases": len(rows),
        "exact_match": round(float(np.mean([r["exact_match"] for r in rows])), 6),
        "exact_match_by_gold_items": by_n,
        "per_case": rows,
    }
    path = os.path.join(HERE, "results_count_baseline.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("count baseline exact match", results["exact_match"], by_n)
    print("wrote", path)


if __name__ == "__main__":
    main()
