"""Paired statistics for the Deep study (post-v1.0 upgrade).

Reads level-3-deep/results_deep.json per_case (144 rows, already
recorded) and adds the uncertainty the v1.0 write-up did not report:
- percentile bootstrap 95% CIs on exact match, per question type
  (resampling the 24 cases of that type, one per document) and
  overall (resampling the 24 documents with all 6 of their cases);
- the exact McNemar test, vision against text, on the paired
  exact-match outcomes (two-sided binomial test on the discordant pairs);
- the count question broken down by gold item count (1 to 4).
No model is run. Deterministic (fixed bootstrap seed), standard
library plus NumPy. Writes results_stats.json beside this file.
"""
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
B = 10000
SEED = 20261107
TYPES = ["colour", "shape", "both", "count", "title", "left_shape"]


def mcnemar_exact(vision, text):
    b = int(sum(1 for v, t in zip(vision, text) if v == 1 and t == 0))
    c = int(sum(1 for v, t in zip(vision, text) if v == 0 and t == 1))
    n, k = b + c, min(b, c)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"vision_only_correct": b, "text_only_correct": c, "discordant": n, "p_value": p}


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def main():
    with open(os.path.join(ROOT, "level-3-deep", "results_deep.json"), encoding="utf-8") as fh:
        rows = json.load(fh)["per_case"]
    rng = np.random.default_rng(SEED)
    out = {"per_question_type": {}}
    for qt in TYPES:
        r = [x for x in rows if x["question_type"] == qt]
        v = np.array([x["vision_exact_match"] for x in r])
        t = np.array([x["text_exact_match"] for x in r])
        idx = rng.integers(0, len(r), size=(B, len(r)))
        out["per_question_type"][qt] = {
            "cases": len(r),
            "vision_em": round(float(v.mean()), 6), "text_em": round(float(t.mean()), 6),
            "vision_em_ci95": ci(v[idx].mean(axis=1)), "text_em_ci95": ci(t[idx].mean(axis=1)),
            "difference_ci95": ci((v[idx] - t[idx]).mean(axis=1)),
            "mcnemar": mcnemar_exact(v, t),
        }
    docs = sorted({x["doc_id"] for x in rows})
    by_doc = {d: [x for x in rows if x["doc_id"] == d] for d in docs}
    dv = np.array([[x["vision_exact_match"] for x in by_doc[d]] for d in docs])
    dt = np.array([[x["text_exact_match"] for x in by_doc[d]] for d in docs])
    idx = rng.integers(0, len(docs), size=(B, len(docs)))
    out["overall"] = {
        "cases": len(rows),
        "vision_em": round(float(dv.mean()), 6), "text_em": round(float(dt.mean()), 6),
        "vision_em_ci95": ci(dv[idx].mean(axis=(1, 2))), "text_em_ci95": ci(dt[idx].mean(axis=(1, 2))),
        "difference_ci95": ci((dv[idx] - dt[idx]).mean(axis=(1, 2))),
        "mcnemar": mcnemar_exact([x["vision_exact_match"] for x in rows], [x["text_exact_match"] for x in rows]),
        "bootstrap_unit": "document (24 clusters of 6 cases)",
    }
    count_rows = [x for x in rows if x["question_type"] == "count"]
    out["count_by_gold_items"] = {
        str(n): {
            "cases": len([x for x in count_rows if x["item_count"] == n]),
            "vision_em": round(float(np.mean(
                [x["vision_exact_match"] for x in count_rows if x["item_count"] == n])), 6),
            "text_em": round(float(np.mean([x["text_exact_match"] for x in count_rows if x["item_count"] == n])), 6),
            "vision_answers": sorted({x["vision_answer"] for x in count_rows if x["item_count"] == n}),
        } for n in (1, 2, 3, 4)}
    results = {
        "project": "P22 multimodal-lab",
        "analysis": "paired statistics over results_deep.json per_case (no rerun)",
        "source": "level-3-deep/results_deep.json",
        "bootstrap_resamples": B, "bootstrap_seed": SEED,
        "mcnemar": "exact two-sided binomial test on discordant pairs",
        **out,
    }
    path = os.path.join(HERE, "results_stats.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")
    for k, v in results["per_question_type"].items():
        print(k, v["vision_em_ci95"], v["text_em_ci95"], v["mcnemar"])
    print("overall", results["overall"])
    print("wrote", path)


if __name__ == "__main__":
    main()
