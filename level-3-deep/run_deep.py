"""Recorded Deep run: train the encoder, score both routes.

Deterministic: same seed and training procedure as Learn and
Build, fixed document seeds, no timestamps and no timing fields.
Writes results_deep.json beside this file. Run twice and compare
hashes to verify determinism.
"""
import json
import os
import sys

import numpy as np  # noqa: F401  (kept for parity with runners)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "level-1-learn"))
sys.path.insert(0, os.path.join(_HERE, "..", "level-2-build"))
import clip_numpy as clip  # noqa: E402
import deep_eval as deep  # noqa: E402
import qa_service as qa  # noqa: E402


def _slot():
    return {"cases": 0, "vision_em": 0.0, "vision_f1": 0.0,
            "text_em": 0.0, "text_f1": 0.0}


def _finish(slot):
    n = slot["cases"]
    return {"cases": n,
            "vision_exact_match": round(slot["vision_em"] / n, 6),
            "vision_mean_token_f1": round(slot["vision_f1"] / n, 6),
            "text_exact_match": round(slot["text_em"] / n, 6),
            "text_mean_token_f1": round(slot["text_f1"] / n, 6)}


def main():
    images, tokens, _ = clip.make_pairs(100, 0)
    params = clip.init_params()
    params, first_loss, last_loss = clip.train(params, images, tokens)
    docs = deep.make_documents()
    cases = deep.make_cases(docs)
    per_case = []
    by_type = {}
    vision_by_items = {}
    overall = _slot()
    for case in cases:
        doc = case["doc"]
        v_ans = deep.vision_answer(params, doc, case["question_type"])
        t_ans = deep.text_answer(doc["text"], case["question_type"])
        gold = case["labelled_answer"]
        v_em = qa.exact_match(v_ans, gold)
        v_f1 = qa.token_f1(v_ans, gold)
        t_em = qa.exact_match(t_ans, gold)
        t_f1 = qa.token_f1(t_ans, gold)
        per_case.append({
            "case_id": case["case_id"],
            "doc_id": case["doc_id"],
            "item_count": case["item_count"],
            "question_type": case["question_type"],
            "labelled_answer": gold,
            "vision_answer": v_ans,
            "text_answer": t_ans,
            "vision_exact_match": round(v_em, 6),
            "vision_token_f1": round(v_f1, 6),
            "text_exact_match": round(t_em, 6),
            "text_token_f1": round(t_f1, 6),
        })
        for bucket in (by_type.setdefault(case["question_type"],
                                          _slot()), overall):
            bucket["cases"] += 1
            bucket["vision_em"] += v_em
            bucket["vision_f1"] += v_f1
            bucket["text_em"] += t_em
            bucket["text_f1"] += t_f1
        group = "single" if case["item_count"] == 1 else "multi"
        key = (case["question_type"], group)
        slot = vision_by_items.setdefault(
            key, {"cases": 0, "em": 0.0, "f1": 0.0})
        slot["cases"] += 1
        slot["em"] += v_em
        slot["f1"] += v_f1
    results = {
        "project": "P22 multimodal-lab",
        "level": "3 Deep",
        "study": "vision route (Build QA service, learn_encoder_"
                 "stand_in) against text-only parser route on the "
                 "same synthetic document questions",
        "data": "synthetic document set (24 documents, programmatic, "
                "fixed seeds, labelled synthetic)",
        "data_is_public": False,
        "vlm_used": False,
        "vlm_substitution": "learn_encoder_stand_in: no VLM was "
                            "downloaded or run; the Build QA service "
                            "is the vision route; see DEEP-SPEC.md",
        "text_route": "deterministic parser over the document text "
                      "layer (no OCR, no learned language model)",
        "train_pairs": int(images.shape[0]),
        "training_steps": 2500,
        "train_loss_first": round(float(first_loss), 6),
        "train_loss_last": round(float(last_loss), 6),
        "documents": len(docs),
        "item_count_distribution": {
            str(n): sum(1 for d in docs if d["item_count"] == n)
            for n in (1, 2, 3, 4)},
        "eval_cases": len(cases),
        "overall": _finish(overall),
        "per_question_type": {
            k: _finish(v) for k, v in sorted(by_type.items())},
        "vision_by_item_group": {
            qtype + "_" + group: {
                "cases": v["cases"],
                "vision_exact_match": round(v["em"] / v["cases"], 6),
                "vision_mean_token_f1": round(v["f1"] / v["cases"], 6)}
            for (qtype, group), v in sorted(vision_by_items.items())},
        "per_case": per_case,
    }
    out_path = os.path.join(_HERE, "results_deep.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("cases", len(cases),
          "vision_em", results["overall"]["vision_exact_match"],
          "text_em", results["overall"]["text_exact_match"])
    for k, v in sorted(results["per_question_type"].items()):
        print(k, "vision", v["vision_exact_match"],
              "text", v["text_exact_match"])
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
