"""Recorded Build run: train the Learn encoder, serve 48 QA cases.

Deterministic: same seed and training procedure as Learn, fixed
evaluation seeds, no timestamps. Writes results_build.json beside
this file. Run twice and compare hashes to verify determinism.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "level-1-learn"))
import clip_numpy as clip  # noqa: E402
import qa_service as qa  # noqa: E402


def main():
    images, tokens, _ = clip.make_pairs(100, 0)
    params = clip.init_params()
    params, first_loss, last_loss = clip.train(params, images, tokens)
    cases = qa.make_eval_cases()
    per_case = []
    schema_pass = 0
    by_type = {}
    for case in cases:
        out = qa.answer_question(params, case["image"],
                                 case["question"])
        errors = qa.schema_validate(out)
        if not errors:
            schema_pass += 1
        em = qa.exact_match(out["answer"], case["labelled_answer"])
        f1 = qa.token_f1(out["answer"], case["labelled_answer"])
        per_case.append({
            "case_id": case["case_id"],
            "question_type": case["question_type"],
            "labelled_answer": case["labelled_answer"],
            "predicted_answer": out["answer"],
            "exact_match": round(em, 6),
            "token_f1": round(f1, 6),
            "schema_valid": not errors,
        })
        slot = by_type.setdefault(case["question_type"],
                                  {"cases": 0, "em": 0.0, "f1": 0.0})
        slot["cases"] += 1
        slot["em"] += em
        slot["f1"] += f1
    n = len(cases)
    vt = qa.visual_token_count()
    results = {
        "project": "P22 multimodal-lab",
        "level": "2 Build",
        "data": "synthetic evaluation set (coloured shapes, "
                "programmatic, held-out seeds)",
        "data_is_public": False,
        "vlm_used": False,
        "vlm_substitution": "learn_encoder_stand_in: no VLM was "
                            "downloaded or run; see BUILD-SPEC.md",
        "train_pairs": int(images.shape[0]),
        "training_steps": 2500,
        "train_loss_first": round(float(first_loss), 6),
        "train_loss_last": round(float(last_loss), 6),
        "eval_cases": n,
        "schema_pass": schema_pass,
        "schema_pass_rate": round(schema_pass / n, 6),
        "exact_match": round(
            sum(c["exact_match"] for c in per_case) / n, 6),
        "mean_token_f1": round(
            sum(c["token_f1"] for c in per_case) / n, 6),
        "per_question_type": {
            k: {"cases": v["cases"],
                "exact_match": round(v["em"] / v["cases"], 6),
                "mean_token_f1": round(v["f1"] / v["cases"], 6)}
            for k, v in sorted(by_type.items())},
        "tokenization": {
            "image_patches": vt["patches"],
            "visual_tokens_with_class_token": vt[
                "with_class_token"],
            "patch_size": qa.PATCH_SIZE,
            "caption_text_tokens": 3,
            "audio_frames_measured": 0,
            "question_text_tokens": {
                k: qa.text_token_count(v)
                for k, v in sorted(qa.QUESTIONS.items())},
            "unified_tokens_example_colour_question":
                qa.unified_token_count(qa.QUESTIONS["colour"]),
        },
        "per_case": per_case,
    }
    out_path = os.path.join(_HERE, "results_build.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("cases", n, "exact_match", results["exact_match"],
          "mean_token_f1", results["mean_token_f1"],
          "schema", schema_pass, "of", n)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
