"""Seed sensitivity and Learn confusion analysis (post-v1.0 upgrade).

Reruns the recorded procedures with five encoder seeds instead of one:
- Learn: run_learn.py data and training (train seed offset 100000,
  held-out offset 900000), held-out recall at 1 and 5.
- Deep: run_deep.py training (train offset 0) and the 144-case study.
  Only the vision route depends on the seed; the text route is a
  deterministic parser and is reported once.
It also dumps the held-out image-to-text confusion matrix for the
recorded seed (42), to explain image-to-text Recall at 1 of 0.983.

Synthetic data only, CPU, NumPy, about 40 s. Deterministic: no
timestamps, so two runs write a byte-identical results_seeds.json.
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
import qa_service as qa  # noqa: E402

SEEDS = [42, 1, 2, 3, 4]
COMBOS = [c + " " + s for c in clip.COLOURS for s in clip.SHAPES]


def learn_run(seed, data):
    tr_i, tr_t, held_i, held_t, held_c = data
    params, first, last = clip.train(clip.init_params(seed), tr_i, tr_t)
    metrics = clip.evaluate(params, held_i, held_t, held_c)
    return params, first, last, metrics


def deep_run(seed, train, cases):
    params, _, _ = clip.train(clip.init_params(seed), *train)
    qa._CAPTION_CACHE.clear()
    em = {}
    for case in cases:
        ans = deep.vision_answer(params, case["doc"], case["question_type"])
        em.setdefault(case["question_type"], []).append(
            qa.exact_match(ans, case["labelled_answer"]))
    out = {k: round(float(np.mean(v)), 6) for k, v in sorted(em.items())}
    out["overall"] = round(float(np.mean([x for v in em.values() for x in v])), 6)
    return out


def summary(values):
    a = np.array(values, dtype=np.float64)
    return {"mean": round(float(a.mean()), 6), "sd": round(float(a.std(ddof=1)), 6),
            "min": round(float(a.min()), 6), "max": round(float(a.max()), 6)}


def main():
    tr_i, tr_t, _ = clip.make_pairs(100, 100000)
    held_i, held_t, held_c = clip.make_pairs(10, 900000)
    learn_data = (tr_i, tr_t, held_i, held_t, held_c)
    deep_train = clip.make_pairs(100, 0)[:2]
    cases = deep.make_cases(deep.make_documents())

    learn, vision, confusion = [], [], None
    for seed in SEEDS:
        params, first, last, m = learn_run(seed, learn_data)
        learn.append({"seed": seed, "train_loss_first": round(float(first), 6),
                      "train_loss_last": round(float(last), 6),
                      **{k: round(float(v), 6) for k, v in m.items()}})
        if seed == 42:
            sims = clip.encode_image(params, held_i) @ clip.encode_text(params, held_t).T
            pred = held_c[np.argmax(sims, axis=1)]
            mat = np.zeros((12, 12), dtype=int)
            for g, p in zip(held_c, pred):
                mat[g, p] += 1
            confusion = {
                "seed": 42,
                "labels": COMBOS,
                "matrix_rows_gold_cols_predicted": mat.tolist(),
                "errors": [{"gold": COMBOS[g], "predicted": COMBOS[p], "count": int(mat[g, p])}
                           for g in range(12) for p in range(12) if g != p and mat[g, p]],
                "note": "Held-out captions repeat 10 times per combination and identical "
                        "captions get identical embeddings, so an image's top 5 captions are "
                        "always 5 copies of one caption: image-to-text Recall at 5 equals "
                        "Recall at 1 by construction.",
            }
        vision.append({"seed": seed, **deep_run(seed, deep_train, cases)})
        print("seed", seed, "learn i2t@1", learn[-1]["image_to_text_recall_at_1"],
              "deep vision overall", vision[-1]["overall"])

    keys = ["image_to_text_recall_at_1", "image_to_text_recall_at_5",
            "text_to_image_recall_at_1", "text_to_image_recall_at_5", "train_loss_last"]
    vkeys = list(deep.QUESTION_TYPES) + ["overall"]
    results = {
        "project": "P22 multimodal-lab",
        "analysis": "seed sensitivity (encoder init seed only) and Learn confusion",
        "data_is_public": False,
        "vlm_used": False,
        "seeds": SEEDS,
        "learn_per_seed": learn,
        "learn_summary": {k: summary([r[k] for r in learn]) for k in keys},
        "deep_vision_em_per_seed": vision,
        "deep_vision_em_summary": {k: summary([r[k] for r in vision]) for k in sorted(vkeys)},
        "deep_text_route": "deterministic parser, seed-independent: see results_deep.json",
        "learn_confusion_seed_42": confusion,
    }
    out = os.path.join(HERE, "results_seeds.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
