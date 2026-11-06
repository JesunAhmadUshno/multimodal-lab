"""P22 Learn runner: train the NumPy CLIP-style dual encoder once,
end to end, on the synthetic teaching set, and write the measured
results to results_learn.json.

Synthetic data only (see LEARN-SPEC.md). The SPEC public-data step is
carried at Learn as this synthetic set; real public data is the Build
step. Deterministic: fixed seeds, full-batch training, no wall-clock
fields, so two runs produce a byte-identical results file.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_numpy as clip  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 2500
LEARNING_RATE = 0.01


def main():
    train_images, train_tokens, train_combos = clip.make_pairs(
        n_per_combo=100, seed_offset=100000)
    held_images, held_tokens, held_combos = clip.make_pairs(
        n_per_combo=10, seed_offset=900000)

    untrained = clip.init_params()
    before = clip.evaluate(untrained, held_images, held_tokens,
                           held_combos)

    params = clip.init_params()
    params, first_loss, last_loss = clip.train(
        params, train_images, train_tokens,
        epochs=EPOCHS, lr=LEARNING_RATE)
    after = clip.evaluate(params, held_images, held_tokens, held_combos)

    def rounded(d):
        return {k: round(float(v), 6) for k, v in d.items()}

    results = {
        "project": "P22 multimodal-lab",
        "level": "1 Learn",
        "data": "synthetic teaching set (coloured shapes, programmatic)",
        "data_is_public": False,
        "train_pairs": int(train_images.shape[0]),
        "held_out_pairs": int(held_images.shape[0]),
        "combinations": int(len(np.unique(train_combos))),
        "vocab_size": int(len(clip.VOCAB)),
        "image_shape": [clip.IMAGE_SIZE, clip.IMAGE_SIZE, 3],
        "embed_dim": int(clip.EMBED_DIM),
        "temperature": clip.TEMPERATURE,
        "training_steps": EPOCHS,
        "batch_pairs_per_step": 12,
        "master_seed": clip.MASTER_SEED,
        "train_loss_first": round(float(first_loss), 6),
        "train_loss_last": round(float(last_loss), 6),
        "before_training": rounded(before),
        "after_training": rounded(after),
        "recall_at_1_gain_image_to_text": round(
            float(after["image_to_text_recall_at_1"]
                  - before["image_to_text_recall_at_1"]), 6),
        "recall_at_1_gain_text_to_image": round(
            float(after["text_to_image_recall_at_1"]
                  - before["text_to_image_recall_at_1"]), 6),
    }
    out = os.path.join(HERE, "results_learn.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    print("wrote", out)


if __name__ == "__main__":
    main()
