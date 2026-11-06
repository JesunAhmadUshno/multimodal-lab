"""P22 Learn tests: hand-computed values for the NumPy CLIP-style
dual encoder. Every expected number below was computed by hand from
the definitions in LEARN-SPEC.md before the run, not copied from the
model output.
Run: python tests/test_learn_multimodal.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "level-1-learn"))

import clip_numpy as clip  # noqa: E402


def test_loss_oracle_identity_temperature_1():
    eye = np.eye(2)
    loss = clip.contrastive_loss(eye, eye, temperature=1.0)
    # softmax diag exp(1)/(exp(1)+exp(0)) = 0.7310585786
    # loss = -log(0.7310585786) = 0.3132616875 in both directions
    assert abs(loss - 0.3132616875182228) < 1e-9


def test_loss_oracle_uniform_embeddings():
    same = np.array([[1.0, 0.0], [1.0, 0.0]])
    loss = clip.contrastive_loss(same, same, temperature=1.0)
    # every logit equal, softmax uniform over 2, loss = log(2)
    assert abs(loss - 0.6931471805599453) < 1e-9


def test_loss_oracle_temperature_half():
    eye = np.eye(2)
    loss = clip.contrastive_loss(eye, eye, temperature=0.5)
    # logits double: -log(exp(2)/(exp(2)+exp(0))) = 0.1269280110
    assert abs(loss - 0.1269280110429725) < 1e-9


def test_embedding_shape_and_unit_norms():
    params = clip.init_params()
    images, tokens, _ = clip.make_pairs(n_per_combo=1, seed_offset=5000)
    img_emb = clip.encode_image(params, images)
    txt_emb = clip.encode_text(params, tokens)
    assert img_emb.shape == (12, clip.EMBED_DIM)
    assert txt_emb.shape == (12, clip.EMBED_DIM)
    assert np.allclose(np.linalg.norm(img_emb, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(txt_emb, axis=1), 1.0, atol=1e-9)


def test_data_is_deterministic_and_labelled_by_combo():
    a = clip.make_pairs(n_per_combo=2, seed_offset=7000)
    b = clip.make_pairs(n_per_combo=2, seed_offset=7000)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])
    assert a[0].shape == (24, 768)
    # first combo is red circle: caption tokens are a, red, circle
    assert list(a[1][0]) == [clip.TOKEN["a"], clip.TOKEN["red"],
                              clip.TOKEN["circle"]]
    assert a[2][0] == 0
    # same combo, different per-sample seed: pixels differ
    assert not np.array_equal(a[0][0], a[0][1])


def test_gradients_match_finite_differences():
    params = clip.init_params()
    images, tokens, _ = clip.make_pairs(n_per_combo=1, seed_offset=3000)
    images, tokens = images[:4], tokens[:4]
    _, grads = clip.loss_and_grads(params, images, tokens)
    rng = np.random.default_rng(7)
    eps = 1e-6
    checked = 0
    for key in ["W1", "W2", "E", "Wt"]:
        flat = params[key].reshape(-1)
        grad_flat = grads[key].reshape(-1)
        for _ in range(6):
            i = int(rng.integers(0, flat.size))
            old = flat[i]
            flat[i] = old + eps
            plus_loss = clip.loss_and_grads(params, images, tokens)[0]
            flat[i] = old - eps
            minus_loss = clip.loss_and_grads(params, images, tokens)[0]
            flat[i] = old
            numeric = (plus_loss - minus_loss) / (2 * eps)
            denom = max(1.0, abs(numeric), abs(grad_flat[i]))
            assert abs(numeric - grad_flat[i]) / denom < 1e-4, (
                key, i, numeric, grad_flat[i])
            checked += 1
    assert checked == 24


def test_retrieval_on_trivially_separable_set():
    # Hand-built embeddings: each pair shares its own axis, so the
    # correct match must rank first in both directions.
    emb = np.eye(4, dtype=np.float64)
    combos = np.array([0, 1, 2, 3])
    assert clip.recall_at_k(emb, emb, combos, 1, "image_to_text") == 1.0
    assert clip.recall_at_k(emb, emb, combos, 1, "text_to_image") == 1.0
    assert clip.recall_at_k(emb, emb, combos, 5, "image_to_text") == 1.0


def test_short_training_lowers_loss_deterministically():
    images, tokens, _ = clip.make_pairs(n_per_combo=2, seed_offset=11000)
    p1 = clip.init_params()
    _, first1, last1 = clip.train(p1, images, tokens, epochs=30, lr=0.01)
    p2 = clip.init_params()
    _, first2, last2 = clip.train(p2, images, tokens, epochs=30, lr=0.01)
    assert first1 == first2
    assert last1 == last2
    assert last1 < first1


if __name__ == "__main__":
    test_loss_oracle_identity_temperature_1()
    test_loss_oracle_uniform_embeddings()
    test_loss_oracle_temperature_half()
    test_embedding_shape_and_unit_norms()
    test_data_is_deterministic_and_labelled_by_combo()
    test_gradients_match_finite_differences()
    test_retrieval_on_trivially_separable_set()
    test_short_training_lowers_loss_deterministically()
    print("P22 Learn tests: PASS (8 tests)")
