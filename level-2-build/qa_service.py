"""Deterministic QA service over synthetic images for P22 Build.

Synthetic data only: images come from the Learn renderer and are
labelled synthetic everywhere. No vision-language model is used.
The VLM step in PROPOSAL.md (former SPEC.md) is replaced by a labelled stand-in: the
Learn CLIP-style encoder retrieves the nearest of the 12 canonical
captions, and a deterministic composer builds the answer. The
substitution and its limits are named in BUILD-SPEC.md
and the level README. No VLM run is claimed.

Dependencies: NumPy only, plus the Learn module loaded by path.
"""
import os
import sys
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "level-1-learn"))
import clip_numpy as clip  # noqa: E402

SOURCE = "learn_encoder_stand_in"
COLOUR_ENUM = clip.COLOURS + ["unknown"]
SHAPE_ENUM = clip.SHAPES + ["unknown"]
PATCH_SIZE = 4


# ---- P14-style schema validation (inline subset, named errors) ----

def schema_validate(obj):
    """Return a list of error strings; empty means valid.

    Rules mirror the P14 Level 1 subset: required fields, types,
    enums, and numeric ranges, each failure named by its path.
    """
    errors = []
    if not isinstance(obj, dict):
        return ["$: not an object"]
    for field in ("answer", "colour", "shape", "count",
                  "confidence", "source"):
        if field not in obj:
            errors.append("$." + field + ": missing required field")
    if errors:
        return errors
    if not isinstance(obj["answer"], str) or not obj["answer"]:
        errors.append("$.answer: must be a non-empty string")
    if obj["colour"] not in COLOUR_ENUM:
        errors.append("$.colour: not in enum")
    if obj["shape"] not in SHAPE_ENUM:
        errors.append("$.shape: not in enum")
    if not isinstance(obj["count"], int) or isinstance(
            obj["count"], bool):
        errors.append("$.count: must be an integer")
    elif obj["count"] < 0:
        errors.append("$.count: below minimum 0")
    conf = obj["confidence"]
    if not isinstance(conf, (int, float)) or isinstance(conf, bool):
        errors.append("$.confidence: must be a number")
    elif conf < 0.0 or conf > 1.0:
        errors.append("$.confidence: out of range 0 to 1")
    if obj["source"] != SOURCE:
        errors.append("$.source: must equal " + SOURCE)
    return errors


# ---- Tokenization counts (register item 333, from this code) ----

def visual_token_count(image_size=clip.IMAGE_SIZE,
                       patch_size=PATCH_SIZE):
    per_side = image_size // patch_size
    patches = per_side * per_side
    return {"patches": patches, "with_class_token": patches + 1}


def text_token_count(text):
    return len(text.split())


def unified_token_count(question):
    return (visual_token_count()["with_class_token"]
            + text_token_count(question))


# ---- The stand-in prediction step ----

_CAPTION_CACHE = {}


def caption_embeddings(params):
    key = id(params)
    if key not in _CAPTION_CACHE:
        toks = np.stack([clip.caption_tokens(c, s)
                         for c in clip.COLOURS for s in clip.SHAPES])
        _CAPTION_CACHE[key] = (toks, clip.encode_text(params, toks))
    return _CAPTION_CACHE[key][1]


def predict_combo(params, image_flat):
    img_emb = clip.encode_image(params, image_flat.reshape(1, -1))
    sims = (img_emb @ caption_embeddings(params).T)[0]
    best = int(np.argmax(sims))
    colour = clip.COLOURS[best // len(clip.SHAPES)]
    shape = clip.SHAPES[best % len(clip.SHAPES)]
    confidence = float(min(1.0, max(0.0, sims[best])))
    return colour, shape, confidence


def question_type(question):
    q = question.lower()
    if "how many" in q:
        return "count"
    if "colour" in q and "shape" in q:
        return "both"
    if "color" in q or "colour" in q:
        return "colour"
    if "shape" in q:
        return "shape"
    return "unsupported"


def answer_question(params, image_flat, question):
    colour, shape, confidence = predict_combo(params, image_flat)
    qtype = question_type(question)
    if qtype == "colour":
        answer, out_colour, out_shape, count = colour, colour, shape, 1
    elif qtype == "shape":
        answer, out_colour, out_shape, count = shape, colour, shape, 1
    elif qtype == "both":
        answer = colour + " " + shape
        out_colour, out_shape, count = colour, shape, 1
    elif qtype == "count":
        answer, out_colour, out_shape, count = "one", colour, shape, 1
    else:
        answer = "unsupported question"
        out_colour, out_shape, count, confidence = (
            "unknown", "unknown", 0, 0.0)
    result = {"answer": answer, "colour": out_colour,
              "shape": out_shape, "count": count,
              "confidence": round(confidence, 6), "source": SOURCE}
    errors = schema_validate(result)
    if errors:
        raise ValueError("schema failure: " + "; ".join(errors))
    return result


# ---- P06-style scoring ----

def normalise(text):
    return " ".join(text.lower().split())


def exact_match(predicted, labelled):
    return 1.0 if normalise(predicted) == normalise(labelled) else 0.0


def token_f1(predicted, labelled):
    pred = Counter(normalise(predicted).split())
    gold = Counter(normalise(labelled).split())
    if not pred or not gold:
        return 0.0
    overlap = sum((pred & gold).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


# ---- Evaluation set (synthetic, held-out seeds) ----

QUESTIONS = {
    "colour": "what is the main colour",
    "shape": "what shape is in the image",
    "both": "what colour and shape is in the image",
    "count": "how many shapes are in the image",
}
EVAL_SEED_OFFSET = 900000


def labelled_answer(qtype, colour, shape):
    if qtype == "colour":
        return colour
    if qtype == "shape":
        return shape
    if qtype == "both":
        return colour + " " + shape
    return "one"


def make_eval_cases():
    """48 synthetic cases: 12 combos times 4 question types."""
    cases = []
    idx = 0
    for ci, colour in enumerate(clip.COLOURS):
        for si, shape in enumerate(clip.SHAPES):
            rng = np.random.default_rng(EVAL_SEED_OFFSET + idx)
            image = clip.render_image(colour, shape, rng).reshape(-1)
            for qtype in ("colour", "shape", "both", "count"):
                cases.append({
                    "case_id": "b%02d-%s" % (idx, qtype),
                    "combo": colour + " " + shape,
                    "question_type": qtype,
                    "question": QUESTIONS[qtype],
                    "labelled_answer": labelled_answer(
                        qtype, colour, shape),
                    "image": image,
                })
            idx += 1
    return cases
