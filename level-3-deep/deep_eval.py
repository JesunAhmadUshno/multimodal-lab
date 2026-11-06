"""Deep eval study for P22: vision route against text-only route.

Synthetic documents only, labelled synthetic everywhere. No
vision-language model is used: the vision route is the Build QA
service exactly as shipped, whose prediction step is the labelled
learn_encoder_stand_in (see DEEP-SPEC.md). The text-only route is
a deterministic parser over each document text layer. No VLM run
is claimed.

Dependencies: NumPy only, plus the Learn and Build modules loaded
by path.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "level-1-learn"))
sys.path.insert(0, os.path.join(_HERE, "..", "level-2-build"))
import clip_numpy as clip  # noqa: E402
import qa_service as qa  # noqa: E402

DOC_SEED_OFFSET = 700000
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
TITLES = ["alpha", "beta", "gamma", "delta"]
COUNTS_CYCLE = [1, 2, 3, 4]
QUESTION_TYPES = ("colour", "shape", "both", "count", "title",
                  "left_shape")
QUESTIONS = dict(qa.QUESTIONS)
QUESTIONS["title"] = "what is the document title"
QUESTIONS["left_shape"] = "what shape is on the left side"

SLOTS = {
    2: [(3, 7), (11, 7)],
    3: [(2, 7), (7, 7), (12, 7)],
    4: [(3, 3), (11, 3), (3, 11), (11, 11)],
}
SLOT_SIZE = 2


def combo(idx):
    idx = idx % (len(clip.COLOURS) * len(clip.SHAPES))
    return (clip.COLOURS[idx // len(clip.SHAPES)],
            clip.SHAPES[idx % len(clip.SHAPES)])


def _draw(img, colour, shape, cx, cy, size):
    ys, xs = np.mgrid[0:clip.IMAGE_SIZE, 0:clip.IMAGE_SIZE]
    if shape == "circle":
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= size ** 2
    elif shape == "square":
        mask = (np.abs(xs - cx) <= size) & (np.abs(ys - cy) <= size)
    elif shape == "triangle":
        in_rows = (ys >= cy - size) & (ys <= cy + size)
        half = np.where(in_rows, (ys - (cy - size)) / 2.0, -1.0)
        mask = in_rows & (np.abs(xs - cx) <= half)
    else:
        raise ValueError("unknown shape: " + shape)
    img[mask] = np.array(clip.COLOUR_RGB[colour], dtype=np.float64)
    return img


def render_document(items, doc_index):
    """Render one synthetic document figure (16 by 16 by 3)."""
    if len(items) == 1:
        colour, shape = items[0]
        rng = np.random.default_rng(DOC_SEED_OFFSET + doc_index)
        return clip.render_image(colour, shape, rng)
    img = np.full((clip.IMAGE_SIZE, clip.IMAGE_SIZE, 3),
                  clip.BACKGROUND, dtype=np.float64)
    for (colour, shape), (cx, cy) in zip(items, SLOTS[len(items)]):
        img = _draw(img, colour, shape, cx, cy, SLOT_SIZE)
    return img


def text_layer(title, items):
    shapes = " ".join(shape for _, shape in items)
    return "title " + title + " figure shapes " + shapes


def make_documents():
    """24 synthetic documents, deterministic, per DEEP-SPEC.md."""
    docs = []
    for i in range(24):
        n = COUNTS_CYCLE[i % len(COUNTS_CYCLE)]
        items = [combo(i + k) for k in range(n)]
        title = TITLES[i % len(TITLES)]
        docs.append({
            "doc_id": "d%02d" % i,
            "item_count": n,
            "items": [c + " " + s for c, s in items],
            "first_colour": items[0][0],
            "first_shape": items[0][1],
            "title": title,
            "text": text_layer(title, items),
            "image": render_document(items, i).reshape(-1),
        })
    return docs


def gold_answer(doc, qtype):
    if qtype == "colour":
        return doc["first_colour"]
    if qtype in ("shape", "left_shape"):
        return doc["first_shape"]
    if qtype == "both":
        return doc["first_colour"] + " " + doc["first_shape"]
    if qtype == "count":
        return NUM_WORDS[doc["item_count"]]
    if qtype == "title":
        return doc["title"]
    raise ValueError("unknown question type: " + qtype)


def make_cases(docs):
    cases = []
    for doc in docs:
        for qtype in QUESTION_TYPES:
            cases.append({
                "case_id": doc["doc_id"] + "-" + qtype,
                "doc_id": doc["doc_id"],
                "item_count": doc["item_count"],
                "question_type": qtype,
                "question": QUESTIONS[qtype],
                "labelled_answer": gold_answer(doc, qtype),
                "doc": doc,
            })
    return cases


# ---- Text-only route: deterministic parser over the text layer ----

def _first_word(tokens, vocab):
    for tok in tokens:
        if tok in vocab:
            return tok
    return "unknown"


def text_answer(text, qtype):
    tokens = text.lower().split()
    if qtype == "colour":
        return _first_word(tokens, clip.COLOURS)
    if qtype in ("shape", "left_shape"):
        return _first_word(tokens, clip.SHAPES)
    if qtype == "both":
        return (_first_word(tokens, clip.COLOURS) + " "
                + _first_word(tokens, clip.SHAPES))
    if qtype == "count":
        if "shapes" not in tokens:
            return "unknown"
        legend = tokens[tokens.index("shapes") + 1:]
        n = sum(1 for tok in legend if tok in clip.SHAPES)
        return NUM_WORDS.get(n, "unknown")
    if qtype == "title":
        if "title" in tokens:
            idx = tokens.index("title")
            if idx + 1 < len(tokens):
                return tokens[idx + 1]
        return "unknown"
    raise ValueError("unknown question type: " + qtype)


# ---- Vision route: the Build QA service, unchanged ----

def vision_answer(params, doc, qtype):
    out = qa.answer_question(params, doc["image"], QUESTIONS[qtype])
    return out["answer"]
