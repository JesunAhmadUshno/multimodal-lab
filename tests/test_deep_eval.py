"""Deep tests for P22: hand-computed oracles from DEEP-SPEC.md."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-3-deep"))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
import numpy as np  # noqa: E402
import clip_numpy as clip  # noqa: E402
import qa_service as qa  # noqa: E402
import deep_eval as deep  # noqa: E402

passed = 0


def check(name, cond):
    global passed
    assert cond, name
    passed += 1
    print("PASS", name)


# Parser oracles on the DEEP-SPEC sample text.
sample = "title alpha figure shapes circle square"
check("parser colour is unknown",
      deep.text_answer(sample, "colour") == "unknown")
check("parser shape is first legend shape",
      deep.text_answer(sample, "shape") == "circle")
check("parser both is unknown circle",
      deep.text_answer(sample, "both") == "unknown circle")
check("parser count counts legend entries",
      deep.text_answer(sample, "count") == "two")
check("parser title reads title token",
      deep.text_answer(sample, "title") == "alpha")
check("parser left shape is first legend shape",
      deep.text_answer(sample, "left_shape") == "circle")
check("parser count with no legend is unknown",
      deep.text_answer("title alpha", "count") == "unknown")
check("parser title missing is unknown",
      deep.text_answer("figure shapes circle", "title") == "unknown")

# Text-route hand oracles over the whole eval set (no training).
docs = deep.make_documents()
cases = deep.make_cases(docs)


def text_em(qtype):
    sel = [c for c in cases if c["question_type"] == qtype]
    return (sum(qa.exact_match(
        deep.text_answer(c["doc"]["text"], qtype),
        c["labelled_answer"]) for c in sel) / len(sel), len(sel))


check("documents are 24", len(docs) == 24)
check("cases are 144", len(cases) == 144)
check("item counts are six each of 1 to 4",
      [sum(1 for d in docs if d["item_count"] == n)
       for n in (1, 2, 3, 4)] == [6, 6, 6, 6])
check("text colour oracle is 0 of 24", text_em("colour") == (0.0, 24))
check("text shape oracle is 24 of 24", text_em("shape") == (1.0, 24))
check("text both oracle is 0 of 24", text_em("both") == (0.0, 24))
check("text count oracle is 24 of 24", text_em("count") == (1.0, 24))
check("text title oracle is 24 of 24", text_em("title") == (1.0, 24))
check("text left shape oracle is 24 of 24",
      text_em("left_shape") == (1.0, 24))
both_f1 = [qa.token_f1(deep.text_answer(c["doc"]["text"], "both"),
                       c["labelled_answer"])
           for c in cases if c["question_type"] == "both"]
check("text both token F1 oracle is 0.5 per case",
      all(abs(v - 0.5) < 1e-12 for v in both_f1) and len(both_f1) == 24)

# Gold label oracles for the first documents, by the spec formula.
check("doc 0 is one red circle titled alpha",
      docs[0]["items"] == ["red circle"] and docs[0]["title"] == "alpha"
      and docs[0]["text"] == "title alpha figure shapes circle")
check("doc 1 has two items, first red square",
      docs[1]["item_count"] == 2
      and docs[1]["items"][0] == "red square")
check("doc 3 count gold is four",
      deep.gold_answer(docs[3], "count") == "four")
check("doc 4 count gold cycles back to one",
      deep.gold_answer(docs[4], "count") == "one")

# Renderer oracles.
one = deep.render_document([("red", "circle")], 0)
rng = np.random.default_rng(deep.DOC_SEED_OFFSET + 0)
check("one-item document equals the Learn renderer",
      np.array_equal(one, clip.render_image("red", "circle", rng)))
two = deep.render_document([("red", "circle"), ("blue", "square")], 1)
check("two-item document paints both halves",
      bool((two[:, :8] != clip.BACKGROUND).any())
      and bool((two[:, 8:] != clip.BACKGROUND).any()))

# Vision route construction oracles (untrained params are enough:
# the count and title branches do not depend on the prediction).
params = clip.init_params()
multi = [d for d in docs if d["item_count"] == 3][0]
check("vision count answer is constant one on a 3-item document",
      deep.vision_answer(params, multi, "count") == "one")
check("vision title answer is the unsupported string",
      deep.vision_answer(params, docs[0], "title")
      == "unsupported question")
vision_count_em = sum(
    qa.exact_match(deep.vision_answer(params, c["doc"], "count"),
                   c["labelled_answer"])
    for c in cases if c["question_type"] == "count") / 24
check("vision count oracle is exactly 0.25 with any params",
      vision_count_em == 0.25)
check("build routing sends left shape question to shape branch",
      qa.question_type(deep.QUESTIONS["left_shape"]) == "shape")
check("build routing sends title question to unsupported",
      qa.question_type(deep.QUESTIONS["title"]) == "unsupported")

print("deep tests: PASS,", passed, "checks")
