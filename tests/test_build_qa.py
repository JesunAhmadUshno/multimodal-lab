"""Build tests for P22: hand-computed oracles from BUILD-SPEC.md."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
import numpy as np  # noqa: E402
import clip_numpy as clip  # noqa: E402
import qa_service as qa  # noqa: E402

passed = 0


def check(name, cond):
    global passed
    assert cond, name
    passed += 1
    print("PASS", name)


# Token F1 oracles, computed by hand in BUILD-SPEC.md.
check("f1 half overlap is 0.5",
      abs(qa.token_f1("red circle", "red square") - 0.5) < 1e-12)
check("f1 identical is 1.0",
      qa.token_f1("red circle", "red circle") == 1.0)
check("f1 disjoint is 0.0",
      qa.token_f1("red circle", "blue square") == 0.0)
check("exact match normalises case and spaces",
      qa.exact_match("Red  Circle", "red circle") == 1.0)
check("exact match rejects different text",
      qa.exact_match("red circle", "red square") == 0.0)

# Tokenization oracles.
vt = qa.visual_token_count()
check("16x16 at patch 4 gives 16 patches", vt["patches"] == 16)
check("visual tokens with class token is 17",
      vt["with_class_token"] == 17)
check("colour question has 5 text tokens",
      qa.text_token_count("what colour is the shape") == 5)
check("unified example is 22",
      qa.unified_token_count("what colour is the shape") == 22)
check("caption tokens are 3",
      len(clip.caption_tokens("red", "circle")) == 3)

# Schema oracles.
good = {"answer": "red", "colour": "red", "shape": "circle",
        "count": 1, "confidence": 0.5, "source": qa.SOURCE}
check("valid answer passes schema", qa.schema_validate(good) == [])
bad = dict(good)
del bad["count"]
check("missing count fails with named path",
      qa.schema_validate(bad) == ["$.count: missing required field"])
bad2 = dict(good, colour="purple")
check("bad colour enum fails",
      qa.schema_validate(bad2) == ["$.colour: not in enum"])
bad3 = dict(good, confidence=1.5)
check("confidence 1.5 fails range",
      qa.schema_validate(bad3) == ["$.confidence: out of range 0 to 1"])

# Question routing.
check("both checked before single types",
      qa.question_type("what colour and shape is in the image")
      == "both")
check("count routing",
      qa.question_type("how many shapes are in the image") == "count")
check("unsupported routing",
      qa.question_type("who painted this") == "unsupported")

# Eval set shape and labels.
cases = qa.make_eval_cases()
check("eval set has 48 cases", len(cases) == 48)
check("eval labels cover 12 combos",
      len({c["combo"] for c in cases}) == 12)
check("count label is one",
      all(c["labelled_answer"] == "one" for c in cases
          if c["question_type"] == "count"))

# Service end to end on an untrained model: schema always holds,
# and the unsupported path stays inside the schema.
params = clip.init_params()
img = cases[0]["image"]
out = qa.answer_question(params, img, cases[0]["question"])
check("service output passes schema", qa.schema_validate(out) == [])
out2 = qa.answer_question(params, img, "who painted this")
check("unsupported answer passes schema",
      qa.schema_validate(out2) == []
      and out2["answer"] == "unsupported question")

print("build tests: PASS,", passed, "checks")
