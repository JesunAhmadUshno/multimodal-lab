# Build Spec: multimodal-lab Level 2

## Objective
Close Build per SPEC.md: a question-answering service over images
with P14 schema rules on its outputs, P06 harness style scoring of
its answers, and a multimodal tokenization note with counts
(register item 333).

## Named substitution, stated up front
SPEC says: wire one open vision-language model into the service.
Checked in this cycle (2026-11-05, Muse VM): torch is not
installed, transformers is not installed, and no VLM weights are
present on any available node. Installing a large model is out of
scope for this cycle (FREE-ONLY, no large installs). The service
is therefore built on the Learn encoder with a labelled stand-in
for the VLM step: image embedding retrieval over the 12 caption
embeddings predicts the colour and shape combination, and a
deterministic answer composer turns that prediction into the
answer text. No VLM was downloaded or run, and no VLM measurement
is claimed anywhere in this level. The substitution is a grade
deduction (minus 700), not a silent change. Swapping in a real
VLM later needs only a new prediction step behind the same schema
and scoring code.

## Carry-over first step: public captioned set
The Learn README names a license-checked public captioned set as
the first Build step. No such set was fetched in this cycle:
per-image license verification cannot be done cheaply and
honestly inside this cycle, and an unverified download would
break the house rule. The evaluation set is therefore synthetic,
derived from the Learn renderer with held-out seeds, and labelled
synthetic in every file. This is a grade deduction (minus 500).

## Service design (stdlib plus NumPy only)
- qa_service.py imports the Learn encoder (clip_numpy) by path.
- Prediction: encode the image once, encode the 12 canonical
  captions ("a <colour> <shape>") once, take the argmax cosine
  similarity. The winning caption gives colour and shape.
- Questions: four supported types, selected by keyword.
  colour ("what colour" or "what color"), shape ("what shape"),
  both ("what colour and shape" checked before the single types),
  count ("how many"). Any other question returns the unsupported
  answer under the same schema, never an exception.
  Routing note, found during the build: a colour question that
  also contains the word shape ("what colour is the shape")
  routes to both under this rule, so the service colour question
  is phrased "what is the main colour" (5 tokens, no shape word).
  The first recorded run exposed this (colour exact match 0 of
  12 with every colour answer returning both words); the question
  wording was fixed, not the labels, and the run was repeated.
- Every answer is a JSON object validated by an inline P14-style
  schema validator (schema_validate in qa_service.py): required
  fields, types, enums, and ranges, with named error paths in the
  style of P14 Level 1 (for example $.colour: not in enum).
  An answer that fails validation is a service failure and is
  counted as such by the runner; it is never shipped silently.
- Answer schema: answer (string), colour (one of red, green,
  blue, yellow, unknown), shape (one of circle, square, triangle,
  unknown), count (integer, 0 or more), confidence (number, 0 to
  1, the winning cosine similarity clipped to that range),
  source (the fixed string learn_encoder_stand_in).

## Scoring, P06 harness style
- Exact match: normalised answer text equals the labelled answer.
- Token F1: token multiset F1 between predicted and labelled
  answer text, the same shape as the P06 lexical scoring.
- The runner reports exact match and mean token F1 overall and
  per question type, plus the schema pass count. Every output is
  validated; the schema pass rate is part of the result.

## Evaluation set (synthetic, labelled synthetic)
- 48 cases: 12 colour and shape combinations times 4 question
  types (colour, shape, both, count). Images come from the Learn
  renderer at seed offset 900000, seeds the Learn train and
  held-out sets never used (Learn used offsets 0 and 5000 ranges;
  this offset is disjoint by construction).
- Labelled answers: colour question answers the colour word,
  shape answers the shape word, both answers "<colour> <shape>",
  count answers "one" (every synthetic image holds one shape).
- The encoder is trained inside the runner with the exact Learn
  procedure (same seed, same 2,500 steps, same training pairs),
  so the service uses trained weights, not a fresh random model.

## Hand-computed oracles, worked before coding
- Token F1: prediction "red circle" against label "red square"
  shares 1 token (red); precision 1/2, recall 1/2, F1 = 0.5.
  Identical strings give 1.0; disjoint strings give 0.0.
- Exact match after normalisation: "Red  Circle" matches
  "red circle" (lowercase, whitespace collapsed).
- Patch count: a 16 by 16 image with 4 by 4 patches gives
  (16/4) * (16/4) = 16 patches; plus one class token = 17 visual
  tokens.
- Unified stream example: the question "what colour is the
  shape" has 5 whitespace tokens; unified length for that case is
  17 visual + 5 text = 22 tokens.
- Schema oracle: an answer missing the count field fails with
  $.count: missing required field; colour "purple" fails with
  $.colour: not in enum; confidence 1.5 fails the range check.
- Count answer: every evaluation image holds exactly one shape,
  so the count prediction is 1 and its text is "one".

## Multimodal tokenization note (counts from this code)
Written in the level README from values the runner computes:
image patches (16 patches, 17 with the class token, patch size
4), text tokens (caption 3, per-question counts), audio frames
(no audio input exists in this service: 0 frames measured; the
note gives the standard framing formula as a labelled formula,
not a measurement), and one unified stream example per case
type with its exact total.

## Determinism rules
- Same master seed and training procedure as Learn; evaluation
  seeds fixed; no timestamps or host names in results_build.json.
- Metrics rounded to 6 decimal places. Two consecutive runs of
  run_build.py must produce a byte-identical results_build.json.

## Commands
- Test: python tests/test_build_qa.py
- Run: python level-2-build/run_build.py
- CI: smoke, Learn tests, Build tests, Learn runner, Build
  runner, on Python 3.11 and 3.12.

## Success criteria
- All Build tests pass, including the hand oracles above.
- Every one of the 48 service outputs passes schema validation.
- results_build.json is byte-identical across two runs.
- No VLM claim, no public-data claim, em dash count 0.
