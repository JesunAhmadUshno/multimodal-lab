# DEEP-SPEC: multimodal-lab (P22), Level 3 Deep

Written before the Deep code, per the house pipeline. Compressed
Deep per the December Compression Protocol: the single hardest
capability in SPEC.md Deep, built, tested, benchmarked, with one
failure class written up. The rest of the v2 Deep scope is named
in the repo ROADMAP.md as the v1.1 backlog.

## SPEC Deep text (source of truth)

"an eval study: the VLM against a text-only pipeline on the same
document questions from P16 (where vision pays, where it does not,
one table); one failure class written up (counting, small text,
spatial relations)."

## Substitutions, named up front (never hidden)

1. No VLM run. Verified on this VM on 2026-11-05 before coding:
   `import torch` and `import transformers` both fail with
   ModuleNotFoundError (Python 3.12.3, NumPy 1.26.4 present), and
   no model weights are on the VM. Nothing is installed (FREE-ONLY,
   no installs in this cycle). The vision route is therefore the
   Build service exactly as shipped, whose prediction step is the
   labelled `learn_encoder_stand_in`: the Learn CLIP-style encoder
   retrieves the nearest of 12 captions and a deterministic
   composer answers. Every result names this substitution. No
   sentence in this level may claim a VLM ran.
2. Synthetic documents only, labelled synthetic everywhere. No
   public document set and no P16 data is used or claimed. The
   document set is built in the spirit of P16 document
   intelligence: each document has a visual figure and a text
   layer, and the same questions are asked of both.
3. Text-only route is a deterministic parser over the document
   text layer, not an OCR system and not a learned language
   model. Its rules are fixed below before any run.

## The document set (synthetic, deterministic)

24 documents. Document i (0 to 23):

- Item count n cycles [1, 2, 3, 4], six documents per count.
- Item k (0-based) takes colour and shape combination
  (i + k) mod 12 over the Learn 12 combinations in Learn order
  (colours red, green, blue, yellow; shapes circle, square,
  triangle). The first item is item 0.
- Title cycles [alpha, beta, gamma, delta] by i mod 4.
- Figure image: a 16 by 16 by 3 array. For n = 1 the image is
  exactly the Learn renderer output for the single item with
  rng seed DOC_SEED_OFFSET + i. For n = 2, 3, 4 a document
  renderer in this level draws the items at fixed slots with
  radius or half-size 2: n = 2 at centres (3, 7) and (11, 7);
  n = 3 at (2, 7), (7, 7), (12, 7); n = 4 at (3, 3), (11, 3),
  (3, 11), (11, 11). Item order is left to right, then top to
  bottom.
- Text layer, exact form: "title <title> figure shapes <shape
  names of the items in order, space separated>". The text layer
  carries the title and the shape names in order. It carries no
  colour word and no count word. This partial text layer is the
  design: it mirrors a document whose embedded text names its
  figures but whose figure colours and item count live only in
  the visual figure and in counting the legend entries.

## The two routes

Vision route: the Build QA service, unchanged.
`qa_service.answer_question(params, image_flat, question)` with
the encoder trained by the exact Learn and Build procedure
(1,200 train pairs, 2,500 steps, master seed 42). Its answer
field is the prediction. Substitutions carried in from Build
and repeated here: encoder stand-in for a VLM, synthetic data,
and the count branch that returns count 1 and the answer "one"
for every count question, by construction.

Text-only route: a deterministic parser over the text layer.
- colour: the first colour word in the text, else "unknown".
- shape: the first shape word in the text, else "unknown".
- both: "<colour result> <shape result>".
- count: the number of shape words after the token "shapes",
  mapped 1 to "one", 2 to "two", 3 to "three", 4 to "four".
- title: the token after "title", else "unknown".
- left_shape: the first shape word in the text (the legend
  lists items left to right, so the first entry is the
  leftmost item).

## Questions (six types, 144 cases: 24 documents times 6)

| Type | Question text | Gold answer |
| --- | --- | --- |
| colour | what is the main colour | first item colour |
| shape | what shape is in the image | first item shape |
| both | what colour and shape is in the image | first item colour + shape |
| count | how many shapes are in the image | count word for n |
| title | what is the document title | document title |
| left_shape | what shape is on the left side | first item shape |

The first four question texts are the Build QUESTIONS verbatim.
Title routes to "unsupported" in the Build service (no colour,
shape, or how-many token), so the vision answer is
"unsupported question" by construction. Left_shape contains
"shape", so the Build service routes it to its shape branch,
which predicts one overall shape, not a left-specific one.

Scoring: P06-style exact match and token F1 from the Build
`qa_service` module (normalise, exact_match, token_f1), applied
identically to both routes.

## Hand-computed oracles (fixed before the run)

Text route, by construction of the text layer and parser:
- colour exact match 0 of 24 (answer is always "unknown", gold
  is a colour), token F1 0.0 ("unknown" shares no token).
- shape exact match 24 of 24 = 1.0 (first legend shape is the
  first item shape).
- both exact match 0 of 24 (colour slot is "unknown"), token
  F1 exactly 0.5 per case (predicted "unknown <shape>" shares
  exactly the shape token with gold "<colour> <shape>":
  precision 1/2, recall 1/2, F1 0.5).
- count exact match 24 of 24 = 1.0 (legend entry count is n).
- title exact match 24 of 24 = 1.0.
- left_shape exact match 24 of 24 = 1.0.

Vision route, by construction of the Build service:
- count: the answer is "one" for all 24 documents, so exact
  match is exactly 6 of 24 = 0.25 (only the six n = 1
  documents), whatever the encoder predicts.
- title: exact match 0 of 24, token F1 0.0 ("unsupported
  question" shares no token with any title word).
- colour, shape, both, left_shape: no exact value is claimed
  in advance; these are the measured cells. The runner also
  reports them split by single-item documents (n = 1) and
  multi-item documents (n = 2 to 4), because the single-item
  figure is in the encoder training distribution and the
  composite figure is not.

Parser oracles on the sample text "title alpha figure shapes
circle square": colour "unknown", shape "circle", both
"unknown circle", count "two", title "alpha", left_shape
"circle".

Renderer oracles:
- A one-item document image equals the Learn renderer output
  for the same item and seed, element for element.
- A two-item document image differs from the flat background
  value in both its left half and its right half.

## Failure class written up

Counting, in depth, in the level README: the Build count
branch is constant one by construction, so on this set the
vision route gets exactly the n = 1 slice and misses every
multi-item document, while the text route counts legend
entries and gets all 24. The write-up names the mechanism, the
measured numbers, and what a real counting step would need.
Spatial relations (left_shape) is reported as a measured table
cell only; it is not a second write-up. Small text is not
measured in this level and is named in ROADMAP.md v1.1.

## Determinism rules

- Same training seeds and procedure as Learn and Build;
  document seeds fixed; no timestamps, host names, or timing
  fields anywhere in results_deep.json.
- Metrics rounded to 6 decimal places. Two consecutive runs of
  run_deep.py must produce a byte-identical results_deep.json.

## Commands

- Test: python tests/test_deep_eval.py
- Run: python level-3-deep/run_deep.py
- CI: smoke, Learn tests, Build tests, Deep tests, Learn
  runner, Build runner, Deep runner, on Python 3.11 and 3.12.

## Success criteria

- All Deep tests pass, including the hand oracles above.
- The comparison table is filled with measured values only;
  the oracle cells above match the run exactly (any mismatch
  is a bug to fix, not a number to edit).
- results_deep.json is byte-identical across two runs.
- No VLM claim, no public-data claim, em dash count 0.
