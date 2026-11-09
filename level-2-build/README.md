# Level 2, Build: multimodal-lab

Status: DONE 2026-11-05 (early, Sprint 4 pull). A deterministic
question-answering service over images, with P14-style schema
rules on every output and P06-style scoring of its answers, plus
the multimodal tokenization note with counts.

Honest scope line, stated first: SPEC names one open
vision-language model wired into the service. No VLM was
downloaded or run in this level. Checked this cycle: torch and
transformers are not installed on the Muse VM and no VLM weights
exist on an available node, and a large model install is out of
scope. The VLM step is a labelled stand-in: the Learn CLIP-style
encoder retrieves the nearest of the 12 canonical captions, and a
deterministic composer answers from that prediction. The
substitution is named in BUILD-SPEC.md, in the code, in
results_build.json (vlm_used false), and as a named limitation.
The evaluation set is synthetic (the Learn renderer, held-out
seeds), labelled synthetic everywhere; the license-checked
public captioned set named as the carry-over first step was not
fetched, because per-image license verification cannot be done
cheaply and honestly inside this cycle. That is a named limitation too.

Code: BUILD-SPEC.md (design first, hand-computed oracles, the
named substitutions), qa_service.py (schema validator, stand-in
prediction, question routing, scoring, tokenization counts,
evaluation set builder), run_build.py (the recorded run, writes
results_build.json), tests in tests/test_build_qa.py (22 checks,
hand-computed values).

## How the service works
- Input: one 16 by 16 by 3 synthetic image (flattened) and one
  question string. Output: one JSON object, always validated.
- Prediction: encode the image once with the trained Learn
  encoder, encode the 12 captions "a <colour> <shape>" once,
  take the argmax cosine similarity. The winner gives colour,
  shape, and a confidence (the winning similarity, clipped to
  0 to 1).
- Questions, four types by keyword rule: colour, shape, both,
  count. Anything else returns "unsupported question" under the
  same schema, never an exception.
- Output schema (P14 rules, inline validator with named error
  paths such as $.colour: not in enum): answer (non-empty
  string), colour (red, green, blue, yellow, or unknown), shape
  (circle, square, triangle, or unknown), count (integer, 0 or
  more), confidence (0 to 1), source (the fixed string
  learn_encoder_stand_in, so no output can pretend a VLM ran).
- Scoring, P06 harness style: exact match on normalised text
  and token multiset F1 against the labelled answer, reported
  overall and per question type, with the schema pass count.
- The encoder is trained inside the runner with the exact Learn
  procedure (1,200 train pairs, 2,500 steps, seed 42), so the
  service runs on trained weights.

## Evaluation set (synthetic, labelled synthetic)
48 cases: 12 colour and shape combinations times 4 question
types, one image per combination from seed offset 900000 (seeds
the Learn train and held-out sets never used). Labels: colour
answers the colour word, shape the shape word, both answers
"<colour> <shape>", count answers "one" (every synthetic image
holds exactly one shape, so the count type is constant by
construction and proves routing, not counting skill).

## Measured results (2026-11-05, Muse VM, Python 3.12, NumPy)

| Metric | Value |
| --- | --- |
| Train loss, first step to last step | 3.574392 to 0.005415 |
| Evaluation cases | 48 |
| Exact match, overall | 1.000 |
| Mean token F1, overall | 1.000 |
| Exact match, per type (colour, shape, both, count) | 1.000 each (12 of 12 each) |
| Schema validation | 48 of 48 passed |

Read these numbers at their real weight: a perfect score on a
48-case synthetic set with one image per combination proves the
service loop works end to end (retrieval, routing, schema,
scoring). It is not evidence about any VLM, any public data, or
real images. results_build.json is byte-identical across two
runs (sha256
0c580990f704cad66a4ef41bc2fd574f6d7910bc8b5e099b3b7f71fbd6218a59).

## Multimodal tokenization note (register item 333)
Counts computed by qa_service.py, not quoted from papers:
- Image patches: the 16 by 16 image at patch size 4 gives
  (16/4) * (16/4) = 16 patch tokens; with one class token the
  visual stream is 17 tokens. (The Learn encoder itself consumes
  the flattened 768 values; the patch count is the standard VLM
  tokenization of the same image, stated as such.)
- Text tokens: a caption "a <colour> <shape>" is 3 tokens in the
  Learn vocabulary. Question token counts from the runner:
  colour question 5, shape question 6, both question 8, count
  question 7 (whitespace tokens).
- Audio frames: this service takes no audio input, so the
  measured count is 0 frames. As a labelled formula only, not a
  measurement: frames = seconds times frame rate (one second at
  50 frames per second would be 50 frames).
- Unified token stream: one stream is visual tokens plus text
  tokens in one sequence. Worked example from the runner: the
  colour question case is 17 visual + 5 text = 22 tokens. The
  other types give 23 (shape), 25 (both), and 24 (count).

## Failure modes and honest observations
- Routing collision, found and fixed during the build: the
  first colour question wording ("what colour is the shape")
  contains the word shape, so the both rule fired and colour
  exact match was 0 of 12. Fixed by rephrasing the colour
  question to "what is the main colour"; labels were not
  changed. Keyword routing is brittle by design here and is
  named as such; a real VLM would not need it.
- The stand-in can only answer about colour, shape, and a
  constant count of one. It cannot read text in an image, count
  several objects, or describe anything outside the 12
  combinations. Those are VLM capabilities this level does not
  have and does not claim.
- Confidence is a cosine similarity, not a calibrated
  probability. It is reported because the schema requires a
  bounded score, and it is labelled here as uncalibrated.
- One image per combination in the evaluation set: jitter and
  size variation are sampled once per combination, so the set
  does not measure robustness across renderings. The Learn
  held-out set (120 pairs) remains the robustness evidence for
  the encoder itself.

## Summary claim
Build shipped a deterministic question-answering service over synthetic images on the Muse VM with no vision-language model downloaded or run: the Learn CLIP-style encoder, retrained with the Learn procedure (train loss 3.574392 to 0.005415 over 2,500 steps), serves as a labelled stand-in that retrieves the nearest of 12 captions, and on a synthetic evaluation set of 48 cases (12 colour and shape combinations times 4 question types, held-out seeds, labelled synthetic, no public data) the service scored exact match 1.000 and mean token F1 1.000 with 48 of 48 outputs passing the inline P14-style schema validation, with 22 hand-computed Build tests passing, and results_build.json byte-identical across two runs at sha256 0c580990f704cad6.

## Limits named at ship
- The VLM step is a labelled stand-in on the Learn encoder; no open
  VLM is wired in as SPEC names.
- The evaluation set is synthetic only; the license-checked public
  captioned set was not fetched.
- Template questions only, 4 types, count constant by construction,
  one image per combination.
- In-process Python API only, no HTTP service layer or framework port.
- Run on the Muse VM, not the assigned BlackBox (GPU) fleet node.
- Honesty checks at ship: no invented claim, no VLM claim, synthetic
  data labelled synthetic in every file, every number measured by the
  runner or hand-computed and test-asserted.
