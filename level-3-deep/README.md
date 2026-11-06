# Level 3, Deep: multimodal-lab

Status: DONE 2026-11-05 (early, Sprint 4 pull). An eval study on
synthetic documents (labelled synthetic): the Build QA service
as the vision route against a deterministic text-only parser on
the same document questions, one comparison table, and one
failure class (counting) written up in depth.

Summary claim: Deep ran a synthetic document eval study (labelled synthetic, 24 documents, 144 cases, Muse VM CPU, Python 3.12, NumPy) comparing the Build QA service (learn_encoder_stand_in, no VLM downloaded or run) against a deterministic text-only parser on the same questions: vision exact match 0.750 on colour and 0.375 on both against text 0.000 on both, text exact match 1.000 on shape, count, title and left_shape against vision 0.375, 0.250, 0.000 and 0.375, overall vision exact match 0.354167 and text exact match 0.666667, with 29 hand-computed Deep tests passing and results_deep.json byte-identical across two runs at sha256 4678206d4585330a.

Code: DEEP-SPEC.md (design, substitutions, hand-computed
oracles fixed before the run), deep_eval.py (document set,
renderer, text-only parser, vision route), run_deep.py (the
recorded run, writes results_deep.json), tests in
tests/test_deep_eval.py (29 tests, hand-computed values).

## Substitutions, named again (from Build, carried into Deep)

1. No VLM was downloaded or run at any level of this project.
   Verified on this VM on 2026-11-05: `import torch` and
   `import transformers` both fail with ModuleNotFoundError,
   and no model weights are present. Nothing was installed.
   The vision route is the Build service, whose prediction
   step is the labelled `learn_encoder_stand_in`: the Learn
   CLIP-style encoder (retrained here with the Learn procedure,
   train loss 3.574392 to 0.005415 over 2,500 steps) retrieves
   the nearest of 12 captions and a deterministic composer
   answers. This study measures that stand-in against a text
   pipeline. It does not measure any real VLM, and no result
   here transfers to one.
2. Synthetic documents only. No public document set and no
   P16 data was used. Documents follow the P16 shape in
   miniature: a visual figure plus a text layer, with the same
   questions asked of both routes.
3. The text-only route is a hand-written deterministic parser
   over the text layer, not OCR and not a learned model.

## The document set (synthetic)

24 documents, six each with 1, 2, 3, and 4 figure items. Each
document has a 16 by 16 figure image (one-item figures are the
Learn renderer output; multi-item figures are drawn at fixed
slots by the renderer in deep_eval.py) and a text layer of the
exact form "title <title> figure shapes <shape names in
order>". The text layer carries the title and the shape names
in left-to-right order. It carries no colour word and no count
word, so colour lives only in the figure, and count lives in
the figure and in the number of legend entries. 144 cases:
24 documents times 6 question types (colour, shape, both,
count, title, left_shape). Full construction rules and the
oracles fixed before the run are in DEEP-SPEC.md.

## Benchmark table (measured 2026-11-05, Muse VM CPU)

Exact match (EM) and mean token F1, per question type, 24
cases per type. Full record: results_deep.json.

| Question type | Vision EM | Vision F1 | Text EM | Text F1 | Where the win is |
| --- | --- | --- | --- | --- | --- |
| colour | 0.750 | 0.750 | 0.000 | 0.000 | vision pays: colour is only in the figure |
| shape | 0.375 | 0.375 | 1.000 | 1.000 | text pays: the legend names the shapes |
| both | 0.375 | 0.5625 | 0.000 | 0.500 | vision pays: text has no colour to complete the pair |
| count | 0.250 | 0.250 | 1.000 | 1.000 | text pays: vision count is constant one (failure class below) |
| title | 0.000 | 0.000 | 1.000 | 1.000 | text pays: the vision service cannot read document text |
| left_shape | 0.375 | 0.375 | 1.000 | 1.000 | text pays: legend order encodes position, the vision route has no position step |
| overall (144) | 0.354167 | 0.385417 | 0.666667 | 0.750 | text route wins this set, because the text layer covers 4 of 6 types |

Single-item against multi-item documents (vision route, EM):
colour 1.000 single, 0.667 multi; shape 0.833 single, 0.222
multi; both 0.833 single, 0.222 multi; left_shape 0.833
single, 0.222 multi; count 1.000 single, 0.000 multi; title
0.000 in both groups. Reading: the encoder stand-in reads
colour off composite figures fairly well (colour fills many
pixels), but its shape reading collapses once several shapes
share one small figure, because the Build prediction step
returns exactly one caption for the whole image.

## Where vision pays, where it does not (the study answer)

- Vision pays exactly where the answer exists only in the
  pixels: colour (0.750 against 0.000) and the combined
  colour-and-shape question (0.375 against 0.000), where the
  text route cannot complete the pair at all.
- Vision does not pay where the document text layer already
  carries the answer: shape names, the title, legend order
  (left_shape), and counting legend entries all score 1.000
  for the text route, against 0.375, 0.000, 0.375, and 0.250
  for the vision route.
- The overall text win (0.667 against 0.354) is a property of
  this text layer, which was designed to cover four of six
  question types. The table, not the overall number, is the
  result: route document questions to the text layer first,
  and spend vision only on what the text does not carry.

## Failure modes

### The failure class in depth: counting

- Mechanism, by construction: the Build QA service answers
  every count question from a constant. In qa_service.py the
  count branch sets count = 1 and the answer text "one",
  whatever the image holds, because every Build evaluation
  image held exactly one shape. The prediction step (nearest
  caption) has no counting operation in it at all: one image
  embedding, one winning caption, no enumeration.
- Measured here: on the 24 count cases the vision route
  scores exact match 0.250, which is exactly the six
  single-item documents (oracle fixed in DEEP-SPEC.md before
  the run, matched exactly). Split by group: 1.000 on
  single-item documents, 0.000 on the 18 multi-item
  documents. It does not undercount or overcount; it cannot
  count, and the 0.250 is the share of the set where the
  constant happens to be right.
- The text route scores 1.000 on the same cases, not because
  it sees better, but because counting legend entries is a
  counting operation and nearest-caption retrieval is not.
- What a real fix needs: a step that enumerates instances
  before answering (detection, tiling with per-tile reads, or
  a VLM whose counting behaviour is itself evaluated per
  count value, since VLMs are known to be weak counters).
  Any of these is new scope; it is named in ROADMAP.md as a
  v1.1 item, not claimed here.
- Honest limit: this failure is a property of the stand-in
  service, proven by construction and confirmed by the
  numbers. It is evidence about this pipeline, and a warning
  to eval any real VLM per count value. It is not a measured
  claim about any real VLM.

### Other observed failure modes

- Composite figures break shape reading: shape EM falls from
  0.833 on single-item documents to 0.222 on multi-item
  documents, while colour EM falls only from 1.000 to 0.667.
  One global embedding mixes the shapes; colour survives
  mixing better than shape does.
- The vision route has no document-text reading at all:
  title scores 0.000, with the service returning its
  unsupported-question answer by construction.
- Spatial relations are a measured table cell only
  (left_shape: vision 0.375, text 1.000, because legend order
  carries position and the vision route predicts one overall
  shape). No deeper spatial claim is made. Small text was
  not measured in this level; it is a ROADMAP.md v1.1 item.

## Determinism

results_deep.json contains no timestamps and no timing
fields. Verified byte-identical across repeated runs of
run_deep.py in this cycle (sha256
4678206d4585330acf56402676212ecac4f0cb5c51b43dade563a0b7125955b2).
The full test suite in this repo: smoke test, 8 Learn tests,
22 Build tests, 29 Deep tests, all passing locally on
2026-11-05.

## Grade

8,000 of 10,000. Deductions: no real VLM, the vision route is
the labelled encoder stand-in (-900); synthetic documents
only, no public or P16 document data (-400); the text route
is a hand parser, not OCR or a learned pipeline (-300); small
scale, 24 documents and one training seed (-200); small text
not measured and spatial relations reported as a table cell
only, not a second study (-200). Inside the compressed Deep
band of 7,500 to 8,700 set by the December Compression
Protocol.
