# Paper candidate brief: vision against text-only routing on synthetic documents, and a counting failure class

Project: P22 multimodal-lab
Status: CANDIDATE, brief dated 2026-11-05
Submission note: submission is Jesun's tap. Nothing is submitted by
this brief, and no venue has been contacted.

## Working title

Vision pays only where the text layer is silent: a synthetic
document eval of an encoder stand-in against a text-only parser,
with a counting failure class

## Candidate claim, in one line

On synthetic documents, a nearest-caption encoder stand-in beats
a text-only parser exactly where the answer lives only in the
pixels (colour), and loses wherever the text layer carries the
answer, with counting failing by construction; the claim is the
routing lesson and the failure analysis on this stand-in, not
any result about a real VLM, because no VLM was downloaded or
run and every dataset is synthetic only.

## Abstract draft

Based only on the measured v1.0 results recorded in the repo.
Scope limits are stated inside the abstract, not after it.

Document question answering is often routed to a vision model
by default. This candidate reports a three level open study
that measures the routing question on synthetic data. Learn
trained a NumPy CLIP-style dual encoder on 1,200 synthetic
train pairs and 120 held-out pairs: loss 3.124841 to 0.005265,
held-out Recall at 1 0.9833 image-to-text and 1.0 text-to-image
after training. Build wrapped that encoder, as a labelled
stand-in with no VLM downloaded or run, in a schema-validated
QA service scoring exact match 1.0 on 48 synthetic cases. Deep
compared that vision route against a deterministic text-only
parser on 144 synthetic document cases (24 documents, 6 types):
vision exact match 0.354 and F1 0.385 against text 0.667 and
0.750 overall; vision 0.750 against text 0.000 on colour and
0.375 against 0.000 on both; text 1.000 against vision 0.375,
0.250, 0.000, and 0.375 on shape, count, title, and left_shape.
The counting failure class is analysed in depth: the vision
count is constant one by construction, scoring 0.250 overall
(exactly the single-item documents) and 0.000 on multi-item
documents. Every expected value was fixed in the level specs
before the recorded runs and asserted by 59 checks. No VLM
was downloaded or run, no public data was used, and no result
transfers to a real VLM. The candidate claims the routing
table and the per-count-value eval warning, on this synthetic
stand-in study only.

## Method outline

1. Learn encoder. Flattened image encoder and mean-pooled text
   encoder into a shared 32 unit space, two-direction InfoNCE
   at temperature 0.1, hand-written gradients checked against
   finite differences, Adam over 2,500 steps, seed 42.
2. Build service. Nearest-caption prediction over 12 canonical
   captions, keyword question routing, inline schema validation
   with named error paths, exact match and token F1 scoring.
3. Deep study. 24 synthetic documents with a figure and a text
   layer, 6 question types, both routes on the same 144 cases,
   per-type and single against multi-item splits.
4. Counting analysis. Mechanism proven from the code (constant
   count branch, no enumeration step), then confirmed by the
   0.250 and 0.000 measurements against the pre-stated oracle.
5. Determinism. All three results files are byte-identical
   across runs; no timing fields.

## Results table

Synthetic, 2026-11-05, Muse VM CPU, NumPy, from the level
READMEs and results files in the repo.

| Item | Result |
| --- | --- |
| Learn | Loss 3.124841 to 0.005265; Recall at 1 0.9833 and 1.0 after training |
| Build | 48 cases, exact match 1.0, F1 1.0, schema 48 of 48 |
| Deep overall | Vision EM 0.354 F1 0.385; text EM 0.667 F1 0.750, 144 cases |
| Deep colour | Vision 0.750 against text 0.000 |
| Deep count | Vision 0.250 overall, 0.000 multi-item; text 1.000 |
| Tests | 59 checks in total, plus the scaffold smoke test |

## Limits a reviewer will raise, stated first

- No real VLM: the vision route is an encoder stand-in.
- Synthetic documents only: no public or real document data.
- The text route is a hand parser, not OCR or a learned model.
- Small scale: 24 documents, one seed, 16 by 16 figures.
- Small text was not measured; spatial relations are one
  table cell (left_shape), not a second study.

## What would strengthen the candidate

Each item is v1.1 backlog in ROADMAP.md, not done work: a real
VLM route on the same questions, real license-checked public
documents, a counting fix with per-count-value scoring, small
text measurement, and several training seeds with spread.

## Venue note

No venue is named as submitted or contacted. Venue choice and
any submission click are Jesun's tap only.
