# Level 1, Learn: multimodal-lab

Status: DONE 2026-11-05 (early, Sprint 4 pull). A NumPy CLIP-style
dual encoder trained once, end to end, on a synthetic teaching set,
plus a document inspection of one open VLM. Honest scope line: SPEC
names public data for this level; no dataset was downloaded,
because no captioned image set was license-verified inside this
cycle. The training set is synthetic, built programmatically in
code, and labelled synthetic everywhere it appears. A
license-checked public captioned image set is the named first step
of Build. No VLM was downloaded or run at this level.

Code: LEARN-SPEC.md (the design, with the hand-computed loss
oracle, the pre-stated retrieval bar, and two recorded design
revisions), clip_numpy.py (data, model, hand-written backprop,
Adam, retrieval), run_learn.py (the recorded run, writes
results_learn.json), ANATOMY.md (the VLM anatomy page), tests in
tests/test_learn_multimodal.py (8 tests, hand-computed values).

## The idea in one paragraph
A CLIP-style system learns two encoders at once: one turns an image
into a vector, one turns a caption into a vector, and training
pulls matched pairs together in a shared space while pushing
unmatched pairs apart. The loss that does this is contrastive
(InfoNCE): for each image, the correct caption should win a softmax
over all captions in the batch, and for each caption, the correct
image should win the mirror softmax. Nothing else is taught. If
the space is learned, retrieval is just nearest neighbour search
in it, which is exactly how this level is scored.

## The synthetic teaching set (synthetic, not public, not real)
- Images: 16 by 16 by 3 arrays. One shape (circle, square,
  triangle) in one colour (red, green, blue, yellow) on a dark
  background, with centre jitter and size variation from per-sample
  seeds. 12 colour and shape combinations.
- Captions: the token sequence "a <colour> <shape>". Vocabulary
  size 9 including a pad token.
- Train: 1,200 pairs (100 per combination). Held-out: 120 pairs
  (10 per combination) from seeds training never used.

## Method
- Image encoder: flatten 768, linear to 128 hidden units with
  ReLU, linear to a 32 unit embedding, L2 normalised.
- Text encoder: token embeddings (9 by 24), mean pooling over
  non-pad tokens, linear to the same 32 unit space, L2 normalised.
- Loss: two-direction InfoNCE at fixed temperature 0.1. Gradients
  are hand-written and checked against finite differences in the
  tests (24 sampled entries across 4 parameter tensors, relative
  error below 1e-4).
- Training: Adam, learning rate 0.01, 2,500 steps, one minibatch
  of 12 pairs per step, exactly one pair per combination, cycling
  deterministically. Master seed 42. Run on the Muse VM CPU; this
  tiny teaching train has no GPU step, and no GPU use is claimed.
- Scoring: combination-match Recall at 1 and at 5 on held-out
  pairs (rule fixed in LEARN-SPEC.md before the recorded run),
  measured before training (same seed initialisation) and after.

## Measured results (2026-11-05, Muse VM, Python 3.12.3, NumPy 1.26.4)

| Metric | Before training | After training |
| --- | --- | --- |
| Train loss (minibatch, first step / last step) | 3.124841 | 0.005265 |
| Image-to-text Recall at 1 | 0.050000 | 0.983333 |
| Image-to-text Recall at 5 | 0.050000 | 0.983333 |
| Text-to-image Recall at 1 | 0.000000 | 1.000000 |
| Text-to-image Recall at 5 | 0.416667 | 1.000000 |

The pre-stated pass bar (trained Recall at 1 beats untrained by at
least 0.30 in both directions, and final loss below first loss)
is met: gains are plus 0.933333 image-to-text and plus 1.000000
text-to-image. results_learn.json is byte-identical across two
runs (sha256
dee704a6b07abdee4a7e77d22cfee76cbb6038c18eae4e2941e7d30000bc503d).

## Failure modes and honest observations
Observed in this level's own runs:
- Duplicate captions break naive InfoNCE. The first design trained
  full-batch over 20 identical captions per combination; the loss
  floored near log(20) = 2.9957 and plateaued at 3.000696, because
  the objective asks the model to pick one correct caption among
  identical copies. Fixed by one-pair-per-combination minibatches
  (design revision 1 in LEARN-SPEC.md).
- A flattened encoder is position-sensitive. At 20 train examples
  per combination the train loss fell to 0.005952 while held-out
  image-to-text Recall at 1 stayed at 0.575: the model memorised
  positions instead of learning shape and colour. More coverage of
  the jitter and size space (100 per combination) closed most of
  the gap (design revision 2). Two held-out images of 120 still
  miss at Recall at 1 after training.
- Image-to-text Recall at 5 equals Recall at 1 by construction:
  the 10 held-out captions of a combination are identical token
  sequences, so they share one embedding and form a tied block at
  the top of the ranking. This is a scoring property of repeated
  captions, not extra evidence of quality.
Known failure classes from the anatomy read, labelled as such and
not measured here (this set has one shape per image and no text in
any image): counting, small text reading, and spatial relations
are the classes the LLaVA inspection in ANATOMY.md traces to the
vision encoder and projector joints. They are Deep level scope in
SPEC.md, to be measured there, not claimed here.

## The VLM anatomy page
ANATOMY.md walks LLaVA part by part: the CLIP vision encoder, the
projector (the joint between the two pre-trained models), the
combined token sequence, and the language model, each with its
job, input, output, and one named failure tendency, plus how an
image becomes tokens end to end. It is a document inspection of
the published design. No VLM was downloaded or run, and the page
contains no measurement about LLaVA.

## Summary claim
Learn trained a NumPy CLIP-style dual encoder once, end to end, on the Muse VM on a synthetic teaching set of 1,200 train pairs and 120 held-out pairs over 12 colour and shape combinations with a vocabulary of 9: train loss fell from 3.124841 to 0.005265 over 2,500 steps, held-out image-to-text Recall at 1 rose from 0.050000 to 0.983333 and text-to-image Recall at 1 rose from 0.000000 to 1.000000, with 8 hand-computed Learn tests passing plus the scaffold smoke test, results_learn.json byte-identical across two runs at sha256 dee704a6b07abdee4a7e77d22cfee76cbb6038c18eae4e2941e7d30000bc503d, and the LLaVA anatomy page written as a document inspection with no VLM downloaded or run.

## Grade
Grade: P22 Learn 8,700 of 10,000.
Deductions: synthetic teaching set instead of the SPEC public data,
no license-verified public captioned image set used (minus 500);
tiny teaching scale only, 16 by 16 images, 12 combinations, one
shape per image, vocabulary of 9 (minus 300); VLM anatomy is a
document inspection, no VLM downloaded or run (minus 200);
NumPy only, no framework port, one seed and one training
configuration (minus 200); run on the Muse VM CPU, not the
assigned BlackBox (GPU) fleet node (minus 100).
Zero-tolerance: PASS (no invented claim, every number measured by
the runner or hand-computed and test-asserted, synthetic data
labelled synthetic in every file, em dash count 0 in all touched
files, scaffold never graded as a build).
