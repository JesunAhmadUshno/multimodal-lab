# ROADMAP: multimodal-lab (P22)

## Shipped

- v1.0 (2026-11-05): Learn, Build, and Deep (compressed Deep
  per the December Compression Protocol). Deep is the eval
  study in level-3-deep: the Build QA service (labelled
  learn_encoder_stand_in, no VLM run) against a text-only
  parser on 144 synthetic document cases, one comparison
  table, and the counting failure class written up.

## v1.1 backlog (the rest of the v2 Deep scope, not in v1.0)

1. Real VLM eval: run one open vision-language model as the
   vision route and rerun the same document study, so the
   table compares a real VLM against the text-only pipeline
   as SPEC Deep originally states. Needs a machine with the
   model stack and weights, or an approved free GPU window.
2. Real document data: a license-checked public document
   question set, or P16 document-intelligence cases, reported
   separately from the synthetic set.
3. Counting fix: an enumeration step (detection or per-tile
   reads) in front of the answer composer, evaluated per
   count value, replacing the constant-one count branch.
4. Small text failure class: a study at several text sizes,
   the second failure class named in SPEC Deep, not measured
   in v1.0.
5. Spatial relations in depth: position-specific questions
   (left, right, above, below) with a position-aware vision
   step, beyond the single left_shape table cell in v1.0.
6. Learned text route: OCR or a learned text pipeline in
   place of the hand parser, so the text route is measured,
   not stipulated.
7. Scale: more documents, more items per document, several
   training seeds with spread reported, and a human-checked
   subset of gold labels.

Nothing in this backlog is built, measured, or claimed in
v1.0. Each item closes only with code, tests, and measured
results, like the levels before it.
