# Learn Spec: multimodal-lab Level 1

## Objective
Close Learn per SPEC.md: train a small CLIP-style dual encoder on
paired image and caption data, once, end to end (register item 310),
and write the VLM anatomy page with one open VLM inspected part by
part (register item 311).

Honest scope substitution, named up front: SPEC says public data.
No dataset was downloaded in this level, because no captioned image
set was license-verified inside this cycle. Learn therefore trains
on a synthetic teaching set built programmatically in code and
labelled synthetic in every file. A license-checked public captioned
image set is the named first step of Build. This substitution is a
grade deduction, not a silent change.

## The synthetic teaching set (synthetic, not public, not real)
- Images: 16 by 16 by 3 NumPy arrays. One coloured shape on a dark
  background (pixel value 0.05). Shapes: circle, square, triangle.
  Colours: red, green, blue, yellow, as fixed RGB triples.
  Centre jitter plus or minus 3 pixels and size variation are drawn
  from a per-sample seed, so the same shape in the same colour never
  renders as the same pixels twice, while the whole set is exactly
  reproducible from the master seed.
- Captions: the token sequence for "a <colour> <shape>", for
  example "a red circle". Vocabulary size 9 including a pad token:
  pad, a, red, green, blue, yellow, circle, square, triangle.
- Pairs: 12 colour and shape combinations. Train 1,200 pairs (100
  per combination), held-out 120 pairs (10 per combination).
  Held-out pairs use seeds the train set never used, so they are
  unseen images and unseen caption instances of known combinations.
  They test position and size generalisation, not new words or
  shapes.

## Model
- Image encoder: flatten 768 inputs, one linear layer to 128 hidden
  units with ReLU, one linear layer to a 32 unit embedding, then
  row-wise L2 normalisation.
- Text encoder: token embedding table (vocabulary 9 by 24), mean
  pooling over non-pad tokens, one linear layer to the same 32 unit
  embedding space, then row-wise L2 normalisation.
- Shared space: cosine similarity is the dot product of the two
  normalised embeddings. Logits are similarities divided by a fixed
  temperature of 0.1.
- Loss: temperature-scaled contrastive loss (InfoNCE) in both
  directions, the mean of image-to-text and text-to-image cross
  entropy over the batch diagonal.
- Optimiser: Adam, hand-written, learning rate 0.01, 2,500 steps.
  Each step is one minibatch of 12 pairs, exactly one pair per
  colour and shape combination, cycling deterministically through
  the 100 train examples of each combination. All gradients are
  hand-written backpropagation in NumPy. No framework, no
  autograd, no torch.
- Design revision 1, recorded during the build on 2026-11-05: the
  first run used full-batch training over all 240 pairs (20 per
  combination). That design is flawed for this set: each batch
  holds 20 identical caption sequences per combination, and
  InfoNCE asks the model to pick one of them as the single correct
  match, so the loss has a floor near log(20) = 2.9957 however well
  the combinations are separated. The first run showed exactly
  that signature: train loss plateaued at 3.000696 and image-to-text
  Recall at 1 reached only 0.6 while text-to-image reached 1.0.
  The minibatch rule above (one pair per combination per step)
  makes every caption in a batch unique, so the objective is well
  posed.
- Design revision 2, recorded the same day: with the minibatch rule
  at 20 train examples per combination, 64 hidden units, and 800
  steps, the loss fell to 0.005952 but held-out image-to-text
  Recall at 1 stayed at 0.575 while text-to-image was 1.0. The
  flattened image encoder is position-sensitive, and 20 examples
  do not cover the jitter and size space (7 by 7 centre positions
  by 2 sizes per combination). The recorded run therefore trains
  on 100 examples per combination with 128 hidden units and 2,500
  steps. Both superseded runs are named here, not hidden.
- Master seed: 42 for parameter initialisation and for every
  per-sample data seed. No other randomness exists in the run.

## Hand-computed oracle for the contrastive loss
Worked by hand before any code ran, and asserted in the tests.
Take 2 pairs, embedding dimension 2, temperature 1.0, and both
embedding matrices equal to the identity. Similarities are then the
identity matrix. Each softmax row is exp(1) over exp(1) plus exp(0),
which is 0.7310585786, so each direction has cross entropy
minus log(0.7310585786) = 0.3132616875, and the two-direction loss
is the same value: 0.3132616875.
Second oracle: if all four embeddings are the single vector (1, 0),
every logit is 1, every softmax is uniform over 2, and the loss is
log(2) = 0.6931471806 in both directions.
Third oracle: with the identity embeddings and temperature 0.5 the
logits double, the diagonal softmax is 0.8807970780, and the loss is
0.1269280110. Temperature sharpens; it does not change the ranking.

## Retrieval expectations, stated before the run
- Metric: Recall at 1 and Recall at 5, image-to-text and
  text-to-image, on the 120 held-out pairs. Because 10 held-out
  pairs share each exact caption text, retrieval is scored by
  combination match: a retrieved caption counts as correct when its
  colour and shape combination equals the query combination, and a
  retrieved image counts as correct when its combination equals the
  caption combination. Exact-instance retrieval would punish
  correct answers, because duplicate captions are identical token
  sequences by construction. This scoring rule is fixed here, in
  the spec, before the run.
- Known scoring property, stated before the recorded run:
  held-out captions repeat, so the 10 captions of one combination
  have identical embeddings and identical similarities to any
  image. The image-to-text top 5 is therefore a tied block of
  copies of the nearest few distinct captions, and image-to-text
  Recall at 5 can equal Recall at 1 without any error. Text-to-image
  has no such tie, because every image renders differently.
- Untrained baseline: the same model, same seed initialisation,
  evaluated before any training step.
- Pass bar, fixed before the run: trained Recall at 1 must beat the
  untrained Recall at 1 by at least 0.30 in both directions, and
  trained train loss last must be below trained train loss first.
  If the bar is missed, the level is not done and the numbers are
  reported as a failure, not tuned away silently. Any tuning change
  would be recorded here with its reason.

## Commands
- Test: python tests/test_learn_multimodal.py
- Run: python level-1-learn/run_learn.py
- Full: python tests/test_smoke.py plus the Learn tests, which is
  exactly what CI runs on Python 3.11 and 3.12, followed by the
  Learn runner itself.

## Determinism rules
- Fixed master seed, fixed data seeds, deterministic minibatch
  cycling (no shuffling, so no order randomness), no wall-clock
  fields, no timestamps, no host names in results_learn.json.
- The runner writes counts and metrics rounded to 6 decimal places.
- Two consecutive runs must produce a byte-identical
  results_learn.json. The sha256 is recorded in the level README.

## Boundaries
- Always: NumPy only, deterministic seeded behaviour, numbers only
  from the real run, synthetic data labelled synthetic everywhere.
- Never: a downloaded dataset without a verified license, a claim
  that synthetic results are public-data results, a VLM measurement
  (the anatomy page is a document inspection; no VLM was downloaded
  or run), an em dash character in any file.

## Success criteria
- The hand-computed loss oracles pass in tests to 1e-9.
- Embedding outputs have shape (batch, 32) and unit row norms.
- Retrieval on a trivially separable set is perfect in tests.
- The real run clears the retrieval pass bar stated above.
- Full test suite passes, and results_learn.json is byte-identical
  across two runs.
