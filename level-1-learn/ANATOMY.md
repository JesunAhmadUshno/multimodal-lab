# VLM anatomy: LLaVA, inspected part by part (register item 311)

Status: document inspection, written 2026-11-05 for P22 Learn.
No VLM was downloaded or run in this level, and no measurement in
this file comes from a run. Every statement below describes the
published LLaVA design (Liu and co-authors, "Visual Instruction
Tuning", 2023, and the LLaVA-1.5 revision) as public documentation.
Where a behaviour is a known tendency of this design family, it is
labelled a tendency, not a measured result of this project.

## How an image becomes tokens, in one paragraph
LLaVA connects a pre-trained vision encoder to a pre-trained
language model with one small trained bridge. The image is encoded
into a grid of visual features. A projector maps those features
into the language model's word-embedding space, so they can be
placed in the token sequence next to the text tokens. The language
model then reads the mixed sequence (visual tokens first, then the
question) and generates the answer as ordinary text. The joints are
the point of this page: the encoder never sees the question, and
the language model never sees a pixel.

## Part 1: the vision encoder
- Job: turn pixels into a grid of feature vectors that carry what
  is in the image and roughly where.
- In LLaVA: a pre-trained CLIP vision transformer (ViT-L/14 in the
  original LLaVA). LLaVA takes the feature grid from the layer
  before the last transformer layer, not the final pooled vector,
  so spatial layout survives into the language model.
- Input: the image, resized and normalised to the encoder input
  size (224 by 224 in the original, 336 by 336 in LLaVA-1.5).
- Output: one feature vector per image patch (a 24 by 24 grid, 576
  vectors, at 336 input with patch size 14 in LLaVA-1.5).
- Failure tendency: the encoder was trained for image and caption
  matching, not for reading. Small text, exact counts, and fine
  spatial relations are weakly represented in its features, and no
  later part can recover detail the encoder discarded.

## Part 2: the projector
- Job: translate visual features into the language model's
  embedding space. This is the joint between two models that were
  trained separately, on different data, with different objectives.
- In LLaVA: the original design used a single linear layer;
  LLaVA-1.5 replaced it with a two-layer MLP with a GELU activation,
  which improved results. It is small compared with both neighbours.
- Input: the encoder feature grid, one vector per patch.
- Output: the same count of vectors, now shaped and scaled like
  word embeddings. These are the visual tokens.
- Failure tendency: the projector is trained on far less data than
  either neighbour, and it is the narrowest part of the system. It
  can only pass what fits through the alignment it learned; visual
  distinctions that captions in its training data never named tend
  to be flattened away here.

## Part 3: the token sequence (the second joint)
- Job: present visual tokens and text tokens to the language model
  as one sequence, in a fixed order, inside its context window.
- In LLaVA: the projected visual tokens are placed where the image
  placeholder sits in the instruction template, before the user
  question, and the sequence is trained with the answer tokens as
  the prediction target.
- Input: visual tokens plus the tokenised instruction and question.
- Output: one combined sequence; the language model cannot tell
  from the mechanics alone which positions came from the image,
  only from what training taught it about those positions.
- Failure tendency: visual tokens consume context. At 576 tokens
  per image in LLaVA-1.5, high resolution detail competes with text
  length, and multi-image inputs multiply the cost. Compression of
  the visual grid trades exactly the detail that counting and
  reading tasks need.

## Part 4: the language model
- Job: reason over the mixed sequence and generate the answer text.
  All question understanding, dialogue behaviour, and answer style
  live here.
- In LLaVA: Vicuna (a LLaMA fine-tune) in the original and
  LLaVA-1.5 designs. In stage 1 training only the projector is
  trained, with both neighbours frozen; in stage 2 the projector
  and the language model are fine-tuned on visual instruction data
  while the vision encoder stays frozen.
- Input: the combined visual and text token sequence.
- Output: generated answer tokens, sampled as in any language
  model.
- Failure tendency: the language model is a strong text prior. When
  the visual evidence is weak or the question is leading, it can
  answer from what text usually says instead of what the image
  shows, which is the standard description of visual hallucination
  in this design family. It also inherits the frozen encoder
  limits: it cannot count objects the encoder features merge.

## What this inspection teaches the Build step
- The dual encoder trained in this level is the same idea as
  Part 1 with a text tower attached: two encoders, one shared
  space, contrastive alignment. A VLM replaces the text tower with
  a full language model and the contrastive objective with
  next-token prediction, but the visual bottleneck stays.
- Build (PROPOSAL.md (former SPEC.md)) wires one open VLM into a question-answering
  service. The parts to instrument are exactly the joints above:
  encoder resolution, projector behaviour, and visual token count,
  because the failure classes named for Deep (counting, small text,
  spatial relations) are born at those joints, not in the answer
  sampling.
