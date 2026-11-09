# 0001: Use the Learn encoder as a labelled stand-in for the VLM

Date: 2026-11-05 (recorded 2026-11-08). Status: accepted. Decided by:
Jesun Ahmad Ushno.

## Context

The scaffold (PROPOSAL.md, former SPEC.md) asked for one open
vision-language model in the Build service and the Deep study. The
machine used for the build had no model stack: on 2026-11-05
`import torch` and `import transformers` both failed and no weights
were present (level-3-deep/README.md). Running a VLM needs GPU time,
which needs the Owner's approval.

## Decision

The vision route is the Learn CLIP-style encoder retrieving the
nearest of 12 captions plus a deterministic composer, labelled
`learn_encoder_stand_in` in the code and `vlm_used: false` in every
results file. All data is synthetic and labelled so.

## Consequences

The scaffold's Deep question (a real VLM against a text pipeline) is
still open, and no result transfers to a real VLM. A real VLM can be
dropped in behind the same schema and scoring code; that run is the
confirmatory cycle drafted in PROPOSAL.md.
