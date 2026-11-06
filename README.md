# multimodal-lab

Vision-language systems: a small CLIP, an open VLM in a question-answering service, a VLM eval study.

Status: Learn DONE 2026-11-05 (ROADMAP-V3-BATCH). Build and Deep not started.
Project: P22. Fleet node: BlackBox (GPU). Wave: Wave 8.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: DONE 2026-11-05. A NumPy CLIP-style dual encoder trained end to end on a synthetic teaching set (labelled synthetic; the SPEC public-data step moves to Build with a license-checked set), plus the LLaVA anatomy page as a document inspection.
- Level 2, Build: a real working system other people can run.
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases.

## Benchmarks

Method: synthetic teaching set, 1,200 train pairs and 120 held-out pairs over 12 colour and shape combinations, combination-match recall on held-out pairs, before and after training, Muse VM CPU, NumPy. Full record in level-1-learn/README.md and level-1-learn/results_learn.json.

| Test | Result |
| --- | --- |
| Learn train loss, first step to last step | 3.124841 to 0.005265 |
| Learn image-to-text Recall at 1, before to after | 0.050000 to 0.983333 |
| Learn text-to-image Recall at 1, before to after | 0.000000 to 1.000000 |

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
