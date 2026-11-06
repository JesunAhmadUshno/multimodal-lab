# multimodal-lab

Vision-language systems: a small CLIP, an open VLM in a question-answering service, a VLM eval study.

Status: Learn DONE 2026-11-05, Build DONE 2026-11-05, Deep DONE 2026-11-05 (ROADMAP-V3-BATCH). v1.0 shipped 2026-11-05; the rest of the v2 Deep scope is the v1.1 backlog in ROADMAP.md.
Project: P22. Fleet node: BlackBox (GPU). Wave: Wave 8.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: DONE 2026-11-05. A NumPy CLIP-style dual encoder trained end to end on a synthetic teaching set (labelled synthetic; the SPEC public-data step moves to Build with a license-checked set), plus the LLaVA anatomy page as a document inspection.
- Level 2, Build: DONE 2026-11-05. A deterministic QA service over synthetic images on the Learn encoder as a labelled VLM stand-in (no VLM run), P14-style schema validation on every output, P06-style scoring, and the multimodal tokenization note with counts.
- Level 3, Deep: DONE 2026-11-05. An eval study on synthetic documents: the Build QA service (labelled encoder stand-in, no VLM run) against a text-only parser on the same 144 document questions, one comparison table (vision pays on colour and both, text pays on shape, count, title, and left_shape), and the counting failure class written up in depth (vision count is constant one: 0.250 overall, 0.000 on multi-item documents).

## Benchmarks

Method: synthetic teaching set, 1,200 train pairs and 120 held-out pairs over 12 colour and shape combinations, combination-match recall on held-out pairs, before and after training, Muse VM CPU, NumPy. Full record in level-1-learn/README.md and level-1-learn/results_learn.json.

| Test | Result |
| --- | --- |
| Learn train loss, first step to last step | 3.124841 to 0.005265 |
| Learn image-to-text Recall at 1, before to after | 0.050000 to 0.983333 |
| Learn text-to-image Recall at 1, before to after | 0.000000 to 1.000000 |
| Build exact match, 48 synthetic QA cases | 1.000 |
| Build mean token F1, 48 synthetic QA cases | 1.000 |
| Build schema validation | 48 of 48 passed |
| Deep vision exact match, 144 synthetic document cases | 0.354167 |
| Deep text-only exact match, same 144 cases | 0.666667 |
| Deep vision colour exact match (text 0.000) | 0.750 |
| Deep vision count exact match (text 1.000) | 0.250 |

## Fan-out

Written fan-out set for v1.0, in fanout/: blog post, tutorial,
website page copy, LinkedIn post draft (DRAFT NOT POSTED, link in
the first comment), paper candidate, and release notes for
v0.1, v0.5, and v1.0. Every piece states the honest scope: no
VLM downloaded or run, the Learn encoder as a labelled stand-in,
synthetic only, no public data claimed. Grades and channel
outcomes are logged in PUBLICITY.md. Overall fan-out grade:
9,457 of 10,000.

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
