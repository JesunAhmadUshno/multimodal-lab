# DOSSIER: multimodal-lab (P22) v1, 2026-11-04

Written under the documentation and publishing standard (past, present, future). Updated at ship time.

## Past

Prior art for this project is named in NOVELTY.md (CLIP and VLM reference repos). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

## Present

Current demand evidence: hiring value 3 of 5 (1 posting read in full, trend years 2027 to 2028) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

## Build ship note (2026-11-05)
Build closed at grade 8,200 of 10,000: a deterministic QA service over synthetic images with the Learn encoder as a labelled stand-in for the SPEC VLM step (no VLM downloaded or run, named deduction), P14-style schema validation (48 of 48 passed), P06-style exact match and token F1 scoring (both 1.000 on the 48-case synthetic set, mechanism proof only), and the tokenization note with computed counts. Full record: level-2-build/README.md.

## Deep ship note (2026-11-05)
Deep closed at grade 8,000 of 10,000 (compressed Deep): an eval study on 24 synthetic documents (144 cases, labelled synthetic) comparing the Build QA service (learn_encoder_stand_in, no VLM downloaded or run, substitution named again) against a deterministic text-only parser on the same questions. Vision pays where the answer is only in the pixels (colour 0.750 against 0.000; both 0.375 against 0.000); text pays where the text layer carries the answer (shape, count, title, left_shape all 1.000 against vision 0.375, 0.250, 0.000, 0.375). Overall vision exact match 0.354167, text 0.666667. The counting failure class is written up in depth: the Build count branch is constant one by construction, scoring exactly the six single-item documents (0.250) and 0.000 on multi-item documents. The rest of the v2 Deep scope (real VLM route, real document data, counting fix, small text, spatial depth, learned text route, scale) is the v1.1 backlog in ROADMAP.md. Full record: level-3-deep/README.md and level-3-deep/results_deep.json (sha256 4678206d4585330acf56402676212ecac4f0cb5c51b43dade563a0b7125955b2, byte-identical across runs).

## Future

Open problems and the project's next steps sit in the Deep level scope in SPEC.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.

## References (added 2026-11-07)

Primary sources for the methods used in this repo. See docs/research-review.md for how each one is used.

- CLIP, the dual-encoder design that Learn rebuilds: Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020.
- InfoNCE, the contrastive loss in clip_numpy.py: van den Oord, A., Li, Y. and Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.
- LLaVA, the design inspected in level-1-learn/ANATOMY.md: Liu, H., Li, C., Wu, Q. and Lee, Y. J. (2023). Visual Instruction Tuning. arXiv:2304.08485.
- LLaVA-1.5: Liu, H., Li, C., Li, Y. and Lee, Y. J. (2023). Improved Baselines with Visual Instruction Tuning. arXiv:2310.03744.
- Adam, the optimiser in clip_numpy.py: Kingma, D. P. and Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980.
- McNemar's test, used in analysis/paired_stats.py: McNemar, Q. (1947). Psychometrika 12(2), 153-157.
- The bootstrap, used in analysis/paired_stats.py: Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. Annals of Statistics 7(1), 1-26.
