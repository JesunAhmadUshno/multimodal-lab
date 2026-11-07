# Research review: multimodal-lab (P22)

Reviewed 2026-11-07 against a PhD-level research rubric. This review grades the repo as a research artefact, not as a teaching project. Every number below comes from a results file in this repo, and the file is named next to it.

## Verdict

**Before this change: about 6.5 of 10. After it: about 7 of 10.**

The engineering and honesty are strong. Gradients are hand-derived and checked against finite differences. Test oracles were fixed before the runs. Every runner writes byte-identical results; this review reran all of them, and `git status` stayed clean. Substitutions are named everywhere.

The scientific reach is still the weak point, and this change cannot fix that. No real vision-language model was run, and all data is synthetic. This change closes the gaps that need only a CPU and minutes: seed variance, uncertainty, paired tests, a confusion analysis, a counting baseline, citations and test ergonomics. The gaps that remain need a model, a dataset or new study design, and each one says what it needs.

## Scorecard

| Criterion | Grade (of 10) | Status after this change |
| --- | --- | --- |
| Research question | 6 | Clear for Learn. The SPEC Deep question (a VLM against a text pipeline) is still unanswered, because the vision route is a stand-in. |
| Hypotheses and pre-specified oracles | 7 | Hand-computed oracles in LEARN-, BUILD- and DEEP-SPEC.md were fixed before the runs. There is no formal hypothesis with a falsification rule for Deep. |
| Method | 7 | Correct and fully specified. The Deep vision route is the Learn encoder plus a keyword composer, so it measures a stand-in by construction. |
| Baselines | 6 (was 5) | Deep compares two routes. **Added:** a region-count enumeration baseline for counting (`analysis/count_baseline.py`). |
| Statistics and uncertainty | 7 (was 3) | **Added:** bootstrap CIs, exact McNemar and five-seed mean and sd (`analysis/`). |
| Reproducibility | 9 | Deterministic runners, recorded sha256, CI reruns, and NumPy pinned per Python version. The new analysis is deterministic too. |
| Threats to validity | 7 (was 6) | Named in each level README. **Added:** the touching-slots flaw in the three-item documents and the clustering caveat on the overall test. |
| Related work | 5 (was 2) | **Added:** primary references in DOSSIER.md. There is still no survey of document-VQA benchmarks or of counting in VLMs. |
| Limitations | 8 | Stated plainly in the READMEs, ROADMAP v1.1 and on the dashboard. |

## What was fixed in this change

1. **Single seed, so no variance reported.** `analysis/run_seeds.py` reruns Learn and the Deep vision route with encoder seeds 42, 1, 2, 3 and 4 (`analysis/results_seeds.json`). Seed 42 reproduces the recorded runs exactly.
   - Learn image-to-text Recall at 1 is 0.990 mean (sd 0.011, range 0.975 to 1.000).
   - Text-to-image Recall at 1 is 1.000 at every seed.
   - Deep vision exact match is 0.371 mean (sd 0.017, range 0.354 to 0.396). The recorded 0.354 sits at the low end of that range.
   - Colour (0.750), count (0.250) and title (0.000) do not move across seeds. Shape, both and left_shape move between 0.375 and 0.458.
   - The text route is a deterministic parser, so it does not depend on the seed.
2. **No uncertainty or paired test.** `analysis/paired_stats.py` (`analysis/results_stats.json`) works on the 144 recorded per-case rows.
   - Per type, it gives percentile bootstrap 95% CIs by resampling the 24 cases of that type (10,000 resamples). Example: vision colour exact match 0.750, CI 0.583 to 0.917.
   - Overall, it resamples the 24 documents (cluster bootstrap): vision 0.354, CI 0.229 to 0.486. The vision-minus-text difference has a CI of -0.438 to -0.181.
   - The exact McNemar test finds every per-type difference significant. The largest p is 0.0039, for both, where vision alone was right on 9 cases and text alone on 0.
   - The overall McNemar test (27 against 72 discordant pairs, p 6.9e-6) treats cases as independent, but the six cases of a document are not. Read it as optimistic. The cluster-bootstrap interval is the honest overall figure.
   - The dashboard recomputes all of this in the browser, and `site/src/lab.check.mjs` asserts that it matches.
3. **Image-to-text R@1 equals R@5 (0.983), not analysed.** The seed-42 confusion matrix (`results_seeds.json`, `learn_confusion_seed_42`) shows the two misses: one red circle retrieved as red square, and one green square retrieved as green circle. Both are circle and square confusions within one colour. R@5 equals R@1 by construction: the held-out captions repeat 10 times per combination, so an image's top five captions are five copies of one caption. Image-to-text R@5 carries no information in this design and should not be reported as a separate result.
4. **Counting failure is true by construction, with no baseline.** `analysis/count_baseline.py` counts 4-connected non-background regions, with no learning, and scores 0.750 exact match on the 24 count questions (`analysis/results_count_baseline.json`).
   - It gets 1.000 on one-, two- and four-item figures and 0.000 on three-item figures.
   - **New threat found:** the three-item slots (x = 2, 7, 12, half-size 2) draw neighbouring shapes touching, so their pixels merge. This flaw is in the v1.0 document set itself.
   - Reading: the pixels carry most counts, so the constant-one composer is the main cause of the failure class. This is a ceiling check on fixed slots, not a counting model.
5. **Literature grounding.** Primary references were added to DOSSIER.md: CLIP, InfoNCE (CPC), LLaVA and LLaVA-1.5, Adam, McNemar and the bootstrap.
6. **Test ergonomics.** The suite runs as scripts, as CI does. The README now says this. The new `tests/test_analysis.py` uses `test_*` functions, so pytest collects it too.
7. **Dashboard self-check.** `site/src/lab.check.mjs` runs inside `npm run build` and checks the following:
   - The JavaScript ports reproduce the deterministic text route exactly: 0.666667 exact match and 0.75 token F1, row by row over 144 cases.
   - They also reproduce the gold labels and the token counts (17 visual tokens, 22 to 25 unified tokens).
   - The trainer passes the repo's loss oracles and a finite-difference gradient check.
   - The README benchmark table matches the results files.

## Gaps that remain, and exactly what each needs

| Gap | Why it cannot be closed here | What it needs |
| --- | --- | --- |
| No real VLM, so the SPEC Deep question is unanswered (ROADMAP v1.1 item 1) | Needs model weights and a GPU-class runtime, which the rules for this change exclude | One open VLM in the LLaVA-1.5 7B class, run on the same 144 cases on one GPU with about 16 GB of memory (an approved free GPU window), with the same scoring code |
| Synthetic data only, and the text layer covers 4 of 6 question types by design (item 2) | Needs a licence-checked public document set and a data-licence review | A public document-VQA set with a clear licence (or P16 document cases), reported separately from the synthetic set |
| Small-text failure class not measured (item 4) | The stand-in encoder cannot read any text (title exact match 0.000), so a small-text sweep would score 0 at every size and teach nothing | An OCR engine or a real VLM as the vision route, plus documents rendered at several text sizes |
| Spatial relations covered by one question type (item 5) | Needs new question types and a position-aware vision step: a new study design, not a fix | Left, right, above and below questions with gold labels, and a per-region vision step |
| Counting fix inside the service (item 3) | The baseline above shows the pixels carry the count; putting an enumeration step into the service changes the v1.0 system | An enumeration step before the composer, evaluated per count value, with the three-item slot layout fixed so shapes do not touch |
| Learn generalisation probed only in-distribution | Held-out pairs come from the same generator, varying only position and size | An out-of-distribution split: unseen sizes, colours with shifted RGB, or held-out colour and shape pairings |
| Small samples (24 cases per type) | The document set is fixed by DEEP-SPEC.md | More documents per item count; the bootstrap widths above give the size needed for a target precision |
| Grades are self-assessed (8,700, 8,200 and 8,000 of 10,000) | Needs a person outside the project | An external reviewer using this rubric |
| NOVELTY.md is a placeholder | No measured difference against prior art exists yet | A comparison against a reference CLIP or VLM repository on a shared task, after the real-VLM run |

## Threats to validity (consolidated)

- **Construct.** "Vision route" means the Learn encoder plus a keyword composer (`learn_encoder_stand_in`), not a VLM. No result transfers to a real VLM.
- **Internal.** The text layer was designed to carry the answers to 4 of 6 question types, which builds in the overall text win. The three-item figures have touching shapes. Count is constant in code.
- **Statistical.** There are 24 cases per type. Cases cluster within documents, so the overall McNemar test is optimistic. Seed variance is small for Learn and moderate for the Deep vision route (sd 0.017 overall).
- **External.** All data is synthetic 16 by 16 images with fixed vocabularies. Nothing here says anything about natural images or real documents.

## References

The same list is in DOSSIER.md.

- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020.
- van den Oord, A., Li, Y. and Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.
- Liu, H., Li, C., Wu, Q. and Lee, Y. J. (2023). Visual Instruction Tuning. arXiv:2304.08485.
- Liu, H., Li, C., Li, Y. and Lee, Y. J. (2023). Improved Baselines with Visual Instruction Tuning. arXiv:2310.03744.
- Kingma, D. P. and Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12(2), 153-157.
- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. Annals of Statistics 7(1), 1-26.
