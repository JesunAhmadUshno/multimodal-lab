# multimodal-lab

**Question.** On synthetic documents, where does a vision route pay
against a text-only parser, and why does the vision route fail at
counting? (The scaffold asked this of a real vision-language model;
none was ever run, so the vision route here is a stand-in.)

**Headline.** On 144 synthetic document questions, a stand-in vision route (a NumPy CLIP-style encoder plus a keyword composer; no real VLM ran) scored exact match 0.354 (document-bootstrap 95% CI 0.229 to 0.486) against 0.667 for a text-only parser; it won only on colour (0.750 against 0.000) and scored 0.000 on counting multi-item documents, while a non-learned region count on the same pixels scored 0.750.

**Field.** ML and AI research. **Job family.** Applied AI and
multimodal engineering (vision-language systems and their evaluation).

**Verdict.** Functional, not pre-registered. Tests pass in CI on Python
3.11 and 3.12, every results file regenerates byte for byte from the
code and seeds in CI, and the system under study (encoder, QA service,
text parser, eval study) has run end to end with its outputs committed.
The system that ran is a stand-in: no VLM was downloaded or run, and
all data is synthetic. No plan was locked before the results
(PROPOSAL.md, docs/decisions/0003), so the verdict is capped at
Functional until a locked confirmatory cycle runs. The old tags v0.1,
v0.5 and v1.0 come from the earlier Learn, Build, Deep ladder and do
not mean "v1.0 Released" under the current standard.

Live dashboard: https://jesunahmadushno.github.io/multimodal-lab/
(retrain the encoder in your browser, replay the recorded Build and
Deep cases, and see the paired statistics; built from site/ by
.github/workflows/pages.yml).

## Reproduce

```text
pip install -r requirements.txt
for t in tests/test_*.py; do python "$t" || exit 1; done
python level-1-learn/run_learn.py && python level-2-build/run_build.py && python level-3-deep/run_deep.py
python analysis/run_seeds.py && python analysis/paired_stats.py && python analysis/count_baseline.py
git diff --exit-code -- level-1-learn/results_learn.json level-2-build/results_build.json level-3-deep/results_deep.json analysis/
```

NumPy only (pinned per Python version), CPU, under a minute in total.
The tests run as scripts, as CI does. Lines three and four rewrite
every results file; the last line proves nothing moved. REPORT.md
traces every headline number to its file and field.

## Lifecycle

- Question: DONE 2026-11-04
- Proposal: NOT PRE-REGISTERED 2026-11-08
- Baseline: DONE 2026-11-05
- Build: DONE 2026-11-05
- Evaluate: OPEN
- Report: DONE 2026-11-08
- Review: OPEN
- Release: OPEN

## Layout

| Path | What it holds |
| --- | --- |
| PROPOSAL.md | Question, post hoc questions, a draft plan for a locked confirmatory cycle (not locked; holds the former SPEC.md, NOVELTY.md and DOSSIER.md) |
| REPORT.md | Background, method, results traced to files, questions scored with misses, threats, verdict, gaps (holds the former ROADMAP.md, docs/research-review.md, docs/index.md and the v1.0 benchmark table) |
| level-1-learn/ | Code: the NumPy CLIP-style encoder (clip_numpy.py), its runner and results_learn.json; LEARN-SPEC.md; ANATOMY.md (LLaVA, a document inspection) |
| level-2-build/ | Code: the QA service on the encoder stand-in (qa_service.py), its runner and results_build.json; BUILD-SPEC.md |
| level-3-deep/ | Code: the document set, text-only parser and eval study (deep_eval.py), its runner and results_deep.json; DEEP-SPEC.md |
| analysis/ | Seeds, paired statistics and the enumeration baseline, with their results_*.json |
| tests/ | Unit tests with hand-computed oracles, and the layout check, run in CI |
| site/ | The live dashboard, built from the results files and REPORT.md |
| docs/decisions/ | Records of hard-to-reverse choices |
| fanout/, PUBLICITY.md | Dated 2026-11-05 publicity drafts and the channel log (nothing posted) |
| CHANGELOG.md, CITATION.cff, SECURITY.md, LICENSE | Change history, citation, security policy, licence |

Evidence and results live next to the code that writes them, in each
level folder and in analysis/, instead of separate experiments/ and
results/ folders: every runner writes one JSON beside itself, and the
dashboard and tests read them at those paths (docs/decisions/0002).
There is no src/ for the same reason.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
