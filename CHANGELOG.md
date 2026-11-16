# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning. The tags v0.1, v0.5 and v1.0 come
from the earlier Learn / Build / Deep ladder; no release is tagged
under the portfolio standard v2 yet.

## [Unreleased]

### Changed
- 2026-11-08: Applied an independent review. The README headline now
  names both vision wins (colour 0.750 and colour-plus-shape 0.375,
  against text 0.000) and compares counting like with like: 0.250
  against the region count's 0.750 on all 24 count questions, 0.000
  against 0.667 on the 18 multi-item ones. REPORT.md Q4, the blog and
  LinkedIn drafts match; the smoke test checks the new numbers.
- 2026-11-08: Restructured to the portfolio standard v2. README is the
  abstract (question, headline with its interval, field, job family,
  verdict: Functional, not pre-registered). PROPOSAL.md (unlocked
  draft) holds the former SPEC.md, NOVELTY.md and DOSSIER.md; REPORT.md
  holds results traced to their files, the post hoc questions scored
  with misses, the former ROADMAP.md, docs/research-review.md,
  docs/index.md and the v1.0 benchmark table. Grades leave the public
  repo and a test keeps them out. The dashboard reads REPORT.md instead
  of ROADMAP.md and docs/index.md, and no longer shows grades.
- 2026-11-08: CI runs with read-only permissions, adds the layout check
  and requires every results file to regenerate byte for byte. CodeQL
  adds JavaScript for the dashboard; Dependabot adds npm for site/.
  Adds CHANGELOG, CITATION, SECURITY and three decision records.

### Added
- 2026-11-07: Pinned vite 6.4.5 for the dashboard build.
- 2026-11-07: Research review and primary references.
- 2026-11-07: Region-count enumeration baseline for the counting
  failure class (analysis/count_baseline.py).
- 2026-11-07: Live dashboard in site/ and its Pages workflow.
- 2026-11-07: Seed sensitivity, Learn confusion and paired statistics
  (analysis/run_seeds.py, analysis/paired_stats.py).
- 2026-11-06: NumPy pinned per Python version (1.26.4 on 3.11, 2.5.4
  on 3.12 and later).

## [1.0] - 2026-11-05 (Learn / Build / Deep ladder)

### Added
- Deep: vision against text-only eval study on 24 synthetic documents
  (144 cases), counting failure class written up. Actions polish pack
  (CI matrix, ruff, CodeQL, Dependabot, release, labeler, greetings,
  stale) and the written fan-out set.

## [0.5] - 2026-11-05 (ladder)

### Added
- Build: deterministic QA service on the Learn encoder stand-in,
  schema-validated and scored on 48 synthetic cases.

## [0.1] - 2026-11-05 (ladder)

### Added
- Learn: NumPy CLIP-style dual encoder on a synthetic teaching set and
  the LLaVA anatomy page.
- 2026-11-04: Wave 0 scaffold (SPEC.md, DOSSIER.md, NOVELTY.md).
