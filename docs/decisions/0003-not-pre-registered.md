# 0003: Label the repo not pre-registered

Date: 2026-11-08. Status: accepted. Decided by: Jesun Ahmad Ushno.

## Context

The standard tags a plan as proposal-v1 only where git history proves
the plan, with hypotheses and pass lines, came before any code or
result. The first commit (46a76da, 2026-11-04) holds SPEC.md with scope
and acceptance criteria, but no hypotheses or pass lines. LEARN-SPEC.md,
BUILD-SPEC.md and DEEP-SPEC.md, with their hand-computed oracles, were
committed in the same commits as the code and results they describe
(620e79a, e471695, 959afbc). The seed, statistics and baseline analysis
came on 2026-11-07, after v1.0.

## Decision

No proposal-v1 tag. The repo carries "not pre-registered" and is capped
at Functional. The questions in PROPOSAL.md are labelled post hoc. The
old release tags v0.1, v0.5 and v1.0 stay as they are; they mark the
earlier Learn / Build / Deep ladder, not a v1.0 under this standard.

## Consequences

The cap lifts only through a confirmatory cycle whose PROPOSAL.md is
locked by tag before its first run.
