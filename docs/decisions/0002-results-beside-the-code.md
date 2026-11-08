# 0002: Keep results beside the code that writes them

Date: 2026-11-08. Status: accepted. Decided by: Jesun Ahmad Ushno.

## Context

The standard's layout puts raw evidence in experiments/ and scored
results in results/. Here every runner writes one deterministic JSON
file beside itself (level-*/results_*.json, analysis/results_*.json),
and the tests, the analysis scripts and the dashboard (site/src/App.jsx,
site/src/lab.check.mjs) import them at those paths.

## Decision

Keep the level folders and analysis/ as the code, evidence and results
homes, and list them in the README layout table. Do not add empty
experiments/, results/ or src/ folders.

## Consequences

Moving the files would change every import for no gain in evidence.
CI proves the results regenerate byte for byte from the code and
seeds, which is the property the experiments/ and results/ split
protects.
