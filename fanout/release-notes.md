# Release notes: multimodal-lab (P22)

One section per release line. Every number below comes from the
level README of that release and the release text on GitHub,
verified with gh release list on 2026-11-05. Every release
states the same scope: no VLM was downloaded or run, the vision
route is the Learn encoder as a labelled stand-in, and every
dataset is synthetic only, with no public data claimed. The full
release text lives on the GitHub releases page:
https://github.com/JesunAhmadUshno/multimodal-lab/releases

## v0.1, Learn (2026-11-05)

- What shipped: a NumPy CLIP-style dual encoder trained end to
  end on a synthetic teaching set, plus the LLaVA anatomy page
  as a document inspection, and 8 hand-computed Learn tests
  plus the scaffold smoke test.
- Measured result: 1,200 train pairs and 120 held-out pairs
  over 12 colour and shape combinations; train loss 3.124841
  to 0.005265 over 2,500 steps; held-out image-to-text Recall
  at 1 0.050 to 0.983333, text-to-image Recall at 1 0.0 to 1.0.
- Level grade: 8,700 of 10,000.
- Release: https://github.com/JesunAhmadUshno/multimodal-lab/releases/tag/v0.1

## v0.5, Build (2026-11-05)

- What shipped: a deterministic QA service over synthetic
  images on the Learn encoder as a labelled VLM stand-in (no
  VLM run), schema validation on every output, P06-style
  scoring, and the multimodal tokenization note, plus 22 Build
  checks (30 checks in total with Learn, plus the smoke test).
- Measured result: 48 synthetic QA cases, exact match 1.000
  and mean token F1 1.000 overall and per type, schema
  validation 48 of 48 passed.
- Level grade: 8,200 of 10,000.
- Release: https://github.com/JesunAhmadUshno/multimodal-lab/releases/tag/v0.5

## v1.0, Learn, Build, and Deep (2026-11-05)

- What shipped: a vision against text-only eval study on
  synthetic documents and the counting failure class in depth,
  plus 29 Deep checks (59 checks in total, plus the scaffold
  smoke test). The rest of the v2 Deep scope is named in
  ROADMAP.md as the v1.1 backlog.
- Measured result: 24 synthetic documents, 144 cases; vision
  exact match 0.354167 and F1 0.385417 against text exact match
  0.666667 and F1 0.750; vision colour 0.750 against text
  0.000; vision count 0.250 overall and 0.000 on multi-item
  documents, text count 1.000.
- Level grade: 8,000 of 10,000.
- Release: https://github.com/JesunAhmadUshno/multimodal-lab/releases/tag/v1.0

## Held channels

- Social posts for every release above: HELD under the standing
  platform rules. The loop prepares drafts and never posts;
  Threads needs Jesun's one tap per post.
- The v1.1 backlog lives in ROADMAP.md: a real VLM route, real
  license-checked public data, a counting fix, small text,
  spatial depth, a learned text route, and scale.
