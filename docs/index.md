# multimodal-lab

Vision-language work at teaching scale, stated honestly first: no VLM was downloaded or run at any level of this project, the vision route is the Learn encoder as a labelled stand-in, and every dataset is synthetic only, with no public data claimed.

- Learn: a NumPy CLIP-style dual encoder trained on a synthetic teaching set of 1,200 train pairs and 120 held-out pairs over 12 colour and shape combinations. Train loss 3.124841 to 0.005265; held-out image-to-text Recall at 1 0.050 to 0.9833, text-to-image Recall at 1 0.0 to 1.0. Grade 8,700 of 10,000.
- Build: a deterministic QA service on the Learn encoder stand-in (no VLM run), with schema validation on every output. 48 synthetic cases: exact match 1.0, mean token F1 1.0, schema 48 of 48. That perfect score proves the service loop on one image per combination, not VLM skill. Grade 8,200 of 10,000.
- Deep: a vision against text-only eval study on synthetic documents, 144 cases. Vision exact match 0.354 and F1 0.385 against text exact match 0.667 and F1 0.750. Vision pays on colour (0.75 against text 0.0) and loses on shape, count, title, and left_shape. The counting failure class: the vision count is constant one, 0.25 overall and 0.0 on multi-item documents. Grade 8,000 of 10,000.

Tests: 59 checks in total (8 Learn, 22 Build, 29 Deep), plus the scaffold smoke test.

Grades on record (of 10,000): Learn 8,700, Build 8,200, Deep 8,000.

Every number above is already published in this repo (README.md, DOSSIER.md, and the three level READMEs and results files) and reproduced by the runners in this repo, checked in CI. The v1.1 backlog in ROADMAP.md names the real VLM route, real public data, and the counting fix that would replace these synthetic stand-in results.

Releases: https://github.com/JesunAhmadUshno/multimodal-lab/releases
