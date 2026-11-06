# Website project page copy: multimodal-lab

Site: jesunahmadushno.com
Status: COPY ONLY, draft for the project page. Dated 2026-11-05.
Nothing on any website was changed by this file.

## Headline

multimodal-lab: where vision pays and where text already has the answer, measured on a synthetic study with no VLM run

## Summary

multimodal-lab is an open vision-language study built by Jesun
Ahmad Ushno as project P22 in Roadmap v3. It trains a NumPy
CLIP-style dual encoder, builds a deterministic QA service on
that encoder, and then studies that service against a text-only
parser on the same synthetic document questions.

One scope fact, stated plainly. No VLM was downloaded or run at
any level. The vision route is the Learn encoder as a labelled
stand-in. Every dataset is synthetic only, and no public data
is claimed.

Status: v1.0, 2026-11-05. Learn, Build, and Deep are done.
Grades on the house 10,000 scale: Learn 8,700, Build 8,200,
Deep 8,000. Total tests: 59 checks, plus the scaffold smoke test.

## Results (all synthetic)

| Level | Set | Key result | Notes |
| --- | --- | --- | --- |
| Learn | 1,200 train, 120 held-out, synthetic | Loss 3.124841 to 0.005265; Recall at 1 image-to-text 0.9833, text-to-image 1.0 after training | 12 colour and shape combinations; 8 checks |
| Build | 48 synthetic QA cases | Exact match 1.0, F1 1.0, schema 48 of 48 | Encoder stand-in, no VLM run; perfect score proves the service loop only; 22 checks |
| Deep | 144 synthetic document cases | Vision EM 0.354 F1 0.385 against text EM 0.667 F1 0.750; vision colour 0.75 against text 0.0; vision count 0.25, 0.0 on multi-item | Counting failure class written up in depth; 29 checks |

## Honest limits

- No VLM downloaded or run; the encoder is a labelled stand-in.
- Synthetic only at every level; no public data claimed.
- Tiny teaching scale and one training seed.
- The text route is a hand parser, not OCR.
- The real VLM route and real data are v1.1 items.

## Links

- GitHub repo: https://github.com/JesunAhmadUshno/multimodal-lab
- Release v0.1 (Learn): https://github.com/JesunAhmadUshno/multimodal-lab/releases/tag/v0.1
- Release v0.5 (Build): https://github.com/JesunAhmadUshno/multimodal-lab/releases/tag/v0.5
- Release v1.0 (Learn, Build, Deep): https://github.com/JesunAhmadUshno/multimodal-lab/releases/tag/v1.0
- Tutorial: fanout/tutorial.md in the GitHub repo
- Blog post draft: fanout/blog-post.md in the GitHub repo

## License note

Code: Apache-2.0. Tutorials: CC BY 4.0.
