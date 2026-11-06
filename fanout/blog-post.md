# When Vision Pays, and When Text Already Has the Answer

By Jesun Ahmad Ushno
Date: 2026-11-05
Status: DRAFT, written fan-out for P22 multimodal-lab v1.0. Not posted.

## The problem

Vision-language systems are usually sold as one model that does
everything. This project asks a smaller, testable question first:
on the same document questions, where does a vision route earn
its cost, and where does the text layer already carry the answer?
It answers that question at teaching scale, with the limits
stated before the results.

Repository: https://github.com/JesunAhmadUshno/multimodal-lab

## Honest scope, stated first

No VLM was downloaded or run at any level of this project. The
vision route is the Learn encoder as a labelled stand-in: a
NumPy CLIP-style encoder that retrieves the nearest of 12
captions, with a deterministic composer answering from that
prediction. Every dataset in this project is synthetic only.
No public data is claimed, and no result here transfers to a
real VLM or to real images.

## What was built

Learn: a NumPy CLIP-style dual encoder with hand-written
backpropagation, trained on a synthetic teaching set of 1,200
train pairs and 120 held-out pairs over 12 colour and shape
combinations. Grade 8,700 of 10,000.

Build: a deterministic QA service on the Learn encoder stand-in
(no VLM run), with P14-style schema validation on every output
and P06-style scoring. Grade 8,200 of 10,000.

Deep: a vision against text-only eval study on synthetic
documents, plus the counting failure class written up in depth.
Grade 8,000 of 10,000.

## The main result

On synthetic data only, with no VLM downloaded or run and the Learn encoder as a labelled stand-in, Deep scored vision exact match 0.354 against text exact match 0.667 on 144 synthetic document cases, with vision paying on colour at 0.750 against text 0.000 and vision counting at 0.250 overall and 0.000 on multi-item documents.

In detail. Learn: train loss fell from 3.124841 to 0.005265
over 2,500 steps, held-out image-to-text Recall at 1 rose from
0.050 to 0.9833 and text-to-image Recall at 1 rose from 0.0 to
1.0. Build: on 48 synthetic QA cases the service scored exact
match 1.0 and mean token F1 1.0, with schema validation 48 of
48. That perfect score proves the service loop on one image per
combination; it is not evidence about any VLM. Deep, per type
on 144 cases: vision exact match 0.354 and F1 0.385 against
text exact match 0.667 and F1 0.750; vision pays on colour
(0.75 against text 0.0) and on both (0.375 against 0.0), and
loses on shape (0.375 against 1.0), count (0.250 against 1.0),
title (0.0 against 1.0), and left_shape (0.375 against 1.0).

## The counting failure

The vision count answer is constant one by construction, because
the Build prediction step has no counting operation: one image
embedding, one winning caption, no enumeration. It scores 0.25
overall, which is exactly the six single-item documents, and
0.0 on the 18 multi-item documents. The text route scores 1.0
by counting legend entries. The lesson is a warning for real
work too: eval any vision system per count value, never only
on an overall average.

## The verdict

Route document questions to the text layer first, and spend
vision only on what the text does not carry. That is the result
of this synthetic study, measured on this stand-in pipeline.
It is not a measured claim about any real VLM.

## Limits

- No VLM downloaded or run; the encoder is a labelled stand-in.
- Synthetic only at every level; no public data claimed.
- Tiny teaching scale: 16 by 16 images, 12 combinations,
  vocabulary of 9, 24 synthetic documents in Deep.
- The text route is a hand parser, not OCR or a learned model.
- Counting, title reading, and spatial depth are named
  failures or v1.1 items, not solved problems.

## Future

The v1.1 backlog is ROADMAP.md in the repo: a real VLM route,
real license-checked public data, a counting fix, small text,
spatial depth, and a learned text route. None of that is
claimed as done.

Tests: 59 checks in total (8 Learn, 22 Build, 29 Deep), plus
the scaffold smoke test.
