# LinkedIn post draft: P22 multimodal-lab v1.0

Status: DRAFT NOT POSTED
Date drafted: 2026-11-05
Author: Jesun Ahmad Ushno
Rule for this platform: the link goes in the first comment, not in the
post body. The first comment line is at the end of this file.

## Post body

Where does vision actually pay on a document, and where does
the text layer already have the answer? I built a small study
to measure that, and I am stating its biggest limit in the
post, not under it: no VLM was downloaded or run.

multimodal-lab v1.0 is done: Learn, Build, and Deep, all open,
NumPy only. The vision route is my Learn CLIP-style encoder as
a labelled stand-in, and every dataset is synthetic only. No
public data is claimed.

Learn: a NumPy dual encoder trained on 1,200 synthetic pairs,
loss 3.124841 to 0.005265, held-out Recall at 1 up to 0.9833
image-to-text and 1.0 text-to-image.

Build: a QA service on that encoder stand-in scored exact
match 1.0 and F1 1.0 on 48 synthetic cases, schema 48 of 48.
That proves the service loop, not VLM skill.

Deep: on 144 synthetic document cases, vision exact match
0.354 against text 0.667. Vision pays on colour, 0.75 against
text 0.0, and on colour-plus-shape, 0.375 against 0.0. It loses on shape, count, title, and left_shape.
The counting failure is the finding I keep coming back to:
the vision count is constant one, 0.25 overall and 0.0 on
multi-item documents.

59 checks in total, plus the smoke test. The real VLM
route is the first v1.1 item, and it is not claimed here.

When did a text layer beat your vision pipeline?

## First comment (post this as the first comment, per the standing rule)

Repo, releases v0.1, v0.5, and v1.0, and the full tutorial: https://github.com/JesunAhmadUshno/multimodal-lab
