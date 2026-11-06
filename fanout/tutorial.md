# Tutorial: P22 multimodal-lab, course module

Module: P22-multimodal-lab
By Jesun Ahmad Ushno
Date: 2026-11-05
License for this tutorial: CC BY 4.0. Code in the repo: Apache-2.0.

## What you will do

You will clone the multimodal-lab repo, run all three levels,
run the test suite for each level, and read your own results
against the recorded v1.0 numbers. Every command below was
verified by running it in this fan-out cycle. The project needs
Python 3.11 or 3.12 and NumPy 1.26.4 (see requirements.txt).

One scope fact governs every step: no VLM is downloaded or run
by any command below, the vision route is the Learn encoder as
a labelled stand-in, and every dataset is synthetic only, with
no public data claimed.

## Step 0: Clone and enter the repo

```text
git clone https://github.com/JesunAhmadUshno/multimodal-lab.git
cd multimodal-lab
pip install -r requirements.txt
```

Optional smoke check, before the levels:

```text
python tests/test_smoke.py
```

Expected: scaffold smoke test: PASS, 8 files present.

## Step 1: Level 1, Learn, NumPy CLIP-style dual encoder

Read level-1-learn/README.md first.

```text
python level-1-learn/run_learn.py
python tests/test_learn_multimodal.py
```

Expected: train loss 3.124841 to 0.005265 over 2,500 steps;
held-out image-to-text Recall at 1 0.983333 and text-to-image
Recall at 1 1.0 after training; P22 Learn tests: PASS (8 tests).
The runner writes level-1-learn/results_learn.json
deterministically, byte-identical across runs.

## Step 2: Level 2, Build, QA service on the encoder stand-in

Read level-2-build/README.md and level-2-build/BUILD-SPEC.md.

```text
python level-2-build/run_build.py
python tests/test_build_qa.py
```

Expected: cases 48, exact match 1.0, mean token F1 1.0, schema
48 of 48; build tests: PASS, 22 checks. Read that perfect score
at its real weight: one image per combination, synthetic only,
no VLM run. The runner writes level-2-build/results_build.json.

## Step 3: Level 3, Deep, vision against text-only study

Read level-3-deep/README.md and level-3-deep/DEEP-SPEC.md.

```text
python level-3-deep/run_deep.py
python tests/test_deep_eval.py
```

Expected: 144 synthetic document cases; vision exact match
0.354167 and text exact match 0.666667; vision colour 0.750
against text 0.000; vision count 0.250 overall and 0.000 on
multi-item documents; deep tests: PASS, 29 checks. The runner
writes level-3-deep/results_deep.json.

## Step 4: Full suite

```text
python tests/test_smoke.py
python tests/test_learn_multimodal.py
python tests/test_build_qa.py
python tests/test_deep_eval.py
```

Expected: 59 checks in total (8 Learn, 22 Build, 29 Deep), all
passing, plus the scaffold smoke test.

## If your numbers differ

The runners are deterministic (master seed 42). Different
numbers mean a different code state or a different NumPy
version: check requirements.txt first, then compare the
results files against the level READMEs before trusting a run.
