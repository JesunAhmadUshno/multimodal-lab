"""Portfolio standard layout check: required files, the abstract, the
headline traced to its evidence, and no public grade.

No lock-hash test: the repo is not pre-registered (PROPOSAL.md is an
unlocked draft written after the results, docs/decisions/0003).
Runs as a script (as CI does) or under unittest.
"""
import json
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED = ["README.md", "PROPOSAL.md", "REPORT.md", "CHANGELOG.md", "CITATION.cff", "SECURITY.md", "LICENSE",
        "requirements.txt", "level-1-learn/README.md", "level-2-build/README.md", "level-3-deep/README.md",
        "level-1-learn/results_learn.json", "level-2-build/results_build.json", "level-3-deep/results_deep.json",
        "analysis/results_seeds.json", "analysis/results_stats.json", "analysis/results_count_baseline.json",
        "docs/decisions/0001-encoder-stand-in-for-the-vlm.md", "docs/decisions/0002-results-beside-the-code.md",
        "docs/decisions/0003-not-pre-registered.md"]
GRADE = re.compile(r"of 10,000|\b\d[\d,]*/10,?000\b|Grade:|grade \d|\b\d+(\.\d+)?/10\b|/1000\b|rubric score"
                   r"|\b\d+(\.\d+)? of 10\b", re.I)
PUBLIC_EXT = (".md", ".jsx", ".js", ".mjs", ".html", ".cff", ".py", ".json")


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return json.load(fh)


class LayoutTests(unittest.TestCase):
    def test_required_files_exist(self):
        missing = [p for p in NEED if not os.path.exists(os.path.join(ROOT, p))]
        self.assertEqual(missing, [], "missing: " + ", ".join(missing))

    def test_readme_opens_with_the_abstract(self):
        text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
        for field in ("**Question.**", "**Headline.**", "**Field.**", "**Job family.**", "**Verdict.**",
                      "## Reproduce", "## Lifecycle", "## Layout"):
            self.assertIn(field, text)
        self.assertIn("**Verdict.** Functional, not pre-registered.", text)

    def test_headline_numbers_match_the_evidence(self):
        text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
        headline = text.split("**Headline.**")[1].split("\n\n")[0]
        deep, stats, count = (load("level-3-deep/results_deep.json"), load("analysis/results_stats.json"),
                              load("analysis/results_count_baseline.json"))
        lo, hi = stats["overall"]["vision_em_ci95"]
        for value in (deep["overall"]["vision_exact_match"], lo, hi, deep["overall"]["text_exact_match"],
                      deep["per_question_type"]["colour"]["vision_exact_match"],
                      deep["per_question_type"]["colour"]["text_exact_match"],
                      deep["vision_by_item_group"]["count_multi"]["vision_exact_match"], count["exact_match"]):
            self.assertIn(f"{value:.3f}", headline)
        self.assertEqual(deep["eval_cases"], 144)
        self.assertFalse(deep["vlm_used"])

    def test_no_public_grade(self):
        hits = []
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in (".git", "node_modules", "dist")]
            for name in files:
                path = os.path.join(d, name)
                if name.endswith(PUBLIC_EXT) and path != os.path.abspath(__file__):
                    for n, line in enumerate(open(path, encoding="utf-8"), 1):
                        if GRADE.search(line):
                            hits.append(f"{os.path.relpath(path, ROOT)}:{n}")
        self.assertEqual(hits, [], "grades are internal only: " + ", ".join(hits))

    def test_grade_pattern_catches_a_grade(self):
        for bad in ("Grade: 7,600 of 10,000", "about 6.5 of 10", "Total /1000", "Deep grade 8,000"):
            self.assertTrue(GRADE.search(bad), bad)
        for ok in ("10,000 resamples", "48 of 48 passed", "9 of 20"):
            self.assertFalse(GRADE.search(ok), ok)


if __name__ == "__main__":
    unittest.main()
