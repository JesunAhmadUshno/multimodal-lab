"""Checks for the post-v1.0 analysis scripts in analysis/.
Run: python tests/test_analysis.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import paired_stats as ps  # noqa: E402


def load(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def test_mcnemar_hand_values():
    # 3 discordant pairs all one way: p = 2 * (1/8) = 0.25
    r = ps.mcnemar_exact([1, 1, 1, 0], [0, 0, 0, 0])
    assert (r["vision_only_correct"], r["text_only_correct"]) == (3, 0)
    assert abs(r["p_value"] - 0.25) < 1e-12
    # no discordant pairs: p = 1
    assert ps.mcnemar_exact([1, 0], [1, 0])["p_value"] == 1.0


def test_stats_match_recorded_deep_summaries():
    deep = load("level-3-deep", "results_deep.json")
    stats = load("analysis", "results_stats.json")
    for qt, row in stats["per_question_type"].items():
        rec = deep["per_question_type"][qt]
        assert row["vision_em"] == rec["vision_exact_match"], qt
        assert row["text_em"] == rec["text_exact_match"], qt
        lo, hi = row["vision_em_ci95"]
        assert lo <= row["vision_em"] <= hi, qt
    assert stats["overall"]["vision_em"] == deep["overall"]["vision_exact_match"]
    # the count failure class: only single-item documents score
    assert stats["count_by_gold_items"]["1"]["vision_em"] == 1.0
    for n in ("2", "3", "4"):
        assert stats["count_by_gold_items"][n]["vision_answers"] == ["one"]


def test_seed_42_reproduces_recorded_runs():
    seeds = load("analysis", "results_seeds.json")
    learn = load("level-1-learn", "results_learn.json")
    deep = load("level-3-deep", "results_deep.json")
    s42 = seeds["learn_per_seed"][seeds["seeds"].index(42)]
    for k, v in learn["after_training"].items():
        assert s42[k] == v, k
    v42 = seeds["deep_vision_em_per_seed"][seeds["seeds"].index(42)]
    assert v42["overall"] == deep["overall"]["vision_exact_match"]
    errors = sum(e["count"] for e in seeds["learn_confusion_seed_42"]["errors"])
    assert errors == round((1 - learn["after_training"]["image_to_text_recall_at_1"]) * 120)


if __name__ == "__main__":
    test_mcnemar_hand_values()
    test_stats_match_recorded_deep_summaries()
    test_seed_42_reproduces_recorded_runs()
    print("analysis tests: PASS, 3 checks")
