"""OcrScorer must never crash a whole batch on one malformed/edge-case prompt.

Prompts are expected to carry the OCR target quoted (`... "target" ...`), but a prompt
missing a quoted target, or with an empty one, must be scored (zero reward) instead of
raising `IndexError` / `ZeroDivisionError` out of `score_batch`.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import numpy as np

from miles.rollout.rm_hub.ocr import OcrScorer

_IMG = np.zeros((4, 4, 3), dtype=np.uint8)


def _scorer_with_recognized_text(text: str) -> OcrScorer:
    scorer = OcrScorer.__new__(OcrScorer)
    scorer.ocr = _FakeOcr(text)
    return scorer


class _FakeOcr:
    def __init__(self, text: str):
        self._text = text

    def ocr(self, img, cls=False):
        return [[[None, (self._text, 0.9)]]]


def test_prompt_without_quoted_target_gets_zero_reward_not_a_crash():
    scorer = _scorer_with_recognized_text("hello world")

    rewards = scorer([_IMG], ["draw a cat"])

    assert rewards == [0.0]


def test_prompt_with_empty_quoted_target_gets_zero_reward_not_a_crash():
    scorer = _scorer_with_recognized_text("hello world")

    rewards = scorer([_IMG], ['a sign that says ""'])

    assert rewards == [0.0]


def test_one_malformed_prompt_does_not_crash_the_rest_of_the_batch():
    scorer = _scorer_with_recognized_text("hello world")

    rewards = scorer([_IMG, _IMG], ["no quotes here", 'a sign that says "hello world"'])

    assert rewards == [0.0, 1.0]


def test_well_formed_prompt_still_scores_normally():
    scorer = _scorer_with_recognized_text("hello world")

    rewards = scorer([_IMG], ['a sign that says "hello world"'])

    assert rewards == [1.0]
