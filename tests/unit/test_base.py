import pytest

from sdmetrics.base import BaseMetric
from sdmetrics.goal import Goal


class TestBaseMetric:

    def test_normalize_bounded(self):
        BaseMetric.max_value = 1
        BaseMetric.min_value = -1
        BaseMetric.goal = Goal.MAXIMIZE

        raw_score = 0
        normalized = BaseMetric.normalize(raw_score)

        assert normalized == .5

    def test_normalize_high_bound(self):
        BaseMetric.max_value = 1
        BaseMetric.min_value = float('-inf')
        BaseMetric.goal = Goal.MAXIMIZE

        raw_score = 1
        normalized = BaseMetric.normalize(raw_score)

        assert normalized == 1

    def test_normalize_low_bound(self):
        BaseMetric.max_value = float('inf')
        BaseMetric.min_value = -1
        BaseMetric.goal = Goal.MAXIMIZE

        raw_score = -1
        normalized = BaseMetric.normalize(raw_score)

        assert normalized == 0

    def test_normalize_unbounded(self):
        BaseMetric.max_value = float('inf')
        BaseMetric.min_value = float('-inf')
        BaseMetric.goal = Goal.MAXIMIZE

        raw_score = 0
        normalized = BaseMetric.normalize(raw_score)

        assert normalized == .5

    def test_normalize_minimize(self):
        BaseMetric.max_value = 1
        BaseMetric.min_value = -1
        BaseMetric.goal = Goal.MINIMIZE

        raw_score = 1
        normalized = BaseMetric.normalize(raw_score)

        assert normalized == 0

    def test_normalize_out_of_bounds(self):
        BaseMetric.max_value = 1
        BaseMetric.min_value = -1
        BaseMetric.goal = Goal.MAXIMIZE

        raw_score = 2
        with pytest.raises(ValueError):
            BaseMetric.normalize(raw_score)
