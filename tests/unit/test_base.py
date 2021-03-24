from sdmetrics.base import BaseMetric
from sdmetrics.goal import Goal


class TestBaseMetric:

    @staticmethod
    def positive():
        BaseMetric.max_value = 2000
        BaseMetric.min_value = 1000
        score = 1600

        return BaseMetric, score

    def test_normalize_positive_bounds_maximize(self):
        BaseMetric, score = self.positive()
        BaseMetric.goal = Goal.MAXIMIZE
        normalized = BaseMetric.normalize(score)

        assert normalized == .6

    def test_normalize_positive_bounds_minimize(self):
        BaseMetric, score = self.positive()
        BaseMetric.goal = Goal.MINIMIZE
        normalized = BaseMetric.normalize(score)

        assert normalized == .4
