import numpy as np
import pandas as pd

from sdmetrics.column_pairs.statistical.kl_divergence import (
    ContinuousKLDivergence, DiscreteKLDivergence)


class TestContinuousKLDivergence:

    @staticmethod
    def ones():
        return pd.DataFrame({
            'a': [1] * 100,
            'b': [1.0] * 100,
        })

    @staticmethod
    def zeros():
        return pd.DataFrame({
            'a': [0] * 100,
            'b': [0.0] * 100,
        })

    @staticmethod
    def real():
        return pd.DataFrame({
            'a': np.random.normal(size=600),
            'b': np.random.randint(0, 10, size=600),
        })

    @staticmethod
    def good():
        return pd.DataFrame({
            'a': np.random.normal(loc=0.01, size=600),
            'b': np.random.randint(0, 10, size=600),
        })

    @staticmethod
    def bad():
        return pd.DataFrame({
            'a': np.random.normal(loc=5, scale=3, size=600),
            'b': np.random.randint(5, 15, size=600),
        })

    def test_perfect(self):
        output = ContinuousKLDivergence.compute(self.ones(), self.ones())
        normalized = ContinuousKLDivergence.normalize(output)

        assert output == 1
        assert normalized == 1

    def test_awful(self):
        output = ContinuousKLDivergence.compute(self.ones(), self.zeros())
        normalized = ContinuousKLDivergence.normalize(output)

        assert 0.0 <= output < 0.1
        assert 0.0 <= normalized < 0.1

    def test_good(self):
        output = ContinuousKLDivergence.compute(self.real(), self.good())
        normalized = ContinuousKLDivergence.normalize(output)

        assert 0.5 < output <= 1
        assert 0.5 < normalized <= 1

    def test_bad(self):
        output = ContinuousKLDivergence.compute(self.real(), self.bad())
        normalized = ContinuousKLDivergence.normalize(output)

        assert 0 <= output < 0.5
        assert 0 <= normalized < 0.5


class TestDiscreteKLDivergence:

    @staticmethod
    def ones():
        return pd.DataFrame({
            'a': ['a'] * 100,
            'b': [True] * 100,
        })

    @staticmethod
    def zeros():
        return pd.DataFrame({
            'a': ['b'] * 100,
            'b': [False] * 100,
        })

    @staticmethod
    def real():
        return pd.DataFrame({
            'a': ['a', 'b', 'b', 'c', 'c', 'c'] * 100,
            'b': [True, True, True, True, True, False] * 100,
        })

    @staticmethod
    def good():
        return pd.DataFrame({
            'a': ['a', 'b', 'b', 'b', 'c', 'c'] * 100,
            'b': [True, True, True, True, False, False] * 100,
        })

    @staticmethod
    def bad():
        return pd.DataFrame({
            'a': ['a', 'a', 'a', 'a', 'b', 'b'] * 100,
            'b': [True, False, False, False, False, False] * 100,
        })

    def test_perfect(self):
        output = DiscreteKLDivergence.compute(self.ones(), self.ones())
        normalized = DiscreteKLDivergence.normalize(output)

        assert output == 1
        assert normalized == 1

    def test_awful(self):
        output = DiscreteKLDivergence.compute(self.ones(), self.zeros())
        normalized = DiscreteKLDivergence.normalize(output)

        assert 0.0 <= output < 0.1
        assert 0.0 <= normalized < 0.1

    def test_good(self):
        output = DiscreteKLDivergence.compute(self.real(), self.good())
        normalized = DiscreteKLDivergence.normalize(output)

        assert 0.5 < output <= 1
        assert 0.5 < normalized <= 1

    def test_bad(self):
        output = DiscreteKLDivergence.compute(self.real(), self.bad())
        normalized = DiscreteKLDivergence.normalize(output)

        assert 0 <= output < 0.5
        assert 0 <= normalized < 0.5
