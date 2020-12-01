import numpy as np
import pandas as pd

from sdmetrics.single_table.single_column import CSTest, KSTest


class TestCSTest:

    def test_max(self):
        data = pd.DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': [True, False, True, False],
        })
        output = CSTest.compute(data, data)

        assert output == 1

    def test_min(self):
        real = pd.DataFrame({
            'a': [True, True, True, True],
        })
        synth = pd.DataFrame({
            'a': [False, False, False, False],
        })
        output = CSTest.compute(real, synth)

        assert output == 0

    def test_point_five(self):
        real = pd.DataFrame({
            'a': ['a', 'b', 'c', 'd'],
            'b': [True, False, True, False],
        })
        synth = pd.DataFrame({
            'a': ['e', 'f', 'g', 'h'],
            'b': [False, True, False, True],
        })
        output = CSTest.compute(real, synth)

        assert output == 0.5


class TestKSTest:

    def test_max(self):
        data = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [0, 0, 0, 0],
        })
        output = KSTest.compute(data, data)

        assert output == 1

    def test_min(self):
        real = pd.DataFrame({
            'a': np.zeros(1000),
        })
        synth = pd.DataFrame({
            'a': np.ones(1000),
        })
        output = KSTest.compute(real, synth)

        assert output == 0

    def test_point_five(self):
        real = pd.DataFrame({
            'a': np.zeros(1000),
            'b': np.ones(1000),
        })
        synth = pd.DataFrame({
            'a': np.ones(1000),
            'b': np.ones(1000),
        })
        output = KSTest.compute(real, synth)

        assert output == 0.5
