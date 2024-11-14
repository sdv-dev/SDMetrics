import pandas as pd

from sdmetrics.single_column import SequenceLengthSimilarity


class TestSequenceLengthSimilarity:
    def test_compute(self):
        """Test it runs."""
        # Setup
        real_data = pd.Series(['id1', 'id2', 'id2', 'id3'])
        synthetic_data = pd.Series(['id4', 'id5', 'id6'])

        # Run
        score = SequenceLengthSimilarity.compute(real_data, synthetic_data)

        # Assert
        assert score == 0.6666666666666667

    def test_compute_one(self):
        """Test it returns 1 when real and synthetic data have the same distribution."""
        # Setup
        real_data = pd.Series(['id1', 'id1', 'id2', 'id2', 'id2', 'id3'])
        synthetic_data = pd.Series(['id4', 'id4', 'id5', 'id6', 'id6', 'id6'])

        # Run
        score = SequenceLengthSimilarity.compute(real_data, synthetic_data)

        # Assert
        assert score == 1

    def test_compute_low_score(self):
        """Test it for distinct distributions."""
        # Setup
        real_data = pd.Series([f'id{i}' for i in range(100)])
        synthetic_data = pd.Series(['id100'] * 100)

        # Run
        score = SequenceLengthSimilarity.compute(real_data, synthetic_data)

        # Assert
        assert score == 0
