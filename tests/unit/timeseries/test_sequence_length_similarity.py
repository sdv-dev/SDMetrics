import pandas as pd

from sdmetrics.timeseries.sequence_length_similarity import SequenceLengthSimilarity


class TestSequenceLengthSimilarity:
    def test_compute_one(self):
        """Test it returns 1 when real and synthetic data have the same distribution."""
        # Setup
        real_data = pd.Series(['id1', 'id1', 'id2', 'id2', 'id2', 'id3'])
        synthetic_data = pd.Series(['id1', 'id1', 'id2', 'id3', 'id3', 'id3'])

        # Run
        score = SequenceLengthSimilarity.compute(real_data, synthetic_data)

        # Assert
        assert score == 1

    def test_compute_low_score(self):
        """Test it for distinct distributions."""
        # Setup
        real_data = pd.Series(['id1', 'id1', 'id2'])
        synthetic_data = pd.Series(['id1', 'id2', 'id3'])

        # Run
        score = SequenceLengthSimilarity.compute(real_data, synthetic_data)

        # Assert
        assert score == 0.5
