import re

import pandas as pd
import pytest

from sdmetrics.timeseries import StatisticMSAS


class TestStatisticMSAS:
    def test_compute_identical_sequences(self):
        """Test it returns 1 when real and synthetic data are identical."""
        # Setup
        real_keys = pd.Series(['id1', 'id1', 'id1', 'id2', 'id2', 'id2'])
        real_values = pd.Series([1, 2, 3, 4, 5, 6])
        synthetic_keys = pd.Series(['id3', 'id3', 'id3', 'id4', 'id4', 'id4'])
        synthetic_values = pd.Series([1, 2, 3, 4, 5, 6])

        # Run and Assert
        for statistic in ['mean', 'median', 'std', 'min', 'max']:
            score = StatisticMSAS.compute(
                real_data=(real_keys, real_values),
                synthetic_data=(synthetic_keys, synthetic_values),
                statistic=statistic,
            )
            assert score == 1

    def test_compute_different_sequences(self):
        """Test it for distinct distributions."""
        # Setup
        # Setup
        real_keys = pd.Series(['id1', 'id1', 'id1', 'id2', 'id2', 'id2'])
        real_values = pd.Series([1, 2, 3, 4, 5, 6])
        synthetic_keys = pd.Series(['id3', 'id3', 'id3', 'id4', 'id4', 'id4'])
        synthetic_values = pd.Series([10, 20, 30, 40, 50, 60])

        # Run and Assert
        for statistic in ['mean', 'median', 'std', 'min', 'max']:
            score = StatisticMSAS.compute(
                real_data=(real_keys, real_values),
                synthetic_data=(synthetic_keys, synthetic_values),
                statistic=statistic,
            )
            assert score == 0

    def test_compute_with_single_sequence(self):
        """Test it with a single sequence."""
        # Setup
        real_keys = pd.Series(['id1', 'id1', 'id1'])
        real_values = pd.Series([1, 2, 3])
        synthetic_keys = pd.Series(['id2', 'id2', 'id2'])
        synthetic_values = pd.Series([1, 2, 3])

        # Run
        score = StatisticMSAS.compute(
            real_data=(real_keys, real_values),
            synthetic_data=(synthetic_keys, synthetic_values),
            statistic='mean',
        )

        # Assert
        assert score == 1

    def test_compute_with_different_sequence_lengths(self):
        """Test it with different sequence lengths."""
        # Setup
        real_keys = pd.Series(['id1', 'id1', 'id1', 'id2', 'id2'])
        real_values = pd.Series([1, 2, 3, 4, 5])
        synthetic_keys = pd.Series(['id2', 'id2', 'id3', 'id4', 'id5'])
        synthetic_values = pd.Series([1, 2, 3, 4, 5])

        # Run
        score = StatisticMSAS.compute(
            real_data=(real_keys, real_values),
            synthetic_data=(synthetic_keys, synthetic_values),
            statistic='mean',
        )

        # Assert
        assert score == 0.75

    def test_compute_with_invalid_statistic(self):
        """Test it raises ValueError for invalid statistic."""
        # Setup
        real_keys = pd.Series(['id1', 'id1', 'id1'])
        real_values = pd.Series([1, 2, 3])
        synthetic_keys = pd.Series(['id2', 'id2', 'id2'])
        synthetic_values = pd.Series([1, 2, 3])

        # Run and Assert
        err_msg = re.escape(
            'Invalid statistic: invalid. Choose from [mean, median, std, min, max].'
        )
        with pytest.raises(ValueError, match=err_msg):
            StatisticMSAS.compute(
                real_data=(real_keys, real_values),
                synthetic_data=(synthetic_keys, synthetic_values),
                statistic='invalid',
            )
