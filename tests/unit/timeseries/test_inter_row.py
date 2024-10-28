import pandas as pd

from sdmetrics.timeseries.inter_row import InterRowMSAS


class TestInterRowMSAS:
    def test_compute_identical_sequences(self):
        """Test it returns 1 when real and synthetic data are identical."""
        # Setup
        real_keys = pd.Series(['id1', 'id1', 'id1', 'id2', 'id2', 'id2'])
        real_values = pd.Series([1, 2, 3, 4, 5, 6])
        synthetic_keys = pd.Series(['id1', 'id1', 'id1', 'id2', 'id2', 'id2'])
        synthetic_values = pd.Series([1, 2, 3, 4, 5, 6])

        # Run
        score = InterRowMSAS.compute(
            real_data=(real_keys, real_values), synthetic_data=(synthetic_keys, synthetic_values)
        )

        # Assert
        assert score == 1

    def test_compute_different_sequences(self):
        """Test it for distinct distributions."""
        # Setup
        real_keys = pd.Series(['id1', 'id1', 'id1', 'id2', 'id2', 'id2'])
        real_values = pd.Series([1, 2, 3, 4, 5, 6])
        synthetic_keys = pd.Series(['id1', 'id1', 'id1', 'id2', 'id2', 'id2'])
        synthetic_values = pd.Series([1, 3, 5, 2, 4, 6])

        # Run
        score = InterRowMSAS.compute(
            real_data=(real_keys, real_values), synthetic_data=(synthetic_keys, synthetic_values)
        )

        # Assert
        assert score == 0

    def test_compute_with_log(self):
        """Test it with logarithmic transformation."""
        # Setup
        real_keys = pd.Series(['id1', 'id1', 'id1', 'id2', 'id2', 'id2'])
        real_values = pd.Series([1, 2, 4, 8, 16, 32])
        synthetic_keys = pd.Series(['id1', 'id1', 'id1', 'id2', 'id2', 'id2'])
        synthetic_values = pd.Series([1, 2, 4, 8, 16, 32])

        # Run
        score = InterRowMSAS.compute(
            real_data=(real_keys, real_values),
            synthetic_data=(synthetic_keys, synthetic_values),
            apply_log=True,
        )

        # Assert
        assert score == 1

    def test_compute_different_n_rows_diff(self):
        """Test it with different n_rows_diff."""
        # Setup
        real_keys = pd.Series(['id1'] * 10 + ['id2'] * 10)
        real_values = pd.Series(list(range(10)) + list(range(10)))
        synthetic_keys = pd.Series(['id1'] * 10 + ['id2'] * 10)
        synthetic_values = pd.Series(list(range(10)) + list(range(10)))

        # Run
        score = InterRowMSAS.compute(
            real_data=(real_keys, real_values),
            synthetic_data=(synthetic_keys, synthetic_values),
            n_rows_diff=3,
        )

        # Assert
        assert score == 1
