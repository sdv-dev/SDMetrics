from unittest.mock import patch

import pandas as pd

from sdmetrics.single_column.statistical import BoundaryAdherence


class TestBoundaryAdherence:

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the number of in-bounds values in the synthetic data is returned.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        real_data = pd.Series([1.0, 2.4, 2.6, 0.8])
        synthetic_data = pd.Series([0.9, 1.8, 2.1, 5.0])

        metric = BoundaryAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert result == 0.75

    @patch('sdmetrics.single_column.statistical.boundary_adherence.SingleColumnMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.

        Input:
        - Raw score

        Output:
        - The output of the inherited ``normalize`` method.
        """
        # Setup
        metric = BoundaryAdherence()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
