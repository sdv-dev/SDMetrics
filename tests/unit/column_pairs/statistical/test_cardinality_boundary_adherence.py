
import pandas as pd

from sdmetrics.column_pairs.statistical import CardinalityBoundaryAdherence


class TestCardinalityBoundaryAdherence:

    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method."""
        # Setup
        real_parent_keys = pd.Series([1, 2, 3, 4, 5])
        real_foreign_keys = pd.Series([1, 1, 2, 3, 4, 5, 5])
        real_data = (real_parent_keys, real_foreign_keys)
        synthetic_parent_keys = pd.Series([1, 2, 3, 4, 5])
        synthetic_foreign_keys = pd.Series([2, 2, 2, 3, 4, 5])
        synthetic_data = (synthetic_parent_keys, synthetic_foreign_keys)

        metric = CardinalityBoundaryAdherence()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        assert result == {'score': 0.6}

    def test_compute(self):
        """Test the ``compute`` method."""
        # Setup
        real_parent_keys = pd.Series([1, 2, 3, 4, 5])
        real_foreign_keys = pd.Series([1, 1, 2, 3, 4, 5, 5])
        real_data = (real_parent_keys, real_foreign_keys)
        synthetic_parent_keys = pd.Series([1, 2, 3, 4, 5])
        synthetic_foreign_keys = pd.Series([2, 2, 2, 3, 4, 5])
        synthetic_data = (synthetic_parent_keys, synthetic_foreign_keys)

        metric = CardinalityBoundaryAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert result == 0.6
