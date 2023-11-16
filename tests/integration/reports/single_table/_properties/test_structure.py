import numpy as np
import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table._properties import Structure


class TestStructure:

    def test_get_score(self):
        """Test the ``get_score`` method."""
        # Setup
        real_data, synthetic_data, metadata = load_demo('single_table')

        # Run
        synthesis_property = Structure()
        score = synthesis_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert score == 1.0

        expected_details = pd.DataFrame({
            'Metric': 'TableStructure',
            'Score': 1.0,
        }, index=[0])

        pd.testing.assert_frame_equal(synthesis_property.details, expected_details)

    def test_get_score_error(self):
        """Test the ``get_score`` method with an error.
        Give an empty synthetic data to get an error.
        """
        # Setup
        real_data, _, metadata = load_demo('single_table')

        # Run
        synthesis_property = Structure()
        score = synthesis_property.get_score(real_data.iloc[:20], [], metadata)

        # Assert
        assert pd.isna(score)

        expected_details = pd.DataFrame({
            'Metric': 'TableStructure',
            'Score': np.nan,
            'Error': "AttributeError: 'list' object has no attribute 'columns'"
        }, index=[0])

        pd.testing.assert_frame_equal(synthesis_property.details, expected_details)
