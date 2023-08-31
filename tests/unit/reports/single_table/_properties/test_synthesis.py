from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from sdmetrics.reports.single_table._properties.synthesis import Synthesis


class TestSynthesis:

    @patch('sdmetrics.reports.single_table._properties.synthesis.'
           'NewRowSynthesis.compute_breakdown')
    def test__generate_details(self, newrowsynthesis_mock):
        """Test the ``_generate_details`` method.

        If the synthetic data is larger than 10000 rows, then the synthetic sample size
        should be 10000. Otherwise, the synthetic sample size should be the size of the
        synthetic data.
        """
        # Setup
        real_data = Mock()
        synthetic_data = [1] * 4
        synthetic_data_20000 = [1] * 20000
        metadata = Mock()

        newrowsynthesis_mock.return_value = {
            'score': 0.25,
            'num_matched_rows': 3,
            'num_new_rows': 1,
        }

        # Run
        synthesis_property = Synthesis()
        details = synthesis_property._generate_details(real_data, synthetic_data_20000, metadata)
        details = synthesis_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls = [
            call(
                real_data=real_data,
                synthetic_data=synthetic_data_20000,
                metadata=metadata,
                synthetic_sample_size=10000
            ),
            call(
                real_data=real_data,
                synthetic_data=synthetic_data,
                metadata=metadata,
                synthetic_sample_size=4
            )
        ]

        newrowsynthesis_mock.assert_has_calls(expected_calls)

        expected__details = pd.DataFrame({
            'Metric': 'NewRowSynthesis',
            'Score': 0.25,
            'Num Matched Rows': 3,
            'Num New Rows': 1,
        }, index=[0])

        pd.testing.assert_frame_equal(details, expected__details)

    @patch('sdmetrics.reports.single_table._properties.synthesis.'
           'NewRowSynthesis.compute_breakdown')
    def test__generate_details_error(self, newrowsynthesis_mock):
        """Test the ``_generate_details`` method when the metric raises an error."""
        # Setup
        newrowsynthesis_mock.side_effect = ValueError('Mock Error')
        real_data = Mock()
        synthetic_data = [1] * 4
        metadata = Mock()

        # Run
        synthesis_property = Synthesis()
        details = synthesis_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        newrowsynthesis_mock.assert_called_once_with(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            synthetic_sample_size=4
        )

        expected_details = pd.DataFrame({
            'Metric': 'NewRowSynthesis',
            'Score': np.nan,
            'Num Matched Rows': np.nan,
            'Num New Rows': np.nan,
            'Error': 'ValueError: Mock Error'
        }, index=[0])

        pd.testing.assert_frame_equal(details, expected_details)

    @patch('sdmetrics.reports.single_table._properties.synthesis.px')
    def test_get_visualization(self, mock_px):
        """Test the ``get_visualization`` method."""
        # Setup
        synthesis_property = Synthesis()
        synthesis_property.details = pd.DataFrame({
            'Metric': 'NewRowSynthesis',
            'Score': 0.25,
            'Num Matched Rows': 3,
            'Num New Rows': 1,
        }, index=[0])

        mock_pie = Mock()
        mock_px.pie.return_value = mock_pie

        # Run
        synthesis_property.get_visualization()

        # Assert
        mock_px.pie.assert_called_once_with(
            values=[3, 1],
            names=['Exact Matches', 'Novel Rows'],
            color=['Exact Matches', 'Novel Rows'],
            color_discrete_map={
                'Exact Matches': '#F16141',
                'Novel Rows': '#36B37E'
            },
            hole=0.4,
            title='Data Diagnostic: Synthesis (Score=0.25)'
        )

        mock_pie.update_traces.assert_called_once_with(
            hovertemplate='<b>%{label}</b><br>%{value} rows'
        )
