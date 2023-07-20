from unittest.mock import Mock, call, patch

import pandas as pd

from sdmetrics.reports.single_table._properties.synthesis import Synthesis


class TestSynthesis:

    @patch('sdmetrics.reports.single_table._properties.synthesis.'
           'NewRowSynthesis.compute_breakdown')
    def test__generate_details(self, newrowsynthesis_mock):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': [False, True, True, False],
            'col3': ['a', 'b', 'c', 'd'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 7, 3, 4],
            'col2': [False, True, True, False],
            'col3': ['a', 'b', 'c', 'd'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime'}
            }
        }

        newrowsynthesis_mock.return_value = {
            'score': 0.25,
            'num_matched_rows': 3,
            'num_new_rows': 1,
        }
        # Run
        synthesis_property = Synthesis()
        details = synthesis_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls = [
            call(real_data, synthetic_data, synthetic_sample_size=4)
        ]

        newrowsynthesis_mock.assert_has_calls(expected_calls)

        expected__details = pd.DataFrame({
            'Metric': 'NewRowSynthesis',
            'Score': 0.25,
            'Num Matched Rows': 3,
            'Num New Rows': 1,
        }, index=[0])

        pd.testing.assert_frame_equal(details, expected__details)

    @patch('sdmetrics.reports.single_table._properties.synthesis.px')
    def test_get_visualization(self, mock_px):
        """Test the ``get_visualization`` method."""
        # Setup
        synthesis_property = Synthesis()
        synthesis_property._details = pd.DataFrame({
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
