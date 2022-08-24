from unittest.mock import Mock, patch

import pandas as pd

from sdmetrics.reports.single_table import QualityReport


class TestQualityReport:

    def test_generate(self):
        """Test the ``generate`` method.

        Expect that the single-table metrics are called.

        Setup:
        - Mock the expected single-table metric compute breakdown calls.

        Input:
        - Real data.
        - Synthetic data.
        - Metadata.

        Side Effects:
        - Expect that each single table metric's ``compute_breakdown`` methods are called once.
        - Expect that the ``_overall_quality_score`` and ``_property_breakdown`` attributes
          are populated.
        """
        # Setup
        real_data = Mock()
        synthetic_data = Mock()
        ks_complement_mock = Mock()
        ks_complement_mock.__name__ = 'KSComplement'
        ks_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        tv_complement_mock = Mock()
        tv_complement_mock.__name__ = 'TVComplement'
        tv_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        corr_sim_mock = Mock()
        corr_sim_mock.__name__ = 'CorrelationSimilarity'
        corr_sim_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        cont_sim_mock = Mock()
        cont_sim_mock.__name__ = 'ContingencySimilarity'
        cont_sim_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }
        metrics_mock = {
            'Column Shapes': [ks_complement_mock, tv_complement_mock],
            'Column Pair Trends': [corr_sim_mock, cont_sim_mock],
        }

        # Run
        with patch.object(
            QualityReport,
            'METRICS',
            metrics_mock,
        ):
            report = QualityReport()
            report.generate(real_data, synthetic_data, Mock())

        # Assert
        ks_complement_mock.compute_breakdown.assert_called_once_with(real_data, synthetic_data)
        tv_complement_mock.compute_breakdown.assert_called_once_with(real_data, synthetic_data)
        corr_sim_mock.compute_breakdown.assert_called_once_with(real_data, synthetic_data)
        cont_sim_mock.compute_breakdown.assert_called_once_with(real_data, synthetic_data)
        assert report._overall_quality_score == 0.15000000000000002
        assert report._property_breakdown == {
            'Column Shapes': 0.15000000000000002,
            'Column Pair Trends': 0.15000000000000002,
        }

    def test_get_score(self):
        """Test the ``get_score`` method.

        Expect that the overall quality score is returned.

        Setup:
        - Mock the ``_overall_quality_score`` attribute.

        Input:
        - None

        Output:
        - The overall quality score.
        """
        # Setup
        report = QualityReport()
        mock_score = Mock()
        report._overall_quality_score = mock_score

        # Run
        score = report.get_score()

        # Assert
        assert score == mock_score

    def test_get_properties(self):
        """Test the ``get_details`` method.

        Expect that the property score breakdown is returned.

        Setup:
        - Mock the ``_property_breakdown`` attribute.

        Input:
        - None

        Output:
        - The metric scores for each property.
        """
        # Setup
        report = QualityReport()
        mock_property_breakdown = {'Column Shapes': 0.1, 'Column Pair Trends': 0.2}
        report._property_breakdown = mock_property_breakdown

        # Run
        properties = report.get_properties()

        # Assert
        pd.testing.assert_frame_equal(
            properties,
            pd.DataFrame({
                'Property': ['Column Shapes', 'Column Pair Trends'],
                'Score': [0.1, 0.2],
            }),
        )
