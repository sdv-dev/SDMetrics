from unittest.mock import call, patch, Mock
import pandas as pd
import pytest

from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes

class TestQualityReport:

    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        report = QualityReport()

        # Assert
        assert report._overall_quality_score is None
        assert not report.is_generated
        assert isinstance(report._properties['Column Shapes'], ColumnShapes)
        assert isinstance(report._properties['Column Pair Trends'], ColumnPairTrends)

    @patch('sys.stdout.write')
    def test__print_results(self, mock_write):
        """Test the ``_print_results`` method."""
        # Setup
        quality_report = QualityReport()
        quality_report._overall_quality_score = 0.5
        quality_report._properties = {
            'Column Shapes': Mock(_compute_average=Mock(return_value=0.6)),
            'Column Pair Trends': Mock(_compute_average=Mock(return_value=0.4))
        }

        # Run
        quality_report._print_results()

        # Assert
        calls = [
            call('\nOverall Quality Score: 50.0%\n\n'),
            call('Properties:\n'),
            call('- Column Shapes: 60.0%\n'),
            call('- Column Pair Trends: 40.0%\n'),
        ]
        mock_write.assert_has_calls(calls, any_order=True)

    def test_generate(self):
        """Test the ``generate`` method."""
        # Setup
        quality_report = QualityReport()
        mock_columnshape_get_score = Mock(return_value=1.0)
        mock_cpt_get_score = Mock(return_value=0.5)
        quality_report._properties['Column Shapes'] = Mock()
        quality_report._properties['Column Shapes'].get_score = mock_columnshape_get_score
        quality_report._properties['Column Pair Trends'] = Mock()
        quality_report._properties['Column Pair Trends'].get_score = mock_cpt_get_score

        real_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        synthetic_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        metadata = {
            'column': {
                'column1': { 'sdtypes': 'numerical'},
                'column2': { 'sdtypes': 'categorical'}
            }
        }

        # Run
        quality_report.generate(real_data, synthetic_data, metadata, verbose=False)

        # Assert
        quality_report._properties['Column Shapes'].get_score.assert_called_with(
            real_data, synthetic_data, metadata
        )
        quality_report._properties['Column Pair Trends'].get_score.assert_called_with(
            real_data, synthetic_data, metadata
        )

    def test__validate_property_generation(self):
        """Test the ``_validate_property_generation`` method."""
        # Setup
        quality_report = QualityReport()
        wrong_property_name = 'Wrong Property Name'
        quality_report.is_generated = False

        # Run and Assert
        expected_message_1 = 'Quality report must be generated before getting details. Call `generate` first.'
        with pytest.raises(ValueError, match=expected_message_1):
            quality_report._validate_property_generation('Column Shapes')

        expected_message_2 = f'Property `{wrong_property_name}` does not exist.'
        with pytest.raises(ValueError, match=expected_message_2):




    def test_get_visualization(self):
        """Test the ``get_visualization`` method."""
        # Setup
        quality_report = QualityReport()
        quality_report._properties['Column Shapes'] = Mock()
        quality_report._properties['Column Pair Trends'] = Mock()
        quality_report.is_generated = True

        # Run
        quality_report.get_visualization('Column Shapes')
        quality_report.get_visualization('Column Pair Trends')

        # Assert
        quality_report._properties['Column Shapes'].get_visualization.assert_called_once()
        quality_report._properties['Column Pair Trends'].get_visualization.assert_called_once()


    def test_get_details(self):
        """Test the ``get_details`` method."""
        # Setup
        quality_report = QualityReport()
        quality_report._properties['Column Shapes'] = Mock()
        quality_report.is_generated = True

        # Run
        quality_report.get_details('Column Shapes')

        # Assert
        quality_report._properties['Column Shapes']._details.copy.assert_called_once()