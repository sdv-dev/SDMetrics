from unittest.mock import Mock

import pandas as pd

from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.single_table._properties import DataValidity, Structure


class TestDiagnosticReport:

    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        report = DiagnosticReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert isinstance(report._properties['Data Validity'], DataValidity)
        assert isinstance(report._properties['Data Structure'], Structure)

    def test__validate_with_data_metadata_mismatch(self):
        """Test the ``_validate`` method doesn't raise an error."""
        # Setup
        base_report = DiagnosticReport()
        mock__validate_metadata_matches_data = Mock(
            side_effect=ValueError('error message')
        )
        base_report._validate_metadata_matches_data = mock__validate_metadata_matches_data

        real_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c'],
            'column3': [4, 5, 6]
        })
        synthetic_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c'],
            'column4': [4, 5, 6]
        })
        metadata = {
            'columns': {
                'column1': {'sdtype': 'numerical'},
                'column2': {'sdtype': 'categorical'},
            }
        }

        # Run
        result = base_report._validate(real_data, synthetic_data, metadata)

        # Assert
        assert result is None
