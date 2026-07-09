import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sdmetrics.reports.base_unified_report import BaseUnifiedReport


class TestBaseUnifiedReport:
    def test__validate_data_format(self):
        """Test the ``_validate_data_format`` method."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({'column1': [1, 2, 3]}),
            'table2': pd.DataFrame({'column2': [4, 5, 6]}),
        }
        synthetic_data = {
            'table1': pd.DataFrame({'column1': [1, 2, 3]}),
            'table2': pd.DataFrame({'column2': [4, 5, 6]}),
        }

        # Run and Assert
        base_report = BaseUnifiedReport()
        base_report._validate_data_format(real_data, synthetic_data)

    def test__validate_data_format_with_invalid_real_data(self):
        """Test the ``_validate_data_format`` method with invalid real data."""
        # Setup
        base_report = BaseUnifiedReport()
        real_data = pd.DataFrame({'column1': [1, 2, 3]})
        synthetic_data = {
            'table1': pd.DataFrame({'column1': [1, 2, 3]}),
        }

        # Run and Assert
        expected_message = re.escape(
            'Please pass in a dictionary mapping tables to dataframes for real_data.'
        )
        with pytest.raises(ValueError, match=expected_message):
            base_report._validate_data_format(real_data, synthetic_data)

    def test__validate_data_format_with_invalid_synthetic_data(self):
        """Test the ``_validate_data_format`` method with invalid synthetic data."""
        # Setup
        base_report = BaseUnifiedReport()
        real_data = {
            'table1': pd.DataFrame({'column1': [1, 2, 3]}),
        }
        synthetic_data = pd.DataFrame({'column1': [1, 2, 3]})

        # Run and Assert
        expected_message = re.escape(
            'Please pass in a dictionary mapping tables to dataframes for synthetic_data.'
        )
        with pytest.raises(ValueError, match=expected_message):
            base_report._validate_data_format(real_data, synthetic_data)

    def test__validate_data_format_with_invalid_table(self):
        """Test the ``_validate_data_format`` method with invalid table data."""
        # Setup
        base_report = BaseUnifiedReport()
        real_data = {
            'table1': pd.DataFrame({'column1': [1, 2, 3]}),
        }
        synthetic_data = {
            'table1': [1, 2, 3],
        }

        # Run and Assert
        expected_message = (
            'BaseUnifiedReport expects real_data and synthetic_data to both be '
            'pandas.DataFrame, or both be dictionaries mapping table names to '
            'pandas.DataFrame. Received real_data=dict and synthetic_data=dict.'
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            base_report._validate_data_format(real_data, synthetic_data)

    def test__validate_data_format_with_invalid_real_table(self):
        """Test the ``_validate_data_format`` method with invalid real table data."""
        # Setup
        base_report = BaseUnifiedReport()
        real_data = {
            'table1': [1, 2, 3],
        }
        synthetic_data = {
            'table1': pd.DataFrame({'column1': [1, 2, 3]}),
        }

        # Run and Assert
        expected_message = re.escape(
            'BaseUnifiedReport expects real_data and synthetic_data to both be '
            'pandas.DataFrame, or both be dictionaries mapping table names to '
            'pandas.DataFrame. Received real_data=dict and synthetic_data=dict.'
        )
        with pytest.raises(ValueError, match=expected_message):
            base_report._validate_data_format(real_data, synthetic_data)

    def test__validate_single_table(self):
        """Test the ``_validate`` method with single-table data."""
        # Setup
        base_report = BaseUnifiedReport()

        real_data = {
            'table1': pd.DataFrame({
                'column1': [1, 2, 3],
                'column2': ['a', 'b', 'c'],
            }),
        }
        synthetic_data = {
            'table1': pd.DataFrame({
                'column1': [1, 2, 3],
                'column2': ['a', 'b', 'c'],
            }),
        }
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'column1': {'sdtype': 'numerical'},
                        'column2': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [],
        }

        # Run
        base_report._validate(real_data, synthetic_data, metadata)

        # Assert
        assert base_report.table_names == ['table1']

    def test__validate_multi_table(self):
        """Test the ``_validate`` method with multi-table data."""
        # Setup
        base_report = BaseUnifiedReport()

        real_data = {
            'table1': pd.DataFrame({'column1': [1, 2, 3]}),
            'table2': pd.DataFrame({'column2': ['a', 'b', 'c']}),
        }
        synthetic_data = {
            'table1': pd.DataFrame({'column1': [1, 2, 3]}),
            'table2': pd.DataFrame({'column2': ['a', 'b', 'c']}),
        }
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'column1': {'sdtype': 'numerical'},
                    },
                },
                'table2': {
                    'columns': {
                        'column2': {'sdtype': 'categorical'},
                    },
                },
            },
            'relationships': [],
        }

        # Run
        base_report._validate(real_data, synthetic_data, metadata)

        # Assert
        assert base_report.table_names == ['table1', 'table2']

    def test__get_skipped_properties_single_table(self):
        """Test single-table unified data returns relationship properties."""
        # Setup
        base_report = BaseUnifiedReport()
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'column1': {'sdtype': 'numerical'},
                    },
                },
            },
            'relationships': [],
        }

        # Run
        skipped_properties = base_report._get_skipped_properties(metadata)

        # Assert
        assert skipped_properties == {'Relationship Validity', 'Cardinality', 'Intertable Trends'}

    def test__get_skipped_properties_multi_table(self):
        """Test multi-table unified data does not skip relationship properties."""
        # Setup
        base_report = BaseUnifiedReport()
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'column1': {'sdtype': 'numerical'},
                    },
                },
                'table2': {
                    'columns': {
                        'column2': {'sdtype': 'numerical'},
                    },
                },
            },
            'relationships': [],
        }

        # Run
        skipped_properties = base_report._get_skipped_properties(metadata)

        # Assert
        assert skipped_properties == set()

    @patch('sdmetrics.reports.base_unified_report._validate_unified_metadata')
    def test__validate_metadata_error(self, mock__validate_metadata):
        """Test the ``_validate`` method when metadata validation fails."""
        # Setup
        error_msg = 'Invalid metadata'
        mock__validate_metadata.side_effect = ValueError(error_msg)
        base_report = BaseUnifiedReport()
        base_report._validate_data_format = Mock()
        base_report._validate_metadata_matches_data = Mock()

        real_data = pd.DataFrame({'column1': [1, 2, 3]})
        synthetic_data = pd.DataFrame({'column1': [1, 2, 3]})
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'column1': {'sdtype': 'numerical'},
                    },
                },
            },
            'relationships': [],
        }

        # Run and Assert
        with pytest.raises(ValueError, match=error_msg):
            base_report._validate(real_data, synthetic_data, metadata)

        mock__validate_metadata.assert_called_once_with(metadata)
        assert base_report.table_names == []
        base_report._validate_data_format.assert_not_called()
        base_report._validate_metadata_matches_data.assert_not_called()

    def test_get_properties_single_table_skips_relationship_properties(self):
        """Test ``get_properties`` skips relationship properties for single-table data."""
        # Setup
        base_report = BaseUnifiedReport()
        base_report._check_report_generated = Mock()
        base_report.report_info = {'num_tables': 1}

        column_shapes = Mock()
        column_shapes._compute_average.return_value = 0.8

        column_pair_trends = Mock()
        column_pair_trends._compute_average.return_value = 0.6

        relationship_validity = Mock()
        relationship_validity._compute_average.return_value = 0.4

        cardinality = Mock()
        cardinality._compute_average.return_value = 0.3

        intertable_trends = Mock()
        intertable_trends._compute_average.return_value = 0.2

        base_report._properties = {
            'Column Shapes': column_shapes,
            'Column Pair Trends': column_pair_trends,
            'Relationship Validity': relationship_validity,
            'Cardinality': cardinality,
            'Intertable Trends': intertable_trends,
        }

        expected_properties = pd.DataFrame({
            'Property': ['Column Shapes', 'Column Pair Trends'],
            'Score': [0.8, 0.6],
        })

        # Run
        properties = base_report.get_properties()

        # Assert
        base_report._check_report_generated.assert_called_once()
        pd.testing.assert_frame_equal(properties, expected_properties)

        column_shapes._compute_average.assert_called_once()
        column_pair_trends._compute_average.assert_called_once()
        relationship_validity._compute_average.assert_not_called()
        cardinality._compute_average.assert_not_called()
        intertable_trends._compute_average.assert_not_called()
