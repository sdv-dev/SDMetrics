"""Test ColumnPairTrends multi-table class."""
from sdmetrics.reports.multi_table._properties import ColumnPairTrends
from sdmetrics.reports.single_table._properties import (
    ColumnPairTrends as SingleTableColumnPairTrends)


def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    column_pair_trends = ColumnPairTrends()

    # Assert
    assert column_pair_trends._properties == {}
    assert column_pair_trends._single_table_property == SingleTableColumnPairTrends


def test__get_num_iterations():
    """Test the ``_get_num_iterations`` method."""
    # Setup
    metadata = {
        'tables': {
            'Table_1': {
                'columns': {
                    'col1': {},
                    'col2': {},
                },
            },
            'Table_2': {
                'columns': {
                    'col3': {},
                    'col4': {},
                    'col5': {},
                },
            },
        }
    }
    column_pair_trends = ColumnPairTrends()

    # Run
    num_iterations = column_pair_trends._get_num_iterations(metadata)

    # Assert
    assert num_iterations == 4
