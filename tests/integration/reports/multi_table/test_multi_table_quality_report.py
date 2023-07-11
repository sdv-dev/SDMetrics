from datetime import date, datetime

import numpy as np
import pandas as pd

from sdmetrics.reports.multi_table.quality_report import QualityReport


def load_test_data():
    real_data = {
        'table1': pd.DataFrame({
            'col1': [0, 1, 2, 3],
            'col2': ['a', 'b', 'c', 'd'],
            'col3': [True, False, False, True],
        }),
        'table2': pd.DataFrame({
            'col4': [
                datetime(2020, 10, 1),
                datetime(2021, 1, 2),
                datetime(2021, 9, 12),
                datetime(2022, 10, 1),
            ],
            'col5': [date(2020, 9, 13), date(2020, 12, 1), date(2021, 1, 12), date(2022, 8, 13)],
            'col6': [0, 1, 1, 0],
            'col7': [0.1, 0.2, 0.3, 0.4],
        }),
    }

    synthetic_data = {
        'table1': pd.DataFrame({
            'col1': [0, 2, 2, 3],
            'col2': ['a', 'c', 'c', 'b'],
            'col3': [False, False, False, True],
        }),
        'table2': pd.DataFrame({
            'col4': [
                datetime(2020, 11, 4),
                datetime(2021, 2, 1),
                datetime(2021, 8, 1),
                datetime(2022, 12, 1),
            ],
            'col5': [date(2020, 10, 13), date(2020, 2, 4), date(2021, 3, 11), date(2022, 7, 23)],
            'col6': [0, 1, 1, 0],
            'col7': [0.1, 0.2, 0.3, 0.4],
        }),
    }

    metadata = {
        'tables': {
            'table1': {
                'columns': {
                    'col1': {'sdtype': 'id'},
                    'col2': {'sdtype': 'categorical'},
                    'col3': {'sdtype': 'boolean'},
                },
            },
            'table2': {
                'columns': {
                    'col4': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                    'col5': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                    'col6': {'sdtype': 'id'},
                    'col7': {'sdtype': 'numerical'},
                },
            }
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'parent_primary_key': 'col1',
                'child_table_name': 'table2',
                'child_foreign_key': 'col6'
            }
        ]
    }

    return (real_data, synthetic_data, metadata)


def test_multi_table_quality_report():
    """Test the multi table quality report."""
    # Setup
    real_data, synthetic_data, metadata = load_test_data()
    report = QualityReport()

    # Run
    report.generate(real_data, synthetic_data, metadata)
    properties = report.get_properties()
    score = report.get_score()
    visualization, details = [], []
    for property_ in report._properties_instances:
        visualization.append(report.get_visualization(property_, 'table1'))
        details.append(report.get_details(property_, 'table1'))

    details.append(report.get_details('Cardinality'))

    # Assert
    np.testing.assert_almost_equal(score, .72)
    pd.testing.assert_frame_equal(properties, pd.DataFrame({
        'Property': ['Column Shapes', 'Column Pair Trends', 'Cardinality'],
        'Score': [0.79, 0.62, 0.75],
    }))

    # Assert Column Shapes details
    pd.testing.assert_frame_equal(details[0], pd.DataFrame({
        'Column': ['col2', 'col3'],
        'Metric': ['TVComplement', 'TVComplement'],
        'Score': [.75, .75]
    }))

    # Assert Column Pair Trends details
    pd.testing.assert_frame_equal(details[1], pd.DataFrame({
        'Column 1': ['col2'],
        'Column 2': ['col3'],
        'Metric': ['ContingencySimilarity'],
        'Score': [.25],
        'Real Correlation': [np.nan],
        'Synthetic Correlation': [np.nan],
    }))

    # Assert Cardinality details
    assert details[2] == details[3] == {('table1', 'table2'): {'score': 0.75}}
