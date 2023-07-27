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
    """Test the multi table QualityReport.

    Run all the public methods for QualityReport, and check that all the scores for
    all the properties are correct.
    """
    # Setup
    real_data, synthetic_data, metadata = load_test_data()
    report = QualityReport()

    # Run `generate`, `get_properties` and `get_score`,
    # as well as `get_visualization` and `get_details` for every property:
    # 'Column Shapes', 'Column Pair Trends', 'Cardinality'
    report.generate(real_data, synthetic_data, metadata)
    properties = report.get_properties()
    score = report.get_score()
    visualization, details = [], []
    for property_ in report._properties_instances:
        visualization.append(report.get_visualization(property_, 'table1'))
        details.append(report.get_details(property_, 'table1'))

    # Run `get_details` for every property without passing a table_name
    for property_ in report._properties_instances:
        details.append(report.get_details(property_))

    # Assert score
    assert score == 0.7190730021414969
    pd.testing.assert_frame_equal(properties, pd.DataFrame({
        'Property': ['Column Shapes', 'Column Pair Trends', 'Cardinality'],
        'Score': [0.7916666666666667, 0.615552339757824, 0.75],
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
    assert details[2] == details[5] == {('table1', 'table2'): {'score': 0.75}}

    # Assert Column Shapes details without table_name
    pd.testing.assert_frame_equal(details[3]['table1'], pd.DataFrame({
        'Column': ['col2', 'col3'],
        'Metric': ['TVComplement', 'TVComplement'],
        'Score': [.75, .75]
    }))
    pd.testing.assert_frame_equal(details[3]['table2'], pd.DataFrame({
        'Column': ['col4', 'col5', 'col7'],
        'Metric': ['KSComplement', 'KSComplement', 'KSComplement'],
        'Score': [.75, .75, 1]
    }))

    # Assert Column Pair Trends details without table_name
    pd.testing.assert_frame_equal(details[4]['table1'], pd.DataFrame({
        'Column 1': ['col2'],
        'Column 2': ['col3'],
        'Metric': ['ContingencySimilarity'],
        'Score': [.25],
        'Real Correlation': [np.nan],
        'Synthetic Correlation': [np.nan],
    }))
    pd.testing.assert_frame_equal(details[4]['table2'], pd.DataFrame({
        'Column 1': ['col4', 'col4', 'col5'],
        'Column 2': ['col5', 'col7', 'col7'],
        'Metric': ['CorrelationSimilarity', 'CorrelationSimilarity', 'CorrelationSimilarity'],
        'Score': [0.9901306731066666, 0.9853027960145061, 0.9678805694257717],
        'Real Correlation': [0.946664, 0.966247, 0.862622],
        'Synthetic Correlation': [0.926925, 0.936853, 0.798384],
    }))
