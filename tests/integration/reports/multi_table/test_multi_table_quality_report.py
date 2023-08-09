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
    expected = pd.DataFrame({
        'Table': ['table1', 'table1'],
        'Column': ['col2', 'col3'],
        'Metric': ['TVComplement', 'TVComplement'],
        'Score': [.75, .75]
    })
    pd.testing.assert_frame_equal(details[0], expected)

    # Assert Column Pair Trends details
    expected = pd.DataFrame({
        'Table': ['table1'],
        'Column 1': ['col2'],
        'Column 2': ['col3'],
        'Metric': ['ContingencySimilarity'],
        'Score': [.25],
        'Real Correlation': [np.nan],
        'Synthetic Correlation': [np.nan],
    })
    pd.testing.assert_frame_equal(details[1], expected)

    # Assert Cardinality details, with and without table name
    # In this case they are the same because the only existing row contains the table_name
    expected = pd.DataFrame({
        'Child Table': ['table1'],
        'Parent Table': ['table2'],
        'Metric': ['CardinalityShapeSimilariy'],
        'Score': [.75]
    })
    pd.testing.assert_frame_equal(details[2], expected)
    pd.testing.assert_frame_equal(details[5], expected)

    # Assert Column Shapes details without table_name
    expected = pd.DataFrame({
        'Table': ['table1', 'table1', 'table2', 'table2', 'table2'],
        'Column': ['col2', 'col3', 'col4', 'col5', 'col7'],
        'Metric': ['TVComplement', 'TVComplement', 'KSComplement', 'KSComplement', 'KSComplement'],
        'Score': [.75, .75, .75, .75, 1]
    })

    # Assert Column Pair Trends details without table_name
    expected = pd.DataFrame({
        'Table': ['table1', 'table2', 'table2', 'table2'],
        'Column 1': ['col2', 'col4', 'col4', 'col5'],
        'Column 2': ['col3', 'col5', 'col7', 'col7'],
        'Metric': [
            'ContingencySimilarity',
            'CorrelationSimilarity',
            'CorrelationSimilarity',
            'CorrelationSimilarity'
        ],
        'Score': [.25, 0.9901306731066666, 0.9853027960145061, 0.9678805694257717],
        'Real Correlation': [np.nan, 0.946664, 0.966247, 0.862622],
        'Synthetic Correlation': [np.nan, 0.926925, 0.936853, 0.798384],
    })
    pd.testing.assert_frame_equal(details[4], expected)


def test_correlation_similarity_constant_real_data():
    """Error out when CorrelationSimilarity is used with a constant pair of columns."""
    # Setup
    data = {
        'table1': pd.DataFrame({'id': [0, 1, 2, 3], 'col': [1, 1, 1, 1], 'col2': [1, 1, 1, 1]}),
        'table2': pd.DataFrame({'id': [0, 1, 2, 3], 'col': [1, 1, 1, 1], 'col2': [1, 1, 1, 1]}),
    }
    metadata = {
        'tables': {
            'table1': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'col': {'sdtype': 'numerical'},
                    'col2': {'sdtype': 'numerical'},
                },
            },
            'table2': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'col': {'sdtype': 'numerical'},
                    'col2': {'sdtype': 'numerical'},
                },
            }
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'parent_primary_key': 'id',
                'child_table_name': 'table2',
                'child_foreign_key': 'id'
            }
        ]
    }
    report = QualityReport()

    # Run and Assert
    report.generate(data, data, metadata)
    details = report.get_details(property_name='Column Pair Trends')

    # Assert
    error_msg1 = details['Error'][0]
    assert error_msg1 == (
        "ConstantInputError: The real data in columns 'col, col2' "
        'contains a constant value. Correlation is undefined for constant data.'
    )

    error_msg1 = details['Error'][1]
    assert error_msg1 == (
        "ConstantInputError: The real data in columns 'col, col2' "
        'contains a constant value. Correlation is undefined for constant data.'
    )
