import time
from datetime import date, datetime

import numpy as np
import pandas as pd

from sdmetrics.demos import load_demo
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
    generate_start_time = time.time()
    report.generate(real_data, synthetic_data, metadata)
    generate_end_time = time.time()
    properties = report.get_properties()
    property_names = list(properties['Property'])
    score = report.get_score()
    visualization, details = [], []
    for property_ in property_names:
        visualization.append(report.get_visualization(property_, 'table1'))
        details.append(report.get_details(property_, 'table1'))

    # Run `get_details` for every property without passing a table_name
    for property_ in property_names:
        details.append(report.get_details(property_))

    # Assert score
    assert score == 0.649582127409184
    pd.testing.assert_frame_equal(properties, pd.DataFrame({
        'Property': ['Column Shapes', 'Column Pair Trends', 'Cardinality', 'Intertable Trends'],
        'Score': [0.8, 0.7983285096367361, 0.75, 0.25],
    }))

    # Assert Column Shapes details
    expected_df_0 = pd.DataFrame({
        'Table': ['table1', 'table1'],
        'Column': ['col2', 'col3'],
        'Metric': ['TVComplement', 'TVComplement'],
        'Score': [.75, .75]
    })
    pd.testing.assert_frame_equal(details[0], expected_df_0)

    # Assert Column Pair Trends details
    expected_df_1 = pd.DataFrame({
        'Table': ['table1'],
        'Column 1': ['col2'],
        'Column 2': ['col3'],
        'Metric': ['ContingencySimilarity'],
        'Score': [.25],
        'Real Correlation': [np.nan],
        'Synthetic Correlation': [np.nan],
    })
    pd.testing.assert_frame_equal(details[1], expected_df_1)

    # Assert Cardinality details
    expected_df_2 = pd.DataFrame({
        'Child Table': ['table2'],
        'Parent Table': ['table1'],
        'Metric': ['CardinalityShapeSimilarity'],
        'Score': [0.75],
    })
    pd.testing.assert_frame_equal(details[2], expected_df_2)
    pd.testing.assert_frame_equal(details[6], expected_df_2)

    # Assert Intertable Trends details
    expected_df_3 = pd.DataFrame({
        'Parent Table': ['table1', 'table1', 'table1', 'table1', 'table1', 'table1'],
        'Child Table': ['table2', 'table2', 'table2', 'table2', 'table2', 'table2'],
        'Foreign Key': ['col6', 'col6', 'col6', 'col6', 'col6', 'col6'],
        'Column 1': ['col2', 'col2', 'col2', 'col3', 'col3', 'col3'],
        'Column 2': ['col4', 'col5', 'col7', 'col4', 'col5', 'col7'],
        'Metric': [
            'ContingencySimilarity', 'ContingencySimilarity', 'ContingencySimilarity',
            'ContingencySimilarity', 'ContingencySimilarity', 'ContingencySimilarity'
        ],
        'Score': [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        'Real Correlation': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'Synthetic Correlation': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    pd.testing.assert_frame_equal(details[3], expected_df_3)
    pd.testing.assert_frame_equal(details[7], expected_df_3)

    # Assert Column Shapes details without table_name
    expected_df_3 = pd.DataFrame({
        'Table': ['table1', 'table1', 'table2', 'table2', 'table2'],
        'Column': ['col2', 'col3', 'col4', 'col5', 'col7'],
        'Metric': ['TVComplement', 'TVComplement', 'KSComplement', 'KSComplement', 'KSComplement'],
        'Score': [0.75, 0.75, 0.75, 0.75, 1.0]
    })
    pd.testing.assert_frame_equal(details[4], expected_df_3)

    # Assert Column Pair Trends details without table_name
    expected_df_4 = pd.DataFrame({
        'Table': ['table1', 'table2', 'table2', 'table2'],
        'Column 1': ['col2', 'col4', 'col4', 'col5'],
        'Column 2': ['col3', 'col5', 'col7', 'col7'],
        'Metric': [
            'ContingencySimilarity', 'CorrelationSimilarity', 'CorrelationSimilarity',
            'CorrelationSimilarity'
        ],
        'Score': [0.25, 0.9901306731066666, 0.9853027960145061, 0.9678805694257717],
        'Real Correlation': [np.nan, 0.946664, 0.966247, 0.862622],
        'Synthetic Correlation': [np.nan, 0.926925, 0.936853, 0.798384],
    })
    pd.testing.assert_frame_equal(details[5], expected_df_4)

    # Assert report info saved
    report_info = report.get_info()
    assert report_info == report.report_info

    expected_info_keys = {
        'report_type', 'generated_date', 'sdmetrics_version', 'num_tables', 'num_rows_real_data',
        'num_rows_synthetic_data', 'generation_time'
    }
    assert report_info.keys() == expected_info_keys
    assert report_info['report_type'] == 'QualityReport'
    assert report_info['num_tables'] == 2
    assert report_info['num_rows_real_data'] == {'table1': 4, 'table2': 4}
    assert report_info['num_rows_synthetic_data'] == {'table1': 4, 'table2': 4}
    assert report_info['generation_time'] <= generate_end_time - generate_start_time


def test_quality_report_end_to_end():
    """Test the multi table QualityReport end to end."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    report = QualityReport()

    # Run
    report.generate(real_data, synthetic_data, metadata)
    score = report.get_score()
    properties = report.get_properties()
    info = report.get_info()

    # Assert
    expected_properties = pd.DataFrame({
        'Property': ['Column Shapes', 'Column Pair Trends', 'Cardinality', 'Intertable Trends'],
        'Score': [0.7922619047619048, 0.4249665433225429, 0.8, 0.48240740740740734],
    })
    assert score == 0.6249089638729638
    pd.testing.assert_frame_equal(properties, expected_properties)
    expected_info_keys = {
        'report_type', 'generated_date', 'sdmetrics_version', 'num_tables', 'num_rows_real_data',
        'num_rows_synthetic_data', 'generation_time'
    }
    assert info.keys() == expected_info_keys
    assert info['report_type'] == 'QualityReport'
    assert info['num_tables'] == 3
    assert info['num_rows_real_data'] == {'sessions': 10, 'users': 10, 'transactions': 10}
    assert info['num_rows_synthetic_data'] == {'sessions': 9, 'users': 10, 'transactions': 10}


def test_quality_report_with_object_datetimes():
    """Test the multi table QualityReport with object datetimes."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    for table, table_meta in metadata['tables'].items():
        for column, column_meta in table_meta['columns'].items():
            if column_meta['sdtype'] == 'datetime':
                dt_format = column_meta['datetime_format']
                real_data[table][column] = real_data[table][column].dt.strftime(dt_format)

    report = QualityReport()

    # Run
    report.generate(real_data, synthetic_data, metadata)
    score = report.get_score()
    properties = report.get_properties()

    # Assert
    expected_properties = pd.DataFrame({
        'Property': ['Column Shapes', 'Column Pair Trends', 'Cardinality', 'Intertable Trends'],
        'Score': [0.7922619047619048, 0.4249665433225429, 0.8, 0.48240740740740734],
    })
    assert score == 0.6249089638729638
    pd.testing.assert_frame_equal(properties, expected_properties)


def test_quality_report_with_errors():
    """Test the multi table QualityReport with errors when computing metrics."""
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    real_data['users']['age'].iloc[0] = 'error_1'
    real_data['transactions']['timestamp'].iloc[0] = 'error_2'
    real_data['transactions']['amount'].iloc[0] = 'error_3'

    report = QualityReport()

    # Run
    report.generate(real_data, synthetic_data, metadata)
    score = report.get_score()
    properties = report.get_properties()
    details_column_shapes = report.get_details('Column Shapes')

    # Assert
    expected_properties = pd.DataFrame({
        'Property': ['Column Shapes', 'Column Pair Trends', 'Cardinality', 'Intertable Trends'],
        'Score': [0.8276190476190475, 0.5666666666666667, 0.8, 0.6092592592592593]
    })
    expected_details = pd.DataFrame({
        'Table': [
            'users', 'users', 'users', 'sessions', 'sessions', 'transactions',
            'transactions', 'transactions'
        ],
        'Column': [
            'country', 'gender', 'age', 'device', 'os', 'timestamp', 'amount', 'approved'
        ],
        'Metric': [
            'TVComplement', 'TVComplement', 'KSComplement', 'TVComplement', 'TVComplement',
            'KSComplement', 'KSComplement', 'TVComplement'
        ],
        'Score': [
            0.7, 0.9714285714285714, np.nan, 0.9333333333333333, 0.7333333333333334,
            np.nan, np.nan, 0.8
        ],
        'Error': [
            None, None, "TypeError: '<' not supported between instances of 'int' and 'str'",
            np.nan, np.nan,
            "TypeError: '<' not supported between instances of 'Timestamp' and 'str'",
            "TypeError: '<' not supported between instances of 'float' and 'str'",
            None
        ]
    })
    assert score == 0.7008862433862433
    pd.testing.assert_frame_equal(properties, expected_properties)
    pd.testing.assert_frame_equal(details_column_shapes, expected_details)


def test_quality_report_with_no_relationships():
    # Setup
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')

    del metadata['relationships']
    report = QualityReport()

    # Run
    report.generate(real_data, synthetic_data, metadata, verbose=True)
    score = report.get_score()

    # Assert
    expected_properties = pd.DataFrame({
        'Property': ['Column Shapes', 'Column Pair Trends', 'Cardinality', 'Intertable Trends'],
        'Score': [0.792262, 0.424967, np.nan, np.nan]
    })
    properties = report.get_properties()
    pd.testing.assert_frame_equal(properties, expected_properties)
    assert score == 0.6086142240422239
