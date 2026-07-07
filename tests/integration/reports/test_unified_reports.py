from datetime import date, datetime

import numpy as np
import pandas as pd

from sdmetrics.demos import load_multi_table_demo, load_single_table_demo
from sdmetrics.reports import DiagnosticReport, QualityReport


def _set_thresholds_zero(report):
    report.real_correlation_threshold = 0
    report.real_association_threshold = 0


def _assert_report_info(report, report_type, expected_real_rows, expected_synthetic_rows):
    report_info = report.get_info()
    expected_info_keys = {
        'report_type',
        'generated_date',
        'sdmetrics_version',
        'num_tables',
        'num_rows_real_data',
        'num_rows_synthetic_data',
        'generation_time',
    }

    assert report_info == report.report_info
    assert report_info.keys() == expected_info_keys
    assert report_info['report_type'] == report_type
    assert report_info['num_tables'] == len(expected_real_rows)
    assert report_info['num_rows_real_data'] == expected_real_rows
    assert report_info['num_rows_synthetic_data'] == expected_synthetic_rows
    assert report_info['generation_time'] >= 0


def _load_quality_report_data():
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
            },
        },
        'relationships': [
            {
                'parent_table_name': 'table1',
                'parent_primary_key': 'col1',
                'child_table_name': 'table2',
                'child_foreign_key': 'col6',
            }
        ],
    }

    return real_data, synthetic_data, metadata


def test_unified_diagnostic_report_single_table():
    # Setup
    real_data, synthetic_data, metadata = load_single_table_demo()

    # Run
    report = DiagnosticReport()
    report.generate(real_data, synthetic_data, metadata, verbose=False)

    # Assert
    expected_properties = pd.DataFrame({
        'Property': ['Data Validity', 'Data Structure'],
        'Score': [1.0, 1.0],
    })
    expected_details_data_validity = pd.DataFrame({
        'Table': ['student_placements'] * 17,
        'Column': [
            'start_date',
            'end_date',
            'salary',
            'duration',
            'student_id',
            'high_perc',
            'high_spec',
            'mba_spec',
            'second_perc',
            'gender',
            'degree_perc',
            'placed',
            'experience_years',
            'employability_perc',
            'mba_perc',
            'work_experience',
            'degree_type',
        ],
        'Metric': [
            'BoundaryAdherence',
            'BoundaryAdherence',
            'BoundaryAdherence',
            'BoundaryAdherence',
            'KeyUniqueness',
            'BoundaryAdherence',
            'CategoryAdherence',
            'CategoryAdherence',
            'BoundaryAdherence',
            'CategoryAdherence',
            'BoundaryAdherence',
            'CategoryAdherence',
            'BoundaryAdherence',
            'BoundaryAdherence',
            'BoundaryAdherence',
            'CategoryAdherence',
            'CategoryAdherence',
        ],
        'Score': [1.0] * 17,
    })
    expected_details_data_structure = pd.DataFrame({
        'Table': ['student_placements'],
        'Metric': ['TableStructure'],
        'Score': [1.0],
    })
    expected_details_relationship_validity = pd.DataFrame()

    pd.testing.assert_frame_equal(report.get_properties(), expected_properties)
    pd.testing.assert_frame_equal(
        report.get_details('Data Validity'), expected_details_data_validity
    )
    pd.testing.assert_frame_equal(
        report.get_details('Data Structure'), expected_details_data_structure
    )
    pd.testing.assert_frame_equal(
        report.get_details('Relationship Validity'),
        expected_details_relationship_validity,
        check_dtype=False,
        check_index_type=False,
    )
    assert report.get_score() == 1.0
    _assert_report_info(
        report, 'DiagnosticReport', {'student_placements': 215}, {'student_placements': 215}
    )


def test_unified_diagnostic_report_single_table_verbose_skips_relationship_validity(capsys):
    """Test verbose mode skips Relationship Validity for single-table unified data."""
    # Setup
    real_data, synthetic_data, metadata = load_single_table_demo()

    # Run
    report = DiagnosticReport()
    report.generate(real_data, synthetic_data, metadata, verbose=True)
    output = capsys.readouterr().out

    # Assert
    expected_lines = [
        'Generating report ...',
        '(1/3) Evaluating Data Validity:',
        'Data Validity Score: 100.0%',
        '(2/3) Evaluating Data Structure:',
        'Data Structure Score: 100.0%',
        '(3/3) Evaluating Relationship Validity: N/A',
        'This property does not apply to single-table data.',
        'Overall Score (Average): 100.0%',
    ]
    for line in expected_lines:
        assert line in output

    assert '0/0' not in output
    assert 'Relationship Validity Score: nan%' not in output
    assert list(report.get_properties()['Property']) == ['Data Validity', 'Data Structure']
    assert report.get_score() == 1.0


def test_unified_quality_report_single_table():
    # Setup
    real_data, synthetic_data, metadata = load_single_table_demo()
    column_names = ['student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience']
    metadata['tables']['student_placements']['columns'] = {
        key: val
        for key, val in metadata['tables']['student_placements']['columns'].items()
        if key in column_names
    }

    # Run
    report = QualityReport()
    _set_thresholds_zero(report)
    report.num_rows_subsample = None
    report.generate(
        {'student_placements': real_data['student_placements'][column_names]},
        {'student_placements': synthetic_data['student_placements'][column_names]},
        metadata,
        verbose=False,
    )

    # Assert
    expected_properties = pd.DataFrame({
        'Property': ['Column Shapes', 'Column Pair Trends'],
        'Score': [0.8736798729642614, 0.805070155812896],
    })
    expected_details_column_shapes = pd.DataFrame({
        'Table': [
            'student_placements',
            'student_placements',
            'student_placements',
            'student_placements',
        ],
        'Column': ['start_date', 'second_perc', 'work_experience', 'degree_type'],
        'Metric': ['KSComplement', 'KSComplement', 'TVComplement', 'TVComplement'],
        'Score': [
            0.6621621621621622,
            0.8976744186046511,
            0.9953488372093023,
            0.9395348837209302,
        ],
    })
    expected_details_cpt = pd.DataFrame({
        'Table': ['student_placements'] * 6,
        'Column 1': [
            'start_date',
            'start_date',
            'start_date',
            'second_perc',
            'second_perc',
            'work_experience',
        ],
        'Column 2': [
            'second_perc',
            'work_experience',
            'degree_type',
            'work_experience',
            'degree_type',
            'degree_type',
        ],
        'Metric': [
            'CorrelationSimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
        ],
        'Score': [
            0.9187918131436303,
            0.6744186046511629,
            0.7162790697674419,
            0.813953488372093,
            0.772093023255814,
            0.9348837209302325,
        ],
        'Real Correlation': [0.04735340044317632, np.nan, np.nan, np.nan, np.nan, np.nan],
        'Synthetic Correlation': [
            -0.11506297326956305,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        'Real Association': [np.nan] * 6,
        'Meets Threshold?': pd.Series([True] * 6, dtype='boolean'),
    })
    expected_details_cardinality = pd.DataFrame()
    expected_details_intertable_trends = pd.DataFrame()

    pd.testing.assert_frame_equal(report.get_properties(), expected_properties)
    pd.testing.assert_frame_equal(
        report.get_details('Column Shapes'), expected_details_column_shapes
    )
    pd.testing.assert_frame_equal(report.get_details('Column Pair Trends'), expected_details_cpt)
    pd.testing.assert_frame_equal(
        report.get_details('Cardinality'),
        expected_details_cardinality,
        check_dtype=False,
        check_index_type=False,
    )
    pd.testing.assert_frame_equal(
        report.get_details('Intertable Trends'), expected_details_intertable_trends
    )
    assert report.get_score() == 0.8393750143888287
    assert report._properties['Column Shapes'].num_rows_subsample is None
    assert report._properties['Column Pair Trends'].num_rows_subsample is None
    _assert_report_info(
        report, 'QualityReport', {'student_placements': 215}, {'student_placements': 215}
    )


def test_unified_quality_report_single_table_verbose_skips_relationship_properties(capsys):
    """Test verbose mode skips relationship-only quality properties for single-table data."""
    # Setup
    real_data, synthetic_data, metadata = load_single_table_demo()
    column_names = ['student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience']
    metadata['tables']['student_placements']['columns'] = {
        key: val
        for key, val in metadata['tables']['student_placements']['columns'].items()
        if key in column_names
    }

    # Run
    report = QualityReport()
    _set_thresholds_zero(report)
    report.num_rows_subsample = None
    report.generate(
        {'student_placements': real_data['student_placements'][column_names]},
        {'student_placements': synthetic_data['student_placements'][column_names]},
        metadata,
        verbose=True,
    )
    output = capsys.readouterr().out

    # Assert
    expected_lines = [
        'Generating report ...',
        '(1/4) Evaluating Column Shapes:',
        'Column Shapes Score: 87.37%',
        '(2/4) Evaluating Column Pair Trends:',
        'Column Pair Trends Score: 80.51%',
        '(3/4) Evaluating Cardinality: N/A',
        'This property does not apply to single-table data.',
        '(4/4) Evaluating Intertable Trends: N/A',
        'Overall Score (Average): 83.94%',
    ]
    for line in expected_lines:
        assert line in output

    assert output.count('This property does not apply to single-table data.') == 2
    assert '0/0' not in output
    assert 'Cardinality Score: nan%' not in output
    assert 'Intertable Trends Score: nan%' not in output
    assert list(report.get_properties()['Property']) == ['Column Shapes', 'Column Pair Trends']
    assert report.get_score() == 0.8393750143888287


def test_unified_diagnostic_report_multi_table():
    # Setup
    real_data, synthetic_data, metadata = load_multi_table_demo()

    # Run
    report = DiagnosticReport()
    report.generate(real_data, synthetic_data, metadata, verbose=False)

    # Assert
    expected_properties = pd.DataFrame({
        'Property': ['Data Validity', 'Data Structure', 'Relationship Validity'],
        'Score': [1.0, 1.0, 1.0],
    })
    expected_details_data_validity = pd.DataFrame({
        'Table': [
            'users',
            'users',
            'users',
            'users',
            'sessions',
            'sessions',
            'sessions',
            'transactions',
            'transactions',
            'transactions',
            'transactions',
        ],
        'Column': [
            'user_id',
            'country',
            'gender',
            'age',
            'session_id',
            'device',
            'os',
            'transaction_id',
            'timestamp',
            'amount',
            'approved',
        ],
        'Metric': [
            'KeyUniqueness',
            'CategoryAdherence',
            'CategoryAdherence',
            'BoundaryAdherence',
            'KeyUniqueness',
            'CategoryAdherence',
            'CategoryAdherence',
            'KeyUniqueness',
            'BoundaryAdherence',
            'BoundaryAdherence',
            'CategoryAdherence',
        ],
        'Score': [1.0] * 11,
    })
    expected_details_data_structure = pd.DataFrame({
        'Table': ['users', 'sessions', 'transactions'],
        'Metric': ['TableStructure', 'TableStructure', 'TableStructure'],
        'Score': [1.0, 1.0, 1.0],
    })
    expected_details_users = pd.DataFrame({
        'Table': ['users', 'users', 'users', 'users'],
        'Column': ['user_id', 'country', 'gender', 'age'],
        'Metric': [
            'KeyUniqueness',
            'CategoryAdherence',
            'CategoryAdherence',
            'BoundaryAdherence',
        ],
        'Score': [1.0, 1.0, 1.0, 1.0],
    })

    pd.testing.assert_frame_equal(report.get_properties(), expected_properties)
    pd.testing.assert_frame_equal(
        report.get_details('Data Validity'), expected_details_data_validity
    )
    pd.testing.assert_frame_equal(
        report.get_details('Data Structure'), expected_details_data_structure
    )
    pd.testing.assert_frame_equal(
        report.get_details('Data Validity', 'users'), expected_details_users
    )
    assert report.get_score() == 1.0
    _assert_report_info(
        report,
        'DiagnosticReport',
        {table_name: len(table) for table_name, table in real_data.items()},
        {table_name: len(table) for table_name, table in synthetic_data.items()},
    )


def test_unified_diagnostic_report_multi_table_with_no_relationships_does_not_skip_properties():
    """Test multi-table diagnostic reports do not skip properties when relationships are absent."""
    # Setup
    real_data, synthetic_data, metadata = _load_quality_report_data()
    del metadata['relationships']

    # Run
    report = DiagnosticReport()
    report.generate(real_data, synthetic_data, metadata, verbose=False)
    properties = report.get_properties()

    # Assert
    assert list(properties['Property']) == [
        'Data Validity',
        'Data Structure',
        'Relationship Validity',
    ]
    assert pd.isna(
        properties.loc[properties['Property'] == 'Relationship Validity', 'Score'].iloc[0]
    )


def test_unified_quality_report_multi_table():
    # Setup
    real_data, synthetic_data, metadata = _load_quality_report_data()

    # Run
    report = QualityReport()
    _set_thresholds_zero(report)
    report.generate(real_data, synthetic_data, metadata, verbose=False)
    properties = report.get_properties()
    property_names = list(properties['Property'])
    score = report.get_score()
    details = []
    for property_ in property_names:
        details.append(report.get_details(property_, 'table1'))

    for property_ in property_names:
        details.append(report.get_details(property_))

    # Assert
    assert round(score, 15) == 0.649582127409184
    expected_properties = pd.DataFrame({
        'Property': ['Column Shapes', 'Column Pair Trends', 'Cardinality', 'Intertable Trends'],
        'Score': [0.8, 0.7983285096367361, 0.75, 0.25],
    })
    expected_details_column_shapes = pd.DataFrame({
        'Table': ['table1', 'table1'],
        'Column': ['col2', 'col3'],
        'Metric': ['TVComplement', 'TVComplement'],
        'Score': [0.75, 0.75],
    })
    expected_details_cpt = pd.DataFrame({
        'Table': ['table1'],
        'Column 1': ['col2'],
        'Column 2': ['col3'],
        'Metric': ['ContingencySimilarity'],
        'Score': [0.25],
        'Real Correlation': [np.nan],
        'Synthetic Correlation': [np.nan],
        'Real Association': [np.nan],
        'Meets Threshold?': pd.Series([True], dtype='boolean'),
    })
    expected_details_cardinality = pd.DataFrame({
        'Child Table': ['table2'],
        'Parent Table': ['table1'],
        'Foreign Key': ['col6'],
        'Metric': ['CardinalityShapeSimilarity'],
        'Score': [0.75],
    })
    expected_details_intertable_trends = pd.DataFrame({
        'Parent Table': ['table1', 'table1', 'table1', 'table1', 'table1', 'table1'],
        'Child Table': ['table2', 'table2', 'table2', 'table2', 'table2', 'table2'],
        'Foreign Key': ['col6', 'col6', 'col6', 'col6', 'col6', 'col6'],
        'Column 1': ['col2', 'col2', 'col2', 'col3', 'col3', 'col3'],
        'Column 2': ['col4', 'col5', 'col7', 'col4', 'col5', 'col7'],
        'Metric': [
            'ContingencySimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
        ],
        'Score': [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        'Real Correlation': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'Synthetic Correlation': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'Real Association': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'Meets Threshold?': pd.Series([True, True, True, True, True, True], dtype='boolean'),
    })
    expected_details_all_column_shapes = pd.DataFrame({
        'Table': ['table1', 'table1', 'table2', 'table2', 'table2'],
        'Column': ['col2', 'col3', 'col4', 'col5', 'col7'],
        'Metric': ['TVComplement', 'TVComplement', 'KSComplement', 'KSComplement', 'KSComplement'],
        'Score': [0.75, 0.75, 0.75, 0.75, 1.0],
    })
    expected_details_all_cpt = pd.DataFrame({
        'Table': ['table1', 'table2', 'table2', 'table2'],
        'Column 1': ['col2', 'col4', 'col4', 'col5'],
        'Column 2': ['col3', 'col5', 'col7', 'col7'],
        'Metric': [
            'ContingencySimilarity',
            'CorrelationSimilarity',
            'CorrelationSimilarity',
            'CorrelationSimilarity',
        ],
        'Score': [0.25, 0.9901306731066666, 0.9853027960145061, 0.9678805694257717],
        'Real Correlation': [np.nan, 0.946664, 0.966247, 0.862622],
        'Synthetic Correlation': [np.nan, 0.926925, 0.936853, 0.798384],
        'Real Association': [np.nan, np.nan, np.nan, np.nan],
        'Meets Threshold?': pd.Series([True, True, True, True], dtype='boolean'),
    })

    pd.testing.assert_frame_equal(properties, expected_properties)
    pd.testing.assert_frame_equal(details[0], expected_details_column_shapes)
    pd.testing.assert_frame_equal(details[1], expected_details_cpt)
    pd.testing.assert_frame_equal(details[2], expected_details_cardinality)
    pd.testing.assert_frame_equal(details[3], expected_details_intertable_trends)
    pd.testing.assert_frame_equal(details[4], expected_details_all_column_shapes)
    pd.testing.assert_frame_equal(details[5], expected_details_all_cpt)
    pd.testing.assert_frame_equal(details[6], expected_details_cardinality)
    pd.testing.assert_frame_equal(details[7], expected_details_intertable_trends)
    _assert_report_info(
        report,
        'QualityReport',
        {'table1': 4, 'table2': 4},
        {'table1': 4, 'table2': 4},
    )


def test_unified_quality_report_multi_table_with_no_relationships_does_not_skip_properties():
    """Test multi-table quality reports do not skip properties when relationships are absent."""
    # Setup
    real_data, synthetic_data, metadata = _load_quality_report_data()
    del metadata['relationships']

    # Run
    report = QualityReport()
    _set_thresholds_zero(report)
    report.generate(real_data, synthetic_data, metadata, verbose=False)
    properties = report.get_properties()

    # Assert
    assert list(properties['Property']) == [
        'Column Shapes',
        'Column Pair Trends',
        'Cardinality',
        'Intertable Trends',
    ]
    assert pd.isna(properties.loc[properties['Property'] == 'Cardinality', 'Score'].iloc[0])
    assert pd.isna(properties.loc[properties['Property'] == 'Intertable Trends', 'Score'].iloc[0])
