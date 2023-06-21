from datetime import date, datetime

import pandas as pd

from sdmetrics.reports.single_table.quality_report import QualityReport


def load_test_data():
    real_data = pd.DataFrame({
        'col1': [0, 1, 2, 3],
        'col2': ['a', 'b', 'c', 'd'],
        'col3': [True, False, False, True],
        'col4': [
            datetime(2020, 10, 1),
            datetime(2021, 1, 2),
            datetime(2021, 9, 12),
            datetime(2022, 10, 1),
        ],
        'col5': [date(2020, 9, 13), date(2020, 12, 1), date(2021, 1, 12), date(2022, 8, 13)],
    })

    synthetic_data = pd.DataFrame({
        'col1': [0, 2, 2, 3],
        'col2': ['a', 'c', 'c', 'b'],
        'col3': [False, False, False, True],
        'col4': [
            datetime(2020, 11, 4),
            datetime(2021, 2, 1),
            datetime(2021, 8, 1),
            datetime(2022, 12, 1),
        ],
        'col5': [date(2020, 10, 13), date(2020, 2, 4), date(2021, 3, 11), date(2022, 7, 23)],
    })

    metadata = {
        'columns': {
            'col1': {'sdtype': 'numerical'},
            'col2': {'sdtype': 'categorical'},
            'col3': {'sdtype': 'boolean'},
            'col4': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'col5': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
        }
    }

    return (real_data, synthetic_data, metadata)


def test_single_table_quality_report():
    """Test the single table quality report."""
    real_data, synthetic_data, metadata = load_test_data()

    report = QualityReport()
    report.generate(real_data, synthetic_data, metadata)

    properties = report.get_properties()
    pd.testing.assert_frame_equal(properties, pd.DataFrame({
        'Property': ['Column Shapes', 'Column Pair Trends'],
        'Score': [0.750000, 0.550575448192246],
    }))
