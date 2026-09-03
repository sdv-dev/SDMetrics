"""Fixtures shared by the integration tests."""

import pytest

from sdmetrics.demos import load_demo, load_timeseries_demo
from sdmetrics.reports.base_report import BaseReport


@pytest.fixture
def converted_datetime_single_table_demo():
    """Single table demo data with the datetime columns converted to ``datetime64``."""
    real_data, synthetic_data, metadata = load_demo(modality='single_table')
    table_name = 'student_placements'
    BaseReport.convert_datetimes(
        real_data[table_name], synthetic_data[table_name], metadata['tables'][table_name]
    )

    return real_data, synthetic_data, metadata


@pytest.fixture
def converted_datetime_multi_table_demo():
    """Multi table demo data with the datetime columns converted to ``datetime64``."""
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    for table_name, table_metadata in metadata['tables'].items():
        BaseReport.convert_datetimes(
            real_data[table_name], synthetic_data[table_name], table_metadata
        )

    return real_data, synthetic_data, metadata


@pytest.fixture
def converted_datetime_timeseries_demo():
    """Timeseries demo data with the datetime columns converted to ``datetime64``."""
    real_data, synthetic_data, metadata = load_timeseries_demo()
    table_name = 'timeseries'
    BaseReport.convert_datetimes(
        real_data[table_name], synthetic_data[table_name], metadata['tables'][table_name]
    )

    return real_data, synthetic_data, metadata
