import pandas as pd
import pytest

from sdmetrics.timeseries.base import TimeSeriesMetric


def test__validate_inputs_for_TimeSeriesMetric():
    """Test ``_validate_inputs`` crashes when datetime column doesn't match metadata."""
    # Setup
    df1 = pd.DataFrame({
        's_key': [1, 2, 3],
        'visits': pd.to_datetime(['1/1/2019', '1/2/2019', '1/3/2019']),
    })
    df1['visits'] = df1['visits'].dt.date
    df2 = pd.DataFrame({
        's_key': [1, 2, 3],
        'visits': ['not', 'a', 'datetime'],
    })
    metadata = {
        'columns': {
            's_key': {'sdtype': 'numerical'},
            'visits': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d %H:%M:%S'},
        },
        'sequence_key': 's_key',
    }

    # Run and Assert
    with pytest.raises(ValueError, match="Column 'visits' is not a valid datetime"):
        TimeSeriesMetric._validate_inputs(
            real_data=df1, synthetic_data=df2, sequence_key=['s_key'], metadata=metadata
        )
