import re

import pandas as pd
import pytest

from sdmetrics.single_table.privacy.util import (
    closest_neighbors,
    detect_time_granularity,
    validate_num_samples_num_iteration,
)


def test_closest_neighbors_exact():
    samples = [
        ('a', '1'),
        ('a', '2'),
        ('a', '3'),
        ('b', '1'),
        ('b', '2'),
        ('b', '3'),
    ]
    target = ('a', '2')
    results = closest_neighbors(samples, target)
    assert len(results) == 1
    assert results[0] == ('a', '2')


def test_closest_neighbors_non_exact():
    samples = [
        ('a', '1'),
        ('a', '3'),
        ('b', '1'),
        ('b', '2'),
        ('b', '3'),
    ]
    target = ('a', '2')
    results = closest_neighbors(samples, target)
    assert len(results) == 3
    assert ('a', '1') in results
    assert ('a', '3') in results
    assert ('b', '2') in results


def test_validate_num_samples_num_iteration():
    # Run and Assert
    num_subsample_error_post = re.escape('must be an integer greater than 1.')

    with pytest.raises(ValueError, match=num_subsample_error_post):
        validate_num_samples_num_iteration(0, 1)

    with pytest.raises(ValueError, match=num_subsample_error_post):
        validate_num_samples_num_iteration('X', 1)

    subsample_none_msg = re.escape(
        'num_iterations should not be greater than 1 if there is no subsampling.'
    )
    num_iterations = 3
    with pytest.raises(ValueError, match=subsample_none_msg):
        validate_num_samples_num_iteration(None, num_iterations)

    zero_iteration_msg = re.escape('num_iterations (0) must be an integer greater than 1.')
    with pytest.raises(ValueError, match=zero_iteration_msg):
        validate_num_samples_num_iteration(1, 0)


def test_microsecond_detect_time_granularity():
    """Test `detect_time_granularity` to see if microseconds are detected."""
    # Setup
    series = pd.Series(pd.to_datetime(['2025-01-01 12:00:00.123456', '2025-01-01 12:00:00.654321']))

    # Run and Assert
    assert detect_time_granularity(series) == 'us'


def test_second_detect_time_granularity():
    """Test `detect_time_granularity` to see if seconds are detected."""
    # Setup
    series = pd.Series(pd.to_datetime(['2025-01-01 12:00:01.123', '2025-01-01 12:00:02.123']))

    # Run and Assert
    assert detect_time_granularity(series) == 's'


def test_minute_detect_time_granularity():
    """Test `detect_time_granularity` to see if minutes are detected."""
    # Setup
    series = pd.Series(pd.to_datetime(['2025-01-01 12:01:00', '2025-01-01 12:02:00']))

    # Run and Assert
    assert detect_time_granularity(series) == 'm'


def test_hour_detect_time_granularity():
    """Test `detect_time_granularity` to see if hours are detected."""
    # Setup
    series = pd.Series(pd.to_datetime(['2025-01-01 12:00:05', '2025-01-01 13:00:05']))

    # Run and Assert
    assert detect_time_granularity(series) == 'h'


def test_day_detect_time_granularity():
    """Test `detect_time_granularity` to see if days are detected."""
    # Setup
    series = pd.Series(pd.to_datetime(['2025-01-01', '2024-03-20']))
    single_series = pd.Series(pd.to_datetime(['2025-01-01']))
    identical_values = pd.Series(
        pd.to_datetime(['2025-01-01 12:00:00.123', '2025-01-01 12:00:00.123'])
    )

    # Run and Assert
    assert detect_time_granularity(series) == 'D'
    assert detect_time_granularity(single_series) == 'D'
    assert detect_time_granularity(identical_values) == 'D'
