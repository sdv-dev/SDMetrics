import re

import pytest

from sdmetrics.single_table.privacy.util import (
    closest_neighbors,
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
    data_length = 2
    num_subsample_error_post = re.escape(
        f'must be an integer greater than 1 and less than num of rows in the data ({data_length}).'
    )

    with pytest.raises(ValueError, match=num_subsample_error_post):
        validate_num_samples_num_iteration(0, 1, data_length)

    with pytest.raises(ValueError, match=num_subsample_error_post):
        validate_num_samples_num_iteration('X', 1, data_length)

    large_subsample_value = 5
    with pytest.raises(ValueError, match=num_subsample_error_post):
        validate_num_samples_num_iteration(large_subsample_value, 1, data_length)

    subsample_none_msg = re.escape(
        'num_iterations should not be greater than 1 if there is no subsampling.'
    )
    num_iterations = 3
    with pytest.raises(ValueError, match=subsample_none_msg):
        validate_num_samples_num_iteration(None, num_iterations, data_length)

    zero_iteration_msg = re.escape('num_iterations (0) must be an integer greater than 1.')
    with pytest.raises(ValueError, match=zero_iteration_msg):
        validate_num_samples_num_iteration(1, 0, data_length)
