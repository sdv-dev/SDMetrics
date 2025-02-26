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
    zero_subsample_msg = re.escape('num_rows_subsample (0) must be an integer greater than 1.')
    with pytest.raises(ValueError, match=zero_subsample_msg):
        validate_num_samples_num_iteration(0, 1)

    subsample_none_msg = re.escape(
        'num_iterations should not be greater than 1 if there is no subsampling.'
    )
    with pytest.raises(ValueError, match=subsample_none_msg):
        validate_num_samples_num_iteration(None, 2)

    zero_iteration_msg = re.escape('num_iterations (0) must be an integer greater than 1.')
    with pytest.raises(ValueError, match=zero_iteration_msg):
        validate_num_samples_num_iteration(1, 0)
