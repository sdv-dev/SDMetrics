import numpy as np

from sdmetrics.column_pairs.statistical import ContingencySimilarity
from sdmetrics.demos import load_demo


def test_with_num_rows_subsample():
    """Test the metric with `num_rows_subsample`.

    Here the `real_data` and `syntehtic_data` have 218 rows.
    """
    # Setup
    np.random.seed(42)
    real_data, synthetic_data, _ = load_demo('single_table')
    real_data = real_data[['degree_type', 'high_spec']]
    synthetic_data = synthetic_data[['degree_type', 'high_spec']]
    num_rows_subsample = 100

    # Run
    result_1 = ContingencySimilarity.compute(
        real_data=real_data,
        synthetic_data=synthetic_data,
        num_rows_subsample=num_rows_subsample,
    )
    result_2 = ContingencySimilarity.compute(
        real_data=real_data,
        synthetic_data=synthetic_data,
        num_rows_subsample=num_rows_subsample,
    )
    result_entire_data = ContingencySimilarity.compute(
        real_data=real_data,
        synthetic_data=synthetic_data,
        num_rows_subsample=None,
    )

    # Assert
    assert result_1 != result_2
    assert result_1 != result_entire_data
    assert result_2 != result_entire_data
    assert np.isclose(result_1, result_entire_data, atol=0.1)
    assert np.isclose(result_2, result_entire_data, atol=0.1)
