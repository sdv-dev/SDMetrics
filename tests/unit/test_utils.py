import pandas as pd

from sdmetrics.utils import get_cardinality_distribution


def test_get_cardinality_distribution():
    """Test the ``get_cardinality_distribution`` utility function.

    Input:
    - parent column
    - child column

    Output:
    - the expected cardinality distribution.
    """
    # Setup
    parent_column = pd.Series([1, 2, 3, 4, 5])
    child_column = pd.Series([3, 4, 1, 4, 4, 5, 1])

    # Run
    cardinality_distribution = get_cardinality_distribution(parent_column, child_column)

    # Assert
    assert cardinality_distribution.to_list() == [2.0, 0.0, 1.0, 3.0, 1.0]
