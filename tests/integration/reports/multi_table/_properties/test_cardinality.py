"""Test multi-table cardinality properties."""
from plotly.graph_objs._figure import Figure

from sdmetrics.demos import load_multi_table_demo
from sdmetrics.reports.multi_table._properties import Cardinality


def test_cardinality_property():
    """Test the ``Cardinality`` with the multi table demo."""
    # Setup
    cardinality_property = Cardinality()
    real_data, synthetic_data, metadata = load_multi_table_demo()

    # Run
    score = cardinality_property.get_score(real_data, synthetic_data, metadata)
    figure = cardinality_property.get_visualization('users')

    # Assert
    assert score == 0.8
    assert isinstance(figure, Figure)
