"""Test multi-table cardinality properties."""
import pandas as pd
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
    assert score == 0.95
    assert isinstance(figure, Figure)


def test_with_multi_foreign_key():
    """Test the ``Cardinality`` with multiple foreign keys."""
    # Setup
    real_data = {
        'bank': pd.DataFrame({
            'primary_key': [1, 2, 3, 4, 5],
            'category': ['a', 'b', 'c', 'd', 'e'],
            'numerical': [1, 2, 3, 4, 5],
        }),
        'transactions': pd.DataFrame({
            'f_key_1': [1, 2, 3, 2, 1],
            'f_key_2': [1, 5, 3, 2, 4],
        })
    }

    synthetic_data = {
        'bank': pd.DataFrame({
            'primary_key': [1, 2, 3, 4, 5],
            'category': ['a', 'b', 'c', 'd', 'e'],
            'numerical': [1, 2, 3, 4, 5]
        }),
        'transactions': pd.DataFrame({
            'f_key_1': [5, 2, 3, 4, 1],
            'f_key_2': [1, 5, 5, 2, 4],
        })
    }

    metadata = {
        'tables': {
            'bank': {
                'primary_key': 'primary_key',
                'columns': {
                    'primary_key': {'sdtype': 'id'},
                    'category': {'sdtype': 'categorical'},
                    'numerical': {'sdtype': 'numerical'}
                }
            },
            'transactions': {
                'columns': {
                    'f_key_1': {'sdtype': 'id'},
                    'f_key_2': {'sdtype': 'id'}
                }
            }
        },
        'relationships': [
            {
                'parent_table_name': 'bank',
                'child_table_name': 'transactions',
                'parent_primary_key': 'primary_key',
                'child_foreign_key': 'f_key_1'
            },
            {
                'parent_table_name': 'bank',
                'child_table_name': 'transactions',
                'parent_primary_key': 'primary_key',
                'child_foreign_key': 'f_key_2'
            }
        ]
    }

    cardinality_property = Cardinality()

    # Run
    cardinality_property.get_score(real_data, synthetic_data, metadata)
    fig = cardinality_property.get_visualization('bank')

    # Assert
    expected_labels = ['transactions (f_key_1) → bank', 'transactions (f_key_2) → bank']
    assert fig.data[0].x.tolist() == expected_labels
