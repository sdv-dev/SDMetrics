"""Tests that are common to all properties."""

import numpy as np
import pytest

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table import _properties as multi_table_properties
from sdmetrics.reports.single_table import _properties as single_table_properties

REAL_DATA_ST, SYNTHETIC_DATA_ST, METADATA_ST = load_demo(modality='single_table')
REAL_DATA_MT, SYNTHETIC_DATA_MT, METADATA_MT = load_demo(modality='multi_table')
SINGLE_TABLE_PROPERTIES = [
    property
    for property_name, property in vars(single_table_properties).items()
    if property_name != 'BaseSingleTableProperty' and isinstance(property, type)
]
MULTI_TABLE_PROPERTIES = [
    property
    for property_name, property in vars(multi_table_properties).items()
    if property_name != 'BaseMultiTableProperty' and isinstance(property, type)
]


@pytest.mark.parametrize('property', SINGLE_TABLE_PROPERTIES)
def test_shuffling_data_single_table(property):
    """Test the property score is the same when shuffling the data for single-table."""
    # Setup
    property_instance = property()
    real_data = REAL_DATA_ST[list(REAL_DATA_ST)[0]]
    synth_data = SYNTHETIC_DATA_ST[list(SYNTHETIC_DATA_ST)[0]]

    # Run
    score = property_instance.get_score(real_data, synth_data, METADATA_ST)
    score_shuffled = property_instance.get_score(
        real_data.sample(frac=1), synth_data.sample(frac=1), METADATA_ST
    )

    # Assert
    assert score_shuffled == score


@pytest.mark.parametrize('property', MULTI_TABLE_PROPERTIES)
def test_shuffling_data_multi_table(property):
    """Test the property score is the same when shuffling the data for multi-table."""
    # Setup
    property_instance = property()
    real_data_shuffled = {
        table_name: table.sample(frac=1) for table_name, table in REAL_DATA_MT.items()
    }
    synthetic_data_shuffled = {
        table_name: SYNTHETIC_DATA_MT[table_name].sample(frac=1) for table_name in SYNTHETIC_DATA_MT
    }
    kwargs = {}
    if property is multi_table_properties.ConstraintValidity:
        kwargs['constraints'] = [
            {
                'class_name': 'FixedCombinations',
                'parameters': {'table_name': 'sessions', 'column_names': ['device', 'os']},
            },
        ]

    # Run
    score = property_instance.get_score(REAL_DATA_MT, SYNTHETIC_DATA_MT, METADATA_MT, **kwargs)
    score_shuffled = property_instance.get_score(
        real_data_shuffled, synthetic_data_shuffled, METADATA_MT, **kwargs
    )

    # Assert
    assert np.isclose(score, score_shuffled, rtol=1e-12)
