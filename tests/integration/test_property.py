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

    # Run
    score = property_instance.get_score(REAL_DATA_ST, SYNTHETIC_DATA_ST, METADATA_ST)
    score_shuffled = property_instance.get_score(
        REAL_DATA_ST.sample(frac=1), SYNTHETIC_DATA_ST.sample(frac=1), METADATA_ST
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

    # Run
    score = property_instance.get_score(REAL_DATA_MT, SYNTHETIC_DATA_MT, METADATA_MT)
    score_shuffled = property_instance.get_score(
        real_data_shuffled, synthetic_data_shuffled, METADATA_MT
    )

    # Assert
    assert np.isclose(score, score_shuffled, rtol=1e-12)
