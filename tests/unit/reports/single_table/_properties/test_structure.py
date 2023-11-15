from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.errors import VisualizationUnavailableError
from sdmetrics.reports.single_table._properties.structure import Structure


class TestStructure:

    @patch('sdmetrics.reports.single_table._properties.structure.'
           'TableStructure.compute')
    def test__generate_details(self, table_format_mock):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, np.nan],
            'col2': [False, True, True],
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
            }
        }

        table_format_mock.return_value = 0.75

        # Run
        structure_property = Structure()
        result = structure_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        table_format_mock.assert_called_once_with(
            real_data, synthetic_data,
        )

        expected_details = pd.DataFrame({
            'Metric': 'TableStructure',
            'Score': 0.75,
        }, index=[0])
        pd.testing.assert_frame_equal(result, expected_details)

    @patch('sdmetrics.reports.single_table._properties.structure.'
           'TableStructure.compute')
    def test__generate_details_with_id_column(self, table_format_mock):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, np.nan],
            'col2': [False, True, True],
            'id': [1, 2, 3]
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'id': [1, 2, 3]
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'id': {'sdtype': 'id'}
            }
        }

        table_format_mock.return_value = 0.75

        # Run
        structure_property = Structure()
        result = structure_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        table_format_mock.assert_called_once_with(
            real_data, synthetic_data
        )

        expected_details = pd.DataFrame({
            'Metric': 'TableStructure',
            'Score': 0.75,
        }, index=[0])
        pd.testing.assert_frame_equal(result, expected_details)

    def test_get_visualization(self):
        """Test the ``get_visualization`` method."""
        # Setup
        structure_property = Structure()

        # Run and Assert
        expected_message = (
            'The single table Structure property does not have a'
            ' supported visualization.'
        )
        with pytest.raises(VisualizationUnavailableError, match=expected_message):
            structure_property.get_visualization()
