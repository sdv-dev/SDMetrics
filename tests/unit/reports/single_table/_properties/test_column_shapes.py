
import re
from unittest.mock import call, patch

import pandas as pd
import pytest

from sdmetrics.reports.single_table._properties.column_shapes import ColumnShapes


class TestColumnShapes:

    @patch('sdmetrics.reports.single_table._properties.column_shapes.KSComplement.compute')
    @patch('sdmetrics.reports.single_table._properties.column_shapes.TVComplement.compute')
    def test__generate_details(self, tv_complement_compute_mock, ks_complement_compute_mock):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        })
        synthetic_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [False, True, True],
            'col3': ['a', 'b', 'c'],
            'col4': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'boolean'},
                'col3': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'datetime'}
            }
        }

        # Run
        column_shape_property = ColumnShapes()
        column_shape_property._generate_details(real_data, synthetic_data, metadata)

        # Assert
        expected_calls_ksc = [
            call(real_data['col1'], synthetic_data['col1']),
            call(real_data['col4'], synthetic_data['col4']),
        ]
        expected_calls_tvc = [
            call(real_data['col2'], synthetic_data['col2']),
            call(real_data['col3'], synthetic_data['col3']),
        ]

        ks_complement_compute_mock.assert_has_calls(expected_calls_ksc)
        tv_complement_compute_mock.assert_has_calls(expected_calls_tvc)

    def test__generate_details_warning(self):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = pd.DataFrame({'col1': [1, '2', 3]})
        synthetic_data = pd.DataFrame({'col1': [4, 5, 6]})
        metadata = {'columns': {'col1': {'sdtype': 'numerical'}}}

        # Run and Assert
        column_shape_property = ColumnShapes()
        expected_message = re.escape(
            "Unable to compute Column Shape for column 'col1'. Encountered Error: "
            "TypeError '<' not supported between instances of 'str' and 'int'"
        )
        with pytest.warns(UserWarning, match=expected_message):
            column_shape_property._generate_details(real_data, synthetic_data, metadata)

    def test_get_visualization(self):
        assert True
