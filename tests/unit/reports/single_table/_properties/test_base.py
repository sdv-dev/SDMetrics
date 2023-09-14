"""Test BaseSingleTableProperty class."""
import re
from unittest.mock import Mock

import pandas as pd
import pytest

from sdmetrics.reports.single_table._properties import BaseSingleTableProperty


class TestBaseSingleTableProperty:

    def test__generate_details_raises_error(self):
        """Test that the method raises a ``NotImplementedError``."""
        # Setup
        base_property = BaseSingleTableProperty()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            base_property._generate_details(None, None, None, None)

    def test__get_num_iterations(self):
        """Test the ``_get_num_iterations``."""
        # Setup
        base_property = BaseSingleTableProperty()

        metadata = {
            'columns': {
                'col1': {}, 'col2': {}, 'col3': {},
                'col4': {}, 'col5': {},
            }
        }

        # Run and Assert
        base_property._num_iteration_case = 'column'
        assert base_property._get_num_iterations(metadata) == 5

        base_property._num_iteration_case = 'table'
        assert base_property._get_num_iterations(metadata) == 1

        base_property._num_iteration_case = 'column_pair'
        assert base_property._get_num_iterations(metadata) == 10

    def test_get_visualization_raises_error(self):
        """Test that the method raises a ``NotImplementedError``."""
        # Setup
        base_property = BaseSingleTableProperty()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            base_property.get_visualization()

    def test__compute_average_raises_error(self):
        """Test that the method raises a ``ValueError`` when _details has not been computed."""
        # Setup
        base_property = BaseSingleTableProperty()

        # Run and Assert
        expected_error_message = re.escape(
            "The property details must be a DataFrame with a 'Score' column."
        )
        with pytest.raises(ValueError, match=expected_error_message):
            base_property._compute_average()

        base_property.details = pd.DataFrame({'Column': ['a', 'b', 'c']})
        with pytest.raises(ValueError, match=expected_error_message):
            base_property._compute_average()

    def test_get_score(self):
        """Test the ``get_score`` method."""
        # Setup

        real_data = Mock()
        synthetic_data = Mock()
        metadata = Mock()
        progress_bar = Mock()

        mock_compute_average = Mock()
        mock__generate_details = Mock(return_value=None)

        base_property = BaseSingleTableProperty()
        base_property._compute_average = mock_compute_average
        base_property._generate_details = mock__generate_details

        # Run
        base_property.get_score(real_data, synthetic_data, metadata, progress_bar)

        # Assert
        mock__generate_details.assert_called_once_with(
            real_data, synthetic_data, metadata, progress_bar
        )
        mock_compute_average.assert_called_once()
