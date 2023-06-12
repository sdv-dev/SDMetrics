"""Test BaseSingleTableProperty class."""
import re
from unittest.mock import Mock, patch

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

        base_property._details = pd.DataFrame({'Column Name': ['a', 'b', 'c']})
        with pytest.raises(ValueError, match=expected_error_message):
            base_property._compute_average()

    @patch('sdmetrics.reports.single_table._properties.base.validate_single_table_inputs')
    def test_get_score(self, validate_single_table_inputs_mock):
        """Test the ``get_score`` method."""
        # Setup
        validate_single_table_inputs_mock.return_value = None

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
        validate_single_table_inputs_mock.assert_called_once_with(
            real_data, synthetic_data, metadata
        )
        mock__generate_details.assert_called_once_with(
            real_data, synthetic_data, metadata, progress_bar
        )
        mock_compute_average.assert_called_once()
