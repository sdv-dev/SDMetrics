import re

import pytest

from sdmetrics.multi_table.statistical.constraints import BaseConstraint, FixedCombinations


class TestBaseConstraint:
    def test_load_constraint_from_dict(self):
        """Test ``load_constraint_from_dict`` method."""
        # Setup
        constraint_dict = {
            'class_name': 'FixedCombinations',
            'parameters': {
                'table_name': 'tableA',
                'column_names': ['col1', 'col2'],
            },
        }

        # Run
        instance = BaseConstraint.load_constraint_from_dict(constraint_dict)

        # Assert
        assert isinstance(instance, FixedCombinations)
        assert instance.table_name == 'tableA'
        assert instance.column_names == ['col1', 'col2']

    def test_load_constraint_from_dict_unknown_class_name(self):
        """Test ``load_constraint_from_dict`` errors if the constraint class is not supported."""
        # Setup
        expected_error = re.escape("Unsupported constraint class 'Unknown'.")

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            BaseConstraint.load_constraint_from_dict({'class_name': 'Unknown', 'parameters': {}})

    def test_load_constraint_from_dict_missing_class_name(self):
        """Test ``load_constraint_from_dict`` errors if the ``class_name`` key is missing."""
        # Setup
        expected_error = re.escape("Invalid constraint. Missing the required key 'class_name'.")

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            BaseConstraint.load_constraint_from_dict({'parameters': {}})

    def test_load_constraint_from_dict_invalid_parameters(self):
        """Test ``load_constraint_from_dict`` errors if a parameter is not supported."""
        # Setup
        constraint_dict = {
            'class_name': 'FixedCombinations',
            'parameters': {'table_name': 'tableA', 'unknown': 'value'},
        }

        expected_error = re.escape("Invalid parameter(s) 'unknown' for constraint")

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            BaseConstraint.load_constraint_from_dict(constraint_dict)

    def test_load_constraint_from_dict_missing_required_parameter(self):
        """Test it errors if a required parameter is not passed in."""
        # Setup
        constraint_dict = {
            'class_name': 'FixedCombinations',
            'parameters': {'table_name': 'tableA'},
        }

        expected_error = re.escape("Unable to create the constraint 'FixedCombinations'")

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            BaseConstraint.load_constraint_from_dict(constraint_dict)
