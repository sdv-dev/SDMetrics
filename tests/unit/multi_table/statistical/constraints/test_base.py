import re

import pytest

from sdmetrics.multi_table.statistical.constraints import BaseConstraint, DenormalizedTable


class TestBaseConstraint:
    def test_load_constraint_from_dict(self):
        """Test ``load_constraint_from_dict`` method."""
        # Setup
        constraint_dict = {
            'class_name': 'DenormalizedTable',
            'parameters': {
                'table_name': 'UserTransaction',
                'denormalized_primary_key': 'User ID',
                'denormalized_column_names': ['FirstName'],
            },
        }

        # Run
        instance = BaseConstraint.load_constraint_from_dict(constraint_dict)

        # Assert
        assert isinstance(instance, DenormalizedTable)
        assert instance.table_name == 'UserTransaction'
        assert instance.denormalized_primary_key == 'User ID'
        assert instance.denormalized_column_names == ['FirstName']

    def test_load_constraint_from_dict_unknown_class_name(self):
        """Test ``load_constraint_from_dict`` errors if the constraint class is not supported."""
        # Setup
        expected_error = re.escape("Unsupported constraint class 'Unknown'.")

        # Run & Assert
        with pytest.raises(ValueError, match=expected_error):
            BaseConstraint.load_constraint_from_dict({'class_name': 'Unknown', 'parameters': {}})

    def test_load_constraint_from_dict_missing_class_name(self):
        """Test ``load_constraint_from_dict`` errors if the ``class_name`` key is missing."""
        # Setup
        expected_error = re.escape("Invalid constraint. Missing the required key 'class_name'.")

        # Run & Assert
        with pytest.raises(ValueError, match=expected_error):
            BaseConstraint.load_constraint_from_dict({'parameters': {}})

    def test_load_constraint_from_dict_invalid_parameters(self):
        """Test ``load_constraint_from_dict`` errors if a parameter is not supported."""
        # Setup
        constraint_dict = {
            'class_name': 'DenormalizedTable',
            'parameters': {'table_name': 'UserTransaction', 'unknown': 'value'},
        }

        expected_error = re.escape("Invalid parameter(s) 'unknown' for constraint")

        # Run & Assert
        with pytest.raises(ValueError, match=expected_error):
            BaseConstraint.load_constraint_from_dict(constraint_dict)

    def test_load_constraint_from_dict_missing_required_parameter(self):
        """Test it errors if a required parameter is not passed in."""
        # Setup
        constraint_dict = {
            'class_name': 'DenormalizedTable',
            'parameters': {'table_name': 'UserTransaction'},
        }

        expected_error = re.escape("Unable to create the constraint 'DenormalizedTable'")

        # Run
        with pytest.raises(ValueError, match=expected_error):
            BaseConstraint.load_constraint_from_dict(constraint_dict)
