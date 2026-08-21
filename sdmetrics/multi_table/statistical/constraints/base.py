"""Base Constraint."""

import inspect

import numpy as np
import pandas as pd


class BaseConstraint:
    """Base class for all constraints.

    A constraint knows how to check, row by row, whether some data adheres to it.
    Subclasses must define ``_validate_data`` and ``_is_valid``, and accept every
    constraint parameter as a keyword argument in their ``__init__``.
    """
    _is_single_table = True

    @classmethod
    def _get_subclasses(cls):
        """Return a mapping of every (recursive) subclass name to the subclass."""
        subclasses = {}
        for subclass in cls.__subclasses__():
            if not inspect.isabstract(subclass):
                subclasses[subclass.__name__] = subclass

            subclasses.update(subclass._get_subclasses())

        return subclasses

    @classmethod
    def _get_init_parameters(cls):
        parameters = inspect.signature(cls.__init__).parameters
        return [name for name in parameters if name != 'self']

    @classmethod
    def _validate_constraint(cls, constraint_dict):
        """Validate a constraint in its dictionary representation.

        Args:
            constraint_dict (dict):
                A dictionary with a ``class_name`` key (the name of the constraint
                class) and ``parameters`` key.
        """
        if not isinstance(constraint_dict, dict):
            raise ValueError(
                'Invalid constraint. Please pass in a dictionary with the keys '
                "'class_name' and 'parameters'."
            )

        invalid_keys = set(constraint_dict) - {'class_name', 'parameters'}
        if invalid_keys:
            invalid_keys = "', '".join(sorted(invalid_keys))
            raise ValueError(
                f"Invalid key(s) '{invalid_keys}' in the constraint. Only the keys "
                "'class_name' and 'parameters' are allowed."
            )

        class_name = constraint_dict.get('class_name')
        if not isinstance(class_name, str):
            raise ValueError("Invalid constraint. Missing the required key 'class_name'.")

        constraint_classes = BaseConstraint._get_subclasses()
        if class_name not in constraint_classes:
            raise ValueError(f"Unsupported constraint class '{class_name}'.")

        parameters = constraint_dict.get('parameters') or {}
        if not isinstance(parameters, dict):
            raise ValueError("Invalid constraint. The 'parameters' key must be a dictionary.")

        constraint_class = constraint_classes[class_name]
        expected_parameters = constraint_class._get_init_parameters()
        invalid_parameters = set(parameters) - set(expected_parameters)
        if invalid_parameters:
            invalid_parameters = "', '".join(sorted(invalid_parameters))
            expected = "', '".join(expected_parameters)
            raise ValueError(
                f"Invalid parameter(s) '{invalid_parameters}' for constraint '{class_name}'. "
                f"The supported parameters are: '{expected}'."
            )

    @classmethod
    def load_constraint_from_dict(cls, constraint_dict):
        """Uses the given parameters to recreate an instance of the constraint."""
        cls._validate_constraint(constraint_dict)
        constraint_classes = BaseConstraint._get_subclasses()

        class_name = constraint_dict['class_name']
        constraint_class = constraint_classes[class_name]
        try:
            return constraint_class(**constraint_dict['parameters'])
        except TypeError as ex:
            raise ValueError(
                f"Unable to create the constraint '{class_name}': {ex}"
            ) from ex

    def __init__(self):
        self.metadata = None
        self._fitted = False
        self._single_table = False
        self._dtypes = None
        self._formatters = {}
        self._datetime_min_max_value = {}

    def _get_single_table_name(self, metadata):
        if not hasattr(self, 'table_name'):
            raise ValueError('No ``table_name`` attribute has been set.')

        return metadata._get_single_table_name() if self.table_name is None else self.table_name

    def _validate_data(self, data, metadata=None):
        """Check that this constraint can be applied to the given data.

        Args:
            data (dict[str, pandas.DataFrame]):
                A dictionary mapping each table name to its data.
            metadata (dict):
                The multi table metadata.

        Raises:
            ConstraintNotApplicableError:
                If the constraint cannot be checked against the data.
        """
        raise NotImplementedError()

    def _is_valid(self, data, metadata=None):
        """Determine whether each row in the data adheres to this constraint.

        Args:
            data (dict[str, pandas.DataFrame]):
                A dictionary mapping each table name to its data.
            metadata (dict):
                The multi table metadata.

        Returns:
            dict[str, pandas.Series]:
                A dictionary mapping each table name that is relevant to this
                constraint to a boolean ``pandas.Series`` that states whether each
                row of that table adheres to the constraint.
        """
        raise NotImplementedError()

    def is_valid(self, data, metadata=None):
        """Say whether the given table rows are valid.

        Args:
            data (pd.DataFrame or dict[pd.DataFrame]):
                Table data.

        Returns:
            pd.Series or dict[pd.Series]:
                Series of boolean values indicating if the row is valid for the constraint or not.
        """
        metadata = self.metadata if metadata is None else metadata
        self._validate_data(data, metadata)

        is_valid_data = self._is_valid(data, metadata)
        if isinstance(data, pd.DataFrame) or self._single_table:
            table_name = (
                self._get_single_table_name(metadata)
                if getattr(self, '_table_name', None) is None
                else self._table_name
            )
            return is_valid_data[table_name]

        return is_valid_data

    def get_score(self, data, metadata=None):
        """Get the proportion of rows in the data that adhere to this constraint.

        Args:
            data (dict[str, pandas.DataFrame]):
                A dictionary mapping each table name to its data.
            metadata (dict):
                The multi table metadata.

        Returns:
            float:
                The proportion of valid rows, or ``np.nan`` if there are no rows to
                check.

        """
        self._validate_data(data, metadata)
        validity = self.is_valid(data, metadata)
        num_rows = sum(len(is_valid) for is_valid in validity.values())
        if num_rows == 0:
            return np.nan

        num_valid_rows = sum(int(is_valid.sum()) for is_valid in validity.values())

        return num_valid_rows / num_rows
