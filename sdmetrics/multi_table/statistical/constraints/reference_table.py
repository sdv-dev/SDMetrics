"""Reference Table Constraint."""

from sdmetrics.multi_table.statistical.constraints._utils import (
    _get_table_to_valid_rows,
)
from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


class ReferenceTable(BaseConstraint):
    """Constraint for tables whose rows connect to a reference table.

    Args:
        reference_table_names (list[str]):
            Names of the reference tables.
    """

    _is_single_table = False

    def __init__(self, reference_table_names):
        super().__init__()

        if not isinstance(reference_table_names, list) or not all(
            isinstance(name, str) for name in reference_table_names
        ):
            raise ValueError("'reference_table_names' must be a list of strings.")

        self.reference_table_names = reference_table_names

    def _validate_constraint_with_metadata(self, metadata):
        """Validate the metadata for the constraint.

        This method:
        - Validates that each reference table exists in the metadata.
        - Validates that no reference table is a child of another table.
          A reference table can be the child of another reference table.

        Args:
            metadata (dict):
                The metadata for the dataset.

        Raises:
            ConstraintNotMetError:
                If any reference table is missing from metadata
                or is a child of a non-reference table.
        """
        if any(table not in metadata['tables'] for table in self.reference_table_names):
            missing = set(self.reference_table_names) - set(metadata['tables'])
            raise ConstraintNotApplicableError(
                f"Reference table(s) '{sorted(missing)}' missing from metadata."
            )

        invalid_pairs = set()
        for relationship in metadata['relationships']:
            parent = relationship['parent_table_name']
            child = relationship['child_table_name']
            if child in self.reference_table_names and parent not in self.reference_table_names:
                invalid_pairs.add((child, parent))

        if invalid_pairs:
            raise ConstraintNotApplicableError(
                'Reference tables cannot be children of non-reference tables. '
                f"The following child-parent pairs are invalid: '{sorted(invalid_pairs)}'"
            )

    def _validate_data(self, data, metadata=None):
        """No data validation needed for reference tables."""
        pass

    def _is_valid(self, data, metadata=None):
        """Get valid rows.

        All rows are valid.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.Series]:
                A dictionary mapping the table name to a Series where each row is=True or False
                depending on if it's valid.
        """
        if metadata is not None:
            self._validate_constraint_with_metadata(metadata)

        return _get_table_to_valid_rows(data)
