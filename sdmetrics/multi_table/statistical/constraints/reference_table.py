"""Reference Table Constraint."""

import pandas as pd

from sdmetrics.multi_table.statistical.constraints._utils import (
    _get_row_tuples,
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
        self._reference_rows = None
        self._reference_columns = None

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
        """Check that every reference table is in the data and kept its columns.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Raises:
            ConstraintNotApplicableError:
                If a reference table is missing from the data, or if it does not have the
                same columns that it has in the real data.
        """
        for table_name in self.reference_table_names:
            if table_name not in data:
                raise ConstraintNotApplicableError(
                    f"The table '{table_name}' is missing from the data."
                )

            if self._reference_columns is None:
                continue

            if set(data[table_name].columns) != set(self._reference_columns[table_name]):
                raise ConstraintNotApplicableError(
                    f"The columns of the table '{table_name}' do not match the ones of the "
                    'real data.'
                )

    def _fit(self, data, metadata=None):
        """Learn the rows of the reference table.

        Args:
            data (dict[str, pd.DataFrame]):
                A dictionary mapping each table name to its real data.
            metadata (dict):
                The multi table metadata.
        """
        self._reference_columns = {
            table_name: list(data[table_name].columns) for table_name in self.reference_table_names
        }
        self._reference_rows = {
            table_name: set(_get_row_tuples(data[table_name], self._reference_columns[table_name]))
            for table_name in self.reference_table_names
        }

    def _get_scored_tables(self, metadata=None):
        return set(self.reference_table_names)

    def _is_valid(self, data, metadata=None):
        """Get valid rows.

        A row of a reference table is valid when the real data holds that same row. The
        rows of every other table are valid, since this constraint does not check them.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.Series]:
                A dictionary mapping the table name to a Series where each row is=True or False
                depending on if it's valid.
        """
        if not self._fitted:
            raise ConstraintNotApplicableError(
                'ReferenceTable constraint must be called with ``fit`` first.'
            )

        if metadata is not None:
            self._validate_constraint_with_metadata(metadata)

        table_to_valid_rows = _get_table_to_valid_rows(data)
        for table_name in self.reference_table_names:
            table_data = data[table_name]
            reference_rows = self._reference_rows[table_name]
            rows = _get_row_tuples(table_data, self._reference_columns[table_name])
            table_to_valid_rows[table_name] = pd.Series(
                [row in reference_rows for row in rows], index=table_data.index, dtype=bool
            )

        return table_to_valid_rows
