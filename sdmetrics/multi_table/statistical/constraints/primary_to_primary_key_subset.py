"""Primary To Primary Key Subset Constraint."""

from sdmetrics.multi_table.statistical.constraints._utils import (
    _cast_to_iterable,
    _create_unique_name,
    _get_primary_key,
    _get_table_to_valid_rows,
)
from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


class PrimaryToPrimaryKeySubset(BaseConstraint):
    """Constraint for tables that hold a subset of the rows of a main table.

    Args:
        main_table_name (str):
            The name of the main table, the one that holds every possible row.
        conditional_column_name (str):
            The name of the column of the main table that controls whether a connection
            is allowed.
        relationships (dict):
            A dictionary that maps the name of every connected table to the list of
            conditional values that allow the connection.
    """

    _is_single_table = False

    @staticmethod
    def _validate_inputs(main_table_name, conditional_column_name, relationships):
        if not all(isinstance(value, str) for value in [main_table_name, conditional_column_name]):
            raise ValueError('`main_table_name` and `conditional_column_name` must be strings')

        if not isinstance(relationships, dict) or not all(
            isinstance(k, str) and isinstance(v, list) for k, v in relationships.items()
        ):
            raise ValueError(
                '`relationships` must be a a dict that maps the name of the connected table to a '
                'list of values that are acceptable for a connection to be made.'
            )

    def __init__(self, main_table_name, conditional_column_name, relationships):
        super().__init__()
        self._validate_inputs(main_table_name, conditional_column_name, relationships)
        self.main_table_name = main_table_name
        self.conditional_column_name = conditional_column_name
        self.relationships = relationships

    def _validate_data(self, data, metadata=None):
        """Check that every table and all the referenced columns exist in the data."""
        if self.main_table_name not in data:
            raise ConstraintNotApplicableError(
                f"The table '{self.main_table_name}' is missing from the data."
            )

        main_columns = data[self.main_table_name].columns
        main_primary_key = _get_primary_key(metadata, self.main_table_name)
        missing_columns = [
            column_name
            for column_name in [main_primary_key, self.conditional_column_name]
            if column_name not in main_columns
        ]
        if missing_columns:
            missing_columns = "', '".join(missing_columns)
            raise ConstraintNotApplicableError(
                f"The column(s) '{missing_columns}' are missing from the table "
                f"'{self.main_table_name}'."
            )

        for table_name in self.relationships:
            if table_name not in data:
                raise ConstraintNotApplicableError(
                    f"The table '{table_name}' is missing from the data."
                )

            primary_key = _get_primary_key(metadata, table_name)
            if primary_key not in data[table_name].columns:
                raise ConstraintNotApplicableError(
                    f"The column(s) '{primary_key}' are missing from the table '{table_name}'."
                )

    def _get_metadata_parameters(self, metadata):
        """Get the metadata parameters for the constraint.

        Return all the necessary metadata parameters to compute the updated metadata:
        - table_to_pk: A dictionary that maps the table name to its primary key.
        - main_table_columns: The columns of the main table.
        - tables_to_column_names: A dictionary that maps the table name to the column names mapping
            of its related table. The column names mapping is a dictionary that maps the original
            column names to the new column names after merging the related table
            into the main table.

        Args:
            metadata (dict):
                The input metadata for the constraint.
        """
        table_to_pk = {}
        table_to_pk[self.main_table_name] = _get_primary_key(metadata, self.main_table_name)
        main_table_columns = metadata['tables'][self.main_table_name]['columns'].keys()
        tables_to_column_names = {}
        main_table_columns = list(main_table_columns)
        existing_column_names = list(main_table_columns)
        for table_name in self.relationships:
            table_to_pk[table_name] = _get_primary_key(metadata, table_name)
            table_columns = metadata['tables'][table_name]['columns'].keys()
            column_names_to_merge = [f'{table_name}_{column_name}' for column_name in table_columns]
            column_names_to_merge = [
                _create_unique_name(column_name, existing_column_names)
                for column_name in column_names_to_merge
            ]
            conditional_value_to_column_name = dict(zip(table_columns, column_names_to_merge))
            for pk_col in _cast_to_iterable(table_to_pk[table_name]):
                del conditional_value_to_column_name[pk_col]

            tables_to_column_names[table_name] = conditional_value_to_column_name
            existing_column_names.extend(column_names_to_merge)

        return table_to_pk, main_table_columns, tables_to_column_names

    def _is_valid(self, data, metadata=None):
        """Get all valid rows.

        A valid row is a row in the related table that has primary key value that is a subset
        of the main table's primary key values and the condition matches.

        Args:
            data (dict <str: pandas.DataFrame>):
                Table data.

        Returns:
            dict <str: pandas.Series>:
                A dictionary mapping the table name to a Series where each row is=True or False
                depending on if it's valid.
        """
        table_to_pks, _, _ = self._get_metadata_parameters(metadata)
        table_to_valid_rows = _get_table_to_valid_rows(data)
        main_table_pk = _cast_to_iterable(table_to_pks[self.main_table_name])
        main_table_keys = data[self.main_table_name][main_table_pk]
        conditional_col = data[self.main_table_name][self.conditional_column_name]
        for table_name, conditional_values in self.relationships.items():
            valid_primary_key_values = main_table_keys[conditional_col.isin(conditional_values)]
            table_pk = _cast_to_iterable(table_to_pks[table_name])
            indicator_col = _create_unique_name('_merge', table_pk + main_table_pk)
            valid_pk_mask = (
                data[table_name][table_pk]
                .astype('object')
                .merge(
                    valid_primary_key_values.astype('object'),
                    left_on=table_pk,
                    right_on=main_table_pk,
                    how='left',
                    indicator=indicator_col,
                )
            )
            table_to_valid_rows[table_name][valid_pk_mask[indicator_col] == 'left_only'] = False

        return table_to_valid_rows
