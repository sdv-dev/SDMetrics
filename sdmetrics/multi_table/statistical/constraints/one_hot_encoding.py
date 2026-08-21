"""OneHotEncoding Constraint."""


from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError
from sdmetrics.multi_table.statistical.constraints.utils import _get_is_valid_dict


class OneHotEncoding(BaseConstraint):
    """Constraint for a table one hot encoded columns.

    Args:
        column_names (list[str]):
            Names of the columns containing one hot rows.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
    """
    def __init__(self, column_names, table_name=None):
        super().__init__()
        if not isinstance(table_name, str):
            raise ValueError("The 'table_name' parameter must be a string.")

        is_list_of_strings = isinstance(column_names, list) and all(
            isinstance(column_name, str) for column_name in column_names
        )
        if not is_list_of_strings:
            raise ValueError(
                "The 'column_names' parameter must be a list of strings."
            )

        self._column_names = column_names
        self.table_name = table_name

    def _validate_data(self, data, metadata=None):
        """Check that the table and all the referenced columns exist in the data."""
        if self.table_name not in data:
            raise ConstraintNotApplicableError(
                f"The table '{self.table_name}' is missing from the data."
            )

        columns = data[self.table_name].columns
        missing_columns = [
            column_name
            for column_name in self._column_names
            if column_name not in columns
        ]
        if missing_columns:
            missing_columns = "', '".join(missing_columns)
            raise ConstraintNotApplicableError(
                f"The column(s) '{missing_columns}' are missing from the table "
                f"'{self.table_name}'."
            )

    def _get_valid_table_data(self, table_data):
        one_hot_data = table_data[self._column_names]

        sum_one = one_hot_data.sum(axis=1) == 1.0
        max_one = one_hot_data.max(axis=1) == 1.0
        min_zero = one_hot_data.min(axis=1) == 0.0
        no_nans = ~one_hot_data.isna().any(axis=1)

        return sum_one & max_one & min_zero & no_nans

    def _is_valid(self, data, metadata=None):
        """Check whether the data satisfies the one-hot constraint.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.Series]:
                Whether each row is valid.
        """
        table_name = self._get_single_table_name(metadata)
        is_valid = _get_is_valid_dict(data, table_name)
        is_valid[table_name] = self._get_valid_table_data(data[table_name])

        return is_valid
