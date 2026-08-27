"""Self Referential Hierarchy Constraint."""

import pandas as pd

from sdmetrics.multi_table.statistical.constraints._utils import (
    _create_unique_name,
    _get_table_to_valid_rows,
)
from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


def _values_match(expected_value, value):
    """Say whether two values are the same, treating two missing values as equal."""
    if pd.isna(expected_value) and pd.isna(value):
        return True

    return expected_value == value


class SelfReferentialHierarchy(BaseConstraint):
    """Constraint for a table that references itself.

    Args:
        table_name (str):
            The name of the table that references itself.
        base_column_name (str):
            The name of the column that the reference is based on. This may be the
            primary key.
        parent_column_name (str):
            The name of the column that holds the parent of the base column.
        grandparent_column_name (str, optional):
            The name of the column that holds the grandparent of the base column.
            Defaults to None.
        root_column_name (str, optional):
            The name of the column that holds the root of the tree. Defaults to None.
        scaling_method (str, optional):
            How the synthetic data grows the hierarchy. One of ``'branch'``, ``'depth'``
            or ``'multiply'``. It does not affect the validity of any row.
            Defaults to ``'branch'``.
    """

    def __init__(
        self,
        table_name,
        base_column_name,
        parent_column_name,
        grandparent_column_name=None,
        root_column_name=None,
        scaling_method='branch',
    ):
        super().__init__()

        if not all(
            isinstance(value, str) for value in [table_name, base_column_name, parent_column_name]
        ):
            raise TypeError(
                'The `table_name`, `base_column_name` and `parent_column_name` '
                'must be all be strings.'
            )

        if not all(
            (value is None or isinstance(value, str))
            for value in [grandparent_column_name, root_column_name]
        ):
            raise TypeError(
                'The `grandparent_column_name` and `root_column_name` '
                'must be all be strings or `None`.'
            )

        if base_column_name == parent_column_name:
            raise ValueError(
                'The `base_column_name` and `parent_column_name` must be different columns.'
            )

        if scaling_method not in ['branch', 'depth', 'multiply']:
            raise ValueError(
                f"Unrecognized scaling_method '{scaling_method}'. The scaling method "
                "must be one of 'branch', 'depth' or 'multiply'."
            )

        self.table_name = table_name
        self._base_column = base_column_name
        self._parent_column = parent_column_name
        self._grandparent_column = grandparent_column_name
        self._root_column = root_column_name
        self._scaling_method = scaling_method

    def _validate_data(self, data, metadata=None):
        """Check that the table and all the referenced columns exist in the data."""
        if self.table_name not in data:
            raise ConstraintNotApplicableError(
                f"The table '{self.table_name}' is missing from the data."
            )

        columns_to_check = {
            'base': self._base_column,
            'parent': self._parent_column,
        }
        if self._grandparent_column:
            columns_to_check['grandparent'] = self._grandparent_column

        if self._root_column:
            columns_to_check['root'] = self._root_column

        columns = data[self.table_name].columns
        missing_columns = [
            column_name for column_name in columns_to_check.values() if column_name not in columns
        ]
        if missing_columns:
            missing_columns = "', '".join(missing_columns)
            raise ConstraintNotApplicableError(
                f"The column(s) '{missing_columns}' are missing from the table '{self.table_name}'."
            )

    def _get_grandparent_column(self, data):
        """Get the actual grandparent column from the data."""
        index_col = _create_unique_name('_index', data.columns)
        data = data.rename_axis(index_col).reset_index().set_index(self._base_column)
        roots = (data.index == data[self._parent_column]) | (data[self._parent_column].isna())
        root_children = data[self._parent_column].isin(data[roots].index) | (
            ~data[self._parent_column].isin(data.index)
        )
        base_index = data[~root_children][index_col]
        parent_values = data[~root_children][self._parent_column]
        grandparent_values = data.loc[parent_values].set_index(base_index)
        grandparent_values = grandparent_values[self._parent_column].reindex(data[index_col])
        return grandparent_values.reset_index(drop=True)

    def _validate_grandparent_column(self, data):
        """Validate that every grandparent points to the parent's parent."""
        actual_grandparents = self._get_grandparent_column(data)
        grandparent_column = data[self._grandparent_column]
        grandparent_match = (actual_grandparents == grandparent_column) | (
            actual_grandparents.isna() & grandparent_column.isna()
        )

        return ~grandparent_match

    def _get_root_column(self, data, root_nodes):
        """Get the actual root column from the data."""
        index_col = _create_unique_name('_index', data.columns)
        data = data.rename_axis(index_col).reset_index().set_index(self._base_column)

        known_root_values = {root_node: root_node for root_node in root_nodes}

        def get_root_node(row, known_roots):
            """Recursively traverse up the hierarchy to find the root node."""
            if row.name in known_roots:
                return known_roots[row.name]

            if row[self._parent_column] in root_nodes:
                known_roots[row.name] = row[self._parent_column]
                return row[self._parent_column]

            if row[self._parent_column] in data.index:
                parent_row = data.loc[row[self._parent_column]]
                root_value = get_root_node(parent_row, known_roots)
                known_roots[row.name] = root_value
                return root_value

        data['root'] = data.apply(get_root_node, axis=1, known_roots=known_root_values)
        return data.set_index(index_col)['root'].reset_index(drop=True)

    def _validate_root_column(self, data, root_nodes):
        """Validate that every root value points to the root of the tree."""
        non_root_rows = data[~data[self._base_column].isin(root_nodes)][self._root_column]
        actual_roots = self._get_root_column(data, root_nodes).loc[non_root_rows.index]
        root_node_match = non_root_rows == actual_roots
        invalid_rows = pd.Series(False, index=data.index)
        invalid_rows.loc[root_node_match.index] = ~root_node_match

        return invalid_rows

    @staticmethod
    def _create_referential_tree(base_column, parent_column):
        """Create the referential hierarchy tree.

        Builds the referential tree by starting from the root nodes and
        traversing references to assign each referenced base value to a
        depth. Also returns a dictionary containing the minimum/maximum
        number of references (branches) for base values at that depth.

        Args:
            base_column (pd.Series):
                The base column.
            parent_column (pd.Series):
                The parent column.

        Returns:
            tuple:
                * dict:
                    Dictionary mapping each base value to its associated depth.
                * dict:
                    Dictionary mapping a depth to the min/max branching factor at that depth.
        """
        depth_counter = 0
        tree_depth = {}
        branch_factor = {}
        null_roots = set(base_column[parent_column.isna()])
        self_roots = set(base_column[base_column == parent_column])
        missing_roots = set(parent_column[~parent_column.isin(base_column)].dropna())
        next_depth = set.union(null_roots, self_roots, missing_roots)
        while next_depth:
            tree_depth.update({base: depth_counter for base in next_depth})
            next_depth_references = parent_column.isin(next_depth) & ~base_column.isin(next_depth)
            reference_counts = parent_column[next_depth_references].value_counts()
            unreferenced = set(next_depth) - set(reference_counts.index)
            if not reference_counts.empty:
                branch_factor[depth_counter] = (
                    int(reference_counts.min()) if not unreferenced else 0,
                    int(reference_counts.max()),
                )

            next_depth = set(base_column[next_depth_references])
            depth_counter += 1

        return tree_depth, branch_factor

    def _validate_self_referential_hierarchy(self, data):
        table_data = data[self.table_name]
        base_data = table_data[self._base_column]
        parent_data = table_data[self._parent_column]
        table_to_valid_rows = _get_table_to_valid_rows(data)
        nan_bases = base_data.isna()
        duplicated_bases = base_data.duplicated()
        if any(nan_bases):
            table_to_valid_rows[self.table_name].loc[nan_bases] = False

        if any(duplicated_bases):
            table_to_valid_rows[self.table_name].loc[duplicated_bases] = False

        filtered_base_data = base_data.loc[table_to_valid_rows[self.table_name]]
        filtered_parent_data = parent_data.loc[table_to_valid_rows[self.table_name]]
        hierarchy_tree_depths, _ = SelfReferentialHierarchy._create_referential_tree(
            filtered_base_data, filtered_parent_data
        )
        keys_in_loop = ~base_data.isin(hierarchy_tree_depths.keys())
        if any(keys_in_loop):
            table_to_valid_rows[self.table_name].loc[keys_in_loop] = False

        if self._grandparent_column:
            invalid_grandparent_rows = self._validate_grandparent_column(table_data)
            if not invalid_grandparent_rows.empty:
                table_to_valid_rows[self.table_name].loc[invalid_grandparent_rows] = False

        if self._root_column:
            root_nodes = {node for node, depth in hierarchy_tree_depths.items() if depth == 0}
            invalid_root_rows = self._validate_root_column(table_data, root_nodes)
            if not invalid_root_rows.empty:
                table_to_valid_rows[self.table_name].loc[invalid_root_rows] = False

        return table_to_valid_rows

    def _is_valid(self, data, metadata=None):
        """Check whether the data satisfies the self referential hierarchy constraint.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.
            metadata (dict):
                Metadata as a dictionary.

        Returns:
            dict[str, pd.Series]:
                Whether each row is valid.
        """
        return self._validate_self_referential_hierarchy(data)
