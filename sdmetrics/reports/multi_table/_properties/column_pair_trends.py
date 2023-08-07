"""Column pair trends property for multi-table."""
from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import (
    ColumnPairTrends as SingleTableColumnPairTrends)


class ColumnPairTrends(BaseMultiTableProperty):
    """Column pair trends property for multi-table.

    This property evaluates the matching in trends between pairs of real
    and synthetic data columns. Each pair's correlation is calculated and
    the final score represents the average of these measures across all column pairs
    """

    _single_table_property = SingleTableColumnPairTrends

    def _get_num_iterations(self, metadata):
        num_columns = [len(table['columns']) for table in metadata['tables'].values()]
        return sum([(n_cols * (n_cols - 1)) // 2 for n_cols in num_columns])