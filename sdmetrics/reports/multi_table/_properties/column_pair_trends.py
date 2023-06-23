"""Column pair trends property for multi-table."""
from sdmetrics.reports.single_table._properties import (
    ColumnPairTrends as SingleTableColumnPairTrends)
from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty


class ColumnPairTrends(BaseMultiTableProperty):
    """Column pair trends property for multi-table.

    This property evaluates the matching in trends between pairs of real
    and synthetic data columns. Each pair's correlation is calculated and
    the final score represents the average of these measures across all column pairs
    """

    def __init__(self):
        super().__init__()
        self._single_table_property = SingleTableColumnPairTrends
