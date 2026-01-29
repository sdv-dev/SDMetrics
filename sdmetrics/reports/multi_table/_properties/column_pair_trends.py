"""Column pair trends property for multi-table."""

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import (
    ColumnPairTrends as SingleTableColumnPairTrends,
)


class ColumnPairTrends(BaseMultiTableProperty):
    """Column pair trends property for multi-table.

    This property evaluates the matching in trends between pairs of real
    and synthetic data columns. Each pair's correlation is calculated and
    the final score represents the average of these measures across all column pairs
    """

    _single_table_property = SingleTableColumnPairTrends
    _num_iteration_case = 'column_pair'

    def __init__(self):
        super().__init__()
        self.real_correlation_threshold = 0
        self.real_association_threshold = 0

    def _compute_average(self):
        """Average the scores for each column pair, honoring contribution rules."""
        return self._compute_average_with_threshold('Meets Threshold?')

    def _configure_single_table_property(self, table_name):
        self._properties[table_name].real_correlation_threshold = self.real_correlation_threshold
        self._properties[table_name].real_association_threshold = self.real_association_threshold
