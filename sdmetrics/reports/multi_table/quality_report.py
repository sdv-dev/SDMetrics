"""Multi table quality report."""

from sdmetrics.reports.multi_table._properties import (
    Cardinality,
    ColumnPairTrends,
    ColumnShapes,
    InterTableTrends,
)
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport


class QualityReport(BaseMultiTableReport):
    """Multi table quality report.

    This class creates a quality report for multi-table data. It calculates the quality
    score along three properties - Column Shapes, Column Pair Trends, and Cardinality.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends(),
            'Cardinality': Cardinality(),
            'Intertable Trends': InterTableTrends(),
        }
