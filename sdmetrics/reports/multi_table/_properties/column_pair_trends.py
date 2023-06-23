from sdmetrics.reports.single_table._properties import ColumnPairTrends as SingleTableColumnPairTrends
from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty


class ColumnPairTrends(BaseMultiTableProperty):

    def __init__(self):
        super().__init__()
        self._single_table_property = SingleTableColumnPairTrends
