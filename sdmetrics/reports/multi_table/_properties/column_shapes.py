from sdmetrics.reports.single_table._properties import ColumnShapes as SingleTableColumnShapes
from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty

class ColumnShapes(BaseMultiTableProperty):

    def __init__(self):
        super().__init__()
        self._single_table_property = SingleTableColumnShapes
