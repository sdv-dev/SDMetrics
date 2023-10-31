"""Structure property for multi-table."""
from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import Structure as SingleTableStructure


class Structure(BaseMultiTableProperty):
    """Structure property class for multi-table.

    This property checks to see whether the overall structure of the synthetic
    data is the same as the real data. The property is calculated for each table.
    """

    _single_table_property = SingleTableStructure
    _num_iteration_case = 'table'
