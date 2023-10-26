"""Data validity property for multi-table."""
from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import DataValidity as SingleTableDataValidity


class DataValidity(BaseMultiTableProperty):
    """Data Validitys property class for multi-table.

    This property computes, at base, whether each column contains valid data.
    The metric is based on the type data in each column.
    A metric score is computed column-wise and the final score is the average over all columns.
    The KSComplement metric is used for numerical and datetime columns while the TVComplement
    is used for categorical and boolean columns.
    The other column types are ignored by this property.
    """

    _single_table_property = SingleTableDataValidity
    _num_iteration_case = 'column'
