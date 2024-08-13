"""Column shapes property for multi-table."""

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import ColumnShapes as SingleTableColumnShapes


class ColumnShapes(BaseMultiTableProperty):
    """Column Shapes property class for multi-table.

    This property assesses the shape similarity between the real and synthetic data.
    A metric score is computed column-wise and the final score is the average over all columns.
    The KSComplement metric is used for numerical and datetime columns while the TVComplement
    is used for categorical and boolean columns.
    The other column types are ignored by this property.
    """

    _single_table_property = SingleTableColumnShapes
    _num_iteration_case = 'column'
