"""Coverage property for multi-table."""

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import Coverage as SingleTableCoverage


class Coverage(BaseMultiTableProperty):
    """Coverage property class for multi-table.

    This property assesses data coverage between the real and synthetic data.
    A metric score is computed column-wise and the final score is the average over all columns.
    The ``RangeCoverage`` metric is used for numerical and datetime columns while the
    ``CategoryCoverage`` is used for categorical and boolean columns.
    The other column types are ignored by this property.
    """

    _single_table_property = SingleTableCoverage
    _num_iteration_case = 'column'
