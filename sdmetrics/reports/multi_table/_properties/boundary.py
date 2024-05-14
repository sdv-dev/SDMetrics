"""Boundary property for multi-table."""

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import Boundary as SingleTableBoundary


class Boundary(BaseMultiTableProperty):
    """Boundary property class for multi-table.

    This property assesses the boundary adherence of the synthetic data over the real data.
    The ``BoundaryAdherence`` metric is computed column-wise and the final score is the average
    over all columns. This metric is computed over numerical and datetime columns only.
    The other column types are ignored by this property.
    """

    _single_table_property = SingleTableBoundary
    _num_iteration_case = 'column'
