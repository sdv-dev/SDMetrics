"""Synthesis property for multi-table."""

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import Synthesis as SingleTableSynthesis


class Synthesis(BaseMultiTableProperty):
    """Synthesis property class for multi-table.

    This property assesses the novelty of the syntetic data over the real data.
    The ``NewRowSynthesis`` metric is computed over the real and synthetic for each table
    to score the proportion of new rows in the synthetic data.
    The final score is the average over all tables.
    """

    _single_table_property = SingleTableSynthesis
    _num_iteration_case = 'table'
