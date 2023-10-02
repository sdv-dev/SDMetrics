"""Reports for sdmetrics."""
from sdmetrics.reports.multi_table import DiagnosticReport as MultiTableDiagnosticReport
from sdmetrics.reports.multi_table import QualityReport as MultiTableQualityReport
from sdmetrics.reports.single_table import DiagnosticReport as SingleTableDiagnosticReport
from sdmetrics.reports.single_table import QualityReport as SingleTableQualityReport
from sdmetrics.reports.utils import get_column_pair_plot

__all__ = [
    'get_column_pair_plot',
    'SingleTableQualityReport',
    'SingleTableDiagnosticReport',
    'MultiTableQualityReport',
    'MultiTableDiagnosticReport',
]
