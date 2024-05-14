"""Reports for sdmetrics."""

from sdmetrics.reports.multi_table import DiagnosticReport as MultiTableDiagnosticReport
from sdmetrics.reports.multi_table import QualityReport as MultiTableQualityReport
from sdmetrics.reports.single_table import DiagnosticReport as SingleTableDiagnosticReport
from sdmetrics.reports.single_table import QualityReport as SingleTableQualityReport

__all__ = [
    'SingleTableQualityReport',
    'SingleTableDiagnosticReport',
    'MultiTableQualityReport',
    'MultiTableDiagnosticReport',
]
