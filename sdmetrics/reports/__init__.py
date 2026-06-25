"""Reports for sdmetrics."""

from sdmetrics.reports.multi_table import DiagnosticReport as MultiTableDiagnosticReport
from sdmetrics.reports.multi_table import QualityReport as MultiTableQualityReport
from sdmetrics.reports.single_table import DiagnosticReport as SingleTableDiagnosticReport
from sdmetrics.reports.single_table import QualityReport as SingleTableQualityReport
from sdmetrics.reports.diagnostic_report import DiagnosticReport
from sdmetrics.reports.quality_report import QualityReport

__all__ = [
    'DiagnosticReport',
    'QualityReport',
    'SingleTableQualityReport',
    'SingleTableDiagnosticReport',
    'MultiTableQualityReport',
    'MultiTableDiagnosticReport',
]
