"""Reports for sdmetrics."""

from sdmetrics.reports.diagnostic_report import DiagnosticReport
from sdmetrics.reports.quality_report import QualityReport

__all__ = [
    'DiagnosticReport',
    'QualityReport',
    'SingleTableQualityReport',  # noqa: F822
    'SingleTableDiagnosticReport',  # noqa: F822
    'MultiTableQualityReport',  # noqa: F822
    'MultiTableDiagnosticReport',  # noqa: F822
]


def __getattr__(name):
    """Lazy load deprecated report class aliases."""
    if name == 'SingleTableQualityReport':
        from sdmetrics.reports.single_table.quality_report import QualityReport

        return QualityReport

    if name == 'SingleTableDiagnosticReport':
        from sdmetrics.reports.single_table.diagnostic_report import DiagnosticReport

        return DiagnosticReport

    if name == 'MultiTableQualityReport':
        from sdmetrics.reports.multi_table.quality_report import QualityReport

        return QualityReport

    if name == 'MultiTableDiagnosticReport':
        from sdmetrics.reports.multi_table.diagnostic_report import DiagnosticReport

        return DiagnosticReport

    raise AttributeError(f"module 'sdmetrics.reports' has no attribute '{name}'")
