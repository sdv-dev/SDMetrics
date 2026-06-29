"""Single table reports for sdmetrics."""

__all__ = [
    'QualityReport',  # noqa: F822
    'DiagnosticReport',  # noqa: F822
]


def __getattr__(name):
    """Lazy load deprecated report classes."""
    if name == 'QualityReport':
        from sdmetrics.reports.single_table.quality_report import QualityReport

        return QualityReport

    if name == 'DiagnosticReport':
        from sdmetrics.reports.single_table.diagnostic_report import DiagnosticReport

        return DiagnosticReport

    raise AttributeError(f"module 'sdmetrics.reports.single_table' has no attribute '{name}'")
