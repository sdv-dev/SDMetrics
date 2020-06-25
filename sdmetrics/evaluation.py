# -*- coding: utf-8 -*-

"""Evaluation module."""

from sdmetrics import constraint, detection, statistical
from sdmetrics.report import MetricsReport


def _validate(metadata, real, synthetic):
    """
    This checks to make sure the real and synthetic databases correspond to
    the given metadata object.
    """
    metadata.validate(real)
    metadata.validate(synthetic)


def _metrics(metadata, real, synthetic):
    """
    This function takes in (1) a `sdv.Metadata` object which describes a set of
    relational tables, (2) a set of "real" tables corresponding to the metadata,
    and (3) a set of "synthetic" tables corresponding to the metadata. It yields
    a sequence of `Metric` objects.

    Args:
        metadata (sdv.Metadata): The Metadata object from SDV.
        real_tables (dict): A dictionary mapping table names to dataframes.
        synthetic_tables (dict): A dictionary mapping table names to dataframes.

    Yields:
        Metric: The next metric.
    """
    _validate(metadata, real, synthetic)

    yield from constraint.metrics(metadata, real, synthetic)
    yield from detection.metrics(metadata, real, synthetic)
    yield from statistical.metrics(metadata, real, synthetic)


def evaluate(metadata, real, synthetic):
    """
    This generates a MetricsReport for the given metadata and tables with the
    default/built-in metrics.

    Args:
        metadata (sdv.Metadata): The Metadata object from SDV.
        real_tables (dict): A dictionary mapping table names to dataframes.
        synthetic_tables (dict): A dictionary mapping table names to dataframes.

    Returns:
        MetricsReport: A report containing the default metrics.
    """
    _validate(metadata, real, synthetic)
    report = MetricsReport()
    report.add_metrics(_metrics(metadata, real, synthetic))
    return report
