"""
This module implements statistical methods for comparing the distributions of
the two databases.
"""
from sdmetrics.statistical.bivariate import ContinuousDivergence, DiscreteDivergence
from sdmetrics.statistical.univariate import CSTest, KSTest


def metrics(metadata, real_tables, synthetic_tables):
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
    for method in [CSTest(), KSTest(), DiscreteDivergence(), ContinuousDivergence()]:
        yield from method.metrics(metadata, real_tables, synthetic_tables)
