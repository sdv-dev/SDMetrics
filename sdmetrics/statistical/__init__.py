"""
This module implements statistical methods for comparing the distributions of
the two databases.
"""
from .bivariate import ContinuousDivergence, DiscreteDivergence
from .univariate import CSTest, KSTest

methods = [CSTest(), KSTest(), DiscreteDivergence(), ContinuousDivergence()]


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
    for method in methods:
        yield from method.metrics(metadata, real_tables, synthetic_tables)
