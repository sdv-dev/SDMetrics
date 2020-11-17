"""
This module implements constraint checking which makes sure the statistical
properties of the synthetic data match the specified metadata.
"""
from sdmetrics.report import Goal, Metric


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
    for table_name in set(real_tables):
        key = metadata.get_primary_key(table_name)
        for child_name in metadata.get_children(table_name):
            child_key = metadata.get_foreign_key(table_name, child_name)

            parent_keys = set(synthetic_tables[table_name][key].values)
            child_keys = set(synthetic_tables[child_name][child_key].values)

            yield Metric(
                name="foreign-key",
                value=float(parent_keys.issuperset(child_keys)),
                tags=set([
                    "table:%s" % table_name,
                    "child:%s" % child_name,
                ]),
                goal=Goal.MAXIMIZE,
                unit="binary",
                domain=(0.0, 1.0)
            )
