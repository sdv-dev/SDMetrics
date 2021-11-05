"""Functions to load demos with real and synthetic data of different data modalities."""

import pathlib
import pickle


def load_demo(modality='multi_table'):
    """Load demo data of the indicated data modality.

    By default, multi_table demo is loaded.

    Output is the real data, the synthetic data and the metadata dict.

    Args:
        modality (str):
            Data modality to load. It can be multi_table, single_table
            or timeseries.

    Returns:
        tuple:
            Real data, Synthetic data, Metadata.
    """
    demo_path = pathlib.Path(__file__).parent / 'demos' / f'{modality}.pkl'
    with open(demo_path, 'rb') as demo_file:
        return pickle.load(demo_file)


def load_multi_table_demo():
    """Load multi-table demo data.

    The dataset is the ``SDV`` demo data, which consists of three
    tables, ``users``, ``sessions`` and ``transactions``, with
    simulated data about user browsing sessions and transactions
    made during those sessions, and a synthetic copy of it made
    by the ``sdv.relational.HMA1`` model.

    Returns:
        tuple:
            * dict: Real tables.
            * dict: Synthetic tables.
            * dict: Dataset Metadata.
    """
    return load_demo('multi_table')


def load_single_table_demo():
    """Load multi-table demo data.

    The dataset is the ``student_placements`` tabular demo from SDV
    and a synthetic copy of it made using he ``sdv.tabular.CTGAN``
    model.

    Returns:
        tuple:
            * pandas.DataFrame: Real table.
            * pandas.DataFrame: Synthetic table.
            * dict: Table Metadata.
    """
    return load_demo('single_table')


def load_timeseries_demo():
    """Load time series demo data.

    The dataset is the ``sunglasses`` demo data from the DeepEcho
    project, which contains simulated data from a chain of sunglasses
    stores, and a synthetic copy of it made by the ``sdv.timeseries.PAR``
    model.

    It has 1 entity column, 1 context column and a datetime sequence index.

    Returns:
        tuple:
            * pandas.DataFrame: Real table.
            * pandas.DataFrame: Synthetic table.
            * dict: Table Metadata.
    """
    return load_demo('timeseries')
