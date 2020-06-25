"""
This module provides simulated datasets than can be used to experiment with
the SDMetrics library.
"""
import os
from glob import glob

import pandas as pd
from sdv import Metadata

_DIR_ = os.path.dirname(__file__)


def list_datasets():
    """
    This function returns the list of datasets that are built-in. These
    dataset names can be passed to `Dataset.load`.

    Returns:
        (List[str]): A list of dataset names.
    """
    datasets = []
    for path_to_metadata in glob(os.path.join(_DIR_, "**/metadata.json")):
        path_to_dataset = os.path.dirname(path_to_metadata)
        dataset_name = os.path.basename(path_to_dataset)
        datasets.append(dataset_name)
    return datasets


class Dataset():
    """
    The Dataset object represents a simulated dataset with metadata, real data, and two
    tiers of synthetic data.

    Attributes:
        metadata (str): The SDV Metadata object.
        tables (Dict[str, DataFrame]): A mapping from table names to the real tables.
        lq_synthetic (Dict[str, DataFrame]): A low quality synthetic copy of the tables.
        hq_synthetic (Dict[str, DataFrame]): A high quality synthetic copy of the tables.
    """

    def __init__(self, metadata, tables, lq_synthetic, hq_synthetic):
        self.metadata = metadata
        self.tables = tables
        self.lq_synthetic = lq_synthetic
        self.hq_synthetic = hq_synthetic

    @staticmethod
    def load(dataset, is_path=False):
        """This function loads a SDMetrics dataset which consists of a metadata
        object, a set of real tables, a set of low quality synthetic tables, and
        a set of high quality synthetic tables.

        Arguments:
            dataset (str): The name of the dataset (or the path to the dataset).

        Returns:
            (Dataset): An instance of the Dataset object.
        """
        if is_path:
            path_to_dataset = dataset
        else:
            path_to_dataset = os.path.join(_DIR_, dataset)

        metadata = Metadata(os.path.join(path_to_dataset, "metadata.json"))
        tables = Dataset._load_tables(os.path.join(path_to_dataset))
        lq_synthetic = Dataset._load_tables(os.path.join(path_to_dataset, "low_quality"))
        hq_synthetic = Dataset._load_tables(os.path.join(path_to_dataset, "high_quality"))
        return Dataset(metadata, tables, lq_synthetic, hq_synthetic)

    def save(self, path_to_dataset):
        """This exports the dataset to disk at the specified directory.

        Arguments:
            path_to_dataset (str): The location to store the dataset.
        """
        self.metadata.to_json(os.path.join(path_to_dataset, "metadata.json"))
        self._save_tables(path_to_dataset, self.tables)
        self._save_tables(os.path.join(path_to_dataset, "low_quality"), self.lq_synthetic)
        self._save_tables(os.path.join(path_to_dataset, "high_quality"), self.hq_synthetic)

    @staticmethod
    def _load_tables(path_to_tables):
        tables = {}
        for path_to_csv in glob(os.path.join(path_to_tables, "*.csv")):
            table_name = os.path.basename(path_to_csv).replace(".csv", "")
            tables[table_name] = pd.read_csv(path_to_csv)
        return tables

    @staticmethod
    def _save_tables(path_to_tables, tables):
        for table_name, df in tables.items():
            df.to_csv(os.path.join(path_to_tables, "%s.csv" % table_name), index=False)
