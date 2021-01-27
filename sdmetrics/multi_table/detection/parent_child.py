"""Base class for Machine Learning Detection metrics that work on parent-child pairs of tables."""

import numpy as np

from sdmetrics.multi_table.detection.base import DetectionMetric
from sdmetrics.single_table.detection import LogisticDetection, SVCDetection
from sdmetrics.utils import NestedAttrsMeta


class ParentChildDetectionMetric(DetectionMetric,
                                 metaclass=NestedAttrsMeta('single_table_metric')):
    """Base class for Multi-table Detection metrics based on parent-child relationships.

    These metrics denormalize the parent-child relationships from the dataset and then
    apply a Single Table Detection metric on the resulting tables.

    The output of the metric is one minus the average ROC AUC score obtained.

    A part from the real and synthetic data, these metrics need to be passed
    a list with the foreign key relationships that exist between the tables.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        single_table_metric (sdmetrics.single_table.detection.base.DetectionMetric):
            The single table detection metric to use.
    """

    single_table_metric = None

    @staticmethod
    def _extract_foreign_keys(metadata):
        if not isinstance(metadata, dict):
            metadata = metadata.to_dict()

        foreign_keys = []
        for child_table, child_meta in metadata['tables'].items():
            for child_key, field_meta in child_meta['fields'].items():
                ref = field_meta.get('ref')
                if ref:
                    foreign_keys.append((ref['table'], ref['field'], child_table, child_key))

        return foreign_keys

    @staticmethod
    def _denormalize(data, foreign_key):
        """Denormalize the child table over the parent."""
        parent_table, parent_key, child_table, child_key = foreign_key

        flat = data[parent_table].set_index(parent_key).merge(
            data[child_table].set_index(child_key),
            how='outer',
            left_index=True,
            right_index=True,
        ).reset_index(drop=True)

        return flat

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, foreign_keys=None):
        """Compute this metric.

        This denormalizes the parent-child relationships from the dataset and then
        applies a Single Table Detection metric on the resulting tables.

        The output of the metric is one minus the average ROC AUC score obtained.

        A part from the real and synthetic data, either a ``foreign_keys`` list
        containing the relationships between the tables or a ``metadata`` that can be
        used to create such list must be passed.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset.
            metadata (dict):
                Multi-table metadata dict. If not passed, foreign keys must be
                passed.
            foreign_keys (list[tuple[str, str, str, str]]):
                List of foreign key relationships specified as tuples
                that contain (parent_table, parent_key, child_table, child_key).
                Ignored if metada is given.

        Returns:
            float:
                Average of the scores obtained by the single table metric.
        """
        if metadata:
            foreign_keys = cls._extract_foreign_keys(metadata)
        if not foreign_keys:
            raise ValueError('No foreign keys given')

        scores = []
        for foreign_key in foreign_keys:
            real = cls._denormalize(real_data, foreign_key)
            synth = cls._denormalize(synthetic_data, foreign_key)
            scores.append(cls.single_table_metric.compute(real, synth))

        return np.mean(scores)


class LogisticParentChildDetection(ParentChildDetectionMetric):
    """ParentChild detection metric based on a LogisticRegression."""

    single_table_metric = LogisticDetection


class SVCParentChildDetection(ParentChildDetectionMetric):
    """ParentChild detection metric based on a SVC."""

    single_table_metric = SVCDetection
