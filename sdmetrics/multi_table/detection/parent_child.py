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
    def _denormalize(parent_table, parent_key, child_table, child_key):
        """Denormalize the child table over the parent."""
        flat = parent_table.merge(
            child_table,
            how='outer',
            left_on=parent_key,
            right_on=child_key
        )

        del flat[parent_key]
        if child_key != parent_key:
            del flat[child_key]

        return flat

    @classmethod
    def compute(cls, real_data, synthetic_data, foreign_keys):
        """Compute this metric.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset.
            foreign_keys (list[tuple[str, str, str, str]]):
                List of foreign key relationships specified as tuples
                that contain (parent_table, parent_key, child_table, child_key).

        Returns:
            float:
                One minus the ROC AUC Score obtained by the classifier.
        """
        scores = []
        for parent_table, parent_key, child_table, child_key in foreign_keys:
            real = cls._denormalize(
                real_data[parent_table],
                parent_key,
                real_data[child_table],
                child_key
            )
            synth = cls._denormalize(
                synthetic_data[parent_table],
                parent_key,
                synthetic_data[child_table],
                child_key
            )
            scores.append(cls.single_table_metric.compute(real, synth))

        return np.mean(scores)


class LogisticParentChildDetection(ParentChildDetectionMetric):

    single_table_metric = LogisticDetection


class SVCParentChildDetection(ParentChildDetectionMetric):

    single_table_metric = SVCDetection
