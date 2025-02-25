"""MultiTable metrics based on applying SingleTable metrics on all the tables."""

import warnings
from collections import defaultdict

import numpy as np

from sdmetrics import single_table
from sdmetrics._utils_metadata import _validate_multi_table_metadata
from sdmetrics.errors import IncomputableMetricError
from sdmetrics.multi_table.base import MultiTableMetric
from sdmetrics.utils import nested_attrs_meta
from sdmetrics.warnings import SDMetricsWarning


class MultiSingleTableMetric(MultiTableMetric, metaclass=nested_attrs_meta('single_table_metric')):
    """MultiTableMetric subclass that applies a SingleTableMetric on each table.

    This class can either be used by creating a subclass that inherits from it and
    sets the SingleTable Metric as the ``single_table_metric`` attribute,
    or by creating an instance of this class passing the underlying SingleTable
    metric as an argument.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        single_table_metric (sdmetrics.single_table.base.SingleTableMetric):
            SingleTableMetric to apply.
    """

    single_table_metric = None

    def __init__(self, single_table_metric):
        self.single_table_metric = single_table_metric
        self.compute = self._compute

    @staticmethod
    def _multitable_warning(caught_warnings, table_name):
        for warning in caught_warnings:
            if issubclass(warning.category, SDMetricsWarning):
                prefixes = ['The real data in', 'The synthetic data in']
                message = str(warning.message)
                for prefix in prefixes:
                    message = message.replace(prefix, f"{prefix[:-3]} in table '{table_name}',")

                warnings.warn(warning.category(message))

    def _compute(self, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric.

        This applies the underlying single table metric to all the tables
        found in the dataset and then returns the average score obtained.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset.
            metadata (dict):
                Multi-table metadata dict. If not passed, it is built based on the
                real_data fields and dtypes.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single table metric

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        if set(real_data.keys()) != set(synthetic_data.keys()):
            raise ValueError('`real_data` and `synthetic_data` must have the same tables')

        if metadata is None:
            metadata = {'tables': defaultdict(type(None))}

        _validate_multi_table_metadata(metadata)
        scores = {}
        errors = {}
        for table_name, real_table in real_data.items():
            synthetic_table = synthetic_data[table_name]
            table_meta = metadata['tables'][table_name]

            with warnings.catch_warnings(record=True) as caught_warnings:
                try:
                    score_breakdown = self.single_table_metric.compute_breakdown(
                        real_table, synthetic_table, table_meta, **kwargs
                    )
                    scores[table_name] = score_breakdown
                except AttributeError:
                    score = self.single_table_metric.compute(
                        real_table, synthetic_table, table_meta, **kwargs
                    )
                    scores[table_name] = score
                except Exception as error:
                    errors[table_name] = error

            if caught_warnings:
                self._multitable_warning(caught_warnings, table_name)

        if not scores:
            raise IncomputableMetricError(f'Encountered the following errors: {errors}')

        return scores

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric.

        This applies the underlying single table metric to all the tables
        found in the dataset and then returns the average score obtained.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset.
            metadata (dict):
                Multi-table metadata dict. If not passed, it is built based on the
                real_data fields and dtypes.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single table metric

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        scores = cls._compute(cls, real_data, synthetic_data, metadata, **kwargs)
        scores = list(scores.values())
        if len(scores) > 0 and isinstance(scores[0], dict):
            all_scores = []
            for table_scores in scores:
                if 'score' in table_scores:
                    all_scores.append(table_scores['score'])
                else:
                    all_scores.extend([
                        result['score'] for result in table_scores.values() if 'score' in result
                    ])

            scores = all_scores

        return np.nanmean(scores)

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric broken down by tables and columns.

        This applies the underlying single table metric to all the tables
        found in the dataset and then returns the breakdown of the obtained scores.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset.
            metadata (dict):
                Multi-table metadata dict. If not passed, it is built based on the
                real_data fields and dtypes.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single table metric

        Returns:
            dict[string -> dict[string -> Union[float, tuple[float]]]]:
                A mapping of table name to column metric breakdowns.
        """
        return cls._compute(cls, real_data, synthetic_data, metadata, **kwargs)

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        assert cls.min_value == 0.0
        return super().normalize(raw_score)


class CSTest(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable CSTest."""

    single_table_metric = single_table.multi_single_column.CSTest


class KSComplement(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable KSComplement."""

    single_table_metric = single_table.multi_single_column.KSComplement


class StatisticSimilarity(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable StatisticSimilarity."""

    single_table_metric = single_table.multi_single_column.StatisticSimilarity


class BoundaryAdherence(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable BoundaryAdherence."""

    single_table_metric = single_table.multi_single_column.BoundaryAdherence


class MissingValueSimilarity(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable MissingValueSimilarity."""

    single_table_metric = single_table.multi_single_column.MissingValueSimilarity


class CategoryCoverage(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable CategoryCoverage."""

    single_table_metric = single_table.multi_single_column.CategoryCoverage


class TVComplement(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable TVComplement."""

    single_table_metric = single_table.multi_single_column.TVComplement


class RangeCoverage(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable RangeCoverage."""

    single_table_metric = single_table.multi_single_column.RangeCoverage


class CorrelationSimilarity(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable CorrelationSimilarity."""

    single_table_metric = single_table.multi_column_pairs.CorrelationSimilarity


class ContingencySimilarity(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable ContingencySimilarity."""

    single_table_metric = single_table.multi_column_pairs.ContingencySimilarity


class LogisticDetection(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable LogisticDetection."""

    single_table_metric = single_table.detection.LogisticDetection


class SVCDetection(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable SVCDetection."""

    single_table_metric = single_table.detection.SVCDetection


class BNLikelihood(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable BNLikelihood."""

    single_table_metric = single_table.bayesian_network.BNLikelihood


class NewRowSynthesis(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable NewRowSynthesis."""

    single_table_metric = single_table.new_row_synthesis.NewRowSynthesis


class BNLogLikelihood(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable BNLogLikelihood."""

    single_table_metric = single_table.bayesian_network.BNLogLikelihood

    @classmethod
    def normalize(cls, raw_score):
        """Normalize the log-likelihood value.

        Note that this is not the mean likelihood but rather the exponentiation
        of the mean log-likelihood.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)
