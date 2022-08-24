"""Single table quality report."""

import itertools
import sys

import numpy as np
import pandas as pd
import tqdm

from sdmetrics.single_table import (
    ContingencySimilarity, CorrelationSimilarity, KSComplement, TVComplement)


class QualityReport():
    """Single table quality report."""

    METRICS = {
        'Column Shapes': [KSComplement, TVComplement],
        'Column Pair Trends': [CorrelationSimilarity, ContingencySimilarity],
    }

    _metric_results = {}

    def generate(self, real_data, synthetic_data, metadata):
        """Generate report.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
        """
        metrics = list(itertools.chain.from_iterable(self.METRICS.values()))

        for metric in tqdm.tqdm(metrics, desc='Creating report:'):
            self._metric_results[metric.__name__] = metric.compute_breakdown(
                real_data, synthetic_data)

        self._property_breakdown = {}
        for prop, metrics in self.METRICS.items():
            prop_scores = []
            for metric in metrics:
                score = np.mean(
                    [
                        breakdown['score'] for _, breakdown
                        in self._metric_results[metric.__name__].items()
                    ]
                )
                prop_scores.append(score)

            self._property_breakdown[prop] = np.mean(prop_scores)

        self._overall_quality_score = np.mean(list(self._property_breakdown.values()))

        self._print_results()

    def _print_results(self, out=sys.stdout):
        """Print the quality report results."""
        out.write(f'Overall Quality Score: {self._overall_quality_score}\n')

        if len(self._property_breakdown) > 0:
            out.write('Properties:')

        for prop, score in self._property_breakdown.items():
            out.write(f'{prop}: {score * 100}%')

    def get_score(self):
        """Return the overall quality score.

        Returns:
            float
                The overall quality score.
        """
        return self._overall_quality_score

    def get_properties(self):
        """Return the property score breakdown.

        Returns:
            pandas.DataFrame
                The property score breakdown.
        """
        return pd.DataFrame({
            'Property': self._property_breakdown.keys(),
            'Score': self._property_breakdown.values(),
        })

    def get_details(self, property_name):
        """Return the details for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.

        Returns:
            pandas.DataFrame
                The score breakdown.
        """
