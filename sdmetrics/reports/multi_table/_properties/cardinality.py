from sdmetrics.multi_table.statistical import CardinalityShapeSimilarity
from sdmetrics.reports.multi_table._properties.base import BaseMultiTableProperty
from sdmetrics.reports.multi_table.plot_utils import get_table_relationships_plot


class Cardinality(BaseMultiTableProperty):
    """``Cardinality`` class.

    Property that uses ``sdmetrics.multi_table.statistical.CardinalityShapeSimilarity`` metric
    in order to compute and plot the scores of cardinality shape similarity in the given tables.
    """

    def __init__(self):
        self._details = {}

    def _get_num_iterations(self, metadata):
        return len(metadata['relationships'])

    def get_score(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Get the average score of cardinality shape similarity in the given tables.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            progress_bar (tqdm.tqdm or None):
                The progress bar object. Defaults to ``None``.

        Returns:
            float:
                The average score for the property for all the individual metric scores computed.
        """
        for relation in metadata.get('relationships', []):
            relationships_metadata = {'relationships': [relation]}
            try:
                self._details = CardinalityShapeSimilarity.compute_breakdown(
                    real_data,
                    synthetic_data,
                    relationships_metadata
                )
            except Exception as error:
                errors = self._details.get('Errors', {})
                errors[relation] = str(error)
                self._details['Errors'] = errors

            if progress_bar is not None:
                progress_bar.update()

        if progress_bar is not None:
            progress_bar.close()

        score = 0
        for result in self._details.values():
            score += result.get('score', 0)

        average_score = score / len(self._details) if len(self._details) else score
        return average_score

    def get_visualization(self, table_name):
        """Return a visualization for each score in the property.

        Args:
            table_name (str):
                Table name to get the visualization for.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the property.
        """
        score_breakdowns = {'CardinalityShapeSimilarity': self._details}
        for metric, details in score_breakdowns.items():
            score_breakdowns[metric] = {
                tables: results for tables, results in details.items()
                if table_name in tables
            }

        fig = get_table_relationships_plot(score_breakdowns)
        return fig
