import numpy as np
import pandas as pd

from sdmetrics.multi_table.statistical import CardinalityShapeSimilarity
from sdmetrics.reports.multi_table._properties.base import BaseMultiTableProperty
from sdmetrics.reports.multi_table.plot_utils import get_table_relationships_plot


class Cardinality(BaseMultiTableProperty):
    """``Cardinality`` class.

    Property that uses ``sdmetrics.multi_table.statistical.CardinalityShapeSimilarity`` metric
    in order to compute and plot the scores of cardinality shape similarity in the given tables.
    """

    def __init__(self):
        super().__init__()
        self._only_multi_table = True

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
        child_table, parent_table = [], []
        metric_names, scores, error_messages = [], [], []
        for relation in metadata.get('relationships', []):
            relationships_metadata = {'relationships': [relation]}
            try:
                relation_score = CardinalityShapeSimilarity.compute(
                    real_data,
                    synthetic_data,
                    relationships_metadata
                )
                error_message = None
            except Exception as e:
                relation_score = np.nan
                error_message = f'Error: {type(e).__name__} {e}'
            finally:
                if progress_bar is not None:
                    progress_bar.update()

            child_table.append(relation['child_table_name'])
            parent_table.append(relation['parent_table_name'])
            metric_names.append('CardinalityShapeSimilarity')
            scores.append(relation_score)
            error_messages.append(error_message)

        self.details_property = pd.DataFrame({
            'Child Table': child_table,
            'Parent Table': parent_table,
            'Metric': metric_names,
            'Score': scores,
            'Error': error_messages,
        })

        if self.details_property['Error'].isna().all():
            self.details_property = self.details_property.drop('Error', axis=1)

        return self._compute_average()

    def _get_details_for_table_name(self, table_name):
        """Return the details for the given table name.

        Args:
            table_name (str):
                Table name to get the details for.

        Returns:
            pandas.DataFrame:
                The details for the given table name.
        """
        is_child = self.details_property['Child Table'] == table_name
        is_parent = self.details_property['Parent Table'] == table_name
        return self.details_property[is_child | is_parent].copy()

    def get_details(self, table_name=None):
        """Return the details for the property.

        Args:
            table_name (str):
                Table name to get the details for.
                Defaults to ``None``.

        Returns:
            pandas.DataFrame:
                The details for the property.
        """
        if table_name is None:
            return self.details_property.copy()

        return self._get_details_for_table_name(table_name)

    def get_visualization(self, table_name):
        """Return a visualization for each score in the property.

        Args:
            table_name (str):
                Table name to get the visualization for.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the property.
        """
        fig = get_table_relationships_plot(self._get_details_for_table_name(table_name))
        return fig
