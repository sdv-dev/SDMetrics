import numpy as np
import pandas as pd

from sdmetrics.errors import VisualizationUnavailableError
from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.single_table import TableStructure


class Structure(BaseSingleTableProperty):
    """Structure property class for single table.

    This property checks to see whether the overall structure of the synthetic
    data is the same as the real data.
    """

    _num_iteration_case = 'table'

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Generate the _details dataframe for the structure property.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata of the table
            progress_bar (tqdm.tqdm or None):
                The progress bar to use. Defaults to None.

        Returns:
            pandas.DataFrame
        """
        try:
            score = TableStructure.compute(real_data, synthetic_data)
            error_message = None

        except Exception as e:
            score = np.nan
            error_message = f'{type(e).__name__}: {e}'

        finally:
            if progress_bar:
                progress_bar.update()

        result = pd.DataFrame(
            {
                'Metric': 'TableStructure',
                'Score': score,
                'Error': error_message,
            },
            index=[0],
        )

        if result['Error'].isna().all():
            result = result.drop('Error', axis=1)

        return result

    def get_visualization(self):
        """Return the visualization for the property.

        Raise an error in this case because the single table Structure property
        does not have a supported visualization.
        """
        raise VisualizationUnavailableError(
            'The single table Structure property does not have a supported visualization.'
        )
