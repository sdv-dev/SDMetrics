"""Single table quality report."""
import sys

from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes
from sdmetrics.reports.single_table.base_report import BaseReport


class QualityReport(BaseReport):
    """Single table quality report.

    This class creates a quality report for single-table data. It calculates the quality
    score along two properties - Column Shapes and Column Pair Trends.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends()
        }

    def _get_num_iterations(self, property_name, metadata):
        """Get the number of iterations for the property.

        Args:
            property_name (str):
                The name of the property.
            metadata (dict):
                The metadata of the table.
        """
        self._check_property_name(property_name)

        num_columns = len(metadata['columns'])

        if property_name == 'Column Shapes':
            return num_columns
        elif property_name == 'Column Pair Trends':
            # if n is the number of columns in the dataset, then the number of
            # combinations between 2 different columns is n * (n - 1) / 2
            return int(num_columns * (num_columns - 1) / 2)

    def _print_results(self):
        """Print the quality report results."""
        sys.stdout.write(
            f'\nOverall Quality Score: {round(self._overall_score * 100, 2)}%\n\n'
        )
        sys.stdout.write('Properties:\n')

        for property_name in self._properties:
            property_score = round(self._properties[property_name]._compute_average(), 4)
            sys.stdout.write(
                f'- {property_name}: {property_score * 100}%\n'
            )

    def get_score(self):
        """Return the overall quality score.

        Returns:
            float
                The overall quality score.
        """
        self._check_report_generated()
        return self._overall_score
