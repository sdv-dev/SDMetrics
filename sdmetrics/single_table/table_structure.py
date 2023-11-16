"""Table Format metric."""
from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric


class TableStructure(SingleTableMetric):
    """TableStructure Single Table metric.

    This metric computes whether the names and data types of each column are
    the same in the real and synthetic data.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = 'TableStructure'
    goal = Goal.MAXIMIZE
    min_value = 0
    max_value = 1

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the score breakdown of the table format metric.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
        """
        synthetic_columns = set(synthetic_data.columns)
        real_columns = set(real_data.columns)
        intersection_columns = real_columns & synthetic_columns
        union_columns = real_columns | synthetic_columns
        score = len(intersection_columns)/len(union_columns)

        return {'score': score}

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute the table format metric score.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.

        Returns:
            float:
                The metric score.
        """
        return cls.compute_breakdown(real_data, synthetic_data)['score']
