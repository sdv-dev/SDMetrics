"""Table Format metric."""
from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric


class TableFormat(SingleTableMetric):
    """TableFormat Single Table metric.

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

    name = 'TableFormat'
    goal = Goal.MAXIMIZE
    min_value = 0
    max_value = 1

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, ignore_dtype_columns=None):
        """Compute the score breakdown of the table format metric.

        Args:
        real_data (pandas.DataFrame):
            The real data.
        synthetic_data (pandas.DataFrame):
            The synthetic data.
        ignore_dtype_columns (list[str]):
            List of column names to ignore when comparing data types.
            Defaults to ``None``.
        """
        ignore_dtype_columns = ignore_dtype_columns or []
        missing_columns_in_synthetic = set(real_data.columns) - set(synthetic_data.columns)
        invalid_names = []
        invalid_sdtypes = []
        for column in synthetic_data.columns:
            if column not in real_data.columns:
                invalid_names.append(column)
                continue

            if column in ignore_dtype_columns:
                continue

            if synthetic_data[column].dtype != real_data[column].dtype:
                invalid_sdtypes.append(column)

        proportion_correct_columns = 1 - len(missing_columns_in_synthetic) / len(real_data.columns)
        proportion_valid_names = 1 - len(invalid_names) / len(synthetic_data.columns)
        proportion_valid_sdtypes = 1 - len(invalid_sdtypes) / len(synthetic_data.columns)

        score = proportion_correct_columns * proportion_valid_names * proportion_valid_sdtypes
        return {'score': score}

    @classmethod
    def compute(cls, real_data, synthetic_data, ignore_dtype_columns=None):
        """Compute the table format metric score.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            ignore_dtype_columns (list[str]):
                List of column names to ignore when comparing data types.
                Defaults to ``None``.

        Returns:
            float:
                The metric score.
        """
        return cls.compute_breakdown(real_data, synthetic_data, ignore_dtype_columns)['score']
