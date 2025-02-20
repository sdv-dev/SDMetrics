"""DCR Baseline Protection metrics."""

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.privacy.dcr_utils import calculate_dcr
from sdmetrics.utils import get_columns_from_metadata, get_type_from_column_meta


class DCRBaselineProtection(SingleTableMetric):
    """DCR Baseline Protection metric."""

    name = 'DCRBaselineProtection'
    goal = Goal.IGNORE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def _validate_inputs(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        num_rows_subsample,
        num_iterations,
    ):
        if num_rows_subsample is not None:
            if not isinstance(num_rows_subsample, int):
                raise ValueError('num_rows_subsample must be an integer.')
            if num_rows_subsample < 1:
                raise ValueError(
                    f'num_rows_subsample ({num_rows_subsample}) must be greater than 1.'
                )
        elif num_rows_subsample is None and num_iterations > 1:
            raise ValueError(
                'num_iterations should not be greater than 1 if there is not subsampling.'
            )

        if not isinstance(num_iterations, int):
            raise ValueError('num_iterations must be an integer.')

        if num_iterations < 1:
            raise ValueError(f'num_iterations ({num_iterations}) must be greater than 1.')

        if metadata is not None:
            if not isinstance(metadata, dict):
                metadata = metadata.to_dict()

        valid_sdtypes = {'numerical', 'categorical', 'boolean', 'datetime'}
        drop_columns = []
        for column, column_meta in get_columns_from_metadata(metadata).items():
            sdtype = get_type_from_column_meta(column_meta)
            if sdtype not in valid_sdtypes:
                drop_columns.append(column)

        super()._validate_inputs(real_training_data, synthetic_data, metadata)
        super()._validate_inputs(real_validation_data, synthetic_data, metadata)

        sanitized_real_training_data = real_training_data.drop(columns=drop_columns)
        sanitized_synthetic_data = synthetic_data.drop(columns=drop_columns)
        sanitized_real_validation_data = real_validation_data.drop(columns=drop_columns)

        if (
            sanitized_real_training_data.empty
            or sanitized_synthetic_data.empty
            or sanitized_real_validation_data.empty
        ):
            raise ValueError(
                'There are no valid sdtypes in the dataframes to run the '
                'DCRBaselineProtection metric.'
            )

        return (
            sanitized_real_training_data,
            sanitized_synthetic_data,
            sanitized_real_validation_data,
        )

    @classmethod
    def compute_breakdown(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        num_rows_subsample=None,
        num_iterations=1,
    ):
        """Compute the DCRBaselineProtection metric.

        Args:
            real_training_data (pd.DataFrame):
                A pd.DataFrame object containing the real data used for training the synthesizer.
            synthetic_data (pd.DataFrame):
                A pandas.DataFrame object containing the synthetic data sampled
                from the synthesizer.
            real_validation_data (pd.DataFrame):
                A pandas.DataFrame object containing a holdout set of real data.
                This data should not have been used to train the synthesizer.
            metadata (dict):
                A metadata dictionary that describes the table of data.
            num_rows_subsample (int or None):
                The number of synthetic data rows to subsample from the synthetic data.
                This is used to increase the speed of the computation, if the dataset is large.
                Defaults to None which means no subsampling will be done.
            num_iterations (int):
                The number of iterations to perform when subsampling.
                The final score will be the average of all iterations. Default is 1 iteration.

        Returns:
            dict:
                Returns a dictionary that contains the overall score,
                the median DCR score between the synthetic data and real data,
                and the median DCR score between the random data and real data.
                Averages of the medians are returned in the case of multiple iterations.
        """
        sanitized_data = cls._validate_inputs(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            num_rows_subsample,
            num_iterations,
        )
        training_data, sanitized_synthetic_data, validation_data = sanitized_data

        synthetic_medians = []
        random_medians = []
        scores = []

        for _ in range(num_iterations):
            synthetic_sample = sanitized_synthetic_data
            if num_rows_subsample is not None:
                synthetic_sample = sanitized_synthetic_data.sample(n=num_rows_subsample)

            dcr_real = calculate_dcr(synthetic_sample, training_data, metadata)
            dcr_random = calculate_dcr(synthetic_sample, validation_data, metadata)
            synthetic_data_median = dcr_real.median()
            random_data_median = dcr_random.median()
            score = min((synthetic_data_median / random_data_median), 1.0)
            synthetic_medians.append(synthetic_data_median)
            random_medians.append(random_data_median)
            scores.append(score)

        result = {
            'score': sum(scores) / len(scores),
            'median_DCR_to_real_data': {
                'synthetic_data': sum(synthetic_medians) / len(synthetic_medians),
                'random_data_baseline': sum(random_medians) / len(random_medians),
            },
        }

        return result

    @classmethod
    def compute(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        num_rows_subsample=None,
        num_iterations=1,
    ):
        """Compute the DCRBaselineProtection metric.

        Args:
            real_training_data (pd.DataFrame):
                A pd.DataFrame object containing the real data used for training the synthesizer.
            synthetic_data (pd.DataFrame):
                A pandas.DataFrame object containing the synthetic data sampled
                from the synthesizer.
            real_validation_data (pd.DataFrame):
                A pandas.DataFrame object containing a holdout set of real data.
                This data should not have been used to train the synthesizer.
            metadata (dict):
                A metadata dictionary that describes the table of data.
            num_rows_subsample (int or None):
                The number of synthetic data rows to subsample from the synthetic data.
                This is used to increase the speed of the computation, if the dataset is large.
                Defaults to None which means no subsampling will be done.
            num_iterations (int):
                The number of iterations to perform when subsampling.
                The final score will be the average of all iterations. Default is 1 iteration.

        Returns:
            float:
                The score for the DCRBaselineProtection metric.
        """
        result = cls.compute_breakdown(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            num_rows_subsample,
            num_iterations,
        )

        return result.get('score')
