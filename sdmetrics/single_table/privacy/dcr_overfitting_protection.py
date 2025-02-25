"""DCR Overfitting Protection metrics."""

import warnings

import numpy as np

from sdmetrics._utils_metadata import _process_data_with_metadata
from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.privacy.dcr_utils import calculate_dcr


class DCROverfittingProtection(SingleTableMetric):
    """DCR Overfitting Protection metric.

    This metric uses a DCR (distance to closest record) computation to measure whether the
    synthetic data has been overfit to the real data, as compared to a holdout set.
    """

    name = 'DCROverfittingProtection'
    goal = Goal.MAXIMIZE
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
            if not isinstance(num_rows_subsample, int) or num_rows_subsample < 1:
                raise ValueError(
                    f'num_rows_subsample ({num_rows_subsample}) must be an integer greater than 1.'
                )
        elif num_rows_subsample is None and num_iterations > 1:
            raise ValueError(
                'num_iterations should not be greater than 1 if there is no subsampling.'
            )

        if not isinstance(num_iterations, int) or num_iterations < 1:
            raise ValueError(
                f'num_iterations ({num_iterations}) must be an integer greater than 1.'
            )

        if len(real_training_data) * 0.5 > len(real_validation_data):
            warnings.warn(
                f'Your real_validation_data contains {len(real_validation_data)} rows while your '
                f'real_training_data contains {len(real_training_data)} rows. For most accurate '
                'results, we recommend that the validation data at least half the size of the training data.'
            )

        real_data_copy = real_training_data.copy()
        synthetic_data_copy = synthetic_data.copy()
        real_validation_copy = real_validation_data.copy()
        real_data_copy = _process_data_with_metadata(real_data_copy, metadata, True)
        synthetic_data_copy = _process_data_with_metadata(synthetic_data_copy, metadata, True)
        real_validation_copy = _process_data_with_metadata(real_validation_copy, metadata, True)

        return (real_data_copy, synthetic_data_copy, real_validation_copy)

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
        """Compute the DCROverfittingProtection metric.

        Args:
            real_training_data (pd.DataFrame):
                A pd.DataFrame object containing the real data used for training the synthesizer.
            synthetic_data (pd.DataFrame):
                A pandas.DataFrame object containing the synthetic data sampled
                from the synthesizer.
            real_validation_data (pd.DataFrame):
                A pandas.DataFrame object containing a validation set of real data.
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
                Returns a dictionary that contains the overall score, the % of synthetic data rows
                that were closer to the validation set, and the % of synthetic data rows that were
                closer to the real dataset. Averages of the medians are returned in the case of
                multiple iterations.
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

        sum_of_scores = 0
        sum_of_p_close_to_real = 0
        sum_of_p_close_to_random = 0
        for _ in range(num_iterations):
            synthetic_sample = sanitized_synthetic_data
            if num_rows_subsample is not None:
                synthetic_sample = sanitized_synthetic_data.sample(n=num_rows_subsample)

            dcr_real = calculate_dcr(training_data, synthetic_sample, metadata)
            dcr_holdout = calculate_dcr(validation_data, synthetic_sample, metadata)

            num_rows_closer_to_real = np.where(dcr_real < dcr_holdout, 1.0, 0.0).sum()
            total_rows = dcr_real.size
            percentage_close_to_real = num_rows_closer_to_real / total_rows
            percentage_close_to_random = 1 - percentage_close_to_real
            score = min((1.0 - percentage_close_to_real) * 2, 1.0)
            sum_of_scores += score
            sum_of_p_close_to_real += percentage_close_to_real
            sum_of_p_close_to_random += percentage_close_to_random

        result = {
            'score': sum_of_scores / num_iterations,
            'synthetic_data_percentages': {
                'closer_to_training': sum_of_p_close_to_real / num_iterations,
                'closer_to_holdout': sum_of_p_close_to_random / num_iterations,
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
        """Compute the DCROverfittingProtection metric.

        Args:
            real_training_data (pd.DataFrame):
                A pd.DataFrame object containing the real data used for training the synthesizer.
            synthetic_data (pd.DataFrame):
                A pandas.DataFrame object containing the synthetic data sampled
                from the synthesizer.
            real_validation_data (pd.DataFrame):
                A pandas.DataFrame object containing a validation set of real data.
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
                The score for the DCROverfittingProtection metric.
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
