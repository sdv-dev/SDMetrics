"""Binary classifier recall efficacy metric."""

from sdmetrics.single_table.data_augmentation.base import BaseDataAugmentationMetric


class BinaryClassifierRecallEfficacy(BaseDataAugmentationMetric):
    """Binary classifier recall efficacy metric."""

    name = 'Binary Classifier Recall Efficacy'
    metric_name = 'recall'

    @classmethod
    def compute_breakdown(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        table_name,
        prediction_column_name,
        minority_class_label,
        classifier='XGBoost',
        fixed_precision_value=0.9,
    ):
        """Compute the score breakdown of the metric."""
        return super().compute_breakdown(
            real_training_data=real_training_data,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation_data,
            metadata=metadata,
            table_name=table_name,
            prediction_column_name=prediction_column_name,
            minority_class_label=minority_class_label,
            classifier=classifier,
            fixed_value=fixed_precision_value,
        )

    @classmethod
    def compute(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        table_name,
        prediction_column_name,
        minority_class_label,
        classifier='XGBoost',
        fixed_precision_value=0.9,
    ):
        """Compute the score of the metric.

        Args:
            real_training_data (pandas.DataFrame):
                The real training data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            real_validation_data (pandas.DataFrame):
                The real validation data.
            metadata (dict):
                The metadata dictionary describing the table of data.
            table_name (str):
                The name of the table to use from the metadata and data dictionaries.
            prediction_column_name (str):
                The name of the column to be predicted.
            minority_class_label (int):
                The minority class label.
            classifier (str):
                The ML algorithm to use when building a Binary Classfication.
                Supported options are ``XGBoost``. Defaults to ``XGBoost``.
            fixed_precision_value (float):
                The fixed precision value to be used when calculating the recall score.
                Defaults to 0.9.

        Returns:
            float:
                The score of the metric.
        """
        return super().compute(
            real_training_data=real_training_data,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation_data,
            metadata=metadata,
            table_name=table_name,
            prediction_column_name=prediction_column_name,
            minority_class_label=minority_class_label,
            classifier=classifier,
            fixed_value=fixed_precision_value,
        )
