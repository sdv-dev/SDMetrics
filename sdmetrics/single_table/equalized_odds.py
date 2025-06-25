# flake8: noqa
"""EqualizedOddsImprovement metric for single table datasets."""

import pandas as pd
from sklearn.metrics import confusion_matrix

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.utils import (
    _validate_classifier,
    _validate_column_consistency,
    _validate_column_values_exist,
    _validate_data_and_metadata,
    _validate_prediction_column_name,
    _validate_required_columns,
    _validate_sensitive_column_name,
    _validate_tables,
    _process_data_with_metadata_ml_efficacy_metrics,
)


class EqualizedOddsImprovement(SingleTableMetric):
    """EqualizedOddsImprovement metric.

    This metric evaluates fairness by measuring equalized odds - whether the
    True Positive Rate (TPR) and False Positive Rate (FPR) are the same
    across different values of a sensitive attribute.

    The metric compares the equalized odds between real training data and
    synthetic data, both evaluated on a holdout validation set.
    """

    name = 'EqualizedOddsImprovement'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def _validate_data_sufficiency(
        cls,
        data,
        prediction_column_name,
        sensitive_column_name,
        positive_class_label,
        sensitive_column_value,
    ):
        """Validate that there is sufficient data for training."""
        # Create binary versions of the columns
        prediction_binary = data[prediction_column_name] == positive_class_label
        sensitive_binary = data[sensitive_column_name] == sensitive_column_value

        # Check both sensitive groups (target value and non-target value)
        for is_sensitive_group in [True, False]:
            group_predictions = prediction_binary[sensitive_binary == is_sensitive_group]
            group_name = 'sensitive' if is_sensitive_group else 'non-sensitive'

            if len(group_predictions) == 0:
                raise ValueError(f'No data found for {group_name} group.')

            positive_count = group_predictions.sum()
            negative_count = len(group_predictions) - positive_count

            if positive_count < 5 or negative_count < 5:
                raise ValueError(
                    f'Insufficient data for {group_name} group: {positive_count} positive, '
                    f'{negative_count} negative examples (need ≥5 each).'
                )

    @classmethod
    def _preprocess_data(
        cls,
        data,
        prediction_column_name,
        positive_class_label,
        sensitive_column_name,
        sensitive_column_value,
        metadata,
    ):
        """Preprocess the data for binary classification."""
        data = data.copy()

        # Convert prediction column to binary
        data[prediction_column_name] = (
            data[prediction_column_name] == positive_class_label
        ).astype(int)

        # Convert sensitive column to binary
        if pd.isna(sensitive_column_value):
            data[sensitive_column_name] = data[sensitive_column_name].isna().astype(int)
        else:
            data[sensitive_column_name] = (
                data[sensitive_column_name] == sensitive_column_value
            ).astype(int)

        # Handle categorical columns for XGBoost
        for column, column_meta in metadata['columns'].items():
            if (
                column in data.columns
                and column_meta.get('sdtype') in ['categorical', 'boolean']
                and column != prediction_column_name
                and column != sensitive_column_name
            ):
                data[column] = data[column].astype('category')
            elif column in data.columns and column_meta.get('sdtype') == 'datetime':
                data[column] = pd.to_numeric(data[column], errors='coerce')

        return data

    @classmethod
    def _train_classifier(cls, train_data, prediction_column_name):
        """Train the XGBoost classifier."""
        train_data = train_data.copy()
        train_target = train_data.pop(prediction_column_name)

        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                'XGBoost is required but not installed. Install with: pip install sdmetrics[xgboost]'
            )

        classifier = XGBClassifier(enable_categorical=True)
        classifier.fit(train_data, train_target)

        return classifier

    @classmethod
    def _compute_prediction_counts(cls, predictions, actuals, sensitive_values):
        """Compute prediction counts for each sensitive group."""
        results = {}

        for sensitive_val in [True, False]:
            mask = sensitive_values == sensitive_val
            if not mask.any():
                # No data for this group
                results[f'{sensitive_val}'] = {
                    'true_positive': 0,
                    'false_positive': 0,
                    'true_negative': 0,
                    'false_negative': 0,
                }
                continue

            group_predictions = predictions[mask]
            group_actuals = actuals[mask]

            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                group_actuals, group_predictions, labels=[0, 1]
            ).ravel()

            results[f'{sensitive_val}'] = {
                'true_positive': int(tp),
                'false_positive': int(fp),
                'true_negative': int(tn),
                'false_negative': int(fn),
            }

        return results

    @classmethod
    def _compute_equalized_odds_score(cls, prediction_counts):
        """Compute the equalized odds score from prediction counts."""
        # Extract counts for both groups
        true_group = prediction_counts['True']
        false_group = prediction_counts['False']

        # Compute TPR and FPR for each group using a loop
        tpr = {}
        fpr = {}
        for group_name, group in [('True', true_group), ('False', false_group)]:
            tpr[group_name] = group['true_positive'] / max(
                1, group['true_positive'] + group['false_negative']
            )
            fpr[group_name] = group['false_positive'] / max(
                1, group['false_positive'] + group['true_negative']
            )

        # Compute fairness scores
        tpr_fairness = 1 - abs(tpr['True'] - tpr['False'])
        fpr_fairness = 1 - abs(fpr['True'] - fpr['False'])

        # Final equalized odds score is minimum of the two fairness scores
        return min(tpr_fairness, fpr_fairness)

    @classmethod
    def _evaluate_dataset(
        cls,
        train_data,
        validation_data,
        prediction_column_name,
        sensitive_column_name,
        sensitive_column_value,
    ):
        """Evaluate equalized odds for a single dataset."""
        # Train classifier
        classifier = cls._train_classifier(train_data, prediction_column_name)

        # Make predictions on validation data
        validation_features = validation_data.drop(columns=[prediction_column_name])
        predictions = classifier.predict(validation_features)
        actuals = validation_data[prediction_column_name].values
        sensitive_values = validation_data[sensitive_column_name].values

        # Compute prediction counts
        prediction_counts = cls._compute_prediction_counts(predictions, actuals, sensitive_values)

        # Format the keys to include sensitive column value as in the spec
        formatted_counts = {}
        for key, counts in prediction_counts.items():
            if key == 'True':
                formatted_key = f'{sensitive_column_value}=True'
            else:
                formatted_key = f'{sensitive_column_value}=False'
            formatted_counts[formatted_key] = counts

        # Compute equalized odds score
        equalized_odds_score = cls._compute_equalized_odds_score(prediction_counts)

        return {
            'equalized_odds': equalized_odds_score,
            'prediction_counts_validation': formatted_counts,
        }

    @classmethod
    def _validate_parameters(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        positive_class_label,
        sensitive_column_name,
        sensitive_column_value,
        classifier,
    ):
        """Validate all parameters and inputs for EqualizedOddsImprovement metric.

        Args:
            real_training_data (pandas.DataFrame):
                The real training data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            real_validation_data (pandas.DataFrame):
                The validation data.
            metadata (dict):
                Metadata describing the table.
            prediction_column_name (str):
                Name of the column to predict.
            positive_class_label:
                The positive class label for binary classification.
            sensitive_column_name (str):
                Name of the sensitive attribute column.
            sensitive_column_value:
                The value to consider as positive in the sensitive column.
            classifier (str):
                Classifier to use.
        """
        # Validate using shared utility functions
        _validate_tables(real_training_data, synthetic_data, real_validation_data)
        _validate_prediction_column_name(prediction_column_name)
        _validate_sensitive_column_name(sensitive_column_name)
        _validate_classifier(classifier)

        # Validate that required columns exist in all datasets
        dataframes_dict = {
            'real_training_data': real_training_data,
            'synthetic_data': synthetic_data,
            'real_validation_data': real_validation_data,
        }
        required_columns = [prediction_column_name, sensitive_column_name]
        _validate_required_columns(dataframes_dict, required_columns)

        # Validate data and metadata consistency for prediction column
        _validate_data_and_metadata(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            positive_class_label,
        )

        # Validate sensitive column value exists in all datasets
        column_value_pairs = [(sensitive_column_name, sensitive_column_value)]
        _validate_column_values_exist(dataframes_dict, column_value_pairs)

        # Use base class validation for real_training_data and synthetic_data
        real_training_data, synthetic_data, metadata = cls._validate_inputs(
            real_training_data, synthetic_data, metadata
        )

        # Validate the validation data separately (not part of standard _validate_inputs)
        real_validation_data = real_validation_data.copy()

        # Ensure validation data has same columns as training data
        _validate_column_consistency(real_training_data, synthetic_data, real_validation_data)

    @classmethod
    def compute_breakdown(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        positive_class_label,
        sensitive_column_name,
        sensitive_column_value,
        classifier='XGBoost',
    ):
        """Compute the EqualizedOddsImprovement metric breakdown.

        Args:
            real_training_data (pandas.DataFrame):
                The real data used for training the synthesizer.
            synthetic_data (pandas.DataFrame):
                The synthetic data generated by the synthesizer.
            real_validation_data (pandas.DataFrame):
                The holdout real data for validation.
            metadata (dict):
                Metadata describing the table.
            prediction_column_name (str):
                Name of the column to predict.
            positive_class_label:
                The positive class label for binary classification.
            sensitive_column_name (str):
                Name of the sensitive attribute column.
            sensitive_column_value:
                The value to consider as positive in the sensitive column.
            classifier (str):
                Classifier to use ('XGBoost' only supported currently).

        Returns:
            dict: breakdown of the score
        """
        cls._validate_parameters(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            positive_class_label,
            sensitive_column_name,
            sensitive_column_value,
            classifier,
        )

        (real_training_data, synthetic_data, real_validation_data) = (
            _process_data_with_metadata_ml_efficacy_metrics(
                real_training_data, synthetic_data, real_validation_data, metadata
            )
        )

        processed_data = []
        for data in [real_training_data, synthetic_data, real_validation_data]:
            processed_data.append(
                cls._preprocess_data(
                    data,
                    prediction_column_name,
                    positive_class_label,
                    sensitive_column_name,
                    sensitive_column_value,
                    metadata,
                )
            )

        real_training_processed, synthetic_processed, real_validation_processed = processed_data
        results = []
        for data in [real_training_processed, synthetic_processed]:
            cls._validate_data_sufficiency(
                data,
                prediction_column_name,
                sensitive_column_name,
                1,
                1,  # Using 1 since we converted to binary
            )

            results.append(
                cls._evaluate_dataset(
                    data,
                    real_validation_processed,
                    prediction_column_name,
                    sensitive_column_name,
                    sensitive_column_value,
                )
            )

        # Compute final improvement score
        real_score = results[0]['equalized_odds']
        synthetic_score = results[1]['equalized_odds']
        improvement_score = (synthetic_score - real_score) / 2 + 0.5

        return {
            'score': improvement_score,
            'real_training_data': results[0],
            'synthetic_data': results[1],
        }

    @classmethod
    def compute(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        positive_class_label,
        sensitive_column_name,
        sensitive_column_value,
        classifier='XGBoost',
    ):
        """Compute the EqualizedOddsImprovement metric score.

        Args:
            real_training_data (pandas.DataFrame):
                The real data used for training the synthesizer.
            synthetic_data (pandas.DataFrame):
                The synthetic data generated by the synthesizer.
            real_validation_data (pandas.DataFrame):
                The holdout real data for validation.
            metadata (dict):
                Metadata describing the table.
            prediction_column_name (str):
                Name of the column to predict.
            positive_class_label:
                The positive class label for binary classification.
            sensitive_column_name (str):
                Name of the sensitive attribute column.
            sensitive_column_value:
                The value to consider as positive in the sensitive column.
            classifier (str):
                Classifier to use ('XGBoost' only supported currently).

        Returns:
            float: The improvement score (0.5 = no improvement, 1.0 = maximum improvement,
                  0.0 = maximum degradation).
        """
        breakdown = cls.compute_breakdown(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            positive_class_label,
            sensitive_column_name,
            sensitive_column_value,
            classifier,
        )

        return breakdown['score']
