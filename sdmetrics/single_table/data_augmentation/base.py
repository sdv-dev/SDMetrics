"""Base class for Efficacy metrics for single table datasets."""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score
from xgboost import XGBClassifier

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.data_augmentation.utils import _validate_inputs
from sdmetrics.single_table.utils import _process_data_with_metadata_ml_efficacy_metrics

METRIC_NAME_TO_METHOD = {'recall': recall_score, 'precision': precision_score}


class ClassifierTrainer:
    """Class to train a classifier model."""

    def __init__(
        self,
        prediction_column_name,
        minority_class_label,
        classifier,
        fixed_value,
        metric_name,
    ):
        self.prediction_column_name = prediction_column_name
        self.minority_class_label = minority_class_label
        self.fixed_value = fixed_value
        self.metric_name = metric_name
        self._classifier_name = classifier
        self._classifier = XGBClassifier(enable_categorical=True)
        self._metric_to_fix = 'recall' if metric_name == 'precision' else 'precision'
        self._metric_method = METRIC_NAME_TO_METHOD[self._metric_to_fix]

    def train_model(self, train_data):
        """Train the classifier model."""
        train_target = train_data.pop(self.prediction_column_name)
        self._classifier.fit(train_data, train_target)
        self._best_threshold = self.get_best_threshold(train_data, train_target)
        probabilities = self._classifier.predict_proba(train_data)[:, 1]
        predictions = (probabilities >= self._best_threshold).astype(int)

        return self._metric_method(train_target, predictions)

    def get_best_threshold(self, train_data, train_target):
        """Find the best threshold for the classifier model."""
        target_probabilities = self._classifier.predict_proba(train_data)[:, 1]
        precision, recall, thresholds = precision_recall_curve(train_target, target_probabilities)
        metric_map = {'precision': precision, 'recall': recall}
        metric = metric_map[self._metric_to_fix]
        valid_idx = np.where(metric >= self.fixed_value)[0]
        if valid_idx.size:
            best_idx = valid_idx[np.argmin(metric[valid_idx] - self.fixed_value)]
            return thresholds[best_idx] if best_idx < len(thresholds) else 1.0

        return 1.0

    def compute_validation_scores(self, real_validation_data):
        """Compute the validation scores."""
        real_validation_target = real_validation_data.pop(self.prediction_column_name)
        predictions = self._classifier.predict_proba(real_validation_data)[:, 1]
        predictions = (predictions >= self._best_threshold).astype(int)
        recall = recall_score(real_validation_target, predictions)
        precision = precision_score(real_validation_target, predictions)
        conf_matrix = confusion_matrix(real_validation_target, predictions)
        prediction_counts_validation = {
            'true_positive': int(conf_matrix[1, 1]),
            'false_positive': int(conf_matrix[0, 1]),
            'true_negative': int(conf_matrix[0, 0]),
            'false_negative': int(conf_matrix[1, 0]),
        }

        return recall, precision, prediction_counts_validation

    def get_scores(self, training_table, validation_table):
        """Get the scores of the metric."""
        training_table = deepcopy(training_table)
        validation_table = deepcopy(validation_table)
        training_score = self.train_model(training_table)
        recall, precision, prediction_counts_validation = self.compute_validation_scores(
            validation_table
        )
        return {
            f'{self._metric_to_fix}_score_training': training_score,
            'recall_score_validation': recall,
            'precision_score_validation': precision,
            'prediction_counts_validation': prediction_counts_validation,
        }


class BaseDataAugmentationMetric(SingleTableMetric):
    """Base class for Data Augmentation metrics for single table datasets."""

    name = None
    metric_name = None
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def _fit(cls, data, metadata, prediction_column_name):
        """Fit preprocessing parameters."""
        discrete_columns = []
        datetime_columns = []
        data_columns = data.columns
        metadata_columns = metadata['columns'].keys()
        common_columns = set(data_columns).intersection(metadata_columns)
        for column in sorted(common_columns):
            column_meta = metadata['columns'][column]
            if (column_meta['sdtype'] in ['categorical', 'boolean']) and (
                column != prediction_column_name
            ):
                discrete_columns.append(column)
            elif column_meta['sdtype'] == 'datetime':
                datetime_columns.append(column)

        return discrete_columns, datetime_columns

    @classmethod
    def _transform(
        cls,
        tables,
        discrete_columns,
        datetime_columns,
        prediction_column_name,
        minority_class_label,
    ):
        """Transform by preprocessing the tables.

        Args:
            tables (dict[str, pandas.DataFrame]):
                Dict containing `real_training_data`, `synthetic_data` and `real_validation_data`.
        """
        tables_result = {}
        for table_name, table in tables.items():
            table = table.copy()
            table[discrete_columns] = table[discrete_columns].astype('category')
            table[datetime_columns] = table[datetime_columns].apply(pd.to_numeric)
            table[prediction_column_name] = (
                table[prediction_column_name] == minority_class_label
            ).astype(int)
            tables_result[table_name] = table

        return tables_result

    @classmethod
    def _fit_transform(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        minority_class_label,
    ):
        """Fit and transform the metric."""
        discrete_columns, datetime_columns = cls._fit(
            real_training_data, metadata, prediction_column_name
        )
        tables = {
            'real_training_data': real_training_data,
            'synthetic_data': synthetic_data,
            'real_validation_data': real_validation_data,
        }

        return cls._transform(
            tables,
            discrete_columns,
            datetime_columns,
            prediction_column_name,
            minority_class_label,
        )

    @classmethod
    def compute_breakdown(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        minority_class_label,
        classifier,
        fixed_value,
    ):
        """Compute the score breakdown of the metric."""
        _validate_inputs(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_value,
        )
        (real_training_data, synthetic_data, real_validation_data) = (
            _process_data_with_metadata_ml_efficacy_metrics(
                real_training_data, synthetic_data, real_validation_data, metadata
            )
        )
        preprocessed_tables = cls._fit_transform(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            minority_class_label,
        )
        trainer = ClassifierTrainer(
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_value,
            cls.metric_name,
        )
        metric_to_fix = 'recall' if cls.metric_name == 'precision' else 'precision'
        result = {
            'real_data_baseline': trainer.get_scores(
                preprocessed_tables['real_training_data'],
                preprocessed_tables['real_validation_data'],
            ),
            'augmented_data': trainer.get_scores(
                pd.concat([
                    preprocessed_tables['real_training_data'],
                    preprocessed_tables['synthetic_data'],
                ]).reset_index(drop=True),
                preprocessed_tables['real_validation_data'],
            ),
            'parameters': {
                'prediction_column_name': trainer.prediction_column_name,
                'minority_class_label': trainer.minority_class_label,
                'classifier': trainer._classifier_name,
                f'fixed_{metric_to_fix}_value': trainer.fixed_value,
            },
        }
        augmented_score = result['augmented_data'][f'{cls.metric_name}_score_validation']
        baseline_score = result['real_data_baseline'][f'{cls.metric_name}_score_validation']
        result['score'] = (augmented_score - baseline_score) / 2 + 0.5
        return result

    @classmethod
    def compute(
        cls,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        minority_class_label,
        classifier,
        fixed_value,
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
            prediction_column_name (str):
                The name of the column to be predicted.
            minority_class_label (int):
                The minority class label.
            classifier (str):
                The ML algorithm to use when building a Binary Classfication.
                Supported options are ``XGBoost``.
            fixed_value (float):
                A float value in the range (0, 1.0) that specifies the metric value
                to fix when building the Binary Classification model.

        Returns:
            float:
                The score of the metric.
        """
        breakdown = cls.compute_breakdown(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_value,
        )

        return breakdown['score']
