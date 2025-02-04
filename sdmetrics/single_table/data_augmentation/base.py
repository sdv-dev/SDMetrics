"""Base class for Efficacy metrics for single table datasets."""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.data_augmentation.utils import _validate_inputs

METRIC_NAME_TO_METHOD = {'recall': recall_score, 'precision': precision_score}


class BaseDataAugmentationMetric(SingleTableMetric):
    """Base class for Data Augmentation metrics for single table datasets."""

    name = None
    metric_name = None
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    def _fit_preprocess(self, data, metadata):
        """Fit the preprocessing steps before applying the data augmentation technique."""
        self._discrete_columns = []
        self._datetime_columns = []
        for column, column_meta in metadata['columns'].items():
            if (column_meta['sdtype'] in ['categorical', 'boolean']) and (
                column != self.prediction_column_name
            ):
                self._discrete_columns.append(column)
            elif column_meta['sdtype'] == 'datetime':
                self._datetime_columns.append(column)

        self._ordinal_encoder = OrdinalEncoder()
        self._ordinal_encoder.fit(data[self._discrete_columns])

    def _fit(
        self, data, metadata, prediction_column_name, minority_class_label, classifier, fixed_value
    ):
        """Fit the data augmentation technique."""
        self.prediction_column_name = prediction_column_name
        self.minority_class_label = minority_class_label
        self.fixed_value = fixed_value
        # To assess the preicision efficacy, we have to fix the recall and reciprocally
        self._metric_to_fix = 'recall' if self.metric_name == 'precision' else 'precision'
        self._metric_method = METRIC_NAME_TO_METHOD[self._metric_to_fix]
        self._classifier_name = classifier
        self._classifier = XGBClassifier(enable_categorical=True)

        self._fit_preprocess(data, metadata)

    def _transform_preprocess(self, tables):
        """Transform the tables before applying the data augmentation technique.

        All the preprocessing steps are applied to the tables before applying the
        data augmentation technique.

        Args:
            tables (dict[str, pandas.DataFrame]):
                The tables to transform.
        """
        tables_result = {}
        for table_name, table in tables.items():
            table = table.copy()
            table[self._discrete_columns] = self._ordinal_encoder.transform(
                table[self._discrete_columns]
            )
            table[self._datetime_columns] = table[self._datetime_columns].apply(pd.to_numeric)
            table[self.prediction_column_name] = (
                table[self.prediction_column_name] == self.minority_class_label
            ).astype(int)
            tables_result[table_name] = table

        return tables_result

    def _get_best_threshold(self, train_data, train_target):
        """Find the best threshold for the classifier model."""
        target_probabilities = self._classifier.predict_proba(train_data)[:, 1]
        precision, recall, thresholds = precision_recall_curve(train_target, target_probabilities)
        # To assess the preicision efficacy, we have to fix the recall and reciprocally
        metric = precision if self.metric_name == 'recall' else recall
        best_threshold = 0.0
        valid_idx = np.where(metric >= self.fixed_value)[0]
        if valid_idx.size:
            differences = metric[valid_idx] - self.fixed_value
            best_idx_local = np.argmin(differences)
            best_idx = valid_idx[best_idx_local]
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        else:
            best_threshold = 1.0

        return best_threshold

    def _train_model(self, train_data):
        """Train the classifier model."""
        train_target = train_data.pop(self.prediction_column_name)
        self._classifier.fit(train_data, train_target)
        self._best_threshold = self._get_best_threshold(train_data, train_target)
        probabilities = self._classifier.predict_proba(train_data)[:, 1]
        predictions = (probabilities >= self._best_threshold).astype(int)

        return self._metric_method(train_target, predictions)

    def _compute_validation_scores(self, real_validation_data):
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

    def _get_scores(self, training_table, validation_table):
        """Get the scores of the metric."""
        training_table = deepcopy(training_table)
        validation_table = deepcopy(validation_table)
        training_score = self._train_model(training_table)
        recall, precision, prediction_counts_validation = self._compute_validation_scores(
            validation_table
        )
        scores = {
            f'{self._metric_to_fix}_score_training': training_score,
            'recall_score_validation': recall,
            'precision_score_validation': precision,
            'prediction_counts_validation': prediction_counts_validation,
        }

        return scores

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
        fixed_recall_value,
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
            fixed_recall_value,
        )
        result = {}
        metric = cls()
        metric._fit(
            real_training_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_recall_value,
        )
        tables = {
            'real_training_data': real_training_data,
            'synthetic_data': synthetic_data,
            'real_validation_data': real_validation_data,
        }
        preprocessed_tables = metric._transform_preprocess(tables)
        result['real_data_baseline'] = metric._get_scores(
            preprocessed_tables['real_training_data'], preprocessed_tables['real_validation_data']
        )
        augmented_training_table = pd.concat([
            preprocessed_tables['real_training_data'],
            preprocessed_tables['synthetic_data'],
        ]).reset_index(drop=True)
        result['augmented_data'] = metric._get_scores(
            augmented_training_table, preprocessed_tables['real_validation_data']
        )
        result['parameters'] = {
            'prediction_column_name': metric.prediction_column_name,
            'minority_class_label': metric.minority_class_label,
            'classifier': metric._classifier_name,
            'fixed_recall_value': metric.fixed_value,
        }
        result['score'] = max(
            0,
            (
                result['augmented_data'][f'{metric.metric_name}_score_validation']
                - result['real_data_baseline'][f'{metric.metric_name}_score_validation']
            ),
        )

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
        classifier=None,
        fixed_recall_value=0.9,
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
                Supported options are ``XGBoost``. Defaults to ``XGBoost``.
            fixed_recall_value (float):
                A float in the range (0, 1.0) describing the value to fix for the recall when
                building the Binary Classification model. Defaults to ``0.9``.

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
            fixed_recall_value,
        )

        return breakdown['score']
