"""Test for the base BaseDataAugmentationMetric metrics."""

from unittest.mock import ANY, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_score, recall_score

from sdmetrics.single_table.data_augmentation.base import BaseDataAugmentationMetric


@pytest.fixture
def real_training_data():
    return pd.DataFrame({
        'target': [1, 0, 0],
        'numerical': [1, 2, 3],
        'categorical': ['a', 'b', 'b'],
        'boolean': [True, False, True],
        'datetime': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
    })


@pytest.fixture
def synthetic_data():
    return pd.DataFrame({
        'target': [0, 1, 0],
        'numerical': [2, 2, 3],
        'categorical': ['a', 'b', 'b'],
        'boolean': [True, False, False],
        'datetime': pd.to_datetime(['2021-01-25', '2021-01-02', '2021-01-03']),
    })


@pytest.fixture
def real_validation_data():
    return pd.DataFrame({
        'target': [1, 0, 0],
        'numerical': [3, 3, 3],
        'categorical': ['a', 'b', 'b'],
        'boolean': [True, False, True],
        'datetime': pd.to_datetime(['2021-01-01', '2021-01-12', '2021-01-03']),
    })


@pytest.fixture
def metadata():
    return {
        'columns': {
            'target': {'sdtype': 'categorical'},
            'numerical': {'sdtype': 'numerical'},
            'categorical': {'sdtype': 'categorical'},
            'boolean': {'sdtype': 'boolean'},
            'datetime': {'sdtype': 'datetime'},
        }
    }


class TestBaseDataAugmentationMetric:
    """Test the BaseDataAugmentationMetric class."""

    @patch('sdmetrics.single_table.data_augmentation.base.OrdinalEncoder')
    def test__fit_preprocess(self, mock_ordinal_encoder, real_training_data, metadata):
        """Test the ``_fit_preprocess`` method."""
        # Setup
        metric = BaseDataAugmentationMetric()
        metric.prediction_column_name = 'target'
        mock_ordinal_encoder.fit = Mock()

        # Run
        metric._fit_preprocess(real_training_data, metadata)

        # Assert
        assert metric._discrete_columns == ['categorical', 'boolean']
        assert metric._datetime_columns == ['datetime']
        assert isinstance(metric._ordinal_encoder, Mock)
        args, _ = metric._ordinal_encoder.fit.call_args
        assert args[0].equals(real_training_data[['categorical', 'boolean']])

    def test__fit(self, real_training_data, metadata):
        """Test the ``_fit`` method."""
        # Setup
        metric = BaseDataAugmentationMetric()
        metric.metric_name = 'precision'
        prediction_column_name = 'target'
        minority_class_label = 1
        classifier = 'XGBoost'
        fixed_recall_value = 0.7

        # Run
        metric._fit(
            real_training_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_recall_value,
        )

        # Assert
        assert metric.prediction_column_name == prediction_column_name
        assert metric.minority_class_label == minority_class_label
        assert metric.fixed_value == fixed_recall_value
        assert metric._metric_method == recall_score
        assert metric._classifier_name == classifier
        # assert metric._classifier == 'XGBClassifier()'

    @patch('sdmetrics.single_table.data_augmentation.base.precision_recall_curve')
    def test__get_best_threshold(self, mock_precision_recall_curve, real_training_data):
        """Test the ``_get_best_threshold`` method."""
        # Setup
        metric = BaseDataAugmentationMetric()
        metric._classifier = Mock()
        metric._classifier.predict_proba = Mock(
            return_value=np.array([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
        )
        mock_precision_recall_curve.return_value = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.72, 0.8, 0.9, 1.0]),
            np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]),
            np.array([0.02, 0.15, 0.25, 0.35, 0.42, 0.51, 0.63, 0.77, 0.82, 0.93, 0.97]),
        ]
        metric.metric_name = 'recall'
        metric.fixed_value = 0.69
        train_data = real_training_data[['numerical']]
        train_target = real_training_data['target']

        # Run
        best_threshold = metric._get_best_threshold(train_data, train_target)

        # Assert
        assert best_threshold == 0.63

    def test__train_model(self, real_training_data):
        """Test the ``_train_model`` method.

        Here the true target values are [1, 0, 0] and the predicted ones based on the
        best threshold are [1, 0, 1]. So the precision score should be 0.5.
        """
        # Setup
        metric = BaseDataAugmentationMetric()
        metric.prediction_column_name = 'target'
        metric._classifier = Mock()
        metric.metric_name = 'precision'
        metric.fixed_value = 0.69
        metric._get_best_threshold = Mock(return_value=0.63)
        metric._classifier.predict_proba = Mock(
            return_value=np.array([[0.3, 0.7], [0.4, 0.6], [0.3, 0.7]])
        )
        metric._metric_method = precision_score
        real_training_data_copy = real_training_data.copy()

        # Run
        score = metric._train_model(real_training_data_copy)

        # Assert
        assert score == 0.5
        assert metric._best_threshold == 0.63

    def test__compute_validation_scores(self, real_validation_data):
        """Test the ``_compute_validation_scores`` method."""
        # Setup
        metric = BaseDataAugmentationMetric()
        metric.prediction_column_name = 'target'
        metric._best_threshold = 0.63
        metric._classifier = Mock()
        metric._classifier.predict_proba = Mock(
            return_value=np.array([[0.3, 0.7], [0.4, 0.6], [0.3, 0.7]])
        )

        # Run
        recall, precision, prediction_counts_validation = metric._compute_validation_scores(
            real_validation_data
        )

        # Assert
        assert recall == 1.0
        assert precision == 0.5
        assert prediction_counts_validation == {
            'true_positive': 1,
            'false_positive': 1,
            'true_negative': 1,
            'false_negative': 0,
        }

    def test__get_scores(self, real_training_data, real_validation_data):
        """Test the ``_get_scores`` method."""
        # Setup
        metric = BaseDataAugmentationMetric()
        metric.metric_name = 'precision'
        metric._train_model = Mock(return_value=0.78)
        metric._metric_to_fix = 'recall'
        metric._compute_validation_scores = Mock(
            return_value=(
                1.0,
                0.5,
                {
                    'true_positive': 1,
                    'false_positive': 1,
                    'true_negative': 1,
                    'false_negative': 0,
                },
            )
        )

        # Run
        scores = metric._get_scores(real_training_data, real_validation_data)

        # Assert
        assert scores == {
            'recall_score_training': 0.78,
            'recall_score_validation': 1.0,
            'precision_score_validation': 0.5,
            'prediction_counts_validation': {
                'true_positive': 1,
                'false_positive': 1,
                'true_negative': 1,
                'false_negative': 0,
            },
        }

    @patch('sdmetrics.single_table.data_augmentation.base._validate_inputs')
    @patch(
        'sdmetrics.single_table.data_augmentation.base.BaseDataAugmentationMetric._fit',
        autospec=True,
    )
    @patch(
        'sdmetrics.single_table.data_augmentation.base.BaseDataAugmentationMetric'
        '._transform_preprocess',
        autospec=True,
    )
    @patch(
        'sdmetrics.single_table.data_augmentation.base.BaseDataAugmentationMetric._get_scores',
        autospec=True,
    )
    def test_compute_breakdown(
        self,
        mock_get_scores,
        mock_transform_preprocess,
        mock_fit,
        mock_validate_inputs,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
    ):
        """Test the ``compute_breakdown`` method."""
        # Setup
        prediction_column_name = 'target'
        minority_class_label = 1
        classifier = 'XGBoost'
        fixed_recall_value = 0.9

        def expected_assignments_fit(
            metric,
            data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_value,
        ):
            metric.metric_name = 'precision'
            metric.prediction_column_name = prediction_column_name
            metric.minority_class_label = minority_class_label
            metric._classifier_name = classifier
            metric._classifier = 'XGBClassifier()'
            metric.fixed_value = fixed_value

        mock_fit.side_effect = expected_assignments_fit
        mock_transform_preprocess.return_value = {
            'real_training_data': real_training_data,
            'synthetic_data': synthetic_data,
            'real_validation_data': real_validation_data,
        }
        real_data_baseline = {
            'precision_score_training': 0.43,
            'recall_score_validation': 0.7,
            'precision_score_validation': 0.5,
            'prediction_counts_validation': {
                'true_positive': 1,
                'false_positive': 1,
                'true_negative': 1,
                'false_negative': 0,
            },
        }
        augmented_table_result = {
            'precision_score_training': 0.78,
            'recall_score_validation': 0.9,
            'precision_score_validation': 0.7,
            'prediction_counts_validation': {
                'true_positive': 2,
                'false_positive': 2,
                'true_negative': 1,
                'false_negative': 0,
            },
        }
        mock_get_scores.side_effect = [real_data_baseline, augmented_table_result]

        # Run
        score_breakdown = BaseDataAugmentationMetric.compute_breakdown(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_recall_value,
        )

        # Assert
        expected_result = {
            'score': 0.19999999999999996,
            'real_data_baseline': real_data_baseline,
            'augmented_data': augmented_table_result,
            'parameters': {
                'prediction_column_name': prediction_column_name,
                'minority_class_label': minority_class_label,
                'classifier': classifier,
                'fixed_recall_value': fixed_recall_value,
            },
        }
        assert score_breakdown == expected_result
        mock_validate_inputs.assert_called_once_with(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_recall_value,
        )
        mock_fit.assert_called_once_with(
            ANY,
            real_training_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_recall_value,
        )
        mock_transform_preprocess.assert_called_once_with(
            ANY,
            {
                'real_training_data': real_training_data,
                'synthetic_data': synthetic_data,
                'real_validation_data': real_validation_data,
            },
        )

    @patch(
        'sdmetrics.single_table.data_augmentation.base.BaseDataAugmentationMetric.compute_breakdown'
    )
    def test_compute(
        self,
        mock_compute_breakdown,
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
    ):
        """Test the ``compute`` method."""
        # Setup
        prediction_column_name = 'target'
        minority_class_label = 1
        classifier = 'XGBoost'
        fixed_recall_value = 0.9
        mock_compute_breakdown.return_value = {
            'score': 0.9,
            'other_key': 'other_value',
        }

        # Run
        score = BaseDataAugmentationMetric.compute(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_recall_value,
        )

        # Assert
        assert score == 0.9
