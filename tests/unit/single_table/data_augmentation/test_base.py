"""Test for the base BaseDataAugmentationMetric metrics."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_score
from xgboost import XGBClassifier

from sdmetrics.single_table.data_augmentation.base import (
    BaseDataAugmentationMetric,
    ClassifierTrainer,
)


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
        'categorical': ['b', 'a', 'b'],
        'boolean': [False, True, False],
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


class TestClassifierTrainer:
    def test__init__(self):
        """Test the ``__init__`` method."""
        # Run
        trainer = ClassifierTrainer('target', 1, 'XGBoost', 0.69, 'recall')

        # Assert
        assert trainer.prediction_column_name == 'target'
        assert trainer.minority_class_label == 1
        assert trainer._classifier_name == 'XGBoost'
        assert trainer.fixed_value == 0.69
        assert trainer.metric_name == 'recall'
        assert trainer._metric_to_fix == 'precision'
        assert trainer._metric_method == precision_score
        assert isinstance(trainer._classifier, XGBClassifier)

    @patch('sdmetrics.single_table.data_augmentation.base.precision_recall_curve')
    def test_get_best_threshold(self, mock_precision_recall_curve, real_training_data):
        """Test the ``get_best_threshold`` method."""
        # Setup
        trainer = ClassifierTrainer('target', 1, 'XGBoost', 0.69, 'recall')
        trainer._classifier = Mock()
        trainer._classifier.predict_proba = Mock(
            return_value=np.array([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
        )
        mock_precision_recall_curve.return_value = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.72, 0.8, 0.9, 1.0]),
            np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]),
            np.array([0.02, 0.15, 0.25, 0.35, 0.42, 0.51, 0.63, 0.77, 0.82, 0.93, 0.97]),
        ]
        train_data = real_training_data[['numerical']]
        train_target = real_training_data['target']

        # Run
        best_threshold = trainer.get_best_threshold(train_data, train_target)

        # Assert
        assert best_threshold == 0.63

    def test_train_model(self, real_training_data):
        """Test the ``train_model`` method.

        Here the true target values are [1, 0, 0] and the predicted ones based on the
        best threshold are [1, 0, 1]. So the precision score should be 0.5.
        """
        # Setup
        trainer = ClassifierTrainer('target', 1, 'XGBoost', 0.69, 'recall')
        trainer.get_best_threshold = Mock(return_value=0.63)
        trainer._classifier = Mock()
        trainer._classifier.predict_proba = Mock(
            return_value=np.array([[0.3, 0.7], [0.4, 0.6], [0.3, 0.7]])
        )
        trainer._metric_method = precision_score
        real_training_data_copy = real_training_data.copy()

        # Run
        score = trainer.train_model(real_training_data_copy)

        # Assert
        assert score == 0.5
        assert trainer._best_threshold == 0.63

    def test_compute_validation_scores(self, real_validation_data):
        """Test the ``compute_validation_scores`` method."""
        # Setup
        trainer = ClassifierTrainer('target', 1, 'XGBoost', 0.69, 'recall')
        trainer._best_threshold = 0.63
        trainer._classifier = Mock()
        trainer._classifier.predict_proba = Mock(
            return_value=np.array([[0.3, 0.7], [0.4, 0.6], [0.3, 0.7]])
        )

        # Run
        recall, precision, prediction_counts_validation = trainer.compute_validation_scores(
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

    def test_get_scores(self, real_training_data, real_validation_data):
        """Test the ``get_scores`` method."""
        # Setup
        trainer = ClassifierTrainer('target', 1, 'XGBoost', 0.69, 'precision')
        trainer.train_model = Mock(return_value=0.78)
        trainer.compute_validation_scores = Mock(
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
        scores = trainer.get_scores(real_training_data, real_validation_data)

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


class TestBaseDataAugmentationMetric:
    """Test the BaseDataAugmentationMetric class."""

    def test__fit_preprocess(self, real_training_data, metadata):
        """Test the ``_fit_preprocess`` method."""
        # Setup
        metric = BaseDataAugmentationMetric()

        # Run
        discrete_columns, datetime_columns = metric._fit_preprocess(
            real_training_data, metadata, 'target'
        )

        # Assert
        assert discrete_columns == ['categorical', 'boolean']
        assert datetime_columns == ['datetime']

    def test__transform_preprocess(self, real_training_data, synthetic_data, real_validation_data):
        """Test the ``_transform_preprocess`` method."""
        # Setup
        metric = BaseDataAugmentationMetric()
        discrete_columns = ['categorical', 'boolean']
        datetime_columns = ['datetime']
        tables = {
            'real_training_data': real_training_data,
            'synthetic_data': synthetic_data,
            'real_validation_data': real_validation_data,
        }

        # Run
        transformed = metric._transform_preprocess(
            tables, discrete_columns, datetime_columns, 'target', 1
        )

        # Assert
        expected_transformed = {
            'real_training_data': pd.DataFrame({
                'target': [1, 0, 0],
                'numerical': [1, 2, 3],
                'categorical': pd.Categorical(['a', 'b', 'b']),
                'boolean': pd.Categorical([True, False, True]),
                'datetime': pd.to_numeric(
                    pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
                ),
            }),
            'synthetic_data': pd.DataFrame({
                'target': [0, 1, 0],
                'numerical': [2, 2, 3],
                'categorical': pd.Categorical(['b', 'a', 'b']),
                'boolean': pd.Categorical([False, True, False]),
                'datetime': pd.to_numeric(
                    pd.to_datetime(['2021-01-25', '2021-01-02', '2021-01-03'])
                ),
            }),
            'real_validation_data': pd.DataFrame({
                'target': [1, 0, 0],
                'numerical': [3, 3, 3],
                'categorical': pd.Categorical(['a', 'b', 'b']),
                'boolean': pd.Categorical([True, False, True]),
                'datetime': pd.to_numeric(
                    pd.to_datetime(['2021-01-01', '2021-01-12', '2021-01-03'])
                ),
            }),
        }
        for table_name, table in transformed.items():
            pd.testing.assert_frame_equal(table, expected_transformed[table_name])

    def test__fit_transform(
        self, real_training_data, synthetic_data, real_validation_data, metadata
    ):
        """Test the ``_fit_transform`` method."""
        # Setup
        metric = BaseDataAugmentationMetric()
        BaseDataAugmentationMetric._fit_preprocess = Mock()
        discrete_columns = ['categorical', 'boolean']
        datetime_columns = ['datetime']
        BaseDataAugmentationMetric._fit_preprocess.return_value = (
            discrete_columns,
            datetime_columns,
        )
        tables = {
            'real_training_data': real_training_data,
            'synthetic_data': synthetic_data,
            'real_validation_data': real_validation_data,
        }
        BaseDataAugmentationMetric._transform_preprocess = Mock(return_value=tables)

        # Run
        transformed = metric._fit_transform(
            real_training_data, synthetic_data, real_validation_data, metadata, 'target', 1
        )

        # Assert
        BaseDataAugmentationMetric._fit_preprocess.assert_called_once_with(
            real_training_data, metadata, 'target'
        )
        BaseDataAugmentationMetric._transform_preprocess.assert_called_once_with(
            tables, discrete_columns, datetime_columns, 'target', 1
        )
        for table_name, table in transformed.items():
            assert table.equals(tables[table_name])

    @patch('sdmetrics.single_table.data_augmentation.base._validate_inputs')
    @patch(
        'sdmetrics.single_table.data_augmentation.base.BaseDataAugmentationMetric._fit_transform'
    )
    @patch(
        'sdmetrics.single_table.data_augmentation.base.ClassifierTrainer',
    )
    @patch.object(BaseDataAugmentationMetric, 'metric_name', 'precision')
    def test_compute_breakdown(
        self,
        mock_classifier_trainer,
        mock_fit_transfrom,
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
        mock_fit_transfrom.return_value = {
            'real_training_data': real_training_data,
            'synthetic_data': synthetic_data,
            'real_validation_data': real_validation_data,
        }
        mock_classifier_trainer.return_value.get_scores.side_effect = [
            real_data_baseline,
            augmented_table_result,
        ]
        mock_classifier_trainer.return_value.prediction_column_name = prediction_column_name
        mock_classifier_trainer.return_value.minority_class_label = minority_class_label
        mock_classifier_trainer.return_value._classifier_name = classifier
        mock_classifier_trainer.return_value.fixed_value = fixed_recall_value

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
        mock_fit_transfrom.assert_called_once_with(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            minority_class_label,
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
