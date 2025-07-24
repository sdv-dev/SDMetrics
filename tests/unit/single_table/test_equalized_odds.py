"""Unit tests for EqualizedOddsImprovement metric."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.equalized_odds import EqualizedOddsImprovement


class TestEqualizedOddsImprovement:
    """Unit tests for EqualizedOddsImprovement class."""

    def test_class_attributes(self):
        """Test that class attributes are set correctly."""
        # Assert
        assert EqualizedOddsImprovement.name == 'EqualizedOddsImprovement'
        assert EqualizedOddsImprovement.goal.name == 'MAXIMIZE'
        assert EqualizedOddsImprovement.min_value == 0.0
        assert EqualizedOddsImprovement.max_value == 1.0

    def test_validate_data_sufficiency_valid_data(self):
        """Test _validate_data_sufficiency with sufficient data."""
        # Setup
        data = pd.DataFrame({
            'prediction': ['A'] * 5 + ['B'] * 5 + ['A'] * 5 + ['B'] * 5,  # 5+5 for each group
            'sensitive': [1] * 10 + [0] * 10,  # 10 sensitive, 10 non-sensitive
        })

        # Run
        EqualizedOddsImprovement._validate_data_sufficiency(data, 'prediction', 'sensitive', 'A', 1)

        # Assert

    def test_validate_data_sufficiency_no_data_for_group(self):
        """Test _validate_data_sufficiency when no data exists for a group."""
        # Setup
        data = pd.DataFrame({
            'prediction': ['A'] * 5 + ['B'] * 5,
            'sensitive': [0] * 10,  # Only non-sensitive group, no sensitive
        })

        # Run & Assert
        with pytest.raises(ValueError, match='No data found for sensitive group'):
            EqualizedOddsImprovement._validate_data_sufficiency(
                data, 'prediction', 'sensitive', 'A', 1
            )

    def test_validate_data_sufficiency_insufficient_positive_examples(self):
        """Test _validate_data_sufficiency with insufficient positive examples."""
        # Setup
        data = pd.DataFrame({
            'prediction': ['A'] * 3 + ['B'] * 10,  # Only 3 positive examples
            'sensitive': [1] * 13,
        })

        # Run & Assert
        with pytest.raises(ValueError, match='Insufficient data for sensitive group: 3 positive'):
            EqualizedOddsImprovement._validate_data_sufficiency(
                data, 'prediction', 'sensitive', 'A', 1
            )

    def test_validate_data_sufficiency_insufficient_negative_examples(self):
        """Test _validate_data_sufficiency with insufficient negative examples."""
        # Setup
        data = pd.DataFrame({
            'prediction': ['A'] * 10 + ['B'] * 3,  # Only 3 negative examples
            'sensitive': [1] * 13,
        })

        # Run & Assert
        with pytest.raises(ValueError, match='Insufficient data for sensitive group.*3 negative'):
            EqualizedOddsImprovement._validate_data_sufficiency(
                data, 'prediction', 'sensitive', 'A', 1
            )

    def test_preprocess_data_binary_conversion(self):
        """Test _preprocess_data converts columns to binary correctly."""
        # Setup
        data = pd.DataFrame({
            'prediction': ['True', 'False', 'True'],
            'sensitive': ['A', 'B', 'A'],
            'feature': [1, 2, 3],
        })

        metadata = {
            'columns': {
                'prediction': {'sdtype': 'categorical'},
                'sensitive': {'sdtype': 'categorical'},
                'feature': {'sdtype': 'numerical'},
            }
        }

        # Run
        result = EqualizedOddsImprovement._preprocess_data(
            data, 'prediction', 'True', 'sensitive', 'A', metadata
        )

        # Assert
        expected_prediction = [1, 0, 1]
        expected_sensitive = [1, 0, 1]

        assert result['prediction'].tolist() == expected_prediction
        assert result['sensitive'].tolist() == expected_sensitive
        assert result['feature'].tolist() == [1, 2, 3]

    def test_preprocess_data_categorical_handling(self):
        """Test _preprocess_data handles categorical columns correctly."""
        # Setup
        data = pd.DataFrame({
            'prediction': [1, 0, 1],
            'sensitive': [1, 0, 1],
            'cat_feature': ['X', 'Y', 'Z'],
            'bool_feature': [True, False, True],
        })

        metadata = {
            'columns': {
                'prediction': {'sdtype': 'categorical'},
                'sensitive': {'sdtype': 'categorical'},
                'cat_feature': {'sdtype': 'categorical'},
                'bool_feature': {'sdtype': 'boolean'},
            }
        }

        # Run
        result = EqualizedOddsImprovement._preprocess_data(
            data, 'prediction', 1, 'sensitive', 1, metadata
        )

        # Assert
        assert result['cat_feature'].dtype.name == 'category'
        assert result['bool_feature'].dtype.name == 'category'

    def test_preprocess_data_datetime_handling(self):
        """Test _preprocess_data handles datetime columns correctly."""
        # Setup
        data = pd.DataFrame({
            'prediction': [1, 0, 1],
            'sensitive': [1, 0, 1],
            'datetime_feature': ['2023-01-01', '2023-01-02', '2023-01-03'],
        })

        metadata = {
            'columns': {
                'prediction': {'sdtype': 'categorical'},
                'sensitive': {'sdtype': 'categorical'},
                'datetime_feature': {'sdtype': 'datetime'},
            }
        }

        # Run
        result = EqualizedOddsImprovement._preprocess_data(
            data, 'prediction', 1, 'sensitive', 1, metadata
        )

        # Assert
        assert pd.api.types.is_numeric_dtype(result['datetime_feature'])

    def test_preprocess_data_does_not_modify_original(self):
        """Test _preprocess_data doesn't modify the original data."""
        # Setup
        original_data = pd.DataFrame({
            'prediction': ['True', 'False'],
            'sensitive': ['A', 'B'],
        })

        metadata = {
            'columns': {
                'prediction': {'sdtype': 'categorical'},
                'sensitive': {'sdtype': 'categorical'},
            }
        }

        # Run
        EqualizedOddsImprovement._preprocess_data(
            original_data, 'prediction', 'True', 'sensitive', 'A', metadata
        )

        # Assert
        assert original_data['prediction'].tolist() == ['True', 'False']
        assert original_data['sensitive'].tolist() == ['A', 'B']

    def test_compute_prediction_counts_both_groups(self):
        """Test _compute_prediction_counts with data for both sensitive groups."""
        # Setup
        predictions = np.array([1, 0, 1, 0, 1, 0])
        actuals = np.array([1, 0, 0, 1, 1, 0])
        sensitive_values = np.array([True, True, True, False, False, False])

        # Run
        result = EqualizedOddsImprovement._compute_prediction_counts(
            predictions, actuals, sensitive_values
        )

        # Assert
        expected_true = {
            'true_positive': 1,
            'false_positive': 1,
            'true_negative': 1,
            'false_negative': 0,
        }

        expected_false = {
            'true_positive': 1,
            'false_positive': 0,
            'true_negative': 1,
            'false_negative': 1,
        }

        assert result['True'] == expected_true
        assert result['False'] == expected_false

    def test_compute_prediction_counts_missing_group(self):
        """Test _compute_prediction_counts when one group has no data."""
        # Setup
        predictions = np.array([1, 0, 1])
        actuals = np.array([1, 0, 0])
        sensitive_values = np.array([True, True, True])

        # Run
        result = EqualizedOddsImprovement._compute_prediction_counts(
            predictions, actuals, sensitive_values
        )

        # Assert
        assert result['True'] == {
            'true_positive': 1,
            'false_positive': 1,
            'true_negative': 1,
            'false_negative': 0,
        }
        assert result['False'] == {
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0,
        }

    def test_compute_equalized_odds_score_perfect_fairness(self):
        """Test _compute_equalized_odds_score with perfect fairness."""
        # Setup
        prediction_counts = {
            'True': {
                'true_positive': 10,
                'false_positive': 5,
                'true_negative': 15,
                'false_negative': 5,
            },
            'False': {
                'true_positive': 10,
                'false_positive': 5,
                'true_negative': 15,
                'false_negative': 5,
            },
        }

        # Run
        score = EqualizedOddsImprovement._compute_equalized_odds_score(prediction_counts)

        # Assert
        assert score == 1.0

    def test_compute_equalized_odds_score_maximum_unfairness(self):
        """Test _compute_equalized_odds_score with maximum unfairness."""
        # Setup
        prediction_counts = {
            'True': {
                'true_positive': 10,  # TPR = 10/10 = 1.0
                'false_positive': 0,  # FPR = 0/10 = 0.0
                'true_negative': 10,
                'false_negative': 0,
            },
            'False': {
                'true_positive': 0,  # TPR = 0/10 = 0.0
                'false_positive': 10,  # FPR = 10/10 = 1.0
                'true_negative': 0,
                'false_negative': 10,
            },
        }

        # Run
        score = EqualizedOddsImprovement._compute_equalized_odds_score(prediction_counts)

        # Assert
        assert score == 0.0

    def test_compute_equalized_odds_score_handles_division_by_zero(self):
        """Test _compute_equalized_odds_score handles division by zero gracefully."""
        # Setup
        prediction_counts = {
            'True': {
                'true_positive': 0,
                'false_positive': 0,
                'true_negative': 0,
                'false_negative': 0,
            },
            'False': {
                'true_positive': 5,
                'false_positive': 5,
                'true_negative': 5,
                'false_negative': 5,
            },
        }

        # Run
        score = EqualizedOddsImprovement._compute_equalized_odds_score(prediction_counts)

        # Assert
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @patch.object(EqualizedOddsImprovement, '_train_classifier')
    @patch.object(EqualizedOddsImprovement, '_compute_prediction_counts')
    @patch.object(EqualizedOddsImprovement, '_compute_equalized_odds_score')
    def test_evaluate_dataset(self, mock_compute_score, mock_compute_counts, mock_train):
        """Test _evaluate_dataset integrates all components correctly."""
        # Setup
        mock_classifier = Mock()
        mock_classifier.predict.return_value = np.array([1, 0, 1])
        mock_train.return_value = mock_classifier

        mock_prediction_counts = {'True': {}, 'False': {}}
        mock_compute_counts.return_value = mock_prediction_counts

        mock_compute_score.return_value = 0.8

        train_data = pd.DataFrame({
            'feature': [1, 2, 3],
            'target': [0, 1, 0],
            'sensitive': [1, 0, 1],
        })

        validation_data = pd.DataFrame({
            'feature': [4, 5, 6],
            'target': [1, 0, 1],
            'sensitive': [1, 1, 0],
        })

        # Run
        result = EqualizedOddsImprovement._evaluate_dataset(
            train_data, validation_data, 'target', 'sensitive', 'sensitive_value'
        )

        # Assert
        mock_train.assert_called_once_with(train_data, 'target')

        expected_features = pd.DataFrame({'feature': [4, 5, 6], 'sensitive': [1, 1, 0]})
        mock_classifier.predict.assert_called_once()
        call_features = mock_classifier.predict.call_args[0][0]
        pd.testing.assert_frame_equal(call_features, expected_features)

        mock_compute_counts.assert_called_once()
        call_args = mock_compute_counts.call_args[0]
        np.testing.assert_array_equal(call_args[0], np.array([1, 0, 1]))  # predictions
        np.testing.assert_array_equal(call_args[1], np.array([1, 0, 1]))  # actuals
        np.testing.assert_array_equal(call_args[2], np.array([1, 1, 0]))  # sensitive_values

        mock_compute_score.assert_called_once_with(mock_prediction_counts)

        expected_result = {
            'equalized_odds': 0.8,
            'prediction_counts_validation': {
                'sensitive_value=True': {},
                'sensitive_value=False': {},
            },
        }
        assert result['equalized_odds'] == expected_result['equalized_odds']
        assert list(result['prediction_counts_validation'].keys()) == list(
            expected_result['prediction_counts_validation'].keys()
        )

    @patch('sdmetrics.single_table.equalized_odds._validate_tables')
    @patch('sdmetrics.single_table.equalized_odds._validate_prediction_column_name')
    @patch('sdmetrics.single_table.equalized_odds._validate_sensitive_column_name')
    @patch('sdmetrics.single_table.equalized_odds._validate_classifier')
    @patch('sdmetrics.single_table.equalized_odds._validate_required_columns')
    @patch('sdmetrics.single_table.equalized_odds._validate_data_and_metadata')
    @patch('sdmetrics.single_table.equalized_odds._validate_column_values_exist')
    @patch('sdmetrics.single_table.equalized_odds._validate_column_consistency')
    @patch.object(EqualizedOddsImprovement, '_validate_inputs')
    def test_validate_parameters_calls_all_validators(
        self,
        mock_validate_inputs,
        mock_validate_consistency,
        mock_validate_values,
        mock_validate_data_meta,
        mock_validate_required,
        mock_validate_classifier,
        mock_validate_sensitive,
        mock_validate_prediction,
        mock_validate_tables,
    ):
        """Test _validate_parameters calls all validation functions."""
        # Setup
        mock_validate_inputs.return_value = (pd.DataFrame(), pd.DataFrame(), {'columns': {}})

        real_training = pd.DataFrame({'col': [1, 2]})
        synthetic = pd.DataFrame({'col': [3, 4]})
        validation = pd.DataFrame({'col': [5, 6]})
        metadata = {'columns': {}}

        # Run
        EqualizedOddsImprovement._validate_parameters(
            real_training,
            synthetic,
            validation,
            metadata,
            'pred_col',
            'pos_label',
            'sens_col',
            'sens_val',
            'XGBoost',
        )

        # Assert
        mock_validate_tables.assert_called_once()
        mock_validate_prediction.assert_called_once_with('pred_col')
        mock_validate_sensitive.assert_called_once_with('sens_col')
        mock_validate_classifier.assert_called_once_with('XGBoost')
        mock_validate_required.assert_called_once()
        mock_validate_data_meta.assert_called_once()
        mock_validate_values.assert_called_once()
        mock_validate_consistency.assert_called_once()
        mock_validate_inputs.assert_called_once()

    @patch.object(EqualizedOddsImprovement, '_validate_parameters')
    @patch('sdmetrics.single_table.equalized_odds._process_data_with_metadata_ml_efficacy_metrics')
    @patch.object(EqualizedOddsImprovement, '_preprocess_data')
    @patch.object(EqualizedOddsImprovement, '_validate_data_sufficiency')
    @patch.object(EqualizedOddsImprovement, '_evaluate_dataset')
    def test_compute_breakdown_integration(
        self,
        mock_evaluate,
        mock_validate_sufficiency,
        mock_preprocess,
        mock_process_data,
        mock_validate,
    ):
        """Test compute_breakdown integrates all components correctly."""
        # Setup
        mock_process_data.return_value = (
            pd.DataFrame({'feature': [1, 2], 'target': [0, 1], 'sensitive': [0, 1]}),
            pd.DataFrame({'feature': [3, 4], 'target': [1, 0], 'sensitive': [1, 0]}),
            pd.DataFrame({'feature': [5, 6], 'target': [0, 1], 'sensitive': [0, 1]}),
        )

        mock_preprocess.side_effect = [
            pd.DataFrame({'feature': [1, 2], 'target': [0, 1], 'sensitive': [0, 1]}),  # real
            pd.DataFrame({'feature': [3, 4], 'target': [1, 0], 'sensitive': [1, 0]}),  # synthetic
            pd.DataFrame({'feature': [5, 6], 'target': [0, 1], 'sensitive': [0, 1]}),  # validation
        ]

        mock_evaluate.side_effect = [
            {'equalized_odds': 0.6, 'prediction_counts_validation': {}},  # real results
            {'equalized_odds': 0.8, 'prediction_counts_validation': {}},  # synthetic results
        ]

        real_training = pd.DataFrame({
            'feature': [1, 2],
            'target': ['A', 'B'],
            'sensitive': ['X', 'Y'],
        })
        synthetic = pd.DataFrame({'feature': [3, 4], 'target': ['B', 'A'], 'sensitive': ['Y', 'X']})
        validation = pd.DataFrame({
            'feature': [5, 6],
            'target': ['A', 'B'],
            'sensitive': ['X', 'Y'],
        })
        metadata = {'columns': {}}

        # Run
        result = EqualizedOddsImprovement.compute_breakdown(
            real_training, synthetic, validation, metadata, 'target', 'A', 'sensitive', 'X'
        )

        # Assert
        mock_validate.assert_called_once()

        mock_process_data.assert_called_once()

        assert mock_preprocess.call_count == 3

        assert mock_validate_sufficiency.call_count == 2

        assert mock_evaluate.call_count == 2

        expected_result = {
            'score': 0.6,
            'real_training_data': {'equalized_odds': 0.6, 'prediction_counts_validation': {}},
            'synthetic_data': {'equalized_odds': 0.8, 'prediction_counts_validation': {}},
        }
        assert abs(result['score'] - expected_result['score']) < 1e-10
        assert result['real_training_data'] == expected_result['real_training_data']
        assert result['synthetic_data'] == expected_result['synthetic_data']

    @patch.object(EqualizedOddsImprovement, 'compute_breakdown')
    def test_compute_returns_score_from_breakdown(self, mock_compute_breakdown):
        """Test compute method returns just the score from compute_breakdown."""
        # Setup
        mock_compute_breakdown.return_value = {
            'score': 0.75,
            'real_training_data': 0.6,
            'synthetic_data': 0.9,
        }

        # Run
        result = EqualizedOddsImprovement.compute(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, 'pred', 'pos', 'sens', 'val'
        )

        # Assert
        assert result == 0.75
        mock_compute_breakdown.assert_called_once()
