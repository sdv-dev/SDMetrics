"""Integration tests for EqualizedOddsImprovement metric."""

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table import EqualizedOddsImprovement


class TestEqualizedOddsImprovement:
    """Test the EqualizedOddsImprovement metric."""

    def test_compute_breakdown_basic(self):
        """Test basic functionality of compute_breakdown."""
        # Create synthetic datasets with clear patterns
        np.random.seed(42)

        # Real training data - somewhat biased
        real_training = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.normal(0, 1, 200),
            'race': np.random.choice(['Asian', 'White'], 200, p=[0.3, 0.7]),
            'loan_approved': np.random.choice(['True', 'False'], 200, p=[0.6, 0.4]),
        })

        # Make the real data slightly biased - Asian applicants have slightly lower approval rates
        asian_mask = real_training['race'] == 'Asian'
        real_training.loc[asian_mask, 'loan_approved'] = np.random.choice(
            ['True', 'False'], sum(asian_mask), p=[0.5, 0.5]
        )

        # Synthetic data - more balanced
        synthetic = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.normal(0, 1, 200),
            'race': np.random.choice(['Asian', 'White'], 200, p=[0.3, 0.7]),
            'loan_approved': np.random.choice(['True', 'False'], 200, p=[0.6, 0.4]),
        })

        # Validation data
        validation = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'race': np.random.choice(['Asian', 'White'], 100, p=[0.3, 0.7]),
            'loan_approved': np.random.choice(['True', 'False'], 100, p=[0.6, 0.4]),
        })

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'feature2': {'sdtype': 'numerical'},
                'race': {'sdtype': 'categorical'},
                'loan_approved': {'sdtype': 'categorical'},
            }
        }

        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='loan_approved',
            positive_class_label='True',
            sensitive_column_name='race',
            sensitive_column_value='Asian',
            classifier='XGBoost',
        )

        # Verify structure
        assert 'score' in result
        assert 'real_training_data' in result
        assert 'synthetic_data' in result

        # Verify score is in valid range
        assert 0.0 <= result['score'] <= 1.0

        # Verify real_training_data structure
        real_data = result['real_training_data']
        assert 'equalized_odds' in real_data
        assert 'prediction_counts_validation' in real_data
        assert 0.0 <= real_data['equalized_odds'] <= 1.0

        # Verify prediction counts structure
        pred_counts = real_data['prediction_counts_validation']
        assert 'race=Asian' in pred_counts
        assert 'race≠Asian' in pred_counts

        for group in pred_counts.values():
            assert 'true_positive' in group
            assert 'false_positive' in group
            assert 'true_negative' in group
            assert 'false_negative' in group
            assert all(isinstance(v, int) for v in group.values())

        # Verify synthetic_data has same structure
        synthetic_data = result['synthetic_data']
        assert 'equalized_odds' in synthetic_data
        assert 'prediction_counts_validation' in synthetic_data
        assert 0.0 <= synthetic_data['equalized_odds'] <= 1.0

    def test_compute_basic(self):
        """Test basic functionality of compute method."""
        np.random.seed(42)

        # Create simple datasets
        real_training = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        synthetic = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        validation = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'sensitive': np.random.choice([0, 1], 50),
            'target': np.random.choice([0, 1], 50),
        })

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'sensitive': {'sdtype': 'categorical'},
                'target': {'sdtype': 'categorical'},
            }
        }

        score = EqualizedOddsImprovement.compute(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='target',
            positive_class_label=1,
            sensitive_column_name='sensitive',
            sensitive_column_value=1,
            classifier='XGBoost',
        )

        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0

    def test_insufficient_data_error(self):
        """Test that insufficient data raises appropriate error."""
        # Create dataset with insufficient examples for one group
        # The sensitive group (sensitive=1) will have all positive examples (target=1)
        # but insufficient negative examples
        real_training = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'sensitive': [0] * 45 + [1] * 5,  # Very few in sensitive group
            'target': [0] * 40 + [1] * 5 + [1] * 5,  # Sensitive group has all positive examples
        })

        synthetic = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'sensitive': np.random.choice([0, 1], 50),
            'target': np.random.choice([0, 1], 50),
        })

        validation = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 30),
            'sensitive': np.random.choice([0, 1], 30),
            'target': np.random.choice([0, 1], 30),
        })

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'sensitive': {'sdtype': 'categorical'},
                'target': {'sdtype': 'categorical'},
            }
        }

        with pytest.raises(ValueError, match='Insufficient .* examples'):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='target',
                positive_class_label=1,
                sensitive_column_name='sensitive',
                sensitive_column_value=1,
                classifier='XGBoost',
            )

    def test_missing_columns_error(self):
        """Test that missing required columns raise appropriate error."""
        real_training = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100),
            # Missing sensitive column
        })

        synthetic = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        validation = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'sensitive': np.random.choice([0, 1], 50),
            'target': np.random.choice([0, 1], 50),
        })

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'sensitive': {'sdtype': 'categorical'},
                'target': {'sdtype': 'categorical'},
            }
        }

        with pytest.raises(ValueError, match='Missing columns in real_training_data'):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='target',
                positive_class_label=1,
                sensitive_column_name='sensitive',
                sensitive_column_value=1,
                classifier='XGBoost',
            )

    def test_unsupported_classifier_error(self):
        """Test that unsupported classifier raises appropriate error."""
        real_training = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        synthetic = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        validation = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'sensitive': np.random.choice([0, 1], 50),
            'target': np.random.choice([0, 1], 50),
        })

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'sensitive': {'sdtype': 'categorical'},
                'target': {'sdtype': 'categorical'},
            }
        }

        with pytest.raises(ValueError, match='Currently only `XGBoost` is supported as classifier'):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='target',
                positive_class_label=1,
                sensitive_column_name='sensitive',
                sensitive_column_value=1,
                classifier='RandomForest',  # Unsupported
            )

    def test_string_labels(self):
        """Test the metric with string labels for sensitive and prediction columns."""
        np.random.seed(42)

        real_training = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'hired': np.random.choice(['Yes', 'No'], 100),
        })

        synthetic = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'hired': np.random.choice(['Yes', 'No'], 100),
        })

        validation = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50),
            'gender': np.random.choice(['Male', 'Female'], 50),
            'hired': np.random.choice(['Yes', 'No'], 50),
        })

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'feature2': {'sdtype': 'numerical'},
                'gender': {'sdtype': 'categorical'},
                'hired': {'sdtype': 'categorical'},
            }
        }

        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='hired',
            positive_class_label='Yes',
            sensitive_column_name='gender',
            sensitive_column_value='Female',
            classifier='XGBoost',
        )

        # Verify the result structure
        assert 'score' in result
        assert 0.0 <= result['score'] <= 1.0

        # Check that sensitive column labels are properly formatted
        pred_counts = result['real_training_data']['prediction_counts_validation']
        assert 'gender=Female' in pred_counts
        assert 'gender≠Female' in pred_counts

    def test_perfect_fairness_case(self):
        """Test case where both datasets have perfect fairness."""
        np.random.seed(42)

        # Create perfectly fair datasets
        def create_fair_data(n):
            data = pd.DataFrame({
                'feature1': np.random.normal(0, 1, n),
                'sensitive': np.random.choice([0, 1], n),
                'target': np.random.choice([0, 1], n),
            })
            # Ensure perfect balance within each sensitive group
            for sensitive_val in [0, 1]:
                mask = data['sensitive'] == sensitive_val
                n_group = sum(mask)
                if n_group > 0:
                    # Make exactly half positive in each group
                    targets = [1] * (n_group // 2) + [0] * (n_group - n_group // 2)
                    data.loc[mask, 'target'] = targets
            return data

        real_training = create_fair_data(100)
        synthetic = create_fair_data(100)
        validation = create_fair_data(60)

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'sensitive': {'sdtype': 'categorical'},
                'target': {'sdtype': 'categorical'},
            }
        }

        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='target',
            positive_class_label=1,
            sensitive_column_name='sensitive',
            sensitive_column_value=1,
            classifier='XGBoost',
        )

        # Both should have high equalized odds scores
        assert result['real_training_data']['equalized_odds'] >= 0.5
        assert result['synthetic_data']['equalized_odds'] >= 0.5

        # Final score should be around 0.5 (no improvement needed)
        assert 0.3 <= result['score'] <= 0.7  # Allow some variance due to randomness

    def test_parameter_validation_type_errors(self):
        """Test that parameter validation catches type errors."""
        real_training = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        synthetic = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        validation = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'sensitive': np.random.choice([0, 1], 50),
            'target': np.random.choice([0, 1], 50),
        })

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'sensitive': {'sdtype': 'categorical'},
                'target': {'sdtype': 'categorical'},
            }
        }

        # Test non-string column names
        with pytest.raises(TypeError, match='`prediction_column_name` must be a string'):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name=123,  # Should be string
                positive_class_label=1,
                sensitive_column_name='sensitive',
                sensitive_column_value=1,
                classifier='XGBoost',
            )

        with pytest.raises(TypeError, match='`sensitive_column_name` must be a string'):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='target',
                positive_class_label=1,
                sensitive_column_name=456,  # Should be string
                sensitive_column_value=1,
                classifier='XGBoost',
            )

        # Test non-DataFrame inputs
        with pytest.raises(
            ValueError,
            match='`real_training_data`, `synthetic_data` and `real_validation_data` '
            'must be pandas DataFrames',
        ):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data='not_a_dataframe',
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='target',
                positive_class_label=1,
                sensitive_column_name='sensitive',
                sensitive_column_value=1,
                classifier='XGBoost',
            )

    def test_parameter_validation_value_errors(self):
        """Test that parameter validation catches value errors."""
        real_training = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        synthetic = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        validation = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'sensitive': np.random.choice([0, 1], 50),
            'target': np.random.choice([0, 1], 50),
        })

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'sensitive': {'sdtype': 'categorical'},
                'target': {'sdtype': 'categorical'},
            }
        }

        # Test positive_class_label not found
        with pytest.raises(
            ValueError,
            match='The value `999` is not present in the column `target` for the '
            'real training data',
        ):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='target',
                positive_class_label=999,  # Doesn't exist in data
                sensitive_column_name='sensitive',
                sensitive_column_value=1,
                classifier='XGBoost',
            )

        # Test sensitive_column_value not found
        with pytest.raises(
            ValueError, match="Value '999' not found in real_training_data\\['sensitive'\\]"
        ):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='target',
                positive_class_label=1,
                sensitive_column_name='sensitive',
                sensitive_column_value=999,  # Doesn't exist in data
                classifier='XGBoost',
            )

    def test_validation_data_column_mismatch(self):
        """Test that validation data with different columns raises error."""
        real_training = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        synthetic = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'sensitive': np.random.choice([0, 1], 100),
            'target': np.random.choice([0, 1], 100),
        })

        validation = pd.DataFrame({
            'different_feature': np.random.normal(0, 1, 50),  # Different column name
            'sensitive': np.random.choice([0, 1], 50),
            'target': np.random.choice([0, 1], 50),
        })

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'sensitive': {'sdtype': 'categorical'},
                'target': {'sdtype': 'categorical'},
            }
        }

        with pytest.raises(ValueError, match='real_validation_data must have the same columns'):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='target',
                positive_class_label=1,
                sensitive_column_name='sensitive',
                sensitive_column_value=1,
                classifier='XGBoost',
            )
