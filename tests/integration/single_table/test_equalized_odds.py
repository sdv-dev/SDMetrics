"""Integration tests for EqualizedOddsImprovement metric."""

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table import EqualizedOddsImprovement


@pytest.fixture
def get_data_metadata():
    # Real training data - somewhat biased
    real_training = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 200),
        'feature2': np.random.normal(0, 1, 200),
        'race': np.random.choice(['A', 'B'], 200, p=[0.3, 0.7]),
        'loan_approved': np.random.choice(['True', 'False'], 200, p=[0.6, 0.4]),
    })

    # Make the real data slightly biased - A applicants have slightly lower approval rates
    group_a_mask = real_training['race'] == 'A'
    real_training.loc[group_a_mask, 'loan_approved'] = np.random.choice(
        ['True', 'False'], sum(group_a_mask), p=[0.5, 0.5]
    )

    synthetic = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 200),
        'feature2': np.random.normal(0, 1, 200),
        'race': np.random.choice(['A', 'B'], 200, p=[0.3, 0.7]),
        'loan_approved': np.random.choice(['True', 'False'], 200, p=[0.6, 0.4]),
    })

    validation = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'race': np.random.choice(['A', 'B'], 100, p=[0.3, 0.7]),
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

    return real_training, synthetic, validation, metadata


class TestEqualizedOddsImprovement:
    """Test the EqualizedOddsImprovement metric."""

    def test_compute_breakdown_basic(self, get_data_metadata):
        """Test basic functionality of compute_breakdown."""
        # Setup
        real_training, synthetic, validation, metadata = get_data_metadata

        # Run
        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='loan_approved',
            positive_class_label='True',
            sensitive_column_name='race',
            sensitive_column_value='A',
            classifier='XGBoost',
        )

        # Assert
        # Verify all scores are in valid range
        assert 0.0 <= result['score'] <= 1.0
        assert 0.0 <= result['real_training_data']['equalized_odds'] <= 1.0
        assert 0.0 <= result['synthetic_data']['equalized_odds'] <= 1.0

    def test_compute_breakdown_biased_real(self, get_data_metadata):
        """Test with heavily biased real data and balanced synthetic data."""
        # Setup
        np.random.seed(42)
        real_training, synthetic, validation, metadata = get_data_metadata

        # Make real data heavily biased - group A has very low approval rate
        group_a_mask = real_training['race'] == 'A'
        group_b_mask = real_training['race'] == 'B'

        real_training.loc[group_a_mask, 'loan_approved'] = np.random.choice(
            ['True', 'False'], sum(group_a_mask), p=[0.1, 0.9]
        )
        real_training.loc[group_b_mask, 'loan_approved'] = np.random.choice(
            ['True', 'False'], sum(group_b_mask), p=[0.9, 0.1]
        )

        # Run
        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='loan_approved',
            positive_class_label='True',
            sensitive_column_name='race',
            sensitive_column_value='A',
            classifier='XGBoost',
        )

        # Assert
        # Verify all scores are in valid range
        assert result['score'] > 0.5
        assert result['real_training_data']['equalized_odds'] < 0.5
        assert result['synthetic_data']['equalized_odds'] > 0.5

    def test_compute_breakdown_biased_synthetic(self, get_data_metadata):
        """Test with heavily biased synthetic data and balanced real data."""
        # Setup
        np.random.seed(42)
        real_training, synthetic, validation, metadata = get_data_metadata

        # Make synthetic data heavily biased - group A has very low approval rate
        group_a_mask = synthetic['race'] == 'A'
        group_b_mask = synthetic['race'] == 'B'

        synthetic.loc[group_a_mask, 'loan_approved'] = np.random.choice(
            ['True', 'False'], sum(group_a_mask), p=[0.9, 0.1]
        )
        synthetic.loc[group_b_mask, 'loan_approved'] = np.random.choice(
            ['True', 'False'], sum(group_b_mask), p=[0.1, 0.9]
        )

        # Run
        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='loan_approved',
            positive_class_label='True',
            sensitive_column_name='race',
            sensitive_column_value='A',
            classifier='XGBoost',
        )

        # Assert
        # Verify all scores are in valid range
        assert result['score'] < 0.5
        assert result['real_training_data']['equalized_odds'] > 0.5
        assert result['synthetic_data']['equalized_odds'] < 0.5

    def test_compute_breakdown_output_format(self, get_data_metadata):
        """Test that compute_breakdown produces the expected output format."""
        # Setup
        real_training, synthetic, validation, metadata = get_data_metadata

        # Run
        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='loan_approved',
            positive_class_label='True',
            sensitive_column_name='race',
            sensitive_column_value='A',
            classifier='XGBoost',
        )

        # Assert
        assert isinstance(result, dict)
        expected_top_keys = {'score', 'real_training_data', 'synthetic_data'}
        assert set(result.keys()) == expected_top_keys

        assert isinstance(result['score'], float)
        assert 0.0 <= result['score'] <= 1.0

        for data in [result['real_training_data'], result['synthetic_data']]:
            assert isinstance(data, dict)
            expected_data_keys = {'equalized_odds', 'prediction_counts_validation'}
            assert set(data.keys()) == expected_data_keys

            assert isinstance(data['equalized_odds'], float)
            assert 0.0 <= data['equalized_odds'] <= 1.0

            pred_counts = data['prediction_counts_validation']
            assert isinstance(pred_counts, dict)
            expected_group_keys = {'A=True', 'A=False'}
            assert set(pred_counts.keys()) == expected_group_keys

            expected_confusion_keys = {
                'true_positive',
                'false_positive',
                'true_negative',
                'false_negative',
            }
            for group_key in expected_group_keys:
                group_counts = pred_counts[group_key]
                assert isinstance(group_counts, dict)
                assert set(group_counts.keys()) == expected_confusion_keys

                for count_key in expected_confusion_keys:
                    count_value = group_counts[count_key]
                    assert isinstance(count_value, int)
                    assert count_value >= 0

    def test_compute_basic(self, get_data_metadata):
        """Test basic functionality of compute method."""
        # Setup
        real_training, synthetic, validation, metadata = get_data_metadata

        # Run
        score = EqualizedOddsImprovement.compute(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='loan_approved',
            positive_class_label='True',
            sensitive_column_name='race',
            sensitive_column_value='A',
            classifier='XGBoost',
        )

        # Assert
        assert 0.0 <= score <= 1.0

    def test_insufficient_data_error(self, get_data_metadata):
        """Test that insufficient data raises appropriate error."""
        # Setup
        real_training, synthetic, validation, metadata = get_data_metadata

        for data in [real_training, synthetic]:
            group_a_mask = data['race'] == 'A'
            data.loc[group_a_mask, 'loan_approved'] = 'True'

        # Run & Assert
        with pytest.raises(ValueError, match='Insufficient .* examples'):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='loan_approved',
                positive_class_label='True',
                sensitive_column_name='race',
                sensitive_column_value='A',
                classifier='XGBoost',
            )

        # Setup
        for data in [real_training, synthetic]:
            group_a_mask = data['race'] == 'A'
            data.loc[group_a_mask, 'loan_approved'] = 'False'

        # Run & Assert
        with pytest.raises(ValueError, match='Insufficient .* examples'):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='loan_approved',
                positive_class_label='True',
                sensitive_column_name='race',
                sensitive_column_value='A',
                classifier='XGBoost',
            )

    def test_missing_columns_error(self):
        """Test that missing required columns raise appropriate error."""
        # Setup
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

        # Run & Assert
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
        # Setup
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

        # Run & Assert
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

    def test_three_classes(self):
        """Test the metric with three classes."""
        # Setup
        real_training = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'race': np.random.choice(['A', 'B', 'C'], 100),
            'loan_approved': np.random.choice(['True', 'False', 'Unknown'], 100),
        })

        synthetic = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'race': np.random.choice(['A', 'B', 'C'], 100),
            'loan_approved': np.random.choice(['True', 'False', 'Unknown'], 100),
        })

        validation = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50),
            'race': np.random.choice(['A', 'B', 'C'], 50),
            'loan_approved': np.random.choice(['True', 'False', 'Unknown'], 50),
        })

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'feature2': {'sdtype': 'numerical'},
                'race': {'sdtype': 'categorical'},
                'loan_approved': {'sdtype': 'categorical'},
            }
        }

        # Run
        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='loan_approved',
            positive_class_label='True',
            sensitive_column_name='race',
            sensitive_column_value='A',
            classifier='XGBoost',
        )

        # Assert
        # Verify all scores are in valid range
        assert 0.0 <= result['score'] <= 1.0
        assert 0.0 <= result['real_training_data']['equalized_odds'] <= 1.0
        assert 0.0 <= result['synthetic_data']['equalized_odds'] <= 1.0

    def test_perfect_fairness_case(self):
        """Test case where both datasets have perfect fairness."""

        # Setup
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

        # Run
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

        # Assert
        # Both should have high equalized odds scores
        assert 0.0 <= result['score'] <= 1.0
        assert 0.0 <= result['real_training_data']['equalized_odds'] <= 1.0
        assert 0.0 <= result['synthetic_data']['equalized_odds'] <= 1.0

    def test_parameter_validation_type_errors(self, get_data_metadata):
        """Test that parameter validation catches type errors."""
        # Setup
        real_training, synthetic, validation, metadata = get_data_metadata

        # Run & Assert
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

    def test_parameter_validation_value_errors(self, get_data_metadata):
        """Test that parameter validation catches value errors."""
        # Setup
        real_training, synthetic, validation, metadata = get_data_metadata

        # Run & Assert
        # Test positive_class_label not found
        with pytest.raises(
            ValueError,
            match='The value `999` is not present in the column `loan_approved` for the '
            'real training data',
        ):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='loan_approved',
                positive_class_label=999,
                sensitive_column_name='race',
                sensitive_column_value='A',
                classifier='XGBoost',
            )

        # Test sensitive_column_value not found
        with pytest.raises(
            ValueError, match="Value '999' not found in real_training_data\\['race'\\]"
        ):
            EqualizedOddsImprovement.compute_breakdown(
                real_training_data=real_training,
                synthetic_data=synthetic,
                real_validation_data=validation,
                metadata=metadata,
                prediction_column_name='loan_approved',
                positive_class_label='True',
                sensitive_column_name='race',
                sensitive_column_value=999,
                classifier='XGBoost',
            )

    def test_validation_data_column_mismatch(self):
        """Test that validation data with different columns raises error."""
        # Setup
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

        # Run & Assert
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

    def test_sensitive_column_nan(self):
        """Test that the metric handles NaN values in the sensitive column."""
        # Setup
        n = 1000

        # Create data with NaN values in sensitive column
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n),
            'feature2': np.random.normal(0, 1, n),
            'sensitive': np.random.choice(['A', 'B', np.nan], n),
            'target': np.random.choice(['True', 'False'], n),
        })

        real_training = data.iloc[: int(0.4 * n)].reset_index(drop=True)
        synthetic = data.iloc[int(0.4 * n) : int(0.8 * n)].reset_index(drop=True)
        validation = data.iloc[int(0.8 * n) :].reset_index(drop=True)

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'feature2': {'sdtype': 'numerical'},
                'sensitive': {'sdtype': 'categorical'},
                'target': {'sdtype': 'categorical'},
            }
        }

        # Run
        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='target',
            positive_class_label='True',
            sensitive_column_name='sensitive',
            sensitive_column_value='A',
        )

        # Assert
        assert 0.0 <= result['score'] <= 1.0
        assert 0.0 <= result['real_training_data']['equalized_odds'] <= 1.0
        assert 0.0 <= result['synthetic_data']['equalized_odds'] <= 1.0

        # Run
        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='target',
            positive_class_label='True',
            sensitive_column_name='sensitive',
            sensitive_column_value=str(np.nan),  # NaN value in sensitive column
        )

        # Assert
        assert 0.0 <= result['score'] <= 1.0
        assert 0.0 <= result['real_training_data']['equalized_odds'] <= 1.0
        assert 0.0 <= result['synthetic_data']['equalized_odds'] <= 1.0

    def test_sensitive_column_nan_integers(self):
        """Test that the metric handles NaN values in the sensitive column."""
        # Setup
        n = 1000

        # Create data with NaN values in sensitive column
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n),
            'feature2': np.random.normal(0, 1, n),
            'sensitive': np.random.choice([0, 1, np.nan], n),
            'target': np.random.choice(['True', 'False'], n),
        })

        real_training = data.iloc[: int(0.4 * n)].reset_index(drop=True)
        synthetic = data.iloc[int(0.4 * n) : int(0.8 * n)].reset_index(drop=True)
        validation = data.iloc[int(0.8 * n) :].reset_index(drop=True)

        metadata = {
            'columns': {
                'feature1': {'sdtype': 'numerical'},
                'feature2': {'sdtype': 'numerical'},
                'sensitive': {'sdtype': 'categorical'},
                'target': {'sdtype': 'categorical'},
            }
        }

        # Run
        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='target',
            positive_class_label='True',
            sensitive_column_name='sensitive',
            sensitive_column_value=1,
        )

        # Assert
        assert 0.0 <= result['score'] <= 1.0
        assert 0.0 <= result['real_training_data']['equalized_odds'] <= 1.0
        assert 0.0 <= result['synthetic_data']['equalized_odds'] <= 1.0

        # Run
        result = EqualizedOddsImprovement.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic,
            real_validation_data=validation,
            metadata=metadata,
            prediction_column_name='target',
            positive_class_label='True',
            sensitive_column_name='sensitive',
            sensitive_column_value=np.nan,  # NaN value in sensitive column
        )

        # Assert
        assert 0.0 <= result['score'] <= 1.0
        assert 0.0 <= result['real_training_data']['equalized_odds'] <= 1.0
        assert 0.0 <= result['synthetic_data']['equalized_odds'] <= 1.0
