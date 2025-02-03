from sdmetrics.demos import load_demo
from sdmetrics.single_table.data_augmentation import BinaryClassifierPrecisionEfficacy


class TestBinaryClassifierPrecisionEfficacy:
    def test_end_to_end(self):
        """Test the metric end-to-end."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        real_training, real_validation = real_data.train_test_split(test_size=0.2, random_state=0)

        # Run
        score_breakdown = BinaryClassifierPrecisionEfficacy.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation,
            metadata=metadata,
            prediction_column_name='gender',
            minority_class_label='F',
            classifier='XGBoost',
            fixed_recall_value=0.8,
        )

        # Assert
        assert 'real_data_baseline' in score_breakdown
