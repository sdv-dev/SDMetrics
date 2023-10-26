from unittest.mock import patch

import pandas as pd

from sdmetrics.column_pairs.statistical import ReferentialIntegrity


class TestReferentialIntegrity:

    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method."""
        # Setup
        real_data = pd.DataFrame({
            'primary_key': [1, 2, 3, 4, 5],
            'foreign_key': [1, 2, 3, 2, 1]
        })
        synthetic_data = pd.DataFrame({
            'primary_key': [1, 2, 3, 4, 5],
            'foreign_key': [1, 6, 3, 4, 5]
        })

        metric = ReferentialIntegrity()
        tuple_real = (real_data['primary_key'], real_data['foreign_key'])
        tuple_synthetic = (synthetic_data['primary_key'], synthetic_data['foreign_key'])

        # Run
        result = metric.compute_breakdown(tuple_real, tuple_synthetic)

        # Assert
        assert result == {'score': 0.8}

    @patch('sdmetrics.column_pairs.statistical.referential_integrity.LOGGER')
    def test_compute_breakdown_with_missing_relations_real_data(self, logger_mock):
        """Test the ``compute_breakdown`` when there is missing relationships in the real data."""
        # Setup
        real_data = pd.DataFrame({
            'primary_key': [1, 2, 3, 4, 5],
            'foreign_key': [1, 2, 6, 2, 1]
        })
        synthetic_data = pd.DataFrame({
            'primary_key': [1, 2, 3, 4, 5],
            'foreign_key': [1, 6, 3, 4, 5]
        })

        metric = ReferentialIntegrity()
        tuple_real = (real_data['primary_key'], real_data['foreign_key'])
        tuple_synthetic = (synthetic_data['primary_key'], synthetic_data['foreign_key'])

        # Run
        result = metric.compute_breakdown(tuple_real, tuple_synthetic)

        # Assert
        expected_message = "The real data has foreign keys that don't reference any primary key."
        assert result == {'score': 0.8}
        logger_mock.info.assert_called_once_with(expected_message)

    @patch('sdmetrics.column_pairs.statistical.referential_integrity.'
           'ReferentialIntegrity.compute_breakdown')
    def test_compute(self, compute_breakdown_mock):
        """Test the ``compute`` method."""
        # Setup
        real_data = pd.Series(['A', 'B', 'C', 'B', 'A'])
        synthetic_data = pd.Series(['A', 'B', 'C', 'D', 'E'])
        metric = ReferentialIntegrity()
        compute_breakdown_mock.return_value = {'score': 0.6}

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        compute_breakdown_mock.assert_called_once_with(real_data, synthetic_data)
        assert result == 0.6
