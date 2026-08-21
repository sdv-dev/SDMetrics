import pandas as pd
import pytest

from sdmetrics.multi_table import ConstraintAdherence


@pytest.fixture
def real_data():
    return {
        'tableA': pd.DataFrame({
            'id': [1, 1, 2, 2, 3],
            'dob': ['1990-01-01', '1990-01-01', '1985-05-05', '1985-05-05', '1970-02-02'],
            'name': ['Ann', 'Ann', 'Bob', 'Bob', 'Cam'],
            'last_name': ['A', 'A', 'B', 'B', 'C'],
        })
    }


@pytest.fixture
def metadata():
    return {'tables': {'tableA': {'columns': {}}}}


@pytest.fixture
def constraint():
    return {
        'class_name': 'DenormalizedTable',
        'parameters': {
            'table_name': 'tableA',
            'denormalized_primary_key': 'id',
            'denormalized_column_names': ['dob', 'name', 'last_name'],
        },
    }


class TestConstraintAdherence:
    def test_compute(self, real_data, metadata, constraint):
        """Test ``compute`` returns the proportion of valid synthetic rows."""
        # Setup
        synthetic_data = {'tableA': real_data['tableA'].copy()}
        synthetic_data['tableA'].loc[3, 'name'] = 'Cam'

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 0.6

    def test_compute_invalid_real_data(self, real_data, metadata, constraint):
        """Test ``compute`` warns about the real data but still scores the synthetic data."""
        # Setup
        invalid_real_data = {'tableA': real_data['tableA'].copy()}
        invalid_real_data['tableA'].loc[0, 'name'] = 'Zoe'

        # Run
        warning_message = 'The real data does not adhere'
        with pytest.warns(UserWarning, match=warning_message):
            score = ConstraintAdherence.compute(invalid_real_data, real_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_compute_missing_table(self, real_data, metadata, constraint):
        """Test ``compute`` warns and returns NaN if the constraint can't be checked."""
        # Setup
        constraint['parameters']['table_name'] = 'MissingTable'
        expected_msg = (
            "Unable to check the constraint against the real data: "
            "The table 'MissingTable' is missing from the data."
        )

        # Run
        with pytest.warns(UserWarning, match=expected_msg):
            score = ConstraintAdherence.compute(real_data, real_data, metadata, constraint)

        # Assert
        assert pd.isna(score)

    def test_compute_unsupported_constraint(self, real_data, metadata):
        """Test ``compute`` warns and returns NaN if the constraint is not supported."""
        # Setup
        constraint = {'class_name': 'Unsupported', 'parameters': {}}

        # Run
        with pytest.warns(UserWarning, match='Unable to check the constraint'):
            score = ConstraintAdherence.compute(real_data, real_data, metadata, constraint)

        # Assert
        assert pd.isna(score)
