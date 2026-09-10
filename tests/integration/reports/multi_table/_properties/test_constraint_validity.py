import sys

import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from sdmetrics.demos import load_demo
from sdmetrics.errors import VisualizationUnavailableError
from sdmetrics.reports.multi_table._properties import ConstraintValidity


@pytest.fixture
def constraints():
    return [
        {
            'class_name': 'FixedCombinations',
            'parameters': {'table_name': 'sessions', 'column_names': ['device', 'os']},
        },
        {
            'class_name': 'Inequality',
            'parameters': {
                'table_name': 'transactions',
                'low_column_name': 'transaction_id',
                'high_column_name': 'amount',
            },
        },
    ]


class TestConstraintValidity:
    def test_end_to_end(self, constraints):
        """Test the constraint validity property end to end."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        constraint_validity = ConstraintValidity()
        expected_details = pd.DataFrame({
            'Constraint': ['FixedCombinations', 'Inequality'],
            'Metric': ['ConstraintAdherence', 'ConstraintAdherence'],
            'Parameters': [constraints[0]['parameters'], constraints[1]['parameters']],
            'Score': [1.0, 1.0],
        })

        # Run
        result = constraint_validity.get_score(real_data, synthetic_data, metadata, constraints)
        details = constraint_validity.get_details()

        # Assert
        assert result == 1.0
        pd.testing.assert_frame_equal(details, expected_details)

    def test_end_to_end_with_invalid_rows(self, constraints):
        """Test the score reflects the proportion of rows that break the constraints."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        synthetic_data['sessions'] = synthetic_data['sessions'].copy()
        synthetic_data['sessions']['os'] = 'unknown'
        constraint_validity = ConstraintValidity()

        # Run
        result = constraint_validity.get_score(real_data, synthetic_data, metadata, constraints)
        details = constraint_validity.get_details()

        # Assert
        assert result == 0.5
        assert details['Score'].tolist() == [0.0, 1.0]

    def test_end_to_end_with_unsupported_constraint(self, constraints):
        """Test an unsupported constraint gets a NaN score and does not affect the average."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        constraints.append({'class_name': 'Unsupported', 'parameters': {}})
        constraint_validity = ConstraintValidity()
        expected_details = pd.DataFrame({
            'Constraint': ['FixedCombinations', 'Inequality', 'Unsupported'],
            'Metric': ['ConstraintAdherence', 'ConstraintAdherence', 'ConstraintAdherence'],
            'Parameters': [constraints[0]['parameters'], constraints[1]['parameters'], {}],
            'Score': [1.0, 1.0, np.nan],
            'Error': [None, None, "ValueError: Unsupported constraint class 'Unsupported'."],
        })

        # Run
        result = constraint_validity.get_score(real_data, synthetic_data, metadata, constraints)
        details = constraint_validity.get_details()

        # Assert
        assert result == 1.0
        pd.testing.assert_frame_equal(details, expected_details)

    def test_end_to_end_without_constraints(self):
        """Test the score is NaN when there are no constraints."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        constraint_validity = ConstraintValidity()

        # Run
        result = constraint_validity.get_score(real_data, synthetic_data, metadata, [])
        details = constraint_validity.get_details()

        # Assert
        assert pd.isna(result)
        assert details.empty
        assert list(details.columns) == ['Constraint', 'Metric', 'Parameters', 'Score']

    def test_with_progress_bar(self, constraints, capsys):
        """Test that the progress bar is updated once per constraint."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        constraint_validity = ConstraintValidity()
        num_constraints = constraint_validity._get_num_iterations(metadata, constraints)
        progress_bar = tqdm(total=num_constraints, file=sys.stdout)

        # Run
        result = constraint_validity.get_score(
            real_data, synthetic_data, metadata, constraints, progress_bar
        )
        progress_bar.close()
        output = capsys.readouterr().out

        # Assert
        assert result == 1.0
        assert num_constraints == 2
        assert '100%' in output
        assert f'{num_constraints}/{num_constraints}' in output

    def test_get_visualization(self, constraints):
        """Test ``get_visualization`` raises an error."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        constraint_validity = ConstraintValidity()
        constraint_validity.get_score(real_data, synthetic_data, metadata, constraints)
        expected_message = (
            'Error: No visualization is available for Constraint Validity. To see the '
            "detailed score breakdowns, use the 'get_details' function."
        )

        # Run and Assert
        with pytest.raises(VisualizationUnavailableError, match=expected_message):
            constraint_validity.get_visualization()
