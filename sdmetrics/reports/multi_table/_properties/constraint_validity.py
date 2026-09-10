"""Constraint validity property for multi-table."""

import numpy as np
import pandas as pd

from sdmetrics.errors import VisualizationUnavailableError
from sdmetrics.multi_table.statistical import ConstraintAdherence
from sdmetrics.reports.multi_table._properties.base import BaseMultiTableProperty


class ConstraintValidity(BaseMultiTableProperty):
    """Constraint Validity property class for multi-table.

    This property evaluates whether the synthetic data is valid according
    to the defined constraints. The constraint adherence metric is computed for
    every constraint and the final score is the average over all the constraints.
    """

    _num_iteration_case = 'constraint'

    def _get_num_iterations(self, metadata, constraints=None):
        """Get the number of iterations for the property, which is one per constraint."""
        return len(constraints) if constraints else 0

    def _generate_details(
        self, real_data, synthetic_data, metadata, constraints, progress_bar=None
    ):
        """Generate the details dataframe for the constraint validity property.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The real data.
            synthetic_data (dict[str, pandas.DataFrame]):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            constraints (list[dict]):
                The constraints to check, each represented as a dictionary with a
                ``class_name`` and a ``parameters`` key.
            progress_bar (tqdm.tqdm or None):
                The progress bar object. Defaults to None.
        """
        constraint_names, constraint_parameters, scores, error_messages = [], [], [], []
        for constraint in constraints or []:
            try:
                score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)
                error_message = None
            except Exception as e:
                score = np.nan
                error_message = f'{type(e).__name__}: {e}'
            finally:
                if progress_bar:
                    progress_bar.update()

            is_dict = isinstance(constraint, dict)
            constraint_names.append(constraint.get('class_name') if is_dict else None)
            constraint_parameters.append(constraint.get('parameters') if is_dict else None)
            scores.append(score)
            error_messages.append(error_message)

        self.details = pd.DataFrame({
            'Constraint': constraint_names,
            'Metric': [ConstraintAdherence.__name__] * len(constraint_names),
            'Parameters': constraint_parameters,
            'Score': scores,
            'Error': error_messages
        })

    def get_score(self, real_data, synthetic_data, metadata, constraints=None, progress_bar=None):
        """Get the average score of all the individual metric scores computed.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The real data.
            synthetic_data (dict[str, pandas.DataFrame]):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            constraints (list[dict]):
                The constraints to check, each represented as a dictionary with a
                ``class_name`` and a ``parameters`` key.
            progress_bar (tqdm.tqdm or None):
                The progress bar object. Defaults to None.

        Returns:
            float:
                The average score for the property for all the individual metric scores computed.
        """
        self._generate_details(real_data, synthetic_data, metadata, constraints, progress_bar)

        self.is_computed = True

        if 'Error' in self.details.columns and self.details['Error'].isna().all():
            self.details = self.details.drop('Error', axis=1)
        elif 'Error' in self.details.columns:
            self.details['Error'] = self.details['Error'].replace({np.nan: None})

        return self._compute_average()

    def get_visualization(self):
        """Raise an error because there is no visualization for this property.

        Raises:
            VisualizationUnavailableError
        """
        raise VisualizationUnavailableError(
            'Error: No visualization is available for Constraint Validity. To see the '
            "detailed score breakdowns, use the 'get_details' function."
        )

    def get_details(self, table_name=None):
        """Return the details table for the property.

        Args:
            table_name (str):
                The name of the table to return details for.
                Defaults to None.

        Returns:
            pandas.DataFrame
        """
        return self.details.copy()
