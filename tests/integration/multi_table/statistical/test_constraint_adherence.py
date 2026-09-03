import re
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table import ConstraintAdherence


class TestConstraintAdherence:
    def test_fixed_combinations(self):
        """Test the score is 1.0 when the synthetic data keeps the real combinations."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'A': [1, 2, 3, 1, 2, 1],
                'B': [10, 20, 30, 10, 20, 10],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'A': {'sdtype': 'categorical'},
                        'B': {'sdtype': 'categorical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'FixedCombinations',
            'parameters': {'column_names': ['A', 'B'], 'table_name': 'table1'},
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_fixed_combinations_with_a_new_combination(self):
        """Test the score counts a combination that the real data does not have."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'A': [1, 2, 3, 1, 2, 1],
                'B': [10, 20, 30, 10, 20, 10],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['table1'].loc[0, 'B'] = 20
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'A': {'sdtype': 'categorical'},
                        'B': {'sdtype': 'categorical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'FixedCombinations',
            'parameters': {'column_names': ['A', 'B'], 'table_name': 'table1'},
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 5 / 6

    @pytest.mark.parametrize('column_name', ['A', 'B'])
    def test_fixed_increments(self, column_name):
        """Test the score is 1.0 when every value is a multiple of the increment."""
        # Setup
        increment_value = 1000
        a_values = np.random.randint(low=1, high=10, size=10) * increment_value
        b_values = np.random.randint(low=1, high=100, size=10) * increment_value
        real_data = {
            'table1': pd.DataFrame({
                'A': pd.Series(a_values, dtype='int64'),
                'B': pd.Series(b_values, dtype='Int64'),
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'A': {'sdtype': 'numerical'},
                        'B': {'sdtype': 'numerical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'FixedIncrements',
            'parameters': {
                'column_name': column_name,
                'increment_value': increment_value,
                'table_name': 'table1',
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_fixed_increments_with_a_value_out_of_the_increment(self):
        """Test the score counts a value that is not a multiple of the increment."""
        # Setup
        increment_value = 1000
        a_values = np.random.randint(low=1, high=10, size=10) * increment_value
        b_values = np.random.randint(low=1, high=100, size=10) * increment_value
        real_data = {
            'table1': pd.DataFrame({
                'A': pd.Series(a_values, dtype='int64'),
                'B': pd.Series(b_values, dtype='Int64'),
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['table1'].loc[0, 'A'] = 1500
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'A': {'sdtype': 'numerical'},
                        'B': {'sdtype': 'numerical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'FixedIncrements',
            'parameters': {
                'column_name': 'A',
                'increment_value': increment_value,
                'table_name': 'table1',
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 9 / 10

    def test_inequality(self):
        """Test the score is 1.0 when every low value is below its high value."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'A': [1, 2, 3, 1, 2, 1],
                'B': [10, 20, 30, 10, 20, 10],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        real_data['table1'].loc[0, 'B'] = 0

        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'A': {'sdtype': 'numerical'},
                        'B': {'sdtype': 'numerical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'Inequality',
            'parameters': {
                'low_column_name': 'A',
                'high_column_name': 'B',
                'table_name': 'table1',
            },
        }

        # Run
        warning_message = re.escape(
            "The real data does not adhere to the 'Inequality' constraint "
            '(83.33% of the rows are valid).'
        )

        # Run
        with pytest.warns(UserWarning, match=warning_message):
            score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_inequality_with_an_invalid_row(self):
        """Test the score counts a row whose low value is above its high value."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'A': [1, 2, 3, 1, 2, 1],
                'B': [10, 20, 30, 10, 20, 10],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['table1'].loc[0, 'B'] = 0
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'A': {'sdtype': 'numerical'},
                        'B': {'sdtype': 'numerical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'Inequality',
            'parameters': {
                'low_column_name': 'A',
                'high_column_name': 'B',
                'table_name': 'table1',
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 5 / 6

    def test_inequality_zero_score(self):
        """Test the metric warns and returns zero if none is valid."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'A': [1, 2, 3, 1, 2, 1],
                'B': [10, 20, 30, 10, 20, 10],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['table1']['A'] *= 100
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'A': {'sdtype': 'numerical'},
                        'B': {'sdtype': 'numerical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'Inequality',
            'parameters': {
                'low_column_name': 'A',
                'high_column_name': 'B',
                'table_name': 'table1',
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 0

    def test_one_hot_encoding(self):
        """Test the score is 1.0 when every row has exactly one value set."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'a': [1, 0, 0],
                'b': [0, 1, 0],
                'c': [0, 0, 1],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'a': {'sdtype': 'numerical'},
                        'b': {'sdtype': 'numerical'},
                        'c': {'sdtype': 'numerical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'OneHotEncoding',
            'parameters': {'column_names': ['a', 'b', 'c'], 'table_name': 'table1'},
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_one_hot_encoding_with_two_values_set(self):
        """Test the score counts a row that has more than one value set."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'a': [1, 0, 0],
                'b': [0, 1, 0],
                'c': [0, 0, 1],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['table1'].loc[0, 'b'] = 1
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'a': {'sdtype': 'numerical'},
                        'b': {'sdtype': 'numerical'},
                        'c': {'sdtype': 'numerical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'OneHotEncoding',
            'parameters': {'column_names': ['a', 'b', 'c'], 'table_name': 'table1'},
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 2 / 3

    def test_range(self):
        """Test the score is 1.0 when every middle value is between its bounds."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'A': [1, 2, 3, 1, 2, 1],
                'B': [10, 20, 30, 10, 20, 10],
                'C': [100, 200, 300, 100, 200, 100],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'A': {'sdtype': 'numerical'},
                        'B': {'sdtype': 'numerical'},
                        'C': {'sdtype': 'numerical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'Range',
            'parameters': {
                'low_column_name': 'A',
                'middle_column_name': 'B',
                'high_column_name': 'C',
                'strict_boundaries': True,
                'table_name': 'table1',
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_range_with_a_value_out_of_range(self):
        """Test the score counts a row whose middle value is above its high value."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'A': [1, 2, 3, 1, 2, 1],
                'B': [10, 20, 30, 10, 20, 10],
                'C': [100, 200, 300, 100, 200, 100],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['table1'].loc[0, 'B'] = 1000
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'A': {'sdtype': 'numerical'},
                        'B': {'sdtype': 'numerical'},
                        'C': {'sdtype': 'numerical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'Range',
            'parameters': {
                'low_column_name': 'A',
                'middle_column_name': 'B',
                'high_column_name': 'C',
                'strict_boundaries': True,
                'table_name': 'table1',
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 5 / 6

    def test_unsupported_constraint(self):
        """Test the metric warns and returns NaN if the constraint is not supported."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'A': [1, 2, 3, 1, 2, 1],
                'B': [10, 20, 30, 10, 20, 10],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'A': {'sdtype': 'numerical'},
                        'B': {'sdtype': 'numerical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {'class_name': 'NotAConstraint', 'parameters': {}}
        warning_message = re.escape("Unsupported constraint class 'NotAConstraint'.")

        # Run
        with pytest.warns(UserWarning, match=warning_message):
            score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert pd.isna(score)
