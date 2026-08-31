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

    def test_denormalized_table(self):
        """Test the score is 1.0 when every row of a key repeats the same values."""
        # Setup
        real_data = {
            'orders': pd.DataFrame({
                'order_id': [1, 2, 3, 4],
                'customer_id': [10, 10, 11, 11],
                'customer_name': ['A', 'A', 'B', 'B'],
                'item': ['x', 'y', 'z', 'w'],
            })
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'orders': {
                    'columns': {
                        'order_id': {'sdtype': 'id'},
                        'customer_id': {'sdtype': 'id'},
                        'customer_name': {'sdtype': 'pii'},
                        'item': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'order_id',
                }
            }
        }
        constraint = {
            'class_name': 'DenormalizedTable',
            'parameters': {
                'table_name': 'orders',
                'denormalized_primary_key': 'customer_id',
                'denormalized_column_names': ['customer_name'],
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_denormalized_table_with_an_inconsistent_key(self):
        """Test the score flags every row of a key that does not repeat the same values."""
        # Setup
        real_data = {
            'orders': pd.DataFrame({
                'order_id': [1, 2, 3, 4],
                'customer_id': [10, 10, 11, 11],
                'customer_name': ['A', 'A', 'B', 'B'],
                'item': ['x', 'y', 'z', 'w'],
            })
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['orders'].loc[1, 'customer_name'] = 'Z'
        metadata = {
            'tables': {
                'orders': {
                    'columns': {
                        'order_id': {'sdtype': 'id'},
                        'customer_id': {'sdtype': 'id'},
                        'customer_name': {'sdtype': 'pii'},
                        'item': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'order_id',
                }
            }
        }
        constraint = {
            'class_name': 'DenormalizedTable',
            'parameters': {
                'table_name': 'orders',
                'denormalized_primary_key': 'customer_id',
                'denormalized_column_names': ['customer_name'],
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 0.5

    def test_foreign_to_primary_key_subset(self):
        """Test the score is 1.0 when every guest stays in a hotel of an allowed city."""
        # Setup
        real_data = {
            'hotels': pd.DataFrame({
                'hotel_id': ['HID_001', 'HID_002', 'HID_003', 'HID_004', 'HID_005'],
                'city': ['Boston', 'San Francisco', 'Austin', 'New York City', 'Denver'],
            }),
            'guests': pd.DataFrame({
                'guest_id': range(6),
                'hotel_id': ['HID_001', 'HID_001', 'HID_002', 'HID_004', 'HID_004', 'HID_002'],
            }),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'hotels': {
                    'columns': {'hotel_id': {'sdtype': 'id'}, 'city': {'sdtype': 'categorical'}},
                    'primary_key': 'hotel_id',
                },
                'guests': {
                    'columns': {'guest_id': {'sdtype': 'id'}, 'hotel_id': {'sdtype': 'id'}},
                    'primary_key': 'guest_id',
                },
            }
        }
        constraint = {
            'class_name': 'ForeignToPrimaryKeySubset',
            'parameters': {
                'parent_table_name': 'hotels',
                'child_table_name': 'guests',
                'child_foreign_key': 'hotel_id',
                'conditional_column_name': 'city',
                'conditional_values': ['Boston', 'San Francisco', 'New York City'],
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_foreign_to_primary_key_subset_with_a_guest_out_of_the_subset(self):
        """Test the score counts a guest that stays in a hotel of another city."""
        # Setup
        real_data = {
            'hotels': pd.DataFrame({
                'hotel_id': ['HID_001', 'HID_002', 'HID_003', 'HID_004', 'HID_005'],
                'city': ['Boston', 'San Francisco', 'Austin', 'New York City', 'Denver'],
            }),
            'guests': pd.DataFrame({
                'guest_id': range(6),
                'hotel_id': ['HID_001', 'HID_001', 'HID_002', 'HID_004', 'HID_004', 'HID_002'],
            }),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['guests'].loc[0, 'hotel_id'] = 'HID_003'
        metadata = {
            'tables': {
                'hotels': {
                    'columns': {'hotel_id': {'sdtype': 'id'}, 'city': {'sdtype': 'categorical'}},
                    'primary_key': 'hotel_id',
                },
                'guests': {
                    'columns': {'guest_id': {'sdtype': 'id'}, 'hotel_id': {'sdtype': 'id'}},
                    'primary_key': 'guest_id',
                },
            }
        }
        constraint = {
            'class_name': 'ForeignToPrimaryKeySubset',
            'parameters': {
                'parent_table_name': 'hotels',
                'child_table_name': 'guests',
                'child_foreign_key': 'hotel_id',
                'conditional_column_name': 'city',
                'conditional_values': ['Boston', 'San Francisco', 'New York City'],
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 5 / 6

    def test_mixed_scales(self):
        """Test the score is 1.0 when every value stays in the bounds of its segment."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'col_A': [1.0, 5.0, 9.0, 10.0, np.nan, 1000.000, -1.00],
                'str': ['Test', 'Test', 'Test', 'pro', 'pro', 'pro', 'pro'],
                'int': [1, 1, 1, 2, 2, 3, 3],
                'float': [3.14, 3.14, 3.14, 0.0, 0.0, 0.0, 3.14],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'col_A': {'sdtype': 'numerical'},
                        'str': {'sdtype': 'categorical'},
                        'int': {'sdtype': 'categorical'},
                        'float': {'sdtype': 'categorical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'MixedScales',
            'parameters': {
                'mixed_scale_column_name': 'col_A',
                'segment_column_names': ['str'],
                'table_name': 'table1',
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_mixed_scales_with_a_value_out_of_its_segment(self):
        """Test the score counts a value that is out of the bounds of its segment."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({
                'col_A': [1.0, 5.0, 9.0, 10.0, np.nan, 1000.000, -1.00],
                'str': ['Test', 'Test', 'Test', 'pro', 'pro', 'pro', 'pro'],
                'int': [1, 1, 1, 2, 2, 3, 3],
                'float': [3.14, 3.14, 3.14, 0.0, 0.0, 0.0, 3.14],
            }),
            'table2': pd.DataFrame({'id': range(5)}),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['table1'].loc[0, 'col_A'] = 500.0
        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'col_A': {'sdtype': 'numerical'},
                        'str': {'sdtype': 'categorical'},
                        'int': {'sdtype': 'categorical'},
                        'float': {'sdtype': 'categorical'},
                    }
                },
                'table2': {'columns': {'id': {'sdtype': 'id'}}},
            }
        }
        constraint = {
            'class_name': 'MixedScales',
            'parameters': {
                'mixed_scale_column_name': 'col_A',
                'segment_column_names': ['str'],
                'table_name': 'table1',
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 6 / 7

    def test_polymorphic_relationship(self):
        """Test the score is 1.0 when every guest points at the table of its type."""
        # Setup
        real_data = {
            'hotels': pd.DataFrame({
                'hotel_id': ['HID_001', 'HID_002'],
                'classification': ['BASIC', 'BASIC'],
            }),
            'resorts': pd.DataFrame({
                'hotel_id': ['HID_003', 'HID_004'],
                'classification': ['RESORT', 'RESORT'],
            }),
            'guests': pd.DataFrame({
                'guest_id': range(6),
                'hotel_id': ['HID_001', 'HID_002', 'HID_001', 'HID_003', 'HID_004', 'HID_003'],
                'hotel_type': ['BASIC', 'BASIC', 'BASIC', 'RESORT', 'RESORT', 'RESORT'],
            }),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'hotels': {
                    'columns': {
                        'hotel_id': {'sdtype': 'id'},
                        'classification': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'hotel_id',
                },
                'resorts': {
                    'columns': {
                        'hotel_id': {'sdtype': 'id'},
                        'classification': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'hotel_id',
                },
                'guests': {
                    'columns': {
                        'guest_id': {'sdtype': 'id'},
                        'hotel_id': {'sdtype': 'id'},
                        'hotel_type': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'guest_id',
                },
            },
            'relationships': [],
        }
        constraint = {
            'class_name': 'PolymorphicRelationship',
            'parameters': {
                'table_name': 'guests',
                'foreign_key': 'hotel_id',
                'parent_table_names': ['hotels', 'resorts'],
                'type_column_name': 'hotel_type',
                'type_value_to_table': {'BASIC': 'hotels', 'RESORT': 'resorts'},
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_polymorphic_relationship_pointing_at_the_wrong_table(self):
        """Test the score counts a guest whose type does not match the table it points at."""
        # Setup
        real_data = {
            'hotels': pd.DataFrame({
                'hotel_id': ['HID_001', 'HID_002'],
                'classification': ['BASIC', 'BASIC'],
            }),
            'resorts': pd.DataFrame({
                'hotel_id': ['HID_003', 'HID_004'],
                'classification': ['RESORT', 'RESORT'],
            }),
            'guests': pd.DataFrame({
                'guest_id': range(6),
                'hotel_id': ['HID_001', 'HID_002', 'HID_001', 'HID_003', 'HID_004', 'HID_003'],
                'hotel_type': ['BASIC', 'BASIC', 'BASIC', 'RESORT', 'RESORT', 'RESORT'],
            }),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['guests'].loc[0, 'hotel_type'] = 'RESORT'
        metadata = {
            'tables': {
                'hotels': {
                    'columns': {
                        'hotel_id': {'sdtype': 'id'},
                        'classification': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'hotel_id',
                },
                'resorts': {
                    'columns': {
                        'hotel_id': {'sdtype': 'id'},
                        'classification': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'hotel_id',
                },
                'guests': {
                    'columns': {
                        'guest_id': {'sdtype': 'id'},
                        'hotel_id': {'sdtype': 'id'},
                        'hotel_type': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'guest_id',
                },
            },
            'relationships': [],
        }
        constraint = {
            'class_name': 'PolymorphicRelationship',
            'parameters': {
                'table_name': 'guests',
                'foreign_key': 'hotel_id',
                'parent_table_names': ['hotels', 'resorts'],
                'type_column_name': 'hotel_type',
                'type_value_to_table': {'BASIC': 'hotels', 'RESORT': 'resorts'},
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 5 / 6

    def test_primary_to_primary_key_subset(self):
        """Test the score is 1.0 when every attribute row is allowed to connect."""
        # Setup
        real_data = {
            'main_table': pd.DataFrame({
                'main_pk': [1, 2, 3, 4, 5, 6],
                'conditional_value_column': ['car', 'car', 'plane', 'plane', 'bus', 'bus'],
            }),
            'attr_car': pd.DataFrame({'car_pk': [1, 2], 'height': [600, 700]}),
            'attr_plane': pd.DataFrame({'plane_pk': [3, 4], 'departure city': ['Madrid', 'Paris']}),
            'attr_bus': pd.DataFrame({'bus_pk': [5, 6], 'route': ['A', 'B']}),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'main_table': {
                    'columns': {
                        'main_pk': {'sdtype': 'id'},
                        'conditional_value_column': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'main_pk',
                },
                'attr_car': {
                    'columns': {'car_pk': {'sdtype': 'id'}, 'height': {'sdtype': 'numerical'}},
                    'primary_key': 'car_pk',
                },
                'attr_plane': {
                    'columns': {
                        'plane_pk': {'sdtype': 'id'},
                        'departure city': {'sdtype': 'city'},
                    },
                    'primary_key': 'plane_pk',
                },
                'attr_bus': {
                    'columns': {'bus_pk': {'sdtype': 'id'}, 'route': {'sdtype': 'categorical'}},
                    'primary_key': 'bus_pk',
                },
            }
        }
        constraint = {
            'class_name': 'PrimaryToPrimaryKeySubset',
            'parameters': {
                'main_table_name': 'main_table',
                'conditional_column_name': 'conditional_value_column',
                'relationships': {
                    'attr_car': ['car'],
                    'attr_plane': ['plane'],
                    'attr_bus': ['bus'],
                },
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_primary_to_primary_key_subset_with_a_row_out_of_the_subset(self):
        """Test the score counts an attribute row whose main row has another value."""
        # Setup
        real_data = {
            'main_table': pd.DataFrame({
                'main_pk': [1, 2, 3, 4, 5, 6],
                'conditional_value_column': ['car', 'car', 'plane', 'plane', 'bus', 'bus'],
            }),
            'attr_car': pd.DataFrame({'car_pk': [1, 2], 'height': [600, 700]}),
            'attr_plane': pd.DataFrame({'plane_pk': [3, 4], 'departure city': ['Madrid', 'Paris']}),
            'attr_bus': pd.DataFrame({'bus_pk': [5, 6], 'route': ['A', 'B']}),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['attr_car'] = pd.DataFrame({'car_pk': [1, 3], 'height': [600, 700]})
        metadata = {
            'tables': {
                'main_table': {
                    'columns': {
                        'main_pk': {'sdtype': 'id'},
                        'conditional_value_column': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'main_pk',
                },
                'attr_car': {
                    'columns': {'car_pk': {'sdtype': 'id'}, 'height': {'sdtype': 'numerical'}},
                    'primary_key': 'car_pk',
                },
                'attr_plane': {
                    'columns': {
                        'plane_pk': {'sdtype': 'id'},
                        'departure city': {'sdtype': 'city'},
                    },
                    'primary_key': 'plane_pk',
                },
                'attr_bus': {
                    'columns': {'bus_pk': {'sdtype': 'id'}, 'route': {'sdtype': 'categorical'}},
                    'primary_key': 'bus_pk',
                },
            }
        }
        constraint = {
            'class_name': 'PrimaryToPrimaryKeySubset',
            'parameters': {
                'main_table_name': 'main_table',
                'conditional_column_name': 'conditional_value_column',
                'relationships': {
                    'attr_car': ['car'],
                    'attr_plane': ['plane'],
                    'attr_bus': ['bus'],
                },
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 5 / 6

    def test_self_referential_hierarchy(self):
        """Test the score is 1.0 when every employee reports up to a root."""
        # Setup
        real_data = {
            'employees': pd.DataFrame({
                'employee_id': [f'employee_{i}' for i in range(6)],
                'supervisor_id': [
                    None,
                    'employee_0',
                    'employee_0',
                    'employee_1',
                    'employee_1',
                    'employee_2',
                ],
                'department': ['sales', 'eng', 'sales', 'eng', 'sales', 'eng'],
            }),
            'departments': pd.DataFrame({
                'department_id': ['sales', 'eng'],
                'city': ['New York', 'Austin'],
            }),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'employees': {
                    'columns': {
                        'employee_id': {'sdtype': 'id'},
                        'supervisor_id': {'sdtype': 'id'},
                        'department': {'sdtype': 'id'},
                    },
                    'primary_key': 'employee_id',
                },
                'departments': {
                    'columns': {
                        'department_id': {'sdtype': 'id'},
                        'city': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'department_id',
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'departments',
                    'child_table_name': 'employees',
                    'parent_primary_key': 'department_id',
                    'child_foreign_key': 'department',
                }
            ],
        }
        constraint = {
            'class_name': 'SelfReferentialHierarchy',
            'parameters': {
                'table_name': 'employees',
                'base_column_name': 'employee_id',
                'parent_column_name': 'supervisor_id',
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_self_referential_hierarchy_with_a_cycle(self):
        """Test the score counts every employee that reports into a cycle."""
        # Setup
        real_data = {
            'employees': pd.DataFrame({
                'employee_id': [f'employee_{i}' for i in range(6)],
                'supervisor_id': [
                    None,
                    'employee_0',
                    'employee_0',
                    'employee_1',
                    'employee_1',
                    'employee_2',
                ],
                'department': ['sales', 'eng', 'sales', 'eng', 'sales', 'eng'],
            }),
            'departments': pd.DataFrame({
                'department_id': ['sales', 'eng'],
                'city': ['New York', 'Austin'],
            }),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['employees']['supervisor_id'] = [
            None,
            'employee_0',
            'employee_0',
            'employee_4',
            'employee_3',
            'employee_2',
        ]
        metadata = {
            'tables': {
                'employees': {
                    'columns': {
                        'employee_id': {'sdtype': 'id'},
                        'supervisor_id': {'sdtype': 'id'},
                        'department': {'sdtype': 'id'},
                    },
                    'primary_key': 'employee_id',
                },
                'departments': {
                    'columns': {
                        'department_id': {'sdtype': 'id'},
                        'city': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'department_id',
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'departments',
                    'child_table_name': 'employees',
                    'parent_primary_key': 'department_id',
                    'child_foreign_key': 'department',
                }
            ],
        }
        constraint = {
            'class_name': 'SelfReferentialHierarchy',
            'parameters': {
                'table_name': 'employees',
                'base_column_name': 'employee_id',
                'parent_column_name': 'supervisor_id',
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 4 / 6

    def test_carry_over_columns(self):
        """Test the score is 1.0 when every key carries the same value everywhere."""
        # Setup
        real_data = {
            'main_table': pd.DataFrame({
                'primary_key': [1, 2, 3],
                'parent_1': ['a', 'b', 'c'],
            }),
            'carry_over_1': pd.DataFrame({
                'child_1': ['a', 'a', 'c'],
                'child_2': ['b', 'b', 'c'],
                'key_column_1': [1, 1, 3],
                'key_column_2': [2, 2, 3],
            }),
            'carry_over_2': pd.DataFrame({
                'child_3': ['a', 'b', 'c'],
                'foreign_key': [1, 2, 3],
            }),
        }
        synthetic_data = deepcopy(real_data)
        metadata = {
            'tables': {
                'main_table': {
                    'columns': {
                        'primary_key': {'sdtype': 'id'},
                        'parent_1': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'primary_key',
                },
                'carry_over_1': {
                    'columns': {
                        'child_1': {'sdtype': 'categorical'},
                        'child_2': {'sdtype': 'categorical'},
                        'key_column_1': {'sdtype': 'id'},
                        'key_column_2': {'sdtype': 'id'},
                    }
                },
                'carry_over_2': {
                    'columns': {
                        'child_3': {'sdtype': 'categorical'},
                        'foreign_key': {'sdtype': 'id'},
                    }
                },
            }
        }
        constraint = {
            'class_name': 'CarryOverColumns',
            'parameters': {
                'common_column_info': [
                    {
                        'table_name': 'main_table',
                        'key_column_name': 'primary_key',
                        'carryover_column_name': 'parent_1',
                    },
                    {
                        'table_name': 'carry_over_1',
                        'key_column_name': 'key_column_1',
                        'carryover_column_name': 'child_1',
                    },
                    {
                        'table_name': 'carry_over_1',
                        'key_column_name': 'key_column_2',
                        'carryover_column_name': 'child_2',
                    },
                    {
                        'table_name': 'carry_over_2',
                        'key_column_name': 'foreign_key',
                        'carryover_column_name': 'child_3',
                    },
                ]
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_carry_over_columns_with_a_value_that_does_not_match(self):
        """Test the score counts a row whose carried over value does not match."""
        # Setup
        real_data = {
            'main_table': pd.DataFrame({
                'primary_key': [1, 2, 3],
                'parent_1': ['a', 'b', 'c'],
            }),
            'carry_over_1': pd.DataFrame({
                'child_1': ['a', 'a', 'c'],
                'child_2': ['b', 'b', 'c'],
                'key_column_1': [1, 1, 3],
                'key_column_2': [2, 2, 3],
            }),
            'carry_over_2': pd.DataFrame({
                'child_3': ['a', 'b', 'c'],
                'foreign_key': [1, 2, 3],
            }),
        }
        synthetic_data = deepcopy(real_data)
        synthetic_data['carry_over_2'].loc[0, 'child_3'] = 'z'
        metadata = {
            'tables': {
                'main_table': {
                    'columns': {
                        'primary_key': {'sdtype': 'id'},
                        'parent_1': {'sdtype': 'categorical'},
                    },
                    'primary_key': 'primary_key',
                },
                'carry_over_1': {
                    'columns': {
                        'child_1': {'sdtype': 'categorical'},
                        'child_2': {'sdtype': 'categorical'},
                        'key_column_1': {'sdtype': 'id'},
                        'key_column_2': {'sdtype': 'id'},
                    }
                },
                'carry_over_2': {
                    'columns': {
                        'child_3': {'sdtype': 'categorical'},
                        'foreign_key': {'sdtype': 'id'},
                    }
                },
            }
        }
        constraint = {
            'class_name': 'CarryOverColumns',
            'parameters': {
                'common_column_info': [
                    {
                        'table_name': 'main_table',
                        'key_column_name': 'primary_key',
                        'carryover_column_name': 'parent_1',
                    },
                    {
                        'table_name': 'carry_over_1',
                        'key_column_name': 'key_column_1',
                        'carryover_column_name': 'child_1',
                    },
                    {
                        'table_name': 'carry_over_1',
                        'key_column_name': 'key_column_2',
                        'carryover_column_name': 'child_2',
                    },
                    {
                        'table_name': 'carry_over_2',
                        'key_column_name': 'foreign_key',
                        'carryover_column_name': 'child_3',
                    },
                ]
            },
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 8 / 9
