import pandas as pd
from sdmetrics.single_table.privacy.dcr_overfitting_protection import DCROverfittingProtection


def test():
    train_data = pd.DataFrame({
        'num_col_1': [1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
        'num_col_2': [8.0, 3.0, 8.0, 8.0, 3.0, 8.0]
    })
    validation_data = pd.DataFrame({
        'num_col_1': [1.0, 3.0, 5.0, 7.0, 10.0, 9.0],
        'num_col_2': [4.0, 6.0, 2.0, 9.0, 1.0, 2.0]
    })
    synthetic_data = pd.DataFrame({
        'num_col_1': [1.0, 10.0, 1.0, 10.0, 10.0, 10.0],
        'num_col_2': [4.0, 3.0, 3.0, 8.0, 3.0, 8.0]
    })
    metadata = {
        'columns': {
            'num_col_1': {
                'sdtype': 'numerical',
            },
            'num_col_2': {
                'sdtype': 'numerical',
            },
        }
    }

    DCROverfittingProtection.compute_breakdown(
        train_data,
        synthetic_data,
        validation_data,
        metadata,
        num_rows_subsample=3,
        num_iterations=5
    )

    assert False
