from unittest.mock import patch

import pandas as pd

from sdmetrics.single_table import LogisticDetection
from tests.utils import DataFrameMatcher


class TestDetectionMetric:

    @patch('sdmetrics.utils.HyperTransformer.transform')
    @patch('sdmetrics.utils.HyperTransformer.fit_transform')
    def test_primary_key_detection_metrics(self, fit_transform_mock, transform_mock):
        """This test checks that ``primary_key`` columns of dataset are ignored.

        Ensure that ``primary_keys`` are ignored for Detection metrics expect that the match
        is made correctly.
        """

        # Setup
        real_data = pd.DataFrame({
            'ID_1': [1, 2, 1, 3, 4],
            'col2': [1.0, 2.0, 3.0, 4.0, 5.0],
            'ID_2': ['aa', 'bb', 'cc', 'dd', 'bb'],
            'col4': [5, 6, 7, 8, 9]
        })
        synthetic_data = pd.DataFrame({
            'ID_1': [1, 3, 4, 2, 2],
            'col2': [11.0, 22.0, 33.0, 44.0, 55.0],
            'ID_2': ['aa', 'bb', 'cc', 'dd', 'ee'],
            'col4': [55, 66, 77, 88, 99]
        })
        metadata = {
            'columns': {
                'ID_1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'numerical'},
                'ID_2': {'sdtype': 'categorical'},
                'col4': {'sdtype': 'numerical'}
            },
            'primary_key': {'ID_1', 'ID_2'}
        }

        expected_real_dataframe = pd.DataFrame({
            'col2': [1.0, 2.0, 3.0, 4.0, 5.0],
            'col4': [5, 6, 7, 8, 9]
        })
        expected_synthetic_dataframe = pd.DataFrame({
            'col2': [11.0, 22.0, 33.0, 44.0, 55.0],
            'col4': [55, 66, 77, 88, 99]
        })
        expected_return_real = DataFrameMatcher(expected_real_dataframe)
        expected_return_synthetic = DataFrameMatcher(expected_synthetic_dataframe)
        fit_transform_mock.return_value = expected_real_dataframe
        transform_mock.return_value = expected_synthetic_dataframe

        # Run
        LogisticDetection().compute(real_data, synthetic_data, metadata)

        # Assert

        # check that ``fit_transform`` and  ``transform`` received the good argument.
        call_1 = pd.DataFrame(fit_transform_mock.call_args_list[0][0][0])
        call_2 = pd.DataFrame(transform_mock.call_args_list[0][0][0])

        transform_mock.assert_called_with(expected_return_synthetic)
        assert expected_return_real == call_1
        assert expected_return_synthetic == call_2
