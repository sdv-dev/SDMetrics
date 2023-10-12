import pickle
import re
import sys
from datetime import datetime
from unittest.mock import Mock, call, mock_open, patch

import pandas as pd
import pytest

from sdmetrics.reports.base_report import BaseReport


class TestBaseReport:
    def test__validate_metadata_matches_data(self):
        """Test the ``_validate_metadata_matches_data`` method.

        This test checks that the method raises an error when there is a column
        mismatch between the data and the metadata.
        At the first call, there is a mismatch, not in the second call.
        """
        # Setup
        base_report = BaseReport()
        real_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c'],
            'column3': [4, 5, 6]
        })
        synthetic_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c'],
            'column4': [4, 5, 6]
        })
        metadata = {
            'columns': {
                'column1': {'sdtype': 'numerical'},
                'column5': {'sdtype': 'categorical'},
            }
        }

        # Run and Assert
        expected_err_message = re.escape(
            'The metadata does not match the data. The following columns are missing'
            ' in the real/synthetic data or in the metadata: column2, column3, column4, column5'
        )
        with pytest.raises(ValueError, match=expected_err_message):
            base_report._validate_metadata_matches_data(real_data, synthetic_data, metadata)

        real_data['column4'] = [1, 2, 3]
        real_data['column5'] = ['a', 'b', 'c']
        synthetic_data['column3'] = [1, 2, 3]
        synthetic_data['column5'] = ['a', 'b', 'c']

        metadata['columns']['column2'] = {'sdtype': 'categorical'}
        metadata['columns']['column3'] = {'sdtype': 'numerical'}
        metadata['columns']['column4'] = {'sdtype': 'numerical'}

        base_report._validate_metadata_matches_data(real_data, synthetic_data, metadata)

    def test__validate_metadata_matches_data_no_mismatch(self):
        """Test the ``_validate_metadata_matches_data`` method.

        This test checks that the method does not raise an error when there is no column mismatch
        between the data and the metadata.
        """
        # Setup
        base_report = BaseReport()
        real_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c'],
            'column3': [4, 5, 6]
        })
        synthetic_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c'],
            'column3': [4, 5, 6]
        })
        metadata = {
            'columns': {
                'column1': {'sdtype': 'numerical'},
                'column2': {'sdtype': 'categorical'},
                'column3': {'sdtype': 'numerical'},
            }
        }

        # Run and Assert
        base_report._validate_metadata_matches_data(real_data, synthetic_data, metadata)

    def test_validate(self):
        """Test the ``validate`` method."""
        # Setup
        base_report = BaseReport()
        mock__validate_metadata_matches_data = Mock()
        base_report._validate_metadata_matches_data = mock__validate_metadata_matches_data

        real_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c'],
            'column3': [4, 5, 6]
        })
        synthetic_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c'],
            'column4': [4, 5, 6]
        })
        metadata = {
            'columns': {
                'column1': {'sdtype': 'numerical'},
                'column2': {'sdtype': 'categorical'},
            }
        }

        # Run
        base_report.validate(real_data, synthetic_data, metadata)

        # Assert
        mock__validate_metadata_matches_data.assert_called_once_with(
            real_data, synthetic_data, metadata
        )

    def test_convert_datetimes(self):
        """Test that ``_convert_datetimes`` tries to convert datetime columns."""
        # Setup
        base_report = BaseReport()
        real_data = pd.DataFrame({
            'col1': ['2020-01-02', '2021-01-02'],
            'col2': ['a', 'b']
        })
        synthetic_data = pd.DataFrame({
            'col1': ['2022-01-03', '2023-04-05'],
            'col2': ['b', 'a']
        })
        metadata = {
            'columns': {
                'col1': {'sdtype': 'datetime'},
                'col2': {'sdtype': 'datetime'}
            },
        }

        # Run
        base_report.convert_datetimes(real_data, synthetic_data, metadata)

        # Assert
        expected_real_data = pd.DataFrame({
            'col1': [datetime(2020, 1, 2), datetime(2021, 1, 2)],
            'col2': ['a', 'b']
        })
        expected_synthetic_data = pd.DataFrame({
            'col1': [datetime(2022, 1, 3), datetime(2023, 4, 5)],
            'col2': ['b', 'a']
        })

        pd.testing.assert_frame_equal(real_data, expected_real_data)
        pd.testing.assert_frame_equal(synthetic_data, expected_synthetic_data)

    def test__convert_metadata_with_to_dict_method(self):
        """Test ``_convert_metadata`` when the metadata object has a ``to_dict`` method."""
        # Setup
        metadata_example = {
            'column1': {'sdtype': 'numerical'},
            'column2': {'sdtype': 'categorical'},
        }

        class Metadata:
            def __init__(self):
                self.columns = metadata_example

            def to_dict(self):
                return self.columns

        metadata = Metadata()

        # Run
        converted_metadata = BaseReport._convert_metadata(metadata)

        # Assert
        assert converted_metadata == metadata_example

    def test__convert_metadata_without_to_dict_method(self):
        """Test ``_convert_metadata`` when the metadata object has no ``to_dict`` method."""
        # Setup
        metadata_example = {
            'column1': {'sdtype': 'numerical'},
            'column2': {'sdtype': 'categorical'},
        }

        class Metadata:
            def __init__(self):
                self.columns = metadata_example

        metadata = Metadata()

        # Run and Assert
        expected_message = re.escape(
            'The provided metadata is not a dictionary and does not have a to_dict method.'
            'Please convert the metadata to a dictionary.'
        )
        with pytest.raises(TypeError, match=expected_message):
            BaseReport._convert_metadata(metadata)

        result = BaseReport._convert_metadata(metadata_example)
        assert result == metadata_example

    @patch('sdmetrics.reports.base_report.BaseReport._convert_metadata')
    def test_generate(self, mock__convert_metadata):
        """Test the ``generate`` method.

        This test checks that the method calls the ``validate`` method and the ``get_score``
        method for each property.
        """
        # Setup
        base_report = BaseReport()
        mock_validate = Mock()
        mock_handle_results = Mock()
        base_report._handle_results = mock_handle_results
        base_report.validate = mock_validate
        base_report._properties['Property 1'] = Mock()
        base_report._properties['Property 1'].get_score.return_value = 1.0
        base_report._properties['Property 2'] = Mock()
        base_report._properties['Property 2'].get_score.return_value = 1.0

        real_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        synthetic_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        metadata = {
            'columns': {
                'column1': {'sdtype': 'numerical'},
                'column2': {'sdtype': 'categorical'}
            }
        }
        mock__convert_metadata.return_value = metadata

        # Run
        base_report.generate(real_data, synthetic_data, metadata, verbose=False)

        # Assert
        mock__convert_metadata.assert_called_once_with(metadata)
        mock_validate.assert_called_once_with(real_data, synthetic_data, metadata)
        mock_handle_results.assert_called_once_with(False)
        base_report._properties['Property 1'].get_score.assert_called_with(
            real_data, synthetic_data, metadata, progress_bar=None
        )
        base_report._properties['Property 2'].get_score.assert_called_with(
            real_data, synthetic_data, metadata, progress_bar=None
        )

    def test__handle_results(self):
        """Test the ``_handle_results`` method."""
        # Setup
        base_report = BaseReport()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            base_report._handle_results(True)

    @patch('tqdm.tqdm')
    def test_generate_verbose(self, mock_tqdm):
        """Test the ``generate`` method with verbose=True."""
        # Setup
        base_report = BaseReport()
        base_report._handle_results = Mock()
        mock_validate = Mock()
        base_report.validate = mock_validate
        base_report._print_results = Mock()
        base_report._properties['Property 1'] = Mock()
        base_report._properties['Property 1'].get_score.return_value = 1.0
        base_report._properties['Property 1']._get_num_iterations.return_value = 4
        base_report._properties['Property 2'] = Mock()
        base_report._properties['Property 2'].get_score.return_value = 1.0
        base_report._properties['Property 2']._get_num_iterations.return_value = 6
        base_report._properties['Property 1']._compute_average.return_value = 1.0
        base_report._properties['Property 2']._compute_average.return_value = 1.0

        real_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c'],
            'column3': [4, 5, 6],
            'column4': [7, 8, 9],
        })
        synthetic_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c'],
            'column3': [4, 5, 6],
            'column4': [7, 8, 9],
        })
        metadata = {
            'columns': {
                'column1': {'sdtype': 'numerical'},
                'column2': {'sdtype': 'categorical'},
                'column3': {'sdtype': 'numerical'},
                'column4': {'sdtype': 'numerical'},
            }
        }

        # Run
        base_report.generate(real_data, synthetic_data, metadata, verbose=True)

        # Assert
        calls = [call(total=4, file=sys.stdout), call(total=6, file=sys.stdout)]
        mock_tqdm.assert_has_calls(calls, any_order=True)
        base_report._handle_results.assert_called_once_with(True)

    def test__check_report_generated(self):
        """Test the ``check_report_generated`` method."""
        # Setup
        base_report = BaseReport()
        base_report.is_generated = False

        # Run and Assert
        expected_message = (
            'The report has not been generated. Please call `generate` first.'
        )
        with pytest.raises(ValueError, match=expected_message):
            base_report._check_report_generated()

        base_report.is_generated = True
        base_report._check_report_generated()

    def test__validate_property_generated(self):
        """Test the ``_validate_property_generated`` method."""
        # Setup
        base_report = BaseReport()
        base_report._properties['Valid Property Name'] = Mock()
        wrong_property_name = 'Wrong Property Name'
        base_report.is_generated = False

        # Run and Assert
        expected_message_1 = (
            'The report has not been generated. Please call `generate` first.'
        )
        with pytest.raises(ValueError, match=expected_message_1):
            base_report._validate_property_generated('Valid Property Name')

        base_report.is_generated = True
        expected_message_2 = (
            "Invalid property name 'Wrong Property Name'. Valid property names"
            " are 'Valid Property Name'."
        )
        with pytest.raises(ValueError, match=expected_message_2):
            base_report._validate_property_generated(wrong_property_name)

    def test_get_properties(self):
        """Test the ``get_details`` method."""
        # Setup
        base_report = BaseReport()
        mock_cs_compute_average = Mock(return_value=1.0)
        mock_cpt_compute_averag = Mock(return_value=1.0)
        base_report._properties['Property 1'] = Mock()
        base_report._properties['Property 1']._compute_average = mock_cs_compute_average
        base_report._properties['Property 2'] = Mock()
        base_report._properties['Property 2']._compute_average = mock_cpt_compute_averag
        base_report.is_generated = True

        # Run
        properties = base_report.get_properties()

        # Assert
        pd.testing.assert_frame_equal(
            properties,
            pd.DataFrame({
                'Property': ['Property 1', 'Property 2'],
                'Score': [1.0, 1.0],
            }),
        )

    def test_get_visualization(self):
        """Test the ``get_visualization`` method."""
        # Setup
        base_report = BaseReport()
        base_report._properties['Property 1'] = Mock()
        base_report._properties['Property 2'] = Mock()
        base_report.is_generated = True

        # Run
        base_report.get_visualization('Property 1')
        base_report.get_visualization('Property 2')

        # Assert
        base_report._properties['Property 1'].get_visualization.assert_called_once()
        base_report._properties['Property 2'].get_visualization.assert_called_once()

    def test_get_details(self):
        """Test the ``get_details`` method."""
        # Setup
        base_report = BaseReport()
        mock_validate_property_generated = Mock()
        base_report._validate_property_generated = mock_validate_property_generated
        base_report._properties['Property 1'] = Mock()
        base_report._properties['Property 2'] = Mock()
        base_report.is_generated = True

        # Run
        base_report.get_details('Property 1')
        base_report.get_details('Property 2')

        # Assert
        mock_validate_property_generated.assert_has_calls([
            call('Property 1'), call('Property 2')
        ])
        base_report._properties['Property 1'].details.copy.assert_called_once()
        base_report._properties['Property 2'].details.copy.assert_called_once()

    @patch('sdmetrics.reports.base_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.base_report.pickle')
    def test_save(self, pickle_mock, get_distribution_mock):
        """Test the ``save`` method.

        Expect that the instance is passed to pickle.

        Input:
        - filepath

        Side Effects:
        - ``pickle`` is called with the instance.
        """
        # Setup
        report = Mock()
        open_mock = mock_open(read_data=pickle.dumps('test'))

        # Run
        with patch('sdmetrics.reports.base_report.open', open_mock):
            BaseReport.save(report, 'test-file.pkl')

        # Assert
        get_distribution_mock.assert_called_once_with('sdmetrics')
        open_mock.assert_called_once_with('test-file.pkl', 'wb')
        pickle_mock.dump.assert_called_once_with(report, open_mock())
        assert report._package_version == get_distribution_mock.return_value.version

    @patch('sdmetrics.reports.base_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.base_report.pickle')
    def test_load(self, pickle_mock, get_distribution_mock):
        """Test the ``load`` method.

        Expect that the report's load method is called with the expected args.

        Input:
        - filepath

        Output:
        - the loaded model

        Side Effects:
        - Expect that ``pickle`` is called with the filepath.
        """
        # Setup
        open_mock = mock_open(read_data=pickle.dumps('test'))
        report = Mock()
        pickle_mock.load.return_value = report
        report._package_version = get_distribution_mock.return_value.version

        # Run
        with patch('sdmetrics.reports.base_report.open', open_mock):
            loaded = BaseReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        assert loaded == pickle_mock.load.return_value

    @patch('sdmetrics.reports.base_report.warnings')
    @patch('sdmetrics.reports.base_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.base_report.pickle')
    def test_load_mismatched_versions(self, pickle_mock, get_distribution_mock, warnings_mock):
        """Test the ``load`` method with mismatched sdmetrics versions.

        Expect that the report's load method is called with the expected args.

        Input:
        - filepath

        Output:
        - the loaded model

        Side Effects:
        - Expect that ``pickle`` is called with the filepath.
        """
        # Setup
        open_mock = mock_open(read_data=pickle.dumps('test'))
        report = Mock()
        pickle_mock.load.return_value = report
        report._package_version = 'previous_version'
        get_distribution_mock.return_value.version = 'new_version'

        # Run
        with patch('sdmetrics.reports.base_report.open', open_mock):
            loaded = BaseReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        warnings_mock.warn.assert_called_once_with(
            'The report was created using SDMetrics version `previous_version` but you are '
            'currently using version `new_version`. Some features may not work as intended.'
        )
        assert loaded == pickle_mock.load.return_value
