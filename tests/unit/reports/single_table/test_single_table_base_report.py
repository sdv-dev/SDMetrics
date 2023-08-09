import pickle
import re
import sys
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

    def test_generate(self):
        """Test the ``generate`` method.

        This test checks that the method calls the ``validate`` method and the ``get_score``
        method for each property.
        """
        # Setup
        base_report = BaseReport()
        mock_validate = Mock()
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
                'column1': {'sdtypes': 'numerical'},
                'column2': {'sdtypes': 'categorical'}
            }
        }

        # Run
        base_report.generate(real_data, synthetic_data, metadata, verbose=False)

        # Assert
        mock_validate.assert_called_once_with(real_data, synthetic_data, metadata)
        base_report._properties['Property 1'].get_score.assert_called_with(
            real_data, synthetic_data, metadata, progress_bar=None
        )
        base_report._properties['Property 2'].get_score.assert_called_with(
            real_data, synthetic_data, metadata, progress_bar=None
        )

    def test__print_result(self):
        """Test the ``_print_result`` method."""
        # Setup
        base_report = BaseReport()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            base_report._print_results()

    @patch('tqdm.tqdm')
    def test_generate_verbose(self, mock_tqdm):
        """Test the ``generate`` method with verbose=True."""
        # Setup
        base_report = BaseReport()
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
                'column1': {'sdtypes': 'numerical'},
                'column2': {'sdtypes': 'categorical'},
                'column3': {'sdtypes': 'numerical'},
                'column4': {'sdtypes': 'numerical'},
            }
        }

        # Run
        base_report.generate(real_data, synthetic_data, metadata, verbose=True)

        # Assert
        calls = [call(total=4, file=sys.stdout), call(total=6, file=sys.stdout)]
        mock_tqdm.assert_has_calls(calls, any_order=True)
        base_report._print_results.assert_called_once()

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
        base_report._properties['Property 1']._details.copy.assert_called_once()
        base_report._properties['Property 2']._details.copy.assert_called_once()

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
