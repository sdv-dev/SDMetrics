import pickle
import re
from unittest.mock import Mock, call, mock_open, patch

import pandas as pd
import pytest

from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes


class TestQualityReport:

    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        report = QualityReport()

        # Assert
        assert report._overall_quality_score is None
        assert not report.is_generated
        assert isinstance(report._properties['Column Shapes'], ColumnShapes)
        assert isinstance(report._properties['Column Pair Trends'], ColumnPairTrends)

    @patch('sys.stdout.write')
    def test__print_results(self, mock_write):
        """Test the ``_print_results`` method."""
        # Setup
        quality_report = QualityReport()
        quality_report._overall_quality_score = 0.5
        quality_report._properties = {
            'Column Shapes': Mock(_compute_average=Mock(return_value=0.6)),
            'Column Pair Trends': Mock(_compute_average=Mock(return_value=0.4))
        }

        # Run
        quality_report._print_results()

        # Assert
        calls = [
            call('\nOverall Quality Score: 50.0%\n\n'),
            call('Properties:\n'),
            call('- Column Shapes: 60.0%\n'),
            call('- Column Pair Trends: 40.0%\n'),
        ]
        mock_write.assert_has_calls(calls, any_order=True)

    def test__validate_metadata_matches_data(self):
        """Test the ``_validate_metadata_matches_data`` method.

        This test checks that the method raises an error when there is a column
        mismatch between the data and the metadata.
        At the first call, there is a mismatch, not in the second call.
        """
        # Setup
        quality_report = QualityReport()
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
            quality_report._validate_metadata_matches_data(real_data, synthetic_data, metadata)

        real_data['column4'] = [1, 2, 3]
        real_data['column5'] = ['a', 'b', 'c']
        synthetic_data['column3'] = [1, 2, 3]
        synthetic_data['column5'] = ['a', 'b', 'c']

        metadata['columns']['column2'] = {'sdtype': 'categorical'}
        metadata['columns']['column3'] = {'sdtype': 'numerical'}
        metadata['columns']['column4'] = {'sdtype': 'numerical'}

        quality_report._validate_metadata_matches_data(real_data, synthetic_data, metadata)

    def test__validate_metadata_matches_data_no_mismatch(self):
        """Test the ``_validate_metadata_matches_data`` method.

        This test checks that the method does not raise an error when there is no column mismatch
        between the data and the metadata
        """
        # Setup
        quality_report = QualityReport()
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
        quality_report._validate_metadata_matches_data(real_data, synthetic_data, metadata)

    @patch('sdmetrics.reports.single_table.quality_report._validate_categorical_values')
    def test_validate(self, mock_validate_categorical_values):
        # Setup
        quality_report = QualityReport()
        mock__validate_metadata_matches_data = Mock()
        quality_report._validate_metadata_matches_data = mock__validate_metadata_matches_data

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
        quality_report.validate(real_data, synthetic_data, metadata)

        # Assert
        mock__validate_metadata_matches_data.assert_called_once_with(
            real_data, synthetic_data, metadata
        )
        mock_validate_categorical_values.assert_called_once_with(
            real_data, synthetic_data, metadata
        )

    def test_generate(self):
        """Test the ``generate`` method."""
        # Setup
        quality_report = QualityReport()
        mock_validate = Mock()
        quality_report.validate = mock_validate
        mock_columnshape_get_score = Mock(return_value=1.0)
        mock_cpt_get_score = Mock(return_value=1.0)
        quality_report._properties['Column Shapes'] = Mock()
        quality_report._properties['Column Shapes'].get_score = mock_columnshape_get_score
        quality_report._properties['Column Pair Trends'] = Mock()
        quality_report._properties['Column Pair Trends'].get_score = mock_cpt_get_score

        real_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        synthetic_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        metadata = {
            'column': {
                'column1': {'sdtypes': 'numerical'},
                'column2': {'sdtypes': 'categorical'}
            }
        }

        # Run
        quality_report.generate(real_data, synthetic_data, metadata, verbose=False)

        # Assert
        mock_validate.assert_called_once_with(real_data, synthetic_data, metadata)
        quality_report._properties['Column Shapes'].get_score.assert_called_with(
            real_data, synthetic_data, metadata
        )
        quality_report._properties['Column Pair Trends'].get_score.assert_called_with(
            real_data, synthetic_data, metadata
        )

    def test__validate_property_generated(self):
        """Test the ``_validate_property_generated`` method."""
        # Setup
        quality_report = QualityReport()
        wrong_property_name = 'Wrong Property Name'
        quality_report.is_generated = False

        # Run and Assert
        expected_message_1 = (
            'Quality report must be generated before ''getting details. Call `generate` first.'
        )
        with pytest.raises(ValueError, match=expected_message_1):
            quality_report._validate_property_generated('Column Shapes')

        quality_report.is_generated = True
        expected_message_2 = (
            "Invalid property name 'Wrong Property Name'. Valid property names"
            " are 'Column Shapes' and 'Column Pair Trends'."
        )
        with pytest.raises(ValueError, match=expected_message_2):
            quality_report._validate_property_generated(wrong_property_name)

    def test_get_score(self):
        """Test the ``get_score`` method."""
        # Setup
        report = QualityReport()
        mock_score = Mock()
        report._overall_quality_score = mock_score

        # Run
        score = report.get_score()

        # Assert
        assert score == mock_score

    def test_get_properties(self):
        """Test the ``get_details`` method."""
        # Setup
        quality_report = QualityReport()
        mock_cs_compute_average = Mock(return_value=1.0)
        mock_cpt_compute_averag = Mock(return_value=1.0)
        quality_report._properties['Column Shapes'] = Mock()
        quality_report._properties['Column Shapes']._compute_average = mock_cs_compute_average
        quality_report._properties['Column Pair Trends'] = Mock()
        quality_report._properties['Column Pair Trends']._compute_average = mock_cpt_compute_averag

        # Run
        properties = quality_report.get_properties()

        # Assert
        pd.testing.assert_frame_equal(
            properties,
            pd.DataFrame({
                'Property': ['Column Shapes', 'Column Pair Trends'],
                'Score': [1.0, 1.0],
            }),
        )

    def test_get_visualization(self):
        """Test the ``get_visualization`` method."""
        # Setup
        quality_report = QualityReport()
        quality_report._properties['Column Shapes'] = Mock()
        quality_report._properties['Column Pair Trends'] = Mock()
        quality_report.is_generated = True

        # Run
        quality_report.get_visualization('Column Shapes')
        quality_report.get_visualization('Column Pair Trends')

        # Assert
        quality_report._properties['Column Shapes'].get_visualization.assert_called_once()
        quality_report._properties['Column Pair Trends'].get_visualization.assert_called_once()

    def test_get_details(self):
        """Test the ``get_details`` method."""
        # Setup
        quality_report = QualityReport()
        mock_validate_property_generated = Mock()
        quality_report._validate_property_generated = mock_validate_property_generated
        quality_report._properties['Column Shapes'] = Mock()
        quality_report._properties['Column Pair Trends'] = Mock()
        quality_report.is_generated = True

        # Run
        quality_report.get_details('Column Shapes')
        quality_report.get_details('Column Pair Trends')

        # Assert
        mock_validate_property_generated.assert_has_calls([
            call('Column Shapes'), call('Column Pair Trends')
        ])
        quality_report._properties['Column Shapes']._details.copy.assert_called_once()
        quality_report._properties['Column Pair Trends']._details.copy.assert_called_once()

    @patch('sdmetrics.reports.single_table.quality_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.single_table.quality_report.pickle')
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
        with patch('sdmetrics.reports.single_table.quality_report.open', open_mock):
            QualityReport.save(report, 'test-file.pkl')

        # Assert
        get_distribution_mock.assert_called_once_with('sdmetrics')
        open_mock.assert_called_once_with('test-file.pkl', 'wb')
        pickle_mock.dump.assert_called_once_with(report, open_mock())
        assert report._package_version == get_distribution_mock.return_value.version

    @patch('sdmetrics.reports.single_table.quality_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.single_table.quality_report.pickle')
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
        with patch('sdmetrics.reports.single_table.quality_report.open', open_mock):
            loaded = QualityReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        assert loaded == pickle_mock.load.return_value

    @patch('sdmetrics.reports.single_table.quality_report.warnings')
    @patch('sdmetrics.reports.single_table.quality_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.single_table.quality_report.pickle')
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
        with patch('sdmetrics.reports.single_table.quality_report.open', open_mock):
            loaded = QualityReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        warnings_mock.warn.assert_called_once_with(
            'The report was created using SDMetrics version `previous_version` but you are '
            'currently using version `new_version`. Some features may not work as intended.'
        )
        assert loaded == pickle_mock.load.return_value
