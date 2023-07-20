import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

import sdmetrics
from sdmetrics import _find_addons


@pytest.fixture()
def mock_sdmetrics():
    sdmetrics_module = sys.modules['sdmetrics']
    sdmetrics_mock = Mock()
    sdmetrics_mock.submodule.__name__ = 'sdmetrics.submodule'
    sys.modules['sdmetrics'] = sdmetrics_mock
    yield sdmetrics_mock
    sys.modules['sdmetrics'] = sdmetrics_module


@patch.object(sdmetrics, 'iter_entry_points')
def test__find_addons_module(entry_points_mock, mock_sdmetrics):
    """Test loading an add-on."""
    # Setup
    add_on_mock = Mock(spec=ModuleType)
    entry_point = Mock()
    entry_point.name = 'sdmetrics.submodule.entry_name'
    entry_point.load.return_value = add_on_mock
    entry_points_mock.return_value = [entry_point]

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='sdmetrics_modules')
    assert mock_sdmetrics.submodule.entry_name == add_on_mock
    assert sys.modules['sdmetrics.submodule.entry_name'] == add_on_mock


@patch.object(sdmetrics, 'iter_entry_points')
def test__find_addons_object(entry_points_mock, mock_sdmetrics):
    """Test loading an add-on."""
    # Setup
    entry_point = Mock()
    entry_point.name = 'sdmetrics.submodule:entry_object.entry_method'
    entry_point.load.return_value = 'new_method'
    entry_points_mock.return_value = [entry_point]

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='sdmetrics_modules')
    assert mock_sdmetrics.submodule.entry_object.entry_method == 'new_method'


@patch('warnings.warn')
@patch('sdmetrics.iter_entry_points')
def test__find_addons_bad_addon(entry_points_mock, warning_mock):
    """Test failing to load an add-on generates a warning."""
    # Setup
    def entry_point_error():
        raise ValueError()

    bad_entry_point = Mock()
    bad_entry_point.name = 'bad_entry_point'
    bad_entry_point.module_name = 'bad_module'
    bad_entry_point.load.side_effect = entry_point_error
    entry_points_mock.return_value = [bad_entry_point]
    msg = 'Failed to load "bad_entry_point" from "bad_module".'

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='sdmetrics_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('sdmetrics.iter_entry_points')
def test__find_addons_wrong_base(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = Mock()
    bad_entry_point.name = 'bad_base.bad_entry_point'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'bad_base.bad_entry_point': expected base module to be 'sdmetrics', found "
        "'bad_base'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='sdmetrics_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('sdmetrics.iter_entry_points')
def test__find_addons_missing_submodule(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = Mock()
    bad_entry_point.name = 'sdmetrics.missing_submodule.new_submodule'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'sdmetrics.missing_submodule.new_submodule': module 'sdmetrics' has no "
        "attribute 'missing_submodule'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='sdmetrics_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('sdmetrics.iter_entry_points')
def test__find_addons_module_and_object(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = Mock()
    bad_entry_point.name = 'sdmetrics.missing_submodule:new_object'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'sdmetrics.missing_submodule:new_object': cannot add 'new_object' to "
        "unknown submodule 'sdmetrics.missing_submodule'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='sdmetrics_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch.object(sdmetrics, 'iter_entry_points')
def test__find_addons_missing_object(entry_points_mock, warning_mock, mock_sdmetrics):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = Mock()
    bad_entry_point.name = 'sdmetrics.submodule:missing_object.new_method'
    entry_points_mock.return_value = [bad_entry_point]
    msg = ("Failed to set 'sdmetrics.submodule:missing_object.new_method': missing_object.")

    del mock_sdmetrics.submodule.missing_object

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='sdmetrics_modules')
    warning_mock.assert_called_once_with(msg)
