from sdmetrics.demos import load_demo


def test_load_single_table_demo():
    """Test loading the single tale demo data and expect the correct demo data to be returned."""
    # Setup
    modality = 'single_table'

    # Run
    real_data, synthetic_data, metadata = load_demo(modality)

    # Assert
    assert metadata['columns']['duration'] == {
        'sdtype': 'numerical',
        'computer_representation': 'Int64'
    }
    assert real_data['duration'].dtype == 'float64'
    assert synthetic_data['duration'].dtype == 'float64'


def test_load_multi_table_demo():
    """Test loading the multi table demo data and expect the correct demo data to be returned."""
    # Setup
    modality = 'multi_table'

    # Run
    real_data, synthetic_data, metadata = load_demo(modality)

    # Assert
    assert metadata['tables']['transactions']['columns']['timestamp'] == {
        'sdtype': 'datetime',
        'datetime_format': '%Y-%m-%d %H:%M:%S',
    }
    assert real_data['transactions']['timestamp'].dtype == 'datetime64[ns]'
    assert synthetic_data['transactions']['timestamp'].dtype == 'datetime64[ns]'
