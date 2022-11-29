from sdmetrics.demos import load_demo


def test_load_single_table_demo():
    """Test loading the single tale demo data and expect the correct demo data to be returned."""
    # Setup
    modality = 'single_table'

    # Run
    real_data, synthetic_data, metadata = load_demo(modality)

    # Assert
    assert metadata['fields']['duration'] == {'type': 'numerical', 'subtype': 'integer'}
    assert real_data['duration'].dtype == 'float64'
    assert synthetic_data['duration'].dtype == 'float64'
