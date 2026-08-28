import pytest

from sdmetrics.demos import load_demo


@pytest.fixture
def single_table_demo_data_and_metadata():
    real_data, synthetic_data, metadata = load_demo(modality='single_table')
    return real_data['student_placements'], synthetic_data['student_placements'], metadata


@pytest.fixture
def composite_keys_single_table_demo():
    real_data, synthetic_data, metadata = load_demo(modality='single_table')
    metadata['tables']['student_placements']['primary_key'] = ['student_id', 'degree_type']
    return real_data['student_placements'], synthetic_data['student_placements'], metadata


@pytest.fixture(scope='module')
def composite_keys_multi_table_demo():
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    metadata['tables']['users']['columns']['user_type'] = {'sdtype': 'categorical'}
    metadata['tables']['users']['primary_key'] = ['user_id', 'user_type']
    metadata['tables']['sessions']['columns']['user_type'] = {'sdtype': 'categorical'}
    metadata['tables']['sessions']['columns']['user_type'] = {'sdtype': 'categorical'}
    metadata['tables']['sessions']['primary_key'] = ['session_id', 'device']
    metadata['tables']['transactions']['columns']['device'] = {'sdtype': 'categorical'}

    metadata['relationships'][0]['parent_primary_key'] = ['user_id', 'user_type']
    metadata['relationships'][0]['child_foreign_key'] = ['user_id', 'user_type']
    metadata['relationships'][1]['parent_primary_key'] = ['session_id', 'device']
    metadata['relationships'][1]['child_foreign_key'] = ['session_id', 'device']

    real_data['users']['user_type'] = ['PREMIUM'] * 5 + [None] * 5
    synthetic_data['users']['user_type'] = ['PREMIUM'] * 5 + [None] * 5
    for data in [real_data, synthetic_data]:
        data['sessions']['user_type'] = (
            data['users']
            .set_index('user_id')
            .loc[data['sessions']['user_id']]['user_type']
            .to_numpy()
        )
        data['transactions']['device'] = (
            data['sessions']
            .set_index('session_id')
            .loc[data['transactions']['session_id']]['device']
            .to_numpy()
        )
        premium_mask = data['users']['user_type'] == 'PREMIUM'
        data['users'].loc[premium_mask, 'user_id'] = range(5)
        data['users'].loc[~premium_mask, 'user_id'] = range(5)
        data['sessions'].loc[data['sessions']['user_type'].isna(), 'user_id'] -= 5

    return real_data, synthetic_data, metadata


@pytest.fixture
def metadata_v2_single_table_demo():
    """Demo data with a V2 metadata that defines the valid range of the columns."""
    real_data, synthetic_data, metadata = load_demo(modality='single_table')
    columns_meta = metadata['tables']['student_placements']['columns']
    for column_name in [
        'second_perc',
        'high_perc',
        'degree_perc',
        'employability_perc',
        'mba_perc',
    ]:
        columns_meta[column_name].update({
            'range_min': 0,
            'range_max': 100,
            'range_is_nullable': False,
        })

    for column_name in ['start_date', 'end_date']:
        columns_meta[column_name].update({
            'range_min': '2020-01-01',
            'range_max': '2021-12-31',
            'range_is_nullable': True,
        })

    columns_meta['salary'].update({
        'range_min': 0,
        'range_max': 200000,
        'range_is_nullable': True,
    })
    columns_meta['duration'].update({'range_min': 1, 'range_max': 24, 'range_is_nullable': True})
    columns_meta['experience_years'].update({
        'range_min': 0,
        'range_max': 50,
        'range_is_nullable': False,
    })
    columns_meta['high_spec'].update({
        'range_values': ['Arts', 'Commerce', 'Science', 'Engineering'],
        'range_is_nullable': False,
    })
    columns_meta['mba_spec'].update({
        'range_values': ['Mkt&Fin', 'Mkt&HR'],
        'range_is_nullable': False,
    })
    columns_meta['degree_type'].update({
        'range_values': ['Comm&Mgmt', 'Sci&Tech', 'Others'],
        'range_is_nullable': False,
    })

    return real_data['student_placements'], synthetic_data['student_placements'], metadata


@pytest.fixture
def metadata_v2_multi_table_demo():
    """Multi table demo data with a V2 metadata that defines the valid range of the columns."""
    real_data, synthetic_data, metadata = load_demo(modality='multi_table')
    users_meta = metadata['tables']['users']['columns']
    users_meta['age'].update({'range_min': 0, 'range_max': 120, 'range_is_nullable': False})
    users_meta['country'].update({
        'range_values': ['BG', 'DE', 'ES', 'FR', 'IT', 'UK', 'US'],
        'range_is_nullable': False,
    })

    sessions_meta = metadata['tables']['sessions']['columns']
    sessions_meta['device'].update({
        'range_values': ['mobile', 'tablet', 'desktop'],
        'range_is_nullable': False,
    })
    sessions_meta['os'].update({
        'range_values': ['android', 'ios', 'windows'],
        'range_is_nullable': False,
    })

    transactions_meta = metadata['tables']['transactions']['columns']
    transactions_meta['timestamp'].update({
        'range_min': '2019-01-01 00:00:00',
        'range_max': '2019-12-31 23:59:59',
        'range_is_nullable': False,
    })
    transactions_meta['amount'].update({
        'range_min': 0,
        'range_max': 1000,
        'range_is_nullable': False,
    })
    transactions_meta['approved'].update({
        'range_values': [True, False],
        'range_is_nullable': False,
    })

    return real_data, synthetic_data, metadata
