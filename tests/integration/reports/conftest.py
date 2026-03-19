import pytest

from sdmetrics.demos import load_demo


@pytest.fixture(scope='module')
def composite_keys_single_table_demo():
    real_data, synthetic_data, metadata = load_demo(modality='single_table')
    metadata['primary_key'] = ['student_id', 'degree_type']
    return real_data, synthetic_data, metadata


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
