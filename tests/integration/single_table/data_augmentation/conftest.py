import pytest
from xgboost import XGBClassifier


@pytest.fixture
def xgboost_init():
    xgb_init = XGBClassifier.__init__

    def patched_init(self, *args, **kwargs):
        kwargs['eta'] = 0.3
        kwargs['gamma'] = 6
        kwargs['max_depth'] = 2
        kwargs['min_child_weight'] = 10
        kwargs['base_score'] = 0.5
        kwargs['eval_metric'] = 'logloss'
        kwargs['tree_method'] = 'hist'
        return xgb_init(self, *args, **kwargs)

    return patched_init
