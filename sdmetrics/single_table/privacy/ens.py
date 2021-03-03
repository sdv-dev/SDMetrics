import numpy as np

from sdmetrics.single_table.privacy.base import CatPrivacyMetric, PrivacyAttackerModel
from sdmetrics.single_table.privacy.util import majority


class CatENSAttacker(PrivacyAttackerModel):
    """The Categorical ENS (ensemble 'majority vote' classifier) privacy attacker will
    predict the majority of the specified sub-attackers's predicions, and the privacy score will
    be calculated based on the accuracy of its prediction.
    """

    def __init__(self, attackers=[]):
        self.synthetic_dict = {}  # table_name -> {key attribute: [sensitive attribute]}
        self.attackers = [attacker() for attacker in attackers]

    def fit(self, synthetic_data, key, sensitive):
        for attacker in self.attackers:
            attacker.fit(synthetic_data, key, sensitive)

    def predict(self, key_data):
        predictions = [attacker.predict(key_data) for attacker in self.attackers]
        return majority(predictions)


class CatENS(CatPrivacyMetric):
    """The Categorical ENS privacy metric. Scored based on the ENSAttacker.
    When calling cls.compute, please make sure to pass in the following argument:
        model_kwargs (dict):
            {attackers: list[PrivacyAttackerModel]}
    """

    name = 'ENS'
    MODEL = CatENSAttacker
    ACCURACY_BASE = True

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, key=[], sensitive=[],
                model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = cls.MODEL_KWARGS
        if 'attackers' not in model_kwargs:  # no attackers specfied
            return np.nan
        elif not isinstance(model_kwargs['attackers'], list) or\
                len(model_kwargs['attackers']) == 0:
            # zero attackers specfied
            return np.nan
        return super().compute(real_data, synthetic_data, metadata, key, sensitive, model_kwargs)
