import numpy as np
import pandas as pd

from sdmetrics.single_table.privacy.base import CatPrivacyMetric, PrivacyAttackerModel
from sdmetrics.single_table.privacy.util import majority

class ENSAttacker(PrivacyAttackerModel):
    """The ENS (ensemble classifier; i.e. 'majority vote') privacy attacker's predicion will
    be the majority of the specified sub-attackers's predicions, and the privacy score will
    be calculated based on the accuracy of its prediction.
    """
    def __init__(self, attackers = []):
        self.synthetic_dict = {} #table_name -> {key attribute: [sensitive attribute]}
        self.attackers = attackers

    def fit(self, synthetic_data, key, sensitive):
        for attacker in self.attackers:
            attacker.fit(synthetic_data, key, sensitive)

    def predict(self, key_data):
        predictions = [attacker.predict(key_data) for attacker in self.attackers]
        return majority(predictions)

class ENS(CatPrivacyMetric):
    """The ENS privacy metric. Scored based on the ENSAttacker.
    When calling cls.compute, please make sure to pass in the following argument:
        model_kwargs (dict):
            {attackers: list[PrivacyAttackerModel]}
    """

    name = 'ENS'
    MODEL = ENSAttacker
    ACCURACY_BASE = True