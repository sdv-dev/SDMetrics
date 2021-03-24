import numpy as np

from sdmetrics.single_table.privacy.base import CategoricalPrivacyMetric, PrivacyAttackerModel
from sdmetrics.single_table.privacy.util import majority


class CategoricalEnsembleAttacker(PrivacyAttackerModel):
    """The Categorical ENS (ensemble 'majority vote' classifier) privacy attacker.

    It will predict the majority of the specified sub-attackers's predicions, and the privacy
    score will be calculated based on the accuracy of its prediction.
    """

    def __init__(self, attackers=[]):
        self.attackers = [attacker() for attacker in attackers]

    def fit(self, synthetic_data, key_fields, sensitive_fields):
        """Fit the CategoricalEnsembleAttacker on the synthetic data.

        Args:
            synthetic_data(pandas.DataFrame):
                The synthetic data table used for adverserial learning.
            key_fields(list[str]):
                The names of the key columns.
            sensitive_fields(list[str]):
                The names of the sensitive columns.
        """
        for attacker in self.attackers:
            attacker.fit(synthetic_data, key_fields, sensitive_fields)

    def predict(self, key_data):
        """Make a prediction of the sensitive data given keys.

        Args:
            key_data(tuple):
                The key data.

        Returns:
            tuple:
                The predicted sensitive data.
        """
        predictions = [attacker.predict(key_data) for attacker in self.attackers]
        return majority(predictions)


class CategoricalEnsemble(CategoricalPrivacyMetric):
    """The Categorical Ensemble privacy metric. Scored based on the CategoricalEnsembleAttacker.

    When calling `cls.compute`, please make sure to pass in the argument
    `model_kwargs (dict): {attackers: list[PrivacyAttackerModel]}`.
    """

    name = 'Ensemble'
    MODEL = CategoricalEnsembleAttacker
    ACCURACY_BASE = True

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, key_fields=None,
                sensitive_fields=None, model_kwargs=None):
        """Compute this metric.

        This fits the CategoricalEnsembleAttacker on the synthetic data and
        then evaluates it making predictions on the real data.

        A ``key_fields`` column(s) name must be given, either directly or as a first level
        entry in the ``metadata`` dict, which will be used as the key column(s) for the
        attack.

        A ``sensitive_fields`` column(s) name must be given, either directly or as a first level
        entry in the ``metadata`` dict, which will be used as the sensitive_fields column(s)
        for the attack.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            key_fields (list(str)):
                Name of the column(s) to use as the key attributes.
            sensitive_fields (list(str)):
                Name of the column(s) to use as the sensitive attributes.
            model_kwargs (dict):
                Key word arguments of the attacker model. cls.MODEL_KWARGS will be used
                if none is provided.

        Returns:
            union[float, tuple[float]]:
                Score obtained by the CategoricalEnsembleAttacker when evaluated on the real data.
        """
        if model_kwargs is None:
            model_kwargs = cls.MODEL_KWARGS

        if 'attackers' not in model_kwargs:  # no attackers specfied
            return np.nan
        elif (not isinstance(model_kwargs['attackers'], list)
              or len(model_kwargs['attackers']) == 0):  # zero attackers specfied
            return np.nan

        return super().compute(
            real_data,
            synthetic_data,
            metadata,
            key_fields,
            sensitive_fields,
            model_kwargs
        )
