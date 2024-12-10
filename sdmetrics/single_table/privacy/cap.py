"""CAP modules and their attackers."""

import warnings

from sdmetrics.single_table.privacy.base import CategoricalPrivacyMetric, PrivacyAttackerModel
from sdmetrics.single_table.privacy.util import closest_neighbors, count_frequency, majority

DEPRECATION_MSG = (
    'Computing CAP metrics directly is deprecated. For improved privacy metrics, '
    "please use the 'DisclosureProtection' and 'DisclosureProtectionEstimate' "
    'metrics instead.'
)


class CAPAttacker(PrivacyAttackerModel):
    """The CAP (Correct Attribution Probability) privacy attacker.

    It will find out all rows in synthetic table that match the target key attributes, and
    predict the sensitive entry that appears most frequently among them. The privacy score will
    be the frequency the correct sensitive entry appears among all such entries. In the case that
    no such row is found, the attack will be ignored and not counted towards the privacy score.
    """

    def __init__(self):
        self.synthetic_dict = {}  # {key attribute: [sensitive attribute]}

    def fit(self, synthetic_data, key_fields, sensitive_fields):
        """Fit the attacker on the synthetic data.

        Args:
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            key_fields (list(str)):
                Name of the column(s) to use as the key attributes.
            sensitive_fields (list(str)):
                Name of the column(s) to use as the sensitive attributes.
        """
        for idx in range(len(synthetic_data)):
            key_value = tuple(synthetic_data[key_fields].iloc[idx])
            sensitive_value = tuple(synthetic_data[sensitive_fields].iloc[idx])
            if key_value in self.synthetic_dict:
                self.synthetic_dict[key_value].append(sensitive_value)
            else:
                self.synthetic_dict[key_value] = [sensitive_value]

    def predict(self, key_data):
        """Make a prediction of the sensitive data given keys.

        Args:
            key_data (tuple):
                The key data.

        Returns:
            tuple:
                The predicted sensitive data.
        """
        if key_data not in self.synthetic_dict:
            return None  # target key attribute not found in synthetic table

        return majority(self.synthetic_dict[key_data])

    def score(self, key_data, sensitive_data):
        """Score based on the belief of the attacker, in the form P(sensitive_data|key|data).

        Args:
            key_data (tuple):
                The key data.
            sensitive_data (tuple):
                The sensitive data.

        Returns:
            float or None:
                The frequency of the correct sensitive entry.
                Returns `None` if the key is not in the data.
        """
        if key_data in self.synthetic_dict:
            return count_frequency(self.synthetic_dict[key_data], sensitive_data)
        else:
            return None


class CategoricalCAP(CategoricalPrivacyMetric):
    """The Categorical CAP privacy metric. Scored based on the CAPAttacker."""

    name = 'CategoricalCAP'
    MODEL = CAPAttacker
    ACCURACY_BASE = False

    @classmethod
    def _compute(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        key_fields=None,
        sensitive_fields=None,
        model_kwargs=None,
    ):
        return super().compute(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            key_fields=key_fields,
            sensitive_fields=sensitive_fields,
            model_kwargs=model_kwargs,
        )

    @classmethod
    def compute(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        key_fields=None,
        sensitive_fields=None,
        model_kwargs=None,
    ):
        """Compute this metric.

        This fits an adversial attacker model on the synthetic data and
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
                Scores obtained by the attackers when evaluated on the real data.
        """
        warnings.warn(DEPRECATION_MSG, DeprecationWarning)
        return cls._compute(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            key_fields=key_fields,
            sensitive_fields=sensitive_fields,
            model_kwargs=model_kwargs,
        )


class ZeroCAPAttacker(CAPAttacker):
    """The 0CAP privacy attacker, which operates in the same way as CAP does.

    The difference is that when a match in key attribute is not found, the attack will
    be classified as failed and a score of 0 will be recorded.
    """

    def score(self, key_data, sensitive_data):
        """Score based on the belief of the attacker, in the form P(sensitive_data|key|data).

        Args:
            key_data (tuple):
                The key data.
            sensitive_data (tuple):
                The sensitive data.

        Returns:
            float or None:
                The frequency of the correct sensitive entry.
                Returns `0` if the key is not in the data.
        """
        if key_data in self.synthetic_dict:
            return count_frequency(self.synthetic_dict[key_data], sensitive_data)
        else:
            return 0


class CategoricalZeroCAP(CategoricalPrivacyMetric):
    """The Categorical 0CAP privacy metric. Scored based on the ZeroCAPAttacker."""

    name = '0CAP'
    MODEL = ZeroCAPAttacker
    ACCURACY_BASE = False

    @classmethod
    def _compute(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        key_fields=None,
        sensitive_fields=None,
        model_kwargs=None,
    ):
        return super().compute(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            key_fields=key_fields,
            sensitive_fields=sensitive_fields,
            model_kwargs=model_kwargs,
        )

    @classmethod
    def compute(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        key_fields=None,
        sensitive_fields=None,
        model_kwargs=None,
    ):
        """Compute this metric.

        This fits an adversial attacker model on the synthetic data and
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
                Scores obtained by the attackers when evaluated on the real data.
        """
        warnings.warn(DEPRECATION_MSG, DeprecationWarning)
        return cls._compute(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            key_fields=key_fields,
            sensitive_fields=sensitive_fields,
            model_kwargs=model_kwargs,
        )


class GeneralizedCAPAttacker(CAPAttacker):
    """The GeneralizedCAP privacy attacker.

    It will find out all rows in synthetic table that are closest (in hamming distance) to the
    target key attributes, and predict the sensitive entry that appears most frequently among
    them. The privacy score for each row in the real table will be calculated as the frequency
    that the true sensitive attribute appears among all rows in the synthetic table with closest
    key attribute.
    """

    def predict(self, key_data):
        """Make a prediction of the sensitive data given keys.

        Args:
            key_data (tuple):
                The key data.

        Returns:
            tuple:
                The predicted sensitive data.
        """
        ref_key_attributes = closest_neighbors(self.synthetic_dict.keys(), key_data)
        ref_sensitive_attributes = []
        for key in ref_key_attributes:
            ref_sensitive_attributes.extend(self.synthetic_dict[key])

        return majority(ref_sensitive_attributes)

    def score(self, key_data, sensitive_data):
        """Score based on the belief of the attacker, in the form P(sensitive_data|key|data).

        Args:
            key_data (tuple):
                The key data.
            sensitive_data (tuple):
                The sensitive data.

        Returns:
            float or None:
                The frequency of the correct sensitive entry.
        """
        ref_key_attributes = closest_neighbors(self.synthetic_dict.keys(), key_data)
        ref_sensitive_attributes = []
        for key in ref_key_attributes:
            ref_sensitive_attributes.extend(self.synthetic_dict[key])

        return count_frequency(ref_sensitive_attributes, sensitive_data)


class CategoricalGeneralizedCAP(CategoricalPrivacyMetric):
    """The GeneralizedCAP privacy metric. Scored based on the ZeroCAPAttacker."""

    name = 'Categorical GeneralizedCAP'
    MODEL = GeneralizedCAPAttacker
    ACCURACY_BASE = False

    @classmethod
    def _compute(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        key_fields=None,
        sensitive_fields=None,
        model_kwargs=None,
    ):
        return super().compute(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            key_fields=key_fields,
            sensitive_fields=sensitive_fields,
            model_kwargs=model_kwargs,
        )

    @classmethod
    def compute(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        key_fields=None,
        sensitive_fields=None,
        model_kwargs=None,
    ):
        """Compute this metric.

        This fits an adversial attacker model on the synthetic data and
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
                Scores obtained by the attackers when evaluated on the real data.
        """
        warnings.warn(DEPRECATION_MSG, DeprecationWarning)
        return cls._compute(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            key_fields=key_fields,
            sensitive_fields=sensitive_fields,
            model_kwargs=model_kwargs,
        )
