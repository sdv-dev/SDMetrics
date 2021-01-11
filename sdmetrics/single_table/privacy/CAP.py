from sdmetrics.single_table.privacy.base import CatPrivacyMetric, PrivacyAttackerModel
from sdmetrics.single_table.privacy.util import count_frequency, closest_neighbors, majority


class CAPAttacker(PrivacyAttackerModel):
    """The CAP (Correct attribution probability) privacy attacker will find out all rows
    in synthetic table that matches the target key attributes, and predict the sensitive entry
    that appears most frequently among them. The privacy score will be the frequency the correct
    sensitive entry appears among all such entries. In the case that no such row is found, the
    attack will be ignored and not counted towards the privacy score.
    """
    def __init__(self):
        self.synthetic_dict = {}  # {key attribute: [sensitive attribute]}

    def fit(self, synthetic_data, key, sensitive):
        for idx in range(len(synthetic_data)):
            key_value = tuple(synthetic_data[key].iloc[idx])
            sensitive_value = tuple(synthetic_data[sensitive].iloc[idx])
            if key_value in self.synthetic_dict:
                self.synthetic_dict[key_value].append(sensitive_value)
            else:
                self.synthetic_dict[key_value] = [sensitive_value]

    def predict(self, key_data):
        if key_data not in self.synthetic_dict:
            return None #target key attribute not found in synthetic table
        return majority(self.synthetic_dict[key_data])

    def score(self, key_data, sensitive_data):
        if key_data in self.synthetic_dict:
            return count_frequency(self.synthetic_dict[key_data], sensitive_data)
        else:
            return None


class CAP(CatPrivacyMetric):
    """The CAP privacy metric. Scored based on the CAPAttacker.
    """

    name = 'CAP'
    MODEL = CAPAttacker
    ACCURACY_BASE = False


class ZeroCAPAttacker(CAPAttacker):
    """The 0CAP privacy attacker, which operates in the same way as CAP does.
    The difference is that when a match in key attribute is not found, the attack will
    be classified as failed and a score of 0 will be recorded.
    """
    def score(self, key_data, sensitive_data):
        if key_data in self.synthetic_dict:
            return count_frequency(self.synthetic_dict[key_data], sensitive_data)
        else:
            return 0


class ZeroCAP(CatPrivacyMetric):
    """The 0CAP privacy metric. Scored based on the ZeroCAPAttacker.
    """

    name = '0CAP'
    MODEL = ZeroCAPAttacker
    ACCURACY_BASE = False


class GCAPAttacker(CAPAttacker):
    """The GCAP privacy attacker will find out all rows in synthetic table
    that are closest (in hamming distance) to the target key attributes, and predict
    the sensitive entry that appears most frequently among them. The privacy score for each
    row in the real table will be calculated as the frequency that the true sensitive
    attribute appears among all rows in the synthetic table with closest key attribute.
    """
    def predict(self, key_data):
        ref_key_attributes = closest_neighbors(self.synthetic_dict.keys(), key_data)
        ref_sensitive_attributes = []
        for key in ref_key_attributes:
            ref_sensitive_attributes.extend(self.synthetic_dict[key])
        return majority(ref_sensitive_attributes)

    def score(self, key_data, sensitive_data):
        ref_key_attributes = closest_neighbors(self.synthetic_dict.keys(), key_data)
        ref_sensitive_attributes = []
        for key in ref_key_attributes:
            ref_sensitive_attributes.extend(self.synthetic_dict[key])

        return count_frequency(ref_sensitive_attributes, sensitive_data)


class GCAP(CatPrivacyMetric):
    """The GCAP (General CAP) privacy metric. Scored based on the ZeroCAPAttacker.
    """

    name = 'GCAP'
    MODEL = GCAPAttacker
    ACCURACY_BASE = False
