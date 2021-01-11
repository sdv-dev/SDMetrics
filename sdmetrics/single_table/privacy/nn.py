import numpy as np

from sdmetrics.single_table.privacy.base import NumPrivacyMetric, PrivacyAttackerModel
from sdmetrics.single_table.privacy.loss import CdfInvLp


class RadiusNearestNeighborAttacker(PrivacyAttackerModel):
    """The Radius Nearest Neighbor Attacker will predict the sensitive value to be a
    weighted mean of the entries in the synthetic table. Where this weight is given by a
    separate function, and typically describes the closeness between the given key and
    the corresponding entry in the table.
    """
    def __init__(self, weight_func=None, weight_func_kwargs={}):
        """
        Args:
            weight_func (Class):
                The weight function to use.
            weight_func_kwargs (dict):
                Parameters of the weight function.
        """
        self.weight_func = weight_func(**weight_func_kwargs)
        self.synthetic_data = None
        self.key = None
        self.sensitive = None

    def fit(self, synthetic_data, key, sensitive):
        self.weight_func.fit(synthetic_data, key)
        self.synthetic_data = synthetic_data
        self.key = key
        self.sensitive = sensitive

    def predict(self, key_data):
        weights = 0
        summ = None
        modified = False
        for idx in range(len(self.synthetic_data)):
            ref_key = tuple(self.synthetic_data[self.key].iloc[idx])
            sensitive_data = np.array(self.synthetic_data[self.sensitive].iloc[idx])
            weight = self.weight_func.measure(key_data, ref_key)
            weights += weight
            if not modified:
                summ = sensitive_data.copy()
                modified = True
            else:
                summ += sensitive_data
        if weights == 0:
            return (0,) * len(self.sensitive)
        else:
            return tuple(summ / weights)


class CdfInvCutoff(CdfInvLp):
    """Gives weight = 1 if the Lp averaged distance between the entries is below a given cutoff.
    Formally, suppose given key = (k1,..,kn), while the reference key is (k1',...,kn').
    Suppose the cdfs of each entry are c1,...,cn, resp.
    Then weight = 1 if and only if (sum |c_i(ki) - c_i(ki')|**p) / n <= cutoff**p.
    """
    def __init__(self, p=2, cutoff=0.1):
        self.p = p
        self.cdfs = []
        self.cutoff = cutoff**p

    def fit(self, data, cols):
        CdfInvLp.fit(self, data, cols)
        self.cutoff *= len(cols)

    def measure(self, pred, real):
        dist = CdfInvLp.measure(self, pred, real)
        return 1 if dist < self.cutoff else 0


class RadiusNearestNeighbor(NumPrivacyMetric):
    """The Radius Nearest Neighbor privacy metric. Scored based on the RadiusNearestNeighbor.
    """

    name = 'Radius Nearest Neighbor'
    MODEL = RadiusNearestNeighborAttacker
    MODEL_KWARGS = {'weight_func': CdfInvCutoff, 'weight_func_kwargs': {'p': 2, 'cutoff': 0.1}}
