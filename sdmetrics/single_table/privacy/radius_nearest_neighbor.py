import numpy as np

from sdmetrics.single_table.privacy.base import NumericalPrivacyMetric, PrivacyAttackerModel
from sdmetrics.single_table.privacy.loss import InverseCDFDistance


class NumericalRadiusNearestNeighborAttacker(PrivacyAttackerModel):
    """The Radius Nearest Neighbor Attacker.

    It will predict the sensitive value to be a weighted mean of the entries in the
    synthetic table. Where this weight is given by a separate function, and typically
    describes the closeness between the given key and the corresponding entry in the table.
    """

    def __init__(self, weight_func=None, weight_func_kwargs=None):
        """
        Args:
            weight_func (Class):
                The weight function to use.
            weight_func_kwargs (dict):
                Parameters of the weight function.
        """
        if weight_func_kwargs is None:
            weight_func_kwargs = {}

        self.weight_func = weight_func(**weight_func_kwargs)
        self.synthetic_data = None
        self.key = None
        self.sensitive_fields = None
        self.key_fields = None

    def fit(self, synthetic_data, key_fields, sensitive_fields):
        """Fit the NumericalRadiusNearestNeighborAttacker on the synthetic data.

        Args:
            synthetic_data(pandas.DataFrame):
                The synthetic data table used for adverserial learning.
            key_fields(list[str]):
                The names of the key columns.
            sensitive_fields(list[str]):
                The names of the sensitive columns.
        """
        self.weight_func.fit(synthetic_data, key_fields)
        self.synthetic_data = synthetic_data
        self.key_fields = key_fields
        self.sensitive_fields = sensitive_fields

    def predict(self, key_data):
        """Make a prediction of the sensitive data given keys.

        Args:
            key_data(tuple):
                The key data.

        Returns:
            tuple:
                The predicted sensitive data.
        """
        weights = 0
        summ = None
        modified = False
        for idx in range(len(self.synthetic_data)):
            ref_key = tuple(self.synthetic_data[self.key_fields].iloc[idx])
            sensitive_data = np.array(self.synthetic_data[self.sensitive_fields].iloc[idx])
            weight = self.weight_func.measure(key_data, ref_key)
            weights += weight
            if not modified:
                summ = sensitive_data.copy()
                modified = True
            else:
                summ += sensitive_data

        if weights == 0:
            return (0,) * len(self.sensitive_fields)
        else:
            return tuple(summ / weights)


class InverseCDFCutoff(InverseCDFDistance):
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
        """Fits univariate distributions (automatically selected).

        Args:
            data (DataFrame):
                Data, where each column in `cols` is a continuous column.
            cols (list[str]):
                Column names.
        """
        InverseCDFDistance.fit(self, data, cols)
        self.cutoff *= len(cols)

    def measure(self, pred, real):
        """Compute the distance (L_p norm) between the pred and real values.

        This uses the probability integral transform to map the pred/real values
        to a CDF value (between 0.0 and 1.0). Then, it computes the L_p norm
        between the CDF(pred) and CDF(real).

        Args:
            pred (tuple):
                Predicted value(s) corresponding to the columns specified in fit.
            real (tuple):
                Real value(s) corresponding to the columns specified in fit.

        Returns:
            float:
                The L_p norm of the CDF value.
        """
        dist = InverseCDFDistance.measure(self, pred, real)
        return 1 if dist < self.cutoff else 0


class NumericalRadiusNearestNeighbor(NumericalPrivacyMetric):
    """The Radius Nearest Neighbor privacy metric.

    Scored based on the NumericalRadiusNearestNeighborAttacker.
    """

    name = 'Numerical Radius Nearest Neighbor'
    MODEL = NumericalRadiusNearestNeighborAttacker
    MODEL_KWARGS = {
        'weight_func': InverseCDFCutoff, 'weight_func_kwargs': {'p': 2, 'cutoff': 0.3}
    }
