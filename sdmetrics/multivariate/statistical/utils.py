import warnings
from collections import Counter


def frequencies(real, synthetic):
    """
    Given two iterators containing categorical data, this transforms it into
    observed/expected frequencies which can be used for statistical tests. It
    adds a regularization term to handle cases where the synthetic data contains
    values that don't exist in the real data.

    Args:
        real (list): A list of hashable objects.
        synthetic (list): A list of hashable objects.

    Yields:
        (list, list): The observed and expected frequencies (as a percent).
    """
    f_obs, f_exp = [], []
    real, synthetic = Counter(real), Counter(synthetic)
    for value in synthetic:
        if value not in real:
            warnings.warn("Unexpected value %s in synthetic data." % (value,))
            real[value] += 1e-6  # Regularization to prevent NaN.
    for value in real:
        f_obs.append(synthetic[value] / sum(synthetic.values()))
        f_exp.append(real[value] / sum(real.values()))
    return f_obs, f_exp
