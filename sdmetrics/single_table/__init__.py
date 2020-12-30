"""Metrics for single table datasets."""

from sdmetrics.single_table import (
    base, bayesian_network, detection, efficacy, gaussian_mixture, multi_single_column)
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.bayesian_network import BNLikelihood, BNLogLikelihood
from sdmetrics.single_table.detection.base import DetectionMetric
from sdmetrics.single_table.detection.sklearn import (
    LogisticDetection, ScikitLearnClassifierDetectionMetric, SVCDetection)
from sdmetrics.single_table.efficacy.base import MLEfficacyMetric
from sdmetrics.single_table.efficacy.binary import (
    BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, BinaryEfficacyMetric,
    BinaryLogisticRegression, BinaryMLPClassifier)
from sdmetrics.single_table.efficacy.multiclass import (
    MulticlassDecisionTreeClassifier, MulticlassEfficacyMetric, MulticlassMLPClassifier)
from sdmetrics.single_table.efficacy.regression import (
    LinearRegression, MLPRegressor, RegressionEfficacyMetric)
from sdmetrics.single_table.gaussian_mixture import GMLogLikelihood
from sdmetrics.single_table.multi_column_pairs import (
    ContinuousKLDivergence, DiscreteKLDivergence, MultiColumnPairsMetric)
from sdmetrics.single_table.multi_single_column import (
    CSTest, KSTest, KSTestExtended, MultiSingleColumnMetric)

__all__ = [
    'bayesian_network',
    'base',
    'detection',
    'efficacy',
    'gaussian_mixture',
    'multi_single_column',
    'SingleTableMetric',
    'BNLikelihood',
    'BNLogLikelihood',
    'DetectionMetric',
    'LogisticDetection',
    'SVCDetection',
    'ScikitLearnClassifierDetectionMetric',
    'MLEfficacyMetric',
    'BinaryEfficacyMetric',
    'BinaryDecisionTreeClassifier',
    'BinaryAdaBoostClassifier',
    'BinaryLogisticRegression',
    'BinaryMLPClassifier',
    'MulticlassEfficacyMetric',
    'MulticlassDecisionTreeClassifier',
    'MulticlassMLPClassifier',
    'RegressionEfficacyMetric',
    'LinearRegression',
    'MLPRegressor',
    'GMLogLikelihood',
    'MultiColumnPairsMetric',
    'ContinuousKLDivergence',
    'DiscreteKLDivergence',
    'MultiSingleColumnMetric',
    'CSTest',
    'KSTest',
    'KSTestExtended',
]
