"""Metrics for single table datasets."""

from sdmetrics.single_table import (
    base,
    bayesian_network,
    detection,
    efficacy,
    gaussian_mixture,
    multi_single_column,
    privacy,
)
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.bayesian_network import BNLikelihood, BNLogLikelihood
from sdmetrics.single_table.detection.base import DetectionMetric
from sdmetrics.single_table.detection.sklearn import (
    LogisticDetection,
    ScikitLearnClassifierDetectionMetric,
    SVCDetection,
)
from sdmetrics.single_table.efficacy.base import MLEfficacyMetric
from sdmetrics.single_table.efficacy.binary import (
    BinaryAdaBoostClassifier,
    BinaryDecisionTreeClassifier,
    BinaryEfficacyMetric,
    BinaryLogisticRegression,
    BinaryMLPClassifier,
)
from sdmetrics.single_table.efficacy.multiclass import (
    MulticlassDecisionTreeClassifier,
    MulticlassEfficacyMetric,
    MulticlassMLPClassifier,
)
from sdmetrics.single_table.efficacy.regression import (
    LinearRegression,
    MLPRegressor,
    RegressionEfficacyMetric,
)
from sdmetrics.single_table.gaussian_mixture import GMLogLikelihood
from sdmetrics.single_table.multi_column_pairs import (
    ContingencySimilarity,
    ContinuousKLDivergence,
    CorrelationSimilarity,
    DiscreteKLDivergence,
    MultiColumnPairsMetric,
)
from sdmetrics.single_table.multi_single_column import (
    BoundaryAdherence,
    CategoryCoverage,
    CSTest,
    KSComplement,
    MissingValueSimilarity,
    MultiSingleColumnMetric,
    RangeCoverage,
    StatisticSimilarity,
    TVComplement,
)
from sdmetrics.single_table.new_row_synthesis import NewRowSynthesis
from sdmetrics.single_table.privacy.base import CategoricalPrivacyMetric, NumericalPrivacyMetric
from sdmetrics.single_table.privacy.cap import (
    CategoricalCAP,
    CategoricalGeneralizedCAP,
    CategoricalZeroCAP,
)
from sdmetrics.single_table.privacy.categorical_sklearn import (
    CategoricalKNN,
    CategoricalNB,
    CategoricalRF,
    CategoricalSVM,
)
from sdmetrics.single_table.privacy.disclosure_protection import (
    DisclosureProtection,
    DisclosureProtectionEstimate,
)
from sdmetrics.single_table.privacy.dcr_baseline_protection import DCRBaselineProtection
from sdmetrics.single_table.privacy.dcr_overfitting_protection import DCROverfittingProtection
from sdmetrics.single_table.privacy.ensemble import CategoricalEnsemble
from sdmetrics.single_table.privacy.numerical_sklearn import NumericalLR, NumericalMLP, NumericalSVR
from sdmetrics.single_table.privacy.radius_nearest_neighbor import NumericalRadiusNearestNeighbor
from sdmetrics.single_table.table_structure import TableStructure
from sdmetrics.single_table.equalized_odds import EqualizedOddsImprovement

__all__ = [
    'bayesian_network',
    'base',
    'detection',
    'privacy',
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
    'KSComplement',
    'CategoricalCAP',
    'CategoricalZeroCAP',
    'CategoricalGeneralizedCAP',
    'DisclosureProtection',
    'DisclosureProtectionEstimate',
    'NumericalMLP',
    'NumericalLR',
    'NumericalSVR',
    'CategoricalKNN',
    'CategoricalNB',
    'CategoricalRF',
    'CategoricalSVM',
    'CategoricalPrivacyMetric',
    'NumericalPrivacyMetric',
    'CategoricalEnsemble',
    'NumericalRadiusNearestNeighbor',
    'ContingencySimilarity',
    'CorrelationSimilarity',
    'BoundaryAdherence',
    'CategoryCoverage',
    'MissingValueSimilarity',
    'StatisticSimilarity',
    'TVComplement',
    'RangeCoverage',
    'NewRowSynthesis',
    'TableStructure',
    'DCRBaselineProtection',
    'DCROverfittingProtection',
    'EqualizedOddsImprovement',
]
