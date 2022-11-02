from sdmetrics.single_table import SingleTableMetric

SINGLE_TABLE_METRICS = [
    'BNLikelihood',
    'BNLogLikelihood',
    'LogisticDetection',
    'SVCDetection',
    'BinaryDecisionTreeClassifier',
    'BinaryAdaBoostClassifier',
    'BinaryLogisticRegression',
    'BinaryMLPClassifier',
    'MulticlassDecisionTreeClassifier',
    'MulticlassMLPClassifier',
    'LinearRegression',
    'MLPRegressor',
    'GMLogLikelihood',
    'CSTest',
    'KSComplement',
    'StatisticSimilarity',
    'BoundaryAdherence',
    'MissingValueSimilarity',
    'CategoryCoverage',
    'TVComplement',
    'RangeCoverage',
    'CategoricalCAP',
    'CategoricalZeroCAP',
    'CategoricalGeneralizedCAP',
    'CategoricalNB',
    'CategoricalKNN',
    'CategoricalRF',
    'CategoricalSVM',
    'CategoricalEnsemble',
    'NumericalLR',
    'NumericalMLP',
    'NumericalSVR',
    'NumericalRadiusNearestNeighbor',
    'ContinuousKLDivergence',
    'DiscreteKLDivergence',
    'ContingencySimilarity',
    'CorrelationSimilarity',
    'NewRowSynthesis',
]


def test_get_single_table_subclasses():
    single_table_metrics = SingleTableMetric.get_subclasses()
    for single_table_metric in SINGLE_TABLE_METRICS:
        assert single_table_metric in single_table_metrics
