# Single Table Metrics

The metrics found on this folder operate on individual tables, passed as two `pandas.DataFrame`s.

Implemented metrics:

* BayesianNetwork Metrics: Metrics that fit a BayesianNetwork to the distribution of the real data
  and later on evaluate the likelihood of the synthetic data having been sampled from that
  distribution.
    * `BNLikelihood`: Returns the average likelihood across all the rows in the synthetic dataset.
    * `BNLogLikelihood`: Returns the average log likelihood across all the rows in the synthetic
      dataset.
* GaussianMixture Metrics: Metrics that fit a GaussianMixture model to the distribution of the
  real data and later on evaluate the likelihood of the synthetic data having been sampled from that
  distribution.
    * `GMLogLikelihood`: Fits multiple GMMs to the real data using different numbers of components
      and returns the average log likelihood given by them to the synthetic data.
* Detection Metrics: Metrics that train a Machine Learning Classifier to distinguish between
  the real and the synthetic data. The score obtained by these metrics is the complementary of the
  score obtained by the classifier when cross validated.
    * `LogisticDetection`: Detection metric based on a LogisticRegression from scikit learn.
    * `SVCDetection`: Detection metric based on a SVC from scikit learn.
* ML Efficacy Metrics: Metrics that evaluate the score obtained by a Machine Learning model
  when fitted on the synthetic data and then evaluated on the real data. The output is the score
  obtained by the model. **warning**: These metrics can only be run on datasets that represent
  machine learning problems, and the metric score range depends on the difficulty of the
  corresponding problem.
    * `BinaryDecisionTreeClassifier`: ML Efficacy metric for binary classifications problems, based
      on a DecisionTreeClassifier from scikit-learn.
    * `BinaryAdaBoostClassifier`: ML Efficacy metric for binary classifications problems, based
      on an AdaBoostClassifier from scikit-learn.
    * `BinaryLogisticRegression`: ML Efficacy metric for binary classifications problems, based
      on a LogisticRegression from scikit-learn.
    * `BinaryMLPClassifier`: ML Efficacy metric for binary classifications problems, based
      on an MLPClassifier from scikit-learn.
    * `MulticlassDecisionTreeClassifier`: ML Efficacy metric for multiclass classifications problems, based
      on a DecisionTreeClassifier from scikit-learn.
    * `MulticlassMLPClassifier`: ML Efficacy metric for multiclass classifications problems, based
      on an MLPClassifier from scikit-learn.
    * `LinearRegression`: ML Efficacy metric for regression problems, based
      on a LinearRegression from scikit-learn.
    * `MLPRegressor`: ML Efficacy metric for regression problems, based
      on an MLPRegressor from scikit-learn.
    * `MLEfficacy`: Generic ML Efficacy metric that detects the type of ML Problem associated
      with the dataset by analyzing the target column type and then applies all the metrics
      that are compatible with it.
* Privacy Metrics: Metrics that fit an adversial attacker model on the synthetic data and
    then evaluate its accuracy (or probability of making the correct attack) on the real data.
    * `CategoricalCAP`: Privacy Metric for categorical columns, based
    on the Correct Attribution Probability method.
    * `CategoricalZeroCAP`: Privacy Metric for categorical columns, based
    on the Correct Attribution Probability method.
    * `CategoricalGeneralizedCAP`: Privacy Metric for categorical columns, based
    on the Correct Attribution Probability method.
    * `NumericalMLP`: Privacy Metric for numerical columns, based
    on MLPRegressor from scikit-learn.
    * `NumericalLR`: Privacy Metric for numerical columns, based
    on LinearRegression from scikit-learn.
    * `NumericalSVR`: Privacy Metric for numerical columns, based
    on SVR from scikit-learn.
    * `CategoricalKNN`: Privacy Metric for categorical columns, based
    on KNeighborsClassifier from scikit-learn.
    * `CategoricalNB`: Privacy Metric for categorical columns, based
    on CategoricalNB from scikit-learn.
    * `CategoricalRF`: Privacy Metric for categorical columns, based
    on RandomForestClassifier from scikit-learn.
    * `CategoricalEnsemble`: Privacy Metric for categorical columns, based
    on an 'ensemble' of other categorical Privacy Metrics.
    * `NumericalRadiusNearestNeighbor`: Privacy Metric for numerical columns, based
    on an implementation of the Radius Nearest Neighbor method.
* MultiSingleColumn Metrics: Metrics that apply a Single Column metric on each column from
  the table that is compatible with it and then compute the average across all the columns.
    * `CSTest`: MultiSingleColumn metric based on applying the Single Column CSTest on all
      the categorical variables.
    * `KSComplement`: MultiSingleColumn metric based on applying the Single Column KSComplement on all
      the numerical variables.
* MultiColumnPairs Metrics: Metrics that apply a ColumnPairs metric on each pair of columns from
  the tables which are compatible with it and then compute the average across all the columns pairs.
    * `ContinuousKLDivergence`: MultiColumnPairs metric based on applying the ColumnPairs
      ContinuousKLDivergence on all the possible pairs of numerical columns.
    * `DiscreteKLDivergence`: MultiColumnPairs metric based on applying the ColumnPairs
      DiscreteKLDivergence on all the possible pairs of categorical and boolean columns.

## SingleTableMetric

All the single table metrics are subclasses form the `sdmetrics.single_table.SingleTableMetric`
class, which can be used to locate all of them:

```python3
In [1]: from sdmetrics.single_table import SingleTableMetric

In [2]: SingleTableMetric.get_subclasses()
Out[2]:
{'BNLogLikelihood': sdmetrics.single_table.bayesian_network.BNLogLikelihood,
 'LogisticDetection': sdmetrics.single_table.detection.sklearn.LogisticDetection,
 'SVCDetection': sdmetrics.single_table.detection.sklearn.SVCDetection,
 'BinaryDecisionTreeClassifier': sdmetrics.single_table.efficacy.binary.BinaryDecisionTreeClassifier,
 'BinaryAdaBoostClassifier': sdmetrics.single_table.efficacy.binary.BinaryAdaBoostClassifier,
 'BinaryLogisticRegression': sdmetrics.single_table.efficacy.binary.BinaryLogisticRegression,
 'BinaryMLPClassifier': sdmetrics.single_table.efficacy.binary.BinaryMLPClassifier,
 'MulticlassDecisionTreeClassifier': sdmetrics.single_table.efficacy.multiclass.MulticlassDecisionTreeClassifier,
 'MulticlassMLPClassifier': sdmetrics.single_table.efficacy.multiclass.MulticlassMLPClassifier,
 'LinearRegression': sdmetrics.single_table.efficacy.regression.LinearRegression,
 'MLPRegressor': sdmetrics.single_table.efficacy.regression.MLPRegressor,
 'GMLogLikelihood': sdmetrics.single_table.gaussian_mixture.GMLogLikelihood,
 'CSTest': sdmetrics.single_table.multi_single_column.CSTest,
 'KSComplement': sdmetrics.single_table.multi_single_column.KSComplement,
 'ContinuousKLDivergence': sdmetrics.single_table.multi_column_pairs.ContinuousKLDivergence,
 'DiscreteKLDivergence': sdmetrics.single_table.multi_column_pairs.DiscreteKLDivergence,
 'CategoricalCAP': sdmetrics.single_table.privacy.cap,
 'CategoricalGeneralizedCAP': sdmetrics.single_table.privacy.cap,
 'CategoricalZeroCAP': sdmetrics.single_table.privacy.cap,
 'CategoricalKNN': sdmetrics.single_table.privacy.cap,
 'CategoricalNB': sdmetrics.single_table.privacy.cap,
 'CategoricalRF': sdmetrics.single_table.privacy.cap,
 'CategoricalEnsemble': sdmetrics.single_table.privacy.ensemble,
 'NumericalLR': sdmetrics.single_table.privacy.numerical_sklearn,
 'NumericalMLP': sdmetrics.single_table.privacy.numerical_sklearn,
 'NumericalSVR': sdmetrics.single_table.privacy.numerical_sklearn,
 'NumericalRadiusNearestNeighbor': sdmetrics.single_table.privacy.radius_nearest_neighbor}
```

## Single Table Inputs and Outputs

All the single table metrics operate on at least two inputs:

* `real_data`: A `pandas.DataFrame` with the data from the real dataset.
* `synthetic_data`: A `pandas.DataFrame` with the data from the synthetic dataset.

For example, a `LogisticDetection` metric can be used on the `users` table from the
demo data as follows:

```python3
In [3]: from sdmetrics.single_table import LogisticDetection

In [4]: from sdmetrics import load_demo

In [5]: real_data, synthetic_data, metadata = load_demo()

In [6]: real_table = real_data['users']

In [7]: synthetic_table = synthetic_data['users']

In [8]: LogisticDetection.compute(real_table, synthetic_table)
Out[8]: 1.0
```

Some metrics also require additional information, such as the `target` column to use
when running an ML Efficacy metric.

For example, this is how you would use a `MulticlassDecisionTreeClassifier` on the `country`
column from the demo table `users`:

```python3
In [9]: from sdmetrics.single_table import MulticlassDecisionTreeClassifier

In [10]: MulticlassDecisionTreeClassifier.compute(real_table, synthetic_table, target='country')
Out[10]: (0.05555555555555555,)
```

Additionally, all the metrics accept a `metadata` argument which must be a dict following
the Metadata JSON schema from SDV, which will be used to determine which columns are compatible
with each one of the different metrics, as well as to extract any additional information required
by the metrics, such as the `target` column to use for ML Efficacy metrics.

If this dictionary is not passed it will be built based on the data found in the real table,
but in this case some field types may not represent the data accurately (e.g. categorical
columns that contain only integer values will be seen as numerical), and any additional
information required by the metrics will not be populated.

For example, we could execute the same metric as before by adding the `target` entry to the
metadata dict:

```python
In [11]: users_metadata = metadata['tables']['users'].copy()

In [12]: users_metadata['target'] = 'country'

In [13]: MulticlassDecisionTreeClassifier.compute(real_table, synthetic_table, metadata=users_metadata)
Out[13]: (0.05555555555555555,)
```
