# Multi Table Metrics

The metrics found on this folder operate on multi-table datasets, passed as two python `dict`s
containing tables as `pandas.DataFrame`s.

Implemented metrics:

* Parent-Child Detection metrics: Metrics that de-normalize each parent-child relationship
  in the dataset and then execute a *Single Table Detection Metric* on the generated tables.
    * `LogisticParentChildDetection`: Parent-child detection metric based on a `LogisticDetection`.
    * `SVCParentChildDetection`: Parent-child detection metric based on a `SVCDetection`.
* Multi Single Table Metrics: Metrics that execute a Single Table Metric on each table from the
  dataset and then return the average score obtained by it.
  * `CSTest`: Multi Single Table metric based on the Single Table CSTest metric.
  * `KSComplement`: Multi Single Table metric based on the Single Table KSComplement metric.
  * `LogisticDetection`: Multi Single Table metric based on the Single Table LogisticDetection metric.
  * `SVCDetection`: Multi Single Table metric based on the Single Table SVCDetection metric.
  * `BNLikelihood`: Multi Single Table metric based on the Single Table BNLikelihood metric.
  * `BNLogLikelihood`: Multi Single Table metric based on the Single Table BNLogLikelihood metric.

## MultiTableMetric

All the multi table metrics are subclasses form the `sdmetrics.multi_table.MultiTableMetric`
class, which can be used to locate all of them:

```python3
In [1]: from sdmetrics.multi_table import MultiTableMetric

In [2]: MultiTableMetric.get_subclasses()
Out[2]:
{'CSTest': sdmetrics.multi_table.multi_single_table.CSTest,
 'KSComplement': sdmetrics.multi_table.multi_single_table.KSComplement,
 'LogisticDetection': sdmetrics.multi_table.multi_single_table.LogisticDetection,
 'SVCDetection': sdmetrics.multi_table.multi_single_table.SVCDetection,
 'BNLikelihood': sdmetrics.multi_table.multi_single_table.BNLikelihood,
 'BNLogLikelihood': sdmetrics.multi_table.multi_single_table.BNLogLikelihood,
 'LogisticParentChildDetection': sdmetrics.multi_table.detection.parent_child.LogisticParentChildDetection,
 'SVCParentChildDetection': sdmetrics.multi_table.detection.parent_child.SVCParentChildDetection}
```

## Multi Table Inputs and Outputs

All the multi table metrics operate on at least two inputs:

* `real_data`: A dict containing the table names and data from the real dataset passed as
  `pandas.DataFrame`s
* `synthetic_data`: A dict containing the table names and data from the synthetic dataset passed
  as `pandas.DataFrame`s

For example, a `KStest` metric can be used as follows:

```python3
In [3]: from sdmetrics.multi_table import KSComplement

In [4]: from sdmetrics import load_demo

In [5]: real_data, synthetic_data, metadata = load_demo()

In [6]: KSComplement.compute(real_data, synthetic_data)
Out[6]: 0.8194444444444443
```

Some metrics also require additional information, such as the relationships that exist between
the tables.

For example, this is how you would use a `LogisticParentChildDetection` metric:

```python3
In [7]: from sdmetrics.multi_table import LogisticParentChildDetection

In [8]: foreign_keys = [
   ...: ('users', 'user_id', 'sessions', 'user_id'),
   ...: ('sessions', 'session_id', 'transactions', 'session_id')
   ...: ]

In [9]: LogisticParentChildDetection.compute(real_data, synthetic_data, foreign_keys=foreign_keys)
Out[9]: 0.7569444444444444
```

Additionally, all the metrics accept a `metadata` argument which must be a dict following
the Metadata JSON schema from SDV, which will be used to determine which columns are compatible
with each one of the different metrics, as well as to extract any additional information required
by the metrics, such as the mentioned relationships.

If this dictionary is not passed it will be built based on the data found in the real table,
but in this case some field types may not represent the data accurately (e.g. categorical
columns that contain only integer values will be seen as numerical), and any additional
information required by the metrics will not be populated.

For example, we could execute the same metric as before by passing the `metadata` dict instead
of having to specify the individual `foreign_keys`:

```python
In [10]: LogisticParentChildDetection.compute(real_data, synthetic_data, metadata)
Out[10]: 0.7569444444444444
```
