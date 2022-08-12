# Single Column Metrics

The metrics found on this folder operate on individual columns (or univariate random variables),
passed as two 1 dimensional arrays.

Implemented metrics:

* Statistical: Metrics that compare the arrays using statistical tests
    * `CSTest`: Chi-Squared test to compare the distributions of two categorical columns.
    * `KSComplement`: Complement to the Kolmogorov-Smirnov statistic to compare the distributions
      of two numerical columns using their empirical CDF.

## SingleColumnMetric

All the single column metrics are subclasses form the `sdmetrics.single_column.SingleColumnMetric`
class, which can be used to locate all of them:

```python3
In [1]: from sdmetrics.single_column import SingleColumnMetric

In [2]: SingleColumnMetric.get_subclasses()
Out[2]:
{'CSTest': sdmetrics.single_column.statistical.cstest.CSTest,
 'KSComplement': sdmetrics.single_column.statistical.kscomplement.KSComplement}
```

## Single Column Inputs and Outputs

All the single column metrics operate on just two inputs:

* `real_data`: A 1d numpy array, coming from the real dataset.
* `synthetic_data`: A 1d numpy array, coming from the synthetic dataset.

For example, this how the KSComplement metric can be computed for the `age` column
from the demo data:

```python3
In [3]: from sdmetrics import load_demo

In [4]: real_data, synthetic_data, metadata = load_demo()

In [5]: from sdmetrics.single_column import KSComplement

In [6]: real_column = real_data['users']['age'].to_numpy()

In [7]: synthetic_column = synthetic_data['users']['age'].to_numpy()

In [8]: KSComplement.compute(real_column, synthetic_column)
Out[8]: 0.8
```
