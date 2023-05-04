# Time Series Metrics

The metrics found on this folder operate on individual tables which represent sequencial data.
The tables need to be passed as two `pandas.DataFrame`s alongside optional lists of
`sequence_key` and `context_columns` or a `metadata` dict which contains them.

Implemented metrics:

* Detection Metrics: Metrics that train a Machine Learning Classifier to distinguish between
  the real and the synthetic data. The score obtained by these metrics is the complementary of the
  score obtained by the classifier when cross validated.
    * `LSTMDetection`: Detection metric based on an LSTM Classifier implemented on PyTorch.
* ML Efficacy Metrics: Metrics that compare the score obtained by a Machine Learning model
  when fitted on the synthetic data or real data, and then evaluated on held out real data.
  The output is the score obtained by the model fitted on synthetic data divided by the score
  obtained when fitted on real data. **warning**: These metrics can only be run on datasets
  that represent machine learning problems which are relatively easy to solve. If the performance
  of the models when fitted on real data is too low, the output from these metrics may be
  meaningless.
    * `LSTMClassifierEfficacy`: Efficacy metric based on an LSTM Classifier implemented on PyTorch.

## TimeSeriesMetric

All the timeseries metrics are subclasses form the `sdmetrics.timeseries.TimeSeriesMetric`
class, which can be used to locate all of them:

```python3
In [1]: from sdmetrics.timeseries import TimeSeriesMetric

In [2]: TimeSeriesMetric.get_subclasses()
Out[2]:
{'LSTMDetection': sdmetrics.timeseries.detection.LSTMDetection}
```

## Time Series Inputs and Outputs

All the timeseries metrics operate on at least three inputs:

* `real_data`: A `pandas.DataFrame` with the data from the real dataset.
* `synthetic_data`: A `pandas.DataFrame` with the data from the synthetic dataset.
* `sequence_key`: A `list` indicating which columns represent entities to which
  the different senquences from the dataset belong.

For example, an `LSTMDetection` metric can be used on the `sunglasses` demo data as follows:

```python3
In [3]: from sdmetrics.timeseries import LSTMDetection

In [4]: from sdmetrics.demos import load_timeseries_demo

In [5]: real_data, synthetic_data, metadata = load_timeseries_demo()

In [6]: LSTMDetection.compute(real_data, synthetic_data, sequence_key=['store_id'])
Out[6]: 0.5
```

Additionally, all the metrics accept a `metadata` argument which must be a dict following
the Metadata JSON schema from SDV, which will be used to determine which columns are compatible
with each one of the different metrics, as well as to extract any additional information required
by the metrics, such as the `sequence_key`.

If this dictionary is not passed it will be built based on the data found in the real table,
but in this case some field types may not represent the data accurately (e.g. categorical
columns that contain only integer values will be seen as numerical), and any additional
information required by the metrics will not be populated.

For example, we could execute the same metric as before by adding the `target` entry to the
metadata dict:

```python
In [7]: metadata['sequence_key'] = 'store_id'

In [8]: LSTMDetection.compute(real_data, synthetic_data, metadata=metadata)
Out[8]: 0.5
```
