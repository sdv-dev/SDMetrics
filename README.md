<p align="left">
  <a href="https://dai.lids.mit.edu">
    <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
  </a>
  <i>An Open Source Project from the <a href="https://dai.lids.mit.edu">Data to AI Lab, at MIT</a></i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPI Shield](https://img.shields.io/pypi/v/sdmetrics.svg)](https://pypi.python.org/pypi/sdmetrics)
[![Downloads](https://pepy.tech/badge/sdmetrics)](https://pepy.tech/project/sdmetrics)
[![Tests](https://github.com/sdv-dev/SDMetrics/workflows/Run%20Tests/badge.svg)](https://github.com/sdv-dev/SDMetrics/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster)
[![Coverage Status](https://codecov.io/gh/sdv-dev/SDMetrics/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/SDMetrics)

<img align="center" width=30% src="docs/resources/header.png">

Metrics for Synthetic Data Generation Projects

* Website: https://sdv.dev
* Documentation: https://sdv.dev/SDV
* Repository: https://github.com/sdv-dev/SDMetrics
* License: [MIT](https://github.com/sdv-dev/SDMetrics/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)

# Overview

The **SDMetrics** library provides a set of **dataset-agnostic tools** for evaluating the **quality
of a synthetic database** by comparing it to the real database that it is modeled after.

It supports multiple data modalities:

* **Single Columns**: Compare 1 dimensional `numpy` arrays representing individual columns.
* **Column Pairs**: Compare how columns in a `pandas.DataFrame` relate to each other, in groups of 2.
* **Single Table**: Compare an entire table, represented as a `pandas.DataFrame`.
* **Multi Table**: Compare multi-table and relational datasets represented as a python `dict` with
  multiple tables passed as `pandas.DataFrame`s.
* **Time Series**: Compare tables representing ordered sequences of events.

It includes a variety of metrics such as:

* **Statistical metrics** which use statistical tests to compare the distributions of the real
  and synthetic distributions.
* **Detection metrics** which use machine learning to try to distinguish between real and synthetic data.
* **Efficacy metrics** which compare the performance of machine learning models when run on the synthetic and real data.
* **Bayesian Network and Gaussian Mixture metrics** which learn the distribution of the real data
  and evaluate the likelihood of the synthetic data belonging to the learned distribution.
* **Privacy metrics** which evaluate whether the synthetic data is leaking information about the real data.

# Install

**SDMetrics** is part of the **SDV** project and is automatically installed alongside it. For
details about this process please visit the [SDV Installation Guide](
https://sdv.dev/SDV/getting_started/install.html)

Optionally, **SDMetrics** can also be installed as a standalone library using the following commands:

**Using `pip`:**

```bash
pip install sdmetrics
```

**Using `conda`:**

```bash
conda install -c sdv-dev -c conda-forge -c pytorch sdmetrics
```

For more installation options please visit the [SDMetrics installation Guide](INSTALL.md)

# Usage

**SDMetrics** is included as part of the framework offered by SDV to evaluate the quality of
your synthetic dataset. For more details about how to use it please visit the corresponding
User Guide:

* [Evaluating Synthetic Data](https://sdv.dev/SDV/user_guides/evaluation/index.html)

## Standalone usage

**SDMetrics** can also be used as a standalone library to run metrics individually.

In this short example we show how to use it to evaluate a toy multi-table dataset and its
synthetic replica by running all the compatible multi-table metrics on it:

```python3
import sdmetrics

# Load the demo data, which includes:
# - A dict containing the real tables as pandas.DataFrames.
# - A dict containing the synthetic clones of the real data.
# - A dict containing metadata about the tables.
real_data, synthetic_data, metadata = sdmetrics.load_demo()

# Obtain the list of multi table metrics, which is returned as a dict
# containing the metric names and the corresponding metric classes.
metrics = sdmetrics.multi_table.MultiTableMetric.get_subclasses()

# Run all the compatible metrics and get a report
sdmetrics.compute_metrics(metrics, real_data, synthetic_data, metadata=metadata)
```

The output will be a table with all the details about the executed metrics and their score:

| metric                       | name                                    |      score |   min_value |   max_value | goal     |
|------------------------------|-----------------------------------------|------------|-------------|-------------|----------|
| CSTest                       | Chi-Squared                             |   0.76651  |           0 |           1 | MAXIMIZE |
| KSTest                       | Inverted Kolmogorov-Smirnov D statistic |   0.75     |           0 |           1 | MAXIMIZE |
| KSTestExtended               | Inverted Kolmogorov-Smirnov D statistic |   0.777778 |           0 |           1 | MAXIMIZE |
| LogisticDetection            | LogisticRegression Detection            |   0.882716 |           0 |           1 | MAXIMIZE |
| SVCDetection                 | SVC Detection                           |   0.833333 |           0 |           1 | MAXIMIZE |
| BNLikelihood                 | BayesianNetwork Likelihood              | nan        |           0 |           1 | MAXIMIZE |
| BNLogLikelihood              | BayesianNetwork Log Likelihood          | nan        |        -inf |           0 | MAXIMIZE |
| LogisticParentChildDetection | LogisticRegression Detection            |   0.619444 |           0 |           1 | MAXIMIZE |
| SVCParentChildDetection      | SVC Detection                           |   0.916667 |           0 |           1 | MAXIMIZE |

# What's next?

If you want to read more about each individual metric, please visit the following folders:

* Single Column Metrics: [sdmetrics/single_column](sdmetrics/single_column)
* Single Table Metrics: [sdmetrics/single_table](sdmetrics/single_table)
* Multi Table Metrics: [sdmetrics/multi_table](sdmetrics/multi_table)
* Time Series Metrics: [sdmetrics/timeseries](sdmetrics/timeseries)

# The Synthetic Data Vault

<p>
  <a href="https://sdv.dev">
    <img width=30% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/SDV-Logo-Color-Tagline.png?raw=true">
  </a>
  <p><i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a></i></p>
</p>

* Website: https://sdv.dev
* Documentation: https://sdv.dev/SDV
