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
* **Bayesian Network and Gaussian Mixture metrics** which learn the distribution of the real data and evaluate the likelihood of the synthetic data belonging to the learned distribution.
* **Privacy metrics** which evaluate whether the synthetic data is leaking information about the real data.

# Install

## Requirements

**SDMetrics** has been developed and tested on [Python 3.6, 3.7 and 3.8](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](
https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **SDMetrics** is run.

## Install with pip

The easiest and recommended way to install **SDMetrics** is using [pip](
https://pip.pypa.io/en/stable/):

```bash
pip install sdmetrics
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://sdv.dev/SDMetrics/contributing.html#get-started).

## Install with conda

**SDMetrics** can also be installed using [conda](https://docs.conda.io/en/latest/):

```bash
conda install -c sdv-dev -c conda-forge sdmetrics
```

This will pull and install the latest stable release from [Anaconda](https://anaconda.org/).

# Basic Usage

In this small code snippet we show an example of how to use SDMetrics to evaluate how similar
a toy multi-table dataset and its synthetic replica are:

1. The demo data is loaded.
2. The list of available multi-table metrics is retreived.
3. All the metrics are run to compare the real and synthetic data.
4. A `pandas.DataFrame` is built with the results.

```python3
import pandas as pd
import sdmetrics

# Load the demo data, which includes:
# - A dict containing the real tables as pandas.DataFrames.
# - A dict containing the synthetic clones of the real data.
# - A dict containing metadata about the tables.
real_data, synthetic_data, metadata = sdmetrics.load_demo()

# Obtain the list of multi table metrics, which is returned as a dict
# containing the metric names and the corresponding metric classes.
metrics = sdmetrics.multi_table.MultiTableMetric.get_subclasses()

# Iterate over the metrics and compute them, capturing the scores obtained.
scores = []
for name, metric in metrics.items():
    try:
        scores.append({
        'metric': name,
        'score': metric.compute(real_data, synthetic_data, metadata)
        })
    except ValueError:
        pass   # Ignore metrics that do not support this data

# Put the results in a DataFrame for pretty printing.
scores = pd.DataFrame(scores)
```

The result will be a table containing the list of metrics that have been
computed and the scores obtained, similar to this one:

| metric                       |    score |
|------------------------------|----------|
| CSTest                       | 0.76651  |
| KSTest                       | 0.75     |
| KSTestExtended               | 0.777778 |
| LogisticDetection            | 0.925926 |
| SVCDetection                 | 0.703704 |
| LogisticParentChildDetection | 0.541667 |
| SVCParentChildDetection      | 0.923611 |

# What's next?

For more details about **SDMetrics** and **SDV** please visit the [documentation site](
https://sdv.dev/SDV/).

More details about each individual type of metrics can also be found here:

* Single Column Metrics: [sdmetrics/single_column](sdmetrics/single_column)
* Single Table Metrics: [sdmetrics/single_table](sdmetrics/single_table)
* Multi Table Metrics: [sdmetrics/multi_table](sdmetrics/multi_table)

# The Synthetic Data Vault

<p>
  <a href="https://sdv.dev">
    <img width=30% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/SDV-Logo-Color-Tagline.png?raw=true">
  </a>
  <p><i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a></i></p>
</p>

* Website: https://sdv.dev
* Documentation: https://sdv.dev/SDV
