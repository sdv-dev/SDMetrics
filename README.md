<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPI Shield](https://img.shields.io/pypi/v/sdmetrics.svg)](https://pypi.python.org/pypi/sdmetrics)
[![Downloads](https://pepy.tech/badge/sdmetrics)](https://pepy.tech/project/sdmetrics)
[![Tests](https://github.com/sdv-dev/SDMetrics/workflows/Run%20Tests/badge.svg)](https://github.com/sdv-dev/SDMetrics/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster)
[![Coverage Status](https://codecov.io/gh/sdv-dev/SDMetrics/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/SDMetrics)

<p>
  <img width=15% src="docs/resources/header.png">
</p>

Metrics for Synthetic Data Generation Projects

* License: [MIT](https://github.com/sdv-dev/SDMetrics/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Documentation: https://sdv-dev.github.io/SDMetrics
* Homepage: https://github.com/sdv-dev/SDMetrics

# Overview

The **SDMetrics** library provides a set of **dataset-agnostic tools** for evaluating the **quality of a synthetic database** by comparing it to the real database that it is modeled after. It includes a variety of metrics such as:

 - **Statistical metrics** which use statistical tests to compare the distributions of the real and synthetic distributions.
 - **Detection metrics** which use machine learning to try to distinguish between real and synthetic data.
 - **Descriptive metrics** which compute descriptive statistics on the real and synthetic datasets independently and then compare the values.

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
[Contributing Guide](https://sdv-dev.github.io/SDMetrics/contributing.html#get-started).

## Install with conda

**SDMetrics** can also be installed using [conda](https://docs.conda.io/en/latest/):

```bash
conda install -c sdv-dev -c conda-forge sdmetrics
```

This will pull and install the latest stable release from [Anaconda](https://anaconda.org/).


# Basic Usage

TODO

# What's next?

For more details about **SDMetrics** and all its possibilities and features, please check
the [documentation site](https://sdv-dev.github.io/SDMetrics/).
