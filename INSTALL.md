# Installing SDMetrics

## Requirements

**SDMetrics** has been developed and tested on [Python 3.8, 3.9, 3.10, 3.11 and 3.12](https://www.python.org/downloads/)

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

## Install with conda

**SDMetrics** can also be installed using [conda](https://docs.conda.io/en/latest/):

```bash
conda install -c sdv-dev -c conda-forge sdmetrics
```

This will pull and install the latest stable release from [Anaconda](https://anaconda.org/).

## Install from source

If you want to install **SDMetrics** from source you need to first clone the repository
and then execute the `make install` command inside the `stable` branch. Note that this
command works only on Unix based systems like GNU/Linux and macOS:

```bash
git clone https://github.com/sdv-dev/SDMetrics
cd SDMetrics
git checkout stable
make install
```

## Install for development

If you intend to modify the source code or contribute to the project you will need to
install it from the source using the `make install-develop` command. In this case, we
recommend you to branch from `main` first:

```bash
git clone git@github.com:sdv-dev/SDMetrics
cd SDMetrics
git checkout main
git checkout -b <your-branch-name>
make install-develp
```

For more details about how to contribute to the project please visit the [Contributing Guide](
CONTRIBUTING.rst).
