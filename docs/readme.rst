Overview
========
The SDMetrics library provides a set of dataset-agnostic tools for evaluating the 
quality of a synthetic database by comparing it to the real database that it is 
modeled after.

Installation
============
To get started with **SDMetrics**, run the following commands to install the library 
in development mode:

.. code-block:: bash

    git clone git@github.com:k15z/SDMetrics.git
    cd SDMetrics
    make install

You should now be able to run the tests with `make test` to verify that the 
installation was successful.

Quickstart
============
Let's run the demo code from **SDV** to generate a simple synthetic dataset:

.. code-block:: python

    from sdv import load_demo, SDV
    metadata, real_tables = load_demo(metadata=True)
    sdv = SDV()
    sdv.fit(metadata, real_tables)
    synthetic_tables = sdv.sample_all(20)

Now that we have a synthetic dataset, we can evaluate it using **SDMetrics** by 
calling the `evaluate` function which returns an instance of `MetricsReport` 
with the default metrics:

.. code-block:: python

    from sdmetrics import evaluate
    report = evaluate(metadata, real_tables, synthetic_tables)

This report object provides a variety of functionality, from computing an overall 
quality score to generating visualizations for exploring the metrics. To learn 
more, check out the API Reference for :py:class:`sdmetrics.MetricsReport`.
