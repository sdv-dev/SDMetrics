# History

## v0.6.0 - 2022-08-12

This release removes SDMetric's dependency on the RDT library, and also introduces new quality and diagnostic metrics. Additionally, we introduce a new `compute_breakdown` method that returns a breakdown of metric results.

### New Features
* Handle null values correctly - Issue [#194](https://github.com/sdv-dev/SDMetrics/issues/194) by @katxiao
* Add wrapper classes for new single and multi table metrics - Issue [#169](https://github.com/sdv-dev/SDMetrics/issues/169) by @katxiao
* Add CorrelationSimilarity metric - Issue [#143](https://github.com/sdv-dev/SDMetrics/issues/143) by @katxiao
* Add CardinalityShapeSimilarity metric - Issue [#160](https://github.com/sdv-dev/SDMetrics/issues/160) by @katxiao
* Add CardinalityStatisticSimilarity metric - Issue [#145](https://github.com/sdv-dev/SDMetrics/issues/145) by @katxiao
* Add ContingencySimilarity Metric - Issue [#159](https://github.com/sdv-dev/SDMetrics/issues/159) by @katxiao
* Add TVComplement metric - Issue [#142](https://github.com/sdv-dev/SDMetrics/issues/142) by @katxiao
* Add MissingValueSimilarity metric - Issue [#139](https://github.com/sdv-dev/SDMetrics/issues/139) by @katxiao
* Add CategoryCoverage metric - Issue [#140](https://github.com/sdv-dev/SDMetrics/issues/140) by @katxiao
* Add compute breakdown column for single column - Issue [#152](https://github.com/sdv-dev/SDMetrics/issues/152) by @katxiao
* Add BoundaryAdherence metric - Issue [#138](https://github.com/sdv-dev/SDMetrics/issues/138) by @katxiao
* Get KSComplement Score Breakdown - Issue [#130](https://github.com/sdv-dev/SDMetrics/issues/130) by @katxiao
* Add StatisticSimilarity Metric - Issue [#137](https://github.com/sdv-dev/SDMetrics/issues/130) by @katxiao
* New features for KSTest.compute - Issue [#129](https://github.com/sdv-dev/SDMetrics/issues/129) by @amontanez24

### Internal Improvements
* Add integration tests and fixes - Issue [#183](https://github.com/sdv-dev/SDMetrics/issues/183) by @katxiao
* Remove rdt hypertransformer dependency in timeseries metrics - Issue [#176](https://github.com/sdv-dev/SDMetrics/issues/178) by @katxiao
* Replace rdt LabelEncoder with sklearn - Issue [#178](https://github.com/sdv-dev/SDMetrics/issues/178) by @katxiao
* Remove rdt as a dependency - Issue [#182](https://github.com/sdv-dev/SDMetrics/issues/182) by @katxiao
* Use sklearn's OneHotEncoder instead of rdt - Issue [#170](https://github.com/sdv-dev/SDMetrics/issues/170) by @katxiao
* Remove KSTestExtended - Issue [#180](https://github.com/sdv-dev/SDMetrics/issues/180) by @katxiao
* Remove TSFClassifierEfficacy and TSFCDetection metrics - Issue [#171](https://github.com/sdv-dev/SDMetrics/issues/171) by @katxiao
* Update the default tags for a feature request - Issue [#172](https://github.com/sdv-dev/SDMetrics/issues/172) by @katxiao
* Bump github macos version - Issue [#174](https://github.com/sdv-dev/SDMetrics/issues/174) by @katxiao
* Fix pydocstyle to check sdmetrics - Issue [#153](https://github.com/sdv-dev/SDMetrics/issues/153) by @pvk-developer
* Update the RDT version to 1.0 - Issue [#150](https://github.com/sdv-dev/SDMetrics/issues/150) by @pvk-developer
* Update slack invite link - Issue [#132](https://github.com/sdv-dev/SDMetrics/issues/132) by @pvk-developer

## v0.5.0 - 2022-05-11

This release fixes an error  where the relational `KSTest` crashes if a table doesn't have numerical columns.
It also includes some housekeeping, updating the pomegranate and copulas version requirements.

### Issues closed

* Cap pomegranate to <0.14.7 - Issue [#116](https://github.com/sdv-dev/SDMetrics/issues/116) by @csala
* Relational KSTest crashes with IncomputableMetricError if a table doesn't have numerical columns - Issue [#109](https://github.com/sdv-dev/SDMetrics/pull/109) by @katxiao

## v0.4.1 - 2021-12-09

This release improves the handling of metric errors, and updates the default transformer behavior used in SDMetrics.

### Issues closed

* Report metric errors from compute_metrics - Issue [#107](https://github.com/sdv-dev/SDMetrics/issues/107) by @katxiao
* Specify default categorical transformers - Issue [#105](https://github.com/sdv-dev/SDMetrics/pull/105) by @katxiao

## v0.4.0 - 2021-11-16

This release adds support for Python 3.9 and updates dependencies to ensure compatibility with the
rest of the SDV ecosystem, and upgrades to the latests [RDT](https://github.com/sdv-dev/RDT/releases/tag/v0.6.1)
release.

### Issues closed

* Replace `sktime` for `pyts` - Issue [#103](https://github.com/sdv-dev/SDMetrics/issues/103) by @pvk-developer
* Add support for Python 3.9 - Issue [#102](https://github.com/sdv-dev/SDMetrics/issues/102) by @pvk-developer
* Increase code style lint - Issue [#80](https://github.com/sdv-dev/SDMetrics/issues/80) by @fealho
* Add `pip check` to `CI` workflows - Issue [#79](https://github.com/sdv-dev/SDMetrics/issues/79) by @pvk-developer
* Upgrade dependency ranges - Issue [#69](https://github.com/sdv-dev/SDMetrics/issues/69) by @katxiao

## v0.3.2 - 2021-08-16

This release makes `pomegranate` an optional dependency.

### Issues closed

* Make pomegranate an optional dependency - Issue [#63](https://github.com/sdv-dev/SDMetrics/issues/63) by @fealho

## v0.3.1 - 2021-07-12

This release fixes a bug to make the privacy metrics available in the API docs.
It also updates dependencies to ensure compatibility with the rest of the SDV ecosystem.

### Issues closed

* `CategoricalSVM` not being imported - Issue [#65](https://github.com/sdv-dev/SDMetrics/issues/65) by @csala

## v0.3.0 - 2021-03-30

This release includes privacy metrics to evaluate if the real data could be obtained or
deduced from the synthetic samples. Additionally all the metrics have a `normalize` method
which takes the `raw_score` generated by the metric and returns a value between `0 ` and `1`.

### Issues closed

* Add normalize method to metrics - Issue [#51](https://github.com/sdv-dev/SDMetrics/issues/51) by @csala and @fealho
* Implement privacy metrics - Issue [#36](https://github.com/sdv-dev/SDMetrics/issues/36) by @ZhuofanXie and @fealho

## v0.2.0 - 2021-02-24

Dependency upgrades to ensure compatibility with the rest of the SDV ecosystem.

## v0.1.3 - 2021-02-13

Updates the required dependecies to facilitate a conda release.

### Issues closed

* Upgrade sktime - Issue [#49](https://github.com/sdv-dev/SDMetrics/issues/49) by @fealho

## v0.1.2 - 2021-01-27

Big fixing release that addresses several minor errors.

### Issues closed

* More splits than classes - Issue [#46](https://github.com/sdv-dev/SDMetrics/issues/46) by @fealho
* Scipy 1.6.0 causes an AttributeError - Issue [#44](https://github.com/sdv-dev/SDMetrics/issues/44) by @fealho
* Time series metrics fails with variable length timeseries - Issue [#42](https://github.com/sdv-dev/SDMetrics/issues/42) by @fealho
* ParentChildDetection metrics KeyError - Issue [#39](https://github.com/sdv-dev/SDMetrics/issues/39) by @csala

## v0.1.1 - 2020-12-30

This version adds Time Series Detection and Efficacy metrics, as well as a fix
to ensure that Single Table binary classification efficacy metrics work well
with binary targets which are not boolean.

### Issues closed

* Timeseries efficacy metrics - Issue [#35](https://github.com/sdv-dev/SDMetrics/issues/35) by @csala
* Timeseries detection metrics - Issue [#34](https://github.com/sdv-dev/SDMetrics/issues/34) by @csala
* Ensure binary classification targets are bool - Issue [#33](https://github.com/sdv-dev/SDMetrics/issues/33) by @csala

## v0.1.0 - 2020-12-18

This release introduces a new project organization and API, with metrics
grouped by data modality, with a common API:

* Single Column
* Column Pair
* Single Table
* Multi Table
* Time Series

Within each data modality, different families of metrics have been implemented:

* Statistical
* Detection
* Bayesian Network and Gaussian Mixture Likelihood
* Machine Learning Efficacy

## v0.0.4 - 2020-11-27

Patch release to relax dependencies and avoid conflicts when using the latest SDV version.

## v0.0.3 - 2020-11-20

Fix error on detection metrics when input data contains infinity or NaN values.

### Issues closed

* ValueError: Input contains infinity or a value too large for dtype('float64') - Issue [#11](https://github.com/sdv-dev/SDMetrics/issues/11) by @csala

## v0.0.2 - 2020-08-08

Add support for Python 3.8 and a broader range of dependencies.

## v0.0.1 - 2020-06-26

First release to PyPI.
