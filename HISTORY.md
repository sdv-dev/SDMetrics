# History

## v0.10.0 - 2023-05-03

This release makes the `DiagnosticReport` more fault tolerant by preventing it from crashing if a metric it uses fails. It also adds support for Pandas 2.0!

Additionally, support for the old `SDV` metadata format (pre `SDV` 1.0) has been dropped.

### New Features

* Cleanup SDMetrics to only accept SDV 1.0 metadata format - Issue [#331](https://github.com/sdv-dev/SDMetrics/issues/331) by @amontanez24
* Make the diagnostic report more fault-tolerant - Issue [#332](https://github.com/sdv-dev/SDMetrics/issues/332) by @frances-h

### Maintenance

* Remove upper bound for pandas - Issue [#338](https://github.com/sdv-dev/SDMetrics/issues/338) by @pvk-developer

## v0.9.3 - 2023-04-12

This release improves the clarity of warning/error messages. We also add a version add-on, update the workflow to optimize the runtime and fix a bug in the `NewRowSynthesis` metric when computing the `synthetic_sample_size` for multi-table.

### New Features
* Add functionality to find version add-on - Issue [#321](https://github.com/sdv-dev/SDMetrics/issues/321) by @frances-h
* More detailed warning in QualityReport when there is a constant input - Issue [#316](https://github.com/sdv-dev/SDMetrics/issues/316) by @pvk-developer
* Make error more informative in QualityReport when tables cannot be merged - Issue [#317](https://github.com/sdv-dev/SDMetrics/issues/317) by @frances-h
* More detailed warning in QualityReport for unexpected category values - Issue [#315](https://github.com/sdv-dev/SDMetrics/issues/315) by @frances-h

### Bug Fixes
* Multi table DiagnosticReport sets synthetic_sample_size too low for NewRowSynthesis - Issue [#320](https://github.com/sdv-dev/SDMetrics/issues/320) by @pvk-developer


## v0.9.2 - 2023-03-08

This release fixes bugs in the  `NewRowSynthesis` metric when too many columns were present. It also fixes bugs around datetime columns that are formatted as strings in both `get_column_pair_plot` and `get_column_plot`.

### Bug Fixes
* Method get_column_pair_plot: Does not plot synthetic data if datetime column is formatted as a string - Issue [#310] (https://github.com/sdv-dev/SDMetrics/issues/310) by @frances-h
* Method get_column_plot: ValueError if a datetime column is formatted as a string - Issue [#309](https://github.com/sdv-dev/SDMetrics/issues/309) by @frances-h
* Fix ValueError in the NewRowSynthesis metric (also impacts DiagnosticReport) - Issue [#307](https://github.com/sdv-dev/SDMetrics/issues/307) by @frances-h

## v0.9.1 - 2023-02-17

This release fixes bugs in the existing metrics and reports.

### Bug Fixes
* Fix issue-296 for discrete and continuous columns - Issue [#296](https://github.com/sdv-dev/SDMetrics/issues/296) by @R-Palazzo
* Support new metadata for datetime_format - Issue [#303](https://github.com/sdv-dev/SDMetrics/issues/303) by @frances-h

## v0.9.0 - 2023-01-18

This release supports Python 3.10 and drops support for Python 3.6. We also add a verbosity argument to report generation.

### New Features
* Silent mode when creating reports. - Issue [#269](https://github.com/sdv-dev/SDMetrics/issues/269) by @katxiao
* Support Python versions >=3.7 and <3.11 - Issue [287](https://github.com/sdv-dev/SDMetrics/issues/287) by @katxiao

## v0.8.1 - 2022-12-09

This release fixes bugs in the existing metrics and reports. We also make the reports compatible with future SDV versions.

### New Features
* Filter out additional sdtypes that will be available in future versions of SDV - Issue [#265](https://github.com/sdv-dev/SDMetrics/issues/265) by @katxiao
* NewRowSynthesis should ignore PrimaryKey column - Issue [#260](https://github.com/sdv-dev/SDMetrics/issues/260) by @katxiao

### Bug Fixes
* Visualization crashes if there are metric errors - Issue [#272](https://github.com/sdv-dev/SDMetrics/issues/272) by @katxiao
* Score for TVComplement if synthetic data only has missing values - Issue [#271](https://github.com/sdv-dev/SDMetrics/issues/271) by @katxiao
* Fix 'timestamp' column metadata in the multi table demo - Issue [#267](https://github.com/sdv-dev/SDMetrics/issues/267) by @katxiao
* Fix 'duration' column in the single table demo - Issue [#266](https://github.com/sdv-dev/SDMetrics/issues/266) by @katxiao
* README.md example has a bug - Issue [#262](https://github.com/sdv-dev/SDMetrics/issues/262) by @katxiao
* Update README.md to fix a bug - Issue [#263](https://github.com/sdv-dev/SDMetrics/issues/263) by @katxiao
* Visualization get_column_pair_plot: update parameter name to column_names - Issue [#258](https://github.com/sdv-dev/SDMetrics/issues/258) by @katxiao
* "Column Shapes" and "Column Pair Trends" Calculation Inconsistency - Issue [#254](https://github.com/sdv-dev/SDMetrics/issues/254) by @katxiao
* Diagnostic Report missing RangeCoverage for numerical columns - Issue [#255](https://github.com/sdv-dev/SDMetrics/issues/255) by @katxiao

## v0.8.0 - 2022-11-02

This release introduces the `DiagnosticReport`, which helps a user verify – at a quick glance – that their data is valid. We also fix an existing bug with detection metrics.

### New Features
* Fixes for new metadata - Issue [#253](https://github.com/sdv-dev/SDMetrics/issues/253) by @katxiao
* Add default synthetic sample size to DiagnosticReport - Issue [#248](https://github.com/sdv-dev/SDMetrics/issues/248) by @katxiao
* Exclude pii columns from single table metrics - Issue [#245](https://github.com/sdv-dev/SDMetrics/issues/245) by @katxiao
* Accept both old and new metadata - Issue [#244](https://github.com/sdv-dev/SDMetrics/issues/244) by @katxiao
* Address Diagnostic Report and metric edge cases - Issue [#243](https://github.com/sdv-dev/SDMetrics/issues/243) by @katxiao
* Update visualization average per table - Issue [#242](https://github.com/sdv-dev/SDMetrics/issues/242) by @katxiao
* Add save and load functionality to multi-table DiagnosticReport - Issue [#218](https://github.com/sdv-dev/SDMetrics/issues/218) by @katxiao
* Visualization methods for the multi-table DiagnosticReport - Issue [#217](https://github.com/sdv-dev/SDMetrics/issues/217) by @katxiao
* Add getter methods to multi-table DiagnosticReport - Issue [#216](https://github.com/sdv-dev/SDMetrics/issues/216) by @katxiao
* Create multi-table DiagnosticReport - Issue [#215](https://github.com/sdv-dev/SDMetrics/issues/215) by @katxiao
* Visualization methods for the single-table DiagnosticReport - Issue [#211](https://github.com/sdv-dev/SDMetrics/issues/211) by @katxiao
* Add getter methods to single-table DiagnosticReport - Issue [#210](https://github.com/sdv-dev/SDMetrics/issues/210) by @katxiao
* Create single-table DiagnosticReport - Issue [#209](https://github.com/sdv-dev/SDMetrics/issues/209) by @katxiao
* Add save and load functionality to single-table DiagnosticReport - Issue [#212](https://github.com/sdv-dev/SDMetrics/issues/212) by @katxiao
* Add single table diagnostic report - Issue [#237](https://github.com/sdv-dev/SDMetrics/issues/237) by @katxiao

### Bug Fixes
* Detection test test doesn't look at metadata when determining which columns to use - Issue [#119](https://github.com/sdv-dev/SDMetrics/issues/119) by @R-Palazzo

### Internal Improvements
* Remove torch dependency - Issue [#233](https://github.com/sdv-dev/SDMetrics/issues/233) by @katxiao
* Update README - Issue [#250](https://github.com/sdv-dev/SDMetrics/issues/250) by @katxiao


## v0.7.0 - 2022-09-27

This release introduces the `QualityReport`, which evaluates how well synthetic data captures mathematical properties from the real data. The `QualityReport` incorporates the new metrics introduced in the previous release, and allows users to get detailed results, visualize the scores, and save the report for future viewing. We also add utility methods for visualizing columns and pairs of columns.

### New Features
* Catch typeerror in new row synthesis query - Issue [#234](https://github.com/sdv-dev/SDMetrics/issues/234) by @katxiao
* Add NewRowSynthesis Metric - Issue [#207](https://github.com/sdv-dev/SDMetrics/issues/207) by @katxiao
* Update plot utilities API - Issue [#228](https://github.com/sdv-dev/SDMetrics/issues/228) by @katxiao
* Fix column pairs visualization bug - Issue [#230](https://github.com/sdv-dev/SDMetrics/issues/230) by @katxiao
* Save version - Issue [#229](https://github.com/sdv-dev/SDMetrics/issues/229) by @katxiao
* Update efficacy metrics API - Issue [#227](https://github.com/sdv-dev/SDMetrics/issues/227) by @katxiao
* Add RangeCoverage Metric - Issue [#208](https://github.com/sdv-dev/SDMetrics/issues/208) by @katxiao
* Add get_column_pairs_plot utility method - Issue [#223](https://github.com/sdv-dev/SDMetrics/issues/223) by @katxiao
* Parse date as datetime - Issue [#222](https://github.com/sdv-dev/SDMetrics/issues/222) by @katxiao
* Update error handling for reports - Issue [#221](https://github.com/sdv-dev/SDMetrics/issues/221) by @katxiao
* Visualization API update - Issue [#220](https://github.com/sdv-dev/SDMetrics/issues/220) by @katxiao
* Bug fixes for QualityReport - Issue [#219](https://github.com/sdv-dev/SDMetrics/issues/219) by @katxiao
* Update column pair metric calculation - Issue [#214](https://github.com/sdv-dev/SDMetrics/issues/214) by @katxiao
* Add get score methods for multi table QualityReport - Issue [#190](https://github.com/sdv-dev/SDMetrics/issues/190) by @katxiao
* Add multi table QualityReport visualization methods - Issue [#192](https://github.com/sdv-dev/SDMetrics/issues/192) by @katxiao
* Add plot_column visualization utility method - Issue [#193](https://github.com/sdv-dev/SDMetrics/issues/193) by @katxiao
* Add save and load behavior to multi table QualityReport - Issue [#188](https://github.com/sdv-dev/SDMetrics/issues/188) by @katxiao
* Create multi-table QualityReport - Issue [#186](https://github.com/sdv-dev/SDMetrics/issues/186) by @katxiao
* Add single table QualityReport visualization methods - Issue [#191](https://github.com/sdv-dev/SDMetrics/issues/191) by @katxiao
* Add save and load behavior to single table QualityReport - Issue [#187](https://github.com/sdv-dev/SDMetrics/issues/187) by @katxiao
* Add get score methods for single table Quality Report - Issue [#189](https://github.com/sdv-dev/SDMetrics/issues/189) by @katxiao
* Create single-table QualityReport - Issue [#185](https://github.com/sdv-dev/SDMetrics/issues/185) by @katxiao

### Internal Improvements
* Auto apply "new" label instead of "pending review" - Issue [#164](https://github.com/sdv-dev/SDMetrics/issues/164) by @katxiao
* fix typo - Issue [#195](https://github.com/sdv-dev/SDMetrics/issues/195) by @fealho


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
