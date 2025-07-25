# History

## v0.22.0 - 2025-07-24

### New Features

* Add a fairness metric that computes Equalized Odds - Issue [#772](https://github.com/sdv-dev/SDMetrics/issues/772) by @fealho
* Add a violin plot visualizations to compare a pair of columns - Issue [#759](https://github.com/sdv-dev/SDMetrics/issues/759) by @R-Palazzo

### Internal

* Fix test failures with pandas 2.3 - Issue [#773](https://github.com/sdv-dev/SDMetrics/issues/773) by @rwedge
* Check pyproject for release candidate dependencies - Issue [#770](https://github.com/sdv-dev/SDMetrics/issues/770) by @rwedge

### Maintenance

* Update python setup step in workflows to use latest python version - Issue [#653](https://github.com/sdv-dev/SDMetrics/issues/653) by @frances-h

### Miscellaneous

* Add workflow to release SDMetrics on PyPI - Issue [#783](https://github.com/sdv-dev/SDMetrics/issues/783) by @gsheni

## v0.21.0 - 2025-05-29

### New Features

* Add a violin plot visualizations to compare a pair of columns - Issue [#759](https://github.com/sdv-dev/SDMetrics/issues/759) by @R-Palazzo

## v0.20.1 - 2025-04-14

### Bugs Fixed

* Importing sdmetrics crashes because of torch dependency - Issue [#764](https://github.com/sdv-dev/SDMetrics/issues/764) by @amontanez24

## v0.20.0 - 2025-04-11

### New Features

* Consolidate handling of datetime columns - Issue [#741](https://github.com/sdv-dev/SDMetrics/issues/741) by @frances-h
* Improve handling of datetime columns for metrics - Issue [#740](https://github.com/sdv-dev/SDMetrics/issues/740) by @lajohn4747
* DCR metrics should allow the possibility of constant numerical columns - Issue [#739](https://github.com/sdv-dev/SDMetrics/issues/739) by @lajohn4747
* Add `DCRBaselineProtection` metric - Issue [#720](https://github.com/sdv-dev/SDMetrics/issues/720)
* Add `DCROverfittingProtection` metric - Issue [#719](https://github.com/sdv-dev/SDMetrics/issues/719) by @lajohn4747

### Bugs Fixed

* Visualization `get_column_plot` crashes if I have a column of all null values - Issue [#757](https://github.com/sdv-dev/SDMetrics/issues/757) by @lajohn4747
* The `DCRBaselineProtection` metric is not creating the correct size of random data - Issue [#743](https://github.com/sdv-dev/SDMetrics/issues/743) by @lajohn4747
* The `DCRBaselineProtection` metric is not producing the expected score - Issue [#742](https://github.com/sdv-dev/SDMetrics/issues/742) by @lajohn4747
* The DCRBaselineProtection metric crashes when the distance between random data and real data is 0 - Issue [#738](https://github.com/sdv-dev/SDMetrics/issues/738) by @lajohn4747
* DCROverfitting and DCRBaseline metrics produce too many warnings about missing columns. - Issue [#737](https://github.com/sdv-dev/SDMetrics/issues/737) by @lajohn4747
* Wrong default values for the `classifier` parameter of the `BinaryClassifierPrecisionEfficacy` - Issue [#734](https://github.com/sdv-dev/SDMetrics/issues/734) by @R-Palazzo

### Maintenance

* Upgrade `pomegranate` dependency to Python 3.13 - Issue [#717](https://github.com/sdv-dev/SDMetrics/issues/717) by @R-Palazzo

## v0.19.0 - 2025-02-24

### New Features

* Use metadata to validate inputs to `BinaryClassifierEfficacy` precision and recall metrics - Issue [#727](https://github.com/sdv-dev/SDMetrics/issues/727) by @R-Palazzo
* Speed up calculation of the QualityReport - Issue [#718](https://github.com/sdv-dev/SDMetrics/issues/718) by @R-Palazzo
* Allow subsampling when computing the `ContingencySimilarity` metric - Issue [#716](https://github.com/sdv-dev/SDMetrics/issues/716) by @R-Palazzo
* Add `BinaryClassifierRecallEfficacy` metric - Issue [#712](https://github.com/sdv-dev/SDMetrics/issues/712) by @R-Palazzo
* Add `BinaryClassifierPrecisionEfficacy` metric - Issue [#711](https://github.com/sdv-dev/SDMetrics/issues/711) by @R-Palazzo
* The `ContingencySimilarity` metric should be able to discretize continuous columns - Issue [#700](https://github.com/sdv-dev/SDMetrics/issues/700)

### Maintenance

* Delete conda folder - Issue [#708](https://github.com/sdv-dev/SDMetrics/issues/708) by @amontanez24
* Cancel previous workflow runs on a new push - Issue [#705](https://github.com/sdv-dev/SDMetrics/issues/705) by @R-Palazzo
* Support Python 3.13 (excluding `pomegranate`) - Issue [#645](https://github.com/sdv-dev/SDMetrics/issues/645) by @rwedge

### Miscellaneous

* BNLikelihood - Issue [#630](https://github.com/sdv-dev/SDMetrics/issues/630)

## v0.18.0 - 2024-12-13

### Bugs Fixed

* Missing whitespace in `DisclosureProtection` warning - Issue [#694](https://github.com/sdv-dev/SDMetrics/issues/694) by @frances-h
* `DisclosureProtection` should be NaN if baseline score is zero - Issue [#693](https://github.com/sdv-dev/SDMetrics/issues/693) by @frances-h
* `CategoricalCAP` metric returns 0 if no overlap in known fields - Issue [#692](https://github.com/sdv-dev/SDMetrics/issues/692) by @frances-h

### New Features

* Add `DisclosureProtectionEstimate` metric - Issue [#676](https://github.com/sdv-dev/SDMetrics/issues/676) by @frances-h
* Add `DisclosureProtection` metric - Issue [#675](https://github.com/sdv-dev/SDMetrics/issues/675) by @frances-h

## v0.17.1 - 2024-12-04

### Maintenance

* Create Prepare Release workflow - Issue [#674](https://github.com/sdv-dev/SDMetrics/issues/674) by @amontanez24
* Update codecov and add flag for integration tests - Issue [#644](https://github.com/sdv-dev/SDMetrics/issues/644) by @pvk-developer

### Bugs Fixed

* `InterRowMSAS` ignores sequences with missing values - Issue [#679](https://github.com/sdv-dev/SDMetrics/issues/679) by @fealho
* Improve error handling for datetime values when `apply_log = True` for `InterRowMSAS` - Issue [#672](https://github.com/sdv-dev/SDMetrics/issues/672) by @fealho
* Improve warning handling for non-positive values when `apply_log = True` for `InterRowMSAS` - Issue [#670](https://github.com/sdv-dev/SDMetrics/issues/670) by @fealho
* `StatisticMSAS` raises undesirable `FutureWarning` - Issue [#665](https://github.com/sdv-dev/SDMetrics/issues/665) by @fealho
* `KSComplement` can be unstable for constant float values - Issue [#652](https://github.com/sdv-dev/SDMetrics/issues/652) by @fealho

## v0.17.0 - 2024-11-14

This release adds a number of Multi-Sequence Aggregate Similarity (MSAS) metrics!

### Bugs Fixed

* Relocate timeseries metrics modules - Issue [#661](https://github.com/sdv-dev/SDMetrics/issues/661) by @fealho
* Fix `SequenceLengthSimilarity` docstrings - Issue [#660](https://github.com/sdv-dev/SDMetrics/issues/660) by @fealho
* When running Quality Report, ContingencySimilarity produces a RuntimeWarning (`The values in the array are unorderable.`) - Issue [#656](https://github.com/sdv-dev/SDMetrics/issues/656) by @R-Palazzo

### New Features

* Add metric for inter-row MSAS - Issue [#640](https://github.com/sdv-dev/SDMetrics/issues/640) by @fealho
* Add metric for general MSAS statistics - Issue [#639](https://github.com/sdv-dev/SDMetrics/issues/639) by @fealho
* Add metric for sequence length similarity - Issue [#638](https://github.com/sdv-dev/SDMetrics/issues/638) by @fealho

## v0.16.0 - 2024-09-25

This release improves the performance of the `contingency_similarity` metric. It also factors dtypes into the score of the `TableStructure` metric.

### Internal

* Try to improve performance of contingency_similarity - Issue [#622](https://github.com/sdv-dev/SDMetrics/issues/622) by @amontanez24

### New Features

* Add dtype comparison in `TableStructure` metric (used in Diagnostic report) - Issue [#631](https://github.com/sdv-dev/SDMetrics/issues/631) by @R-Palazzo

## v0.15.1 - 2024-08-13

### Bugs Fixed

* X-axis for the bar plot should be labeled `Value` instead of `Category` - Issue [#620](https://github.com/sdv-dev/SDMetrics/issues/620) by @R-Palazzo
* LinAlgError when plotting data that is constant - Issue [#616](https://github.com/sdv-dev/SDMetrics/issues/616) by @R-Palazzo
* Wrong chart title when generating a box plot for just the real data using `get_column_pair_plot()`  - Issue [#615](https://github.com/sdv-dev/SDMetrics/issues/615) by @R-Palazzo

### New Features

* Better error message when passing an SDV Metadata object - Issue [#610](https://github.com/sdv-dev/SDMetrics/issues/610) by @R-Palazzo
* Check that every property score are index-free - Issue [#583](https://github.com/sdv-dev/SDMetrics/issues/583) by @R-Palazzo

## v0.15.0 - 2024-07-15

This release adds support for NumPy 2.0! Additionally, the visualization utilities no longer require both real and synthetic data to be provided, and they can now be used to visualize only real or only synthetic data.

### Maintenance

* Switch to using ruff for Python linting and code formatting - Issue [#536](https://github.com/sdv-dev/SDMetrics/issues/536) by @gsheni
* Change job names in integration workflow to "integration" - Issue [#577](https://github.com/sdv-dev/SDMetrics/issues/577) by @rwedge
* Cap numpy to less than 2.0.0 until SDMetrics supports - Issue [#591](https://github.com/sdv-dev/SDMetrics/issues/591) by @gsheni

### Internal

* Switch to using ruff for Python linting and code formatting - Issue [#536](https://github.com/sdv-dev/SDMetrics/issues/536) by @gsheni

### New Features

* Allow me to visualize just the real or synthetic data - Issue [#581](https://github.com/sdv-dev/SDMetrics/issues/581) by @lajohn4747
* Update Referential Integrity metric to support NaNs in child column  - Issue [#587](https://github.com/sdv-dev/SDMetrics/issues/587) by @R-Palazzo
* Add support for numpy 2.0.0 - Issue [#593](https://github.com/sdv-dev/SDMetrics/issues/593) by @R-Palazzo

### Bugs Fixed

* ColumnPairTrends score depends on the data index - Issue [#582](https://github.com/sdv-dev/SDMetrics/issues/582) by @R-Palazzo
* Datetime columns set to Object pandas dtype breaks LSTMDetection - Issue [#584](https://github.com/sdv-dev/SDMetrics/issues/584) by @fealho

## v0.14.1 - 2024-05-13

This release patches a bug on the `LSTMDetection` metric.

### Bugs Fixed

* `LSTMDetection` metric crashes when there are multiple context columns - Issue [#298](https://github.com/sdv-dev/SDMetrics/issues/298) by @frances-h

### Maintenance

* Cleanup automated PR workflows - Issue [#566](https://github.com/sdv-dev/SDMetrics/issues/566) by @R-Palazzo
* Only run unit and integration tests on oldest and latest python versions for macos - Issue [#569](https://github.com/sdv-dev/SDMetrics/issues/569) by @R-Palazzo

## v0.14.0 - 2024-04-11

This release adds support for Python 3.12! It also improves the way the reports print in verbose mode.

### Maintenance

* Support Python 3.12 - Issue [#529](https://github.com/sdv-dev/SDMetrics/issues/529) by @fealho
* Add dependency checker - Issue [#547](https://github.com/sdv-dev/SDMetrics/issues/547) by @lajohn4747
* Add bandit workflow - Issue [#552](https://github.com/sdv-dev/SDMetrics/issues/552) by @R-Palazzo
* Fix minimum version workflow when pointing to github branch - Issue [#555](https://github.com/sdv-dev/SDMetrics/issues/555) by @R-Palazzo

### New Features

* Improve readability of the report scores when verbosity is on - Issue [#538](https://github.com/sdv-dev/SDMetrics/issues/538) by @lajohn4747

## v0.13.1 - 2024-03-14

### Maintenance

* Transition from using setup.py to pyroject.toml to specify project metadata - Issue [#534](https://github.com/sdv-dev/SDMetrics/issues/534) by @lajohn4747
* Remove bumpversion and use bump-my-version - Issue [#535](https://github.com/sdv-dev/SDMetrics/issues/535) by @R-Palazzo
* Add support for Copulas 0.10 - Issue [#541](https://github.com/sdv-dev/SDMetrics/issues/541) by @amontanez24

## v0.13.0 - 2023-12-04

This release makes significant improvements to the Diagnostic Reports! The report now runs a diagnostic to calculate scores for three basic but important properties of your data: data validity, data structure and in the multi table case, relationship validity. Data validity checks that the columns of your data are valid (eg. correct range or values). Data structure makes sure the synthetic data has the correct columns. Relationship validity checks to make sure key references are correct and the cardinality is within ranges seen in the real data. These changes are meant to make the `DiagnosticReport` a quick way for you to see if there are any major problems with your synthetic data.

Additionally, some general improvements were made and bugs were resolved. The `LogisticDetection` and `SVCDetection` metrics were fixed to only use boolean, categorical, datetime and numeric columns in their calculations. A bug that prevented visualizations from displaying on Jupyter notebooks was patched. The cardinality property in the multi table `QualityReport` can now handle multiple foreign keys to the same parent. Finally, a new visualization was added for sequential/timeseries data called `get_column_line_plot`.

### New Features

* Detection metrics should only use statistically modeled columns (filter out the rest) - Issue [#286](https://github.com/sdv-dev/SDMetrics/issues/286) by @lajohn4747
* Add visualization for timeseries / sequential data  - Issue [#376](https://github.com/sdv-dev/SDMetrics/issues/376) by @lajohn4747
* Multi table quality report should handle multi-foreign keys (to same parent) - Issue [#406](https://github.com/sdv-dev/SDMetrics/issues/406) by @R-Palazzo
* Add `KeyUniqueness` metric - Issue [#460](https://github.com/sdv-dev/SDMetrics/issues/460) by @R-Palazzo
* Add `ReferentialIntegrity` metric - Issue [#461](https://github.com/sdv-dev/SDMetrics/issues/461) by @R-Palazzo
* Add `CategoryAdherence` metric - Issue [#462](https://github.com/sdv-dev/SDMetrics/issues/462) by @R-Palazzo
* Add `TableFormat` metric - Issue [#463](https://github.com/sdv-dev/SDMetrics/issues/463) by @R-Palazzo
* Add `CardinalityBoundaryAdherence` metric - Issue [#464](https://github.com/sdv-dev/SDMetrics/issues/464) by @frances-h
* Add `DataValidity` property - Issue [#467](https://github.com/sdv-dev/SDMetrics/issues/467) by @R-Palazzo
* Add `Structure` property - Issue [#468](https://github.com/sdv-dev/SDMetrics/issues/468) by @R-Palazzo
* Add `Relationship Validity` property - Issue [#469](https://github.com/sdv-dev/SDMetrics/issues/469) by @R-Palazzo
* Update `DiagnosticReport` to calculate base correctness of synthetic data - Issue [#471](https://github.com/sdv-dev/SDMetrics/issues/471) by @R-Palazzo
* Update the synthetic data that's available for the multi-table demo - Issue [#501](https://github.com/sdv-dev/SDMetrics/issues/501) by @R-Palazzo
* Update the synthetic data that's available for the single-table demo - Issue [#502](https://github.com/sdv-dev/SDMetrics/issues/502) by @R-Palazzo
* Update `TableFormat` metric to `TableStructure` + fix its computation - Issue [#518](https://github.com/sdv-dev/SDMetrics/issues/518) by @R-Palazzo

### Bugs Fixed

* Sometimes graphs don't show when using Jupyter notebook - Issue [#322](https://github.com/sdv-dev/SDMetrics/issues/322) by @pvk-developer
* Fix ReferentialIntegrity NaN handling - Issue [#494](https://github.com/sdv-dev/SDMetrics/issues/494) by @R-Palazzo
* KeyUniqueness metric should only be applied to primary and alternate keys - Issue [#503](https://github.com/sdv-dev/SDMetrics/issues/503) by @R-Palazzo
* Single table Structure property should not have visualization - Issue [#504](https://github.com/sdv-dev/SDMetrics/issues/504) by @R-Palazzo
* Multi table Structure property visualization has incorrect styling - Issue [#505](https://github.com/sdv-dev/SDMetrics/issues/505) by @R-Palazzo
* `UserWarning: KeyError: 'relationships'` in DiagnosticReport if metadata missing relationships - Issue [#506](https://github.com/sdv-dev/SDMetrics/issues/506) by @R-Palazzo
* Report `validate` method should be private - Issue [#507](https://github.com/sdv-dev/SDMetrics/issues/507) by @R-Palazzo
* `ValueError` in DiagnosticReport if synthetic data does not match metadata - Issue [#508](https://github.com/sdv-dev/SDMetrics/issues/508) by @R-Palazzo
* Check if QualityReport needs the synthetic data to match the metadata - Issue [#509](https://github.com/sdv-dev/SDMetrics/issues/509) by @R-Palazzo
* Running single table report on multi table data (or vice versa) results in confusing error - Issue [#510](https://github.com/sdv-dev/SDMetrics/issues/510) by @R-Palazzo
* Add metadata validation - Issue [#526](https://github.com/sdv-dev/SDMetrics/issues/526) by @R-Palazzo

## v0.12.1 - 2023-11-01

This release fixes a bug with the new Intertable Trends property and older pandas versions and a bug with how the ML Efficacy metric handled train and test data. Reports handle missing relationships more gracefully.

### Bugs Fixed

* Multiple FutureWarning lines printed out when running the Quality Report (Intertable Trends property) - Issue [#490](https://github.com/sdv-dev/SDMetrics/issues/490) by @frances-h
* Transformer should not be fit on test data - Issue [#291](https://github.com/sdv-dev/SDMetrics/issues/291) by @fealho
* Reports should not crash if there are no relationships - Issue [#481](https://github.com/sdv-dev/SDMetrics/issues/481) by @lajohn4747

## v0.12.0 - 2023-10-31

This release adds a new property, InterTable Trends. Several plots were moved from the reports module into the new visualizations module.  The `metadata` parameter was removed for these plots, and the `plot_types` parameter was added. `plot_types` lets the user control which plot type is used. Several crashes have been resolved.

### New Features

* Provide meta information about the reports - Pull [#472](https://github.com/sdv-dev/SDMetrics/pull/472) by @frances-h
* Validate that the metadata is always a dict - Issue [#428](https://github.com/sdv-dev/SDMetrics/issues/428) by @R-Palazzo
* Expose reports module in top-level init - Pull [#459](https://github.com/sdv-dev/SDMetrics/pull/459) by @frances-h
* Add new get_column_pair_plot - Issue [#444](https://github.com/sdv-dev/SDMetrics/issues/444) by @pvk-developer
* Add InterTable Trends property - Issue [#451](https://github.com/sdv-dev/SDMetrics/issues/451) by @frances-h
* Add new get_column_plot - Issue [#443](https://github.com/sdv-dev/SDMetrics/issues/443) by @pvk-developer
* Add new get_cardinality_plot - Issue [#445](https://github.com/sdv-dev/SDMetrics/issues/445) by @frances-h
* Create visualizations module - Issue [#442](https://github.com/sdv-dev/SDMetrics/issues/442) by @frances-h, @pvk-developer

### Bugs Fixed

* Fix `NewRowSynthesis` on datetime columns without formats - Issue [#473](https://github.com/sdv-dev/SDMetrics/issues/473) by @fealho
* Intertable trends property crashes if a table has no statistical columns - Issue [#476](https://github.com/sdv-dev/SDMetrics/issues/476) by @lajohn4747
* Fix BoundaryAdherence NaN handling - Issue [#470](https://github.com/sdv-dev/SDMetrics/issues/470) by @frances-h
* The Intertable Trends visualization is mislabeled as 'Column Shapes' - Issue [#477](https://github.com/sdv-dev/SDMetrics/issues/477) by @lajohn4747
* ValueError when using get_cardinality_plot on some schemas - Issue [#447](https://github.com/sdv-dev/SDMetrics/issues/447) by @frances-h

### Internal

* Switch default branch from master to main - Issue [#420](https://github.com/sdv-dev/SDMetrics/issues/420) by @amontanez24

## v0.11.1 - 2023-09-14

This release makes multiple changes to better handle errors that get raised from the `DiagnosticReport`. The report should be able to run to completion now and have any errors that it encounters reported in a column on the details that can be observed from running `get_details`. It also resolves many warnings that were interrupting the printing of the report's results and progress.

### New Features

* Create single table coverage property - Issue [#389](https://github.com/sdv-dev/SDMetrics/issues/389) by @R-Palazzo
* Create single table synthesis property - Issue [#390](https://github.com/sdv-dev/SDMetrics/issues/390) by @R-Palazzo
* Create single table Boundaries property - Issue [#391](https://github.com/sdv-dev/SDMetrics/issues/391) by @R-Palazzo
* Add multi table Coverage, Synthesis and Boundaries property - Issue [#393](https://github.com/sdv-dev/SDMetrics/issues/393) by @R-Palazzo

### Bugs Fixed

* Ensure that the `Synthesis` property score doesn't change - Issue [#425](https://github.com/sdv-dev/SDMetrics/issues/425) by @amontanez24
* The Error column contains a mix of `NaN` and `None` values - Issue [#427](https://github.com/sdv-dev/SDMetrics/issues/427) by @pvk-developer
* Always show the `Table` column in `get_details` - Issue [#429](https://github.com/sdv-dev/SDMetrics/issues/429) by @frances-h
* Diagnostic explanations should not repeat if I generate multiple times - Issue [#430](https://github.com/sdv-dev/SDMetrics/issues/430) by @amontanez24
* RangeCoverage errors on datetime columns in DiagnosticReport - Issue [#431](https://github.com/sdv-dev/SDMetrics/issues/431) by @frances-h
* The coverage visualization shows empty bar graph for nan values - Issue [#432](https://github.com/sdv-dev/SDMetrics/issues/432) by @frances-h
* Diagnostic report should skip over all NaN columns - Issue [#433](https://github.com/sdv-dev/SDMetrics/issues/433) by @pvk-developer
* Quality report is printing out a long warning message (hundreds of lines) - Issue [#448](https://github.com/sdv-dev/SDMetrics/issues/448) by @amontanez24

### Internal

* Use property classes in single table DiagnosticReport - Issue [#392](https://github.com/sdv-dev/SDMetrics/issues/392) by @R-Palazzo
* Use property classes in multi table DiagnosticReport - Issue [#394](https://github.com/sdv-dev/SDMetrics/issues/394) by @R-Palazzo

## v0.11.0 - 2023-08-10

This release adds a function that allows users to plot the cardinality of foreign and primary keys in synthetic data. More specifically, it graphs the frequency that each number of children per parent row occurs in the parent table.

Additionally, architectural changes are made to improve the efficiency and error handling of the `QualityReport`! The progress bar is also enhanced to be more informative when the report is generating.

This release also adds support for Python 3.11 and drops support for Python 3.7.

### New Features

* Visualize cardinality of foreign key columns - Issue [#283](https://github.com/sdv-dev/SDMetrics/issues/283) by @R-Palazzo
* Create single table BaseProperty class - Issue [#354](https://github.com/sdv-dev/SDMetrics/issues/354) by @amontanez24
* Create single table column shapes property - Issue [#355](https://github.com/sdv-dev/SDMetrics/issues/355) by @R-Palazzo
* Create single table column pair trends property - Issue [#356](https://github.com/sdv-dev/SDMetrics/issues/356) by @R-Palazzo
* Create multi table BaseProperty class - Issue [#357](https://github.com/sdv-dev/SDMetrics/issues/357) by @pvk-developer
* Create multi table column shapes and column pair trends properties - Issue [#358](https://github.com/sdv-dev/SDMetrics/issues/358) by @R-Palazzo
* Create Parent Child Relationships property class - Issue [#359](https://github.com/sdv-dev/SDMetrics/issues/359) by @pvk-developer
* In Multi Table Quality Report: Rename "Table Relationships" property to "Cardinality" - Issue [#360](https://github.com/sdv-dev/SDMetrics/issues/360) by @frances-h
* More accurate progress bar for single table Quality Report - Issue [#361](https://github.com/sdv-dev/SDMetrics/issues/361) by @R-Palazzo
* More accurate progress bar for multi table Quality Report - Issue [#362](https://github.com/sdv-dev/SDMetrics/issues/362) by @fealho
* Raise error in CorrelationSimilarity if either column is constant - Issue [#407](https://github.com/sdv-dev/SDMetrics/issues/407) by @fealho

### Bug Fixes

* Issue in building the denormalized table inside the Parent-Child Detection metrics - Issue [#328](https://github.com/sdv-dev/SDMetrics/issues/328) by @fealho
* Don't modify the rounding in the quality report - Issue [#401](https://github.com/sdv-dev/SDMetrics/issues/401) by @R-Palazzo
* The Cardinality property is missing some relationships - Issue [#404](https://github.com/sdv-dev/SDMetrics/issues/404) by @pvk-developer
* The Cardinality property is not returning a DataFrame - Issue [#405](https://github.com/sdv-dev/SDMetrics/issues/405) by @fealho
* Overall property score should be the average across all breakdowns - Issue [#415](https://github.com/sdv-dev/SDMetrics/issues/415) by @amontanez24

### Internal

* Use property classes in single table QualityReport - Issue [#370](https://github.com/sdv-dev/SDMetrics/issues/370) by @R-Palazzo
* Use property classes in multi table QualityReport - Issue [#371](https://github.com/sdv-dev/SDMetrics/issues/371) by @fealho
* Add add-on detection for premium metrics - Issue [#388](https://github.com/sdv-dev/SDMetrics/issues/388) by @amontanez24

### Maintenance

* Add support for Python 3.11 - Issue [#353](https://github.com/sdv-dev/SDMetrics/issues/353) by @amontanez24
* Drop support for Python 3.7 - Issue [#380](https://github.com/sdv-dev/SDMetrics/issues/380) by @amontanez24

## v0.10.1 - 2023-06-06

This release fixes a bug that was causing the `DiagnosticReport` to crash on the `NewRowSynthesis` metric. It also adds support for PyTorch 2.0!

### Bug Fixes

* ValueError: multi-line expressions (NewRowSynthesis metric in DiagnosticReport) - Issue [#327](https://github.com/sdv-dev/SDMetrics/issues/327) by @R-Palazzo

### Maintenance

* Upgrade to torch 2.0 - Issue [#347](https://github.com/sdv-dev/SDMetrics/issues/347) by @fealho

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
