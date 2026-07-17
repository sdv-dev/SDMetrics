# Historical EqualizedOdds Metric Experiments

This document provides a historical record of experiments that were conducted to test and validate the `EqualizedOddsImprovement` metric in SDMetrics. These experiments have since been removed from the codebase and can only be accessed through the git history. Thery are documented here to avoid repeating similar approaches in the future.

## Background

The EqualizedOdds metric measures fairness by ensuring that the true positive rate and false positive rate are similar across different groups defined by a sensitive attribute. The historical experiments documented below all used the Adult dataset with:

- **Target variable**: `income` (where `'>50K'` is the positive class)
- **Sensitive attribute**: `sex` (potential gender-based bias)

These experiments were removed because they either failed to achieve a EqualizedOddsImprovement score of over 0.5, or because the experiments were deemed flawed.

## Historical Experiment Overview

### Experiment 1: Basic SDV Synthesis with Conditional Sampling

**Objective**: Test whether conditional sampling could reduce bias in synthetic data. This is detailed in GitHub Issue #776.

**Methodology**:

1. Split the Adult dataset from single-table demo datasets into training and test sets
2. Ensured both sets contained all combinations of prediction target and sensitive attributes
3. Trained an SDV synthesizer (TVAESynthesizer) on the training set
4. Generated synthetic data and measured EqualizedOddsImprovement against real data
5. Applied conditional sampling to generate balanced synthetic data:
   - 25% with `income='>50K'` and `sex='Female'`
   - 25% with `income='<=50K'` and `sex='Male'`
   - 25% with `income='>50K'` and `sex='Female'`
   - 25% with `income='<=50K'` and `sex='Male'`
6. Compared results between regular and conditionally sampled synthetic data

### Experiment 2: Artificially Introducing Bias

**Objective**: Create a more biased dataset to better demonstrate the metric's effectiveness.

**Methodology**:

1. Started with setup from Experiment 1
2. Introduced artificial bias by flipping income values for `sex='Female'` rows
3. Tested synthetic data generation and conditional sampling on this biased dataset

**Original Hypothesis**:

- Sex would become a significant factor in income prediction
- Baseline equalized odds would be poor
- Synthetic data, especially conditionally sampled, would show improvement

### Experiment 3: Refined Bias Introduction

**Objective**: Create a bias scenario where the positive class remained minority.

**Methodology**:

1. Started with setup from Experiment 1
2. For `sex='Male'` rows only:
   - If `salary='<=50K'`, flipped to `'>50K'` with 25% probability
   - If `salary='>50K'`, kept unchanged
3. Kept `sex='Female'` rows unchanged
4. Ran EqualizedOddsImprovement metric with specified parameters

**Original Hypothesis**: Make `'>50K'` several times more likely for males while keeping it as a minority label overall and making females the under-represented group in high earnings.

### Experiment 4: Training/Validation Imbalance

**Objective**: Test the metric with imbalanced training data but fair validation data.

**Methodology**:

1. Created class imbalance in training data
2. Converted validation data into a fair set where the number of high/low earners was equal across Female/Male groups
3. Measured how synthetic data performed under these conditions
