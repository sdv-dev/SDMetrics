import pandas as pd
import pytest

from sdmetrics import evaluate
from sdmetrics.datasets import Dataset, list_datasets


@pytest.mark.parametrize('dataset', list_datasets())
def test_sdmetrics(dataset):
    dataset = Dataset.load(dataset)
    hq_report = evaluate(dataset.metadata, dataset.tables, dataset.hq_synthetic)
    lq_report = evaluate(dataset.metadata, dataset.tables, dataset.lq_synthetic)

    details = hq_report.details()
    assert isinstance(details, pd.DataFrame)
    assert len(details.columns) == 7

    highlights = hq_report.highlights()
    assert isinstance(highlights, pd.DataFrame)

    assert hq_report.overall() > lq_report.overall()
