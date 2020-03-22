from unittest import TestCase

from parameterized import parameterized
from sdv import SDV, load_demo

from sdmetrics import evaluate
from sdmetrics.datasets import Dataset, list_datasets


class TestSDMetrics(TestCase):

    def test_integration(self):
        metadata, tables = load_demo(metadata=True)

        sdv = SDV()
        sdv.fit(metadata, tables)
        synthetic = sdv.sample_all(20)

        metrics = evaluate(metadata, tables, synthetic)
        metrics.overall()
        metrics.details()
        metrics.highlights()

    @parameterized.expand(list_datasets())
    def test_data_driven(self, dataset):
        dataset = Dataset.load(dataset)
        hq_report = evaluate(dataset.metadata, dataset.tables, dataset.hq_synthetic)
        lq_report = evaluate(dataset.metadata, dataset.tables, dataset.lq_synthetic)
        assert hq_report.overall() > lq_report.overall()
