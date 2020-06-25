from unittest import TestCase

from sdmetrics.report import Goal, Metric, MetricsReport


class TestMetricsReport(TestCase):

    def test_report(self):
        report = MetricsReport()

        report.add_metric(Metric(
            name="one", value=10.0, tags=set(["a", "b"]), goal=Goal.MINIMIZE))
        assert report.overall() == -10.0
        assert len(report.details()) == 1
        assert len(report.details(lambda metric: False)) == 0

        report.add_metric(
            Metric(
                name="two",
                value=3.0,
                tags=set(["a"])))
        assert report.overall() == -10.0
        assert len(report.details()) == 2
        assert len(report.details(lambda metric: "a" in metric.tags)) == 2
        assert len(report.details(lambda metric: "b" in metric.tags)) == 1

        report.add_metric(
            Metric(
                name="three",
                value=5.0,
                goal=Goal.MAXIMIZE,
                tags=set(["priority:high"])))
        assert report.overall() == -5.0
        assert len(report.highlights()) == 1
