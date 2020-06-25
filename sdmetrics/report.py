# -*- coding: utf-8 -*-

"""MetricsReport module.

This module defines the classes Goal, Metric and MetricsReport, which
are used for reporting the results of the different evaluation
metrics executed on the data.
"""

from enum import Enum

import pandas as pd


class Goal(Enum):
    """
    This enumerates the `goal` for a metric; the value of a metric can be ignored,
    minimized, or maximized.
    """

    IGNORE = "ignore"
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Metric():
    """
    This represents a single instance of a Metric.

    Attributes:
        name (str): The name of the attribute.
        value (float): The value of the attribute.
        tags (set(str)): A set of arbitrary strings/tags for the attribute.
        goal (Goal): Whether the value should maximized, minimized, or ignored.
        unit (str): The "unit" of the metric (i.e. p-value, entropy, mean-squared-error).
        domain (tuple): The range of values the metric can take on.
        description (str): An arbitrary text description of the attribute.
    """

    def __init__(self, name, value, tags=None, goal=Goal.IGNORE,
                 unit="", domain=(float("-inf"), float("inf")), description=""):
        self.name = name
        self.value = value
        self.tags = tags if tags else set()
        self.goal = goal
        self.unit = unit
        self.domain = domain
        self.description = description
        self._validate()

    def _validate(self):
        assert isinstance(self.name, str)
        assert isinstance(self.value, float)
        assert isinstance(self.tags, set)
        assert isinstance(self.goal, Goal)
        assert isinstance(self.unit, str)
        assert isinstance(self.domain, tuple)
        assert isinstance(self.description, str)
        assert self.domain[0] <= self.value and self.value <= self.domain[1]
        assert all(isinstance(t, str) for t in self.tags)

    def __eq__(self, other):
        my_attrs = (self.name, self.value, self.goal, self.unit)
        your_attrs = (other.name, other.value, other.objective, self.unit)
        return my_attrs == your_attrs

    def __hash__(self):
        return hash(self.name) + hash(self.value)

    def __str__(self):
        return """Metric(\n  name=%s, \n  value=%.2f, \n  tags=%s, \n  description=%s\n)""" % (
            self.name, self.value, self.tags, self.description)


class MetricsReport():
    """
    The `MetricsReport` object is responsible for storing metrics and providing a user
    friendly API for accessing them.
    """

    def __init__(self):
        self.metrics = []

    def add_metric(self, metric):
        """
        This adds the given `Metric` object to this report.
        """
        assert isinstance(metric, Metric)
        self.metrics.append(metric)

    def add_metrics(self, iterator):
        """
        This takes an iterator which yields `Metric` objects and adds all
        of these metrics to this report.
        """
        for metric in iterator:
            self.add_metric(metric)

    def overall(self):
        """
        This computes a single scalar score for this report. To produce higher quality
        synthetic data, the model should try to maximize this score.

        Returns:
            float: The scalar value to maximize.
        """
        score = 0.0
        for metric in self.metrics:
            if metric.goal == Goal.MAXIMIZE:
                score += metric.value
            elif metric.goal == Goal.MINIMIZE:
                score -= metric.value
        return score

    def details(self, filter_func=None):
        """
        This returns a DataFrame containing all of the metrics in this report. You can
        optionally use `filter_func` to specify a lambda function which takes in the
        metric and returns True if it should be included in the output.

        Args:
            filter_func (function, optional): A function which takes a Metric object
                and returns True if it should be included. Defaults to accepting all
                Metric objects.

        Returns:
            DataFrame: A table listing all the (selected) metrics.
        """
        if not filter_func:
            def filter_func(metric):
                return True
        rows = []
        for metric in self.metrics:
            if not filter_func(metric):
                continue
            table_tags = [tag for tag in metric.tags if "table:" in tag]
            column_tags = [tag for tag in metric.tags if "column:" in tag]
            misc_tags = metric.tags - set(table_tags) - set(column_tags)
            rows.append({
                "Name": metric.name,
                "Value": metric.value,
                "Goal": metric.goal,
                "Unit": metric.unit,
                "Tables": ",".join(table_tags),
                "Columns": ",".join(column_tags),
                "Misc. Tags": ",".join(misc_tags),
            })
        return pd.DataFrame(rows)

    def highlights(self):
        """
        This returns a DataFrame containing all of the metrics in this report which
        contain the "priority:high" tag.

        Returns:
            DataFrame: A table listing all the high-priority metrics.
        """
        return self.details(lambda metric: "priority:high" in metric.tags)

    def visualize(self):
        """
        This returns a pyplot.Figure which shows some of the key metrics.

        Returns:
            pyplot.Figure: A matplotlib figure visualizing key metricss.
        """
        from matplotlib import rcParams
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['DejaVu Sans']

        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.style.use('seaborn')

        fig = plt.figure(figsize=(10, 12), constrained_layout=True)
        gs = fig.add_gridspec(5, 4)

        # Detectability of synthetic tables
        fig.add_subplot(gs[3:, :])
        labels, scores = [], []
        for metric in self.metrics:
            tables = [tag.replace("table:", "")
                      for tag in metric.tags if "table:" in tag]
            labels.append(" <-> ".join(tables))
            scores.append(metric.value)
        df = pd.DataFrame({"score": scores, "label": labels})
        df = df.groupby("label").agg({"score": "mean"}).reset_index()
        df = df.sort_values(["score"], ascending=False)
        df = df.head(4)
        sns.barplot(
            x="label",
            y="score",
            data=df,
            ci=None,
            palette=sns.color_palette(
                "coolwarm_r",
                7))
        plt.axhline(0.9, color="red", linestyle=":", label="Easy To Detect")
        plt.axhline(0.7, color="green", linestyle=":", label="Hard To Detect")
        plt.legend(loc="lower right")
        plt.title("Detectability of Synthetic Tables", fontweight='bold')
        plt.ylabel("auROC")
        plt.xlabel("")

        # Coming soon.
        fig.add_subplot(gs[1:3, 2:])
        pvalues = np.array([m.value for m in self.metrics if m.unit == "p-value"])
        sizes = [np.sum(pvalues < 0.1), np.sum(pvalues > 0.1)]
        labels = ['Reject (p<0.1)', 'Fail To Reject']
        plt.pie(sizes, labels=labels)
        plt.axis('equal')
        plt.title("Columnwise Statistical Tests", fontweight='bold')
        plt.ylabel("")
        plt.xlabel("")

        # Coming soon.
        fig.add_subplot(gs[:3, :2])
        labels, scores = [], []
        for metric in self.metrics:
            if metric.unit != "entropy":
                continue
            for tag in metric.tags:
                if "column:" not in tag:
                    continue
                labels.append(tag.replace("column:", ""))
                scores.append(metric.value)
        df = pd.DataFrame({"score": scores, "label": labels})
        df = df.groupby("label").agg({"score": "mean"}).reset_index()
        df = df.sort_values(["score"], ascending=False)
        df = df.head(8)
        sns.barplot(x="score", y="label", data=df, ci=None, palette=sns.color_palette("Blues_d"))
        plt.title("Column Divergence", fontweight='bold')
        plt.ylabel("")
        plt.xlabel("")

        # Coming soon.
        fig.add_subplot(gs[:1, 2:])
        plt.text(0.5, 0.7, r'Overall Score', fontsize=14, fontweight='bold', ha="center")
        plt.text(0.5, 0.4, r'%.2f' % self.overall(), fontsize=36, ha="center")
        rectangle = plt.Rectangle((0.2, 0.3), 0.6, 0.6, ec='black', fc='white')
        plt.gca().add_patch(rectangle)
        plt.ylabel("")
        plt.xlabel("")
        plt.axis('off')

        fig.tight_layout(pad=2.0)
        return fig
