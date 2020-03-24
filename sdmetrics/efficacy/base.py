import numpy as np
from sklearn.model_selection import KFold

from sdmetrics import Goal, Metric


class MLEfficacy():

    name = ""
    target_table = ""
    target_column = ""

    metric_unit = ""
    metric_goal = Goal.IGNORE
    metric_domain = (float("-inf"), float("inf"))

    def fit(self, X, y):
        """This function implements a fit procedure which trains a binary
        classification model where class=1 indicates the data is synthetic
        and class=0 indicates that the data is real.

        Arguments:
            X (np.ndarray): The numerical features (i.e. transformed rows).
            y (np.ndarray): The binary classification target.
        """
        raise NotImplementedError()

    def score(self, X, y):
        """This function scores this model on the (test) dataset.

        Arguments:
            X (np.ndarray): The numerical features (i.e. transformed rows).
            y (np.ndarray): The binary classification target.

        Returns:
            float: The value of the appropriate metric.
        """
        raise NotImplementedError()

    def metrics(self, metadata, real_tables, synthetic_tables):
        real_table = real_tables[self.target_table]
        synthetic_table = synthetic_tables[self.target_table]
        delta_score, synthetic_score = self._cross_val_score(real_table, synthetic_table)

        # Score on the synthetic table. Evaluated on real.
        yield Metric(
            name=self.name,
            value=synthetic_score,
            tags=set([
                "efficacy:ml",
                "table:%s" % self.target_table,
                "column:%s" % self.target_column,
            ]),
            goal=self.metric_goal,
            unit=self.metric_unit,
            domain=self.metric_domain,
            description="Score on the real test set when trained on synthetic data."
        )

        # Score on synthetic minus score on real. Evaluated on real.
        domain = self.metric_domain[1] - self.metric_domain[0]
        yield Metric(
            name=self.name,
            value=delta_score,
            tags=set([
                "efficacy:ml",
                "table:%s" % self.target_table,
                "column:%s" % self.target_column,
            ]),
            goal=self.metric_goal,
            unit="delta_%s" % self.metric_unit,
            domain=(-domain, domain),
            description="Diff in score on real when trained on synthetic vs real."
        )

    def _cross_val_score(self, real_table, synthetic_table):
        real_X = real_table.loc[:, real_table.columns != self.target_column].values
        real_y = real_table[self.target_column].values

        synthetic_X = synthetic_table.loc[:, synthetic_table.columns != self.target_column].values
        synthetic_y = synthetic_table[self.target_column].values

        delta_scores = []
        synthetic_scores = []
        kf = KFold(n_splits=3, shuffle=True)

        for train_index, test_index in kf.split(real_X, real_y):
            # Train a model on the real dataset.
            self.fit(real_X[train_index], real_y[train_index])
            real_score = self.score(real_X[test_index], real_y[test_index])

            # Train a model on the synthetic dataset.
            self.fit(synthetic_X, synthetic_y)
            synthetic_score = self.score(real_X[test_index], real_y[test_index])

            delta_scores.append(synthetic_score - real_score)
            synthetic_scores.append(synthetic_score)

        return np.mean(delta_scores), np.mean(synthetic_scores)
