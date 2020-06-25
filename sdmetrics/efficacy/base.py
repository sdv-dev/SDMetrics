import numpy as np
from sklearn.model_selection import KFold

from sdmetrics.report import Goal, Metric


class MLEfficacy():

    name = ""
    target_table_name = ""
    target_column_name = ""

    metric_unit = ""
    metric_goal = Goal.IGNORE
    metric_domain = (float("-inf"), float("inf"))

    def fit(self, X, y):
        """This function implements a fit procedure which trains a supervised
        learning model.

        Arguments:
            X (np.ndarray): The numerical features (i.e. transformed rows).
            y (np.ndarray): The output/target value.
        """
        raise NotImplementedError()

    def score(self, X, y):
        """This function scores this model on the (test) dataset.

        Arguments:
            X (np.ndarray): The numerical features (i.e. transformed rows).
            y (np.ndarray): The output/target value.

        Returns:
            float: The value of the appropriate metric.
        """
        raise NotImplementedError()

    def metrics(self, metadata, real_tables, synthetic_tables):
        real_table = real_tables[self.target_table_name]
        synthetic_table = synthetic_tables[self.target_table_name]
        delta_score, synthetic_score = self._evaluate_score(real_table, synthetic_table)

        # Score on the synthetic table. Evaluated on real.
        yield Metric(
            name=self.name,
            value=synthetic_score,
            tags=set([
                "efficacy:ml",
                "table:%s" % self.target_table_name,
                "column:%s" % self.target_column_name,
            ]),
            goal=self.metric_goal,
            unit=self.metric_unit,
            domain=self.metric_domain,
            description="Score on the real test set using the machine learning"
            " model trained on synthetic data."
        )

        # Score on synthetic minus score on real. Evaluated on real.
        delta_domain = self.metric_domain[1] - self.metric_domain[0]
        yield Metric(
            name=self.name,
            value=delta_score,
            tags=set([
                "efficacy:ml",
                "table:%s" % self.target_table_name,
                "column:%s" % self.target_column_name,
            ]),
            goal=self.metric_goal,
            unit="delta_%s" % self.metric_unit,
            domain=(-delta_domain, delta_domain),
            description="Diff in score on real when trained on synthetic vs real."
        )

    def _evaluate_score(self, real, synthetic):
        """
        This computes and returns the score of the model on the real test set when
        it is trained on the synthetic data. It also returns the difference in score
        of the model on the real data when trained on the synthetic data minus the
        score when trained on the real data.
        """
        real_X = real.loc[:, real.columns != self.target_column_name].values
        real_y = real[self.target_column_name].values

        synthetic_X = synthetic.loc[:, synthetic.columns != self.target_column_name].values
        synthetic_y = synthetic[self.target_column_name].values

        delta_scores = []
        synthetic_scores = []
        kf = KFold(n_splits=3, shuffle=True)

        for train_index, test_index in kf.split(real_X, real_y):
            # Train a model on the real dataset and test on the real dataset.
            self.fit(real_X[train_index], real_y[train_index])
            real_score = self.score(real_X[test_index], real_y[test_index])

            # Train a model on the synthetic dataset and test it on the real test dataset.
            self.fit(synthetic_X, synthetic_y)
            synthetic_score = self.score(real_X[test_index], real_y[test_index])

            delta_scores.append(synthetic_score - real_score)
            synthetic_scores.append(synthetic_score)

        return np.mean(delta_scores), np.mean(synthetic_scores)
