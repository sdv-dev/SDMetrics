import warnings

import numpy as np
from rdt import HyperTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sdmetrics import Goal, Metric


class TabularDetector():

    name = ""

    def fit(self, X, y):
        """This function implements a fit procedure which trains a binary
        classification model where class=1 indicates the data is synthetic
        and class=0 indicates that the data is real.

        Arguments:
            X (np.ndarray): The numerical features (i.e. transformed rows).
            y (np.ndarray): The binary classification target.
        """
        raise NotImplementedError()

    def predict_proba(self, X):
        """This function predicts the probability that each of the samples
        comes from the synthetic dataset.

        Arguments:
            X (np.ndarray): The numerical features (i.e. transformed rows).

        Returns:
            np.ndarray: The probability that the class is 1.
        """
        raise NotImplementedError()

    def metrics(self, metadata, real_tables, fake_tables):
        for table_name in set(real_tables):
            # Single Table Detection
            auroc = self._auroc(
                real_tables[table_name],
                fake_tables[table_name])

            yield Metric(
                name=self.name,
                value=auroc,
                tags=set([
                    "detection:auroc",
                    "table:%s" % table_name
                ]),
                goal=Goal.MINIMIZE,
                unit="auroc",
                domain=(0.0, 1.0)
            )

            # Parent-Child Table Detection
            key = metadata.get_primary_key(table_name)
            for child_name in metadata.get_children(table_name):
                child_key = metadata.get_foreign_key(table_name, child_name)

                real = self._explode(
                    real_tables[table_name],
                    key,
                    real_tables[child_name],
                    child_key)
                fake = self._explode(
                    fake_tables[table_name],
                    key,
                    fake_tables[child_name],
                    child_key)
                auroc = self._auroc(real, fake)

                yield Metric(
                    name=self.name,
                    value=auroc,
                    tags=set([
                        "detection:auroc",
                        "table:%s" % table_name,
                        "child:%s" % child_name,
                    ] + (["priority:high"] if auroc > 0.9 else [])),
                    goal=Goal.MINIMIZE,
                    unit="auroc",
                    domain=(0.0, 1.0)
                )

    def _auroc(self, real_table, fake_table):
        transformer = HyperTransformer()
        real_table = transformer.fit_transform(real_table).values
        fake_table = transformer.transform(fake_table).values

        X = np.concatenate([real_table, fake_table])
        y = np.hstack([np.ones(len(real_table)), np.zeros(len(fake_table))])
        X[np.isnan(X)] = 0.0

        if len(X) < 20:
            warnings.warn("Not enough data, skipping the detection tests.")

        scores = []
        kf = StratifiedKFold(n_splits=3, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            self.fit(X[train_index], y[train_index])
            y_pred = self.predict_proba(X[test_index])
            auroc = roc_auc_score(y[test_index], y_pred)
            if auroc < 0.5:
                auroc = 1.0 - auroc
            scores.append(auroc)
        return np.mean(scores)

    @staticmethod
    def _explode(table, key, child_table, child_key):
        """
        Given a parent table (with a primary key) and a child table (with a foreign key),
        this performs an outer join and returns a single flat table.
        """
        flat = table.merge(
            child_table,
            how='outer',
            left_on=key,
            right_on=child_key)
        return flat
