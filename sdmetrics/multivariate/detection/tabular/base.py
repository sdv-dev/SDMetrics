import warnings

import numpy as np
from rdt import HyperTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sdmetrics.report import Goal, Metric


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

    def metrics(self, metadata, real_tables, synthetic_tables):
        """
        This function yields a sequence of Metric object.

        Args:
            metadata (sdv.Metadata): The Metadata object from SDV.
            real_tables (dict): A dictionary mapping table names to dataframes.
            synthetic_tables (dict): A dictionary mapping table names to dataframes.

        Yields:
            Metric: The next metric.
        """
        yield from self._single_table_detection(metadata, real_tables, synthetic_tables)
        yield from self._parent_child_detection(metadata, real_tables, synthetic_tables)

    def _single_table_detection(self, metadata, real_tables, synthetic_tables):
        # Single Table Detection
        for table_name in set(real_tables):
            table_fields = list(metadata.get_dtypes(table_name, ids=False).keys())
            auroc = self._compute_auroc(
                real_tables[table_name][table_fields],
                synthetic_tables[table_name][table_fields])

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

    def _parent_child_detection(self, metadata, real_tables, synthetic_tables):
        # Parent-Child Table Detection
        for table_name in set(real_tables):
            key = metadata.get_primary_key(table_name)
            table_fields = [key] + list(metadata.get_dtypes(table_name, ids=False))
            for child_name in metadata.get_children(table_name):
                child_key = metadata.get_foreign_key(table_name, child_name)
                child_fields = [child_key] + list(metadata.get_dtypes(child_name, ids=False))

                real = self._denormalize(
                    real_tables[table_name][table_fields],
                    key,
                    real_tables[child_name][child_fields],
                    child_key
                )
                synthetic = self._denormalize(
                    synthetic_tables[table_name][table_fields],
                    key,
                    synthetic_tables[child_name][child_fields],
                    child_key
                )

                auroc = self._compute_auroc(real, synthetic)

                yield Metric(
                    name=self.name,
                    value=auroc,
                    tags=set([
                        "detection:auroc",
                        "table:%s" % table_name,
                        "table:%s" % child_name,
                    ] + (["priority:high"] if auroc > 0.9 else [])),
                    goal=Goal.MINIMIZE,
                    unit="auroc",
                    domain=(0.0, 1.0)
                )

    def _compute_auroc(self, real_table, synthetic_table):
        transformer = HyperTransformer()
        real_table = transformer.fit_transform(real_table).values
        synthetic_table = transformer.transform(synthetic_table).values

        X = np.concatenate([real_table, synthetic_table])
        y = np.hstack([np.ones(len(real_table)), np.zeros(len(synthetic_table))])
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
    def _denormalize(table, key, child_table, child_key):
        """
        Given a parent table (with a primary key) and a child table (with a foreign key),
        this performs an outer join and returns a single flat table.
        """
        flat = table.merge(
            child_table,
            how='outer',
            left_on=key,
            right_on=child_key)

        del flat[key]
        if child_key != key:
            del flat[child_key]

        return flat
