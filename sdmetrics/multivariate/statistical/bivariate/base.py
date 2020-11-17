from itertools import permutations

from sdmetrics.report import Metric


class BivariateMetric():
    """
    Attributes:
        name (str): The name of the bivariate metric.
        dtypes (list[str]): The ordered pairs of data types to accept (i.e.
        [(float, floatt), (float, intt)]).
    """

    name = ""
    dtypes = []

    @staticmethod
    def metric(real_2d, synthetic_2d):
        """This function is expected to perform a statistical test on the two
        samples, each of which contains two columns, and return a tuple containing
        (value, goal, unit, domain). See the Metric object for what these fields
        represent.

        Arguments:
            real_2d (np.ndarray): Two columns from the real database.
            synthetic_2d (np.ndarray): Two columns from the synthetic database.

        Returns:
            (str, Goal, str, tuple): A tuple containing (value, goal, unit, domain)
            which corresponds to the fields in a Metric object.
        """
        raise NotImplementedError()

    def metrics(self, metadata, real_tables, synthetic_tables):
        """
        This function iterates over all the pairs of columns in all the tables
        and, if the data types match, it yields the Metric.

        Args:
            metadata (sdv.Metadata): The Metadata object from SDV.
            real_tables (dict): A dictionary mapping table names to dataframes.
            synthetic_tables (dict): A dictionary mapping table names to dataframes.

        Yields:
            Metric: The next metric.
        """
        tables = set(real_tables).union(synthetic_tables)
        for name in tables:
            dtypes = metadata.get_dtypes(name)
            real = real_tables[name]
            synthetic = synthetic_tables[name]
            yield from self._compute(name, dtypes, real, synthetic)

    def _compute(self, name, dtypes, real, synthetic):
        for (col1, col1_type), (col2, col2_type) in permutations(
                dtypes.items(), r=2):
            if (col1_type, col2_type) not in self.dtypes:
                continue
            X1 = real[[col1, col2]].values
            X2 = synthetic[[col1, col2]].values
            value, goal, unit, domain = self.metric(X1, X2)
            yield Metric(
                name=self.name,
                value=value,
                tags=set([
                    "statistic:bivariate",
                    "table:%s" % name,
                    "column:%s" % col1,
                    "column:%s" % col2,
                ]),
                goal=goal,
                unit=unit,
                domain=domain
            )
