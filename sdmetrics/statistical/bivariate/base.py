from itertools import permutations

from sdmetrics import Metric


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
    def metric(real, fake):
        """This function is expected to perform a statistical test on the two
        samples, each of which is two-dimensional, and return a tuple containing
        (value, goal, unit, domain). See the Metric object for what these fields
        represent.

        Arguments:
            real_column (np.ndarray): The values from the real database.
            fake_column (np.ndarray): The values from the fake database.

        Returns:
            (str, Goal, str, tuple): A tuple containing (value, goal, unit, domain)
            which corresponds to the fields in a Metric object.
        """
        raise NotImplementedError()

    def metrics(self, metadata, real_tables, synthetic_tables):
        tables = set(real_tables).union(synthetic_tables)
        for name in tables:
            dtypes = metadata.get_dtypes(name)
            real = real_tables[name]
            fake = synthetic_tables[name]
            yield from self._handle(name, dtypes, real, fake)

    def _handle(self, name, dtypes, real, fake):
        for (col1, col1_type), (col2, col2_type) in permutations(
                dtypes.items(), r=2):
            if (col1_type, col2_type) not in self.dtypes:
                continue
            X1 = real[[col1, col2]].values
            X2 = fake[[col1, col2]].values
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
