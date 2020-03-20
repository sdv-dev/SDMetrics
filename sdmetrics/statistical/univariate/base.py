
from sdmetrics import Metric


class UnivariateMetric():
    """
    Attributes:
        name (str): The name of the univariate metric.
        dtypes (list[str]): The data types to accept (i.e. [float, str]).
    """

    name = ""
    dtypes = []

    @staticmethod
    def metric(real_column, fake_column):
        """This function is expected to perform a statistical test on the two
        samples and return a tuple containing (value, goal, unit, domain). See the
        Metric object for what these fields represent.

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
        for column_name, column_type in dtypes.items():
            if column_type not in self.dtypes:
                continue
            x1 = real[column_name].values
            x2 = fake[column_name].values
            value, goal, unit, domain = self.metric(x1, x2)
            yield Metric(
                name=self.name,
                value=value,
                tags=set([
                    "statistic:univariate",
                    "table:%s" % name,
                    "column:%s" % column_name
                ] + (["priority:high"] if value < 0.1 and unit == "p-value" else [])),
                goal=goal,
                unit=unit,
                domain=domain
            )
