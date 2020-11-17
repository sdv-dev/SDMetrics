
from sdmetrics.report import Metric


class UnivariateMetric():
    """
    Attributes:
        name (str): The name of the univariate metric.
        dtypes (list[str]): The data types to accept (i.e. [float, str]).
    """

    name = ""
    dtypes = []

    @staticmethod
    def metric(real_column, synthetic_column):
        """This function is expected to perform a statistical test on the two
        samples and return a tuple containing (value, goal, unit, domain). See the
        Metric object for what these fields represent.

        Arguments:
            real_column (np.ndarray): The values from the real database.
            synthetic_column (np.ndarray): The values from the synthetic database.

        Returns:
            (str, Goal, str, tuple): A tuple containing (value, goal, unit, domain)
            which corresponds to the fields in a Metric object.
        """
        raise NotImplementedError()

    def metrics(self, metadata, real_tables, synthetic_tables):
        """This function iterates over all the columns in all the tables and, if
        the data type of a column matches the data types for which this metric is
        defined, it computes the metric for that column and yields it.

        Args:
            metadata (sdv.Metadata): The Metadata object from SDV.
            real_tables (dict): A dictionary mapping table names to dataframes.
            synthetic_tables (dict): A dictionary mapping table names to dataframes.

        Yields:
            Metric: The next metric.
        """
        assert real_tables.keys() == synthetic_tables.keys()
        for table_name in real_tables.keys():
            dtypes = metadata.get_dtypes(table_name)
            real = real_tables[table_name]
            synthetic = synthetic_tables[table_name]
            yield from self._compute(table_name, dtypes, real, synthetic)

    def _compute(self, name, dtypes, real, synthetic):
        for column_name, column_type in dtypes.items():
            if column_type not in self.dtypes:
                continue
            x1 = real[column_name].values
            x2 = synthetic[column_name].values
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
