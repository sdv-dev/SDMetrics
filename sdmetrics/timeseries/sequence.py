from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.single_column.statistical import KSComplement

class SequenceLengthSimilarity(TimeSeriesMetric):
    """Sequence Length test based on Kolmogorov-Smirnov statistic based metric.
    This function uses the two-sample Kolmogorov–Smirnov test to compare
    the distributions of the event sequence length (i.e. number of rows of events linked to each entity) 
    using the empirical CDF between real and synthetic data.
    
    It returns 1 minus the KS Test D statistic, which indicates the maximum
    distance between the expected CDF and the observed CDF values.
    
    As a result, the output value is 1.0 if the distributions are identical
    and 0.0 if they are completely different.
    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = 'Sequence Length Similarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0
       
    @classmethod
    def compute(cls,real_data, synthetic_data, metadata=None, entity_columns=None):
        """Compare event sequence length per entity using a Kolmogorov–Smirnov test.
        Args:
            real_data (Union[pandas.Series]):
                The real dataset.
            synthetic_data (Union[pandas.Series]):
                The synthetic dataset.
        Returns:
            float:
                1 minus the Kolmogorov–Smirnov D statistic.
        """
        _, entity_columns = super()._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)
        
        
        real_data = real_data.groupby(entity_columns).size()
        synthetic_data = synthetic_data.groupby(entity_columns).size()
        output = KSComplement.compute(real_data, synthetic_data)
        return output
    
    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.
        Args:
            raw_score (float):
                The value of the metric from `compute`.
        Returns:
            float:
                The normalized value of the metric
        """
        normalized = KSComplement.normalize(raw_score)
        
        return normalized