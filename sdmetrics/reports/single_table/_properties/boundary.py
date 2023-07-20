import numpy as np
import pandas as pd
import plotly.express as px

from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.single_column import BoundaryAdherence


class Boundary(BaseSingleTableProperty):
    """Boundary property class for single table."""

    metric = BoundaryAdherence

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        return super()._generate_details(real_data, synthetic_data, metadata, progress_bar)
