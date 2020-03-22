import os

import numpy as np
import pandas as pd
from sdv import Metadata

from sdmetrics.datasets import Dataset

size = 100
tables = {
    "table1": pd.DataFrame({
        "x": np.random.random(size=size),
        "y": np.random.normal(size=size, loc=10.0)
    })
}
lq_synthetic = {
    "table1": pd.DataFrame({
        "x": np.random.random(size=size) + np.random.normal(size=size),
        "y": np.random.normal(size=size, loc=10.0) + np.random.normal(size=size)
    })
}
hq_synthetic = {
    "table1": pd.DataFrame({
        "x": np.random.random(size=size) + np.random.normal(size=size) / 10.0,
        "y": np.random.normal(size=size, loc=10.0) + np.random.normal(size=size) / 10.0
    })
}

metadata = Metadata()
for table_name, df in tables.items():
    metadata.add_table(table_name, data=df)
dataset = Dataset(metadata, tables, lq_synthetic, hq_synthetic)
dataset.save(os.path.dirname(__file__))
