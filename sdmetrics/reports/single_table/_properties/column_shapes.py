from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.single_column.statistical.kscomplement import KSComplement
from sdmetrics.single_column.statistical.tv_complement import TVComplement
import pandas as pd
import numpy as np

class ColumnShapes(BaseSingleTableProperty):
    

    metrics = [KSComplement, TVComplement]
    _sdtype_to_metric = {
        'numerical': KSComplement,
        'datetime': KSComplement,
        'categorical': TVComplement,
        'boolean': TVComplement
    }

    def _get_score(self, real_data, synthetic_data, metadata, progress_bar):
        
        column_names, metric_names, scores = [], [], []
        for column_name in progress_bar(metadata['columns']):
            sdtype = metadata[column_name]['sdtype']
            try:
                if sdtype in self._sdtype_to_metric:
                    metric = self._sdtype_to_metric[sdtype]
                    column_score = metric.compute(real_data[column_name], synthetic_data[column_name])
                else:
                    continue

            except Exception as e:
                    column_score = np.nan
                    Warning("Unable to compute Column Shape for column <name>. " + 
                            "Encountered Error: type(e).__name__ e")
    
            column_names.append(column_name)
            metric_names.append(metric.__name__)
            scores.append(column_score)
    
        result = pd.DataFrame({
            'Column': column_names,
            'Metric': metric_names,
            'Score': scores,
        })

        return result