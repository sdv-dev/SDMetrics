from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.single_column.statistical.kscomplement import KSComplement
from sdmetrics.single_column.statistical.tv_complement import TVComplement
import pandas as pd
import numpy as np
import plotly.express as px


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
            'Column name': column_names,
            'Metric': metric_names,
            'Score': scores,
        })

        return result

    def get_visualization(self):
        """Create a plot to show the column shape scores.

        Returns:
            plotly.graph_objects._figure.Figure
        """
        average_score = self._average_scores()
        data = self._details
    
        fig = px.bar(
            data,
            x='Column name',
            y='Score',
            title=f'Data Quality: Column Shapes (Average Score={round(average_score, 2)})',
            category_orders={'group': data['Column Name']},
            color='Metric',
            color_discrete_map={
                'KSComplement': '#000036',
                'TVComplement': '#03AFF1',
            },
            pattern_shape='Metric',
            pattern_shape_sequence=['', '/'],
            hover_name='Column Name',
            hover_data={
                'Column Name': False,
                'Metric': True,
                'Quality Score': True,
            },
        )

        fig.update_yaxes(range=[0, 1])

        fig.update_layout(
            xaxis_categoryorder='total ascending',
            plot_bgcolor='#F5F5F8',
            margin={'t': 150},
        )

        return fig