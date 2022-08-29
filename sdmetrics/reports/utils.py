"""Report utility methods."""

import pandas as pd
import plotly.figure_factory as ff

DATACEBO_DARK = '#000036'
DATACEBO_LIGHT = '#01E0C9'


def plot_column(real_column, synthetic_column, sdtype):
    """Plot the real and synthetic data for a given column.

    Args:
        real_column (pandas.Series):
            The real data for the desired column.
        synthetic_column (pandas.Series):
            The synthetic data for the desired column.
        sdtype (str):
            The data type of the column.
    """
    column_name = real_column.name
    missing_data_real = round((real_column.is_na().sum() / len(real_column)) * 100, 2)
    missing_data_synthetic = round((synthetic_column.is_na().sum() / len(synthetic_column)), 2)

    real_data = real_column.dropna()
    synthetic_data = synthetic_column.dropna()

    if sdtype == 'datetime':
        real_data = real_data.view(int)
        synthetic_data = synthetic_data.view(int)

    fig = ff.create_distplot(
        [real_data, synthetic_data],
        ['Real', 'Synthetic'],
        show_hist=False,
        show_rug=False,
        colors=[DATACEBO_DARK, DATACEBO_LIGHT]
    )

    fig.update_traces(
        x=pd.to_datetime(fig.data[0].x) if sdtype == 'datetime' else fig.data[0].x,
        fill='tozeroy',
        hovertemplate='<b>Real</b><br>Value: %{x}<br>Frequency: %{y}<extra></extra>',
        selector={'name': 'Real'}
    )

    fig.update_traces(
        x=pd.to_datetime(fig.data[1].x) if sdtype == 'datetime' else fig.data[1].x,
        fill='tozeroy',
        hovertemplate='<b>Synthetic</b><br>Value: %{x}<br>Frequency: %{y}<extra></extra>',
        selector={'name': 'Synthetic'}
    )

    fig.update_layout(
        title=f'Real vs. Synthetic Data for column {column_name}',
        xaxis_title='Value',
        yaxis_title='Frequency',
        plot_bgcolor='#F5F5F8',
        annotations=[
            {
                'xref': 'paper',
                'yref': 'paper',
                'x': -0.08,
                'y': -0.2,
                'showarrow': False,
                'text': (
                    f'*Missing Values: Real Data ({missing_data_real}%), '
                    f'Synthetic Data ({missing_data_synthetic}%)'
                ),
            },
        ]
    )

    fig.show()
