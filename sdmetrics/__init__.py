# -*- coding: utf-8 -*-

"""Top-level package for SDMetrics."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0.dev0'

import pathlib
import pickle

from sdmetrics import column_pairs, goal, multi_table, single_column, single_table, timeseries

__all__ = [
    'goal',
    'multi_table',
    'column_pairs',
    'single_column',
    'single_table',
    'timeseries',
]


def load_demo():
    demo_path = pathlib.Path(__file__).parent / 'demo.pkl'
    with open(demo_path, 'rb') as demo_file:
        return pickle.load(demo_file)
