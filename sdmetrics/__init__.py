# -*- coding: utf-8 -*-

"""Top-level package for SDMetrics."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.22.0'

import sys
import warnings as python_warnings
from importlib.metadata import entry_points
from operator import attrgetter
from types import ModuleType

import pandas as pd

from sdmetrics import (
    column_pairs,
    demos,
    goal,
    multi_table,
    reports,
    single_column,
    single_table,
    timeseries,
)
from sdmetrics.demos import load_demo

__all__ = [
    'demos',
    'load_demo',
    'goal',
    'multi_table',
    'column_pairs',
    'single_column',
    'single_table',
    'timeseries',
    'reports',
]


def compute_metrics(metrics, real_data, synthetic_data, metadata=None, **kwargs):
    """Compute a collection of metrics on the given data.

    Args:
        metrics (list[sdmetrics.base.BaseMetric]):
            Metrics to compute.
        real_data:
            Data from the real dataset
        synthetic_data:
            Data from the synthetic dataset
        metadata (dict):
            Dataset metadata.
        **kwargs:
            Any additional arguments to pass to the metrics.

    Returns:
        pandas.DataFrame:
            Dataframe containing the metric scores, as well as information
            about each metric such as the min and max values and its goal.
    """
    # Only add metadata to kwargs if passed, to stay compatible
    # with metrics that do not expect a metadata argument
    if metadata is not None:
        kwargs['metadata'] = metadata

    scores = []
    for name, metric in metrics.items():
        error = None
        try:
            raw_score = metric.compute(real_data, synthetic_data, **kwargs)
            normalized_score = metric.normalize(raw_score)
        except Exception as err:
            raw_score = None
            normalized_score = None
            error = str(err)

        scores.append({
            'metric': name,
            'name': metric.name,
            'raw_score': raw_score,
            'normalized_score': normalized_score,
            'min_value': metric.min_value,
            'max_value': metric.max_value,
            'goal': metric.goal.name,
            'error': error,
        })

    return pd.DataFrame(scores)


def _get_addon_target(addon_path_name):
    """Find the target object for the add-on.

    Args:
        addon_path_name (str):
            The add-on's name. The add-on's name should be the full path of valid Python
            identifiers (i.e. importable.module:object.attr).

    Returns:
        tuple:
            * object:
                The base module or object the add-on should be added to.
            * str:
                The name the add-on should be added to under the module or object.
    """
    module_path, _, object_path = addon_path_name.partition(':')
    module_path = module_path.split('.')

    if module_path[0] != __name__:
        msg = f"expected base module to be '{__name__}', found '{module_path[0]}'"
        raise AttributeError(msg)

    target_base = sys.modules[__name__]
    for submodule in module_path[1:-1]:
        target_base = getattr(target_base, submodule)

    addon_name = module_path[-1]
    if object_path:
        if len(module_path) > 1 and not hasattr(target_base, module_path[-1]):
            msg = f"cannot add '{object_path}' to unknown submodule '{'.'.join(module_path)}'"
            raise AttributeError(msg)

        if len(module_path) > 1:
            target_base = getattr(target_base, module_path[-1])

        split_object = object_path.split('.')
        addon_name = split_object[-1]

        if len(split_object) > 1:
            target_base = attrgetter('.'.join(split_object[:-1]))(target_base)

    return target_base, addon_name


def _find_addons():
    """Find and load all SDMetrics add-ons.

    If the add-on is a module, we add it both to the target module and to
    ``system.modules`` so that they can be imported from the top of a file as follows:

    from top_module.addon_module import x
    """
    group = 'sdmetrics_modules'
    try:
        eps = entry_points(group=group)
    except TypeError:
        # Load-time selection requires Python >= 3.10 or importlib_metadata >= 3.6
        eps = entry_points().get(group, [])

    for entry_point in eps:
        try:
            addon = entry_point.load()
        except Exception:  # pylint: disable=broad-exception-caught
            msg = f'Failed to load "{entry_point.name}" from "{entry_point.value}".'
            python_warnings.warn(msg)
            continue

        try:
            addon_target, addon_name = _get_addon_target(entry_point.name)
        except AttributeError as error:
            msg = f"Failed to set '{entry_point.name}': {error}."
            python_warnings.warn(msg)
            continue

        if isinstance(addon, ModuleType):
            addon_module_name = f'{addon_target.__name__}.{addon_name}'
            if addon_module_name not in sys.modules:
                sys.modules[addon_module_name] = addon

        setattr(addon_target, addon_name, addon)


_find_addons()
