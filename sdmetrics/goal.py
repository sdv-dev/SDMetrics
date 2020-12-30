"""SDMetrics Goal Enumeration."""

from enum import Enum


class Goal(Enum):
    """Goal Enumeration.

    This enumerates the ``goal`` for a metric; the value of a metric can be ignored,
    minimized, or maximized.
    """

    IGNORE = "ignore"
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
