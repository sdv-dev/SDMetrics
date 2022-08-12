"""Warnings for sdmetrics."""


class ConstantInputWarning(RuntimeWarning):
    """Thrown when the input data has all the same values."""

    def __init__(self, message):
        self.message = message
