"""Constraints."""

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.column_formula import ColumnFormula
from sdmetrics.multi_table.statistical.constraints.chained_inequality import ChainedInequality
from sdmetrics.multi_table.statistical.constraints.inequality import Inequality
from sdmetrics.multi_table.statistical.constraints.fixed_combinations import FixedCombinations
from sdmetrics.multi_table.statistical.constraints.fixed_increments import FixedIncrements
from sdmetrics.multi_table.statistical.constraints.fixed_null_combinations import (
    FixedNullCombinations,
)
from sdmetrics.multi_table.statistical.constraints.foreign_to_foreign_key import (
    ForeignToForeignKey,
)
from sdmetrics.multi_table.statistical.constraints.foreign_to_primary_key_subset import (
    ForeignToPrimaryKeySubset,
)
from sdmetrics.multi_table.statistical.constraints.mixed_scales import MixedScales
from sdmetrics.multi_table.statistical.constraints.polymorphic_relationship import (
    PolymorphicRelationship,
)
from sdmetrics.multi_table.statistical.constraints.primary_to_primary_key_subset import (
    PrimaryToPrimaryKeySubset,
)
from sdmetrics.multi_table.statistical.constraints.reference_table import ReferenceTable
from sdmetrics.multi_table.statistical.constraints.one_hot_encoding import OneHotEncoding
from sdmetrics.multi_table.statistical.constraints.range import Range
from sdmetrics.multi_table.statistical.constraints.referential_hierarchy import (
    SelfReferentialHierarchy,
)

__all__ = (
    BaseConstraint,
    ColumnFormula,
    ChainedInequality,
    Inequality,
    FixedIncrements,
    FixedCombinations,
    FixedNullCombinations,
    ForeignToForeignKey,
    ForeignToPrimaryKeySubset,
    MixedScales,
    PolymorphicRelationship,
    PrimaryToPrimaryKeySubset,
    ReferenceTable,
    OneHotEncoding,
    Range,
    SelfReferentialHierarchy,
)
