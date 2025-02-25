import re

import pandas as pd
import pytest

from sdmetrics.single_table.privacy.cap import (
    CategoricalCAP,
    CategoricalGeneralizedCAP,
    CategoricalZeroCAP,
)


@pytest.mark.parametrize('metric', [CategoricalCAP, CategoricalZeroCAP, CategoricalGeneralizedCAP])
def test_CAP_deprecation_message(metric):
    """Test deprecation warning is raised when running the metric directly."""
    # Setup
    real_data = pd.DataFrame({'col1': range(5), 'col2': ['A', 'B', 'C', 'A', 'B']})
    synthetic_data = pd.DataFrame({'col1': range(5), 'col2': ['C', 'A', 'A', 'B', 'C']})

    # Run and Assert
    expected_warning = re.escape(
        'Computing CAP metrics directly is deprecated. For improved privacy metrics, '
        "please use the 'DisclosureProtection' and 'DisclosureProtectionEstimate' "
        'metrics instead.'
    )
    with pytest.warns(DeprecationWarning, match=expected_warning):
        metric.compute(real_data, synthetic_data, key_fields=['col1'], sensitive_fields=['col2'])
