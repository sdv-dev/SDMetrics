
from sklearn.model_selection import train_test_split
from sdmetrics.demos import load_single_table_demo
from sdmetrics.single_table.privacy.dcr_baseline_protection import DCRBaselineProtection


class TestDisclosureProtection:
    def test_end_to_end_with_demo(self):
        real_data, synthetic_data, metadata = load_single_table_demo()
        print(metadata)
        train_df, holdout_df = train_test_split(real_data, test_size=0.2)
        result = DCRBaselineProtection.compute_breakdown(train_df, synthetic_data,  holdout_df, metadata)
        print(result)
        assert False
