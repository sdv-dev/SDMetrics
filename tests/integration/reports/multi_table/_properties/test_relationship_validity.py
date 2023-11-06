import sys

from tqdm import tqdm

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table._properties import RelationshipValidity


class TestRelationshipValidity:

    def test_end_to_end(self):
        """Test the ``RelationshipValidity`` multi-table property end to end."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        relationship_validity = RelationshipValidity()

        # Run
        result = relationship_validity.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert result == 1.0

    def test_with_progress_bar(self, capsys):
        """Test that the progress bar is correctly updated."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        relationship_validity = RelationshipValidity()
        num_relationship = 2

        progress_bar = tqdm(total=num_relationship, file=sys.stdout)

        # Run
        result = relationship_validity.get_score(real_data, synthetic_data, metadata, progress_bar)
        progress_bar.close()
        captured = capsys.readouterr()
        output = captured.out

        # Assert
        assert result == 1.0
        assert '100%' in output
        assert f'{num_relationship}/{num_relationship}' in output
