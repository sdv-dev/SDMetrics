"""Base class for Machine Learning Detection metrics that work on parent-child pairs of tables."""

import numpy as np

from sdmetrics.multi_table.detection.base import DetectionMetric
from sdmetrics.single_table.detection import LogisticDetection, SVCDetection
from sdmetrics.utils import get_columns_from_metadata, nested_attrs_meta


class ParentChildDetectionMetric(DetectionMetric,
                                 metaclass=nested_attrs_meta('single_table_metric')):
    """Base class for Multi-table Detection metrics based on parent-child relationships.

    These metrics denormalize the parent-child relationships from the dataset and then
    apply a Single Table Detection metric on the resulting tables.

    The output of the metric is one minus the average ROC AUC score obtained.

    A part from the real and synthetic data, these metrics need to be passed
    a list with the foreign key relationships that exist between the tables.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        single_table_metric (sdmetrics.single_table.detection.base.DetectionMetric):
            The single table detection metric to use.
    """

    single_table_metric = None

    @staticmethod
    def _extract_foreign_keys(metadata):
        if not isinstance(metadata, dict):
            metadata = metadata.to_dict()

        foreign_keys = []
        for child_table, child_meta in metadata['tables'].items():
            for child_key, field_meta in get_columns_from_metadata(child_meta).items():
                ref = field_meta.get('ref')
                if ref:
                    foreign_keys.append((ref['table'], ref['field'], child_table, child_key))

        return foreign_keys

    @staticmethod
    def _denormalize(data, foreign_key, metadata = None):
        """Denormalize the child table over the parent."""
        parent_table, parent_key, child_table, child_key = foreign_key


        if not isinstance(metadata, dict):
            metadata = metadata.to_dict()

        if metadata is not None:
            table_meta_parent = metadata['tables'][parent_table]
            table_meta_child = metadata['tables'][child_table]
        else:
            table_meta_parent = None
            table_meta_child = None
        to_drop_parent = []
        to_drop_child = []

        if table_meta_child is not None and 'primary_key' in table_meta_child:
            to_drop_child.append(table_meta_child['primary_key'])

        if table_meta_child is not None:
            for field in table_meta_child['fields'].keys():
                if ('ref' in table_meta_child['fields'][field].keys()) and (field!=child_key):
                    to_drop_child.append(field)

        if table_meta_parent is not None:
            for field in table_meta_parent['fields'].keys():
                if 'ref' in table_meta_parent['fields'][field].keys():
                    to_drop_parent.append(field)
        

        flat = data[parent_table].drop(to_drop_parent, axis=1).set_index(parent_key).merge(
            data[child_table].drop(to_drop_child, axis=1).set_index(child_key),
            how='outer',
            left_index=True,
            right_index=True,
        ).reset_index(drop=True)

        return flat

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, foreign_keys=None):
        """Compute this metric.

        This denormalizes the parent-child relationships from the dataset and then
        applies a Single Table Detection metric on the resulting tables.

        The output of the metric is one minus the average ROC AUC score obtained.

        A part from the real and synthetic data, either a ``foreign_keys`` list
        containing the relationships between the tables or a ``metadata`` that can be
        used to create such list must be passed.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset.
            metadata (dict):
                Multi-table metadata dict. If not passed, foreign keys must be
                passed.
            foreign_keys (list[tuple[str, str, str, str]]):
                List of foreign key relationships specified as tuples
                that contain (parent_table, parent_key, child_table, child_key).
                Ignored if metada is given.

        Returns:
            float:
                Average of the scores obtained by the single table metric.
        """
        if metadata:
            foreign_keys = cls._extract_foreign_keys(metadata)
        if not foreign_keys:
            raise ValueError('No foreign keys given')

        scores = []
        
        for foreign_key in foreign_keys:
            parent_table, parent_key, child_table, child_key = foreign_key

            # Keep attributes only
            if not isinstance(metadata, dict):
                metadata = metadata.to_dict()

            no_attribute_parent = []
            no_attribute_child = []
            if metadata is not None:
                table_meta_parent = metadata['tables'][parent_table]
                table_meta_child = metadata['tables'][child_table]
            else:
                table_meta_parent = None
                table_meta_child = None

            if table_meta_parent is not None:
                if 'primary_key' in table_meta_parent:
                    no_attribute_parent.append(table_meta_parent['primary_key'])
                for field in table_meta_parent['fields'].keys():
                    if 'ref' in table_meta_parent['fields'][field].keys():
                        no_attribute_parent.append(field)
            if table_meta_child is not None:
                if 'primary_key' in table_meta_child:
                    no_attribute_child.append(table_meta_child['primary_key'])
                for field in table_meta_child['fields'].keys():
                    if 'ref' in table_meta_child['fields'][field].keys():
                        no_attribute_child.append(field)
            
            for c in real_data[parent_table].columns:
                if c not in no_attribute_parent:
                    real_data[parent_table] = real_data[parent_table].rename(columns={c: "parent."+c}).copy()
                    synthetic_data[parent_table] = synthetic_data[parent_table].rename(columns={c: "parent."+c}).copy()
                    if c in metadata['tables'][parent_table]['fields'].keys():
                        metadata['tables'][parent_table]['fields']["parent."+c] = metadata['tables'][parent_table]['fields'].pop(c)
            for c in real_data[child_table].columns:
                if c not in no_attribute_child:
                    real_data[child_table] = real_data[child_table].rename(columns={c: "child."+c}).copy()
                    synthetic_data[child_table] = synthetic_data[child_table].rename(columns={c: "child."+c}).copy()
                    if c in metadata['tables'][child_table]['fields'].keys():
                        metadata['tables'][child_table]['fields']["child."+c] = metadata['tables'][child_table]['fields'].pop(c)

            # Denormalize and apply model
            real = cls._denormalize(real_data, foreign_key, metadata)
            synth = cls._denormalize(synthetic_data, foreign_key, metadata)


            metadata_merged = {'fields': {}}
            for field in real.columns:
                if field in metadata['tables'][parent_table]['fields'].keys():
                    metadata_merged['fields'][field] = metadata['tables'][parent_table]['fields'][field]
                elif field in metadata['tables'][child_table]['fields'].keys():
                    metadata_merged['fields'][field] = metadata['tables'][child_table]['fields'][field]
            
            for c in real_data[parent_table].columns:
                if c not in no_attribute_parent:
                    to_c = c[-len(c)+len('parent.'):]
                    real_data[parent_table] = real_data[parent_table].rename(columns={c: to_c}).copy()
                    synthetic_data[parent_table] = synthetic_data[parent_table].rename(columns={c: to_c}).copy()
                    if c in metadata['tables'][parent_table]['fields'].keys():
                        metadata['tables'][parent_table]['fields'][to_c] = metadata['tables'][parent_table]['fields'].pop(c)
            for c in real_data[child_table].columns:
                if c not in no_attribute_child:
                    to_c = c[-len(c)+len('child.'):]
                    real_data[child_table] = real_data[child_table].rename(columns={c: to_c}).copy()
                    synthetic_data[child_table] = synthetic_data[child_table].rename(columns={c: to_c}).copy()
                    if c in metadata['tables'][child_table]['fields'].keys():
                        metadata['tables'][child_table]['fields'][to_c] = metadata['tables'][child_table]['fields'].pop(c)

            score = cls.single_table_metric.compute(real, synth, metadata_merged)
            scores.append(score)

        return np.mean(scores)


class LogisticParentChildDetection(ParentChildDetectionMetric):
    """ParentChild detection metric based on a LogisticRegression."""

    single_table_metric = LogisticDetection


class SVCParentChildDetection(ParentChildDetectionMetric):
    """ParentChild detection metric based on a SVC."""

    single_table_metric = SVCDetection
