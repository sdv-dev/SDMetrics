"""Base Constraint."""

class BaseConstraint:

    _is_single_table = True

    def _get_single_table_name(self, metadata):
        if not hasattr(self, 'table_name'):
            raise ValueError('No ``table_name`` attribute has been set.')

        return metadata._get_single_table_name() if self.table_name is None else self.table_name


    def load_constraint(self, constraint_dict):
        pass

    def compute(self):
        raise NotImplementedError()