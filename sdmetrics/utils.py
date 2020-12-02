"""SDMetrics utils to be used across all the project."""


def NestedAttrsMeta(nested):
    """Metaclass factory that defines a Metaclass with a dynamic attribute name."""

    class Metaclass(type):
        """Metaclass which pulls the attributes from a nested object using properties."""

        def __getattr__(cls, attr):
            """If cls does not have the attribute, try to get it from the nested object."""
            if hasattr(cls, attr):
                return getattr(cls, attr)

            nested_obj = getattr(cls, nested)
            if hasattr(nested_obj, attr):
                return getattr(nested_obj, attr)

            # At this point we know that neither cls nor the nested object has the attribute.
            # However, we try getting the attribute from cls again to provoke a crash with
            # the right error message in it.
            return getattr(cls, attr)

    return Metaclass
