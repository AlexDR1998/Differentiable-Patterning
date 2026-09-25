"""Shared base class for the immutable typed configuration dataclasses."""


class ConfigValue:
    """Marker base class for typed configuration dataclasses.

    Read fields as attributes (``cfg.run.t``); every field has a value, so
    there is no need for ``.get(key, default)``.
    """
