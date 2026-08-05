"""Small shared primitives for immutable typed configuration values."""

from dataclasses import fields
from typing import Any


class ConfigValue:
    """Compatibility surface for typed values during configuration migration."""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self):
        return ((item.name, getattr(self, item.name)) for item in fields(self))
