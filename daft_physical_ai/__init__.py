"""Example Python-only Daft extension."""

from __future__ import annotations

import daft


@daft.func
def greet(name: str | None) -> str | None:
    """Greet someone by name."""
    if name is None:
        return None
    return f"Hello, {name}!"


__all__ = ["greet"]
