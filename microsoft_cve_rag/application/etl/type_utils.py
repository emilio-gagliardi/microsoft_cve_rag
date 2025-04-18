"""Utility functions for type conversions and validation."""
from typing import Optional


def convert_to_float(value: str | float | int | None) -> Optional[float]:
    """Convert a value to float with validation.

    Args:
        value: The value to convert to float. Can be string, float, int, or None.

    Returns:
        Optional[float]: The converted float value if valid, None otherwise.
        Returns None if:
        - Input is None
        - Input cannot be converted to float
        - Resulting float is negative
    """
    if value is None:
        return None

    try:
        return float(value.strip()) if value and value.strip() else None
    except (ValueError, AttributeError, TypeError):
        return None
