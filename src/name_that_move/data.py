"""Compatibility imports for the offline data API.

New code may import these functions from ``name_that_move.offline``.
"""

from name_that_move.offline.data import (
    DEFAULT_FILE_NAMES,
    augment_segments,
    load_segments,
    make_dataset,
)

__all__ = [
    "DEFAULT_FILE_NAMES",
    "augment_segments",
    "load_segments",
    "make_dataset",
]
