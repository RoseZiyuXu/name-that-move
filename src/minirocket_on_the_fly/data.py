"""Compatibility imports for the offline data API.

New code may import these functions from ``minirocket_on_the_fly.offline``.
"""

from minirocket_on_the_fly.offline.data import (
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
