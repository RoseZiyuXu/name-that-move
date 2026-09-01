"""Compatibility imports for the offline training API.

New code may import these functions from ``name_that_move.offline``.
"""

from name_that_move.offline.training import (
    extract_features,
    save_artifacts,
    train,
)

__all__ = ["extract_features", "save_artifacts", "train"]
