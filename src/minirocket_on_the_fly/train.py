"""Compatibility imports for the offline training API.

New code may import these functions from ``minirocket_on_the_fly.offline``.
"""

from minirocket_on_the_fly.offline.training import (
    extract_features,
    save_artifacts,
    train,
)

__all__ = ["extract_features", "save_artifacts", "train"]
