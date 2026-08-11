"""Compatibility shim for different tsai versions.

``get_minirocket_features`` and ``default_device`` moved between releases.
Import from here instead of directly from tsai so the rest of the package
doesn't need to care.
"""

from __future__ import annotations

try:
    from tsai.models.utils import default_device, get_minirocket_features
except ImportError:
    # Older tsai versions expose these directly from the MiniRocket module
    # or from tsai.basics.
    try:
        from tsai.models.MINIROCKET_Pytorch import (
            get_minirocket_features,  # type: ignore[no-redef]
        )
        from tsai.models.utils import default_device  # type: ignore[no-redef]
    except ImportError:
        from tsai.basics import (  # type: ignore[no-redef]
            default_device,
            get_minirocket_features,
        )

__all__ = ["default_device", "get_minirocket_features"]
