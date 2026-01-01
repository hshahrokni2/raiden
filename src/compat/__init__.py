"""
Compatibility fixes for third-party packages.

This module patches known issues with dependencies before they're imported.
Import this module early in the application startup.
"""

import sys
from unittest.mock import MagicMock


def patch_collections_abc():
    """
    Fix Python 3.10+ compatibility for packages using old collections imports.

    Many older packages (like geomeppy) use:
        from collections import MutableSequence

    But Python 3.10+ moved these to collections.abc:
        from collections.abc import MutableSequence

    This patch adds the old names back to collections module.
    """
    import collections
    import collections.abc

    # Patch all the moved classes
    for name in [
        'MutableSequence', 'MutableMapping', 'MutableSet',
        'Sequence', 'Mapping', 'Set',
        'Iterable', 'Iterator', 'Callable',
        'Container', 'Hashable', 'Sized',
        'MappingView', 'KeysView', 'ItemsView', 'ValuesView',
    ]:
        if not hasattr(collections, name) and hasattr(collections.abc, name):
            setattr(collections, name, getattr(collections.abc, name))


def patch_tkinter():
    """
    Mock tkinter if not available.

    GeomEppy imports visualization code that requires tkinter.
    On headless servers or some macOS Python installations, tkinter isn't available.
    We don't need visualization, so we mock it.
    """
    try:
        import tkinter
    except (ImportError, ModuleNotFoundError):
        # Create mock tkinter module
        mock_tk = MagicMock()
        mock_tk.TclError = Exception  # Provide the exception class

        sys.modules['tkinter'] = mock_tk
        sys.modules['_tkinter'] = mock_tk
        sys.modules['six.moves.tkinter'] = mock_tk


def patch_all():
    """Apply all compatibility patches."""
    patch_collections_abc()
    patch_tkinter()


# Auto-apply patches on import
patch_all()
