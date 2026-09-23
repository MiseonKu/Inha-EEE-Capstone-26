"""Project-local Python dependency path hook.

This keeps optional local packages under `.python_deps` inside the repository
so verification can run without modifying system Python or user site-packages.
"""

from __future__ import annotations

import sys
from pathlib import Path


DEPS = Path(__file__).resolve().parent / ".python_deps"
if DEPS.exists():
    deps_text = str(DEPS)
    if deps_text not in sys.path:
        sys.path.insert(0, deps_text)
