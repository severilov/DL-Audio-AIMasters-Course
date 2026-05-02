import os
import tempfile


# Newer numba/librosa combinations can fail if numba has no writable cache locator.
# Point it at Tortoise's cache directory before any librosa-backed modules import.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "tortoise-numba-cache"),
)
