"""Return freed heap memory to the OS.

Each round the server receives tens of MB of ciphertext per hospital, and gRPC/protobuf make several
copies. glibc keeps those fragmented pages mapped, so RSS ratchets up round after round; trimming
after aggregation keeps the long-run footprint flat. No-op where libc.malloc_trim is unavailable.
"""
from __future__ import annotations

import ctypes
import gc


def release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
