"""One timestamp clock for traces, metrics samples and the run manifest.

time.time() can step backwards when the system clock is corrected (WSL resyncs
by up to ~1 s; NTP can step a Linux clock), which makes spans appear to overlap
or end before they start. now() is wall-clock seconds since the epoch, anchored
to time.time() once per process and then advanced by the monotonic
perf_counter(): readable as a date, never going backwards within a process.
"""

from __future__ import annotations

import time

_WALL0 = time.time()
_MONO0 = time.perf_counter()


def now() -> float:
    return _WALL0 + (time.perf_counter() - _MONO0)
