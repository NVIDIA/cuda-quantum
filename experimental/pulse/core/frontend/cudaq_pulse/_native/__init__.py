# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around the C++ nanobind extension.

The native extension is required for MLIR dialect bindings. If it is not
built, importing this module raises ``ImportError`` with an actionable
message. Set ``CUDAQ_PULSE_ALLOW_NO_NATIVE=1`` to suppress the error
during pure-Python development/testing only.
"""

from __future__ import annotations

import os

try:
    from ._cudaq_pulse_native import *  # noqa: F401,F403
except ImportError:
    if os.environ.get("CUDAQ_PULSE_ALLOW_NO_NATIVE",
                      "0") not in ("1", "true", "yes"):
        raise ImportError(
            "cudaq-pulse native extension (_cudaq_pulse_native) is not available. "
            "Build it with `pip install .` from the cudaq-pulse root, or set "
            "CUDAQ_PULSE_ALLOW_NO_NATIVE=1 for pure-Python development mode.")
