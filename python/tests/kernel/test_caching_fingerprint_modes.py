# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import subprocess
import sys

import pytest

_CACHE_SUITE = os.path.join(os.path.dirname(__file__), "test_caching.py")


@pytest.mark.parametrize("structural", [False, True])
def test_cache_suite_holds_in_both_fingerprint_modes(structural):
    """The compiled-module cache must behave identically whichever way
    `createProgramFingerprint` reaches its digest.

    The mode is read once per process, so it cannot be changed from inside a
    running test; re-run the whole cache suite in a fresh process instead.

    Both modes answer the same question and expose nothing that tells them
    apart, so this cannot confirm that the variable was honored, only that the
    cache holds whichever digest it ends up with. Renaming the variable without
    updating this test would cost the coverage silently.
    """
    env = dict(os.environ)
    if structural:
        env["CUDAQ_FINGERPRINT_STRUCTURE"] = "1"
    else:
        env.pop("CUDAQ_FINGERPRINT_STRUCTURE", None)

    result = subprocess.run([
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "--no-header", _CACHE_SUITE
    ],
                            capture_output=True,
                            text=True,
                            env=env)

    # A collection error exits 2 and an empty run exits 5, so a zero status
    # also establishes that the suite actually ran.
    assert result.returncode == 0, (
        "cache suite failed with CUDAQ_FINGERPRINT_STRUCTURE="
        f"{env.get('CUDAQ_FINGERPRINT_STRUCTURE', '<unset>')}\n"
        f"{result.stdout}\n{result.stderr}")
