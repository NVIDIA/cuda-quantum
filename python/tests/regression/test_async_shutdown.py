# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Interpreter-shutdown safety for the asynchronous execution APIs.

A `*_async` call enqueues a C++ task that holds a clone of the kernel's MLIR
module. That clone lives in the global `MLIRContext` owned by the Python side.

If the interpreter then shuts down, Python finalization destroys the
`MLIRContext` before C++ static destruction joins the queue thread, and the
in-flight/enqueued task dereferences a bogus pointer.

The crash only manifests during interpreter teardown, so each case runs in its
own subprocess. See `async_shutdown_child.py` for the program under test.
"""

import os
import signal
import subprocess
import sys

import cudaq
import pytest

CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "async_shutdown_child.py")


def _describe(result):
    """Render a subprocess outcome, naming the signal if one killed it."""
    if result.returncode < 0:
        try:
            name = signal.Signals(-result.returncode).name
        except ValueError:
            name = "unknown signal"
        status = f"killed by {name} ({result.returncode})"
    else:
        status = f"exit code {result.returncode}"
    return (f"child {status}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}")


@pytest.mark.parametrize("api", ["run", "sample", "observe"])
def test_async_abandoned_jobs_shut_down_cleanly(api):
    """Exiting with async jobs still in flight must not crash the interpreter."""
    result = subprocess.run([sys.executable, CHILD, api],
                            capture_output=True,
                            text=True,
                            timeout=600)

    assert result.returncode == 0, (
        f"interpreter did not shut down cleanly with outstanding "
        f"{api}_async jobs.\n{_describe(result)}")


def test_async_submit_from_thread_outliving_main():
    """A non-daemon thread is joined before the shutdown hook runs.
    """
    result = subprocess.run([sys.executable, CHILD, "thread"],
                            capture_output=True,
                            text=True,
                            timeout=600)

    assert result.returncode == 0, (
        f"interpreter did not shut down cleanly.\n{_describe(result)}")
    # Verify the SUCCESS end marker.
    assert "SUCCESS" in result.stdout, (
        f"a thread outliving main could not submit asynchronous work.\n"
        f"{_describe(result)}")


@pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia-mqpu")),
    reason="a second platform instance requires the mqpu target")
def test_async_jobs_outstanding_on_swapped_out_platform():
    """Switching targets leaves the previous platform alive, queues included."""
    result = subprocess.run([sys.executable, CHILD, "target_swap"],
                            capture_output=True,
                            text=True,
                            timeout=600)

    assert result.returncode == 0, (
        f"interpreter did not shut down cleanly with jobs outstanding on a "
        f"platform that is no longer the current target.\n{_describe(result)}")
