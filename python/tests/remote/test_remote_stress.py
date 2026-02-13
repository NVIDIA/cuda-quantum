# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Stress test for remote-mqpu localhost connections.

Repeatedly runs the same run_async logic that fails intermittently on macOS CI
(Python 3.12) to gather diagnostic data.

See https://github.com/NVIDIA/cuda-quantum/issues/3910
    https://github.com/NVIDIA/cuda-quantum/issues/3931
"""

import os
import platform
import socket
import subprocess
import sys
import time
import warnings

import pytest

import cudaq

# ---------------------------------------------------------------------------
# Use the same configuration as test_remote_platform.py
# ---------------------------------------------------------------------------
num_qpus = 3
ITERATIONS = 50


# ---------------------------------------------------------------------------
# Fixtures -- identical to test_remote_platform.py
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def startUpMockServer():
    cudaq.set_target("remote-mqpu", auto_launch=str(num_qpus))
    yield
    cudaq.reset_target()


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


# ---------------------------------------------------------------------------
# Kernel -- identical to test_remote_platform.py
# ---------------------------------------------------------------------------
@cudaq.kernel
def simple(numQubits: int) -> int:
    qubits = cudaq.qvector(numQubits)
    h(qubits.front())
    for i, qubit in enumerate(qubits.front(numQubits - 1)):
        x.ctrl(qubit, qubits[i + 1])
    result = 0
    for i in range(numQubits):
        if mz(qubits[i]):
            result += 1
    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def _log_environment():
    """Print environment details useful for diagnosing CI failures."""
    print("\n" + "=" * 70)
    print("STRESS TEST ENVIRONMENT")
    print("=" * 70)
    print(f"Python:      {sys.version}")
    print(f"Platform:    {sys.platform} / {platform.platform()}")
    print(f"Machine:     {platform.machine()}")
    print(f"CPUs:        {os.cpu_count()}")
    print(f"PID:         {os.getpid()}")

    # TCP backlog (macOS)
    try:
        result = subprocess.run(["sysctl", "kern.ipc.somaxconn"],
                                capture_output=True,
                                text=True,
                                timeout=5)
        print(f"somaxconn:   {result.stdout.strip()}")
    except Exception:
        print("somaxconn:   (unavailable)")

    # mDNSResponder status (macOS) -- disabled on CI runners
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["launchctl", "print", "system/com.apple.mDNSResponder"],
                capture_output=True,
                text=True,
                timeout=5)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "state" in line.lower():
                        print(f"mDNS:        {line.strip()}")
                        break
                else:
                    print("mDNS:        running (no state line found)")
            else:
                print("mDNS:        NOT running (launchctl failed)")
        except Exception:
            print("mDNS:        (check unavailable)")

    # Test localhost resolution -- dual-stack (IPv4/IPv6) is relevant
    try:
        addrs = socket.getaddrinfo("localhost", None, socket.AF_UNSPEC,
                                   socket.SOCK_STREAM)
        families = set()
        for af, _, _, _, sa in addrs:
            name = "IPv6" if af == socket.AF_INET6 else "IPv4"
            families.add(f"{name}={sa[0]}")
        print(f"localhost:   {', '.join(sorted(families))}")
    except Exception as e:
        print(f"localhost:   resolution failed: {e}")

    print(f"cudaq:       {cudaq.__version__}")
    target = cudaq.get_target()
    print(f"target:      {target.name}, num_qpus={target.num_qpus()}")
    print("=" * 70 + "\n", flush=True)


def _log_failure(iteration, total, error, elapsed_ms):
    """Log detailed diagnostics when a run_async burst fails."""
    print("\n" + "-" * 70)
    print(f"FAILURE at iteration {iteration}/{total}")
    print(f"  Error:     {type(error).__name__}: {error}")
    print(f"  Elapsed:   {elapsed_ms:.0f} ms")
    print(f"  Time:      {time.strftime('%H:%M:%S')}")

    # Check if cudaq-qpud processes are alive
    try:
        result = subprocess.run(["pgrep", "-lf", "cudaq-qpud"],
                                capture_output=True,
                                text=True,
                                timeout=5)
        lines = result.stdout.strip()
        count = len(lines.splitlines()) if lines else 0
        print(f"  qpud procs: {count} alive")
        if lines:
            for line in lines.splitlines():
                print(f"    {line}")
    except Exception:
        print("  qpud procs: (check failed)")

    # Snapshot of TCP connections on localhost
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["netstat", "-an", "-p", "tcp"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5)
            localhost_lines = [
                l for l in result.stdout.splitlines()
                if "127.0.0.1" in l or "localhost" in l
            ]
            states = {}
            for l in localhost_lines:
                parts = l.split()
                if parts:
                    state = parts[-1]
                    states[state] = states.get(state, 0) + 1
            print(f"  TCP conns (localhost): {len(localhost_lines)}")
            if states:
                print(f"  TCP states: {states}")
        except Exception:
            pass

    print("-" * 70 + "\n", flush=True)


# ---------------------------------------------------------------------------
# Test -- exact same logic as test_run_async, repeated ITERATIONS times
# ---------------------------------------------------------------------------
def test_stress_run_async():
    """Repeatedly fire the same run_async pattern that fails on macOS CI.

    Identical to test_run_async in test_remote_platform.py but iterated
    with diagnostic logging.  Failures are reported as warnings, not
    assertion errors, so this test gathers data without blocking CI.

    See https://github.com/NVIDIA/cuda-quantum/issues/3910
    """
    _log_environment()

    shots = 10
    qubitCount = 4
    failures = []
    timings = []

    for iteration in range(1, ITERATIONS + 1):
        t0 = time.monotonic()
        try:
            # --- begin: identical to test_run_async ---
            result_futures = []
            for i in range(cudaq.get_target().num_qpus()):
                result = cudaq.run_async(simple,
                                         qubitCount,
                                         shots_count=shots,
                                         qpu_id=i)
                result_futures.append(result)

            for idx in range(len(result_futures)):
                res = result_futures[idx].get()
                assert len(res) == shots
            # --- end: identical to test_run_async ---

            elapsed = (time.monotonic() - t0) * 1000
            timings.append(elapsed)
            print(f"  [{iteration}/{ITERATIONS}] PASS  ({elapsed:.0f} ms)",
                  flush=True)

        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            timings.append(elapsed)
            failures.append((iteration, str(e), elapsed))
            _log_failure(iteration, ITERATIONS, e, elapsed)

    # Summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    total = len(timings)
    passes = total - len(failures)
    print(f"Iterations:    {total}")
    print(f"Passed:        {passes}")
    print(f"Failed:        {len(failures)}")
    if failures:
        print(f"Failure rate:  {len(failures)/total*100:.1f}%")
        for it, err, ms in failures:
            print(f"  #{it}: ({ms:.0f} ms) {err[:200]}")
    if timings:
        print(f"Timing (ms):   min={min(timings):.0f}, "
              f"max={max(timings):.0f}, "
              f"avg={sum(timings)/len(timings):.0f}")
    print("=" * 70 + "\n", flush=True)

    if failures:
        warnings.warn(
            f"[#3910 stress] {len(failures)}/{total} run_async bursts "
            f"failed on {sys.platform} Python "
            f"{sys.version_info.major}.{sys.version_info.minor}: "
            f"{failures[0][1][:150]}",
            RuntimeWarning,
            stacklevel=1)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP", "-v", "-s"])
