# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q must keep working after the thread that first used it exits.

Each case runs in a fresh interpreter, since lazy target init happens once.
"""

import subprocess
import sys
import textwrap

import pytest

PRELUDE = textwrap.dedent("""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import cudaq

    @cudaq.kernel
    def k():
        q = cudaq.qvector(2)
        h(q[0])
        x(q[1])

    H = cudaq.SpinOperator.from_word("ZZ")

    def obs():
        return cudaq.observe(k, H).expectation()

    def on_thread(fn):
        out = []
        t = threading.Thread(target=lambda: out.append(fn()))
        t.start()
        t.join()
        assert out, "worker thread raised"
        return out[0]
""")

CASES = {
    # The reproducer from the issue.
    "sequential_threads":
        """
        for _ in range(2):
            assert abs(on_thread(obs)) < 1e-9
        assert abs(obs()) < 1e-9
        assert cudaq.sample(k).most_probable() in ("01", "11")
        """,
    "thread_pool_shutdown":
        """
        with ThreadPoolExecutor(1) as ex:
            assert abs(ex.submit(obs).result()) < 1e-9
        assert abs(obs()) < 1e-9
        """,
    "concurrent_first_use":
        """
        results = []
        ts = [threading.Thread(target=lambda: results.append(obs()))
              for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert len(results) == 3 and all(abs(r) < 1e-9 for r in results)
        assert abs(obs()) < 1e-9
        """,
    # Target and noise configured on a worker must outlive it.
    "worker_configures_target":
        """
        nm = cudaq.NoiseModel()
        nm.add_all_qubit_channel("x", cudaq.BitFlipChannel(1.0))

        def configure():
            cudaq.set_target("density-matrix-cpu")
            cudaq.set_noise(nm)
            return True

        on_thread(configure)
        assert cudaq.get_target().name == "density-matrix-cpu"
        counts = cudaq.sample(k, shots_count=200)
        # The bit flip undoes `x`, so qubit 1 is always measured as 0.
        assert {b for b, _ in counts.items()} <= {"00", "10"}, counts
        on_thread(lambda: cudaq.sample(k))
        """,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_cudaq_usable_after_first_thread_exits(case, tmp_path):
    # Kernels need inspectable source, so the child runs from a file.
    child = tmp_path / f"child_{case}.py"
    child.write_text(PRELUDE + textwrap.dedent(CASES[case]))
    result = subprocess.run([sys.executable, "-W", "ignore",
                             str(child)],
                            capture_output=True,
                            text=True,
                            timeout=600)
    assert result.returncode == 0, (f"exit code {result.returncode}\n"
                                    f"--- stdout ---\n{result.stdout}\n"
                                    f"--- stderr ---\n{result.stderr}")
