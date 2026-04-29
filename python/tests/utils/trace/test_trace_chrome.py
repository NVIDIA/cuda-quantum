# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import json
import subprocess
import sys
import textwrap

import pytest

cudaq = pytest.importorskip("cudaq")
from cudaq.util import trace


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


def test_file_backed_chrome_backend_interleaves_python_and_mlir_pass(tmp_path):
    """A file-backed ChromeBackend captures the Python span plus every
    MLIR pass that runs inside it, and every pass event is contained in the
    outer Python span's time window."""
    # Reset target in case a prior xdist test on this worker set a remote
    # backend (e.g. backends/test_Quantinuum_*) and did not restore.
    cudaq.set_target("qpp-cpu")
    trace_path = tmp_path / "trace.json"

    trace.set_backend(trace.ChromeBackend(str(trace_path)))
    with trace.span("outer"):
        cudaq.sample(bell, shots_count=10)
    trace.reset_backend()

    doc = json.loads(trace_path.read_text())
    by_cat = {}
    for e in doc["traceEvents"]:
        by_cat.setdefault(e.get("cat"), []).append(e)

    outer = next(e for e in by_cat.get("python", []) if e["name"] == "outer")
    assert by_cat.get("mlir_pass"), "no mlir_pass events"

    outer_end = outer["ts"] + outer.get("dur", 0)
    for e in by_cat["mlir_pass"]:
        assert outer["ts"] <= e["ts"] <= e["ts"] + e.get("dur", 0) <= outer_end


def test_in_memory_chrome_backend_exposes_events_without_file():
    """A ChromeBackend constructed with no path is pure in-memory: to_json()
    and to_dict() agree, get_backend() round-trips, and the destructor must
    not write any file."""
    backend = trace.ChromeBackend()
    trace.set_backend(backend)
    assert trace.get_backend() is backend

    with trace.span("inmem"):
        pass
    trace.reset_backend()

    dict_form = backend.to_dict()
    assert dict_form == json.loads(backend.to_json())
    assert any(e["name"] == "inmem" and e["cat"] == "python"
               for e in dict_form["traceEvents"])


def test_builtin_python_phase_spans_wrap_kernel_lifecycle():
    """Built-in @trace.traced decorators emit spans covering the Python
    entry, JIT compile, and prepare_call / clone_module bridge phases."""
    # Reset target in case a prior xdist test on this worker set a remote
    # backend (e.g. backends/test_Quantinuum_*) and did not restore.
    cudaq.set_target("qpp-cpu")

    # Fresh kernel forces a JIT compile inside the traced region.
    @cudaq.kernel
    def fresh():
        q = cudaq.qvector(1)
        h(q[0])
        mz(q)

    backend = trace.ChromeBackend()
    trace.set_backend(backend)
    try:
        cudaq.sample(fresh, shots_count=10)
    finally:
        trace.reset_backend()

    events = backend.to_dict()["traceEvents"]
    names = {e["name"] for e in events if e.get("cat") == "python"}
    assert "cudaq.runtime.sample.sample" in names
    assert "cudaq.kernel.kernel_decorator.PyKernelDecorator.compile" in names
    assert "cudaq.kernel.kernel_decorator.PyKernelDecorator.prepare_call" in names
    assert "kernel.clone_module" in names
    # AST-bridge build span and the AOT pipeline marker emitted from
    # compile_to_mlir. Tooling attributes per-pass events to AOT vs JIT by
    # the cudaq.pipeline.* span ancestry, so a refactor that drops either
    # name needs to update both this test and the contract.
    assert "ast_bridge.build_module" in names
    assert "cudaq.pipeline.aot" in names


def test_file_backed_chrome_backend_writes_on_process_exit(tmp_path):
    """A file-backed ChromeBackend installed in a subprocess writes its JSON
    at process exit even without an explicit reset_backend / write_file /
    del. Pins the shared_ptr shutdown-ordering contract."""
    trace_path = tmp_path / "deferred.json"
    script = textwrap.dedent(f"""
        from cudaq.util import trace
        trace.set_backend(trace.ChromeBackend({str(trace_path)!r}))
        with trace.span("deferred"):
            pass
    """)
    subprocess.run([sys.executable, "-c", script], check=True)

    assert trace_path.exists()
    doc = json.loads(trace_path.read_text())
    assert any(e["name"] == "deferred" and e["cat"] == "python"
               for e in doc["traceEvents"])
