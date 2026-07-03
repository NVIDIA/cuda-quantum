# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for the bytecode kernel capture backend.

Covers all pulse operations, control flow, linear-type rebinding,
constant folding, qudit allocation patterns, and error cases.
"""

import math
import pytest

from cudaq_pulse.kernel.bytecode_bridge import compile_kernel_bytecode
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.kernel.decorator import kernel, qudit_ref, qvec_ref
from cudaq_pulse.ops import (
    get_drive_line,
    get_readout_line,
    gaussian,
    square,
    drag,
    cosine,
    tanh_ramp,
    gaussian_square,
    custom,
    drive,
    readout,
    wait,
    sync,
    shift_phase,
    set_phase,
    shift_frequency,
    set_frequency,
)

# ── Basic compilation ────────────────────────────────────────────────


class TestBasicCompilation:

    def test_empty_kernel(self):

        def empty(q0):
            pass

        ir = compile_kernel_bytecode(empty)(qudit_ref())
        assert ir.name == "empty"
        assert len(ir.ops) == 1  # just pulse.qudit_arg

    def test_kernel_name_preserved(self):

        def my_special_name(q0):
            pass

        ir = compile_kernel_bytecode(my_special_name)(qudit_ref())
        assert ir.name == "my_special_name"

    def test_drive_kernel(self):

        def drive_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(drive_test)(qudit_ref())
        kinds = [op.kind for op in ir.ops]
        assert kinds == [
            "pulse.qudit_arg",
            "pulse.get_drive_line",
            "pulse.gaussian",
            "pulse.drive",
        ]

    def test_multiple_args(self):

        def two_qubit(q0, q1):
            d0, t0 = get_drive_line(q0)
            d1, t1 = get_drive_line(q1)

        ir = compile_kernel_bytecode(two_qubit)(qudit_ref(), qudit_ref())
        arg_ops = [op for op in ir.ops if op.kind == "pulse.qudit_arg"]
        assert len(arg_ops) == 2

    def test_wrong_arg_count_raises(self):

        def one_arg(q0):
            pass

        emitter = compile_kernel_bytecode(one_arg)
        with pytest.raises(CompilationError, match="expected 1 args, got 0"):
            emitter()

    def test_kwargs_rejected(self):

        def one_arg(q0):
            pass

        emitter = compile_kernel_bytecode(one_arg)
        with pytest.raises(CompilationError, match="keyword arguments"):
            emitter(q0=qudit_ref())


# ── Waveform creation ────────────────────────────────────────────────


class TestWaveforms:

    def test_gaussian(self):

        def wf_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(wf_test)(qudit_ref())
        wf_op = next(op for op in ir.ops if op.kind == "pulse.gaussian")
        assert wf_op.attrs == {"duration": 40, "amplitude": 0.3, "sigma": 10.0}

    def test_square(self):

        def wf_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = square(20, 0.5)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(wf_test)(qudit_ref())
        wf_op = next(op for op in ir.ops if op.kind == "pulse.square")
        assert wf_op.attrs == {"duration": 20, "amplitude": 0.5}

    def test_drag(self):

        def wf_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = drag(40, 0.435, 5.0, 0.75)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(wf_test)(qudit_ref())
        wf_op = next(op for op in ir.ops if op.kind == "pulse.drag")
        assert wf_op.attrs == {
            "duration": 40,
            "amplitude": 0.435,
            "sigma": 5.0,
            "beta": 0.75,
        }

    def test_cosine(self):

        def wf_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = cosine(100, 0.5, 1e6)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(wf_test)(qudit_ref())
        assert any(op.kind == "pulse.cosine" for op in ir.ops)

    def test_gaussian_square(self):

        def wf_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = gaussian_square(200, 0.3, 10.0, 150.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(wf_test)(qudit_ref())
        wf_op = next(op for op in ir.ops if op.kind == "pulse.gaussian_square")
        assert wf_op.attrs["width"] == 150.0

    def test_tanh_ramp(self):

        def wf_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = tanh_ramp(50, 0.4, 5.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(wf_test)(qudit_ref())
        assert any(op.kind == "pulse.tanh_ramp" for op in ir.ops)


# ── Readout and measurement ──────────────────────────────────────────


class TestReadout:

    def test_readout_produces_measurement(self):

        def ro_test(q0):
            r0, t0 = get_readout_line(q0)
            wf = square(600, 0.1)
            readout(r0, wf, t0)

        ir = compile_kernel_bytecode(ro_test)(qudit_ref())
        ro_ops = [op for op in ir.ops if op.kind == "pulse.readout"]
        assert len(ro_ops) == 1
        assert len(ro_ops[0].results) == 3
        assert ro_ops[0].results[2].vtype == "measurement"


# ── Tone manipulation ops ────────────────────────────────────────────


class TestToneOps:

    def test_shift_phase(self):

        def phase_test(q0):
            d0, t0 = get_drive_line(q0)
            t0 = shift_phase(t0, 1.5707)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(phase_test)(qudit_ref())
        sp = next(op for op in ir.ops if op.kind == "pulse.shift_phase")
        assert abs(sp.attrs["phase_rad"] - 1.5707) < 1e-10

    def test_set_phase(self):

        def phase_test(q0):
            d0, t0 = get_drive_line(q0)
            t0 = set_phase(t0, 0.0)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(phase_test)(qudit_ref())
        assert any(op.kind == "pulse.set_phase" for op in ir.ops)

    def test_shift_frequency(self):

        def freq_test(q0):
            d0, t0 = get_drive_line(q0)
            t0 = shift_frequency(t0, 1e6)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(freq_test)(qudit_ref())
        sf = next(op for op in ir.ops if op.kind == "pulse.shift_frequency")
        assert sf.attrs["freq_hz"] == 1e6

    def test_set_frequency(self):

        def freq_test(q0):
            d0, t0 = get_drive_line(q0)
            t0 = set_frequency(t0, 5.1e9)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(freq_test)(qudit_ref())
        assert any(op.kind == "pulse.set_frequency" for op in ir.ops)


# ── Wait and sync ────────────────────────────────────────────────────


class TestWaitSync:

    def test_wait(self):

        def wait_test(q0):
            d0, t0 = get_drive_line(q0)
            wait(d0, 100)

        ir = compile_kernel_bytecode(wait_test)(qudit_ref())
        w = next(op for op in ir.ops if op.kind == "pulse.wait")
        assert w.attrs["duration"] == 100

    def test_sync_variadic(self):

        def sync_test(q0, q1):
            d0, t0 = get_drive_line(q0)
            d1, t1 = get_drive_line(q1)
            sync(d0, d1)

        ir = compile_kernel_bytecode(sync_test)(qudit_ref(), qudit_ref())
        s = next(op for op in ir.ops if op.kind == "pulse.sync")
        assert len(s.operands) == 2
        assert all(o.vtype == "drive_line" for o in s.operands)


# ── Linear-type rebinding ────────────────────────────────────────────


class TestLinearRebinding:

    def test_drive_rebinds_line_and_tone(self):

        def rebind_test(q0):
            d0, t0 = get_drive_line(q0)
            wf1 = gaussian(40, 0.3, 10.0)
            wf2 = gaussian(40, 0.5, 10.0)
            drive(d0, wf1, t0)
            drive(d0, wf2, t0)

        ir = compile_kernel_bytecode(rebind_test)(qudit_ref())
        drives = [op for op in ir.ops if op.kind == "pulse.drive"]
        assert len(drives) == 2
        assert drives[0].operands[0] is not drives[1].operands[0]
        assert drives[0].operands[2] is not drives[1].operands[2]

    def test_wait_rebinds_line(self):

        def rebind_test(q0):
            d0, t0 = get_drive_line(q0)
            wait(d0, 50)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(rebind_test)(qudit_ref())
        w = next(op for op in ir.ops if op.kind == "pulse.wait")
        d = next(op for op in ir.ops if op.kind == "pulse.drive")
        assert w.operands[0] is not d.operands[0]

    def test_shift_phase_rebinds_tone(self):

        def phase_test(q0):
            d0, t0 = get_drive_line(q0)
            t0 = shift_phase(t0, 0.5)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(phase_test)(qudit_ref())
        sp = next(op for op in ir.ops if op.kind == "pulse.shift_phase")
        d = next(op for op in ir.ops if op.kind == "pulse.drive")
        assert sp.results[0] is d.operands[2]


# ── For loops ────────────────────────────────────────────────────────


class TestForLoop:

    def test_for_range_emits_scf(self):

        def loop_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 10.0)
            for i in range(5):
                drive(d0, wf, t0)

        ir = compile_kernel_bytecode(loop_test)(qudit_ref())
        kinds = [op.kind for op in ir.ops]
        assert "scf.for" in kinds
        assert "scf.for_end" in kinds
        scf_for = next(op for op in ir.ops if op.kind == "scf.for")
        assert scf_for.attrs["ub"] == 5

    def test_for_loop_with_yield(self):

        def loop_yield(q0):
            d0, t0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 10.0)
            for i in range(3):
                drive(d0, wf, t0)

        ir = compile_kernel_bytecode(loop_yield)(qudit_ref())
        kinds = [op.kind for op in ir.ops]
        assert "scf.yield" in kinds


# ── Qudit allocation patterns ────────────────────────────────────────


class TestQuditAlloc:

    def test_internal_qudit_ref(self):

        def alloc_test():
            q = qudit_ref()
            d0, t0 = get_drive_line(q)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(alloc_test)()
        kinds = [op.kind for op in ir.ops]
        assert "pulse.qudit_alloc" in kinds
        alloc = next(op for op in ir.ops if op.kind == "pulse.qudit_alloc")
        assert alloc.results[0].vtype == "qref"

    def test_attribute_style_alloc(self):
        import cudaq_pulse

        def attr_alloc():
            q = cudaq_pulse.qudit_ref()
            d0, t0 = cudaq_pulse.get_drive_line(q)

        ir = compile_kernel_bytecode(attr_alloc)()
        kinds = [op.kind for op in ir.ops]
        assert "pulse.qudit_alloc" in kinds
        assert "pulse.get_drive_line" in kinds


# ── Constant folding / arithmetic ────────────────────────────────────


class TestConstantFolding:

    def test_binary_add(self):

        def arith_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = gaussian(20 + 20, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(arith_test)(qudit_ref())
        wf = next(op for op in ir.ops if op.kind == "pulse.gaussian")
        assert wf.attrs["duration"] == 40

    def test_binary_mul(self):

        def arith_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = gaussian(40, 0.3 * 2, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(arith_test)(qudit_ref())
        wf = next(op for op in ir.ops if op.kind == "pulse.gaussian")
        assert abs(wf.attrs["amplitude"] - 0.6) < 1e-10

    def test_binary_div(self):

        def arith_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 20.0 / 2)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(arith_test)(qudit_ref())
        wf = next(op for op in ir.ops if op.kind == "pulse.gaussian")
        assert abs(wf.attrs["sigma"] - 10.0) < 1e-10

    def test_unary_negation(self):

        def neg_test(q0):
            d0, t0 = get_drive_line(q0)
            wf = gaussian(40, -0.3, 10.0)
            drive(d0, wf, t0)

        ir = compile_kernel_bytecode(neg_test)(qudit_ref())
        wf = next(op for op in ir.ops if op.kind == "pulse.gaussian")
        assert abs(wf.attrs["amplitude"] - (-0.3)) < 1e-10


# ── Source-less compilation ──────────────────────────────────────────


class TestSourceless:

    def test_exec_compiled_function(self):
        code = compile(
            "def f(q0):\n d0, t0 = get_drive_line(q0)\n",
            "<test>",
            "exec",
        )
        ns = {"get_drive_line": get_drive_line}
        exec(code, ns)
        fn = ns["f"]

        ir = compile_kernel_bytecode(fn)(qudit_ref())
        assert any(op.kind == "pulse.get_drive_line" for op in ir.ops)


# ── Error cases ──────────────────────────────────────────────────────


class TestErrors:

    def test_unknown_op_raises(self):
        """Calling a function that isn't a pulse op or known builtin raises."""

        def bad_kernel(q0):
            d0, t0 = get_drive_line(q0)
            unknown_function_xyz(d0)

        emitter = compile_kernel_bytecode(bad_kernel)
        with pytest.raises(CompilationError):
            emitter(qudit_ref())

    def test_for_list_not_supported(self):
        """for i in [1,2,3] is not range-based, should fail gracefully."""

        def bad_loop(q0):
            d0, t0 = get_drive_line(q0)
            for i in [1, 2, 3]:
                wait(d0, 10)

        emitter = compile_kernel_bytecode(bad_loop)
        q = qudit_ref()
        with pytest.raises((CompilationError, TypeError)):
            emitter(q)


# ── Decorator integration ────────────────────────────────────────────


class TestDecoratorIntegration:

    def test_kernel_decorator_uses_bytecode(self):

        @kernel
        def my_kernel(q0):
            d0, t0 = get_drive_line(q0)
            wf = gaussian(40, 0.3, 10.0)
            drive(d0, wf, t0)

        ir = my_kernel(qudit_ref())
        assert ir is not None
        kinds = [op.kind for op in ir.ops]
        assert "pulse.drive" in kinds

    def test_kernel_caching(self):

        @kernel
        def cached(q0):
            pass

        cached(qudit_ref())
        e1 = cached.__cudaq_pulse_emitter__
        cached(qudit_ref())
        e2 = cached.__cudaq_pulse_emitter__
        assert e1 is e2

    def test_kernel_internal_alloc(self):

        @kernel
        def internal():
            q = qudit_ref()
            d0, t0 = get_drive_line(q)

        ir = internal()
        kinds = [op.kind for op in ir.ops]
        assert "pulse.qudit_alloc" in kinds
