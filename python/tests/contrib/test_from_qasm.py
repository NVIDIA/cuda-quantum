# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the native OpenQASM 2.0 / 3.0 → CUDA-Q translator.

These tests exercise `cudaq.contrib.from_qasm_str` (and `from_qasm` for the
file-based entry point). The native parser has no Qiskit dependency.
"""

import os
import tempfile
import textwrap

import pytest

import cudaq
from cudaq.contrib import from_qasm, from_qasm_str
from cudaq.contrib.qasm_convert import _eval_expr

HEADER_2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _run(qasm_src, shots=1000):
    """Helper: parse source, sample, return counts."""
    kernel = from_qasm_str(qasm_src)
    return cudaq.sample(kernel, shots_count=shots)


# --------------------------------------------------------------------------- #
# Expression evaluator (unit-level tests)
# --------------------------------------------------------------------------- #


class TestExpressionEvaluator:

    def test_numeric_literals(self):
        assert _eval_expr("0") == 0.0
        assert _eval_expr("3.14") == pytest.approx(3.14)
        assert _eval_expr("1e-3") == pytest.approx(1e-3)

    def test_arithmetic(self):
        assert _eval_expr("1 + 2 * 3") == pytest.approx(7.0)
        assert _eval_expr("(1 + 2) * 3") == pytest.approx(9.0)
        assert _eval_expr("10 / 4") == pytest.approx(2.5)
        assert _eval_expr("-5") == pytest.approx(-5.0)
        assert _eval_expr("+5") == pytest.approx(5.0)

    def test_constants(self):
        import math
        assert _eval_expr("pi") == pytest.approx(math.pi)
        assert _eval_expr("pi / 2") == pytest.approx(math.pi / 2)
        assert _eval_expr("2 * pi") == pytest.approx(2 * math.pi)

    def test_functions(self):
        import math
        assert _eval_expr("sin(0)") == pytest.approx(0.0)
        assert _eval_expr("cos(0)") == pytest.approx(1.0)
        assert _eval_expr("sqrt(2)") == pytest.approx(math.sqrt(2))
        assert _eval_expr("exp(0)") == pytest.approx(1.0)
        assert _eval_expr("sin(pi/2)") == pytest.approx(1.0)

    def test_power_operators(self):
        assert _eval_expr("2 ** 3") == pytest.approx(8.0)
        # QASM `^` is exponentiation (Python's `ast` parses as BitXor, handled).
        assert _eval_expr("2 ^ 3") == pytest.approx(8.0)

    def test_env_variables(self):
        assert _eval_expr("theta / 2", {"theta": 1.0}) == pytest.approx(0.5)
        assert _eval_expr("a + b", {"a": 2.0, "b": 3.0}) == pytest.approx(5.0)

    def test_unknown_identifier_raises(self):
        with pytest.raises(ValueError):
            _eval_expr("unknown_var")

    def test_unknown_function_raises(self):
        with pytest.raises(ValueError):
            _eval_expr("floor(1.5)")


# --------------------------------------------------------------------------- #
# Basic single- and two-qubit gates
# --------------------------------------------------------------------------- #


class TestBasicGates:

    def test_identity(self):
        src = HEADER_2 + "qreg q[1];\nid q[0];\n"
        counts = _run(src)
        assert counts['0'] == 1000

    def test_x(self):
        src = HEADER_2 + "qreg q[1];\nx q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_y(self):
        src = HEADER_2 + "qreg q[1];\ny q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_z_on_zero_is_identity(self):
        src = HEADER_2 + "qreg q[1];\nz q[0];\n"
        counts = _run(src)
        assert counts['0'] == 1000

    def test_h_superposition(self):
        src = HEADER_2 + "qreg q[1];\nh q[0];\n"
        counts = _run(src)
        assert '0' in counts
        assert '1' in counts

    def test_s_then_sdg_is_identity(self):
        src = HEADER_2 + "qreg q[1];\nh q[0];\ns q[0];\nsdg q[0];\nh q[0];\n"
        counts = _run(src)
        assert counts['0'] == 1000

    def test_t_then_tdg_is_identity(self):
        src = HEADER_2 + "qreg q[1];\nh q[0];\nt q[0];\ntdg q[0];\nh q[0];\n"
        counts = _run(src)
        assert counts['0'] == 1000

    def test_sx_twice_is_x(self):
        src = HEADER_2 + "qreg q[1];\nsx q[0];\nsx q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_rx_pi(self):
        src = HEADER_2 + "qreg q[1];\nrx(pi) q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_ry_pi(self):
        src = HEADER_2 + "qreg q[1];\nry(pi) q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_rz_does_not_flip(self):
        src = HEADER_2 + "qreg q[1];\nrz(pi) q[0];\n"
        counts = _run(src)
        assert counts['0'] == 1000

    def test_u3(self):
        # u3(pi, 0, pi) ≡ X up to a global phase
        src = HEADER_2 + "qreg q[1];\nu3(pi, 0, pi) q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_u2(self):
        # u2(0, pi) ≡ H up to a global phase
        src = HEADER_2 + "qreg q[1];\nu2(0, pi) q[0];\n"
        counts = _run(src)
        assert '0' in counts and '1' in counts

    def test_u1_phase_on_zero_is_noop(self):
        src = HEADER_2 + "qreg q[1];\nu1(pi/3) q[0];\n"
        counts = _run(src)
        assert counts['0'] == 1000


# --------------------------------------------------------------------------- #
# Multi-qubit gates
# --------------------------------------------------------------------------- #


class TestMultiQubit:

    def test_cx_bell(self):
        src = HEADER_2 + "qreg q[2];\nh q[0];\ncx q[0], q[1];\n"
        counts = _run(src)
        assert '00' in counts
        assert '11' in counts
        assert '01' not in counts
        assert '10' not in counts

    def test_cz_on_11(self):
        # CZ acts as phase on |11>; not observable in Z-basis.
        src = HEADER_2 + "qreg q[2];\nx q[0];\nx q[1];\ncz q[0], q[1];\n"
        counts = _run(src)
        assert counts['11'] == 1000

    def test_swap(self):
        src = HEADER_2 + "qreg q[2];\nx q[0];\nswap q[0], q[1];\n"
        counts = _run(src)
        # q0 had |1>, after swap q1 holds |1>. CUDA-Q big-endian: '01'.
        assert counts['01'] == 1000

    def test_ccx_toffoli(self):
        src = HEADER_2 + "qreg q[3];\nx q[0];\nx q[1];\nccx q[0], q[1], q[2];\n"
        counts = _run(src)
        assert counts['111'] == 1000

    def test_cswap(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            x q[0];
            x q[1];
            cswap q[0], q[1], q[2];
        """)
        counts = _run(src)
        # control=|1>, targets q1=|1>, q2=|0> → swap → q1=|0>, q2=|1>
        assert counts['101'] == 1000

    def test_rxx(self):
        src = HEADER_2 + "qreg q[2];\nrxx(pi) q[0], q[1];\n"
        counts = _run(src)
        # RXX(pi) |00> = -i|11>; measurement gives '11'
        assert counts['11'] == 1000

    def test_rzz_on_zero_is_phase_only(self):
        src = HEADER_2 + "qreg q[2];\nrzz(pi/2) q[0], q[1];\n"
        counts = _run(src)
        assert counts['00'] == 1000

    def test_crx(self):
        # Control=|0>: no effect.
        src = HEADER_2 + "qreg q[2];\ncrx(pi) q[0], q[1];\n"
        counts = _run(src)
        assert counts['00'] == 1000

        # Control=|1>: target flips.
        src = (HEADER_2 + "qreg q[2];\nx q[0];\ncrx(pi) q[0], q[1];\n")
        counts = _run(src)
        assert counts['11'] == 1000

    def test_cy(self):
        # CY with control=|1> flips the target (up to a phase in Z-basis).
        src = HEADER_2 + "qreg q[2];\nx q[0];\ncy q[0], q[1];\n"
        counts = _run(src)
        assert counts['11'] == 1000

    def test_ch(self):
        # CH with control=|1> puts target in superposition.
        src = HEADER_2 + "qreg q[2];\nx q[0];\nch q[0], q[1];\n"
        counts = _run(src)
        assert '10' in counts
        assert '11' in counts

    def test_cu1_noop_on_zero(self):
        # cu1(λ) only applies a phase on |11>; on |00> it is a no-op.
        src = HEADER_2 + "qreg q[2];\ncu1(pi/3) q[0], q[1];\n"
        counts = _run(src)
        assert counts['00'] == 1000

    def test_cu1_phase_on_11_still_measures_11(self):
        # Phase has no effect on Z-basis measurement, so the outcome is still |11>.
        src = (HEADER_2 + "qreg q[2];\nx q[0];\nx q[1];\n"
               "cu1(pi) q[0], q[1];\n")
        counts = _run(src)
        assert counts['11'] == 1000

    def test_cu3(self):
        # cu3(pi, 0, pi) with control=|1> behaves like X on the target
        # (u3(pi, 0, pi) ≡ X up to a global phase).
        src = (HEADER_2 + "qreg q[2];\nx q[0];\n"
               "cu3(pi, 0, pi) q[0], q[1];\n")
        counts = _run(src)
        assert counts['11'] == 1000

    def test_ryy_full_period(self):
        # RYY(2π) = -I, a global phase in the Z-basis: state stays |00>.
        src = HEADER_2 + "qreg q[2];\nryy(2*pi) q[0], q[1];\n"
        counts = _run(src)
        assert counts['00'] == 1000

    def test_ryy_pi_flips_both(self):
        # RYY(π) = exp(-iπ/2 Y⊗Y) = cos(π/2)·I - i sin(π/2)·Y⊗Y = -i Y⊗Y.
        # Applied to |00>: Y⊗Y|00> = -|11>, so RYY(π)|00> = i|11> → '11'.
        src = HEADER_2 + "qreg q[2];\nryy(pi) q[0], q[1];\n"
        counts = _run(src)
        assert counts['11'] == 1000


# --------------------------------------------------------------------------- #
# Register broadcasting
# --------------------------------------------------------------------------- #


class TestBroadcasting:

    def test_h_broadcast_across_register(self):
        src = HEADER_2 + "qreg q[3];\nh q;\n"
        counts = _run(src)
        # 2^3 = 8 distinct outcomes expected, all sampled at least once usually
        assert len(counts) >= 4  # not deterministic but very likely

    def test_x_broadcast(self):
        src = HEADER_2 + "qreg q[3];\nx q;\n"
        counts = _run(src)
        assert counts['111'] == 1000

    def test_cx_parallel_registers(self):
        # `cx` applied parallel element-wise.
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg a[2];
            qreg b[2];
            x a[0];
            x a[1];
            cx a, b;
        """)
        counts = _run(src)
        # a=|11>, after broadcast `cx`: b=|11>, so state |a=11, b=11> = '1111'
        assert counts['1111'] == 1000

    def test_mismatched_register_sizes_raises(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg a[2];
            qreg b[3];
            cx a, b;
        """)
        with pytest.raises(ValueError):
            from_qasm_str(src)


# --------------------------------------------------------------------------- #
# Custom gate definitions
# --------------------------------------------------------------------------- #


class TestCustomGates:

    def test_simple_custom_gate(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            gate flip a { x a; }
            qreg q[1];
            flip q[0];
        """)
        counts = _run(src)
        assert counts['1'] == 1000

    def test_parametric_custom_gate(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            gate my_ry(theta) a { ry(theta) a; }
            qreg q[1];
            my_ry(pi) q[0];
        """)
        counts = _run(src)
        assert counts['1'] == 1000

    def test_custom_gate_multi_qubit(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            gate bell a, b { h a; cx a, b; }
            qreg q[2];
            bell q[0], q[1];
        """)
        counts = _run(src)
        assert '00' in counts and '11' in counts
        assert '01' not in counts and '10' not in counts

    def test_custom_gate_calling_custom_gate(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            gate flip a { x a; }
            gate double_flip a { flip a; flip a; }
            qreg q[1];
            double_flip q[0];
        """)
        counts = _run(src)
        assert counts['0'] == 1000

    def test_custom_gate_with_expression_in_body(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            gate half_ry(theta) a { ry(theta/2) a; }
            qreg q[1];
            half_ry(2*pi) q[0];
        """)
        counts = _run(src)
        # `ry(pi)` on |0> → |1>
        assert counts['1'] == 1000

    def test_custom_gate_wrong_arity_raises(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            gate bell a, b { h a; cx a, b; }
            qreg q[3];
            bell q[0], q[1], q[2];
        """)
        with pytest.raises(ValueError):
            from_qasm_str(src)


# --------------------------------------------------------------------------- #
# Measurement, reset, barrier, comments
# --------------------------------------------------------------------------- #


class TestNonGateStatements:

    def test_measure(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            x q[0];
            measure q[0] -> c[0];
        """)
        counts = _run(src)
        assert counts['1'] == 1000

    def test_reset(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            x q[0];
            reset q[0];
        """)
        counts = _run(src)
        assert counts['0'] == 1000

    def test_barrier_is_noop(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            h q[0];
            barrier q[0], q[1];
            cx q[0], q[1];
        """)
        counts = _run(src)
        assert '00' in counts and '11' in counts

    def test_line_comments_stripped(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;  // version header
            include "qelib1.inc";
            // allocate one qubit
            qreg q[1];
            x q[0]; // flip it
        """)
        counts = _run(src)
        assert counts['1'] == 1000

    def test_block_comments_stripped(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            /* This is a
               multi-line comment */
            qreg q[1];
            x q[0];
        """)
        counts = _run(src)
        assert counts['1'] == 1000


# --------------------------------------------------------------------------- #
# File-based entry point
# --------------------------------------------------------------------------- #


class TestFromQasmFile:

    def test_reads_file(self, tmp_path):
        qasm_path = tmp_path / "bell.qasm"
        qasm_path.write_text(
            textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            h q[0];
            cx q[0], q[1];
        """))
        kernel = from_qasm(str(qasm_path))
        counts = cudaq.sample(kernel)
        assert '00' in counts and '11' in counts

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            from_qasm("/nonexistent/path/no.qasm")


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


class TestErrors:

    def test_missing_header_raises(self):
        src = "qreg q[1];\nx q[0];\n"
        with pytest.raises(ValueError, match="OPENQASM"):
            from_qasm_str(src)

    def test_unknown_qasm_version_raises(self):
        src = 'OPENQASM 4.0;\nqreg q[1];\nx q[0];\n'
        with pytest.raises(NotImplementedError, match="4.0"):
            from_qasm_str(src)

    def test_unsupported_gate_raises(self):
        src = HEADER_2 + "qreg q[1];\nmagic_gate q[0];\n"
        with pytest.raises(ValueError, match="magic_gate"):
            from_qasm_str(src)

    def test_if_statement_raises(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            measure q[0] -> c[0];
            if (c==1) x q[0];
        """)
        with pytest.raises(NotImplementedError):
            from_qasm_str(src)

    def test_unknown_qreg_raises(self):
        src = HEADER_2 + "qreg q[1];\nh r[0];\n"
        with pytest.raises(ValueError, match="Unknown qreg"):
            from_qasm_str(src)


# --------------------------------------------------------------------------- #
# OpenQASM 3.0
# --------------------------------------------------------------------------- #

HEADER_3 = 'OPENQASM 3.0;\ninclude "stdgates.inc";\n'


class TestQASM3Declarations:

    def test_qubit_array_declaration(self):
        src = HEADER_3 + "qubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\n"
        counts = _run(src)
        assert '00' in counts and '11' in counts
        assert '01' not in counts and '10' not in counts

    def test_single_qubit_declaration(self):
        src = HEADER_3 + "qubit q;\nx q;\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_multiple_qubit_arrays(self):
        src = textwrap.dedent("""\
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] a;
            qubit[1] b;
            x a[0];
        """)
        counts = _run(src)
        # a is allocated first → leftmost (big-endian) → '10'.
        assert counts['10'] == 1000

    def test_bit_declarations_ignored(self):
        # `bit[N]` / `bit` are tracked but never observed at runtime.
        src = HEADER_3 + "qubit[1] q;\nbit c;\nbit[3] cc;\nx q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_header_without_stdgates_include(self):
        # `include "stdgates.inc";` is optional — our handler table covers it.
        src = 'OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n'
        counts = _run(src)
        assert '0' in counts and '1' in counts

    def test_qasm_3_no_dot_version(self):
        src = 'OPENQASM 3;\nqubit[1] q;\nx q[0];\n'
        counts = _run(src)
        assert counts['1'] == 1000


class TestQASM3Measurement:

    def test_measurement_assignment_scalar(self):
        src = textwrap.dedent("""\
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[1] q;
            bit c;
            x q[0];
            c = measure q[0];
        """)
        counts = _run(src)
        assert counts['1'] == 1000

    def test_measurement_assignment_register(self):
        src = textwrap.dedent("""\
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            bit[2] c;
            x q[0];
            x q[1];
            c = measure q;
        """)
        counts = _run(src)
        assert counts['11'] == 1000

    def test_measurement_indexed_assignment(self):
        src = textwrap.dedent("""\
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            bit[2] c;
            x q[0];
            c[0] = measure q[0];
        """)
        counts = _run(src)
        # Only q[0] is explicitly measured (q[1] never enters the `mz` list), so
        # the sample bitstring has length 1: q[0] = |1> → '1'.
        assert counts['1'] == 1000


class TestQASM3Builtins:

    def test_U_three_params_as_x(self):
        # U(pi, 0, pi) ≡ X
        src = HEADER_3 + "qubit[1] q;\nU(pi, 0, pi) q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_U_three_params_as_h(self):
        # U(pi/2, 0, pi) ≡ H
        src = HEADER_3 + "qubit[1] q;\nU(pi/2, 0, pi) q[0];\n"
        counts = _run(src)
        assert '0' in counts and '1' in counts

    def test_U_four_params_gamma_unobservable(self):
        # 4-parameter U(θ, φ, λ, γ): γ is a global phase; ignored for sampling.
        src = HEADER_3 + "qubit[1] q;\nU(pi, 0, pi, 1.234) q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_U_wrong_param_count_raises(self):
        src = HEADER_3 + "qubit[1] q;\nU(pi, 0) q[0];\n"
        with pytest.raises(ValueError, match="'U' expects 3 or 4"):
            from_qasm_str(src)

    def test_gphase_is_noop(self):
        src = HEADER_3 + "qubit[1] q;\ngphase(pi/4);\nx q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_CX_alias(self):
        src = HEADER_3 + "qubit[2] q;\nh q[0];\nCX q[0], q[1];\n"
        counts = _run(src)
        assert '00' in counts and '11' in counts
        assert '01' not in counts and '10' not in counts


class TestQASM3Gates:

    def test_bell_state_via_stdgates(self):
        src = HEADER_3 + "qubit[2] q;\nh q[0];\ncx q[0], q[1];\n"
        counts = _run(src)
        assert '00' in counts and '11' in counts

    def test_rotation_gates(self):
        src = HEADER_3 + "qubit[1] q;\nrx(pi) q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000

    def test_broadcast_on_qubit_array(self):
        src = HEADER_3 + "qubit[3] q;\nh q;\n"
        counts = _run(src)
        # All 8 outcomes should be reachable in a uniform superposition.
        assert len(counts) >= 4

    def test_ccx_3qubit(self):
        src = HEADER_3 + "qubit[3] q;\nx q[0];\nx q[1];\nccx q[0], q[1], q[2];\n"
        counts = _run(src)
        assert counts['111'] == 1000


class TestQASM3GateCoverage:
    """Exercises every OpenQASM 3.0 gate listed in the project reference
    table. Each test `parse+sample`s a tiny kernel to confirm the gate name
    is accepted by the translator and reaches the CUDA-Q simulator."""

    # --- 1-qubit / Paulis / Clifford ---------------------------------------

    def test_I_uppercase_identity(self):
        # Per the QASM 3.0 reference, `I` (capital) is the identity.
        src = HEADER_3 + "qubit[1] q;\nI q[0];\n"
        counts = _run(src)
        assert counts['0'] == 1000

    def test_id_lowercase_identity(self):
        # The `stdgates.inc` file used by Qiskit spells identity as `id`; both must work.
        src = HEADER_3 + "qubit[1] q;\nid q[0];\n"
        counts = _run(src)
        assert counts['0'] == 1000

    def test_sx_and_sxdg(self):
        # `sx` ∘ `sx` = X.
        src = HEADER_3 + "qubit[1] q;\nsx q[0];\nsx q[0];\n"
        counts = _run(src)
        assert counts['1'] == 1000
        # `sx` ∘ `sxdg` = I.
        src2 = HEADER_3 + "qubit[1] q;\nx q[0];\nsx q[0];\nsxdg q[0];\n"
        counts2 = _run(src2)
        assert counts2['1'] == 1000

    def test_s_sdg_t_tdg(self):
        # Full round-trip: `s` `t` `tdg` `sdg` on |+> leaves |+>.
        src = HEADER_3 + textwrap.dedent("""\
            qubit[1] q;
            h q[0];
            s q[0];
            t q[0];
            tdg q[0];
            sdg q[0];
            h q[0];
        """)
        counts = _run(src)
        assert counts['0'] == 1000

    # --- 1-qubit phase / parametric ---------------------------------------

    def test_p_phase_on_one(self):
        # p(λ) on |0> is a no-op.
        src = HEADER_3 + "qubit[1] q;\np(pi/4) q[0];\n"
        counts = _run(src)
        assert counts['0'] == 1000

    def test_rx_ry_rz(self):
        # `rx(pi)` ≡ X (up to global phase); `ry(pi)` ≡ Y; `rz` does not flip.
        assert _run(HEADER_3 + "qubit[1] q;\nrx(pi) q[0];\n")['1'] == 1000
        assert _run(HEADER_3 + "qubit[1] q;\nry(pi) q[0];\n")['1'] == 1000
        assert _run(HEADER_3 + "qubit[1] q;\nrz(pi) q[0];\n")['0'] == 1000

    # --- Two-qubit controlled ---------------------------------------------

    def test_cx_cy_cz_ch(self):
        # `cx` / `cy` on |10>: flip the target. `cz` / `ch`: phase or superposition.
        for gate, expected in [('cx', '11'), ('cy', '11')]:
            src = HEADER_3 + f"qubit[2] q;\nx q[0];\n{gate} q[0], q[1];\n"
            counts = _run(src)
            assert counts[expected] == 1000
        # `cz` on |11> leaves phase only → measurement stays '11'.
        src = HEADER_3 + "qubit[2] q;\nx q[0];\nx q[1];\ncz q[0], q[1];\n"
        assert _run(src)['11'] == 1000
        # `ch` on |10>: control=|1> → target gets H → 50/50 '10' vs '11'.
        src = HEADER_3 + "qubit[2] q;\nx q[0];\nch q[0], q[1];\n"
        counts = _run(src)
        assert '10' in counts and '11' in counts

    def test_cs_csdg(self):
        # CS on |11> is a T phase; CS∘CSdg = I.
        src = HEADER_3 + textwrap.dedent("""\
            qubit[2] q;
            x q[0];
            x q[1];
            cs q[0], q[1];
            csdg q[0], q[1];
        """)
        counts = _run(src)
        assert counts['11'] == 1000

    def test_cp_phase(self):
        # `cp(θ)` on |11> → phase only, measurement stays '11'.
        src = HEADER_3 + textwrap.dedent("""\
            qubit[2] q;
            x q[0];
            x q[1];
            cp(pi/3) q[0], q[1];
        """)
        counts = _run(src)
        assert counts['11'] == 1000

    def test_crx_cry_crz(self):
        # `crx(pi)` with control=|1> flips the target (q1: |0> → |1>).
        src = HEADER_3 + "qubit[2] q;\nx q[0];\ncrx(pi) q[0], q[1];\n"
        assert _run(src)['11'] == 1000
        # `cry(pi)` with control=|1> flips the target as well.
        src2 = HEADER_3 + "qubit[2] q;\nx q[0];\ncry(pi) q[0], q[1];\n"
        assert _run(src2)['11'] == 1000
        # `crz(pi)` with control=|1> is phase-only on target.
        src3 = HEADER_3 + "qubit[2] q;\nx q[0];\ncrz(pi) q[0], q[1];\n"
        assert _run(src3)['10'] == 1000

    # --- Swaps / exchanges ------------------------------------------------

    def test_swap(self):
        src = HEADER_3 + "qubit[2] q;\nx q[0];\nswap q[0], q[1];\n"
        counts = _run(src)
        # q0 → q1; q0 now |0>, q1 now |1> → big-endian '01'.
        assert counts['01'] == 1000

    def test_iswap(self):
        # `iswap` on |01> → i|10>; measurement gives '10'.
        src = HEADER_3 + "qubit[2] q;\nx q[1];\niswap q[0], q[1];\n"
        counts = _run(src)
        assert counts['10'] == 1000

    def test_ecr(self):
        # ECR is a hardware-level 2-qubit gate; just confirm it's accepted.
        src = HEADER_3 + "qubit[2] q;\necr q[0], q[1];\n"
        counts = _run(src)
        assert len(counts) >= 1  # reaches the simulator, produces some bits

    # --- Two-qubit parametric (two-body Pauli rotations) -------------------

    def test_rxx_pi(self):
        # RXX(pi) |00> = -i|11>; measurement → '11'.
        src = HEADER_3 + "qubit[2] q;\nrxx(pi) q[0], q[1];\n"
        assert _run(src)['11'] == 1000

    def test_ryy_2pi_is_identity(self):
        # RYY(2π) = -I; measurement unchanged.
        src = HEADER_3 + "qubit[2] q;\nryy(2*pi) q[0], q[1];\n"
        assert _run(src)['00'] == 1000

    def test_rzz_on_zero(self):
        # RZZ(π/2) on |00> is a phase — counts still '00'.
        src = HEADER_3 + "qubit[2] q;\nrzz(pi/2) q[0], q[1];\n"
        assert _run(src)['00'] == 1000

    # --- 3-qubit & multi-qubit --------------------------------------------

    def test_ccx_toffoli(self):
        src = HEADER_3 + "qubit[3] q;\nx q[0];\nx q[1];\nccx q[0], q[1], q[2];\n"
        assert _run(src)['111'] == 1000

    def test_ccz_on_111(self):
        # CCZ is pure phase; measurement stays '111'.
        src = HEADER_3 + "qubit[3] q;\nx q[0];\nx q[1];\nx q[2];\nccz q[0], q[1], q[2];\n"
        assert _run(src)['111'] == 1000

    def test_cswap_fredkin(self):
        src = HEADER_3 + textwrap.dedent("""\
            qubit[3] q;
            x q[0];
            x q[1];
            cswap q[0], q[1], q[2];
        """)
        counts = _run(src)
        # control=|1>, swap(q1,q2): q1 |1> ↔ q2 |0> → q1=|0>, q2=|1> → '101'.
        assert counts['101'] == 1000


class TestQASM3CustomGates:

    def test_custom_gate_definition(self):
        src = textwrap.dedent("""\
            OPENQASM 3.0;
            include "stdgates.inc";
            gate mybell a, b {
                h a;
                cx a, b;
            }
            qubit[2] q;
            mybell q[0], q[1];
        """)
        counts = _run(src)
        assert '00' in counts and '11' in counts
        assert '01' not in counts and '10' not in counts


class TestQASM3Errors:

    def test_ctrl_modifier_rejected(self):
        src = HEADER_3 + "qubit[2] q;\nctrl @ x q[0], q[1];\n"
        with pytest.raises(NotImplementedError, match="modifier"):
            from_qasm_str(src)

    def test_inv_modifier_rejected(self):
        src = HEADER_3 + "qubit[1] q;\ninv @ s q[0];\n"
        with pytest.raises(NotImplementedError, match="modifier"):
            from_qasm_str(src)

    def test_pow_modifier_rejected(self):
        src = HEADER_3 + "qubit[1] q;\npow(2) @ x q[0];\n"
        with pytest.raises(NotImplementedError, match="modifier"):
            from_qasm_str(src)

    def test_def_subroutine_rejected(self):
        src = textwrap.dedent("""\
            OPENQASM 3.0;
            include "stdgates.inc";
            def flip(qubit q) { x q; }
            qubit[1] q;
            flip(q[0]);
        """)
        with pytest.raises(NotImplementedError, match="classical feature"):
            from_qasm_str(src)

    def test_for_loop_rejected(self):
        src = HEADER_3 + "qubit[2] q;\nfor uint i in [0:1] { h q[i]; }\n"
        with pytest.raises(NotImplementedError, match="classical feature"):
            from_qasm_str(src)

    def test_input_declaration_rejected(self):
        src = HEADER_3 + "input float theta;\nqubit[1] q;\nrx(theta) q[0];\n"
        with pytest.raises(NotImplementedError, match="classical feature"):
            from_qasm_str(src)

    def test_missing_header_raises(self):
        src = "qubit[1] q;\nx q[0];\n"
        with pytest.raises(ValueError, match="OPENQASM"):
            from_qasm_str(src)
