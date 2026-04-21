# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Round-trip tests for the QASM / Qiskit / CUDA-Q conversion paths.

Two scenarios:

1. ``TestQasmToQiskitToCudaq`` — an OpenQASM 2.0 source is loaded with
   the native parser provided by Qiskit, and the resulting
   ``QuantumCircuit`` is converted to a CUDA-Q kernel with
   ``cudaq.contrib.from_qiskit``.

2. ``TestQiskitToQasmToCudaq`` — a ``QuantumCircuit`` is built with the
   Qiskit builder, written to OpenQASM 2.0 with the native writer
   provided by Qiskit, and parsed into a CUDA-Q kernel with
   ``cudaq.contrib.from_qasm_str``.

These tests exercise the interoperability story end-to-end. All bitstring
expectations follow the big-endian convention used by CUDA-Q (q[0] is the
leftmost character of the output bitstring).
"""

import math
import textwrap

import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

import cudaq
from cudaq.contrib import from_qasm_str, from_qiskit

# --------------------------------------------------------------------------- #
# Qiskit API compatibility shims (`qasm2.loads` / `qasm2.dumps` preferred;
# older Qiskit versions fall back to the legacy path).
# --------------------------------------------------------------------------- #


def _qiskit_load_qasm(src):
    """Parse an OpenQASM 2.0 string into a Qiskit ``QuantumCircuit``."""
    try:
        from qiskit import qasm2
        return qasm2.loads(src)
    except (ImportError, AttributeError):
        return QuantumCircuit.from_qasm_str(src)


def _qiskit_dump_qasm(qc):
    """Write a Qiskit ``QuantumCircuit`` out as an OpenQASM 2.0 string."""
    try:
        from qiskit import qasm2
        return qasm2.dumps(qc)
    except (ImportError, AttributeError):
        return qc.qasm()


# --------------------------------------------------------------------------- #
# Scenario A: OpenQASM source → Qiskit → CUDA-Q (`from_qiskit`)
# --------------------------------------------------------------------------- #


class TestQasmToQiskitToCudaq:
    """QASM 2.0 text → ``qiskit.qasm2.loads`` → ``from_qiskit`` → CUDA-Q."""

    def _trip(self, qasm_src):
        qc = _qiskit_load_qasm(qasm_src)
        return from_qiskit(qc)

    def test_bell_pair(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q -> c;
        """)
        kernel = self._trip(src)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert set(counts) <= {'00', '11'}
        assert '00' in counts and '11' in counts

    def test_ghz_three_qubit(self):
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cx q[0], q[1];
            cx q[1], q[2];
            measure q -> c;
        """)
        kernel = self._trip(src)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert set(counts) <= {'000', '111'}
        assert '000' in counts and '111' in counts

    def test_deterministic_single_qubit(self):
        # u3(pi, 0, pi) ≡ X up to global phase → measurement = '1'.
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            u3(pi, 0, pi) q[0];
            measure q -> c;
        """)
        kernel = self._trip(src)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert counts['1'] == 1000

    def test_rotations_and_controls(self):
        # `rz` on |0> is a phase → state is |00>; `ccx` with both controls |0>
        # is a no-op; expect exactly '000'.
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            rz(pi/4) q[0];
            cu1(pi/8) q[0], q[1];
            ccx q[0], q[1], q[2];
            measure q -> c;
        """)
        kernel = self._trip(src)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert counts['000'] == 1000

    def test_user_defined_gate_in_qasm(self):
        # Custom ``bell`` gate defined in the QASM source is expanded by
        # the parser provided by Qiskit into its body before `from_qiskit`
        # sees it.
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            gate bell a, b { h a; cx a, b; }
            qreg q[2];
            creg c[2];
            bell q[0], q[1];
            measure q -> c;
        """)
        kernel = self._trip(src)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert set(counts) <= {'00', '11'}
        assert '00' in counts and '11' in counts

    def test_parametric_two_qubit(self):
        # ``rxx`` isn't in ``qelib1.inc``, so define it inline via the
        # standard decomposition. ``rxx(pi)`` on |00> = -i|11> (up to global
        # phase).
        src = textwrap.dedent("""\
            OPENQASM 2.0;
            include "qelib1.inc";
            gate rxx(theta) a, b {
                h a; h b;
                cx a, b;
                rz(theta) b;
                cx a, b;
                h a; h b;
            }
            qreg q[2];
            creg c[2];
            rxx(pi) q[0], q[1];
            measure q -> c;
        """)
        kernel = self._trip(src)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert counts['11'] == 1000


# --------------------------------------------------------------------------- #
# Scenario B: Qiskit → QASM (``qasm2.dumps``) → CUDA-Q (``from_qasm_str``)
# --------------------------------------------------------------------------- #


class TestQiskitToQasmToCudaq:
    """Build with Qiskit → ``qiskit.qasm2.dumps`` → ``from_qasm_str``."""

    def _trip(self, qc):
        return from_qasm_str(_qiskit_dump_qasm(qc))

    def test_bell_pair(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        kernel = self._trip(qc)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert set(counts) <= {'00', '11'}
        assert '00' in counts and '11' in counts

    def test_ghz_four_qubit(self):
        n = 4
        qc = QuantumCircuit(n, n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n), range(n))
        kernel = self._trip(qc)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert set(counts) <= {'0' * n, '1' * n}
        assert '0' * n in counts and '1' * n in counts

    def test_deterministic_flip_big_endian(self):
        # X on q[0] only: q0 = |1>, q1 = |0>. CUDA-Q bitstring is big-endian
        # (q0 leftmost) → '10'.
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure([0, 1], [0, 1])
        kernel = self._trip(qc)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert counts['10'] == 1000

    def test_controlled_rotation(self):
        # ``crx(pi)`` with control=|1> flips target: |10> → |11>.
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.crx(math.pi, 0, 1)
        qc.measure([0, 1], [0, 1])
        kernel = self._trip(qc)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert counts['11'] == 1000

    def test_rotations_phase_only_stays_on_zero(self):
        qc = QuantumCircuit(2, 2)
        qc.rz(math.pi / 4, 0)
        qc.cp(math.pi / 3, 0, 1)
        qc.measure([0, 1], [0, 1])
        kernel = self._trip(qc)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert counts['00'] == 1000

    def test_composite_via_compose(self):
        # Build a Bell-pair sub-circuit, compose into the main, round-trip.
        # The OpenQASM emitter in Qiskit inline-expands ``compose``d sub-circuits
        # into the top-level, so the resulting OpenQASM uses flat ``h`` + ``cx``.
        sub = QuantumCircuit(2, name="mybell")
        sub.h(0)
        sub.cx(0, 1)

        qc = QuantumCircuit(3, 3)
        qc.compose(sub, qubits=[0, 1], inplace=True)
        qc.x(2)
        qc.measure([0, 1, 2], [0, 1, 2])

        kernel = self._trip(qc)
        counts = cudaq.sample(kernel, shots_count=1000)
        # q0,q1 in Bell; q2 = |1>. Big-endian → '001' or '111'.
        assert set(counts) <= {'001', '111'}
        assert '001' in counts and '111' in counts

    def test_ccx_and_swap(self):
        qc = QuantumCircuit(3, 3)
        qc.x(0)
        qc.x(1)
        qc.ccx(0, 1, 2)  # |11> controls → target flips → '111'
        qc.swap(0, 2)  # swap q0 ↔ q2 → state unchanged ('111')
        qc.measure([0, 1, 2], [0, 1, 2])
        kernel = self._trip(qc)
        counts = cudaq.sample(kernel, shots_count=1000)
        assert counts['111'] == 1000


# --------------------------------------------------------------------------- #
# Cross-check: ``QASM → Qiskit → CUDA-Q`` and ``QASM → CUDA-Q`` (direct)
# should match.
# --------------------------------------------------------------------------- #


class TestRoundTripConsistency:
    """The CUDA-Q-bound outcome should be identical whether we go through
    Qiskit or parse the QASM directly with the native translator."""

    @pytest.mark.parametrize("qasm_body,expected", [
        ("qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\n"
         "measure q -> c;\n", {'00', '11'}),
        ("qreg q[2];\ncreg c[2];\nx q[0];\n"
         "measure q -> c;\n", {'10'}),
        ("qreg q[3];\ncreg c[3];\nx q[0];\nx q[1];\n"
         "ccx q[0], q[1], q[2];\nmeasure q -> c;\n", {'111'}),
    ])
    def test_paths_agree(self, qasm_body, expected):
        src = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + qasm_body

        # Direct: ``QASM → CUDA-Q``
        direct = cudaq.sample(from_qasm_str(src), shots_count=1000)

        # Indirect: ``QASM → Qiskit → CUDA-Q``
        qc = _qiskit_load_qasm(src)
        indirect = cudaq.sample(from_qiskit(qc), shots_count=1000)

        assert set(direct) <= expected
        assert set(indirect) <= expected
        # If expected has a single element, both must have produced only it.
        if len(expected) == 1:
            only = next(iter(expected))
            assert direct[only] == 1000
            assert indirect[only] == 1000
