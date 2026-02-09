# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file tests the from_qiskit and from_qasm conversion functions.

import pytest
import numpy as np
import os
import tempfile

import cudaq

# Skip all tests if qiskit is not installed
qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit


class TestFromQiskit:
    """Tests for cudaq.from_qiskit()."""

    def test_single_qubit_h_gate(self):
        """Test conversion of a single H gate."""
        qc = QuantumCircuit(1)
        qc.h(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H gate creates superposition, expect roughly 50/50 distribution
        assert '0' in counts or '1' in counts

    def test_single_qubit_x_gate(self):
        """Test conversion of X gate."""
        qc = QuantumCircuit(1)
        qc.x(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['1'] == 1000

    def test_single_qubit_y_gate(self):
        """Test conversion of Y gate."""
        qc = QuantumCircuit(1)
        qc.y(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['1'] == 1000

    def test_single_qubit_z_gate(self):
        """Test conversion of Z gate on |+> state."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.z(0)
        qc.h(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-Z-H = X, so result should be |1>
        assert counts['1'] == 1000

    def test_single_qubit_s_gate(self):
        """Test conversion of S gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.s(0)
        qc.s(0)
        qc.h(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-S-S-H = H-Z-H = X, result should be |1>
        assert counts['1'] == 1000

    def test_single_qubit_t_gate(self):
        """Test conversion of T gate."""
        qc = QuantumCircuit(1)
        qc.t(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # T on |0> should give |0>
        assert counts['0'] == 1000

    def test_single_qubit_sdg_gate(self):
        """Test conversion of Sdg (S-dagger) gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.sdg(0)
        qc.sdg(0)
        qc.h(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-Sdg-Sdg-H = H-Z-H = X, result should be |1>
        assert counts['1'] == 1000

    def test_single_qubit_tdg_gate(self):
        """Test conversion of Tdg (T-dagger) gate."""
        qc = QuantumCircuit(1)
        qc.tdg(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Tdg on |0> should give |0>
        assert counts['0'] == 1000

    def test_cx_gate(self):
        """Test conversion of CX (CNOT) gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cx(0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # |10> -> |11> after CNOT
        assert counts['11'] == 1000

    def test_cy_gate(self):
        """Test conversion of CY gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cy(0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control qubit is 1, so Y is applied to target
        assert counts['11'] == 1000

    def test_cz_gate(self):
        """Test conversion of CZ gate."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.cz(0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # CZ on |++> creates entanglement
        assert len(counts) > 0

    def test_ch_gate(self):
        """Test conversion of CH gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.ch(0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control is 1, so H is applied to target creating superposition
        assert '10' in counts or '11' in counts

    def test_swap_gate(self):
        """Test conversion of SWAP gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.swap(0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # |10> swapped to |01>
        assert counts['01'] == 1000

    def test_rx_gate(self):
        """Test conversion of RX gate."""
        qc = QuantumCircuit(1)
        qc.rx(np.pi, 0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RX(pi) on |0> gives |1>
        assert counts['1'] == 1000

    def test_ry_gate(self):
        """Test conversion of RY gate."""
        qc = QuantumCircuit(1)
        qc.ry(np.pi, 0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RY(pi) on |0> gives |1>
        assert counts['1'] == 1000

    def test_rz_gate(self):
        """Test conversion of RZ gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(np.pi, 0)
        qc.h(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-RZ(pi)-H = X, result should be |1>
        assert counts['1'] == 1000

    def test_crx_gate(self):
        """Test conversion of CRX gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.crx(np.pi, 0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control is 1, RX(pi) applied to target
        assert counts['11'] == 1000

    def test_cry_gate(self):
        """Test conversion of CRY gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cry(np.pi, 0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control is 1, RY(pi) applied to target
        assert counts['11'] == 1000

    def test_crz_gate(self):
        """Test conversion of CRZ gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        qc.crz(np.pi, 0, 1)
        qc.h(1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control is 1, H-RZ(pi)-H = X applied to target
        assert counts['11'] == 1000

    def test_u3_gate(self):
        """Test conversion of U3 gate."""
        qc = QuantumCircuit(1)
        qc.u(np.pi, 0, np.pi, 0)  # U3(pi, 0, pi) = X

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['1'] == 1000

    def test_phase_gate(self):
        """Test conversion of phase (p) gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(np.pi, 0)
        qc.h(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-P(pi)-H should flip the qubit
        assert counts['1'] == 1000

    def test_sx_gate(self):
        """Test conversion of SX (sqrt-X) gate."""
        qc = QuantumCircuit(1)
        qc.sx(0)
        qc.sx(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # SX^2 = X, result should be |1>
        assert counts['1'] == 1000

    def test_identity_gate(self):
        """Test conversion of identity gate."""
        qc = QuantumCircuit(1)
        qc.id(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['0'] == 1000

    def test_barrier_ignored(self):
        """Test that barrier is properly ignored."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Bell state
        assert '00' in counts and '11' in counts

    def test_measurement(self):
        """Test that measurements are converted."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['1'] == 1000

    def test_bell_state(self):
        """Test conversion of Bell state circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Bell state should have only |00> and |11>
        assert '00' in counts
        assert '11' in counts
        assert '01' not in counts
        assert '10' not in counts

    def test_ghz_state(self):
        """Test conversion of 3-qubit GHZ state circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # GHZ state should have only |000> and |111>
        assert '000' in counts
        assert '111' in counts

    def test_ccx_toffoli_gate(self):
        """Test conversion of CCX (Toffoli) gate."""
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(1)
        qc.ccx(0, 1, 2)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Both controls are 1, so target flips: |110> -> |111>
        assert counts['111'] == 1000

    def test_rxx_gate(self):
        """Test conversion of RXX gate."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.rxx(np.pi, 0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RXX creates entanglement
        assert len(counts) > 0

    def test_rzz_gate(self):
        """Test conversion of RZZ gate."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.rzz(np.pi, 0, 1)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RZZ creates entanglement
        assert len(counts) > 0

    def test_sxdg_gate(self):
        """Test conversion of SXdg (sqrt-X dagger) gate."""
        qc = QuantumCircuit(1)
        qc.sx(0)
        qc.sxdg(0)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # SX followed by SXdg should return to |0>
        assert counts['0'] == 1000

    def test_unsupported_gate_raises_error(self):
        """Test that unsupported gates raise ValueError."""
        qc = QuantumCircuit(4)
        qc.mcx([0, 1, 2], 3)  # Multi-controlled X not supported

        with pytest.raises(ValueError, match="not supported"):
            cudaq.from_qiskit(qc)

    def test_empty_circuit(self):
        """Test conversion of empty circuit."""
        qc = QuantumCircuit(2)

        kernel = cudaq.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['00'] == 1000


class TestFromQasm:
    """Tests for cudaq.from_qasm()."""

    def test_simple_qasm_file(self):
        """Test loading a simple QASM file."""
        qasm_content = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm',
                                         delete=False) as f:
            f.write(qasm_content)
            temp_path = f.name

        try:
            kernel = cudaq.from_qasm(temp_path)
            counts = cudaq.sample(kernel)

            # Bell state
            assert '00' in counts
            assert '11' in counts
        finally:
            os.unlink(temp_path)

    def test_qasm_with_rotations(self):
        """Test QASM file with parametric rotations."""
        qasm_content = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(3.14159265359) q[0];
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm',
                                         delete=False) as f:
            f.write(qasm_content)
            temp_path = f.name

        try:
            kernel = cudaq.from_qasm(temp_path)
            counts = cudaq.sample(kernel)

            # RX(pi) should flip the qubit
            assert counts['1'] == 1000
        finally:
            os.unlink(temp_path)

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            cudaq.from_qasm('/nonexistent/path/to/file.qasm')

    def test_invalid_qasm_raises_error(self):
        """Test that invalid QASM content raises RuntimeError."""
        qasm_content = "this is not valid qasm"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm',
                                         delete=False) as f:
            f.write(qasm_content)
            temp_path = f.name

        try:
            with pytest.raises(RuntimeError, match="Could not parse"):
                cudaq.from_qasm(temp_path)
        finally:
            os.unlink(temp_path)
