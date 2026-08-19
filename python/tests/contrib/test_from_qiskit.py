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

QuantumCircuit = qiskit.QuantumCircuit
Gate = qiskit.circuit.Gate
Parameter = qiskit.circuit.Parameter
CUGate = qiskit.circuit.library.CUGate


def _resource_counts(kernel):
    return cudaq.estimate_resources(kernel).to_dict()


def _from_qasm_str(qasm):
    return cudaq.contrib.from_qiskit(QuantumCircuit.from_qasm_str(qasm))


def _from_qasm_file(qasm):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm",
                                     delete=False) as f:
        f.write(qasm)
        temp_path = f.name

    try:
        return cudaq.contrib.from_qasm(temp_path)
    finally:
        os.unlink(temp_path)


class TestFromQiskit:
    """Tests for cudaq.contrib.from_qiskit()."""

    def test_single_qubit_h_gate(self):
        """Test conversion of a single H gate."""
        qc = QuantumCircuit(1)
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H gate creates superposition, expect roughly 50/50 distribution
        assert "0" in counts or "1" in counts

    def test_single_qubit_x_gate(self):
        """Test conversion of X gate."""
        qc = QuantumCircuit(1)
        qc.x(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["1"] == 1000

    def test_single_qubit_y_gate(self):
        """Test conversion of Y gate."""
        qc = QuantumCircuit(1)
        qc.y(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["1"] == 1000

    def test_single_qubit_z_gate(self):
        """Test conversion of Z gate on |+> state."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.z(0)
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-Z-H = X, so result should be |1>
        assert counts["1"] == 1000

    def test_single_qubit_s_gate(self):
        """Test conversion of S gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.s(0)
        qc.s(0)
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-S-S-H = H-Z-H = X, result should be |1>
        assert counts["1"] == 1000

    def test_single_qubit_t_gate(self):
        """Test conversion of T gate."""
        qc = QuantumCircuit(1)
        qc.t(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # T on |0> should give |0>
        assert counts["0"] == 1000

    def test_single_qubit_sdg_gate(self):
        """Test conversion of Sdg (S-dagger) gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.sdg(0)
        qc.sdg(0)
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-Sdg-Sdg-H = H-Z-H = X, result should be |1>
        assert counts["1"] == 1000

    def test_single_qubit_tdg_gate(self):
        """Test conversion of Tdg (T-dagger) gate."""
        qc = QuantumCircuit(1)
        qc.tdg(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Tdg on |0> should give |0>
        assert counts["0"] == 1000

    def test_cx_gate(self):
        """Test conversion of CX (CNOT) gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cx(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # |10> -> |11> after CNOT
        assert counts["11"] == 1000

    def test_cy_gate(self):
        """Test conversion of CY gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cy(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control qubit is 1, so Y is applied to target
        assert counts["11"] == 1000

    def test_cz_gate(self):
        """Test conversion of CZ gate."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.cz(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # CZ on |++> creates entanglement
        assert len(counts) > 0

    def test_ch_gate(self):
        """Test conversion of CH gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.ch(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control is 1, so H is applied to target creating superposition
        assert "10" in counts or "11" in counts

    def test_swap_gate(self):
        """Test conversion of SWAP gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.swap(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # |10> swapped to |01>
        assert counts["01"] == 1000

    def test_rx_gate(self):
        """Test conversion of RX gate."""
        qc = QuantumCircuit(1)
        qc.rx(np.pi, 0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RX(pi) on |0> gives |1>
        assert counts["1"] == 1000

    def test_ry_gate(self):
        """Test conversion of RY gate."""
        qc = QuantumCircuit(1)
        qc.ry(np.pi, 0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RY(pi) on |0> gives |1>
        assert counts["1"] == 1000

    def test_rz_gate(self):
        """Test conversion of RZ gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(np.pi, 0)
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-RZ(pi)-H = X, result should be |1>
        assert counts["1"] == 1000

    def test_crx_gate(self):
        """Test conversion of CRX gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.crx(np.pi, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control is 1, RX(pi) applied to target
        assert counts["11"] == 1000

    def test_cry_gate(self):
        """Test conversion of CRY gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cry(np.pi, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control is 1, RY(pi) applied to target
        assert counts["11"] == 1000

    def test_crz_gate(self):
        """Test conversion of CRZ gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        qc.crz(np.pi, 0, 1)
        qc.h(1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control is 1, H-RZ(pi)-H = X applied to target
        assert counts["11"] == 1000

    def test_cp_gate_resource_count(self):
        """Test conversion of CP to CUDA-Q CR1."""
        qc = QuantumCircuit(2)
        qc.cp(np.pi / 4, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        resources = cudaq.estimate_resources(kernel)

        assert resources.to_dict().get("cr1", 0) == 1

    def test_cu1_gate_resource_count(self):
        """Test conversion of CU1 to CUDA-Q CR1."""
        if not hasattr(QuantumCircuit, "cu1"):
            pytest.skip("Qiskit QuantumCircuit.cu1 is unavailable.")
        qc = QuantumCircuit(2)
        qc.cu1(np.pi / 4, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        resources = cudaq.estimate_resources(kernel)

        assert resources.to_dict().get("cr1", 0) == 1

    def test_direct_gate_resource_counts(self):
        """Test direct Qiskit gate mappings to native CUDA-Q builder ops."""
        qc = QuantumCircuit(6)
        qc.append(Gate("u1", 1, [0.125]), [0])
        qc.u(np.pi / 2, 0.25, 0.5, 1)
        qc.append(Gate("cu3", 2, [0.125, 0.25, 0.5]), [0, 2])
        qc.cswap(0, 3, 4)
        qc.mcx([0, 1, 2], 5)
        qc.reset(4)
        qc.cs(0, 1)
        qc.append(Gate("ct", 2, []), [1, 2])
        qc.csdg(0, 1)
        qc.append(Gate("ctdg", 2, []), [1, 2])

        kernel = cudaq.contrib.from_qiskit(qc)
        resources = _resource_counts(kernel)

        assert resources.get("r1", 0) == 1
        assert resources.get("u3", 0) == 1
        assert resources.get("cu3", 0) == 1
        assert resources.get("cswap", 0) == 1
        assert resources.get("cccx", 0) == 1
        assert resources.get("reset", 0) == 1
        assert resources.get("cs", 0) == 2
        assert resources.get("ct", 0) == 2

    def test_synthetic_gate_alias_resource_counts(self):
        """Test aliases that Qiskit public APIs may normalize away."""
        qc = QuantumCircuit(3)
        qc.append(Gate("phase", 1, [0.125]), [0])
        qc.append(Gate("cphase", 2, [0.25]), [0, 1])
        qc.append(Gate("CX", 2, []), [1, 2])

        kernel = cudaq.contrib.from_qiskit(qc)
        resources = _resource_counts(kernel)

        assert resources.get("r1", 0) == 1
        assert resources.get("cr1", 0) == 1
        assert resources.get("cx", 0) == 1

    def test_open_control_cx_expands_definition(self):
        """Test Qiskit open-control CX expands through its definition."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1, ctrl_state=0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["01"] == 1000

    def test_open_control_ccx_expands_definition(self):
        """Test Qiskit open-control CCX expands through its definition."""
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.ccx(0, 1, 2, ctrl_state=1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["101"] == 1000

    def test_open_control_mcx_expands_definition(self):
        """Test Qiskit open-control MCX expands through its definition."""
        qc = QuantumCircuit(5)
        qc.x(1)
        qc.x(3)
        qc.mcx([0, 1, 2, 3], 4, ctrl_state=0b1010)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["01011"] == 1000

    def test_csdg_cancels_cs(self):
        """Test CSDG emits the adjoint of CS."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        qc.cs(0, 1)
        qc.csdg(0, 1)
        qc.h(1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["10"] == 1000

    def test_ctdg_cancels_ct(self):
        """Test CTDG emits the adjoint of CT."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        qc.append(Gate("ct", 2, []), [0, 1])
        qc.append(Gate("ctdg", 2, []), [0, 1])
        qc.h(1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["10"] == 1000

    def test_u3_gate(self):
        """Test conversion of U3 gate."""
        qc = QuantumCircuit(1)
        qc.u(np.pi, 0, np.pi, 0)  # U3(pi, 0, pi) = X

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["1"] == 1000

    def test_phase_gate(self):
        """Test conversion of phase (p) gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(np.pi, 0)
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-P(pi)-H should flip the qubit
        assert counts["1"] == 1000

    def test_sx_gate(self):
        """Test conversion of SX (sqrt-X) gate."""
        qc = QuantumCircuit(1)
        qc.sx(0)
        qc.sx(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # SX^2 = X, result should be |1>
        assert counts["1"] == 1000

    def test_identity_gate(self):
        """Test conversion of identity gate."""
        qc = QuantumCircuit(1)
        qc.id(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["0"] == 1000

    def test_barrier_ignored(self):
        """Test that barrier is properly ignored."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Bell state
        assert "00" in counts and "11" in counts

    def test_measurement(self):
        """Test that measurements are converted."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["1"] == 1000

    def test_bell_state(self):
        """Test conversion of Bell state circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Bell state should have only |00> and |11>
        assert "00" in counts
        assert "11" in counts
        assert "01" not in counts
        assert "10" not in counts

    def test_ghz_state(self):
        """Test conversion of 3-qubit GHZ state circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # GHZ state should have only |000> and |111>
        assert "000" in counts
        assert "111" in counts

    def test_ccx_toffoli_gate(self):
        """Test conversion of CCX (Toffoli) gate."""
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(1)
        qc.ccx(0, 1, 2)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Both controls are 1, so target flips: |110> -> |111>
        assert counts["111"] == 1000

    def test_rxx_gate(self):
        """Test conversion of RXX gate."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.rxx(np.pi, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RXX creates entanglement
        assert len(counts) > 0

    def test_rzz_gate(self):
        """Test conversion of RZZ gate."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.rzz(np.pi, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RZZ creates entanglement
        assert len(counts) > 0

    def test_rzz_pi_behavior(self):
        """Test RZZ(pi) lowers with the rotation on the target qubit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.rzz(np.pi, 0, 1)
        qc.h(0)
        qc.h(1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["11"] == 1000

    def test_sxdg_gate(self):
        """Test conversion of SXdg (sqrt-X dagger) gate."""
        qc = QuantumCircuit(1)
        qc.sx(0)
        qc.sxdg(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # SX followed by SXdg should return to |0>
        assert counts["0"] == 1000

    def test_unsupported_gate_raises_error(self):
        """Test that unsupported gates raise ValueError."""
        qc = QuantumCircuit(2)
        qc.append(CUGate(0.1, 0.2, 0.3, 0.4), [0, 1])

        with pytest.raises(ValueError, match="Gate 'cu' is not supported"):
            cudaq.contrib.from_qiskit(qc)

    def test_custom_gate_definition_expands_inline(self):
        """Test custom Qiskit gate definitions expand inline."""
        custom = QuantumCircuit(1, name="mygate")
        custom.h(0)

        qc = QuantumCircuit(1)
        qc.append(custom.to_gate(), [0])

        kernel = cudaq.contrib.from_qiskit(qc)
        resources = _resource_counts(kernel)

        assert resources.get("h", 0) == 1
        assert "mygate" not in resources

    def test_custom_gate_name_collision_expands_inline(self):
        """Test custom gates expand before direct name matching."""
        custom = QuantumCircuit(1, name="x")
        custom.h(0)

        qc = QuantumCircuit(1)
        qc.append(custom.to_gate(), [0])

        kernel = cudaq.contrib.from_qiskit(qc)
        resources = _resource_counts(kernel)

        assert resources.get("h", 0) == 1
        assert "x" not in resources

    @pytest.mark.parametrize(
        "gate,qargs",
        [
            (Gate("x", 2, []), [0, 1]),
            (Gate("cx", 1, []), [0]),
            (Gate("x", 1, [0.1]), [0]),
            (Gate("rx", 1, []), [0]),
        ],
    )
    def test_malformed_direct_gate_raises_value_error(self, gate, qargs):
        """Test malformed direct gates raise ValueError."""
        qc = QuantumCircuit(2)
        qc.append(gate, qargs)

        with pytest.raises(ValueError, match=f"Gate '{gate.name.lower()}'"):
            cudaq.contrib.from_qiskit(qc)

    def test_symbolic_parameter_raises_value_error(self):
        """Test symbolic parameters are rejected with a clear error."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        with pytest.raises(ValueError, match="Gate 'rx'"):
            cudaq.contrib.from_qiskit(qc)

    def test_control_flow_raises_unsupported_input_error(self):
        """Test Qiskit control flow fails before parameter conversion."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        if(c==1) x q[0];
        """
        qc = QuantumCircuit.from_qasm_str(qasm)

        with pytest.raises(ValueError, match="classical control flow"):
            cudaq.contrib.from_qiskit(qc)

    def test_empty_circuit(self):
        """Test conversion of empty circuit."""
        qc = QuantumCircuit(2)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts["00"] == 1000


class TestFromQasm:
    """Tests for cudaq.contrib.from_qasm()."""

    def test_simple_qasm_file(self):
        """Test loading a simple QASM file."""
        qasm_content = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm",
                                         delete=False) as f:
            f.write(qasm_content)
            temp_path = f.name

        try:
            kernel = cudaq.contrib.from_qasm(temp_path)
            counts = cudaq.sample(kernel)

            # Bell state
            assert "00" in counts
            assert "11" in counts
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm",
                                         delete=False) as f:
            f.write(qasm_content)
            temp_path = f.name

        try:
            kernel = cudaq.contrib.from_qasm(temp_path)
            counts = cudaq.sample(kernel)

            # RX(pi) should flip the qubit
            assert counts["1"] == 1000
        finally:
            os.unlink(temp_path)

    def test_qasm_direct_gate_aliases(self):
        """Test OpenQASM aliases imported through Qiskit map directly."""
        kernel = _from_qasm_file("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[6];
        u1(0.125) q[0];
        u2(0.25,0.5) q[1];
        cu3(0.125,0.25,0.5) q[0],q[2];
        cswap q[0],q[3],q[4];
        c3x q[0],q[1],q[2],q[5];
        """)

        resources = _resource_counts(kernel)

        assert resources.get("r1", 0) == 1
        assert resources.get("u3", 0) == 1
        assert resources.get("cu3", 0) == 1
        assert resources.get("cswap", 0) == 1
        assert resources.get("cccx", 0) == 1

    def test_qasm_c4x_imports_as_mcx(self):
        """Test c4x is normalized by Qiskit and emitted as controlled X."""
        kernel = _from_qasm_str("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[5];
        c4x q[0],q[1],q[2],q[3],q[4];
        """)

        resources = _resource_counts(kernel)

        assert resources.get("ccccx", 0) == 1

    def test_custom_qasm_gate_definition_expands_inline(self):
        """Test custom OpenQASM gate definitions expand inline."""
        kernel = _from_qasm_str("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        gate mygate a { h a; }
        mygate q[0];
        """)

        resources = _resource_counts(kernel)

        assert resources.get("h", 0) == 1
        assert "mygate" not in resources

    def test_custom_qasm_macro_expands_inline(self):
        """Test a multi-instruction OpenQASM macro expands inline."""
        kernel = _from_qasm_str("""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        gate majority a,b,c {
          cx c,b;
          cx c,a;
          ccx a,b,c;
        }
        majority q[0],q[1],q[2];
        """)

        resources = _resource_counts(kernel)

        assert resources.get("cx", 0) == 2
        assert resources.get("ccx", 0) == 1
        assert "majority" not in resources

    def test_qasm_control_flow_raises_value_error(self):
        """Test OpenQASM if statements report unsupported control flow."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        if(c==1) x q[0];
        """

        with pytest.raises(ValueError, match="classical control flow"):
            _from_qasm_str(qasm)

    def test_qasm_u0_global_phase_is_unsupported(self):
        """Test u0 remains unsupported because it is a global phase."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u0(5) q[0];
        """

        with pytest.raises(ValueError, match="Gate 'u0' is not supported"):
            _from_qasm_str(qasm)

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            cudaq.contrib.from_qasm("/nonexistent/path/to/file.qasm")

    def test_invalid_qasm_raises_error(self):
        """Test that invalid QASM content raises RuntimeError."""
        qasm_content = "this is not valid qasm"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm",
                                         delete=False) as f:
            f.write(qasm_content)
            temp_path = f.name

        try:
            with pytest.raises(RuntimeError, match="Could not parse"):
                cudaq.contrib.from_qasm(temp_path)
        finally:
            os.unlink(temp_path)
