# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file tests the `from_qiskit` conversion function.
# Tests for `from_qasm` live in `test_from_qasm.py`.

import pytest
import numpy as np

import cudaq
from cudaq import contrib

# Skip all tests if `qiskit` is not installed
qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    U1Gate,
    U2Gate,
    CU1Gate,
    CU3Gate,
    CUGate,
    XXPlusYYGate,
    XXMinusYYGate,
    C3XGate,
    C4XGate,
    GlobalPhaseGate,
)


class TestFromQiskit:
    """Tests for the `cudaq.contrib.from_qiskit` helper."""

    def test_single_qubit_h_gate(self):
        """Test conversion of a single H gate."""
        qc = QuantumCircuit(1)
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H gate creates superposition, expect roughly 50/50 distribution
        assert '0' in counts or '1' in counts

    def test_single_qubit_x_gate(self):
        """Test conversion of X gate."""
        qc = QuantumCircuit(1)
        qc.x(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['1'] == 1000

    def test_single_qubit_y_gate(self):
        """Test conversion of Y gate."""
        qc = QuantumCircuit(1)
        qc.y(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['1'] == 1000

    def test_single_qubit_z_gate(self):
        """Test conversion of Z gate on |+> state."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.z(0)
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
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

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-S-S-H = H-Z-H = X, result should be |1>
        assert counts['1'] == 1000

    def test_single_qubit_t_gate(self):
        """Test conversion of T gate."""
        qc = QuantumCircuit(1)
        qc.t(0)

        kernel = cudaq.contrib.from_qiskit(qc)
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

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-`Sdg`-`Sdg`-H = H-Z-H = X, result should be |1>
        assert counts['1'] == 1000

    def test_single_qubit_tdg_gate(self):
        """Test conversion of Tdg (T-dagger) gate."""
        qc = QuantumCircuit(1)
        qc.tdg(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Tdg on |0> should give |0>
        assert counts['0'] == 1000

    def test_cx_gate(self):
        """Test conversion of CX (CNOT) gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cx(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # |10> -> |11> after CNOT
        assert counts['11'] == 1000

    def test_cy_gate(self):
        """Test conversion of CY gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cy(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control qubit is 1, so Y is applied to target
        assert counts['11'] == 1000

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
        assert '10' in counts or '11' in counts

    def test_swap_gate(self):
        """Test conversion of SWAP gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.swap(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # |10> swapped to |01>
        assert counts['01'] == 1000

    def test_rx_gate(self):
        """Test conversion of RX gate."""
        qc = QuantumCircuit(1)
        qc.rx(np.pi, 0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RX(pi) on |0> gives |1>
        assert counts['1'] == 1000

    def test_ry_gate(self):
        """Test conversion of RY gate."""
        qc = QuantumCircuit(1)
        qc.ry(np.pi, 0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RY(pi) on |0> gives |1>
        assert counts['1'] == 1000

    def test_rz_gate(self):
        """Test conversion of RZ gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(np.pi, 0)
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-`RZ(pi)`-H = X, result should be |1>
        assert counts['1'] == 1000

    def test_crx_gate(self):
        """Test conversion of CRX gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.crx(np.pi, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control is 1, RX(pi) applied to target
        assert counts['11'] == 1000

    def test_cry_gate(self):
        """Test conversion of CRY gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cry(np.pi, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
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

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Control is 1, H-`RZ(pi)`-H = X applied to target
        assert counts['11'] == 1000

    def test_u3_gate(self):
        """Test conversion of U3 gate."""
        qc = QuantumCircuit(1)
        qc.u(np.pi, 0, np.pi, 0)  # U3(pi, 0, pi) = X

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['1'] == 1000

    def test_phase_gate(self):
        """Test conversion of phase (p) gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(np.pi, 0)
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # H-P(pi)-H should flip the qubit
        assert counts['1'] == 1000

    def test_sx_gate(self):
        """Test conversion of SX (`sqrt-X`) gate."""
        qc = QuantumCircuit(1)
        qc.sx(0)
        qc.sx(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # SX^2 = X, result should be |1>
        assert counts['1'] == 1000

    def test_identity_gate(self):
        """Test conversion of identity gate."""
        qc = QuantumCircuit(1)
        qc.id(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['0'] == 1000

    def test_barrier_ignored(self):
        """Test that barrier is properly ignored."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Bell state
        assert '00' in counts and '11' in counts

    def test_measurement(self):
        """Test that measurements are converted."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['1'] == 1000

    def test_bell_state(self):
        """Test conversion of Bell state circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
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

        kernel = cudaq.contrib.from_qiskit(qc)
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

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Both controls are 1, so target flips: |110> -> |111>
        assert counts['111'] == 1000

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

    def test_sxdg_gate(self):
        """Test conversion of SXdg (square-root-X dagger) gate."""
        qc = QuantumCircuit(1)
        qc.sx(0)
        qc.sxdg(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # SX followed by SXdg should return to |0>
        assert counts['0'] == 1000

    # ------------------------------------------------------------------ #
    # Phase / universal 1-qubit                                          #
    # ------------------------------------------------------------------ #

    def test_u1_gate(self):
        """Test conversion of U1 gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.append(U1Gate(np.pi), [0])
        qc.h(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # U1(π) applies a Z phase, so H·U1(π)·H = X
        assert counts['1'] == 1000

    def test_u2_gate(self):
        """Test conversion of U2 gate."""
        qc = QuantumCircuit(1)
        qc.append(U2Gate(0, np.pi), [0])
        qc.append(U2Gate(0, np.pi), [0])

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # U2(0, π) = H, applied twice returns |0>
        assert counts['0'] == 1000

    def test_r_gate(self):
        """Test conversion of R gate."""
        qc = QuantumCircuit(1)
        qc.r(np.pi, 0, 0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # R(π, 0) = -i·X up to global phase, so |0> → |1>
        assert counts['1'] == 1000

    # ------------------------------------------------------------------ #
    # Controlled 1-qubit                                                 #
    # ------------------------------------------------------------------ #

    def test_cs_gate(self):
        """Test conversion of CS gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        qc.cs(0, 1)
        qc.cs(0, 1)
        qc.h(1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # CS² = CZ, so H·CS·CS·H on |1+> yields |11>
        assert counts['11'] == 1000

    def test_csdg_gate(self):
        """Test conversion of CSdg gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cs(0, 1)
        qc.csdg(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # CS · CS† = I, so we stay at |10>
        assert counts['10'] == 1000

    def test_csx_gate(self):
        """Test conversion of CSX gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.csx(0, 1)
        qc.csx(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # CSX² = CX, so |10> → |11>
        assert counts['11'] == 1000

    # ------------------------------------------------------------------ #
    # Controlled phase / universal                                       #
    # ------------------------------------------------------------------ #

    def test_cphase_gate(self):
        """Test conversion of CPhase gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        qc.cp(np.pi, 0, 1)
        qc.h(1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # CP(π) = CZ, so H·CP(π)·H on |1+> yields |11>
        assert counts['11'] == 1000

    def test_cu1_gate(self):
        """Test conversion of CU1 gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        qc.append(CU1Gate(np.pi), [0, 1])
        qc.h(1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # CU1(π) = CZ, same behavior as CPhase(π)
        assert counts['11'] == 1000

    def test_cu3_gate(self):
        """Test conversion of CU3 gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(CU3Gate(np.pi, 0, np.pi), [0, 1])

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # CU3(π, 0, π) acts as CX on control=1
        assert counts['11'] == 1000

    def test_cu_gate(self):
        """Test conversion of CU gate with γ=0."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.cu(np.pi, 0, np.pi, 0, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # CU(π, 0, π, 0) reduces to CX
        assert counts['11'] == 1000

    # ------------------------------------------------------------------ #
    # Swap family                                                        #
    # ------------------------------------------------------------------ #

    def test_iswap_gate(self):
        """Test conversion of `iswap` (`iSWAP`) gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.iswap(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # `iswap` on |10> = i|01>, sampling ignores phase
        assert counts['01'] == 1000

    def test_dcx_gate(self):
        """Test conversion of DCX gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.dcx(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # DCX = CX(0,1)·CX(1,0): |10> → |11> → |01>
        assert counts['01'] == 1000

    def test_ecr_gate(self):
        """Test conversion of ECR gate."""
        qc = QuantumCircuit(2)
        qc.ecr(0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # ECR|00> = (1/√2)|q0=1,q1=0> - (i/√2)|q0=1,q1=1>.
        # In CUDA-Q big-endian counts (q0 leftmost): outcomes '10' and '11'.
        assert '00' not in counts
        assert '01' not in counts
        assert '10' in counts
        assert '11' in counts

    # ------------------------------------------------------------------ #
    # Two-qubit parametric                                               #
    # ------------------------------------------------------------------ #

    def test_ryy_gate(self):
        """Test conversion of RYY gate."""
        qc = QuantumCircuit(2)
        qc.ryy(2 * np.pi, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RYY(2π) = -I, |00> stays |00> up to a global sign
        assert counts['00'] == 1000

    def test_rzx_gate(self):
        """Test conversion of RZX gate."""
        qc = QuantumCircuit(2)
        qc.rzx(np.pi, 0, 1)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RZX(π)|00> = -i·(Z_0⊗X_1)|00> = -i|0,1>
        assert counts['01'] == 1000

    def test_xx_plus_yy_gate(self):
        """Test conversion of XXPlusYY gate."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(XXPlusYYGate(np.pi, 0), [0, 1])

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # At θ=π the gate swaps |01> ↔ |10> in Qiskit's basis;
        # starting from q0=1 the result is q0=0, q1=1.
        assert counts['01'] == 1000

    def test_xx_minus_yy_gate(self):
        """Test conversion of XXMinusYY gate."""
        qc = QuantumCircuit(2)
        qc.append(XXMinusYYGate(np.pi, 0), [0, 1])

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # At θ=π the gate swaps |00> ↔ |11>; from |00> we land on |11>.
        assert counts['11'] == 1000

    # ------------------------------------------------------------------ #
    # Multi-qubit                                                        #
    # ------------------------------------------------------------------ #

    def test_rccx_gate(self):
        """Test conversion of RCCX (relative-phase Toffoli) gate."""
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(1)
        qc.rccx(0, 1, 2)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # RCCX flips the target when both controls are 1, up to a phase
        assert counts['111'] == 1000

    def test_ccz_gate(self):
        """Test conversion of CCZ gate."""
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(1)
        qc.ccz(0, 1, 2)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # CCZ only adds a phase to |111>; the target stays |0>
        assert counts['110'] == 1000

    def test_c3x_gate(self):
        """Test conversion of C3X gate."""
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.x(1)
        qc.x(2)
        qc.append(C3XGate(), [0, 1, 2, 3])

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['1111'] == 1000

    def test_c4x_gate(self):
        """Test conversion of C4X gate."""
        qc = QuantumCircuit(5)
        qc.x(0)
        qc.x(1)
        qc.x(2)
        qc.x(3)
        qc.append(C4XGate(), [0, 1, 2, 3, 4])

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['11111'] == 1000

    def test_mcx_gate(self):
        """Test conversion of MCX gate with 5 controls."""
        qc = QuantumCircuit(6)
        for i in range(5):
            qc.x(i)
        qc.mcx([0, 1, 2, 3, 4], 5)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['111111'] == 1000

    def test_mcphase_gate(self):
        """Test conversion of MCPhase gate."""
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.x(1)
        qc.x(2)
        qc.h(3)
        qc.mcp(np.pi, [0, 1, 2], 3)
        qc.h(3)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # With all controls = 1, `mcp(π)` flips the phase of the |1> component
        # of the target |+>, turning it into |->. H brings it to |1>.
        assert counts['1111'] == 1000

    # ------------------------------------------------------------------ #
    # Other operators                                                    #
    # ------------------------------------------------------------------ #

    def test_reset(self):
        """Test conversion of reset."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.reset(0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['0'] == 1000

    def test_delay_gate(self):
        """Test that delay is converted as a no-op."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.delay(100, 0)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['1'] == 1000

    def test_global_phase_gate(self):
        """Test that GlobalPhaseGate is converted as a no-op."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.append(GlobalPhaseGate(np.pi), [])

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        # Global phase has no effect on sampling.
        assert counts['1'] == 1000

    def test_unbound_parameter_raises_error(self):
        """Test that unbound parameter expressions raise ValueError.

        Gates whose parameters are not concrete floats (e.g., an unbound
        `Parameter`) cannot be converted and should raise.
        """
        theta = Parameter('theta')
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        with pytest.raises(ValueError, match="not supported"):
            cudaq.contrib.from_qiskit(qc)

    def test_empty_circuit(self):
        """Test conversion of empty circuit."""
        qc = QuantumCircuit(2)

        kernel = cudaq.contrib.from_qiskit(qc)
        counts = cudaq.sample(kernel)

        assert counts['00'] == 1000
