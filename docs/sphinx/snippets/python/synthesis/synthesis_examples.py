import cudaq
import numpy as np

# [Begin PY ControlAndAdjointCombined]
# Part 1a: x_gate function for control example
@cudaq.kernel()
def x_gate_for_control_py(q : cudaq.qubit):
    x(q)

# Part 1b: kernel demonstrating cudaq.control
@cudaq.kernel()
def control_example_kernel_py():
    q_reg = cudaq.qvector(3)
    # Initialize q_reg[0] and q_reg[1] to |1> for Toffoli behavior
    x(q_reg[0])
    x(q_reg[1])
    
    ctrl_bits = q_reg.front(2) # First 2 qubits as controls
    # Apply x_gate_for_control_py to q_reg[2] if ctrl_bits are |11>
    cudaq.control(x_gate_for_control_py, ctrl_bits, q_reg[2])
    
    mz(q_reg) # Measure all qubits

# Part 1c: rx_and_h_gate function for adjoint example
@cudaq.kernel()
def rx_and_h_gate_for_adjoint_py(angle: float, q : cudaq.qubit):
    rx(angle, q)
    h(q)

# Part 1d: kernel demonstrating cudaq.adjoint
@cudaq.kernel()
def adjoint_example_kernel_py(N_qubits: int, angle_param: float):
    q_vec = cudaq.qvector(N_qubits)
    if N_qubits == 0: return

    target_qubit = q_vec[0] # Use first qubit as target for simplicity
    
    # Example: Apply original operations first
    # rx_and_h_gate_for_adjoint_py(angle_param, target_qubit)

    # Apply adjoint of rx_and_h_gate_for_adjoint_py
    cudaq.adjoint(rx_and_h_gate_for_adjoint_py, angle_param, target_qubit)
    
    mz(q_vec) # Measure all qubits
# [End PY ControlAndAdjointCombined]

# Helper callable for negated control example, as `kernel` in RST
@cudaq.kernel
def simple_op_py(target_qubit_arg: cudaq.qubit):
    # This is the "kernel" or operation being controlled.
    # The `kernel_arg` from RST is `target_qubit_arg` here.
    x(target_qubit_arg) # Example: apply X to the target

# Kernel demonstrating negated control line from RST
@cudaq.kernel
def negated_control_usage_kernel_py():
    q_reg = cudaq.qvector(3) # For qubit0, qubit1, and kernel_arg (as target)
    qubit0 = q_reg[0]
    qubit1 = q_reg[1]
    kernel_arg_target = q_reg[2]

    # Example setup:
    # x(qubit1) # To make qubit1 |1>, so ~qubit1 (control on |0>) is false.

    kernel_to_control = simple_op_py # This represents `kernel` from RST
    # [Begin PY NegatedControlRSTLine]
    cudaq.control(kernel_to_control, [qubit0, ~qubit1], kernel_arg_target)
    # [End PY NegatedControlRSTLine]
    mz(q_reg)


if __name__ == "__main__":
    print("Python Control Example (Toffoli-like):")
    counts_control_py = cudaq.sample(control_example_kernel_py)
    counts_control_py.dump()

    print("\nPython Adjoint Example:")
    @cudaq.kernel
    def full_adjoint_test_py():
        q = cudaq.qubit()
        rx_and_h_gate_for_adjoint_py(np.pi/2.0, q)
        cudaq.adjoint(rx_and_h_gate_for_adjoint_py, np.pi/2.0, q)
        mz(q)
    counts_adjoint_full_py = cudaq.sample(full_adjoint_test_py)
    counts_adjoint_full_py.dump()
    
    counts_adjoint_rst_py = cudaq.sample(adjoint_example_kernel_py, 3, np.pi/4.0)
    counts_adjoint_rst_py.dump()

    print("\nPython Negated Control Example:")
    counts_neg_ctrl_py = cudaq.sample(negated_control_usage_kernel_py)
    counts_neg_ctrl_py.dump()