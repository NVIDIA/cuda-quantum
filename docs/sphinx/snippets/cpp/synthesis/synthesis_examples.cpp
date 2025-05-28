#include <cudaq.h>
#include <vector>
#include <iostream> 

// [Begin CPP ControlAndAdjointCombined]
// Part 1a: x_gate function for control example
__qpu__ void x_gate_for_control_cpp(cudaq::qubit& q) { x(q); }

// Part 1b: kernel struct demonstrating cudaq::control
struct control_example_kernel_cpp {
  void operator() () __qpu__ {
    cudaq::qarray<3> q_reg;
    // Initialize q_reg[0] and q_reg[1] to |1> for Toffoli behavior
    x(q_reg[0]);
    x(q_reg[1]);
    
    auto ctrl_bits = q_reg.front(2); // First 2 qubits as controls
    // Apply x_gate_for_control_cpp to q_reg[2] if ctrl_bits are |11>
    cudaq::control(x_gate_for_control_cpp, ctrl_bits, q_reg[2]);
    
    mz(q_reg); // Measure all qubits
  }
};

// Part 1c: rx_and_h_gate function for adjoint example
__qpu__ void rx_and_h_gate_for_adjoint_cpp(double angle, cudaq::qubit& q) { 
  rx(angle, q); 
  h(q); 
}

// Part 1d: kernel function demonstrating cudaq::adjoint
// Note: Original RST showed "__qpu__ kernel(int N)". Renamed to avoid conflict.
__qpu__ void adjoint_example_kernel_cpp(int N_qubits, double angle_param) {
  cudaq::qvector q_vec(N_qubits);
  if (N_qubits == 0) return; // Or handle error

  cudaq::qubit& target_qubit = q_vec[0]; // Use first qubit as target for simplicity
  
  // Example: Apply original operations first
  // rx_and_h_gate_for_adjoint_cpp(angle_param, target_qubit);

  // Apply adjoint of rx_and_h_gate_for_adjoint_cpp
  // This will effectively apply h, then rx(-angle_param)
  cudaq::adjoint(rx_and_h_gate_for_adjoint_cpp, angle_param, target_qubit);
  
  mz(q_vec); // Measure all qubits
}
// [End CPP ControlAndAdjointCombined]

// Helper callable for negated control example, as `kernel{}` in RST
struct simple_op_cpp {
    void operator()(cudaq::qubit& target_qubit_arg) __qpu__ {
        // This is the "kernel" or operation being controlled.
        // The `kernel_arg` from RST is `target_qubit_arg` here.
        x(target_qubit_arg); // Example: apply X to the target
    }
};

// Kernel demonstrating negated control line from RST
struct negated_control_usage_kernel_cpp {
    void operator()() __qpu__ {
        cudaq::qarray<3> q_reg; // For qubit0, qubit1, and kernel_arg (as target)
        cudaq::qubit& qubit0 = q_reg[0];
        cudaq::qubit& qubit1 = q_reg[1];
        cudaq::qubit& kernel_arg_target = q_reg[2]; 

        // Example setup:
        // x(qubit1); // To make qubit1 |1>, so !qubit1 (control on |0>) is false.
                   // If qubit1 is |0> (default), !qubit1 is true.

        simple_op_cpp kernel_instance; // This represents `kernel{}` from RST
        // [Begin CPP NegatedControlRSTLine]
        cudaq::control(kernel_instance, {qubit0, !qubit1}, kernel_arg_target);
        // [End CPP NegatedControlRSTLine]
        mz(q_reg);
    }
};


int main() {
  std::cout << "C++ Control Example (Toffoli-like):" << std::endl;
  auto counts_control = cudaq::sample(control_example_kernel_cpp{});
  counts_control.dump(); // Expect |111> if x_gate_for_control_cpp is X

  std::cout << "\nC++ Adjoint Example:" << std::endl;
  // For adjoint, if we apply Op then Op_adj, we should return to initial state (often |0...0>)
  // Let's run a sequence: H, RX(pi/2), then adjoint(RX_H_gate, pi/2)
  // RX_H_gate(pi/2) = RX(pi/2) H
  // Adjoint(RX_H_gate, pi/2) = H RX(-pi/2)
  // So sequence is: H RX(pi/2) H RX(-pi/2)
  struct full_adjoint_test {
    void operator()() __qpu__ {
        cudaq::qubit q;
        rx_and_h_gate_for_adjoint_cpp(M_PI_2, q); // RX(pi/2) H
        cudaq::adjoint(rx_and_h_gate_for_adjoint_cpp, M_PI_2, q); // H RX(-pi/2)
        mz(q);
    }
  };
  auto counts_adjoint_full = cudaq::sample(full_adjoint_test{});
  counts_adjoint_full.dump(); // Result depends on the specific sequence.

  // Test just the adjoint kernel part as in RST
  auto counts_adjoint_rst = cudaq::sample(adjoint_example_kernel_cpp, 3, M_PI_4);
  counts_adjoint_rst.dump();


  std::cout << "\nC++ Negated Control Example:" << std::endl;
  auto counts_neg_ctrl = cudaq::sample(negated_control_usage_kernel_cpp{});
  counts_neg_ctrl.dump(); 
  // If qubit0=|0> (true), qubit1=|0> (!qubit1 is true), then X applied to target.
  // Default state |000> -> control active -> |001>

  return 0;
}