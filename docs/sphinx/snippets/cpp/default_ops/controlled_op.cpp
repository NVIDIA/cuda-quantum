#include <cudaq.h>

int main() {
    // Allocate qubits in a |0> state.
    cudaq::qubit ctrl_1, ctrl_2, target;
    // Create a superposition.
    h(ctrl_1);
    // Qubit ctrl_1 is now in a state (|0> + |1>) / √2.

    // Apply the unitary transformation
    // | 1  0  0  0 |
    // | 0  1  0  0 |
    // | 0  0  0  1 |
    // | 0  0  1  0 |
    x<cudaq::ctrl>(ctrl_1, ctrl_2);
    // The qubits ctrl_1 and ctrl_2 are in a state (|00> + |11>) / √2.

    // Set the state of `target` to |1>:
    x(target);
    // Apply the transformation T only if both 
    // control qubits are in a |1> state:
    t<cudaq::ctrl>(ctrl_1, ctrl_2, target);
    // The qubits ctrl_1, ctrl_2, and target are now in a state
    // (|000> + exp(iπ/4)|111>) / √2.
    
    return 0;
}