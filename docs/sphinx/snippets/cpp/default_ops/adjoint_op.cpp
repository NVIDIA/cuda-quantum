#include <cudaq.h>

int main() {
    // Allocate a qubit in a |0> state.
    cudaq::qubit qubit;

    // Apply the unitary transformation defined by the matrix
    // T = | 1      0     |
    //     | 0  exp(iπ/4) |
    // to the state of the qubit `q`:
    t(qubit);

    // Apply its adjoint transformation defined by the matrix
    // T† = | 1      0     |
    //      | 0  exp(-iπ/4) |
    t<cudaq::adj>(qubit);
    // Qubit `q` is now again in the initial state |0>.
    
    return 0;
}