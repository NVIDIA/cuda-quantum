#include <cudaq.h>

int main() {
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // Rx(θ) = |  cos(θ/2)  -isin(θ/2) |
    //         | -isin(θ/2)  cos(θ/2)  |
    rx(std::numbers::pi, qubit);
    
    return 0;
}