#include <cudaq.h>

int main() {
    cudaq::qubit c, q;
    h(c);
    x<cudaq::ctrl>(!c, q);
    // The qubits c and q are in a state (|01> + |10>) / √2.
    
    return 0;
}