#include <cudaq.h>

int main() {
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // X = | 0  1 |
    //     | 1  0 |
    x(qubit);
    
    return 0;
}