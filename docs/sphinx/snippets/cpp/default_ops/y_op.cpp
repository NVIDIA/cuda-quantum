#include <cudaq.h>

int main() {
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // Y = | 0  -i |
    //     | i   0 |
    y(qubit);
    
    return 0;
}