#include <cudaq.h>

int main() {
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // H = (1 / sqrt(2)) * | 1   1 |
    //                     | 1  -1 |
    h(qubit);
    
    return 0;
}