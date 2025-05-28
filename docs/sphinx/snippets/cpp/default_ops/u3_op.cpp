#include <cudaq.h>

int main() {
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // U3(θ,φ,λ) = | cos(θ/2)            -exp(iλ) * sin(θ/2)       |
    //             | exp(iφ) * sin(θ/2)   exp(i(λ + φ)) * cos(θ/2) |
    u3(M_PI, M_PI, M_PI_2, qubit);
    
    return 0;
}