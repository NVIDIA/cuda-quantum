int main() {
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // T = | 1      0     |
    //     | 0  exp(iπ/4) |
    t(qubit);
    
    return 0;
}