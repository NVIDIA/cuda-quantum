int main() {
    cudaq::qubit qubit_1, qubit_2;

    // Apply the unitary transformation
    // Swap = | 1 0 0 0 |
    //        | 0 0 1 0 |
    //        | 0 1 0 0 |
    //        | 0 0 0 1 |
    swap(qubit_1, qubit_2);
    
    return 0;
}