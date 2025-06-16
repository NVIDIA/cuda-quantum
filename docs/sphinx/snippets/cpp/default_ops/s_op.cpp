int main() {
  cudaq::qubit qubit;

  // Apply the unitary transformation
  // S = | 1   0 |
  //     | 0   i |
  s(qubit);

  return 0;
}
