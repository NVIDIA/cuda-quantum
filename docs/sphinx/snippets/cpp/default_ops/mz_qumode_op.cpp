#include <cudaq.h>

int main() {
  cudaq::qvector<3> qumodes(2);
  mz(qumodes);

  return 0;
}
