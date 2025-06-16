#include <cudaq.h>

int main() {
  cudaq::qvector<3> q(2);
  beam_splitter(q[0], q[1], 0.34);

  return 0;
}
