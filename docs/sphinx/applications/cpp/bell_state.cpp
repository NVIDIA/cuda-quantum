#include <cudaq.h>

struct bell {
  int operator()(int num_iters) __qpu__ {
    cudaq::qarray<2> q;
    int nCorrect = 0;
    for (int i = 0; i < num_iters; i++) {
      h(q[0]);
      x<cudaq::ctrl>(q[0], q[1]);
      auto results = mz(q);
      if (static_cast<int>(results[0]) == static_cast<int>(results[1]))
        nCorrect++;

      reset(q[0]);
      reset(q[1]);
    }
    return nCorrect;
  }
};

int main() { printf("N Correct = %d\n", bell{}(100)); }
