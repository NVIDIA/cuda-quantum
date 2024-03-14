#include <cudaq.h>
namespace cudaq {

void CCNOT(qubit &a, qubit &b, qubit &c) __qpu__ {
  h(c);
  cx(b, c);
  t<cudaq::adj>(c);
  cx(a, c);
  t(c);
  cx(b, c);
  t<cudaq::adj>(c);
  cx(a, c);
  t(b);
  t(c);
  h(c);
  cx(a, b);
  t(a);
  t<cudaq::adj>(b);
  cx(a, b);
}

void CollectControls(cudaq::qview<> ctls, cudaq::qview<> aux,
                     int adjustment) __qpu__ {
  for (int i = 0; i < ctls.size() - 1; i += 2) {
    CCNOT(ctls[i], ctls[i + 1], aux[i / 2]);
  }
  for (int i = 0; i < ctls.size() / 2 - 1 - adjustment; ++i) {
    CCNOT(aux[i * 2], aux[(i * 2) + 1], aux[i + ctls.size() / 2]);
  }
}

void CollectControls(
    const std::vector<std::reference_wrapper<cudaq::qubit>> &ctls,
    cudaq::qview<> aux, int adjustment) __qpu__ {
  for (int i = 0; i < ctls.size() - 1; i += 2) {
    CCNOT(ctls[i], ctls[i + 1], aux[i / 2]);
  }
  for (int i = 0; i < ctls.size() / 2 - 1 - adjustment; ++i) {
    CCNOT(aux[i * 2], aux[(i * 2) + 1], aux[i + ctls.size() / 2]);
  }
}

void AdjustForSingleControl(cudaq::qview<> ctls, cudaq::qview<> aux) __qpu__ {
  if (ctls.size() % 2 != 0)
    CCNOT(ctls[ctls.size() - 1], aux[ctls.size() - 3], aux[ctls.size() - 2]);
}

template <size_t V, typename... T>
decltype(auto) getParameterPackVals(T &&...Args) noexcept {
  return std::get<V>(std::forward_as_tuple(std::forward<T>(Args)...));
}

template <typename mod, typename... QubitTy>
void x(cudaq::qubit& c0, cudaq::qubit& c1, QubitTy &...qubits) __qpu__ {
  static_assert(std::is_same_v<mod, cudaq::ctrl>);
  static constexpr std::size_t qubitCount = sizeof...(qubits) + 2;
  static constexpr std::size_t numCtrls = qubitCount - 1;
  static_assert(numCtrls > 1);
  if constexpr (numCtrls == 2) {
    CCNOT(c0,
          c1,
          getParameterPackVals<0>(qubits...));
  } else {
    cudaq::qvector aux(numCtrls - 2);
    std::vector<std::reference_wrapper<cudaq::qubit>> ctls{{qubits...}};
    ctls.pop_back();
    ctls.emplace_back(c1);
    ctls.emplace_back(c0);
    assert(ctls.size() == numCtrls);
    cudaq::compute_action(
        [&]() { CollectControls(ctls, aux, 1 - (ctls.size() % 2)); },
        [&]() {
          if (ctls.size() % 2 != 0) {
            CCNOT(ctls[ctls.size() - 1], aux[ctls.size() - 3], getParameterPackVals<sizeof...(qubits) - 1>(qubits...));
          } else {
            CCNOT(aux[ctls.size() - 3], aux[ctls.size() - 4], getParameterPackVals<sizeof...(qubits) - 1>(qubits...));
          }
        });
  }
}
} // namespace cudaq

int main() {

  auto kernel = []() __qpu__ {
    cudaq::qarray<5> q;
    x(q);
    x<cudaq::ctrl>(q[0], q[1], q[2], q[3], q[4]);
    mz(q);
  };

  auto counts = cudaq::sample(kernel);
  counts.dump();

  return 0;
}