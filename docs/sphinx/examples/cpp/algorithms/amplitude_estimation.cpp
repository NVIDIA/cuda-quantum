// Compile and run with:
// ```
// nvq++ amplitude_estimation.cpp -o mlae.x && ./mlae.x
// ```

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/optimizers.h>

// This file implements Quantum Monte Carlo integration using Amplitude
// Estimation. To compute a definite integral of the form sin^2(x) from
// `0` to `bmax`, we use the maximum likelihood estimation method [1].
// The kernels have been given similar names as their corresponding
// unitary operators from equation (1) in [1].
//
// [1] Amplitude estimation without phase estimation by Suzuki et al.
// (https://arxiv.org/pdf/1904.10246.pdf)

namespace mlae {

double loglikelihood(std::vector<int> &h, std::vector<int> &mk, int N_shots,
                     double theta_val) {

  double sum = 0;
  for (int i = 0; i < h.size(); i++) {

    double arg = (2 * mk[i] + 1) * theta_val;
    sum += 2 * h[i] * log(fabs(sin(arg))) +
           2 * (N_shots - h[i]) * log(fabs(cos(arg)));
  }

  return -sum;
}

double discretized_integral(const int n_qubits, double bmax) {

  int N = pow(2, n_qubits) - 1;
  double sum = 0.0;
  double denom = pow(2, n_qubits);

  for (int i = 0; i <= N; i++) {

    sum += pow(sin((i + 0.5) * bmax / denom), 2);
  }

  return sum / denom;
}

int num_oracles(std::vector<int> &vec, const int N_shots) {

  int total_oracles = 0;
  for (int i = 0; i < vec.size(); i++) {
    total_oracles += N_shots * (2 * vec[i] + 1);
  }

  return total_oracles;
}
} // namespace mlae

struct statePrep_A {
  void operator()(cudaq::qreg<> &q, const double bmax) __qpu__ {

    int n = q.size();
    // all qubits sans auxiliary
    auto qubit_subset = q.front(n - 1);

    h(qubit_subset);

    ry(bmax / pow(2.0, n - 1), q[n - 1]);

    for (int i = 1; i < n; i++) {
      ry<cudaq::ctrl>(bmax / pow(2.0, n - i - 1), q[i - 1], q[n - 1]);
    }
  }
};

struct S_0 {
  void operator()(cudaq::qreg<> &q) __qpu__ {

    auto ctrl_qubits = q.front(q.size() - 1);
    auto &last_qubit = q.back();

    x(q);
    h(last_qubit);
    x<cudaq::ctrl>(ctrl_qubits, last_qubit);
    h(last_qubit);
    x(q);
  }
};

struct run_circuit {

  auto operator()(const int n_qubits, const int n_itrs,
                  const double bmax) __qpu__ {

    cudaq::qreg q(n_qubits + 1); // last is auxiliary
    auto &last_qubit = q.back();

    // State preparation
    statePrep_A{}(q, bmax);

    // Amplification Q^m_k as per evaluation schedule {m_0,m_1,..,m_k,..}
    for (int i = 0; i < n_itrs; ++i) {

      z(last_qubit);
      cudaq::adjoint(statePrep_A{}, q, bmax);
      S_0{}(q);
      statePrep_A{}(q, bmax);
    }
    // Measure the last auxiliary qubit
    mz(last_qubit);
  }
};

int main() {

  const int n = 10; // number of qubits (not including the auxiliary qubit)
  const double bmax = M_PI * 0.25; // upper bound of the integral

  // Specify your evaluation schedule
  std::vector<int> schedule{0, 1, 2, 4};
  std::vector<int> hits(schedule.size(),
                        0); // #hits, input for post-processing

  for (size_t i = 0; i < schedule.size(); i++) {
    auto counts = cudaq::sample(run_circuit{}, n, schedule[i], bmax);
    hits[i] = counts.count("1");
  }

  // Print the number of hits for the good state for each circuit
  for (size_t i = 0; i < hits.size(); ++i)
    printf("%d ", hits[i]);

  printf("\n");

  // Optimization (Replace with brute force)
  cudaq::optimizers::cobyla optimizer;

  // Specify initial value of optimization parameters
  std::vector<double> theta{0.8}; // COBYLA is very sensitive to the start
  optimizer.initial_parameters = theta;

  // Specify bounds for theta
  optimizer.lower_bounds = {0.0};
  optimizer.upper_bounds = {M_PI / 2.0};

  // Using 1000 shots in log-likelihood, which is the default #shots
  auto [opt_val, opt_params] =
      optimizer.optimize(1, [&](const std::vector<double> &theta) {
        auto f = mlae::loglikelihood(hits, schedule, 1000, theta[0]);
        printf("theta = %18.16e\n", theta[0]);
        return f;
      });

  printf("opt val = %f, theta = %f\n", opt_val, opt_params[0]);

  double a_estimated = sin(opt_params[0]) * sin(opt_params[0]);
  printf("Estimated a is: %f\n", a_estimated);

  // The integral is discretized using 2^n intervals
  double a_discretized = mlae::discretized_integral(n, bmax);
  printf("Discretized a is: %f\n", a_discretized);

  printf("Relative error w.r.t. discretized version is: %3.2f %% \n",
         100 * fabs((a_discretized - a_estimated) / a_estimated));

  return 0;
}
