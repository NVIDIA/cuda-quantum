.. |:spellcheck-disable:| replace:: \

Noisy Simulation with PTSBE
++++++++++++++++++++++++++++

.. _ptsbe:

Pre-Trajectory Sampling with Batch Execution (PTSBE) is an efficient method
for sampling noisy quantum circuits [Patti2025]_. It is a powerful generalization of the
trajectories methods for noisy quantum systems, that stochastically sample
noise operators from a noisy quantum circuit and then subsequently build
and sample from the corresponding pseudo-coherent quantum states rather than sampling from
the full density matrix of a system, as this is quadratically larger than
each pseudo-coherent quantum statevector [Carmichael2007]_. While these trajectory methods
can be much more efficient than constructing and sampling from full density
matrices, they traditionally sampled only once shot per constructed state.
In contrast, PTSBE *pre-samples* a set of noise
realizations (trajectories) from the circuit's noise model and then *batches*
circuit executions by unique trajectory [Patti2025]_. As the noise pre-sampling and state post-sampling
are tasks with only low-degree polynomial complexity, while the state construction
is, in general, of exponential complexity, PTSBE allows us to gather noisy quantum data
orders of magnitude quicker than traditional trajectory sampling methods by allowing finely-tuned
batched sampling. PTSBE can be used to capture millions of times more noisy shot data, which can
then be used as e.g., training data in ML tasks such as AI decoders, or it can be deployed proportionally
capturing the exact statistics of the problem while stiff offering a considerable speedup. In particular,
PTSBE achieves traditional trajectory formalism accuracy at a fraction of
the computational cost when the number of unique trajectories (errors) is much smaller than the total
shot count [Patti2025]_.

Conceptual Overview
^^^^^^^^^^^^^^^^^^^^

A quantum circuit subject to noise can be described by a set of Kraus operators
applied at each gate location. At each noise site, the environment selects one
Kraus operator with some probability. A **trajectory** is one complete
assignment of Kraus operators across all noise sites in the circuit. Its
probability is the product of the probabilities of the chosen operators at
each site.

PTSBE works in three phases:

.. list-table::
   :widths: 10 30 60
   :header-rows: 1

   * - Phase
     - Name
     - Description
   * - 1
     - Trajectory Sampling
     - Draw *T* unique trajectories from the full noise space using a sampling
       strategy. Each trajectory specifies which Kraus operator fires at every
       noise site.
   * - 2
     - Shot Allocation
     - Distribute the total *N* shots across the *T* trajectories according to
       a shot allocation strategy (e.g. proportional to trajectory probability).
   * - 3
     - Batch Execution
     - Simulate each trajectory as a pure-state circuit. The per-trajectory
       measurement outcomes are merged into a single :class:`~cudaq.SampleResult`.

Because trajectories are reused across many shots, the number of circuit
simulations scales with the number of unique trajectories *T*, not the shot
count *N*.

When to Use PTSBE
^^^^^^^^^^^^^^^^^^

PTSBE is most beneficial when:

- The circuit has **few distinct noise sites** so the trajectory space is
  manageable.
- A **large shot count** is required (1 000 – 1 000 000+) so the reuse of
  trajectories provides a significant speed-up.
- The shots are intended for a data-hungry downstream task that is not necessarily
  inhibited by correlated sampling, such as training AI models

Benchmarks from the original paper [Patti2025]_ illustrate the potential
speed-ups:

- **35-qubit** statevector simulation (magic state distillation): up to
  **10⁶×** speedup over conventional trajectory methods, producing one
  trillion shots on 4 NVIDIA H100 GPUs.
- **85-qubit** tensor network simulation (magic state distillation): **16×**
  speedup, producing one million shots.

PTSBE is particularly well-suited for generating large synthetic datasets of
noisy measurement outcomes, such as training data for machine-learning–based
quantum error correction (QEC) decoders [Patti2025]_.

PTSBE requires:

- A **static circuit** — no mid-circuit measurements or
  measurement-dependent conditional logic.
- A **local simulator** backend.

Quick Start
^^^^^^^^^^^^

The example below simulates a two-qubit Bell circuit under depolarizing noise.

.. tab:: Python

   .. code-block:: python

      import cudaq
      from cudaq import ptsbe

      cudaq.set_target("nvidia")

      @cudaq.kernel
      def bell():
          q = cudaq.qvector(2)
          h(q[0])
          cx(q[0], q[1])
          mz(q)

      noise = cudaq.NoiseModel()
      noise.add_channel("h",  [0], cudaq.DepolarizationChannel(0.01))
      noise.add_channel("cx", [0, 1], cudaq.Depolarization2Channel(0.005))

      result = ptsbe.sample(bell, shots_count=10_000, noise_model=noise)
      print(result)

.. tab:: C++

   .. code-block:: cpp

      #include "cudaq.h"
      #include "cudaq/ptsbe/PTSBESample.h"

      auto bell = []() __qpu__ {
        cudaq::qvector<2> q;
        h(q[0]);
        cx(q[0], q[1]);
        mz(q);
      };

      int main() {
        auto noise = cudaq::noise_model{};
        noise.add_channel<cudaq::depolarization_channel>("h",  {0},     0.01);
        noise.add_channel<cudaq::depolarization2_channel>("cx", {0, 1}, 0.005);

        cudaq::ptsbe::sample_options opts;
        opts.shots  = 10000;
        opts.noise  = noise;

        auto result = cudaq::ptsbe::sample(opts, bell);
        result.dump();
      }

Usage Tutorial
^^^^^^^^^^^^^^^

Controlling the Number of Trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, PTSBE generates up to ``shots_count`` unique trajectories.
For large shot counts set ``max_trajectories`` to cap trajectory generation
and gain the batching benefit:

.. tab:: Python

   .. code-block:: python

      result = ptsbe.sample(
          bell,
          shots_count=100_000,
          noise_model=noise,
          max_trajectories=500,   # reuse each trajectory ~200 times on average
      )

.. tab:: C++

   .. code-block:: cpp

      cudaq::ptsbe::sample_options opts;
      opts.shots = 100'000;
      opts.noise = noise;
      opts.ptsbe.max_trajectories = 500;

      auto result = cudaq::ptsbe::sample(opts, bell);

Choosing a Trajectory Sampling Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Four strategies control which trajectories are selected from the noise space:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Strategy
     - Description
   * - **Probabilistic** *(default)*
     - Randomly samples unique trajectories weighted by probability. Produces
       a representative cross-section of the noise space.
   * - **Ordered**
     - Selects the top-*T* highest-probability trajectories. Best when the
       noise space is dominated by a small number of likely error patterns.
   * - **Exhaustive**
     - Enumerates every possible trajectory. Use only when the noise space is
       small (few qubits and low-weight noise).
   * - **Conditional**
     - Keeps only trajectories that satisfy a user-supplied predicate. Useful
       for targeted studies (e.g. only single-qubit error events).

.. tab:: Python

   .. code-block:: python

      from cudaq.ptsbe import (ProbabilisticSamplingStrategy,
                               OrderedSamplingStrategy,
                               ExhaustiveSamplingStrategy,
                               ConditionalSamplingStrategy)

      # Reproducible probabilistic sampling
      result = ptsbe.sample(
          bell,
          shots_count=10_000,
          noise_model=noise,
          sampling_strategy=ProbabilisticSamplingStrategy(seed=42),
      )

      # Top-100 trajectories by probability
      result = ptsbe.sample(
          bell,
          shots_count=10_000,
          noise_model=noise,
          max_trajectories=100,
          sampling_strategy=OrderedSamplingStrategy(),
      )

.. tab:: C++

   .. code-block:: cpp

      #include "cudaq/ptsbe/strategies/ProbabilisticSamplingStrategy.h"
      #include "cudaq/ptsbe/strategies/OrderedSamplingStrategy.h"

      // Reproducible probabilistic sampling
      cudaq::ptsbe::sample_options opts;
      opts.ptsbe.strategy =
          std::make_shared<cudaq::ptsbe::ProbabilisticSamplingStrategy>(/*seed=*/42);

      // Top-100 trajectories
      opts.ptsbe.max_trajectories = 100;
      opts.ptsbe.strategy =
          std::make_shared<cudaq::ptsbe::OrderedSamplingStrategy>();

Shot Allocation Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After trajectories are selected, shots are distributed across them:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Strategy
     - Description
   * - **Proportional** *(default)*
     - Each trajectory receives shots proportional to its probability.
       Uses multinomial sampling — total is always exact and every trajectory
       with non-zero probability receives a fair share.
   * - **Uniform**
     - Equal shots per trajectory regardless of probability.
   * - **Low-weight bias**
     - Biases more shots toward trajectories with fewer errors (lower Kraus
       weight). Useful when low-error events dominate the observable of
       interest.
   * - **High-weight bias**
     - Biases more shots toward high-error trajectories. Useful for studying
       rare error events.

.. tab:: Python

   .. code-block:: python

      from cudaq.ptsbe import ShotAllocationStrategy

      result = ptsbe.sample(
          bell,
          shots_count=10_000,
          noise_model=noise,
          shot_allocation=ShotAllocationStrategy(
              ShotAllocationStrategy.Type.LOW_WEIGHT_BIAS,
              bias_strength=2.0,
          ),
      )

.. tab:: C++

   .. code-block:: cpp

      #include "cudaq/ptsbe/ShotAllocationStrategy.h"

      cudaq::ptsbe::sample_options opts;
      opts.ptsbe.shot_allocation = cudaq::ptsbe::ShotAllocationStrategy(
          cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS,
          /*bias=*/2.0);

Inspecting Execution Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``return_execution_data=True`` to attach the full execution trace —
circuit instructions, sampled trajectories, and per-trajectory counts — to
the result:

.. tab:: Python

   .. code-block:: python

      result = ptsbe.sample(
          bell,
          shots_count=1_000,
          noise_model=noise,
          return_execution_data=True,
      )

      data = result.execution_data()

      # Circuit structure
      for inst in data.instructions:
          print(inst.type, inst.name, inst.targets)

      # Trajectory details
      for trajectory in data.trajectories:
          print(f"id={trajectory.trajectory_id}  p={trajectory.probability:.4f}"
                f"  shots={trajectory.num_shots}  errors={trajectory.count_errors()}")

.. tab:: C++

   .. code-block:: cpp

      cudaq::ptsbe::sample_options opts;
      opts.ptsbe.return_execution_data = true;

      auto result = cudaq::ptsbe::sample(opts, bell);

      if (result.has_execution_data()) {
        const auto &data = result.execution_data();
        for (const auto &trajectory : data.trajectories)
          printf("id=%zu  p=%.4f  shots=%zu\n",
                 trajectory.trajectory_id, trajectory.probability, trajectory.num_shots);
      }

Asynchronous Execution
~~~~~~~~~~~~~~~~~~~~~~~

Use ``sample_async`` to submit the job without blocking:

.. tab:: Python

   .. code-block:: python

      future = ptsbe.sample_async(bell, shots_count=10_000, noise_model=noise)
      # ... do other work ...
      result = future.get()

.. tab:: C++

   .. code-block:: cpp

      auto future = cudaq::ptsbe::sample_async(opts, bell);
      // ... do other work ...
      auto result = future.get();

Trajectory vs Shot Trade-offs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The central tension in PTSBE is between **trajectory count** *T* and
**shots per trajectory** *N/T*.

.. rubric:: Increasing trajectories

- Covers more of the noise space → lower bias in the estimated distribution.
- Each trajectory is simulated independently → linear scaling in simulation
  cost.
- Diminishing returns once *T* approaches the total trajectory space size.

.. rubric:: Decreasing trajectories (fewer, reused more)

- Each trajectory accumulates more shots → lower shot-noise variance for that
  trajectory.
- Fewer circuit simulations → lower wall-clock time.
- Risk of bias if high-probability regions of the noise space are under-sampled.

.. rubric:: Practical guidance

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Scenario
     - Recommendation
   * - Few noise sites, low error rates
     - Use **Exhaustive** strategy; all trajectories have manageable count.
   * - Many noise sites, high error rates
     - Use **Probabilistic** with ``max_trajectories`` ≈ √N; the proportional
       shot allocation handles variance automatically.
   * - Studying low-error observables
     - Use **Ordered** or **Low-weight bias** to concentrate shots on the most
       probable (low-error) trajectories.
   * - Studying rare error events
     - Use **High-weight bias** to allocate more shots to high-error
       trajectories.
   * - Accuracy validation
     - Compare against a standard density-matrix run. Hellinger fidelity
       *F* ≈ 1 indicates PTSBE is faithfully reproducing the full distribution.

As a rule of thumb, ``max_trajectories`` between 100 and 10 000 covers the
majority of practical use cases. Below 100, bias may dominate. Above 10 000,
the simulation cost approaches that of a conventional density-matrix run.

Backend Requirements
^^^^^^^^^^^^^^^^^^^^^

PTSBE requires a backend that supports trajectory-based noisy simulation.
The supported targets are those marked with ``*`` in the
`simulator table <https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html>`_,
plus ``density-matrix-cpu`` and ``qpp-cpu``:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Target
     - Notes
   * - ``nvidia``
     - Single GPU, single/double precision. General purpose.
   * - ``nvidia, option=mgpu``
     - Multi-GPU / multi-node. Suitable for large qubit counts (33+).
   * - ``nvidia, option=mqpu``
     - Multi-QPU asynchronous distribution across multiple GPUs.
   * - ``tensornet``
     - Tensor network method. Exact simulation of shallow-depth circuits.
       Handles thousands of qubits.
   * - ``tensornet-mps``
     - Matrix product state (approximate). Efficient for square-shaped circuits.
   * - ``remote-mqpu``
     - Combines ``mqpu`` with other backends for distributed execution.
   * - ``density-matrix-cpu``
     - CPU density matrix simulator. Recommended for small noisy circuits
       (< 14 qubits) and development/testing.
   * - ``qpp-cpu``
     - CPU state vector simulator. Lightweight option for small circuits
       (< 28 qubits).

Set the target before calling :func:`cudaq.ptsbe.sample`:

.. tab:: Python

   .. code-block:: python

      # Single GPU (most common for production)
      cudaq.set_target("nvidia")

      # CPU density matrix (development / small circuits)
      cudaq.set_target("density-matrix-cpu")

      # Multi-GPU for large circuits
      cudaq.set_target("nvidia", option="mgpu")

      # Tensor network for wide shallow circuits
      cudaq.set_target("tensornet")

.. tab:: C++

   .. code-block:: cpp

      // Set via CMake target or the --target flag at runtime.
      // See backend documentation for available options.

See :doc:`backends/backends` for full details on each target including precision and
qubit count limits.

Related Approaches
^^^^^^^^^^^^^^^^^^^

TUSQ [Dangwal2025]_ is an alternative noisy-simulation framework addressing
the same problem with a complementary set of techniques:

- **Error Realization (ER) Tallying**: Samples noise channels once to identify
  unique error realizations, then simulates each unique circuit once and
  samples from its output multiple times — conceptually similar to PTSBE's
  trajectory deduplication.
- **ER Commutation**: Pushes Pauli noise gates rightward through the circuit
  using gate commutation rules, merging additional circuits whose error
  patterns are functionally equivalent after commutation.
- **Depth-First Tree Traversal with Uncomputation**: Represents circuits that
  share a common gate prefix as a tree, traverses depth-first, and
  *uncomputes* backward using inverse gates before branching to the next
  circuit — achieving computation reuse with zero extra memory overhead.

TUSQ reports an average speedup of 52.5× over Qiskit and 12.53× over CUDA-Q
(up to 30 qubits), with peak gains of 7878× and 439× respectively on larger
benchmarks [Dangwal2025]_.

References
^^^^^^^^^^^

.. [Carmichael2007] Carmichael, H. J. *Quantum jumps revisited: An overview of quantum trajectory theory.* Quantum Future From Volta and
   Como to the Present and Beyond: Proceedings of the Xth Max Born Symposium Held in Przesieka, Poland, 24–27 September 1997. Berlin,
   Heidelberg: Springer Berlin Heidelberg, 2007.
   https://link.springer.com/chapter/10.1007/bfb0105336

.. [Patti2025] Taylor L. Patti, Thien Nguyen, Justin G. Lietz,
   Alexander J. McCaskey, Brucek Khailany,
   *Augmenting Simulated Noisy Quantum Data Collection by Orders of Magnitude Using Pre-Trajectory Sampling with Batched Execution.*
   Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 2025.
   https://dl.acm.org/doi/full/10.1145/3712285.3759871

.. [Dangwal2025] Siddharth Dangwal, Tina Oberoi, Ajay Sailopal,
   Dhirpal Shah, Frederic T. Chong,
   *Noisy Quantum Simulation Using Tracking, Uncomputation and Sampling*,
   arXiv:2508.04880 (2025).
   https://arxiv.org/abs/2508.04880
.. |:spellcheck-enable:| replace:: \