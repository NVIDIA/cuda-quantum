::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../index.html){.icon .icon-home}

::: version
pr-4370
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](../quick_start.html){.reference .internal}
    -   [Install CUDA-Q](../quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](../quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q
        Academic](../quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](../basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q
        Kernel?](../basics/kernel_intro.html){.reference .internal}
    -   [Building your first CUDA-Q
        Program](../basics/build_kernel.html){.reference .internal}
    -   [Running your first CUDA-Q
        Program](../basics/run_kernel.html){.reference .internal}
        -   [Sample](../basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](../basics/run_kernel.html#run){.reference .internal}
        -   [Observe](../basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](../basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](../basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](../basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
        -   [Python
            Stack-Traces](../basics/troubleshooting.html#python-stack-traces){.reference
            .internal}
-   [Examples](../examples/examples.html){.reference .internal}
    -   [Introduction](../examples/introduction.html){.reference
        .internal}
    -   [Building Kernels](../examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](../examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](../examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](../examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](../examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](../examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](../examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](../examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](../examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](../examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](../examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](../examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](../examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](../examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring
        Kernels](../examples/measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](../examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](../../examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](../../examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](../../examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing
        Kernels](../examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](../examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](../examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](../examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](../examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](../examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](../examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](../examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](../examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](../examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](../examples/expectation_values.html){.reference
        .internal}
        -   [Parallelizing across Multiple
            Processors](../examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](../examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](../examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](../examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](../examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends ([`remote-mqpu`{.code .docutils
            .literal
            .notranslate}]{.pre})](../examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
            .internal}
    -   [Optimizers &
        Gradients](../../examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [CUDA-Q Optimizer
            Overview](../../examples/python/optimizers_gradients.html#CUDA-Q-Optimizer-Overview){.reference
            .internal}
            -   [Gradient-Free Optimizers (no gradients
                required):](../../examples/python/optimizers_gradients.html#Gradient-Free-Optimizers-(no-gradients-required):){.reference
                .internal}
            -   [Gradient-Based Optimizers (require
                gradients):](../../examples/python/optimizers_gradients.html#Gradient-Based-Optimizers-(require-gradients):){.reference
                .internal}
        -   [1. Built-in CUDA-Q Optimizers and
            Gradients](../../examples/python/optimizers_gradients.html#1.-Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
            -   [1.1 Adam Optimizer with Parameter
                Configuration](../../examples/python/optimizers_gradients.html#1.1-Adam-Optimizer-with-Parameter-Configuration){.reference
                .internal}
            -   [1.2 SGD (Stochastic Gradient Descent)
                Optimizer](../../examples/python/optimizers_gradients.html#1.2-SGD-(Stochastic-Gradient-Descent)-Optimizer){.reference
                .internal}
            -   [1.3 SPSA (Simultaneous Perturbation Stochastic
                Approximation)](../../examples/python/optimizers_gradients.html#1.3-SPSA-(Simultaneous-Perturbation-Stochastic-Approximation)){.reference
                .internal}
        -   [2. Third-Party
            Optimizers](../../examples/python/optimizers_gradients.html#2.-Third-Party-Optimizers){.reference
            .internal}
        -   [3. Parallel Parameter Shift
            Gradients](../../examples/python/optimizers_gradients.html#3.-Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](../../examples/python/noisy_simulations.html){.reference
        .internal}
    -   [Pre-Trajectory Sampling with Batch
        Execution](../examples/ptsbe.html){.reference .internal}
        -   [Conceptual
            Overview](../examples/ptsbe.html#conceptual-overview){.reference
            .internal}
        -   [When to Use
            PTSBE](../examples/ptsbe.html#when-to-use-ptsbe){.reference
            .internal}
        -   [Quick Start](../examples/ptsbe.html#quick-start){.reference
            .internal}
        -   [Usage
            Tutorial](../examples/ptsbe.html#usage-tutorial){.reference
            .internal}
            -   [Controlling the Number of
                Trajectories](../examples/ptsbe.html#controlling-the-number-of-trajectories){.reference
                .internal}
            -   [Choosing a Trajectory Sampling
                Strategy](../examples/ptsbe.html#choosing-a-trajectory-sampling-strategy){.reference
                .internal}
            -   [Shot Allocation
                Strategies](../examples/ptsbe.html#shot-allocation-strategies){.reference
                .internal}
            -   [Inspecting Execution
                Data](../examples/ptsbe.html#inspecting-execution-data){.reference
                .internal}
    -   [Constructing Operators](../examples/operators.html){.reference
        .internal}
        -   [Constructing Spin
            Operators](../examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](../examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](../../examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](../../examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](../examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](../examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](../examples/hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](../examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](../examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](../examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](../examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](../examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](../examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](../examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum Circuits,
            Inc.](../examples/hardware_providers.html#quantum-circuits-inc){.reference
            .internal}
        -   [Quantum
            Machines](../examples/hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](../examples/hardware_providers.html#quera-computing){.reference
            .internal}
        -   [Scaleway](../examples/hardware_providers.html#scaleway){.reference
            .internal}
        -   [TII](../examples/hardware_providers.html#tii){.reference
            .internal}
    -   [When to Use sample vs.
        run](../examples/sample_vs_run.html){.reference .internal}
        -   [Introduction](../examples/sample_vs_run.html#introduction){.reference
            .internal}
        -   [Usage
            Guidelines](../examples/sample_vs_run.html#usage-guidelines){.reference
            .internal}
        -   [What Is Supported with [`sample`{.docutils .literal
            .notranslate}]{.pre}](../examples/sample_vs_run.html#what-is-supported-with-sample){.reference
            .internal}
        -   [What Is Not Supported with [`sample`{.docutils .literal
            .notranslate}]{.pre}](../examples/sample_vs_run.html#what-is-not-supported-with-sample){.reference
            .internal}
        -   [How to
            Migrate](../examples/sample_vs_run.html#how-to-migrate){.reference
            .internal}
            -   [Step 1: Add a return type to the
                kernel](../examples/sample_vs_run.html#step-1-add-a-return-type-to-the-kernel){.reference
                .internal}
            -   [Step 2: Replace [`sample`{.docutils .literal
                .notranslate}]{.pre} with [`run`{.docutils .literal
                .notranslate}]{.pre}](../examples/sample_vs_run.html#step-2-replace-sample-with-run){.reference
                .internal}
            -   [Step 3: Update result
                processing](../examples/sample_vs_run.html#step-3-update-result-processing){.reference
                .internal}
        -   [Migration
            Examples](../examples/sample_vs_run.html#migration-examples){.reference
            .internal}
            -   [Example 1: Simple conditional
                logic](../examples/sample_vs_run.html#example-1-simple-conditional-logic){.reference
                .internal}
            -   [Example 2: Returning multiple measurement
                results](../examples/sample_vs_run.html#example-2-returning-multiple-measurement-results){.reference
                .internal}
            -   [Example 3: Quantum
                teleportation](../examples/sample_vs_run.html#example-3-quantum-teleportation){.reference
                .internal}
        -   [Additional
            Notes](../examples/sample_vs_run.html#additional-notes){.reference
            .internal}
    -   [Dynamics
        Examples](../examples/dynamics_examples.html){.reference
        .internal}
        -   [Python Examples (Jupyter
            Notebooks)](../examples/dynamics_examples.html#python-examples-jupyter-notebooks){.reference
            .internal}
            -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
                Model)](../../examples/python/dynamics/dynamics_intro_1.html){.reference
                .internal}
            -   [Introduction to CUDA-Q Dynamics (Time Dependent
                Hamiltonians)](../../examples/python/dynamics/dynamics_intro_2.html){.reference
                .internal}
            -   [Superconducting
                Qubits](../../examples/python/dynamics/superconducting.html){.reference
                .internal}
            -   [Spin
                Qubits](../../examples/python/dynamics/spinqubits.html){.reference
                .internal}
            -   [Trapped Ion
                Qubits](../../examples/python/dynamics/iontrap.html){.reference
                .internal}
            -   [Control](../../examples/python/dynamics/control.html){.reference
                .internal}
        -   [C++
            Examples](../examples/dynamics_examples.html#c-examples){.reference
            .internal}
            -   [Introduction: Single Qubit
                Dynamics](../examples/dynamics_examples.html#introduction-single-qubit-dynamics){.reference
                .internal}
            -   [Introduction: Cavity QED (Jaynes-Cummings
                Model)](../examples/dynamics_examples.html#introduction-cavity-qed-jaynes-cummings-model){.reference
                .internal}
            -   [Superconducting Qubits: Cross-Resonance
                Gate](../examples/dynamics_examples.html#superconducting-qubits-cross-resonance-gate){.reference
                .internal}
            -   [Spin Qubits: Heisenberg Spin
                Chain](../examples/dynamics_examples.html#spin-qubits-heisenberg-spin-chain){.reference
                .internal}
            -   [Control: Driven
                Qubit](../examples/dynamics_examples.html#control-driven-qubit){.reference
                .internal}
            -   [State
                Batching](../examples/dynamics_examples.html#state-batching){.reference
                .internal}
            -   [Numerical
                Integrators](../examples/dynamics_examples.html#numerical-integrators){.reference
                .internal}
-   [Applications](../applications.html){.reference .internal}
    -   [Max-Cut with
        QAOA](../../applications/python/qaoa.html){.reference .internal}
    -   [Molecular docking via
        DC-QAOA](../../applications/python/digitized_counterdiabatic_qaoa.html){.reference
        .internal}
        -   [Setting up the Molecular Docking
            Problem](../../applications/python/digitized_counterdiabatic_qaoa.html#Setting-up-the-Molecular-Docking-Problem){.reference
            .internal}
        -   [CUDA-Q
            Implementation](../../applications/python/digitized_counterdiabatic_qaoa.html#CUDA-Q-Implementation){.reference
            .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](../../applications/python/krylov.html){.reference
        .internal}
        -   [Setup](../../applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](../../applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](../../applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Quantum-Selected Configuration Interaction
        (QSCI)](../../applications/python/qsci.html){.reference
        .internal}
        -   [0. Problem
            definition](../../applications/python/qsci.html#0.-Problem-definition){.reference
            .internal}
        -   [1. Prepare an Approximate Quantum
            State](../../applications/python/qsci.html#1.-Prepare-an-Approximate-Quantum-State){.reference
            .internal}
        -   [2 Quantum Sampling to Select
            Configuration](../../applications/python/qsci.html#2-Quantum-Sampling-to-Select-Configuration){.reference
            .internal}
        -   [3. Classical Diagonalization on the Selected
            Subspace](../../applications/python/qsci.html#3.-Classical-Diagonalization-on-the-Selected-Subspace){.reference
            .internal}
        -   [5. Compare
            results](../../applications/python/qsci.html#5.-Compare-results){.reference
            .internal}
        -   [Reference](../../applications/python/qsci.html#Reference){.reference
            .internal}
    -   [Bernstein-Vazirani
        Algorithm](../../applications/python/bernstein_vazirani.html){.reference
        .internal}
        -   [Classical
            case](../../applications/python/bernstein_vazirani.html#Classical-case){.reference
            .internal}
        -   [Quantum
            case](../../applications/python/bernstein_vazirani.html#Quantum-case){.reference
            .internal}
        -   [Implementing in
            CUDA-Q](../../applications/python/bernstein_vazirani.html#Implementing-in-CUDA-Q){.reference
            .internal}
    -   [Cost
        Minimization](../../applications/python/cost_minimization.html){.reference
        .internal}
    -   [Deutsch's
        Algorithm](../../applications/python/deutsch_algorithm.html){.reference
        .internal}
        -   [XOR [\\(\\oplus\\)]{.math .notranslate
            .nohighlight}](../../applications/python/deutsch_algorithm.html#XOR-\oplus){.reference
            .internal}
        -   [Quantum
            oracles](../../applications/python/deutsch_algorithm.html#Quantum-oracles){.reference
            .internal}
        -   [Phase
            oracle](../../applications/python/deutsch_algorithm.html#Phase-oracle){.reference
            .internal}
        -   [Quantum
            parallelism](../../applications/python/deutsch_algorithm.html#Quantum-parallelism){.reference
            .internal}
        -   [Deutsch's
            Algorithm:](../../applications/python/deutsch_algorithm.html#Deutsch's-Algorithm:){.reference
            .internal}
    -   [Divisive Clustering With Coresets Using
        CUDA-Q](../../applications/python/divisive_clustering_coresets.html){.reference
        .internal}
        -   [Data
            preprocessing](../../applications/python/divisive_clustering_coresets.html#Data-preprocessing){.reference
            .internal}
        -   [Quantum
            functions](../../applications/python/divisive_clustering_coresets.html#Quantum-functions){.reference
            .internal}
        -   [Divisive Clustering
            Function](../../applications/python/divisive_clustering_coresets.html#Divisive-Clustering-Function){.reference
            .internal}
        -   [QAOA
            Implementation](../../applications/python/divisive_clustering_coresets.html#QAOA-Implementation){.reference
            .internal}
        -   [Scaling simulations with
            CUDA-Q](../../applications/python/divisive_clustering_coresets.html#Scaling-simulations-with-CUDA-Q){.reference
            .internal}
    -   [Hybrid Quantum Neural
        Networks](../../applications/python/hybrid_quantum_neural_networks.html){.reference
        .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](../../applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](../../applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using [`Sample`{.docutils .literal .notranslate}]{.pre} to
            perform the Hadamard
            test](../../applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](../../applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](../../applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](../../applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](../../applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](../../applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](../../applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](../../applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](../../applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](../../applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](../../applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. [`Compute`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`overlap`{.docutils .literal
                .notranslate}]{.pre}](../../applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](../../applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](../../applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](../../applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](../../applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](../../applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](../../applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](../../applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](../../applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](../../applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum Fourier
        Transform](../../applications/python/quantum_fourier_transform.html){.reference
        .internal}
        -   [Quantum Fourier Transform
            revisited](../../applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform-revisited){.reference
            .internal}
    -   [Quantum
        Teleporation](../../applications/python/quantum_teleportation.html){.reference
        .internal}
        -   [Teleportation
            explained](../../applications/python/quantum_teleportation.html#Teleportation-explained){.reference
            .internal}
    -   [Quantum
        Volume](../../applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](../../applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](../../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](../../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](../../applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Compiling Unitaries Using Diffusion
        Models](../../applications/python/unitary_compilation_diffusion_models.html){.reference
        .internal}
        -   [Diffusion model
            pipeline](../../applications/python/unitary_compilation_diffusion_models.html#Diffusion-model-pipeline){.reference
            .internal}
        -   [Setup and load
            models](../../applications/python/unitary_compilation_diffusion_models.html#Setup-and-load-models){.reference
            .internal}
            -   [Load discrete
                model](../../applications/python/unitary_compilation_diffusion_models.html#Load-discrete-model){.reference
                .internal}
            -   [Load continuous
                model](../../applications/python/unitary_compilation_diffusion_models.html#Load-continuous-model){.reference
                .internal}
            -   [Create helper
                functions](../../applications/python/unitary_compilation_diffusion_models.html#Create-helper-functions){.reference
                .internal}
        -   [Unitary
            compilation](../../applications/python/unitary_compilation_diffusion_models.html#Unitary-compilation){.reference
            .internal}
            -   [Random
                unitary](../../applications/python/unitary_compilation_diffusion_models.html#Random-unitary){.reference
                .internal}
            -   [Discrete
                model](../../applications/python/unitary_compilation_diffusion_models.html#Discrete-model){.reference
                .internal}
            -   [Continuous
                model](../../applications/python/unitary_compilation_diffusion_models.html#Continuous-model){.reference
                .internal}
            -   [Quantum Fourier
                transform](../../applications/python/unitary_compilation_diffusion_models.html#Quantum-Fourier-transform){.reference
                .internal}
            -   [XXZ-Hamiltonian
                evolution](../../applications/python/unitary_compilation_diffusion_models.html#XXZ-Hamiltonian-evolution){.reference
                .internal}
        -   [Choosing the circuit you
            need](../../applications/python/unitary_compilation_diffusion_models.html#Choosing-the-circuit-you-need){.reference
            .internal}
    -   [VQE with gradients, active spaces, and gate
        fusion](../../applications/python/vqe_advanced.html){.reference
        .internal}
        -   [The Basics of
            VQE](../../applications/python/vqe_advanced.html#The-Basics-of-VQE){.reference
            .internal}
        -   [Installing/Loading Relevant
            Packages](../../applications/python/vqe_advanced.html#Installing/Loading-Relevant-Packages){.reference
            .internal}
        -   [Implementing VQE in
            CUDA-Q](../../applications/python/vqe_advanced.html#Implementing-VQE-in-CUDA-Q){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../../applications/python/vqe_advanced.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
        -   [Using an Active
            Space](../../applications/python/vqe_advanced.html#Using-an-Active-Space){.reference
            .internal}
        -   [Gate Fusion for Larger
            Circuits](../../applications/python/vqe_advanced.html#Gate-Fusion-for-Larger-Circuits){.reference
            .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](../../applications/python/afqmc.html){.reference
        .internal}
        -   [Hamiltonian preparation for
            VQE](../../applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](../../applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](../../applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](../../applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](../../applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](../../applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [ADAPT-QAOA
        algorithm](../../applications/python/adapt_qaoa.html){.reference
        .internal}
        -   [Simulation
            input:](../../applications/python/adapt_qaoa.html#Simulation-input:){.reference
            .internal}
        -   [The problem Hamiltonian [\\(H_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](../../applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A_j\\)]{.math .notranslate
            .nohighlight}:](../../applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H_C,A_j\]\\)]{.math .notranslate
            .nohighlight}:](../../applications/python/adapt_qaoa.html#The-commutator-%5BH_C,A_j%5D:){.reference
            .internal}
        -   [Beginning of ADAPT-QAOA
            iteration:](../../applications/python/adapt_qaoa.html#Beginning-of-ADAPT-QAOA-iteration:){.reference
            .internal}
    -   [ADAPT-VQE
        algorithm](../../applications/python/adapt_vqe.html){.reference
        .internal}
        -   [Classical
            pre-processing](../../applications/python/adapt_vqe.html#Classical-pre-processing){.reference
            .internal}
        -   [Jordan
            Wigner:](../../applications/python/adapt_vqe.html#Jordan-Wigner:){.reference
            .internal}
        -   [UCCSD operator
            pool](../../applications/python/adapt_vqe.html#UCCSD-operator-pool){.reference
            .internal}
            -   [Single
                excitation](../../applications/python/adapt_vqe.html#Single-excitation){.reference
                .internal}
            -   [Double
                excitation](../../applications/python/adapt_vqe.html#Double-excitation){.reference
                .internal}
        -   [Commutator \[[\\(H\\)]{.math .notranslate .nohighlight},
            [\\(A_i\\)]{.math .notranslate
            .nohighlight}\]](../../applications/python/adapt_vqe.html#Commutator-%5BH,-A_i%5D){.reference
            .internal}
        -   [Reference
            State:](../../applications/python/adapt_vqe.html#Reference-State:){.reference
            .internal}
        -   [Quantum
            kernels:](../../applications/python/adapt_vqe.html#Quantum-kernels:){.reference
            .internal}
        -   [Beginning of
            ADAPT-VQE:](../../applications/python/adapt_vqe.html#Beginning-of-ADAPT-VQE:){.reference
            .internal}
    -   [Quantum edge
        detection](../../applications/python/edge_detection.html){.reference
        .internal}
        -   [Image](../../applications/python/edge_detection.html#Image){.reference
            .internal}
        -   [Quantum Probability Image Encoding
            (QPIE):](../../applications/python/edge_detection.html#Quantum-Probability-Image-Encoding-(QPIE):){.reference
            .internal}
            -   [Below we show how to encode an image using QPIE in
                cudaq.](../../applications/python/edge_detection.html#Below-we-show-how-to-encode-an-image-using-QPIE-in-cudaq.){.reference
                .internal}
        -   [Flexible Representation of Quantum Images
            (FRQI):](../../applications/python/edge_detection.html#Flexible-Representation-of-Quantum-Images-(FRQI):){.reference
            .internal}
            -   [Building the FRQI
                State:](../../applications/python/edge_detection.html#Building-the-FRQI-State:){.reference
                .internal}
        -   [Quantum Hadamard Edge Detection
            (QHED)](../../applications/python/edge_detection.html#Quantum-Hadamard-Edge-Detection-(QHED)){.reference
            .internal}
            -   [Post-processing](../../applications/python/edge_detection.html#Post-processing){.reference
                .internal}
    -   [Factoring Integers With Shor's
        Algorithm](../../applications/python/shors.html){.reference
        .internal}
        -   [Shor's
            algorithm](../../applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](../../applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](../../applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](../../applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](../../applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](../../applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](../../applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](../../applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
            -   [(a) Generate the molecular Hamiltonian using Restricted
                Hartree Fock molecular
                orbitals](../../applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Restricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(b) Generate the molecular Hamiltonian using
                Unrestricted Hartree Fock molecular
                orbitals](../../applications/python/generate_fermionic_ham.html#(b)-Generate-the-molecular-Hamiltonian-using-Unrestricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(a) Generate the active space hamiltonian using RHF
                molecular
                orbitals.](../../applications/python/generate_fermionic_ham.html#(a)-Generate-the-active-space-hamiltonian-using-RHF-molecular-orbitals.){.reference
                .internal}
            -   [(b) Generate the active space Hamiltonian using the
                natural orbitals computed from MP2
                simulation](../../applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
                .internal}
            -   [(c) Generate the active space Hamiltonian computed from
                the CASSCF molecular
                orbitals](../../applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
                .internal}
            -   [(d) Generate the electronic Hamiltonian using
                ROHF](../../applications/python/generate_fermionic_ham.html#(d)-Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
            -   [(e) Generate electronic Hamiltonian using
                UHF](../../applications/python/generate_fermionic_ham.html#(e)-Generate-electronic-Hamiltonian-using-UHF){.reference
                .internal}
    -   [Grover's
        Algorithm](../../applications/python/grovers.html){.reference
        .internal}
        -   [Overview](../../applications/python/grovers.html#Overview){.reference
            .internal}
        -   [Problem](../../applications/python/grovers.html#Problem){.reference
            .internal}
        -   [Structure of Grover's
            Algorithm](../../applications/python/grovers.html#Structure-of-Grover's-Algorithm){.reference
            .internal}
            -   [Step 1:
                Preparation](../../applications/python/grovers.html#Step-1:-Preparation){.reference
                .internal}
            -   [Good and Bad
                States](../../applications/python/grovers.html#Good-and-Bad-States){.reference
                .internal}
            -   [Step 2: Oracle
                application](../../applications/python/grovers.html#Step-2:-Oracle-application){.reference
                .internal}
            -   [Step 3: Amplitude
                amplification](../../applications/python/grovers.html#Step-3:-Amplitude-amplification){.reference
                .internal}
            -   [Steps 4 and 5: Iteration and
                measurement](../../applications/python/grovers.html#Steps-4-and-5:-Iteration-and-measurement){.reference
                .internal}
    -   [Quantum
        PageRank](../../applications/python/quantum_pagerank.html){.reference
        .internal}
        -   [Problem
            Definition](../../applications/python/quantum_pagerank.html#Problem-Definition){.reference
            .internal}
        -   [Simulating Quantum PageRank by CUDA-Q
            dynamics](../../applications/python/quantum_pagerank.html#Simulating-Quantum-PageRank-by-CUDA-Q-dynamics){.reference
            .internal}
        -   [Breakdown of
            Terms](../../applications/python/quantum_pagerank.html#Breakdown-of-Terms){.reference
            .internal}
    -   [The UCCSD Wavefunction
        ansatz](../../applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](../../applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](../../applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](../../applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](../../applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](../../applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](../../applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [QM/MM simulation: VQE within a Polarizable Embedded
        Framework.](../../applications/python/qm_mm_pe.html){.reference
        .internal}
        -   [Key
            concepts:](../../applications/python/qm_mm_pe.html#Key-concepts:){.reference
            .internal}
        -   [PE-VQE-SCF Algorithm
            Steps](../../applications/python/qm_mm_pe.html#PE-VQE-SCF-Algorithm-Steps){.reference
            .internal}
            -   [Step 1: Initialize (Classical
                pre-processing)](../../applications/python/qm_mm_pe.html#Step-1:-Initialize-(Classical-pre-processing)){.reference
                .internal}
            -   [Step 2: Build the
                Hamiltonian](../../applications/python/qm_mm_pe.html#Step-2:-Build-the-Hamiltonian){.reference
                .internal}
            -   [Step 3: Run
                VQE](../../applications/python/qm_mm_pe.html#Step-3:-Run-VQE){.reference
                .internal}
            -   [Step 4: Update
                Environment](../../applications/python/qm_mm_pe.html#Step-4:-Update-Environment){.reference
                .internal}
            -   [Step 5: Self-Consistency
                Loop](../../applications/python/qm_mm_pe.html#Step-5:-Self-Consistency-Loop){.reference
                .internal}
            -   [Requirments:](../../applications/python/qm_mm_pe.html#Requirments:){.reference
                .internal}
            -   [Example 1: LiH with 2 water
                molecules.](../../applications/python/qm_mm_pe.html#Example-1:-LiH-with-2-water-molecules.){.reference
                .internal}
            -   [VQE, update environment, and scf
                loop.](../../applications/python/qm_mm_pe.html#VQE,-update-environment,-and-scf-loop.){.reference
                .internal}
            -   [Example 2: NH3 with 46 water molecule using active
                space.](../../applications/python/qm_mm_pe.html#Example-2:-NH3-with-46-water-molecule-using-active-space.){.reference
                .internal}
    -   [Sample-Based Krylov Quantum Diagonalization
        (SKQD)](../../applications/python/skqd.html){.reference
        .internal}
        -   [Why
            SKQD?](../../applications/python/skqd.html#Why-SKQD?){.reference
            .internal}
        -   [Understanding Krylov
            Subspaces](../../applications/python/skqd.html#Understanding-Krylov-Subspaces){.reference
            .internal}
            -   [What is a Krylov
                Subspace?](../../applications/python/skqd.html#What-is-a-Krylov-Subspace?){.reference
                .internal}
            -   [The SKQD
                Algorithm](../../applications/python/skqd.html#The-SKQD-Algorithm){.reference
                .internal}
        -   [Problem Setup: 22-Qubit Heisenberg
            Model](../../applications/python/skqd.html#Problem-Setup:-22-Qubit-Heisenberg-Model){.reference
            .internal}
        -   [Krylov State Generation via Repeated
            Evolution](../../applications/python/skqd.html#Krylov-State-Generation-via-Repeated-Evolution){.reference
            .internal}
        -   [Quantum Measurements and
            Sampling](../../applications/python/skqd.html#Quantum-Measurements-and-Sampling){.reference
            .internal}
            -   [The Sampling
                Process](../../applications/python/skqd.html#The-Sampling-Process){.reference
                .internal}
        -   [Classical Post-Processing and
            Diagonalization](../../applications/python/skqd.html#Classical-Post-Processing-and-Diagonalization){.reference
            .internal}
            -   [Matrix Construction
                Details](../../applications/python/skqd.html#Matrix-Construction-Details){.reference
                .internal}
            -   [Approach 1: GPU-Vectorized CSR Sparse
                Matrix](../../applications/python/skqd.html#Approach-1:-GPU-Vectorized-CSR-Sparse-Matrix){.reference
                .internal}
            -   [Approach 2: Matrix-Free Lanczos via
                [`distributed_eigsh`{.docutils .literal
                .notranslate}]{.pre}](../../applications/python/skqd.html#Approach-2:-Matrix-Free-Lanczos-via-distributed_eigsh){.reference
                .internal}
        -   [Results Analysis and
            Convergence](../../applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](../../applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
        -   [Postprocessing Acceleration: CSR matrix approach, single
            GPU vs
            CPU](../../applications/python/skqd.html#Postprocessing-Acceleration:-CSR-matrix-approach,-single-GPU-vs-CPU){.reference
            .internal}
        -   [Postprocessing Scale-Up and Scale-Out: Linear Operator
            Approach, Multi-GPU
            Multi-Node](../../applications/python/skqd.html#Postprocessing-Scale-Up-and-Scale-Out:-Linear-Operator-Approach,-Multi-GPU-Multi-Node){.reference
            .internal}
            -   [Saving Hamiltonian
                Data](../../applications/python/skqd.html#Saving-Hamiltonian-Data){.reference
                .internal}
            -   [Running the Distributed
                Solver](../../applications/python/skqd.html#Running-the-Distributed-Solver){.reference
                .internal}
        -   [Summary](../../applications/python/skqd.html#Summary){.reference
            .internal}
    -   [Entanglement Accelerates Quantum
        Simulation](../../applications/python/entanglement_acc_hamiltonian_simulation.html){.reference
        .internal}
        -   [2. Model
            Definition](../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.-Model-Definition){.reference
            .internal}
            -   [2.1 Initial product
                state](../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.1-Initial-product-state){.reference
                .internal}
            -   [2.2 QIMF
                Hamiltonian](../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.2-QIMF-Hamiltonian){.reference
                .internal}
            -   [2.3 First-Order Trotter Formula
                (PF1)](../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.3-First-Order-Trotter-Formula-(PF1)){.reference
                .internal}
            -   [2.4 PF1 step for the QIMF
                partition](../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.4-PF1-step-for-the-QIMF-partition){.reference
                .internal}
            -   [2.5 Hamiltonian
                helpers](../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.5-Hamiltonian-helpers){.reference
                .internal}
        -   [3. Entanglement
            metrics](../../applications/python/entanglement_acc_hamiltonian_simulation.html#3.-Entanglement-metrics){.reference
            .internal}
        -   [4. Simulation
            workflow](../../applications/python/entanglement_acc_hamiltonian_simulation.html#4.-Simulation-workflow){.reference
            .internal}
            -   [4.1 Single-step Trotter
                error](../../applications/python/entanglement_acc_hamiltonian_simulation.html#4.1-Single-step-Trotter-error){.reference
                .internal}
            -   [4.2 Dual trajectory
                update](../../applications/python/entanglement_acc_hamiltonian_simulation.html#4.2-Dual-trajectory-update){.reference
                .internal}
        -   [5. Reproducing the paper's Figure
            1a](../../applications/python/entanglement_acc_hamiltonian_simulation.html#5.-Reproducing-the-paper’s-Figure-1a){.reference
            .internal}
            -   [5.1 Visualising the joint
                behaviour](../../applications/python/entanglement_acc_hamiltonian_simulation.html#5.1-Visualising-the-joint-behaviour){.reference
                .internal}
            -   [5.2 Interpreting the
                result](../../applications/python/entanglement_acc_hamiltonian_simulation.html#5.2-Interpreting-the-result){.reference
                .internal}
        -   [6. References and further
            reading](../../applications/python/entanglement_acc_hamiltonian_simulation.html#6.-References-and-further-reading){.reference
            .internal}
    -   [Pre-Trajectory Sampling with Batch Execution
        (PTSBE)](../../applications/python/ptsbe.html){.reference
        .internal}
        -   [Set up the
            environment](../../applications/python/ptsbe.html#Set-up-the-environment){.reference
            .internal}
        -   [Define the circuit and noise
            model](../../applications/python/ptsbe.html#Define-the-circuit-and-noise-model){.reference
            .internal}
            -   [Inline noise with [`apply_noise`{.docutils .literal
                .notranslate}]{.pre}](../../applications/python/ptsbe.html#Inline-noise-with-apply_noise){.reference
                .internal}
        -   [Run PTSBE
            sampling](../../applications/python/ptsbe.html#Run-PTSBE-sampling){.reference
            .internal}
            -   [Larger circuit for execution
                data](../../applications/python/ptsbe.html#Larger-circuit-for-execution-data){.reference
                .internal}
        -   [Inspecting trajectories with execution
            data](../../applications/python/ptsbe.html#Inspecting-trajectories-with-execution-data){.reference
            .internal}
        -   [Performance of PTSBE vs standard noisy
            sampling](../../applications/python/ptsbe.html#Performance-of-PTSBE-vs-standard-noisy-sampling){.reference
            .internal}
-   [Backends](../backends/backends.html){.reference .internal}
    -   [Circuit Simulation](../backends/simulators.html){.reference
        .internal}
        -   [State Vector
            Simulators](../backends/sims/svsims.html){.reference
            .internal}
            -   [CPU](../backends/sims/svsims.html#cpu){.reference
                .internal}
            -   [Single-GPU](../backends/sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](../backends/sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network
            Simulators](../backends/sims/tnsims.html){.reference
            .internal}
            -   [Multi-GPU
                multi-node](../backends/sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](../backends/sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](../backends/sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU
            Simulators](../backends/sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](../backends/sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](../backends/sims/mqpusims.html#multi-qpu-other-backends){.reference
                .internal}
        -   [Noisy Simulators](../backends/sims/noisy.html){.reference
            .internal}
            -   [Trajectory Noisy
                Simulation](../backends/sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](../backends/sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](../backends/sims/noisy.html#stim){.reference
                .internal}
        -   [Photonics
            Simulators](../backends/sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](../backends/sims/photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware (QPUs)](../backends/hardware.html){.reference
        .internal}
        -   [Ion Trap
            QPUs](../backends/hardware/iontrap.html){.reference
            .internal}
            -   [IonQ](../backends/hardware/iontrap.html#ionq){.reference
                .internal}
            -   [Quantinuum](../backends/hardware/iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting
            QPUs](../backends/hardware/superconducting.html){.reference
            .internal}
            -   [Anyon Technologies/Anyon
                Computing](../backends/hardware/superconducting.html#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](../backends/hardware/superconducting.html#iqm){.reference
                .internal}
            -   [OQC](../backends/hardware/superconducting.html#oqc){.reference
                .internal}
            -   [Quantum Circuits,
                Inc.](../backends/hardware/superconducting.html#quantum-circuits-inc){.reference
                .internal}
            -   [TII](../backends/hardware/superconducting.html#tii){.reference
                .internal}
        -   [Neutral Atom
            QPUs](../backends/hardware/neutralatom.html){.reference
            .internal}
            -   [Infleqtion](../backends/hardware/neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](../backends/hardware/neutralatom.html#pasqal){.reference
                .internal}
            -   [QuEra
                Computing](../backends/hardware/neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic
            QPUs](../backends/hardware/photonic.html){.reference
            .internal}
            -   [ORCA
                Computing](../backends/hardware/photonic.html#orca-computing){.reference
                .internal}
        -   [Quantum Control
            Systems](../backends/hardware/qcontrol.html){.reference
            .internal}
            -   [Quantum
                Machines](../backends/hardware/qcontrol.html#quantum-machines){.reference
                .internal}
    -   [Dynamics
        Simulation](../backends/dynamics_backends.html){.reference
        .internal}
    -   [Cloud](../backends/cloud.html){.reference .internal}
        -   [Amazon Braket
            (braket)](../backends/cloud/braket.html){.reference
            .internal}
            -   [Setting
                Credentials](../backends/cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submitting](../backends/cloud/braket.html#submitting){.reference
                .internal}
        -   [Scaleway QaaS
            (scaleway)](../backends/cloud/scaleway.html){.reference
            .internal}
            -   [Setting
                Credentials](../backends/cloud/scaleway.html#setting-credentials){.reference
                .internal}
            -   [Submitting](../backends/cloud/scaleway.html#submitting){.reference
                .internal}
            -   [Manage your QPU
                session](../backends/cloud/scaleway.html#manage-your-qpu-session){.reference
                .internal}
-   [Dynamics](../dynamics.html){.reference .internal}
    -   [Quick Start](../dynamics.html#quick-start){.reference
        .internal}
    -   [Operator](../dynamics.html#operator){.reference .internal}
    -   [Time-Dependent
        Dynamics](../dynamics.html#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](../dynamics.html#super-operator-representation){.reference
        .internal}
    -   [Numerical
        Integrators](../dynamics.html#numerical-integrators){.reference
        .internal}
    -   [Batch simulation](../dynamics.html#batch-simulation){.reference
        .internal}
    -   [Multi-GPU Multi-Node
        Execution](../dynamics.html#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](../dynamics.html#examples){.reference .internal}
-   [Realtime](../realtime.html){.reference .internal}
    -   [Installation](installation.html){.reference .internal}
        -   [Prerequisites](installation.html#prerequisites){.reference
            .internal}
        -   [Setup](installation.html#setup){.reference .internal}
        -   [Latency
            Measurement](installation.html#latency-measurement){.reference
            .internal}
    -   [Host API](#){.current .reference .internal}
        -   [What is HSB?](#what-is-hsb){.reference .internal}
        -   [Transport Mechanisms](#transport-mechanisms){.reference
            .internal}
            -   [Supported Transport
                Options](#supported-transport-options){.reference
                .internal}
        -   [The 3-Kernel Architecture (HSB Example)
            {#three-kernel-architecture}](#the-3-kernel-architecture-hsb-example-three-kernel-architecture){.reference
            .internal}
            -   [Data Flow Summary](#data-flow-summary){.reference
                .internal}
            -   [Why 3 Kernels?](#why-3-kernels){.reference .internal}
        -   [Unified Dispatch Mode](#unified-dispatch-mode){.reference
            .internal}
            -   [Architecture](#architecture){.reference .internal}
            -   [Transport-Agnostic
                Design](#transport-agnostic-design){.reference
                .internal}
            -   [When to Use Which
                Mode](#when-to-use-which-mode){.reference .internal}
            -   [Host API Extensions](#host-api-extensions){.reference
                .internal}
            -   [Wiring Example (Unified Mode with
                HSB)](#wiring-example-unified-mode-with-hsb){.reference
                .internal}
        -   [What This API Does (In One
            Paragraph)](#what-this-api-does-in-one-paragraph){.reference
            .internal}
        -   [Scope](#scope){.reference .internal}
        -   [Terms and Components](#terms-and-components){.reference
            .internal}
        -   [Schema Data Structures](#schema-data-structures){.reference
            .internal}
            -   [Type Descriptors](#type-descriptors){.reference
                .internal}
            -   [Handler Schema](#handler-schema){.reference .internal}
        -   [RPC Messaging Protocol](#rpc-messaging-protocol){.reference
            .internal}
        -   [Host API Overview](#host-api-overview){.reference
            .internal}
        -   [Manager and Dispatcher
            Topology](#manager-and-dispatcher-topology){.reference
            .internal}
        -   [Host API Functions](#host-api-functions){.reference
            .internal}
            -   [Occupancy Query and Eager Module
                Loading](#occupancy-query-and-eager-module-loading){.reference
                .internal}
            -   [Graph-Based Dispatch
                Functions](#graph-based-dispatch-functions){.reference
                .internal}
            -   [Kernel Launch Helper
                Functions](#kernel-launch-helper-functions){.reference
                .internal}
        -   [Memory Layout and Ring Buffer
            Wiring](#memory-layout-and-ring-buffer-wiring){.reference
            .internal}
        -   [Step-by-Step: Wiring the Host API
            (Minimal)](#step-by-step-wiring-the-host-api-minimal){.reference
            .internal}
        -   [Device Handler and Function
            ID](#device-handler-and-function-id){.reference .internal}
            -   [Multi-Argument Handler
                Example](#multi-argument-handler-example){.reference
                .internal}
        -   [CUDA Graph Dispatch
            Mode](#cuda-graph-dispatch-mode){.reference .internal}
            -   [Requirements](#requirements){.reference .internal}
            -   [Graph-Based Dispatch
                API](#graph-based-dispatch-api){.reference .internal}
            -   [Graph Handler Setup
                Example](#graph-handler-setup-example){.reference
                .internal}
            -   [Graph Capture and
                Instantiation](#graph-capture-and-instantiation){.reference
                .internal}
            -   [When to Use Graph
                Dispatch](#when-to-use-graph-dispatch){.reference
                .internal}
            -   [Graph vs Device Call
                Dispatch](#graph-vs-device-call-dispatch){.reference
                .internal}
        -   [Building and Sending an RPC
            Message](#building-and-sending-an-rpc-message){.reference
            .internal}
        -   [Reading the Response](#reading-the-response){.reference
            .internal}
        -   [Schema-Driven Argument
            Parsing](#schema-driven-argument-parsing){.reference
            .internal}
        -   [HSB 3-Kernel Workflow
            (Primary)](#hsb-3-kernel-workflow-primary){.reference
            .internal}
        -   [NIC-Free Testing (No HSB / No
            ConnectX-7)](#nic-free-testing-no-hsb-no-connectx-7){.reference
            .internal}
        -   [Troubleshooting](#troubleshooting){.reference .internal}
    -   [Messaging Protocol](protocol.html){.reference .internal}
        -   [Scope](protocol.html#scope){.reference .internal}
        -   [RPC Header /
            Response](protocol.html#rpc-header-response){.reference
            .internal}
        -   [Request ID
            Semantics](protocol.html#request-id-semantics){.reference
            .internal}
        -   [[`PTP`{.docutils .literal .notranslate}]{.pre} Timestamp
            Semantics](protocol.html#ptp-timestamp-semantics){.reference
            .internal}
        -   [Function ID
            Semantics](protocol.html#function-id-semantics){.reference
            .internal}
        -   [Schema and Payload
            Interpretation](protocol.html#schema-and-payload-interpretation){.reference
            .internal}
            -   [Type System](protocol.html#type-system){.reference
                .internal}
        -   [Payload
            Encoding](protocol.html#payload-encoding){.reference
            .internal}
            -   [Single-Argument
                Payloads](protocol.html#single-argument-payloads){.reference
                .internal}
            -   [Multi-Argument
                Payloads](protocol.html#multi-argument-payloads){.reference
                .internal}
            -   [Size
                Constraints](protocol.html#size-constraints){.reference
                .internal}
            -   [Encoding
                Examples](protocol.html#encoding-examples){.reference
                .internal}
            -   [Bit-Packed Data
                Encoding](protocol.html#bit-packed-data-encoding){.reference
                .internal}
            -   [Multi-Bit Measurement
                Encoding](protocol.html#multi-bit-measurement-encoding){.reference
                .internal}
        -   [Response
            Encoding](protocol.html#response-encoding){.reference
            .internal}
            -   [Single-Result
                Response](protocol.html#single-result-response){.reference
                .internal}
            -   [Multi-Result
                Response](protocol.html#multi-result-response){.reference
                .internal}
            -   [Status Codes](protocol.html#status-codes){.reference
                .internal}
        -   [QEC-Specific Usage
            Example](protocol.html#qec-specific-usage-example){.reference
            .internal}
            -   [QEC
                Terminology](protocol.html#qec-terminology){.reference
                .internal}
            -   [QEC Decoder
                Handler](protocol.html#qec-decoder-handler){.reference
                .internal}
            -   [Decoding
                Rounds](protocol.html#decoding-rounds){.reference
                .internal}
-   [CUDA-QX](../cudaqx/cudaqx.html){.reference .internal}
    -   [CUDA-Q
        Solvers](../cudaqx/cudaqx.html#cuda-q-solvers){.reference
        .internal}
    -   [CUDA-Q QEC](../cudaqx/cudaqx.html#cuda-q-qec){.reference
        .internal}
-   [Installation](../install/install.html){.reference .internal}
    -   [Local
        Installation](../install/local_installation.html){.reference
        .internal}
        -   [Introduction](../install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](../install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](../install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](../install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](../install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](../install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](../install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](../install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](../install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](../install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](../install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](../install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](../install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](../install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](../install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](../install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](../install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](../install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](../install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](../install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](../install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](../install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](../install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
        -   [Next
            Steps](../install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](../install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](../install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](../install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](../install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](../install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](../install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](../install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](../install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](../install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](../install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](../install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](../integration/integration.html){.reference .internal}
    -   [Downstream CMake
        Integration](../integration/cmake_app.html){.reference
        .internal}
    -   [Combining CUDA with
        CUDA-Q](../integration/cuda_gpu.html){.reference .internal}
    -   [Integrating with Third-Party
        Libraries](../integration/libraries.html){.reference .internal}
        -   [Calling a CUDA-Q library from
            C++](../integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](../integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](../integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](../extending/extending.html){.reference .internal}
    -   [Add a new Hardware
        Backend](../extending/backend.html){.reference .internal}
        -   [Overview](../extending/backend.html#overview){.reference
            .internal}
        -   [Server Helper
            Implementation](../extending/backend.html#server-helper-implementation){.reference
            .internal}
            -   [Directory
                Structure](../extending/backend.html#directory-structure){.reference
                .internal}
            -   [Server Helper
                Class](../extending/backend.html#server-helper-class){.reference
                .internal}
            -   [[`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](../extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](../extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent [`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](../extending/backend.html#update-parent-cmakelists-txt){.reference
                .internal}
        -   [Testing](../extending/backend.html#testing){.reference
            .internal}
            -   [Unit
                Tests](../extending/backend.html#unit-tests){.reference
                .internal}
            -   [Mock
                Server](../extending/backend.html#mock-server){.reference
                .internal}
            -   [Python
                Tests](../extending/backend.html#python-tests){.reference
                .internal}
            -   [Integration
                Tests](../extending/backend.html#integration-tests){.reference
                .internal}
        -   [Documentation](../extending/backend.html#documentation){.reference
            .internal}
        -   [Example
            Usage](../extending/backend.html#example-usage){.reference
            .internal}
        -   [Code
            Review](../extending/backend.html#code-review){.reference
            .internal}
        -   [Maintaining a
            Backend](../extending/backend.html#maintaining-a-backend){.reference
            .internal}
        -   [Conclusion](../extending/backend.html#conclusion){.reference
            .internal}
    -   [Create a new NVQIR
        Simulator](../extending/nvqir_simulator.html){.reference
        .internal}
        -   [[`CircuitSimulator`{.code .docutils .literal
            .notranslate}]{.pre}](../extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](../extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
    -   [Working with CUDA-Q IR](../extending/cudaq_ir.html){.reference
        .internal}
    -   [Create an MLIR Pass for
        CUDA-Q](../extending/mlir_pass.html){.reference .internal}
-   [Specifications](../../specification/index.html){.reference
    .internal}
    -   [Language
        Specification](../../specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](../../specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](../../specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](../../specification/cudaq/types.html){.reference
            .internal}
            -   [3.1. [`cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}]{.pre}](../../specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. [`cudaq::qubit`{.code .docutils .literal
                .notranslate}]{.pre}](../../specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](../../specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](../../specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. [`cudaq::spin_op`{.code .docutils .literal
                .notranslate}]{.pre}](../../specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](../../specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on [`cudaq::qubit`{.code .docutils
                .literal
                .notranslate}]{.pre}](../../specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](../../specification/cudaq/kernels.html){.reference
            .internal}
        -   [7. Sub-circuit
            Synthesis](../../specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](../../specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](../../specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](../../specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](../../specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11.
            Platform](../../specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](../../specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. [`cudaq::sample`{.code .docutils .literal
                .notranslate}]{.pre}](../../specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. [`cudaq::run`{.code .docutils .literal
                .notranslate}]{.pre}](../../specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. [`cudaq::observe`{.code .docutils .literal
                .notranslate}]{.pre}](../../specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. [`cudaq::optimizer`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../../specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. [`cudaq::gradient`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../../specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](../../specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](../../specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](../../specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](../../specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](../../specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](../../specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](../../specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](../../specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](../../specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](../../specification/quake-dialect.html#motivation){.reference
            .internal}
-   [API Reference](../../api/api.html){.reference .internal}
    -   [C++ API](../../api/languages/cpp_api.html){.reference
        .internal}
        -   [Operators](../../api/languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](../../api/languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](../../api/languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](../../api/languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](../../api/languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](../../api/languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Platform](../../api/languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](../../api/languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](../../api/languages/cpp_api.html#namespaces){.reference
            .internal}
        -   [PTSBE](../../api/languages/cpp_api.html#ptsbe){.reference
            .internal}
            -   [Sampling
                Functions](../../api/languages/cpp_api.html#sampling-functions){.reference
                .internal}
            -   [Options](../../api/languages/cpp_api.html#options){.reference
                .internal}
            -   [Result
                Type](../../api/languages/cpp_api.html#result-type){.reference
                .internal}
            -   [Trajectory Sampling
                Strategies](../../api/languages/cpp_api.html#trajectory-sampling-strategies){.reference
                .internal}
            -   [Shot Allocation
                Strategy](../../api/languages/cpp_api.html#shot-allocation-strategy){.reference
                .internal}
            -   [Execution
                Data](../../api/languages/cpp_api.html#execution-data){.reference
                .internal}
            -   [Trajectory and Selection
                Types](../../api/languages/cpp_api.html#trajectory-and-selection-types){.reference
                .internal}
    -   [Python API](../../api/languages/python_api.html){.reference
        .internal}
        -   [Program
            Construction](../../api/languages/python_api.html#program-construction){.reference
            .internal}
            -   [[`make_kernel()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [[`PyKernel`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [[`Kernel`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [[`PyKernelDecorator`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [[`kernel()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](../../api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [[`sample_async()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [[`run()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [[`run_async()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [[`observe()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [[`observe_async()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [[`get_state()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [[`get_state_async()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [[`vqe()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [[`draw()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [[`translate()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [[`estimate_resources()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
        -   [Backend
            Configuration](../../api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [[`parse_args()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.parse_args){.reference
                .internal}
            -   [[`has_target()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [[`get_target()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [[`get_targets()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [[`set_target()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [[`reset_target()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [[`set_noise()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [[`unset_noise()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [[`register_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [[`unregister_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [[`cudaq.apply_noise()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [[`initialize_cudaq()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [[`num_available_gpus()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [[`set_random_seed()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](../../api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [[`evolve()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [[`evolve_async()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [[`Schedule`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [[`BaseIntegrator`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [[`InitialState`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [[`InitialStateType`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [[`IntermediateResultSave`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](../../api/languages/python_api.html#operators){.reference
            .internal}
            -   [[`OperatorSum`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [[`ProductOperator`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [[`ElementaryOperator`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [[`ScalarOperator`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [[`RydbergHamiltonian`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [[`SuperOperator`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [[`operators.define()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [[`operators.instantiate()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](../../api/languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](../../api/languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](../../api/languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](../../api/languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](../../api/languages/python_api.html#data-types){.reference
            .internal}
            -   [[`SimulationPrecision`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [[`Target`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [[`State`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [[`Tensor`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [[`QuakeValue`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [[`qubit`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [[`qreg`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [[`qvector`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [[`ComplexMatrix`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [[`SampleResult`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [[`AsyncSampleResult`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [[`ObserveResult`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [[`AsyncObserveResult`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [[`AsyncStateResult`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [[`OptimizationResult`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [[`EvolveResult`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [[`AsyncEvolveResult`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [[`Resources`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.Resources){.reference
                .internal}
            -   [Optimizers](../../api/languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](../../api/languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](../../api/languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](../../api/languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [[`initialize()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [[`rank()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [[`num_ranks()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [[`all_gather()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [[`broadcast()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [[`is_initialized()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [[`finalize()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](../../api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
        -   [PTSBE
            Submodule](../../api/languages/python_api.html#ptsbe-submodule){.reference
            .internal}
            -   [Sampling
                Functions](../../api/languages/python_api.html#sampling-functions){.reference
                .internal}
            -   [Result
                Type](../../api/languages/python_api.html#result-type){.reference
                .internal}
            -   [Trajectory Sampling
                Strategies](../../api/languages/python_api.html#trajectory-sampling-strategies){.reference
                .internal}
            -   [Shot Allocation
                Strategy](../../api/languages/python_api.html#shot-allocation-strategy){.reference
                .internal}
            -   [Execution
                Data](../../api/languages/python_api.html#execution-data){.reference
                .internal}
            -   [Trajectory and Selection
                Types](../../api/languages/python_api.html#trajectory-and-selection-types){.reference
                .internal}
    -   [Quantum Operations](../../api/default_ops.html){.reference
        .internal}
        -   [Unitary Operations on
            Qubits](../../api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [[`x`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#x){.reference
                .internal}
            -   [[`y`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#y){.reference
                .internal}
            -   [[`z`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#z){.reference
                .internal}
            -   [[`h`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#h){.reference
                .internal}
            -   [[`r1`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#r1){.reference
                .internal}
            -   [[`rx`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#rx){.reference
                .internal}
            -   [[`ry`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#ry){.reference
                .internal}
            -   [[`rz`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#rz){.reference
                .internal}
            -   [[`s`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#s){.reference
                .internal}
            -   [[`t`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#t){.reference
                .internal}
            -   [[`swap`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#swap){.reference
                .internal}
            -   [[`u3`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#u3){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](../../api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](../../api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#mz){.reference
                .internal}
            -   [[`mx`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#mx){.reference
                .internal}
            -   [[`my`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](../../api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](../../api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [[`create`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#create){.reference
                .internal}
            -   [[`annihilate`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#annihilate){.reference
                .internal}
            -   [[`phase_shift`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#phase-shift){.reference
                .internal}
            -   [[`beam_splitter`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](../../api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](../../versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](../../index.html)

::: wy-nav-content
::: rst-content
::: {role="navigation" aria-label="Page navigation"}
-   [](../../index.html){.icon .icon-home aria-label="Home"}
-   [CUDA-Q Realtime](../realtime.html)
-   CUDA-Q Realtime Host API
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](installation.html "Installation"){.btn .btn-neutral
.float-left accesskey="p"} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](protocol.html "CUDA-Q Realtime Messaging Protocol"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#cuda-q-realtime-host-api .section}
# CUDA-Q Realtime Host API[¶](#cuda-q-realtime-host-api "Permalink to this heading"){.headerlink}

This document explains the C host API for realtime dispatch, the RPC
wire protocol, and complete wiring examples. It is written for external
partners integrating CUDA-QX decoders with their own transport
mechanisms. The API and protocol are **transport-agnostic** and support
multiple data transport options, including NVIDIA HSB (RDMA via ConnectX
NIC's), [`libibverbs`{.docutils .literal .notranslate}]{.pre}, and
proprietary transport layers. Handlers can execute on GPU (via CUDA
kernels) or CPU (via host threads). Examples in this document use HSB's
3-kernel workflow (RX kernel/dispatch/TX kernel) for illustration, but
the same principles apply to other transport mechanisms.

::: {#what-is-hsb .section}
## What is HSB?[¶](#what-is-hsb "Permalink to this heading"){.headerlink}

**HSB** is NVIDIA's low-latency sensor bridge framework that enables
direct GPU memory access from external devices (FPGAs, sensors) over
Ethernet using RDMA (Remote Direct Memory Access) via ConnectX NIC's. In
the context of quantum error correction, HSB is one example of a
transport mechanism that connects the quantum control system (typically
an FPGA) to GPU-based decoders.

**Repository**: [[`nvidia-holoscan`{.docutils .literal
.notranslate}]{.pre}/[`holoscan-sensor-bridge`{.docutils .literal
.notranslate}]{.pre} ([`nvqlink`{.docutils .literal .notranslate}]{.pre}
branch)](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/tree/nvqlink){.reference
.external}

HSB handles:

-   **RX (Receive)**: RX kernel receives data from the FPGA directly
    into GPU memory via RDMA

-   **TX (Transmit)**: TX kernel sends results back to the FPGA via RDMA

-   **RDMA transport**: Zero-copy data movement using ConnectX-7 NIC's
    with GPUDirect support

The CUDA-Q Realtime Host API provides the **middle component** (dispatch
kernel or thread) that sits between the transport's RX and TX
components, executing the actual decoder logic.
:::

::: {#transport-mechanisms .section}
## Transport Mechanisms[¶](#transport-mechanisms "Permalink to this heading"){.headerlink}

The realtime dispatch API is designed to work with multiple transport
mechanisms that move data between the quantum control system (FPGA) and
the decoder. The transport mechanism handles getting RPC messages into
RX ring buffer slots and sending responses from TX ring buffer slots
back to the FPGA.

::: {#supported-transport-options .section}
### Supported Transport Options[¶](#supported-transport-options "Permalink to this heading"){.headerlink}

**HSB (GPU-based with GPUDirect)**:

-   Uses ConnectX-7 NIC's with RDMA for zero-copy data movement

-   RX and TX are persistent GPU kernels that directly access GPU memory

-   Requires GPUDirect support

-   Lowest latency option for GPU-based decoders

**[`libibverbs`{.docutils .literal .notranslate}]{.pre} (CPU-based)**:

-   Standard InfiniBand Verbs API for RDMA on the CPU

-   RX and TX are host threads that poll CPU-accessible memory

-   Works with CPU-based dispatchers

-   Ring buffers reside in host memory ([`cudaHostAlloc`{.docutils
    .literal .notranslate}]{.pre} or regular [`malloc`{.docutils
    .literal .notranslate}]{.pre})

**Proprietary Transport Mechanisms**:

-   Custom implementations with or without GPUDirect support

-   May use different networking technologies or memory transfer methods

-   Must implement the ring buffer + flag protocol defined in this
    document

-   Can target either GPU (with suitable memory access) or CPU execution

The key requirement is that the transport mechanism implements the ring
buffer slot + flag protocol: writing RPC messages to RX slots and
setting [`rx_flags`{.docutils .literal .notranslate}]{.pre}, then
reading TX slots after [`tx_flags`{.docutils .literal
.notranslate}]{.pre} are set.
:::
:::

::: {#the-3-kernel-architecture-hsb-example-three-kernel-architecture .section}
## The 3-Kernel Architecture (HSB Example) {#three-kernel-architecture}[¶](#the-3-kernel-architecture-hsb-example-three-kernel-architecture "Permalink to this heading"){.headerlink}

The HSB workflow separates concerns into three persistent GPU kernels
that communicate via shared ring buffers:

![3-kernel
architecture](data:image/svg+xml;base64,PHN2ZyBpZD0ibWVybWFpZC1zdmciIHdpZHRoPSIxMDAlIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGNsYXNzPSJmbG93Y2hhcnQiIHN0eWxlPSJtYXgtd2lkdGg6IDU1Mi42MDE1NjI1cHg7IiB2aWV3Qm94PSIwIDAgNTUyLjYwMTU2MjUgODg2IiByb2xlPSJncmFwaGljcy1kb2N1bWVudCBkb2N1bWVudCIgYXJpYS1yb2xlZGVzY3JpcHRpb249ImZsb3djaGFydC12MiIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPjxzdHlsZSB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCI+QGltcG9ydCB1cmwoImh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL2ZvbnQtYXdlc29tZS82LjcuMi9jc3MvYWxsLm1pbi5jc3MiKTs8L3N0eWxlPjxzdHlsZT4jbWVybWFpZC1zdmd7Zm9udC1mYW1pbHk6InRyZWJ1Y2hldCBtcyIsdmVyZGFuYSxhcmlhbCxzYW5zLXNlcmlmO2ZvbnQtc2l6ZToxNnB4O2ZpbGw6IzMzMzt9QGtleWZyYW1lcyBlZGdlLWFuaW1hdGlvbi1mcmFtZXtmcm9te3N0cm9rZS1kYXNob2Zmc2V0OjA7fX1Aa2V5ZnJhbWVzIGRhc2h7dG97c3Ryb2tlLWRhc2hvZmZzZXQ6MDt9fSNtZXJtYWlkLXN2ZyAuZWRnZS1hbmltYXRpb24tc2xvd3tzdHJva2UtZGFzaGFycmF5OjksNSFpbXBvcnRhbnQ7c3Ryb2tlLWRhc2hvZmZzZXQ6OTAwO2FuaW1hdGlvbjpkYXNoIDUwcyBsaW5lYXIgaW5maW5pdGU7c3Ryb2tlLWxpbmVjYXA6cm91bmQ7fSNtZXJtYWlkLXN2ZyAuZWRnZS1hbmltYXRpb24tZmFzdHtzdHJva2UtZGFzaGFycmF5OjksNSFpbXBvcnRhbnQ7c3Ryb2tlLWRhc2hvZmZzZXQ6OTAwO2FuaW1hdGlvbjpkYXNoIDIwcyBsaW5lYXIgaW5maW5pdGU7c3Ryb2tlLWxpbmVjYXA6cm91bmQ7fSNtZXJtYWlkLXN2ZyAuZXJyb3ItaWNvbntmaWxsOiM1NTIyMjI7fSNtZXJtYWlkLXN2ZyAuZXJyb3ItdGV4dHtmaWxsOiM1NTIyMjI7c3Ryb2tlOiM1NTIyMjI7fSNtZXJtYWlkLXN2ZyAuZWRnZS10aGlja25lc3Mtbm9ybWFse3N0cm9rZS13aWR0aDoxcHg7fSNtZXJtYWlkLXN2ZyAuZWRnZS10aGlja25lc3MtdGhpY2t7c3Ryb2tlLXdpZHRoOjMuNXB4O30jbWVybWFpZC1zdmcgLmVkZ2UtcGF0dGVybi1zb2xpZHtzdHJva2UtZGFzaGFycmF5OjA7fSNtZXJtYWlkLXN2ZyAuZWRnZS10aGlja25lc3MtaW52aXNpYmxle3N0cm9rZS13aWR0aDowO2ZpbGw6bm9uZTt9I21lcm1haWQtc3ZnIC5lZGdlLXBhdHRlcm4tZGFzaGVke3N0cm9rZS1kYXNoYXJyYXk6Mzt9I21lcm1haWQtc3ZnIC5lZGdlLXBhdHRlcm4tZG90dGVke3N0cm9rZS1kYXNoYXJyYXk6Mjt9I21lcm1haWQtc3ZnIC5tYXJrZXJ7ZmlsbDojMzMzMzMzO3N0cm9rZTojMzMzMzMzO30jbWVybWFpZC1zdmcgLm1hcmtlci5jcm9zc3tzdHJva2U6IzMzMzMzMzt9I21lcm1haWQtc3ZnIHN2Z3tmb250LWZhbWlseToidHJlYnVjaGV0IG1zIix2ZXJkYW5hLGFyaWFsLHNhbnMtc2VyaWY7Zm9udC1zaXplOjE2cHg7fSNtZXJtYWlkLXN2ZyBwe21hcmdpbjowO30jbWVybWFpZC1zdmcgLmxhYmVse2ZvbnQtZmFtaWx5OiJ0cmVidWNoZXQgbXMiLHZlcmRhbmEsYXJpYWwsc2Fucy1zZXJpZjtjb2xvcjojMzMzO30jbWVybWFpZC1zdmcgLmNsdXN0ZXItbGFiZWwgdGV4dHtmaWxsOiMzMzM7fSNtZXJtYWlkLXN2ZyAuY2x1c3Rlci1sYWJlbCBzcGFue2NvbG9yOiMzMzM7fSNtZXJtYWlkLXN2ZyAuY2x1c3Rlci1sYWJlbCBzcGFuIHB7YmFja2dyb3VuZC1jb2xvcjp0cmFuc3BhcmVudDt9I21lcm1haWQtc3ZnIC5sYWJlbCB0ZXh0LCNtZXJtYWlkLXN2ZyBzcGFue2ZpbGw6IzMzMztjb2xvcjojMzMzO30jbWVybWFpZC1zdmcgLm5vZGUgcmVjdCwjbWVybWFpZC1zdmcgLm5vZGUgY2lyY2xlLCNtZXJtYWlkLXN2ZyAubm9kZSBlbGxpcHNlLCNtZXJtYWlkLXN2ZyAubm9kZSBwb2x5Z29uLCNtZXJtYWlkLXN2ZyAubm9kZSBwYXRoe2ZpbGw6I0VDRUNGRjtzdHJva2U6IzkzNzBEQjtzdHJva2Utd2lkdGg6MXB4O30jbWVybWFpZC1zdmcgLnJvdWdoLW5vZGUgLmxhYmVsIHRleHQsI21lcm1haWQtc3ZnIC5ub2RlIC5sYWJlbCB0ZXh0LCNtZXJtYWlkLXN2ZyAuaW1hZ2Utc2hhcGUgLmxhYmVsLCNtZXJtYWlkLXN2ZyAuaWNvbi1zaGFwZSAubGFiZWx7dGV4dC1hbmNob3I6bWlkZGxlO30jbWVybWFpZC1zdmcgLm5vZGUgLmthdGV4IHBhdGh7ZmlsbDojMDAwO3N0cm9rZTojMDAwO3N0cm9rZS13aWR0aDoxcHg7fSNtZXJtYWlkLXN2ZyAucm91Z2gtbm9kZSAubGFiZWwsI21lcm1haWQtc3ZnIC5ub2RlIC5sYWJlbCwjbWVybWFpZC1zdmcgLmltYWdlLXNoYXBlIC5sYWJlbCwjbWVybWFpZC1zdmcgLmljb24tc2hhcGUgLmxhYmVse3RleHQtYWxpZ246Y2VudGVyO30jbWVybWFpZC1zdmcgLm5vZGUuY2xpY2thYmxle2N1cnNvcjpwb2ludGVyO30jbWVybWFpZC1zdmcgLnJvb3QgLmFuY2hvciBwYXRoe2ZpbGw6IzMzMzMzMyFpbXBvcnRhbnQ7c3Ryb2tlLXdpZHRoOjA7c3Ryb2tlOiMzMzMzMzM7fSNtZXJtYWlkLXN2ZyAuYXJyb3doZWFkUGF0aHtmaWxsOiMzMzMzMzM7fSNtZXJtYWlkLXN2ZyAuZWRnZVBhdGggLnBhdGh7c3Ryb2tlOiMzMzMzMzM7c3Ryb2tlLXdpZHRoOjIuMHB4O30jbWVybWFpZC1zdmcgLmZsb3djaGFydC1saW5re3N0cm9rZTojMzMzMzMzO2ZpbGw6bm9uZTt9I21lcm1haWQtc3ZnIC5lZGdlTGFiZWx7YmFja2dyb3VuZC1jb2xvcjpyZ2JhKDIzMiwyMzIsMjMyLCAwLjgpO3RleHQtYWxpZ246Y2VudGVyO30jbWVybWFpZC1zdmcgLmVkZ2VMYWJlbCBwe2JhY2tncm91bmQtY29sb3I6cmdiYSgyMzIsMjMyLDIzMiwgMC44KTt9I21lcm1haWQtc3ZnIC5lZGdlTGFiZWwgcmVjdHtvcGFjaXR5OjAuNTtiYWNrZ3JvdW5kLWNvbG9yOnJnYmEoMjMyLDIzMiwyMzIsIDAuOCk7ZmlsbDpyZ2JhKDIzMiwyMzIsMjMyLCAwLjgpO30jbWVybWFpZC1zdmcgLmxhYmVsQmtne2JhY2tncm91bmQtY29sb3I6cmdiYSgyMzIsIDIzMiwgMjMyLCAwLjUpO30jbWVybWFpZC1zdmcgLmNsdXN0ZXIgcmVjdHtmaWxsOiNmZmZmZGU7c3Ryb2tlOiNhYWFhMzM7c3Ryb2tlLXdpZHRoOjFweDt9I21lcm1haWQtc3ZnIC5jbHVzdGVyIHRleHR7ZmlsbDojMzMzO30jbWVybWFpZC1zdmcgLmNsdXN0ZXIgc3Bhbntjb2xvcjojMzMzO30jbWVybWFpZC1zdmcgZGl2Lm1lcm1haWRUb29sdGlwe3Bvc2l0aW9uOmFic29sdXRlO3RleHQtYWxpZ246Y2VudGVyO21heC13aWR0aDoyMDBweDtwYWRkaW5nOjJweDtmb250LWZhbWlseToidHJlYnVjaGV0IG1zIix2ZXJkYW5hLGFyaWFsLHNhbnMtc2VyaWY7Zm9udC1zaXplOjEycHg7YmFja2dyb3VuZDpoc2woODAsIDEwMCUsIDk2LjI3NDUwOTgwMzklKTtib3JkZXI6MXB4IHNvbGlkICNhYWFhMzM7Ym9yZGVyLXJhZGl1czoycHg7cG9pbnRlci1ldmVudHM6bm9uZTt6LWluZGV4OjEwMDt9I21lcm1haWQtc3ZnIC5mbG93Y2hhcnRUaXRsZVRleHR7dGV4dC1hbmNob3I6bWlkZGxlO2ZvbnQtc2l6ZToxOHB4O2ZpbGw6IzMzMzt9I21lcm1haWQtc3ZnIHJlY3QudGV4dHtmaWxsOm5vbmU7c3Ryb2tlLXdpZHRoOjA7fSNtZXJtYWlkLXN2ZyAuaWNvbi1zaGFwZSwjbWVybWFpZC1zdmcgLmltYWdlLXNoYXBle2JhY2tncm91bmQtY29sb3I6cmdiYSgyMzIsMjMyLDIzMiwgMC44KTt0ZXh0LWFsaWduOmNlbnRlcjt9I21lcm1haWQtc3ZnIC5pY29uLXNoYXBlIHAsI21lcm1haWQtc3ZnIC5pbWFnZS1zaGFwZSBwe2JhY2tncm91bmQtY29sb3I6cmdiYSgyMzIsMjMyLDIzMiwgMC44KTtwYWRkaW5nOjJweDt9I21lcm1haWQtc3ZnIC5pY29uLXNoYXBlIHJlY3QsI21lcm1haWQtc3ZnIC5pbWFnZS1zaGFwZSByZWN0e29wYWNpdHk6MC41O2JhY2tncm91bmQtY29sb3I6cmdiYSgyMzIsMjMyLDIzMiwgMC44KTtmaWxsOnJnYmEoMjMyLDIzMiwyMzIsIDAuOCk7fSNtZXJtYWlkLXN2ZyAubGFiZWwtaWNvbntkaXNwbGF5OmlubGluZS1ibG9jaztoZWlnaHQ6MWVtO292ZXJmbG93OnZpc2libGU7dmVydGljYWwtYWxpZ246LTAuMTI1ZW07fSNtZXJtYWlkLXN2ZyAubm9kZSAubGFiZWwtaWNvbiBwYXRoe2ZpbGw6Y3VycmVudENvbG9yO3N0cm9rZTpyZXZlcnQ7c3Ryb2tlLXdpZHRoOnJldmVydDt9I21lcm1haWQtc3ZnIDpyb290ey0tbWVybWFpZC1mb250LWZhbWlseToidHJlYnVjaGV0IG1zIix2ZXJkYW5hLGFyaWFsLHNhbnMtc2VyaWY7fTwvc3R5bGU+PGc+PG1hcmtlciBpZD0ibWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLXBvaW50RW5kIiBjbGFzcz0ibWFya2VyIGZsb3djaGFydC12MiIgdmlld0JveD0iMCAwIDEwIDEwIiByZWZYPSI1IiByZWZZPSI1IiBtYXJrZXJVbml0cz0idXNlclNwYWNlT25Vc2UiIG1hcmtlcldpZHRoPSI4IiBtYXJrZXJIZWlnaHQ9IjgiIG9yaWVudD0iYXV0byI+PHBhdGggZD0iTSAwIDAgTCAxMCA1IEwgMCAxMCB6IiBjbGFzcz0iYXJyb3dNYXJrZXJQYXRoIiBzdHlsZT0ic3Ryb2tlLXdpZHRoOiAxOyBzdHJva2UtZGFzaGFycmF5OiAxLCAwOyIvPjwvbWFya2VyPjxtYXJrZXIgaWQ9Im1lcm1haWQtc3ZnX2Zsb3djaGFydC12Mi1wb2ludFN0YXJ0IiBjbGFzcz0ibWFya2VyIGZsb3djaGFydC12MiIgdmlld0JveD0iMCAwIDEwIDEwIiByZWZYPSI0LjUiIHJlZlk9IjUiIG1hcmtlclVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgbWFya2VyV2lkdGg9IjgiIG1hcmtlckhlaWdodD0iOCIgb3JpZW50PSJhdXRvIj48cGF0aCBkPSJNIDAgNSBMIDEwIDEwIEwgMTAgMCB6IiBjbGFzcz0iYXJyb3dNYXJrZXJQYXRoIiBzdHlsZT0ic3Ryb2tlLXdpZHRoOiAxOyBzdHJva2UtZGFzaGFycmF5OiAxLCAwOyIvPjwvbWFya2VyPjxtYXJrZXIgaWQ9Im1lcm1haWQtc3ZnX2Zsb3djaGFydC12Mi1jaXJjbGVFbmQiIGNsYXNzPSJtYXJrZXIgZmxvd2NoYXJ0LXYyIiB2aWV3Qm94PSIwIDAgMTAgMTAiIHJlZlg9IjExIiByZWZZPSI1IiBtYXJrZXJVbml0cz0idXNlclNwYWNlT25Vc2UiIG1hcmtlcldpZHRoPSIxMSIgbWFya2VySGVpZ2h0PSIxMSIgb3JpZW50PSJhdXRvIj48Y2lyY2xlIGN4PSI1IiBjeT0iNSIgcj0iNSIgY2xhc3M9ImFycm93TWFya2VyUGF0aCIgc3R5bGU9InN0cm9rZS13aWR0aDogMTsgc3Ryb2tlLWRhc2hhcnJheTogMSwgMDsiLz48L21hcmtlcj48bWFya2VyIGlkPSJtZXJtYWlkLXN2Z19mbG93Y2hhcnQtdjItY2lyY2xlU3RhcnQiIGNsYXNzPSJtYXJrZXIgZmxvd2NoYXJ0LXYyIiB2aWV3Qm94PSIwIDAgMTAgMTAiIHJlZlg9Ii0xIiByZWZZPSI1IiBtYXJrZXJVbml0cz0idXNlclNwYWNlT25Vc2UiIG1hcmtlcldpZHRoPSIxMSIgbWFya2VySGVpZ2h0PSIxMSIgb3JpZW50PSJhdXRvIj48Y2lyY2xlIGN4PSI1IiBjeT0iNSIgcj0iNSIgY2xhc3M9ImFycm93TWFya2VyUGF0aCIgc3R5bGU9InN0cm9rZS13aWR0aDogMTsgc3Ryb2tlLWRhc2hhcnJheTogMSwgMDsiLz48L21hcmtlcj48bWFya2VyIGlkPSJtZXJtYWlkLXN2Z19mbG93Y2hhcnQtdjItY3Jvc3NFbmQiIGNsYXNzPSJtYXJrZXIgY3Jvc3MgZmxvd2NoYXJ0LXYyIiB2aWV3Qm94PSIwIDAgMTEgMTEiIHJlZlg9IjEyIiByZWZZPSI1LjIiIG1hcmtlclVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgbWFya2VyV2lkdGg9IjExIiBtYXJrZXJIZWlnaHQ9IjExIiBvcmllbnQ9ImF1dG8iPjxwYXRoIGQ9Ik0gMSwxIGwgOSw5IE0gMTAsMSBsIC05LDkiIGNsYXNzPSJhcnJvd01hcmtlclBhdGgiIHN0eWxlPSJzdHJva2Utd2lkdGg6IDI7IHN0cm9rZS1kYXNoYXJyYXk6IDEsIDA7Ii8+PC9tYXJrZXI+PG1hcmtlciBpZD0ibWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLWNyb3NzU3RhcnQiIGNsYXNzPSJtYXJrZXIgY3Jvc3MgZmxvd2NoYXJ0LXYyIiB2aWV3Qm94PSIwIDAgMTEgMTEiIHJlZlg9Ii0xIiByZWZZPSI1LjIiIG1hcmtlclVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgbWFya2VyV2lkdGg9IjExIiBtYXJrZXJIZWlnaHQ9IjExIiBvcmllbnQ9ImF1dG8iPjxwYXRoIGQ9Ik0gMSwxIGwgOSw5IE0gMTAsMSBsIC05LDkiIGNsYXNzPSJhcnJvd01hcmtlclBhdGgiIHN0eWxlPSJzdHJva2Utd2lkdGg6IDI7IHN0cm9rZS1kYXNoYXJyYXk6IDEsIDA7Ii8+PC9tYXJrZXI+PGcgY2xhc3M9InJvb3QiPjxnIGNsYXNzPSJjbHVzdGVycyIvPjxnIGNsYXNzPSJlZGdlUGF0aHMiPjxwYXRoIGQ9Ik0zMjMuMDA4LDYyTDMxNC43ODIsNjguMTY3QzMwNi41NTYsNzQuMzMzLDI5MC4xMDQsODYuNjY3LDI4OS41NzEsOTguNkMyODkuMDM3LDExMC41MzQsMzA0LjQyMiwxMjIuMDY3LDMxMi4xMTUsMTI3LjgzNEwzMTkuODA3LDEzMy42MDEiIGlkPSJMX0ZQR0FfUkRNQV8wIiBjbGFzcz0iIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBmbG93Y2hhcnQtbGluayIgc3R5bGU9IjsiIGRhdGEtZWRnZT0idHJ1ZSIgZGF0YS1ldD0iZWRnZSIgZGF0YS1pZD0iTF9GUEdBX1JETUFfMCIgZGF0YS1wb2ludHM9Ilczc2llQ0k2TXpJekxqQXdOelV3TnpNeU5ESXhPRGMxTENKNUlqbzJNbjBzZXlKNElqb3lOek11TmpVeU16UXpOelVzSW5raU9qazVmU3g3SW5naU9qTXlNeTR3TURjMU1EY3pNalF5TVRnM05Td2llU0k2TVRNMmZWMD0iIG1hcmtlci1lbmQ9InVybCgjbWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLXBvaW50RW5kKSIvPjxwYXRoIGQ9Ik0zMDEuOTI5LDE5MEwyODguODg4LDE5Ni4xNjdDMjc1Ljg0OCwyMDIuMzMzLDI0OS43NjgsMjE0LjY2NywyMzYuNzI4LDIyNi4zMzNDMjIzLjY4OCwyMzgsMjIzLjY4OCwyNDksMjIzLjY4OCwyNTQuNUwyMjMuNjg4LDI2MCIgaWQ9IkxfUkRNQV9SWF8wIiBjbGFzcz0iIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBmbG93Y2hhcnQtbGluayIgc3R5bGU9IjsiIGRhdGEtZWRnZT0idHJ1ZSIgZGF0YS1ldD0iZWRnZSIgZGF0YS1pZD0iTF9SRE1BX1JYXzAiIGRhdGEtcG9pbnRzPSJXM3NpZUNJNk16QXhMamt5T0RVNE9EZzJOekU0TnpVc0lua2lPakU1TUgwc2V5SjRJam95TWpNdU5qZzNOU3dpZVNJNk1qSTNmU3g3SW5naU9qSXlNeTQyT0RjMUxDSjVJam95TmpSOVhRPT0iIG1hcmtlci1lbmQ9InVybCgjbWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLXBvaW50RW5kKSIvPjxwYXRoIGQ9Ik0yMjMuNjg4LDMxOEwyMjMuNjg4LDMyNC4xNjdDMjIzLjY4OCwzMzAuMzMzLDIyMy42ODgsMzQyLjY2NywyMjMuNjg4LDM1NC4zMzNDMjIzLjY4OCwzNjYsMjIzLjY4OCwzNzcsMjIzLjY4OCwzODIuNUwyMjMuNjg4LDM4OCIgaWQ9IkxfUlhfUlhfQlVGXzAiIGNsYXNzPSIgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBlZGdlLXRoaWNrbmVzcy1ub3JtYWwgZWRnZS1wYXR0ZXJuLXNvbGlkIGZsb3djaGFydC1saW5rIiBzdHlsZT0iOyIgZGF0YS1lZGdlPSJ0cnVlIiBkYXRhLWV0PSJlZGdlIiBkYXRhLWlkPSJMX1JYX1JYX0JVRl8wIiBkYXRhLXBvaW50cz0iVzNzaWVDSTZNakl6TGpZNE56VXNJbmtpT2pNeE9IMHNleUo0SWpveU1qTXVOamczTlN3aWVTSTZNelUxZlN4N0luZ2lPakl5TXk0Mk9EYzFMQ0o1SWpvek9USjlYUT09IiBtYXJrZXItZW5kPSJ1cmwoI21lcm1haWQtc3ZnX2Zsb3djaGFydC12Mi1wb2ludEVuZCkiLz48cGF0aCBkPSJNMjIzLjY4OCw0NDZMMjIzLjY4OCw0NTIuMTY3QzIyMy42ODgsNDU4LjMzMywyMjMuNjg4LDQ3MC42NjcsMjIzLjY4OCw0ODIuMzMzQzIyMy42ODgsNDk0LDIyMy42ODgsNTA1LDIyMy42ODgsNTEwLjVMMjIzLjY4OCw1MTYiIGlkPSJMX1JYX0JVRl9ESVNQQVRDSF8wIiBjbGFzcz0iIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBmbG93Y2hhcnQtbGluayIgc3R5bGU9IjsiIGRhdGEtZWRnZT0idHJ1ZSIgZGF0YS1ldD0iZWRnZSIgZGF0YS1pZD0iTF9SWF9CVUZfRElTUEFUQ0hfMCIgZGF0YS1wb2ludHM9Ilczc2llQ0k2TWpJekxqWTROelVzSW5raU9qUTBObjBzZXlKNElqb3lNak11TmpnM05Td2llU0k2TkRnemZTeDdJbmdpT2pJeU15NDJPRGMxTENKNUlqbzFNakI5WFE9PSIgbWFya2VyLWVuZD0idXJsKCNtZXJtYWlkLXN2Z19mbG93Y2hhcnQtdjItcG9pbnRFbmQpIi8+PHBhdGggZD0iTTE3MS40NjEsNTk4TDE2MC41MjUsNjA2LjE2N0MxNDkuNTg5LDYxNC4zMzMsMTI3LjcxNiw2MzAuNjY3LDExNi43OCw2NTEuNDkyQzEwNS44NDQsNjcyLjMxNywxMDUuODQ0LDY5Ny42MzMsMTA1Ljg0NCw3MTAuMjkyTDEwNS44NDQsNzIyLjk1IiBpZD0iRElTUEFUQ0gtY3ljbGljLXNwZWNpYWwtMSIgY2xhc3M9IiBlZGdlLXRoaWNrbmVzcy1ub3JtYWwgZWRnZS1wYXR0ZXJuLXNvbGlkIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZmxvd2NoYXJ0LWxpbmsiIHN0eWxlPSI7IiBkYXRhLWVkZ2U9InRydWUiIGRhdGEtZXQ9ImVkZ2UiIGRhdGEtaWQ9IkRJU1BBVENILWN5Y2xpYy1zcGVjaWFsLTEiIGRhdGEtcG9pbnRzPSJXM3NpZUNJNk1UY3hMalEyTVRJNU1qWXhNell6TmpNM0xDSjVJam8xT1RoOUxIc2llQ0k2TVRBMUxqZzBNemMxTENKNUlqbzJORGQ5TEhzaWVDSTZNVEExTGpnME16YzFMQ0o1SWpvM01qSXVPVFE1T1RrNU9UazVNalUwT1gxZCIvPjxwYXRoIGQ9Ik0xMDUuODQ0LDcyMy4wNUwxMDUuODQ0LDczMy43MDhDMTA1Ljg0NCw3NDQuMzY3LDEwNS44NDQsNzY1LjY4MywxMTMuMTg2LDc4N0MxMjAuNTI5LDgwOC4zMTcsMTM1LjIxNCw4MjkuNjMzLDE0Mi41NTcsODQwLjI5MkwxNDkuODk5LDg1MC45NSIgaWQ9IkRJU1BBVENILWN5Y2xpYy1zcGVjaWFsLW1pZCIgY2xhc3M9IiBlZGdlLXRoaWNrbmVzcy1ub3JtYWwgZWRnZS1wYXR0ZXJuLXNvbGlkIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZmxvd2NoYXJ0LWxpbmsiIHN0eWxlPSI7IiBkYXRhLWVkZ2U9InRydWUiIGRhdGEtZXQ9ImVkZ2UiIGRhdGEtaWQ9IkRJU1BBVENILWN5Y2xpYy1zcGVjaWFsLW1pZCIgZGF0YS1wb2ludHM9Ilczc2llQ0k2TVRBMUxqZzBNemMxTENKNUlqbzNNak11TURVd01EQXdNREF3TnpRMU1YMHNleUo0SWpveE1EVXVPRFF6TnpVc0lua2lPamM0TjMwc2V5SjRJam94TkRrdU9EazVNVFE0TlRVNU1EVTNNRFFzSW5raU9qZzFNQzQ1TkRrNU9UazVPVGt5TlRRNWZWMD0iLz48cGF0aCBkPSJNMTQ5Ljk4NCw4NTAuOTU3TDE2Mi4yNjgsODQwLjI5N0MxNzQuNTUyLDgyOS42MzgsMTk5LjEyLDgwOC4zMTksMjExLjQwNCw3ODYuOTkzQzIyMy42ODgsNzY1LjY2NywyMjMuNjg4LDc0NC4zMzMsMjIzLjY4OCw3MjFDMjIzLjY4OCw2OTcuNjY3LDIyMy42ODgsNjcyLjMzMywyMjMuNjg4LDY1Mi4xNjdDMjIzLjY4OCw2MzIsMjIzLjY4OCw2MTcsMjIzLjY4OCw2MDkuNUwyMjMuNjg4LDYwMiIgaWQ9IkRJU1BBVENILWN5Y2xpYy1zcGVjaWFsLTIiIGNsYXNzPSIgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBlZGdlLXRoaWNrbmVzcy1ub3JtYWwgZWRnZS1wYXR0ZXJuLXNvbGlkIGZsb3djaGFydC1saW5rIiBzdHlsZT0iOyIgZGF0YS1lZGdlPSJ0cnVlIiBkYXRhLWV0PSJlZGdlIiBkYXRhLWlkPSJESVNQQVRDSC1jeWNsaWMtc3BlY2lhbC0yIiBkYXRhLXBvaW50cz0iVzNzaWVDSTZNVFE1TGprNE16VTVNemMxTURjME5UQTJMQ0o1SWpvNE5UQXVPVFUyTmpFeU5EWTJPVEV6Tlgwc2V5SjRJam95TWpNdU5qZzNOU3dpZVNJNk56ZzNmU3g3SW5naU9qSXlNeTQyT0RjMUxDSjVJam8zTWpOOUxIc2llQ0k2TWpJekxqWTROelVzSW5raU9qWTBOMzBzZXlKNElqb3lNak11TmpnM05Td2llU0k2TlRrNGZWMD0iIG1hcmtlci1lbmQ9InVybCgjbWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLXBvaW50RW5kKSIvPjxwYXRoIGQ9Ik0yODMuNjY2LDU5OEwyOTYuMjI2LDYwNi4xNjdDMzA4Ljc4NSw2MTQuMzMzLDMzMy45MDQsNjMwLjY2NywzNDYuNDY0LDY0Ni4zMzNDMzU5LjAyMyw2NjIsMzU5LjAyMyw2NzcsMzU5LjAyMyw2ODQuNUwzNTkuMDIzLDY5MiIgaWQ9IkxfRElTUEFUQ0hfVFhfQlVGXzAiIGNsYXNzPSIgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBlZGdlLXRoaWNrbmVzcy1ub3JtYWwgZWRnZS1wYXR0ZXJuLXNvbGlkIGZsb3djaGFydC1saW5rIiBzdHlsZT0iOyIgZGF0YS1lZGdlPSJ0cnVlIiBkYXRhLWV0PSJlZGdlIiBkYXRhLWlkPSJMX0RJU1BBVENIX1RYX0JVRl8wIiBkYXRhLXBvaW50cz0iVzNzaWVDSTZNamd6TGpZMk5Ua3lOamcwTmpVNU1Ea3NJbmtpT2pVNU9IMHNleUo0SWpvek5Ua3VNREl6TkRNM05Td2llU0k2TmpRM2ZTeDdJbmdpT2pNMU9TNHdNak0wTXpjMUxDSjVJam8yT1RaOVhRPT0iIG1hcmtlci1lbmQ9InVybCgjbWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLXBvaW50RW5kKSIvPjxwYXRoIGQ9Ik0zNTkuMDIzLDc1MEwzNTkuMDIzLDc1Ni4xNjdDMzU5LjAyMyw3NjIuMzMzLDM1OS4wMjMsNzc0LjY2NywzNjUuMDU5LDc4Ni41NDJDMzcxLjA5NSw3OTguNDE3LDM4My4xNjYsODA5LjgzNCwzODkuMjAyLDgxNS41NDNMMzk1LjIzOCw4MjEuMjUxIiBpZD0iTF9UWF9CVUZfVFhfMCIgY2xhc3M9IiBlZGdlLXRoaWNrbmVzcy1ub3JtYWwgZWRnZS1wYXR0ZXJuLXNvbGlkIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZmxvd2NoYXJ0LWxpbmsiIHN0eWxlPSI7IiBkYXRhLWVkZ2U9InRydWUiIGRhdGEtZXQ9ImVkZ2UiIGRhdGEtaWQ9IkxfVFhfQlVGX1RYXzAiIGRhdGEtcG9pbnRzPSJXM3NpZUNJNk16VTVMakF5TXpRek56VXNJbmtpT2pjMU1IMHNleUo0SWpvek5Ua3VNREl6TkRNM05Td2llU0k2TnpnM2ZTeDdJbmdpT2pNNU9DNHhORE01T0RFNU16TTFPVE0zTlN3aWVTSTZPREkwZlYwPSIgbWFya2VyLWVuZD0idXJsKCNtZXJtYWlkLXN2Z19mbG93Y2hhcnQtdjItcG9pbnRFbmQpIi8+PHBhdGggZD0iTTQ1NS4yMzksODI0TDQ2MS43NTksODE3LjgzM0M0NjguMjc5LDgxMS42NjcsNDgxLjMxOSw3OTkuMzMzLDQ4Ny44MzksNzgyLjVDNDk0LjM1OSw3NjUuNjY3LDQ5NC4zNTksNzQ0LjMzMyw0OTQuMzU5LDcyMUM0OTQuMzU5LDY5Ny42NjcsNDk0LjM1OSw2NzIuMzMzLDQ5NC4zNTksNjQ1QzQ5NC4zNTksNjE3LjY2Nyw0OTQuMzU5LDU4OC4zMzMsNDk0LjM1OSw1NjFDNDk0LjM1OSw1MzMuNjY3LDQ5NC4zNTksNTA4LjMzMyw0OTQuMzU5LDQ4NUM0OTQuMzU5LDQ2MS42NjcsNDk0LjM1OSw0NDAuMzMzLDQ5NC4zNTksNDE5QzQ5NC4zNTksMzk3LjY2Nyw0OTQuMzU5LDM3Ni4zMzMsNDk0LjM1OSwzNTVDNDk0LjM1OSwzMzMuNjY3LDQ5NC4zNTksMzEyLjMzMyw0OTQuMzU5LDI5MUM0OTQuMzU5LDI2OS42NjcsNDk0LjM1OSwyNDguMzMzLDQ4MS45MjIsMjMxLjc4NUM0NjkuNDg0LDIxNS4yMzcsNDQ0LjYwOSwyMDMuNDczLDQzMi4xNzIsMTk3LjU5Mkw0MTkuNzM0LDE5MS43MSIgaWQ9IkxfVFhfUkRNQV8wIiBjbGFzcz0iIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBmbG93Y2hhcnQtbGluayIgc3R5bGU9IjsiIGRhdGEtZWRnZT0idHJ1ZSIgZGF0YS1ldD0iZWRnZSIgZGF0YS1pZD0iTF9UWF9SRE1BXzAiIGRhdGEtcG9pbnRzPSJXM3NpZUNJNk5EVTFMakl6T0Rnek1EVTJOalF3TmpJMUxDSjVJam80TWpSOUxIc2llQ0k2TkRrMExqTTFPVE0zTlN3aWVTSTZOemczZlN4N0luZ2lPalE1TkM0ek5Ua3pOelVzSW5raU9qY3lNMzBzZXlKNElqbzBPVFF1TXpVNU16YzFMQ0o1SWpvMk5EZDlMSHNpZUNJNk5EazBMak0xT1RNM05Td2llU0k2TlRVNWZTeDdJbmdpT2pRNU5DNHpOVGt6TnpVc0lua2lPalE0TTMwc2V5SjRJam8wT1RRdU16VTVNemMxTENKNUlqbzBNVGw5TEhzaWVDSTZORGswTGpNMU9UTTNOU3dpZVNJNk16VTFmU3g3SW5naU9qUTVOQzR6TlRrek56VXNJbmtpT2pJNU1YMHNleUo0SWpvME9UUXVNelU1TXpjMUxDSjVJam95TWpkOUxIc2llQ0k2TkRFMkxqRXhPREk0TmpFek1qZ3hNalVzSW5raU9qRTVNSDFkIiBtYXJrZXItZW5kPSJ1cmwoI21lcm1haWQtc3ZnX2Zsb3djaGFydC12Mi1wb2ludEVuZCkiLz48cGF0aCBkPSJNMzk1LjAzOSwxMzZMNDAzLjI2NSwxMjkuODMzQzQxMS40OTEsMTIzLjY2Nyw0MjcuOTQzLDExMS4zMzMsNDI4LjQ3Niw5OS40QzQyOS4wMSw4Ny40NjYsNDEzLjYyNSw3NS45MzMsNDA1LjkzMiw3MC4xNjZMMzk4LjI0LDY0LjM5OSIgaWQ9IkxfUkRNQV9GUEdBXzAiIGNsYXNzPSIgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBlZGdlLXRoaWNrbmVzcy1ub3JtYWwgZWRnZS1wYXR0ZXJuLXNvbGlkIGZsb3djaGFydC1saW5rIiBzdHlsZT0iOyIgZGF0YS1lZGdlPSJ0cnVlIiBkYXRhLWV0PSJlZGdlIiBkYXRhLWlkPSJMX1JETUFfRlBHQV8wIiBkYXRhLXBvaW50cz0iVzNzaWVDSTZNemsxTGpBek9UTTJOelkzTlRjNE1USTFMQ0o1SWpveE16WjlMSHNpZUNJNk5EUTBMak01TkRVek1USTFMQ0o1SWpvNU9YMHNleUo0SWpvek9UVXVNRE01TXpZM05qYzFOemd4TWpVc0lua2lPall5ZlYwPSIgbWFya2VyLWVuZD0idXJsKCNtZXJtYWlkLXN2Z19mbG93Y2hhcnQtdjItcG9pbnRFbmQpIi8+PC9nPjxnIGNsYXNzPSJlZGdlTGFiZWxzIj48ZyBjbGFzcz0iZWRnZUxhYmVsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyNzMuNjUyMzQzNzUsIDk5KSI+PGcgY2xhc3M9ImxhYmVsIiBkYXRhLWlkPSJMX0ZQR0FfUkRNQV8wIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzUuMTQ4NDM3NSwgLTEyKSI+PGZvcmVpZ25PYmplY3Qgd2lkdGg9IjE1MC4yOTY4NzUiIGhlaWdodD0iMjQiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIGNsYXNzPSJsYWJlbEJrZyIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlLWNlbGw7IHdoaXRlLXNwYWNlOiBub3dyYXA7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsiPjxzcGFuIGNsYXNzPSJlZGdlTGFiZWwgIj4xLiBTeW5kcm9tZSBwYWNrZXRzPC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJlZGdlTGFiZWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIyMy42ODc1LCAyMjcpIj48ZyBjbGFzcz0ibGFiZWwiIGRhdGEtaWQ9IkxfUkRNQV9SWF8wIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNTEuMTI1LCAtMTIpIj48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMTAyLjI1IiBoZWlnaHQ9IjI0Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBjbGFzcz0ibGFiZWxCa2ciIHN0eWxlPSJkaXNwbGF5OiB0YWJsZS1jZWxsOyB3aGl0ZS1zcGFjZTogbm93cmFwOyBsaW5lLWhlaWdodDogMS41OyBtYXgtd2lkdGg6IDIwMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7Ij48c3BhbiBjbGFzcz0iZWRnZUxhYmVsICI+Mi4gUkRNQSB3cml0ZTwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0iZWRnZUxhYmVsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMjMuNjg3NSwgMzU1KSI+PGcgY2xhc3M9ImxhYmVsIiBkYXRhLWlkPSJMX1JYX1JYX0JVRl8wIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtODguMTA5Mzc1LCAtMTIpIj48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMTc2LjIxODc1IiBoZWlnaHQ9IjI0Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBjbGFzcz0ibGFiZWxCa2ciIHN0eWxlPSJkaXNwbGF5OiB0YWJsZS1jZWxsOyB3aGl0ZS1zcGFjZTogbm93cmFwOyBsaW5lLWhlaWdodDogMS41OyBtYXgtd2lkdGg6IDIwMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7Ij48c3BhbiBjbGFzcz0iZWRnZUxhYmVsICI+My4gV3JpdGUgc2xvdCArIHNldCByeF9mbGFnPC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJlZGdlTGFiZWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIyMy42ODc1LCA0ODMpIj48ZyBjbGFzcz0ibGFiZWwiIGRhdGEtaWQ9IkxfUlhfQlVGX0RJU1BBVENIXzAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC00OC40Njg3NSwgLTEyKSI+PGZvcmVpZ25PYmplY3Qgd2lkdGg9Ijk2LjkzNzUiIGhlaWdodD0iMjQiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIGNsYXNzPSJsYWJlbEJrZyIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlLWNlbGw7IHdoaXRlLXNwYWNlOiBub3dyYXA7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsiPjxzcGFuIGNsYXNzPSJlZGdlTGFiZWwgIj40LiBQb2xsIHJ4X2ZsYWc8L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PGcgY2xhc3M9ImVkZ2VMYWJlbCI+PGcgY2xhc3M9ImxhYmVsIiBkYXRhLWlkPSJESVNQQVRDSC1jeWNsaWMtc3BlY2lhbC0xIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLCAwKSI+PGZvcmVpZ25PYmplY3Qgd2lkdGg9IjAiIGhlaWdodD0iMCI+PGRpdiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCIgY2xhc3M9ImxhYmVsQmtnIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9ImVkZ2VMYWJlbCAiPjwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0iZWRnZUxhYmVsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMDUuODQzNzUsIDc4NykiPjxnIGNsYXNzPSJsYWJlbCIgZGF0YS1pZD0iRElTUEFUQ0gtY3ljbGljLXNwZWNpYWwtbWlkIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtOTcuODQzNzUsIC0xMikiPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIxOTUuNjg3NSIgaGVpZ2h0PSIyNCI+PGRpdiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCIgY2xhc3M9ImxhYmVsQmtnIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9ImVkZ2VMYWJlbCAiPjUuIEV4ZWN1dGUgZGVjb2RlciBoYW5kbGVyPC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJlZGdlTGFiZWwiPjxnIGNsYXNzPSJsYWJlbCIgZGF0YS1pZD0iRElTUEFUQ0gtY3ljbGljLXNwZWNpYWwtMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwgMCkiPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIwIiBoZWlnaHQ9IjAiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIGNsYXNzPSJsYWJlbEJrZyIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlLWNlbGw7IHdoaXRlLXNwYWNlOiBub3dyYXA7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsiPjxzcGFuIGNsYXNzPSJlZGdlTGFiZWwgIj48L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PGcgY2xhc3M9ImVkZ2VMYWJlbCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzU5LjAyMzQzNzUsIDY0NykiPjxnIGNsYXNzPSJsYWJlbCIgZGF0YS1pZD0iTF9ESVNQQVRDSF9UWF9CVUZfMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEwMCwgLTI0KSI+PGZvcmVpZ25PYmplY3Qgd2lkdGg9IjIwMCIgaGVpZ2h0PSI0OCI+PGRpdiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCIgY2xhc3M9ImxhYmVsQmtnIiBzdHlsZT0iZGlzcGxheTogdGFibGU7IHdoaXRlLXNwYWNlOiBicmVhay1zcGFjZXM7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsgd2lkdGg6IDIwMHB4OyI+PHNwYW4gY2xhc3M9ImVkZ2VMYWJlbCAiPjYuIFdyaXRlIHJlc3BvbnNlICsgc2V0IHR4X2ZsYWc8L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PGcgY2xhc3M9ImVkZ2VMYWJlbCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzU5LjAyMzQzNzUsIDc4NykiPjxnIGNsYXNzPSJsYWJlbCIgZGF0YS1pZD0iTF9UWF9CVUZfVFhfMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTQ4LjAzMTI1LCAtMTIpIj48Zm9yZWlnbk9iamVjdCB3aWR0aD0iOTYuMDYyNSIgaGVpZ2h0PSIyNCI+PGRpdiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCIgY2xhc3M9ImxhYmVsQmtnIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9ImVkZ2VMYWJlbCAiPjcuIFBvbGwgdHhfZmxhZzwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0iZWRnZUxhYmVsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0OTQuMzU5Mzc1LCA0ODMpIj48ZyBjbGFzcz0ibGFiZWwiIGRhdGEtaWQ9IkxfVFhfUkRNQV8wIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNTAuMjQyMTg3NSwgLTEyKSI+PGZvcmVpZ25PYmplY3Qgd2lkdGg9IjEwMC40ODQzNzUiIGhlaWdodD0iMjQiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIGNsYXNzPSJsYWJlbEJrZyIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlLWNlbGw7IHdoaXRlLXNwYWNlOiBub3dyYXA7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsiPjxzcGFuIGNsYXNzPSJlZGdlTGFiZWwgIj44LiBSRE1BIHJlYWQ8L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PGcgY2xhc3M9ImVkZ2VMYWJlbCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDQ0LjM5NDUzMTI1LCA5OSkiPjxnIGNsYXNzPSJsYWJlbCIgZGF0YS1pZD0iTF9SRE1BX0ZQR0FfMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTc1LjU5Mzc1LCAtMTIpIj48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMTUxLjE4NzUiIGhlaWdodD0iMjQiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIGNsYXNzPSJsYWJlbEJrZyIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlLWNlbGw7IHdoaXRlLXNwYWNlOiBub3dyYXA7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsiPjxzcGFuIGNsYXNzPSJlZGdlTGFiZWwgIj45LiBDb3JyZWN0aW9uIHBhY2tldHM8L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PC9nPjxnIGNsYXNzPSJub2RlcyI+PGcgY2xhc3M9Im5vZGUgZGVmYXVsdCAgIiBpZD0iZmxvd2NoYXJ0LUZQR0EtMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzU5LjAyMzQzNzUsIDM1KSI+PHJlY3QgY2xhc3M9ImJhc2ljIGxhYmVsLWNvbnRhaW5lciIgc3R5bGU9IiIgeD0iLTExOC45Mjk2ODc1IiB5PSItMjciIHdpZHRoPSIyMzcuODU5Mzc1IiBoZWlnaHQ9IjU0Ii8+PGcgY2xhc3M9ImxhYmVsIiBzdHlsZT0iIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtODguOTI5Njg3NSwgLTEyKSI+PHJlY3QvPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIxNzcuODU5Mzc1IiBoZWlnaHQ9IjI0Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9Im5vZGVMYWJlbCAiPjxwPkZQR0EgLyBRdWFudHVtIENvbnRyb2w8L3A+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJub2RlIGRlZmF1bHQgICIgaWQ9ImZsb3djaGFydC1SRE1BLTEiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM1OS4wMjM0Mzc1LCAxNjMpIj48cmVjdCBjbGFzcz0iYmFzaWMgbGFiZWwtY29udGFpbmVyIiBzdHlsZT0iIiB4PSItOTguMDIzNDM3NSIgeT0iLTI3IiB3aWR0aD0iMTk2LjA0Njg3NSIgaGVpZ2h0PSI1NCIvPjxnIGNsYXNzPSJsYWJlbCIgc3R5bGU9IiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTY4LjAyMzQzNzUsIC0xMikiPjxyZWN0Lz48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMTM2LjA0Njg3NSIgaGVpZ2h0PSIyNCI+PGRpdiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlLWNlbGw7IHdoaXRlLXNwYWNlOiBub3dyYXA7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsiPjxzcGFuIGNsYXNzPSJub2RlTGFiZWwgIj48cD5Db25uZWN0WC03IFJETUE8L3A+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJub2RlIGRlZmF1bHQgICIgaWQ9ImZsb3djaGFydC1SWC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMjMuNjg3NSwgMjkxKSI+PHJlY3QgY2xhc3M9ImJhc2ljIGxhYmVsLWNvbnRhaW5lciIgc3R5bGU9IiIgeD0iLTEwMi40Njg3NSIgeT0iLTI3IiB3aWR0aD0iMjA0LjkzNzUiIGhlaWdodD0iNTQiLz48ZyBjbGFzcz0ibGFiZWwiIHN0eWxlPSIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC03Mi40Njg3NSwgLTEyKSI+PHJlY3QvPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIxNDQuOTM3NSIgaGVpZ2h0PSIyNCI+PGRpdiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlLWNlbGw7IHdoaXRlLXNwYWNlOiBub3dyYXA7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsiPjxzcGFuIGNsYXNzPSJub2RlTGFiZWwgIj48cD5SWCBLZXJuZWwgKEhvbG9saW5rKTwvcD48L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PGcgY2xhc3M9Im5vZGUgZGVmYXVsdCAgIiBpZD0iZmxvd2NoYXJ0LVJYX0JVRi01IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMjMuNjg3NSwgNDE5KSI+PHJlY3QgY2xhc3M9ImJhc2ljIGxhYmVsLWNvbnRhaW5lciIgc3R5bGU9IiIgeD0iLTEwMS42NjQwNjI1IiB5PSItMjciIHdpZHRoPSIyMDMuMzI4MTI1IiBoZWlnaHQ9IjU0Ii8+PGcgY2xhc3M9ImxhYmVsIiBzdHlsZT0iIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzEuNjY0MDYyNSwgLTEyKSI+PHJlY3QvPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIxNDMuMzI4MTI1IiBoZWlnaHQ9IjI0Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9Im5vZGVMYWJlbCAiPjxwPlJYIEJ1ZmZlciArIHJ4X2ZsYWdzPC9wPjwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0ibm9kZSBkZWZhdWx0ICAiIGlkPSJmbG93Y2hhcnQtRElTUEFUQ0gtNyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjIzLjY4NzUsIDU1OSkiPjxyZWN0IGNsYXNzPSJiYXNpYyBsYWJlbC1jb250YWluZXIiIHN0eWxlPSIiIHg9Ii0xMzAiIHk9Ii0zOSIgd2lkdGg9IjI2MCIgaGVpZ2h0PSI3OCIvPjxnIGNsYXNzPSJsYWJlbCIgc3R5bGU9IiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEwMCwgLTI0KSI+PHJlY3QvPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iNDgiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIHN0eWxlPSJkaXNwbGF5OiB0YWJsZTsgd2hpdGUtc3BhY2U6IGJyZWFrLXNwYWNlczsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyB3aWR0aDogMjAwcHg7Ij48c3BhbiBjbGFzcz0ibm9kZUxhYmVsICI+PHA+RGlzcGF0Y2ggS2VybmVsIChDVURBLVEgUmVhbHRpbWUpPC9wPjwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0ibm9kZSBkZWZhdWx0ICAiIGlkPSJmbG93Y2hhcnQtVFhfQlVGLTExIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzNTkuMDIzNDM3NSwgNzIzKSI+PHJlY3QgY2xhc3M9ImJhc2ljIGxhYmVsLWNvbnRhaW5lciIgc3R5bGU9IiIgeD0iLTEwMC4zMzU5Mzc1IiB5PSItMjciIHdpZHRoPSIyMDAuNjcxODc1IiBoZWlnaHQ9IjU0Ii8+PGcgY2xhc3M9ImxhYmVsIiBzdHlsZT0iIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNzAuMzM1OTM3NSwgLTEyKSI+PHJlY3QvPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIxNDAuNjcxODc1IiBoZWlnaHQ9IjI0Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9Im5vZGVMYWJlbCAiPjxwPlRYIEJ1ZmZlciArIHR4X2ZsYWdzPC9wPjwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0ibm9kZSBkZWZhdWx0ICAiIGlkPSJmbG93Y2hhcnQtVFgtMTMiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQyNi42OTE0MDYyNSwgODUxKSI+PHJlY3QgY2xhc3M9ImJhc2ljIGxhYmVsLWNvbnRhaW5lciIgc3R5bGU9IiIgeD0iLTEwMS41NzgxMjUiIHk9Ii0yNyIgd2lkdGg9IjIwMy4xNTYyNSIgaGVpZ2h0PSI1NCIvPjxnIGNsYXNzPSJsYWJlbCIgc3R5bGU9IiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTcxLjU3ODEyNSwgLTEyKSI+PHJlY3QvPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIxNDMuMTU2MjUiIGhlaWdodD0iMjQiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIHN0eWxlPSJkaXNwbGF5OiB0YWJsZS1jZWxsOyB3aGl0ZS1zcGFjZTogbm93cmFwOyBsaW5lLWhlaWdodDogMS41OyBtYXgtd2lkdGg6IDIwMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7Ij48c3BhbiBjbGFzcz0ibm9kZUxhYmVsICI+PHA+VFggS2VybmVsIChIb2xvbGluayk8L3A+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJsYWJlbCBlZGdlTGFiZWwiIGlkPSJESVNQQVRDSC0tLURJU1BBVENILS0tMSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTA1Ljg0Mzc1LCA3MjMpIj48cmVjdCB3aWR0aD0iMC4xIiBoZWlnaHQ9IjAuMSIvPjxnIGNsYXNzPSJsYWJlbCIgc3R5bGU9IiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwgMCkiPjxyZWN0Lz48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMCIgaGVpZ2h0PSIwIj48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAxMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7Ij48c3BhbiBjbGFzcz0ibm9kZUxhYmVsICI+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJsYWJlbCBlZGdlTGFiZWwiIGlkPSJESVNQQVRDSC0tLURJU1BBVENILS0tMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTQ5LjkzMzU5Mzc1LCA4NTEpIj48cmVjdCB3aWR0aD0iMC4xIiBoZWlnaHQ9IjAuMSIvPjxnIGNsYXNzPSJsYWJlbCIgc3R5bGU9IiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwgMCkiPjxyZWN0Lz48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMCIgaGVpZ2h0PSIwIj48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAxMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7Ij48c3BhbiBjbGFzcz0ibm9kZUxhYmVsICI+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjwvZz48L2c+PC9nPjwvc3ZnPg==){width="553"
height="886"}

::: {#data-flow-summary .section}
### Data Flow Summary[¶](#data-flow-summary "Permalink to this heading"){.headerlink}

  Step   Component         Action
  ------ ----------------- -----------------------------------------------------------------------------------------------
  1-2    FPGA → ConnectX   Detection event data sent over Ethernet, RDMA writes to GPU memory
  3      RX Kernel         Frames detection events into RPC message, sets `rx_flags[slot]` (see Message completion note)
  4-5    Dispatch Kernel   Polls for ready slots, looks up handler by `function_id`, executes decoder
  6      Dispatch Kernel   Writes `RPCResponse` + correction, sets `tx_flags[slot]`
  7-8    TX Kernel         Polls for responses, triggers RDMA send back to FPGA
  9      ConnectX → FPGA   Correction delivered to quantum controller
:::

::: {#why-3-kernels .section}
### Why 3 Kernels?[¶](#why-3-kernels "Permalink to this heading"){.headerlink}

1.  **Separation of concerns**: Transport (RX/TX kernels) vs. compute
    (dispatch) are decoupled

2.  **Reusability**: Same dispatch kernel works with any decoder handler

3.  **Testability**: Dispatch kernel can be tested without HSB hardware

4.  **Flexibility**: RX/TX kernels can be replaced with different
    transport mechanisms

5.  **Transport independence**: The protocol works with HSB,
    [`libibverbs`{.docutils .literal .notranslate}]{.pre}, or
    proprietary transports

For use cases where lowest possible latency is needed, see [[Unified
Dispatch Mode]{.std .std-ref}](#unified-dispatch-mode){.reference
.internal} which combines all three kernels into one while retaining
transport independence through a pluggable launch function.
:::
:::

::: {#unified-dispatch-mode .section}
## Unified Dispatch Mode[¶](#unified-dispatch-mode "Permalink to this heading"){.headerlink}

The **unified dispatch mode** ([`CUDAQ_KERNEL_UNIFIED`{.docutils
.literal .notranslate}]{.pre}) is an alternative to the 3-kernel
architecture that combines receive, RPC dispatch, and transmit into a
single GPU kernel. By eliminating the inter-kernel ring-buffer flag
handoff between RX, dispatch, and TX kernels, the unified kernel reduces
round-trip latency for simple (non-cooperative) RPC handlers.

::: {#architecture .section}
### Architecture[¶](#architecture "Permalink to this heading"){.headerlink}

In unified mode, a single GPU thread runs a transport-provided kernel
that combines receive, dispatch, and transmit into one tight loop:

1.  Polls for an incoming message (transport-specific mechanism)

2.  Parses the [`RPCHeader`{.docutils .literal .notranslate}]{.pre} from
    the receive buffer

3.  Looks up and calls the registered handler in-place

4.  Writes the [`RPCResponse`{.docutils .literal .notranslate}]{.pre}
    header (overwriting the request header)

5.  Sends the response (transport-specific mechanism)

6.  Re-posts the receive buffer for the next message

The symmetric ring layout means the response overwrites the request in
the same buffer slot. [`RPCHeader`{.docutils .literal
.notranslate}]{.pre} fields ([`request_id`{.docutils .literal
.notranslate}]{.pre}, [`ptp_timestamp`{.docutils .literal
.notranslate}]{.pre}) are saved to registers before the handler runs.

For example, the [`HSB`{.docutils .literal
.notranslate}]{.pre}/[`DOCA`{.docutils .literal .notranslate}]{.pre}
transport implementation polls a [`DOCA`{.docutils .literal
.notranslate}]{.pre} completion queue ([`CQ`{.docutils .literal
.notranslate}]{.pre}) in step 1, sends via [`DOCA`{.docutils .literal
.notranslate}]{.pre} [`BlueFlame`{.docutils .literal
.notranslate}]{.pre} in step 5, and re-posts a [`DOCA`{.docutils
.literal .notranslate}]{.pre} receive [`WQE`{.docutils .literal
.notranslate}]{.pre} in step 6. Other transport implementations would
substitute their own receive and send primitives.
:::

::: {#transport-agnostic-design .section}
### Transport-Agnostic Design[¶](#transport-agnostic-design "Permalink to this heading"){.headerlink}

The unified dispatch mode is fully transport-agnostic, just like the
3-kernel mode. The core dispatcher library
([`libcudaq-realtime.so`{.docutils .literal .notranslate}]{.pre}) has no
dependency on any specific transport (no [`DOCA`{.docutils .literal
.notranslate}]{.pre}, no [`HSB`{.docutils .literal
.notranslate}]{.pre}). Unified mode introduces:

-   [`CUDAQ_KERNEL_UNIFIED`{.docutils .literal .notranslate}]{.pre} -- a
    new [`cudaq_kernel_type_t`{.docutils .literal .notranslate}]{.pre}
    enum value

-   [`cudaq_unified_launch_fn_t`{.docutils .literal .notranslate}]{.pre}
    -- a launch function type that receives an opaque [`void*`{.docutils
    .literal .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`transport_ctx`{.docutils .literal
    .notranslate}]{.pre} instead of ring-buffer pointers

-   [`cudaq_dispatcher_set_unified_launch()`{.docutils .literal
    .notranslate}]{.pre} -- wires the launch function and transport
    context to the dispatcher

Transport-specific details are packed into an opaque struct and passed
through the [`void*`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`transport_ctx`{.docutils .literal .notranslate}]{.pre}
pointer. The transport provider supplies both the context struct and the
launch function implementation. For example, the [`HSB`{.docutils
.literal .notranslate}]{.pre}/[`DOCA`{.docutils .literal
.notranslate}]{.pre} transport packs [`DOCA`{.docutils .literal
.notranslate}]{.pre} [`QP`{.docutils .literal .notranslate}]{.pre}
handles, memory keys, and ring buffer addresses into a
[`doca_transport_ctx`{.docutils .literal .notranslate}]{.pre} and
provides [`hololink_launch_unified_dispatch`{.docutils .literal
.notranslate}]{.pre} as the launch function (compiled into
[`libcudaq-realtime-bridge-hololink.so`{.docutils .literal
.notranslate}]{.pre}). A different transport would define its own
context struct and launch function; the dispatcher manages them
identically without any transport-specific knowledge.
:::

::: {#when-to-use-which-mode .section}
### When to Use Which Mode[¶](#when-to-use-which-mode "Permalink to this heading"){.headerlink}

**3-kernel mode** ([`CUDAQ_KERNEL_REGULAR`{.docutils .literal
.notranslate}]{.pre} or [`CUDAQ_KERNEL_COOPERATIVE`{.docutils .literal
.notranslate}]{.pre}):

-   Transport-agnostic -- works with any transport that implements the
    ring-buffer flag protocol

-   Required for cooperative handlers that use [`grid.sync()`{.docutils
    .literal .notranslate}]{.pre}

-   Supports [`CUDAQ_DISPATCH_GRAPH_LAUNCH`{.docutils .literal
    .notranslate}]{.pre} mode

**Unified mode** ([`CUDAQ_KERNEL_UNIFIED`{.docutils .literal
.notranslate}]{.pre}):

-   Lowest latency for regular (non-cooperative) handlers

-   Transport-agnostic API -- the transport provides a pluggable launch
    function and opaque context (e.g., [`HSB`{.docutils .literal
    .notranslate}]{.pre}/[`DOCA`{.docutils .literal .notranslate}]{.pre}
    supplies [`hololink_launch_unified_dispatch`{.docutils .literal
    .notranslate}]{.pre})

-   Single-thread, single-block kernel -- no inter-kernel
    synchronization overhead

-   Not compatible with cooperative handlers or
    [`CUDAQ_DISPATCH_GRAPH_LAUNCH`{.docutils .literal
    .notranslate}]{.pre}
:::

::: {#host-api-extensions .section}
### Host API Extensions[¶](#host-api-extensions "Permalink to this heading"){.headerlink}

::: {.highlight-cpp .notranslate}
::: highlight
    typedef enum {
      CUDAQ_KERNEL_REGULAR     = 0,
      CUDAQ_KERNEL_COOPERATIVE = 1,
      CUDAQ_KERNEL_UNIFIED     = 2
    } cudaq_kernel_type_t;

    typedef void (*cudaq_unified_launch_fn_t)(
        void *transport_ctx,
        cudaq_function_entry_t *function_table, size_t func_count,
        volatile int *shutdown_flag, uint64_t *stats,
        cudaStream_t stream);

    cudaq_status_t cudaq_dispatcher_set_unified_launch(
        cudaq_dispatcher_t *dispatcher,
        cudaq_unified_launch_fn_t unified_launch_fn,
        void *transport_ctx);
:::
:::

When [`kernel_type`{.docutils .literal .notranslate}]{.pre}` `{.docutils
.literal .notranslate}[`==`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`CUDAQ_KERNEL_UNIFIED`{.docutils .literal
.notranslate}]{.pre}:

-   [`cudaq_dispatcher_set_ringbuffer()`{.docutils .literal
    .notranslate}]{.pre} and
    [`cudaq_dispatcher_set_launch_fn()`{.docutils .literal
    .notranslate}]{.pre} are **not required** (the unified kernel
    handles transport internally)

-   [`cudaq_dispatcher_set_unified_launch()`{.docutils .literal
    .notranslate}]{.pre} **must** be called instead

-   [`num_slots`{.docutils .literal .notranslate}]{.pre} and
    [`slot_size`{.docutils .literal .notranslate}]{.pre} in the
    configuration may be zero

-   All other wiring ([`set_function_table`{.docutils .literal
    .notranslate}]{.pre}, [`set_control`{.docutils .literal
    .notranslate}]{.pre}) remains the same
:::

::: {#wiring-example-unified-mode-with-hsb .section}
### Wiring Example (Unified Mode with HSB)[¶](#wiring-example-unified-mode-with-hsb "Permalink to this heading"){.headerlink}

::: {.highlight-cpp .notranslate}
::: highlight
    // Pack DOCA transport handles
    hololink_doca_transport_ctx ctx;
    ctx.gpu_dev_qp     = hololink_get_gpu_dev_qp(transceiver);
    ctx.rx_ring_data   = hololink_get_rx_ring_data_addr(transceiver);
    ctx.rx_ring_stride_sz  = hololink_get_page_size(transceiver);
    ctx.rx_ring_mkey   = htonl(hololink_get_rkey(transceiver));
    ctx.rx_ring_stride_num = hololink_get_num_pages(transceiver);
    ctx.frame_size     = frame_size;

    // Configure dispatcher for unified mode
    cudaq_dispatcher_config_t config{};
    config.device_id       = gpu_id;
    config.kernel_type     = CUDAQ_KERNEL_UNIFIED;
    config.dispatch_mode   = CUDAQ_DISPATCH_DEVICE_CALL;

    cudaq_dispatcher_create(manager, &config, &dispatcher);
    cudaq_dispatcher_set_unified_launch(
        dispatcher, &hololink_launch_unified_dispatch, &ctx);
    cudaq_dispatcher_set_function_table(dispatcher, &table);
    cudaq_dispatcher_set_control(dispatcher, d_shutdown_flag, d_stats);
    cudaq_dispatcher_start(dispatcher);
:::
:::
:::
:::

::: {#what-this-api-does-in-one-paragraph .section}
## What This API Does (In One Paragraph)[¶](#what-this-api-does-in-one-paragraph "Permalink to this heading"){.headerlink}

The host API wires a dispatcher (GPU kernel or CPU thread) to shared
ring buffers. The transport mechanism (e.g., HSB RX/TX kernels,
[`libibverbs`{.docutils .literal .notranslate}]{.pre} threads, or
proprietary transport) places incoming RPC messages into RX slots and
retrieves responses from TX slots. The dispatcher polls RX flags (see
Message completion note), looks up a handler by [`function_id`{.docutils
.literal .notranslate}]{.pre}, executes it on the GPU, and writes a
response into the same slot. The transport's RX/TX components handle
I/O; the dispatch kernel sits in the middle and runs the decoder
handler.
:::

::: {#scope .section}
## Scope[¶](#scope "Permalink to this heading"){.headerlink}

-   C host API in [`cudaq_realtime.h`{.docutils .literal
    .notranslate}]{.pre}

-   RPC messaging protocol (header + payload + response)
:::

::: {#terms-and-components .section}
## Terms and Components[¶](#terms-and-components "Permalink to this heading"){.headerlink}

-   **Ring buffer**: Fixed-size slots holding RPC messages (see Message
    completion note). Each slot has an RX flag and a TX flag.

-   **RX flag**: Nonzero means a slot is ready to be processed.

-   **TX flag**: Nonzero means a response is ready to send.

-   **Dispatcher**: Component that processes RPC messages (GPU kernel or
    CPU thread).

-   **Handler**: Function registered in the function table that
    processes specific message types.

-   **Function table**: Array of handler function pointers + IDs +
    schemas.
:::

::: {#schema-data-structures .section}
## Schema Data Structures[¶](#schema-data-structures "Permalink to this heading"){.headerlink}

Each handler registered in the function table includes a schema that
describes its argument and result types.

::: {#type-descriptors .section}
### Type Descriptors[¶](#type-descriptors "Permalink to this heading"){.headerlink}

::: {.highlight-cpp .notranslate}
::: highlight
    // Standardized payload type identifiers
    typedef enum {
      CUDAQ_TYPE_UINT8           = 0x10,
      CUDAQ_TYPE_INT32           = 0x11,
      CUDAQ_TYPE_INT64           = 0x12,
      CUDAQ_TYPE_FLOAT32         = 0x13,
      CUDAQ_TYPE_FLOAT64         = 0x14,
      CUDAQ_TYPE_ARRAY_UINT8     = 0x20,
      CUDAQ_TYPE_ARRAY_INT32     = 0x21,
      CUDAQ_TYPE_ARRAY_FLOAT32   = 0x22,
      CUDAQ_TYPE_ARRAY_FLOAT64   = 0x23,
      CUDAQ_TYPE_BIT_PACKED      = 0x30   // Bit-packed data (LSB-first)
    } cudaq_payload_type_t;

    struct cudaq_type_desc_t {
      uint8_t  type_id;       // cudaq_payload_type_t value
      uint8_t  reserved[3];
      uint32_t size_bytes;    // Total size in bytes
      uint32_t num_elements;  // Interpretation depends on type_id
    };
:::
:::

The [`num_elements`{.docutils .literal .notranslate}]{.pre} field
interpretation:

-   **Scalar types** ([`CUDAQ_TYPE_UINT8`{.docutils .literal
    .notranslate}]{.pre}, [`CUDAQ_TYPE_INT32`{.docutils .literal
    .notranslate}]{.pre}, etc.): unused, set to 1

-   **Array types** ([`CUDAQ_TYPE_ARRAY_*`{.docutils .literal
    .notranslate}]{.pre}): number of array elements

-   **CUDAQ_TYPE_BIT_PACKED**: number of bits (not bytes)
:::

::: {#handler-schema .section}
### Handler Schema[¶](#handler-schema "Permalink to this heading"){.headerlink}

::: {.highlight-cpp .notranslate}
::: highlight
    struct cudaq_handler_schema_t {
      uint8_t  num_args;              // Number of input arguments
      uint8_t  num_results;           // Number of return values
      uint16_t reserved;

      cudaq_type_desc_t args[8];      // Argument type descriptors
      cudaq_type_desc_t results[4];   // Result type descriptors
    };
:::
:::

Limits:

-   Maximum 8 arguments per handler

-   Maximum 4 results per handler

-   Total payload size must fit in slot: [`slot_size`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`-`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`sizeof(RPCHeader)`{.docutils .literal
    .notranslate}]{.pre}
:::
:::

::: {#rpc-messaging-protocol .section}
## RPC Messaging Protocol[¶](#rpc-messaging-protocol "Permalink to this heading"){.headerlink}

Each RX ring buffer slot contains an RPC request. The dispatcher writes
the response to the corresponding TX ring buffer slot.

::: {.highlight-text .notranslate}
::: highlight
    RX Slot: | RPCHeader | request payload bytes |
    TX Slot: | RPCResponse | response payload bytes |
:::
:::

Payload encoding details (type system, multi-argument encoding,
bit-packing, and QEC-specific examples) are defined in documentation for
the CUDA-Q Realtime Messaging Protocol.

Magic values (little-endian 32-bit):

-   [`RPC_MAGIC_REQUEST`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`=`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`0x43555152`{.docutils .literal .notranslate}]{.pre}
    ([`'CUQR'`{.docutils .literal .notranslate}]{.pre})

-   [`RPC_MAGIC_RESPONSE`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`=`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`0x43555153`{.docutils .literal .notranslate}]{.pre}
    ([`'CUQS'`{.docutils .literal .notranslate}]{.pre})

::: {.highlight-cpp .notranslate}
::: highlight
    // Wire format (byte layout must match dispatch_kernel_launch.h)
    struct RPCHeader {
      uint32_t magic;          // RPC_MAGIC_REQUEST
      uint32_t function_id;    // fnv1a_hash("handler_name")
      uint32_t arg_len;        // payload bytes following this header
      uint32_t request_id;     // caller-assigned ID, echoed in the response
      uint64_t ptp_timestamp;  // PTP send timestamp (set by sender; 0 if unused)
    };

    struct RPCResponse {
      uint32_t magic;          // RPC_MAGIC_RESPONSE
      int32_t  status;         // 0 = success
      uint32_t result_len;     // bytes of response payload
      uint32_t request_id;     // echoed from RPCHeader::request_id
      uint64_t ptp_timestamp;  // echoed from RPCHeader::ptp_timestamp
    };
:::
:::

Both structs are 24 bytes, packed with no padding. See
[`cudaq_realtime_message_protocol.bs`{.docutils .literal
.notranslate}]{.pre} for [`request_id`{.docutils .literal
.notranslate}]{.pre} and [`ptp_timestamp`{.docutils .literal
.notranslate}]{.pre} semantics.

Payload conventions:

-   **Request payload**: argument data as specified by handler schema.

-   **Response payload**: result data as specified by handler schema.

-   **Size limit**: payload must fit in one slot.
    [`max_payload_bytes`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`=`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`slot_size`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`-`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`sizeof(RPCHeader)`{.docutils .literal
    .notranslate}]{.pre}.

-   **Multi-argument encoding**: arguments concatenated in schema order
    (see message protocol doc).
:::

::: {#host-api-overview .section}
## Host API Overview[¶](#host-api-overview "Permalink to this heading"){.headerlink}

Header:
[`realtime/include/cudaq/realtime/daemon/dispatcher/cudaq_realtime.h`{.docutils
.literal .notranslate}]{.pre}
:::

::: {#manager-and-dispatcher-topology .section}
## Manager and Dispatcher Topology[¶](#manager-and-dispatcher-topology "Permalink to this heading"){.headerlink}

The manager is a lightweight owner for one or more dispatchers. Each
dispatcher is configured independently (e.g., [`vp_id`{.docutils
.literal .notranslate}]{.pre}, [`kernel_type`{.docutils .literal
.notranslate}]{.pre}, [`dispatch_mode`{.docutils .literal
.notranslate}]{.pre}) and can target different workloads.

![Manager and dispatcher
topology](data:image/svg+xml;base64,PHN2ZyBpZD0ibWVybWFpZC1zdmciIHdpZHRoPSIxMDAlIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGNsYXNzPSJmbG93Y2hhcnQiIHN0eWxlPSJtYXgtd2lkdGg6IDExMjFweDsiIHZpZXdCb3g9IjAgMCAxMTIxIDU0OCIgcm9sZT0iZ3JhcGhpY3MtZG9jdW1lbnQgZG9jdW1lbnQiIGFyaWEtcm9sZWRlc2NyaXB0aW9uPSJmbG93Y2hhcnQtdjIiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48c3R5bGUgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiPkBpbXBvcnQgdXJsKCJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9mb250LWF3ZXNvbWUvNi43LjIvY3NzL2FsbC5taW4uY3NzIik7PC9zdHlsZT48c3R5bGU+I21lcm1haWQtc3Zne2ZvbnQtZmFtaWx5OiJ0cmVidWNoZXQgbXMiLHZlcmRhbmEsYXJpYWwsc2Fucy1zZXJpZjtmb250LXNpemU6MTZweDtmaWxsOiMzMzM7fUBrZXlmcmFtZXMgZWRnZS1hbmltYXRpb24tZnJhbWV7ZnJvbXtzdHJva2UtZGFzaG9mZnNldDowO319QGtleWZyYW1lcyBkYXNoe3Rve3N0cm9rZS1kYXNob2Zmc2V0OjA7fX0jbWVybWFpZC1zdmcgLmVkZ2UtYW5pbWF0aW9uLXNsb3d7c3Ryb2tlLWRhc2hhcnJheTo5LDUhaW1wb3J0YW50O3N0cm9rZS1kYXNob2Zmc2V0OjkwMDthbmltYXRpb246ZGFzaCA1MHMgbGluZWFyIGluZmluaXRlO3N0cm9rZS1saW5lY2FwOnJvdW5kO30jbWVybWFpZC1zdmcgLmVkZ2UtYW5pbWF0aW9uLWZhc3R7c3Ryb2tlLWRhc2hhcnJheTo5LDUhaW1wb3J0YW50O3N0cm9rZS1kYXNob2Zmc2V0OjkwMDthbmltYXRpb246ZGFzaCAyMHMgbGluZWFyIGluZmluaXRlO3N0cm9rZS1saW5lY2FwOnJvdW5kO30jbWVybWFpZC1zdmcgLmVycm9yLWljb257ZmlsbDojNTUyMjIyO30jbWVybWFpZC1zdmcgLmVycm9yLXRleHR7ZmlsbDojNTUyMjIyO3N0cm9rZTojNTUyMjIyO30jbWVybWFpZC1zdmcgLmVkZ2UtdGhpY2tuZXNzLW5vcm1hbHtzdHJva2Utd2lkdGg6MXB4O30jbWVybWFpZC1zdmcgLmVkZ2UtdGhpY2tuZXNzLXRoaWNre3N0cm9rZS13aWR0aDozLjVweDt9I21lcm1haWQtc3ZnIC5lZGdlLXBhdHRlcm4tc29saWR7c3Ryb2tlLWRhc2hhcnJheTowO30jbWVybWFpZC1zdmcgLmVkZ2UtdGhpY2tuZXNzLWludmlzaWJsZXtzdHJva2Utd2lkdGg6MDtmaWxsOm5vbmU7fSNtZXJtYWlkLXN2ZyAuZWRnZS1wYXR0ZXJuLWRhc2hlZHtzdHJva2UtZGFzaGFycmF5OjM7fSNtZXJtYWlkLXN2ZyAuZWRnZS1wYXR0ZXJuLWRvdHRlZHtzdHJva2UtZGFzaGFycmF5OjI7fSNtZXJtYWlkLXN2ZyAubWFya2Vye2ZpbGw6IzMzMzMzMztzdHJva2U6IzMzMzMzMzt9I21lcm1haWQtc3ZnIC5tYXJrZXIuY3Jvc3N7c3Ryb2tlOiMzMzMzMzM7fSNtZXJtYWlkLXN2ZyBzdmd7Zm9udC1mYW1pbHk6InRyZWJ1Y2hldCBtcyIsdmVyZGFuYSxhcmlhbCxzYW5zLXNlcmlmO2ZvbnQtc2l6ZToxNnB4O30jbWVybWFpZC1zdmcgcHttYXJnaW46MDt9I21lcm1haWQtc3ZnIC5sYWJlbHtmb250LWZhbWlseToidHJlYnVjaGV0IG1zIix2ZXJkYW5hLGFyaWFsLHNhbnMtc2VyaWY7Y29sb3I6IzMzMzt9I21lcm1haWQtc3ZnIC5jbHVzdGVyLWxhYmVsIHRleHR7ZmlsbDojMzMzO30jbWVybWFpZC1zdmcgLmNsdXN0ZXItbGFiZWwgc3Bhbntjb2xvcjojMzMzO30jbWVybWFpZC1zdmcgLmNsdXN0ZXItbGFiZWwgc3BhbiBwe2JhY2tncm91bmQtY29sb3I6dHJhbnNwYXJlbnQ7fSNtZXJtYWlkLXN2ZyAubGFiZWwgdGV4dCwjbWVybWFpZC1zdmcgc3BhbntmaWxsOiMzMzM7Y29sb3I6IzMzMzt9I21lcm1haWQtc3ZnIC5ub2RlIHJlY3QsI21lcm1haWQtc3ZnIC5ub2RlIGNpcmNsZSwjbWVybWFpZC1zdmcgLm5vZGUgZWxsaXBzZSwjbWVybWFpZC1zdmcgLm5vZGUgcG9seWdvbiwjbWVybWFpZC1zdmcgLm5vZGUgcGF0aHtmaWxsOiNFQ0VDRkY7c3Ryb2tlOiM5MzcwREI7c3Ryb2tlLXdpZHRoOjFweDt9I21lcm1haWQtc3ZnIC5yb3VnaC1ub2RlIC5sYWJlbCB0ZXh0LCNtZXJtYWlkLXN2ZyAubm9kZSAubGFiZWwgdGV4dCwjbWVybWFpZC1zdmcgLmltYWdlLXNoYXBlIC5sYWJlbCwjbWVybWFpZC1zdmcgLmljb24tc2hhcGUgLmxhYmVse3RleHQtYW5jaG9yOm1pZGRsZTt9I21lcm1haWQtc3ZnIC5ub2RlIC5rYXRleCBwYXRoe2ZpbGw6IzAwMDtzdHJva2U6IzAwMDtzdHJva2Utd2lkdGg6MXB4O30jbWVybWFpZC1zdmcgLnJvdWdoLW5vZGUgLmxhYmVsLCNtZXJtYWlkLXN2ZyAubm9kZSAubGFiZWwsI21lcm1haWQtc3ZnIC5pbWFnZS1zaGFwZSAubGFiZWwsI21lcm1haWQtc3ZnIC5pY29uLXNoYXBlIC5sYWJlbHt0ZXh0LWFsaWduOmNlbnRlcjt9I21lcm1haWQtc3ZnIC5ub2RlLmNsaWNrYWJsZXtjdXJzb3I6cG9pbnRlcjt9I21lcm1haWQtc3ZnIC5yb290IC5hbmNob3IgcGF0aHtmaWxsOiMzMzMzMzMhaW1wb3J0YW50O3N0cm9rZS13aWR0aDowO3N0cm9rZTojMzMzMzMzO30jbWVybWFpZC1zdmcgLmFycm93aGVhZFBhdGh7ZmlsbDojMzMzMzMzO30jbWVybWFpZC1zdmcgLmVkZ2VQYXRoIC5wYXRoe3N0cm9rZTojMzMzMzMzO3N0cm9rZS13aWR0aDoyLjBweDt9I21lcm1haWQtc3ZnIC5mbG93Y2hhcnQtbGlua3tzdHJva2U6IzMzMzMzMztmaWxsOm5vbmU7fSNtZXJtYWlkLXN2ZyAuZWRnZUxhYmVse2JhY2tncm91bmQtY29sb3I6cmdiYSgyMzIsMjMyLDIzMiwgMC44KTt0ZXh0LWFsaWduOmNlbnRlcjt9I21lcm1haWQtc3ZnIC5lZGdlTGFiZWwgcHtiYWNrZ3JvdW5kLWNvbG9yOnJnYmEoMjMyLDIzMiwyMzIsIDAuOCk7fSNtZXJtYWlkLXN2ZyAuZWRnZUxhYmVsIHJlY3R7b3BhY2l0eTowLjU7YmFja2dyb3VuZC1jb2xvcjpyZ2JhKDIzMiwyMzIsMjMyLCAwLjgpO2ZpbGw6cmdiYSgyMzIsMjMyLDIzMiwgMC44KTt9I21lcm1haWQtc3ZnIC5sYWJlbEJrZ3tiYWNrZ3JvdW5kLWNvbG9yOnJnYmEoMjMyLCAyMzIsIDIzMiwgMC41KTt9I21lcm1haWQtc3ZnIC5jbHVzdGVyIHJlY3R7ZmlsbDojZmZmZmRlO3N0cm9rZTojYWFhYTMzO3N0cm9rZS13aWR0aDoxcHg7fSNtZXJtYWlkLXN2ZyAuY2x1c3RlciB0ZXh0e2ZpbGw6IzMzMzt9I21lcm1haWQtc3ZnIC5jbHVzdGVyIHNwYW57Y29sb3I6IzMzMzt9I21lcm1haWQtc3ZnIGRpdi5tZXJtYWlkVG9vbHRpcHtwb3NpdGlvbjphYnNvbHV0ZTt0ZXh0LWFsaWduOmNlbnRlcjttYXgtd2lkdGg6MjAwcHg7cGFkZGluZzoycHg7Zm9udC1mYW1pbHk6InRyZWJ1Y2hldCBtcyIsdmVyZGFuYSxhcmlhbCxzYW5zLXNlcmlmO2ZvbnQtc2l6ZToxMnB4O2JhY2tncm91bmQ6aHNsKDgwLCAxMDAlLCA5Ni4yNzQ1MDk4MDM5JSk7Ym9yZGVyOjFweCBzb2xpZCAjYWFhYTMzO2JvcmRlci1yYWRpdXM6MnB4O3BvaW50ZXItZXZlbnRzOm5vbmU7ei1pbmRleDoxMDA7fSNtZXJtYWlkLXN2ZyAuZmxvd2NoYXJ0VGl0bGVUZXh0e3RleHQtYW5jaG9yOm1pZGRsZTtmb250LXNpemU6MThweDtmaWxsOiMzMzM7fSNtZXJtYWlkLXN2ZyByZWN0LnRleHR7ZmlsbDpub25lO3N0cm9rZS13aWR0aDowO30jbWVybWFpZC1zdmcgLmljb24tc2hhcGUsI21lcm1haWQtc3ZnIC5pbWFnZS1zaGFwZXtiYWNrZ3JvdW5kLWNvbG9yOnJnYmEoMjMyLDIzMiwyMzIsIDAuOCk7dGV4dC1hbGlnbjpjZW50ZXI7fSNtZXJtYWlkLXN2ZyAuaWNvbi1zaGFwZSBwLCNtZXJtYWlkLXN2ZyAuaW1hZ2Utc2hhcGUgcHtiYWNrZ3JvdW5kLWNvbG9yOnJnYmEoMjMyLDIzMiwyMzIsIDAuOCk7cGFkZGluZzoycHg7fSNtZXJtYWlkLXN2ZyAuaWNvbi1zaGFwZSByZWN0LCNtZXJtYWlkLXN2ZyAuaW1hZ2Utc2hhcGUgcmVjdHtvcGFjaXR5OjAuNTtiYWNrZ3JvdW5kLWNvbG9yOnJnYmEoMjMyLDIzMiwyMzIsIDAuOCk7ZmlsbDpyZ2JhKDIzMiwyMzIsMjMyLCAwLjgpO30jbWVybWFpZC1zdmcgLmxhYmVsLWljb257ZGlzcGxheTppbmxpbmUtYmxvY2s7aGVpZ2h0OjFlbTtvdmVyZmxvdzp2aXNpYmxlO3ZlcnRpY2FsLWFsaWduOi0wLjEyNWVtO30jbWVybWFpZC1zdmcgLm5vZGUgLmxhYmVsLWljb24gcGF0aHtmaWxsOmN1cnJlbnRDb2xvcjtzdHJva2U6cmV2ZXJ0O3N0cm9rZS13aWR0aDpyZXZlcnQ7fSNtZXJtYWlkLXN2ZyA6cm9vdHstLW1lcm1haWQtZm9udC1mYW1pbHk6InRyZWJ1Y2hldCBtcyIsdmVyZGFuYSxhcmlhbCxzYW5zLXNlcmlmO308L3N0eWxlPjxnPjxtYXJrZXIgaWQ9Im1lcm1haWQtc3ZnX2Zsb3djaGFydC12Mi1wb2ludEVuZCIgY2xhc3M9Im1hcmtlciBmbG93Y2hhcnQtdjIiIHZpZXdCb3g9IjAgMCAxMCAxMCIgcmVmWD0iNSIgcmVmWT0iNSIgbWFya2VyVW5pdHM9InVzZXJTcGFjZU9uVXNlIiBtYXJrZXJXaWR0aD0iOCIgbWFya2VySGVpZ2h0PSI4IiBvcmllbnQ9ImF1dG8iPjxwYXRoIGQ9Ik0gMCAwIEwgMTAgNSBMIDAgMTAgeiIgY2xhc3M9ImFycm93TWFya2VyUGF0aCIgc3R5bGU9InN0cm9rZS13aWR0aDogMTsgc3Ryb2tlLWRhc2hhcnJheTogMSwgMDsiLz48L21hcmtlcj48bWFya2VyIGlkPSJtZXJtYWlkLXN2Z19mbG93Y2hhcnQtdjItcG9pbnRTdGFydCIgY2xhc3M9Im1hcmtlciBmbG93Y2hhcnQtdjIiIHZpZXdCb3g9IjAgMCAxMCAxMCIgcmVmWD0iNC41IiByZWZZPSI1IiBtYXJrZXJVbml0cz0idXNlclNwYWNlT25Vc2UiIG1hcmtlcldpZHRoPSI4IiBtYXJrZXJIZWlnaHQ9IjgiIG9yaWVudD0iYXV0byI+PHBhdGggZD0iTSAwIDUgTCAxMCAxMCBMIDEwIDAgeiIgY2xhc3M9ImFycm93TWFya2VyUGF0aCIgc3R5bGU9InN0cm9rZS13aWR0aDogMTsgc3Ryb2tlLWRhc2hhcnJheTogMSwgMDsiLz48L21hcmtlcj48bWFya2VyIGlkPSJtZXJtYWlkLXN2Z19mbG93Y2hhcnQtdjItY2lyY2xlRW5kIiBjbGFzcz0ibWFya2VyIGZsb3djaGFydC12MiIgdmlld0JveD0iMCAwIDEwIDEwIiByZWZYPSIxMSIgcmVmWT0iNSIgbWFya2VyVW5pdHM9InVzZXJTcGFjZU9uVXNlIiBtYXJrZXJXaWR0aD0iMTEiIG1hcmtlckhlaWdodD0iMTEiIG9yaWVudD0iYXV0byI+PGNpcmNsZSBjeD0iNSIgY3k9IjUiIHI9IjUiIGNsYXNzPSJhcnJvd01hcmtlclBhdGgiIHN0eWxlPSJzdHJva2Utd2lkdGg6IDE7IHN0cm9rZS1kYXNoYXJyYXk6IDEsIDA7Ii8+PC9tYXJrZXI+PG1hcmtlciBpZD0ibWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLWNpcmNsZVN0YXJ0IiBjbGFzcz0ibWFya2VyIGZsb3djaGFydC12MiIgdmlld0JveD0iMCAwIDEwIDEwIiByZWZYPSItMSIgcmVmWT0iNSIgbWFya2VyVW5pdHM9InVzZXJTcGFjZU9uVXNlIiBtYXJrZXJXaWR0aD0iMTEiIG1hcmtlckhlaWdodD0iMTEiIG9yaWVudD0iYXV0byI+PGNpcmNsZSBjeD0iNSIgY3k9IjUiIHI9IjUiIGNsYXNzPSJhcnJvd01hcmtlclBhdGgiIHN0eWxlPSJzdHJva2Utd2lkdGg6IDE7IHN0cm9rZS1kYXNoYXJyYXk6IDEsIDA7Ii8+PC9tYXJrZXI+PG1hcmtlciBpZD0ibWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLWNyb3NzRW5kIiBjbGFzcz0ibWFya2VyIGNyb3NzIGZsb3djaGFydC12MiIgdmlld0JveD0iMCAwIDExIDExIiByZWZYPSIxMiIgcmVmWT0iNS4yIiBtYXJrZXJVbml0cz0idXNlclNwYWNlT25Vc2UiIG1hcmtlcldpZHRoPSIxMSIgbWFya2VySGVpZ2h0PSIxMSIgb3JpZW50PSJhdXRvIj48cGF0aCBkPSJNIDEsMSBsIDksOSBNIDEwLDEgbCAtOSw5IiBjbGFzcz0iYXJyb3dNYXJrZXJQYXRoIiBzdHlsZT0ic3Ryb2tlLXdpZHRoOiAyOyBzdHJva2UtZGFzaGFycmF5OiAxLCAwOyIvPjwvbWFya2VyPjxtYXJrZXIgaWQ9Im1lcm1haWQtc3ZnX2Zsb3djaGFydC12Mi1jcm9zc1N0YXJ0IiBjbGFzcz0ibWFya2VyIGNyb3NzIGZsb3djaGFydC12MiIgdmlld0JveD0iMCAwIDExIDExIiByZWZYPSItMSIgcmVmWT0iNS4yIiBtYXJrZXJVbml0cz0idXNlclNwYWNlT25Vc2UiIG1hcmtlcldpZHRoPSIxMSIgbWFya2VySGVpZ2h0PSIxMSIgb3JpZW50PSJhdXRvIj48cGF0aCBkPSJNIDEsMSBsIDksOSBNIDEwLDEgbCAtOSw5IiBjbGFzcz0iYXJyb3dNYXJrZXJQYXRoIiBzdHlsZT0ic3Ryb2tlLXdpZHRoOiAyOyBzdHJva2UtZGFzaGFycmF5OiAxLCAwOyIvPjwvbWFya2VyPjxnIGNsYXNzPSJyb290Ij48ZyBjbGFzcz0iY2x1c3RlcnMiLz48ZyBjbGFzcz0iZWRnZVBhdGhzIj48cGF0aCBkPSJNNDMwLjUsODQuNjYyTDM4OCw5My4wNTJDMzQ1LjUsMTAxLjQ0MiwyNjAuNSwxMTguMjIxLDIxOCwxMzAuMTFDMTc1LjUsMTQyLDE3NS41LDE0OSwxNzUuNSwxNTIuNUwxNzUuNSwxNTYiIGlkPSJMX01HUl9EMF8wIiBjbGFzcz0iIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBmbG93Y2hhcnQtbGluayIgc3R5bGU9IjsiIGRhdGEtZWRnZT0idHJ1ZSIgZGF0YS1ldD0iZWRnZSIgZGF0YS1pZD0iTF9NR1JfRDBfMCIgZGF0YS1wb2ludHM9Ilczc2llQ0k2TkRNd0xqVXNJbmtpT2pnMExqWTJNak16TnpZMk1qTXpOelkyZlN4N0luZ2lPakUzTlM0MUxDSjVJam94TXpWOUxIc2llQ0k2TVRjMUxqVXNJbmtpT2pFMk1IMWQiIG1hcmtlci1lbmQ9InVybCgjbWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLXBvaW50RW5kKSIvPjxwYXRoIGQ9Ik01NjAuNSwxMTBMNTYwLjUsMTE0LjE2N0M1NjAuNSwxMTguMzMzLDU2MC41LDEyNi42NjcsNTYwLjUsMTM0LjMzM0M1NjAuNSwxNDIsNTYwLjUsMTQ5LDU2MC41LDE1Mi41TDU2MC41LDE1NiIgaWQ9IkxfTUdSX0QxXzAiIGNsYXNzPSIgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBlZGdlLXRoaWNrbmVzcy1ub3JtYWwgZWRnZS1wYXR0ZXJuLXNvbGlkIGZsb3djaGFydC1saW5rIiBzdHlsZT0iOyIgZGF0YS1lZGdlPSJ0cnVlIiBkYXRhLWV0PSJlZGdlIiBkYXRhLWlkPSJMX01HUl9EMV8wIiBkYXRhLXBvaW50cz0iVzNzaWVDSTZOVFl3TGpVc0lua2lPakV4TUgwc2V5SjRJam8xTmpBdU5Td2llU0k2TVRNMWZTeDdJbmdpT2pVMk1DNDFMQ0o1SWpveE5qQjlYUT09IiBtYXJrZXItZW5kPSJ1cmwoI21lcm1haWQtc3ZnX2Zsb3djaGFydC12Mi1wb2ludEVuZCkiLz48cGF0aCBkPSJNNjkwLjUsODQuNjYyTDczMyw5My4wNTJDNzc1LjUsMTAxLjQ0Miw4NjAuNSwxMTguMjIxLDkwMywxMzAuMTFDOTQ1LjUsMTQyLDk0NS41LDE0OSw5NDUuNSwxNTIuNUw5NDUuNSwxNTYiIGlkPSJMX01HUl9ETl8wIiBjbGFzcz0iIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBmbG93Y2hhcnQtbGluayIgc3R5bGU9IjsiIGRhdGEtZWRnZT0idHJ1ZSIgZGF0YS1ldD0iZWRnZSIgZGF0YS1pZD0iTF9NR1JfRE5fMCIgZGF0YS1wb2ludHM9Ilczc2llQ0k2Tmprd0xqVXNJbmtpT2pnMExqWTJNak16TnpZMk1qTXpOelkyZlN4N0luZ2lPamswTlM0MUxDSjVJam94TXpWOUxIc2llQ0k2T1RRMUxqVXNJbmtpT2pFMk1IMWQiIG1hcmtlci1lbmQ9InVybCgjbWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLXBvaW50RW5kKSIvPjxwYXRoIGQ9Ik0xNzUuNSwyMTRMMTc1LjUsMjE4LjE2N0MxNzUuNSwyMjIuMzMzLDE3NS41LDIzMC42NjcsMTc1LjUsMjM4LjMzM0MxNzUuNSwyNDYsMTc1LjUsMjUzLDE3NS41LDI1Ni41TDE3NS41LDI2MCIgaWQ9IkxfRDBfRDBfQ0ZHXzAiIGNsYXNzPSIgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBlZGdlLXRoaWNrbmVzcy1ub3JtYWwgZWRnZS1wYXR0ZXJuLXNvbGlkIGZsb3djaGFydC1saW5rIiBzdHlsZT0iOyIgZGF0YS1lZGdlPSJ0cnVlIiBkYXRhLWV0PSJlZGdlIiBkYXRhLWlkPSJMX0QwX0QwX0NGR18wIiBkYXRhLXBvaW50cz0iVzNzaWVDSTZNVGMxTGpVc0lua2lPakl4Tkgwc2V5SjRJam94TnpVdU5Td2llU0k2TWpNNWZTeDdJbmdpT2pFM05TNDFMQ0o1SWpveU5qUjlYUT09IiBtYXJrZXItZW5kPSJ1cmwoI21lcm1haWQtc3ZnX2Zsb3djaGFydC12Mi1wb2ludEVuZCkiLz48cGF0aCBkPSJNNTYwLjUsMjE0TDU2MC41LDIxOC4xNjdDNTYwLjUsMjIyLjMzMyw1NjAuNSwyMzAuNjY3LDU2MC41LDIzOC4zMzNDNTYwLjUsMjQ2LDU2MC41LDI1Myw1NjAuNSwyNTYuNUw1NjAuNSwyNjAiIGlkPSJMX0QxX0QxX0NGR18wIiBjbGFzcz0iIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZWRnZS10aGlja25lc3Mtbm9ybWFsIGVkZ2UtcGF0dGVybi1zb2xpZCBmbG93Y2hhcnQtbGluayIgc3R5bGU9IjsiIGRhdGEtZWRnZT0idHJ1ZSIgZGF0YS1ldD0iZWRnZSIgZGF0YS1pZD0iTF9EMV9EMV9DRkdfMCIgZGF0YS1wb2ludHM9Ilczc2llQ0k2TlRZd0xqVXNJbmtpT2pJeE5IMHNleUo0SWpvMU5qQXVOU3dpZVNJNk1qTTVmU3g3SW5naU9qVTJNQzQxTENKNUlqb3lOalI5WFE9PSIgbWFya2VyLWVuZD0idXJsKCNtZXJtYWlkLXN2Z19mbG93Y2hhcnQtdjItcG9pbnRFbmQpIi8+PHBhdGggZD0iTTk0NS41LDIxNEw5NDUuNSwyMTguMTY3Qzk0NS41LDIyMi4zMzMsOTQ1LjUsMjMwLjY2Nyw5NDUuNSwyMzguMzMzQzk0NS41LDI0Niw5NDUuNSwyNTMsOTQ1LjUsMjU2LjVMOTQ1LjUsMjYwIiBpZD0iTF9ETl9ETl9DRkdfMCIgY2xhc3M9IiBlZGdlLXRoaWNrbmVzcy1ub3JtYWwgZWRnZS1wYXR0ZXJuLXNvbGlkIGVkZ2UtdGhpY2tuZXNzLW5vcm1hbCBlZGdlLXBhdHRlcm4tc29saWQgZmxvd2NoYXJ0LWxpbmsiIHN0eWxlPSI7IiBkYXRhLWVkZ2U9InRydWUiIGRhdGEtZXQ9ImVkZ2UiIGRhdGEtaWQ9IkxfRE5fRE5fQ0ZHXzAiIGRhdGEtcG9pbnRzPSJXM3NpZUNJNk9UUTFMalVzSW5raU9qSXhOSDBzZXlKNElqbzVORFV1TlN3aWVTSTZNak01ZlN4N0luZ2lPamswTlM0MUxDSjVJam95TmpSOVhRPT0iIG1hcmtlci1lbmQ9InVybCgjbWVybWFpZC1zdmdfZmxvd2NoYXJ0LXYyLXBvaW50RW5kKSIvPjwvZz48ZyBjbGFzcz0iZWRnZUxhYmVscyI+PGcgY2xhc3M9ImVkZ2VMYWJlbCI+PGcgY2xhc3M9ImxhYmVsIiBkYXRhLWlkPSJMX01HUl9EMF8wIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLCAwKSI+PGZvcmVpZ25PYmplY3Qgd2lkdGg9IjAiIGhlaWdodD0iMCI+PGRpdiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCIgY2xhc3M9ImxhYmVsQmtnIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9ImVkZ2VMYWJlbCAiPjwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0iZWRnZUxhYmVsIj48ZyBjbGFzcz0ibGFiZWwiIGRhdGEtaWQ9IkxfTUdSX0QxXzAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsIDApIj48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMCIgaGVpZ2h0PSIwIj48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBjbGFzcz0ibGFiZWxCa2ciIHN0eWxlPSJkaXNwbGF5OiB0YWJsZS1jZWxsOyB3aGl0ZS1zcGFjZTogbm93cmFwOyBsaW5lLWhlaWdodDogMS41OyBtYXgtd2lkdGg6IDIwMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7Ij48c3BhbiBjbGFzcz0iZWRnZUxhYmVsICI+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJlZGdlTGFiZWwiPjxnIGNsYXNzPSJsYWJlbCIgZGF0YS1pZD0iTF9NR1JfRE5fMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwgMCkiPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIwIiBoZWlnaHQ9IjAiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIGNsYXNzPSJsYWJlbEJrZyIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlLWNlbGw7IHdoaXRlLXNwYWNlOiBub3dyYXA7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsiPjxzcGFuIGNsYXNzPSJlZGdlTGFiZWwgIj48L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PGcgY2xhc3M9ImVkZ2VMYWJlbCI+PGcgY2xhc3M9ImxhYmVsIiBkYXRhLWlkPSJMX0QwX0QwX0NGR18wIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLCAwKSI+PGZvcmVpZ25PYmplY3Qgd2lkdGg9IjAiIGhlaWdodD0iMCI+PGRpdiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCIgY2xhc3M9ImxhYmVsQmtnIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9ImVkZ2VMYWJlbCAiPjwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0iZWRnZUxhYmVsIj48ZyBjbGFzcz0ibGFiZWwiIGRhdGEtaWQ9IkxfRDFfRDFfQ0ZHXzAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsIDApIj48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMCIgaGVpZ2h0PSIwIj48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBjbGFzcz0ibGFiZWxCa2ciIHN0eWxlPSJkaXNwbGF5OiB0YWJsZS1jZWxsOyB3aGl0ZS1zcGFjZTogbm93cmFwOyBsaW5lLWhlaWdodDogMS41OyBtYXgtd2lkdGg6IDIwMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7Ij48c3BhbiBjbGFzcz0iZWRnZUxhYmVsICI+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJlZGdlTGFiZWwiPjxnIGNsYXNzPSJsYWJlbCIgZGF0YS1pZD0iTF9ETl9ETl9DRkdfMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMCwgMCkiPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIwIiBoZWlnaHQ9IjAiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIGNsYXNzPSJsYWJlbEJrZyIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlLWNlbGw7IHdoaXRlLXNwYWNlOiBub3dyYXA7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsiPjxzcGFuIGNsYXNzPSJlZGdlTGFiZWwgIj48L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PC9nPjxnIGNsYXNzPSJub2RlcyI+PGcgY2xhc3M9InJvb3QiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDc3MCwgMjU2KSI+PGcgY2xhc3M9ImNsdXN0ZXJzIj48ZyBjbGFzcz0iY2x1c3RlciAiIGlkPSJETl9DRkciIGRhdGEtbG9vaz0iY2xhc3NpYyI+PHJlY3Qgc3R5bGU9IiIgeD0iOCIgeT0iOCIgd2lkdGg9IjMzNSIgaGVpZ2h0PSIyNzYiLz48ZyBjbGFzcz0iY2x1c3Rlci1sYWJlbCAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDk4LjU3ODEyNSwgOCkiPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIxNTMuODQzNzUiIGhlaWdodD0iMjQiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIHN0eWxlPSJkaXNwbGF5OiB0YWJsZS1jZWxsOyB3aGl0ZS1zcGFjZTogbm93cmFwOyBsaW5lLWhlaWdodDogMS41OyBtYXgtd2lkdGg6IDIwMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7Ij48c3BhbiBjbGFzcz0ibm9kZUxhYmVsICI+PHA+RGlzcGF0Y2hlciBOLTEgY29uZmlnPC9wPjwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48L2c+PGcgY2xhc3M9ImVkZ2VQYXRocyIvPjxnIGNsYXNzPSJlZGdlTGFiZWxzIi8+PGcgY2xhc3M9Im5vZGVzIj48ZyBjbGFzcz0ibm9kZSBkZWZhdWx0ICAiIGlkPSJmbG93Y2hhcnQtRE5BLTExIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNzUuNSwgODIpIj48cmVjdCBjbGFzcz0iYmFzaWMgbGFiZWwtY29udGFpbmVyIiBzdHlsZT0iIiB4PSItMTMwIiB5PSItMzkiIHdpZHRoPSIyNjAiIGhlaWdodD0iNzgiLz48ZyBjbGFzcz0ibGFiZWwiIHN0eWxlPSIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC0xMDAsIC0yNCkiPjxyZWN0Lz48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjQ4Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGU7IHdoaXRlLXNwYWNlOiBicmVhay1zcGFjZXM7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsgd2lkdGg6IDIwMHB4OyI+PHNwYW4gY2xhc3M9Im5vZGVMYWJlbCAiPjxwPktlcm5lbDogQ29vcGVyYXRpdmUgb3IgUmVndWxhcjwvcD48L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PGcgY2xhc3M9Im5vZGUgZGVmYXVsdCAgIiBpZD0iZmxvd2NoYXJ0LUROQi0xMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTc1LjUsIDIxMCkiPjxyZWN0IGNsYXNzPSJiYXNpYyBsYWJlbC1jb250YWluZXIiIHN0eWxlPSIiIHg9Ii0xMzAiIHk9Ii0zOSIgd2lkdGg9IjI2MCIgaGVpZ2h0PSI3OCIvPjxnIGNsYXNzPSJsYWJlbCIgc3R5bGU9IiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEwMCwgLTI0KSI+PHJlY3QvPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iNDgiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIHN0eWxlPSJkaXNwbGF5OiB0YWJsZTsgd2hpdGUtc3BhY2U6IGJyZWFrLXNwYWNlczsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyB3aWR0aDogMjAwcHg7Ij48c3BhbiBjbGFzcz0ibm9kZUxhYmVsICI+PHA+RGlzcGF0Y2g6IERldmljZUNhbGwgb3IgR3JhcGhMYXVuY2g8L3A+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjwvZz48L2c+PGcgY2xhc3M9InJvb3QiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM4NSwgMjU2KSI+PGcgY2xhc3M9ImNsdXN0ZXJzIj48ZyBjbGFzcz0iY2x1c3RlciAiIGlkPSJEMV9DRkciIGRhdGEtbG9vaz0iY2xhc3NpYyI+PHJlY3Qgc3R5bGU9IiIgeD0iOCIgeT0iOCIgd2lkdGg9IjMzNSIgaGVpZ2h0PSIyNzYiLz48ZyBjbGFzcz0iY2x1c3Rlci1sYWJlbCAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEwNy4wMTU2MjUsIDgpIj48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMTM2Ljk2ODc1IiBoZWlnaHQ9IjI0Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9Im5vZGVMYWJlbCAiPjxwPkRpc3BhdGNoZXIgMSBjb25maWc8L3A+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjwvZz48ZyBjbGFzcz0iZWRnZVBhdGhzIi8+PGcgY2xhc3M9ImVkZ2VMYWJlbHMiLz48ZyBjbGFzcz0ibm9kZXMiPjxnIGNsYXNzPSJub2RlIGRlZmF1bHQgICIgaWQ9ImZsb3djaGFydC1EMUEtOSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTc1LjUsIDgyKSI+PHJlY3QgY2xhc3M9ImJhc2ljIGxhYmVsLWNvbnRhaW5lciIgc3R5bGU9IiIgeD0iLTEzMCIgeT0iLTM5IiB3aWR0aD0iMjYwIiBoZWlnaHQ9Ijc4Ii8+PGcgY2xhc3M9ImxhYmVsIiBzdHlsZT0iIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTAwLCAtMjQpIj48cmVjdC8+PGZvcmVpZ25PYmplY3Qgd2lkdGg9IjIwMCIgaGVpZ2h0PSI0OCI+PGRpdiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlOyB3aGl0ZS1zcGFjZTogYnJlYWstc3BhY2VzOyBsaW5lLWhlaWdodDogMS41OyBtYXgtd2lkdGg6IDIwMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7IHdpZHRoOiAyMDBweDsiPjxzcGFuIGNsYXNzPSJub2RlTGFiZWwgIj48cD5LZXJuZWw6IENvb3BlcmF0aXZlIG9yIFJlZ3VsYXI8L3A+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJub2RlIGRlZmF1bHQgICIgaWQ9ImZsb3djaGFydC1EMUItMTAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE3NS41LCAyMTApIj48cmVjdCBjbGFzcz0iYmFzaWMgbGFiZWwtY29udGFpbmVyIiBzdHlsZT0iIiB4PSItMTMwIiB5PSItMzkiIHdpZHRoPSIyNjAiIGhlaWdodD0iNzgiLz48ZyBjbGFzcz0ibGFiZWwiIHN0eWxlPSIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC0xMDAsIC0yNCkiPjxyZWN0Lz48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjQ4Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGU7IHdoaXRlLXNwYWNlOiBicmVhay1zcGFjZXM7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsgd2lkdGg6IDIwMHB4OyI+PHNwYW4gY2xhc3M9Im5vZGVMYWJlbCAiPjxwPkRpc3BhdGNoOiBEZXZpY2VDYWxsIG9yIEdyYXBoTGF1bmNoPC9wPjwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48L2c+PC9nPjxnIGNsYXNzPSJyb290IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLCAyNTYpIj48ZyBjbGFzcz0iY2x1c3RlcnMiPjxnIGNsYXNzPSJjbHVzdGVyICIgaWQ9IkQwX0NGRyIgZGF0YS1sb29rPSJjbGFzc2ljIj48cmVjdCBzdHlsZT0iIiB4PSI4IiB5PSI4IiB3aWR0aD0iMzM1IiBoZWlnaHQ9IjI3NiIvPjxnIGNsYXNzPSJjbHVzdGVyLWxhYmVsICIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTA3LjAxNTYyNSwgOCkiPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIxMzYuOTY4NzUiIGhlaWdodD0iMjQiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIHN0eWxlPSJkaXNwbGF5OiB0YWJsZS1jZWxsOyB3aGl0ZS1zcGFjZTogbm93cmFwOyBsaW5lLWhlaWdodDogMS41OyBtYXgtd2lkdGg6IDIwMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7Ij48c3BhbiBjbGFzcz0ibm9kZUxhYmVsICI+PHA+RGlzcGF0Y2hlciAwIGNvbmZpZzwvcD48L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PC9nPjxnIGNsYXNzPSJlZGdlUGF0aHMiLz48ZyBjbGFzcz0iZWRnZUxhYmVscyIvPjxnIGNsYXNzPSJub2RlcyI+PGcgY2xhc3M9Im5vZGUgZGVmYXVsdCAgIiBpZD0iZmxvd2NoYXJ0LUQwQS03IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNzUuNSwgODIpIj48cmVjdCBjbGFzcz0iYmFzaWMgbGFiZWwtY29udGFpbmVyIiBzdHlsZT0iIiB4PSItMTMwIiB5PSItMzkiIHdpZHRoPSIyNjAiIGhlaWdodD0iNzgiLz48ZyBjbGFzcz0ibGFiZWwiIHN0eWxlPSIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC0xMDAsIC0yNCkiPjxyZWN0Lz48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjQ4Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGU7IHdoaXRlLXNwYWNlOiBicmVhay1zcGFjZXM7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsgd2lkdGg6IDIwMHB4OyI+PHNwYW4gY2xhc3M9Im5vZGVMYWJlbCAiPjxwPktlcm5lbDogQ29vcGVyYXRpdmUgb3IgUmVndWxhcjwvcD48L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PGcgY2xhc3M9Im5vZGUgZGVmYXVsdCAgIiBpZD0iZmxvd2NoYXJ0LUQwQi04IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNzUuNSwgMjEwKSI+PHJlY3QgY2xhc3M9ImJhc2ljIGxhYmVsLWNvbnRhaW5lciIgc3R5bGU9IiIgeD0iLTEzMCIgeT0iLTM5IiB3aWR0aD0iMjYwIiBoZWlnaHQ9Ijc4Ii8+PGcgY2xhc3M9ImxhYmVsIiBzdHlsZT0iIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTAwLCAtMjQpIj48cmVjdC8+PGZvcmVpZ25PYmplY3Qgd2lkdGg9IjIwMCIgaGVpZ2h0PSI0OCI+PGRpdiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94aHRtbCIgc3R5bGU9ImRpc3BsYXk6IHRhYmxlOyB3aGl0ZS1zcGFjZTogYnJlYWstc3BhY2VzOyBsaW5lLWhlaWdodDogMS41OyBtYXgtd2lkdGg6IDIwMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7IHdpZHRoOiAyMDBweDsiPjxzcGFuIGNsYXNzPSJub2RlTGFiZWwgIj48cD5EaXNwYXRjaDogRGV2aWNlQ2FsbCBvciBHcmFwaExhdW5jaDwvcD48L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PC9nPjwvZz48ZyBjbGFzcz0ibm9kZSBkZWZhdWx0ICAiIGlkPSJmbG93Y2hhcnQtTUdSLTAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDU2MC41LCA1OSkiPjxyZWN0IGNsYXNzPSJiYXNpYyBsYWJlbC1jb250YWluZXIiIHN0eWxlPSIiIHg9Ii0xMzAiIHk9Ii01MSIgd2lkdGg9IjI2MCIgaGVpZ2h0PSIxMDIiLz48ZyBjbGFzcz0ibGFiZWwiIHN0eWxlPSIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC0xMDAsIC0zNikiPjxyZWN0Lz48Zm9yZWlnbk9iamVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjcyIj48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGU7IHdoaXRlLXNwYWNlOiBicmVhay1zcGFjZXM7IGxpbmUtaGVpZ2h0OiAxLjU7IG1heC13aWR0aDogMjAwcHg7IHRleHQtYWxpZ246IGNlbnRlcjsgd2lkdGg6IDIwMHB4OyI+PHNwYW4gY2xhc3M9Im5vZGVMYWJlbCAiPjxwPkRpc3BhdGNoZXIgTWFuYWdlcjxiciAvPkNyZWF0ZXMgYW5kIG93bnMgZGlzcGF0Y2hlcnM8L3A+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjxnIGNsYXNzPSJub2RlIGRlZmF1bHQgICIgaWQ9ImZsb3djaGFydC1EMC0yIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNzUuNSwgMTg3KSI+PHJlY3QgY2xhc3M9ImJhc2ljIGxhYmVsLWNvbnRhaW5lciIgc3R5bGU9IiIgeD0iLTk3LjU4NTkzNzUiIHk9Ii0yNyIgd2lkdGg9IjE5NS4xNzE4NzUiIGhlaWdodD0iNTQiLz48ZyBjbGFzcz0ibGFiZWwiIHN0eWxlPSIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC02Ny41ODU5Mzc1LCAtMTIpIj48cmVjdC8+PGZvcmVpZ25PYmplY3Qgd2lkdGg9IjEzNS4xNzE4NzUiIGhlaWdodD0iMjQiPjxkaXYgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGh0bWwiIHN0eWxlPSJkaXNwbGF5OiB0YWJsZS1jZWxsOyB3aGl0ZS1zcGFjZTogbm93cmFwOyBsaW5lLWhlaWdodDogMS41OyBtYXgtd2lkdGg6IDIwMHB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7Ij48c3BhbiBjbGFzcz0ibm9kZUxhYmVsICI+PHA+RGlzcGF0Y2hlciAwIChWUDApPC9wPjwvc3Bhbj48L2Rpdj48L2ZvcmVpZ25PYmplY3Q+PC9nPjwvZz48ZyBjbGFzcz0ibm9kZSBkZWZhdWx0ICAiIGlkPSJmbG93Y2hhcnQtRDEtNCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNTYwLjUsIDE4NykiPjxyZWN0IGNsYXNzPSJiYXNpYyBsYWJlbC1jb250YWluZXIiIHN0eWxlPSIiIHg9Ii05Ny41ODU5Mzc1IiB5PSItMjciIHdpZHRoPSIxOTUuMTcxODc1IiBoZWlnaHQ9IjU0Ii8+PGcgY2xhc3M9ImxhYmVsIiBzdHlsZT0iIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNjcuNTg1OTM3NSwgLTEyKSI+PHJlY3QvPjxmb3JlaWduT2JqZWN0IHdpZHRoPSIxMzUuMTcxODc1IiBoZWlnaHQ9IjI0Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9Im5vZGVMYWJlbCAiPjxwPkRpc3BhdGNoZXIgMSAoVlAxKTwvcD48L3NwYW4+PC9kaXY+PC9mb3JlaWduT2JqZWN0PjwvZz48L2c+PGcgY2xhc3M9Im5vZGUgZGVmYXVsdCAgIiBpZD0iZmxvd2NoYXJ0LUROLTYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDk0NS41LCAxODcpIj48cmVjdCBjbGFzcz0iYmFzaWMgbGFiZWwtY29udGFpbmVyIiBzdHlsZT0iIiB4PSItMTE0LjQ2ODc1IiB5PSItMjciIHdpZHRoPSIyMjguOTM3NSIgaGVpZ2h0PSI1NCIvPjxnIGNsYXNzPSJsYWJlbCIgc3R5bGU9IiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTg0LjQ2ODc1LCAtMTIpIj48cmVjdC8+PGZvcmVpZ25PYmplY3Qgd2lkdGg9IjE2OC45Mzc1IiBoZWlnaHQ9IjI0Ij48ZGl2IHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hodG1sIiBzdHlsZT0iZGlzcGxheTogdGFibGUtY2VsbDsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgbGluZS1oZWlnaHQ6IDEuNTsgbWF4LXdpZHRoOiAyMDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyI+PHNwYW4gY2xhc3M9Im5vZGVMYWJlbCAiPjxwPkRpc3BhdGNoZXIgTi0xIChWUE4tMSk8L3A+PC9zcGFuPjwvZGl2PjwvZm9yZWlnbk9iamVjdD48L2c+PC9nPjwvZz48L2c+PC9nPjwvc3ZnPg==){width="1121"
height="548"}
:::

::: {#host-api-functions .section}
## Host API Functions[¶](#host-api-functions "Permalink to this heading"){.headerlink}

Function usage:

**`cudaq_dispatch_manager_create`** creates the top-level manager that
owns dispatchers.

Parameters:

-   [`out_mgr`{.docutils .literal .notranslate}]{.pre}: receives the
    created manager handle.

Call this once near program startup and keep the manager alive for the
lifetime of the dispatch subsystem.

**`cudaq_dispatch_manager_destroy`** releases the manager and any
internal resources.

Parameters:

-   [`mgr`{.docutils .literal .notranslate}]{.pre}: manager handle to
    destroy.

Call this after all dispatchers have been destroyed and the program is
shutting down.

**`cudaq_dispatcher_create`** allocates a dispatcher instance and
validates the configuration.

Parameters:

-   [`mgr`{.docutils .literal .notranslate}]{.pre}: owning manager.

-   [`config`{.docutils .literal .notranslate}]{.pre}: filled
    [`cudaq_dispatcher_config_t`{.docutils .literal .notranslate}]{.pre}
    with:

    -   [`device_id`{.docutils .literal .notranslate}]{.pre} (default
        0): selects the CUDA device for the dispatcher

    -   [`num_blocks`{.docutils .literal .notranslate}]{.pre} (default
        1)

    -   [`threads_per_block`{.docutils .literal .notranslate}]{.pre}
        (default 32)

    -   [`num_slots`{.docutils .literal .notranslate}]{.pre} (required)

    -   [`slot_size`{.docutils .literal .notranslate}]{.pre} (required)

    -   [`vp_id`{.docutils .literal .notranslate}]{.pre} (default 0):
        tags a dispatcher to a transport channel. Queue pair selection
        and NIC port/IP binding are configured in HSB, not in this API.

    -   [`kernel_type`{.docutils .literal .notranslate}]{.pre} (default
        [`CUDAQ_KERNEL_REGULAR`{.docutils .literal .notranslate}]{.pre})

        -   [`CUDAQ_KERNEL_REGULAR`{.docutils .literal
            .notranslate}]{.pre}: standard kernel launch

        -   [`CUDAQ_KERNEL_COOPERATIVE`{.docutils .literal
            .notranslate}]{.pre}: cooperative launch
            ([`grid.sync()`{.docutils .literal .notranslate}]{.pre}
            capable)

        -   [`CUDAQ_KERNEL_UNIFIED`{.docutils .literal
            .notranslate}]{.pre}: single-kernel dispatch with integrated
            transport (see [[Unified Dispatch Mode]{.std
            .std-ref}](#unified-dispatch-mode){.reference .internal})

    -   [`dispatch_mode`{.docutils .literal .notranslate}]{.pre}
        (default [`CUDAQ_DISPATCH_DEVICE_CALL`{.docutils .literal
        .notranslate}]{.pre})

        -   [`CUDAQ_DISPATCH_DEVICE_CALL`{.docutils .literal
            .notranslate}]{.pre}: direct [`__device__`{.docutils
            .literal .notranslate}]{.pre} handler call (lowest latency)

        -   [`CUDAQ_DISPATCH_GRAPH_LAUNCH`{.docutils .literal
            .notranslate}]{.pre}: CUDA graph launch from device code
            (requires [`sm_90+`{.docutils .literal .notranslate}]{.pre},
            Hopper or later GPUs)

-   [`out_dispatcher`{.docutils .literal .notranslate}]{.pre}: receives
    the created dispatcher handle.

Call this before wiring ring buffers, function tables, or control state.

**`cudaq_dispatcher_destroy`** releases a dispatcher after it has been
stopped.

Parameters:

-   [`dispatcher`{.docutils .literal .notranslate}]{.pre}: dispatcher
    handle to destroy.

Call this when the dispatcher is no longer needed.

**`cudaq_dispatcher_set_ringbuffer`** provides the RX/TX flag and data
pointers the dispatch kernel will poll and use for request/response
slots.

Parameters:

-   [`dispatcher`{.docutils .literal .notranslate}]{.pre}: dispatcher
    handle.

-   [`ringbuffer`{.docutils .literal .notranslate}]{.pre}:
    [`cudaq_ringbuffer_t`{.docutils .literal .notranslate}]{.pre} with:

    -   [`rx_flags`{.docutils .literal .notranslate}]{.pre}:
        device-visible pointer to RX flags.

    -   [`tx_flags`{.docutils .literal .notranslate}]{.pre}:
        device-visible pointer to TX flags.

    -   [`rx_data`{.docutils .literal .notranslate}]{.pre}:
        device-visible pointer to RX slot data (request payloads).

    -   [`tx_data`{.docutils .literal .notranslate}]{.pre}:
        device-visible pointer to TX slot data (response payloads).

    -   [`rx_stride_sz`{.docutils .literal .notranslate}]{.pre}: size in
        bytes of each RX slot.

    -   [`tx_stride_sz`{.docutils .literal .notranslate}]{.pre}: size in
        bytes of each TX slot.

Call this before [`cudaq_dispatcher_start`{.docutils .literal
.notranslate}]{.pre}, after allocating mapped host memory or device
memory for the ring buffers.

**`cudaq_dispatcher_set_function_table`** supplies the function table
containing handler pointers, IDs, and schemas.

Parameters:

-   [`dispatcher`{.docutils .literal .notranslate}]{.pre}: dispatcher
    handle.

-   [`table`{.docutils .literal .notranslate}]{.pre}:
    [`cudaq_function_table_t`{.docutils .literal .notranslate}]{.pre}
    with:

    -   [`entries`{.docutils .literal .notranslate}]{.pre}: device
        pointer to array of [`cudaq_function_entry_t`{.docutils .literal
        .notranslate}]{.pre}.

    -   [`count`{.docutils .literal .notranslate}]{.pre}: number of
        entries in the table.

::: {.highlight-cpp .notranslate}
::: highlight
    // Unified function table entry with schema
    struct cudaq_function_entry_t {
      union {
        void*           device_fn_ptr;   // for CUDAQ_DISPATCH_DEVICE_CALL
        cudaGraphExec_t graph_exec;      // for CUDAQ_DISPATCH_GRAPH_LAUNCH
      } handler;

      uint32_t                function_id;
      uint8_t                 dispatch_mode;   // Per-handler dispatch mode
      uint8_t                 reserved[3];

      cudaq_handler_schema_t  schema;          // Handler interface schema
    };

    struct cudaq_function_table_t {
      cudaq_function_entry_t* entries;   // Device pointer to entry array
      uint32_t                count;     // Number of entries
    };
:::
:::

Call this after initializing the device-side function table entries.
Each entry contains a handler pointer (or graph), function_id, dispatch
mode, and schema describing the handler's interface.

Function ID semantics:

-   [`function_id`{.docutils .literal .notranslate}]{.pre} is the 32-bit
    **[`FNV-1a`{.docutils .literal .notranslate}]{.pre} hash** of the
    handler name string.

-   The handler name is the string you hash when populating entries;
    there is no separate runtime registration call.

-   If no entry matches, the dispatcher clears the slot without a
    response.

-   Suggested: use stable, human-readable handler names (e.g.,
    [`"mock_decode"`{.docutils .literal .notranslate}]{.pre}).

**`cudaq_dispatcher_set_control`** supplies the shutdown flag and stats
buffer the dispatch kernel uses for termination and bookkeeping.

Parameters:

-   [`dispatcher`{.docutils .literal .notranslate}]{.pre}: dispatcher
    handle.

-   [`shutdown_flag`{.docutils .literal .notranslate}]{.pre}:
    device-visible flag used to signal shutdown.

-   [`stats`{.docutils .literal .notranslate}]{.pre}: device-visible
    stats buffer.

Call this before starting the dispatcher; both buffers must remain valid
for the dispatcher's lifetime.

**`cudaq_dispatcher_set_launch_fn`** provides the host-side launch
wrapper that invokes the dispatch kernel with the correct grid/block
dimensions.

Parameters:

-   [`dispatcher`{.docutils .literal .notranslate}]{.pre}: dispatcher
    handle.

-   [`launch_fn`{.docutils .literal .notranslate}]{.pre}: host launch
    function pointer.

Call this once during setup. Typically you pass one of the provided
launch functions:

-   [`cudaq_launch_dispatch_kernel_regular`{.docutils .literal
    .notranslate}]{.pre} - for [`CUDAQ_KERNEL_REGULAR`{.docutils
    .literal .notranslate}]{.pre} mode

-   [`cudaq_launch_dispatch_kernel_cooperative`{.docutils .literal
    .notranslate}]{.pre} - for [`CUDAQ_KERNEL_COOPERATIVE`{.docutils
    .literal .notranslate}]{.pre} mode

**`cudaq_dispatcher_start`** launches the persistent dispatch kernel and
begins processing slots.

Parameters:

-   [`dispatcher`{.docutils .literal .notranslate}]{.pre}: dispatcher
    handle.

Call this only after ring buffers, function table, control buffers, and
launch function are set.

**`cudaq_dispatcher_stop`** signals the dispatch kernel to exit and
waits for it to shut down.

Parameters:

-   [`dispatcher`{.docutils .literal .notranslate}]{.pre}: dispatcher
    handle.

Call this during tear-down before destroying the dispatcher.

**`cudaq_dispatcher_get_processed`** reads the processed‑packet counter
from the stats buffer to support debugging or throughput tracking.

Parameters:

-   [`dispatcher`{.docutils .literal .notranslate}]{.pre}: dispatcher
    handle.

-   [`out_packets`{.docutils .literal .notranslate}]{.pre}: receives the
    processed packet count.

::: {#occupancy-query-and-eager-module-loading .section}
### Occupancy Query and Eager Module Loading[¶](#occupancy-query-and-eager-module-loading "Permalink to this heading"){.headerlink}

Before calling [`cudaq_dispatcher_start`{.docutils .literal
.notranslate}]{.pre}, call the appropriate occupancy query to force
eager loading of the dispatch kernel module. This avoids lazy-load
deadlocks when the dispatch kernel and transport kernels (e.g., HSB
RX/TX) run as persistent kernels.

**`cudaq_dispatch_kernel_query_occupancy`** returns the maximum number
of active blocks per multiprocessor for the **regular** dispatch kernel.

Parameters:

-   [`out_blocks`{.docutils .literal .notranslate}]{.pre}: receives the
    max blocks per SM (or 0 on error).

-   [`threads_per_block`{.docutils .literal .notranslate}]{.pre}: block
    size used for the occupancy calculation.

Returns [`cudaSuccess`{.docutils .literal .notranslate}]{.pre} on
success. Call this when [`kernel_type`{.docutils .literal
.notranslate}]{.pre} is [`CUDAQ_KERNEL_REGULAR`{.docutils .literal
.notranslate}]{.pre}.

**`cudaq_dispatch_kernel_cooperative_query_occupancy`** returns the
maximum number of active blocks per multiprocessor for the
**cooperative** dispatch kernel.

Parameters:

-   [`out_blocks`{.docutils .literal .notranslate}]{.pre}: receives the
    max blocks per SM (or 0 on error).

-   [`threads_per_block`{.docutils .literal .notranslate}]{.pre}: block
    size used for the occupancy calculation (e.g., 128 for cooperative
    decoders).

Returns [`cudaSuccess`{.docutils .literal .notranslate}]{.pre} on
success. Call this when [`kernel_type`{.docutils .literal
.notranslate}]{.pre} is [`CUDAQ_KERNEL_COOPERATIVE`{.docutils .literal
.notranslate}]{.pre}. Use the same [`threads_per_block`{.docutils
.literal .notranslate}]{.pre} value that will be passed to the
dispatcher configuration and launch function.

Call the occupancy function that matches the dispatcher's
[`kernel_type`{.docutils .literal .notranslate}]{.pre} once before
[`cudaq_dispatcher_start`{.docutils .literal .notranslate}]{.pre}; the
result can be used to size the dispatch grid (e.g., to reserve
[`SM`{.docutils .literal .notranslate}]{.pre}'s for transport kernels).

Lifetime/ownership:

-   All resources are assumed to live for the program lifetime.

-   The API does not take ownership of host-allocated memory.

Threading:

-   Single-threaded host usage; create/wire/start/stop from one thread.

Error handling:

-   All calls return [`cudaq_status_t`{.docutils .literal
    .notranslate}]{.pre}.

-   [`CUDAQ_ERR_INVALID_ARG`{.docutils .literal .notranslate}]{.pre} for
    missing pointers or invalid configuration.

-   [`CUDAQ_ERR_CUDA`{.docutils .literal .notranslate}]{.pre} for CUDA
    API failures during start/stop.
:::

::: {#graph-based-dispatch-functions .section}
### Graph-Based Dispatch Functions[¶](#graph-based-dispatch-functions "Permalink to this heading"){.headerlink}

The following functions are only available when using
[`CUDAQ_DISPATCH_GRAPH_LAUNCH`{.docutils .literal .notranslate}]{.pre}
mode with [`sm_90+`{.docutils .literal .notranslate}]{.pre} GPUs:

**`cudaq_create_dispatch_graph_regular`** creates a graph-based dispatch
context that enables device-side graph launching.

Parameters:

-   [`rx_flags`{.docutils .literal .notranslate}]{.pre}: device-visible
    pointer to RX ring buffer flags

-   [`tx_flags`{.docutils .literal .notranslate}]{.pre}: device-visible
    pointer to TX ring buffer flags

-   [`rx_data`{.docutils .literal .notranslate}]{.pre}: device-visible
    pointer to RX slot data (request payloads)

-   [`tx_data`{.docutils .literal .notranslate}]{.pre}: device-visible
    pointer to TX slot data (response payloads)

-   [`rx_stride_sz`{.docutils .literal .notranslate}]{.pre}: size in
    bytes of each RX slot

-   [`tx_stride_sz`{.docutils .literal .notranslate}]{.pre}: size in
    bytes of each TX slot

-   [`function_table`{.docutils .literal .notranslate}]{.pre}: device
    pointer to function table entries

-   [`func_count`{.docutils .literal .notranslate}]{.pre}: number of
    function table entries

-   [`graph_io_ctx`{.docutils .literal .notranslate}]{.pre}: device
    pointer to a [`GraphIOContext`{.docutils .literal
    .notranslate}]{.pre} struct for graph buffer communication

-   [`shutdown_flag`{.docutils .literal .notranslate}]{.pre}:
    device-visible shutdown flag

-   [`stats`{.docutils .literal .notranslate}]{.pre}: device-visible
    stats buffer

-   [`num_slots`{.docutils .literal .notranslate}]{.pre}: number of ring
    buffer slots

-   [`num_blocks`{.docutils .literal .notranslate}]{.pre}: grid size for
    dispatch kernel

-   [`threads_per_block`{.docutils .literal .notranslate}]{.pre}: block
    size for dispatch kernel

-   [`stream`{.docutils .literal .notranslate}]{.pre}: CUDA stream for
    graph operations

-   [`out_context`{.docutils .literal .notranslate}]{.pre}: receives the
    created graph context handle

Returns [`cudaSuccess`{.docutils .literal .notranslate}]{.pre} on
success, or CUDA error code on failure.

This function creates a graph containing the dispatch kernel,
instantiates it with [`cudaGraphInstantiateFlagDeviceLaunch`{.docutils
.literal .notranslate}]{.pre}, and uploads it to the device. The
resulting graph context enables device-side
[`cudaGraphLaunch()`{.docutils .literal .notranslate}]{.pre} calls from
within handlers.

**`cudaq_launch_dispatch_graph`** launches the dispatch graph to begin
processing RPC messages.

Parameters:

-   [`context`{.docutils .literal .notranslate}]{.pre}: graph context
    handle from [`cudaq_create_dispatch_graph_regular`{.docutils
    .literal .notranslate}]{.pre}

-   [`stream`{.docutils .literal .notranslate}]{.pre}: CUDA stream for
    graph launch

Returns [`cudaSuccess`{.docutils .literal .notranslate}]{.pre} on
success, or CUDA error code on failure.

Call this to start the persistent dispatch kernel. The kernel will
continue running until the shutdown flag is set.

**`cudaq_destroy_dispatch_graph`** destroys the graph context and
releases all associated resources.

Parameters:

-   [`context`{.docutils .literal .notranslate}]{.pre}: graph context
    handle to destroy

Returns [`cudaSuccess`{.docutils .literal .notranslate}]{.pre} on
success, or CUDA error code on failure.

Call this after the dispatch kernel has exited (shutdown flag was set)
to clean up graph resources.
:::

::: {#kernel-launch-helper-functions .section}
### Kernel Launch Helper Functions[¶](#kernel-launch-helper-functions "Permalink to this heading"){.headerlink}

The following helper functions are provided for use with
[`cudaq_dispatcher_set_launch_fn()`{.docutils .literal
.notranslate}]{.pre}:

**`cudaq_launch_dispatch_kernel_regular`** launches the dispatch kernel
in regular (non-cooperative) mode.

Parameters:

-   [`rx_flags`{.docutils .literal .notranslate}]{.pre}: device-visible
    pointer to RX ring buffer flags

-   [`tx_flags`{.docutils .literal .notranslate}]{.pre}: device-visible
    pointer to TX ring buffer flags

-   [`rx_data`{.docutils .literal .notranslate}]{.pre}: device-visible
    pointer to RX slot data (request payloads)

-   [`tx_data`{.docutils .literal .notranslate}]{.pre}: device-visible
    pointer to TX slot data (response payloads)

-   [`rx_stride_sz`{.docutils .literal .notranslate}]{.pre}: size in
    bytes of each RX slot

-   [`tx_stride_sz`{.docutils .literal .notranslate}]{.pre}: size in
    bytes of each TX slot

-   [`function_table`{.docutils .literal .notranslate}]{.pre}: device
    pointer to function table entries

-   [`func_count`{.docutils .literal .notranslate}]{.pre}: number of
    function table entries

-   [`shutdown_flag`{.docutils .literal .notranslate}]{.pre}:
    device-visible shutdown flag

-   [`stats`{.docutils .literal .notranslate}]{.pre}: device-visible
    stats buffer

-   [`num_slots`{.docutils .literal .notranslate}]{.pre}: number of ring
    buffer slots

-   [`num_blocks`{.docutils .literal .notranslate}]{.pre}: grid size for
    dispatch kernel

-   [`threads_per_block`{.docutils .literal .notranslate}]{.pre}: block
    size for dispatch kernel

-   [`stream`{.docutils .literal .notranslate}]{.pre}: CUDA stream for
    kernel launch

Use this when [`kernel_type`{.docutils .literal .notranslate}]{.pre} is
set to [`CUDAQ_KERNEL_REGULAR`{.docutils .literal .notranslate}]{.pre}
in the dispatcher configuration.

**`cudaq_launch_dispatch_kernel_cooperative`** launches the dispatch
kernel in cooperative mode.

Parameters: Same as [`cudaq_launch_dispatch_kernel_regular`{.docutils
.literal .notranslate}]{.pre}.

Use this when [`kernel_type`{.docutils .literal .notranslate}]{.pre} is
set to [`CUDAQ_KERNEL_COOPERATIVE`{.docutils .literal
.notranslate}]{.pre} in the dispatcher configuration. This enables the
dispatch kernel and handlers to use grid-wide synchronization via
[`cooperative_groups::this_grid().sync()`{.docutils .literal
.notranslate}]{.pre}.
:::
:::

::: {#memory-layout-and-ring-buffer-wiring .section}
## Memory Layout and Ring Buffer Wiring[¶](#memory-layout-and-ring-buffer-wiring "Permalink to this heading"){.headerlink}

Each slot is a fixed-size byte region:

::: {.highlight-text .notranslate}
::: highlight
    | RPCHeader | payload bytes (arg_len) | unused padding (slot_size - header - payload) |
:::
:::

Unused padding is the remaining bytes in the fixed-size slot after the
header and payload.

Flags (both are [`uint64_t`{.docutils .literal .notranslate}]{.pre}
arrays of slot flags):

-   [`rx_flags[slot]`{.docutils .literal .notranslate}]{.pre} is set by
    the producer to a non-zero value when a slot is ready.

-   [`tx_flags[slot]`{.docutils .literal .notranslate}]{.pre} is set by
    the dispatch kernel to a non-zero value when the response is ready.

Message completion note: An RPC message may be delivered as multiple
RDMA writes into a single slot. Completion is signaled only after the
final write (often an RDMA write with immediate) sets
[`rx_flags[slot]`{.docutils .literal .notranslate}]{.pre} to a non-zero
value. The dispatch kernel treats the slot as complete only after the
flag is set.

In the NIC-free path, flags and data are allocated with
[`cudaHostAllocMapped`{.docutils .literal .notranslate}]{.pre} so the
device and host see the same memory.
:::

::: {#step-by-step-wiring-the-host-api-minimal .section}
## Step-by-Step: Wiring the Host API (Minimal)[¶](#step-by-step-wiring-the-host-api-minimal "Permalink to this heading"){.headerlink}

::: {.highlight-cpp .notranslate}
::: highlight
    // Host API wiring
    ASSERT_EQ(cudaq_dispatch_manager_create(&manager_), CUDAQ_OK);
    cudaq_dispatcher_config_t config{};
    config.device_id = 0;
    config.num_blocks = 1;
    config.threads_per_block = 32;
    config.num_slots = static_cast<uint32_t>(num_slots_);
    config.slot_size = static_cast<uint32_t>(slot_size_);
    config.vp_id = 0;
    config.kernel_type = CUDAQ_KERNEL_REGULAR;
    config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;

    ASSERT_EQ(cudaq_dispatcher_create(manager_, &config, &dispatcher_), CUDAQ_OK);

    cudaq_ringbuffer_t ringbuffer{};
    ringbuffer.rx_flags = rx_flags_;
    ringbuffer.tx_flags = tx_flags_;
    ringbuffer.rx_data = rx_data_;
    ringbuffer.tx_data = tx_data_;
    ringbuffer.rx_stride_sz = slot_size_;
    ringbuffer.tx_stride_sz = slot_size_;
    ASSERT_EQ(cudaq_dispatcher_set_ringbuffer(dispatcher_, &ringbuffer), CUDAQ_OK);

    // Allocate and initialize function table entries
    cudaq_function_entry_t* d_entries;
    cudaMalloc(&d_entries, func_count_ * sizeof(cudaq_function_entry_t));

    // Initialize entries on device (including schemas)
    init_function_table<<<1, 1>>>(d_entries);
    cudaDeviceSynchronize();

    cudaq_function_table_t table{};
    table.entries = d_entries;
    table.count = func_count_;
    ASSERT_EQ(cudaq_dispatcher_set_function_table(dispatcher_, &table), CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_set_control(dispatcher_, d_shutdown_flag_, d_stats_),
              CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_set_launch_fn(
                  dispatcher_,
                  &cudaq::qec::realtime::mock_decode_launch_dispatch_kernel),
              CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_start(dispatcher_), CUDAQ_OK);
:::
:::
:::

::: {#device-handler-and-function-id .section}
## Device Handler and Function ID[¶](#device-handler-and-function-id "Permalink to this heading"){.headerlink}

::: {.highlight-cpp .notranslate}
::: highlight
    // The dispatcher uses function_id to find the handler
    constexpr std::uint32_t MOCK_DECODE_FUNCTION_ID =
        cudaq::realtime::fnv1a_hash("mock_decode");

    /// @brief Initialize the device function table with schema
    __global__ void init_function_table(cudaq_function_entry_t* entries) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Entry 0: Mock decoder
        entries[0].handler.device_fn_ptr =
            reinterpret_cast<void*>(&cudaq::qec::realtime::mock_decode_rpc);
        entries[0].function_id = MOCK_DECODE_FUNCTION_ID;
        entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;

        // Schema: 1 arg (bit-packed detection events), 1 result (correction byte)
        entries[0].schema.num_args = 1;
        entries[0].schema.args[0] = {CUDAQ_TYPE_BIT_PACKED, {0}, 16, 128};  // 128 bits
        entries[0].schema.num_results = 1;
        entries[0].schema.results[0] = {CUDAQ_TYPE_UINT8, {0}, 1, 1};
      }
    }
:::
:::

::: {#multi-argument-handler-example .section}
### Multi-Argument Handler Example[¶](#multi-argument-handler-example "Permalink to this heading"){.headerlink}

::: {.highlight-cpp .notranslate}
::: highlight
    constexpr std::uint32_t ADVANCED_DECODE_FUNCTION_ID =
        cudaq::realtime::fnv1a_hash("advanced_decode");

    __global__ void init_advanced_handler(cudaq_function_entry_t* entries,
                                           uint32_t index) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        entries[index].handler.device_fn_ptr =
            reinterpret_cast<void*>(&advanced_decode_rpc);
        entries[index].function_id = ADVANCED_DECODE_FUNCTION_ID;
        entries[index].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;

        // Schema: 2 args (detection events + calibration), 1 result
        entries[index].schema.num_args = 2;
        entries[index].schema.args[0] = {CUDAQ_TYPE_BIT_PACKED, {0}, 16, 128};
        entries[index].schema.args[1] = {CUDAQ_TYPE_ARRAY_FLOAT32, {0}, 64, 16};  // 16 floats
        entries[index].schema.num_results = 1;
        entries[index].schema.results[0] = {CUDAQ_TYPE_UINT8, {0}, 1, 1};
      }
    }
:::
:::
:::
:::

::: {#cuda-graph-dispatch-mode .section}
## CUDA Graph Dispatch Mode[¶](#cuda-graph-dispatch-mode "Permalink to this heading"){.headerlink}

The [`CUDAQ_DISPATCH_GRAPH_LAUNCH`{.docutils .literal
.notranslate}]{.pre} mode enables handlers to be executed as
pre-captured CUDA graphs launched from device code. This is useful for
complex multi-kernel workflows that benefit from graph optimization and
can reduce kernel launch overhead for sophisticated decoders.

::: {#requirements .section}
### Requirements[¶](#requirements "Permalink to this heading"){.headerlink}

-   **GPU Architecture**: Compute capability 9.0 or higher (Hopper H100
    or later)

-   **CUDA Version**: CUDA 12.0+ with device-side graph launch support

-   **Graph Setup**: Handler graphs must be captured and instantiated
    with [`cudaGraphInstantiateFlagDeviceLaunch`{.docutils .literal
    .notranslate}]{.pre}
:::

::: {#graph-based-dispatch-api .section}
### Graph-Based Dispatch API[¶](#graph-based-dispatch-api "Permalink to this heading"){.headerlink}

The API provides functions to properly wrap the dispatch kernel in a
graph context that enables device-side [`cudaGraphLaunch()`{.docutils
.literal .notranslate}]{.pre}:

::: {.highlight-cpp .notranslate}
::: highlight
    // Opaque handle for graph-based dispatch context
    typedef struct cudaq_dispatch_graph_context cudaq_dispatch_graph_context;

    // Create a graph-based dispatch context
    cudaError_t cudaq_create_dispatch_graph_regular(
        volatile uint64_t *rx_flags, volatile uint64_t *tx_flags,
        uint8_t *rx_data, uint8_t *tx_data,
        size_t rx_stride_sz, size_t tx_stride_sz,
        cudaq_function_entry_t *function_table, size_t func_count,
        void *graph_io_ctx, volatile int *shutdown_flag, uint64_t *stats,
        size_t num_slots, uint32_t num_blocks, uint32_t threads_per_block,
        cudaStream_t stream, cudaq_dispatch_graph_context **out_context);

    // Launch the dispatch graph
    cudaError_t cudaq_launch_dispatch_graph(cudaq_dispatch_graph_context *context,
                                            cudaStream_t stream);

    // Destroy the dispatch graph context
    cudaError_t cudaq_destroy_dispatch_graph(cudaq_dispatch_graph_context *context);
:::
:::
:::

::: {#graph-handler-setup-example .section}
### Graph Handler Setup Example[¶](#graph-handler-setup-example "Permalink to this heading"){.headerlink}

::: {.highlight-cpp .notranslate}
::: highlight
    /// @brief Initialize function table with CUDA graph handler
    __global__ void init_function_table_graph(cudaq_function_entry_t* entries) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        entries[0].handler.graph_exec = /* pre-captured cudaGraphExec_t */;
        entries[0].function_id = DECODE_FUNCTION_ID;
        entries[0].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;

        // Schema: same as device call mode
        entries[0].schema.num_args = 1;
        entries[0].schema.args[0] = {TYPE_BIT_PACKED, {0}, 16, 128};
        entries[0].schema.num_results = 1;
        entries[0].schema.results[0] = {TYPE_UINT8, {0}, 1, 1};
      }
    }
:::
:::
:::

::: {#graph-capture-and-instantiation .section}
### Graph Capture and Instantiation[¶](#graph-capture-and-instantiation "Permalink to this heading"){.headerlink}

Handler graphs must be captured and instantiated with the device launch
flag:

::: {.highlight-cpp .notranslate}
::: highlight
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);

    // Capture the decoder kernel(s) into a graph
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    decode_kernel<<<blocks, threads, 0, capture_stream>>>(args...);
    cudaStreamEndCapture(capture_stream, &graph);

    // Instantiate with device launch flag (required for device-side cudaGraphLaunch)
    cudaGraphExec_t graph_exec;
    cudaGraphInstantiateWithFlags(&graph_exec, graph,
                                  cudaGraphInstantiateFlagDeviceLaunch);

    // Upload graph to device
    cudaGraphUpload(graph_exec, capture_stream);
    cudaStreamSynchronize(capture_stream);
    cudaStreamDestroy(capture_stream);
:::
:::
:::

::: {#when-to-use-graph-dispatch .section}
### When to Use Graph Dispatch[¶](#when-to-use-graph-dispatch "Permalink to this heading"){.headerlink}

Use [`CUDAQ_DISPATCH_GRAPH_LAUNCH`{.docutils .literal
.notranslate}]{.pre} mode with the graph-based dispatch API when
handlers need to launch CUDA graphs from device code. The graph-based
dispatch API ([`cudaq_create_dispatch_graph_regular()`{.docutils
.literal .notranslate}]{.pre} +
[`cudaq_launch_dispatch_graph()`{.docutils .literal
.notranslate}]{.pre}) wraps the dispatch kernel in a graph execution
context, enabling device-side [`cudaGraphLaunch()`{.docutils .literal
.notranslate}]{.pre} calls from within handlers.
:::

::: {#graph-vs-device-call-dispatch .section}
### Graph vs Device Call Dispatch[¶](#graph-vs-device-call-dispatch "Permalink to this heading"){.headerlink}

**Device Call Mode** ([`CUDAQ_DISPATCH_DEVICE_CALL`{.docutils .literal
.notranslate}]{.pre}):

-   Lowest latency for simple handlers

-   Direct [`__device__`{.docutils .literal .notranslate}]{.pre}
    function call from dispatcher

-   Suitable for lightweight decoders and data transformations

-   No special hardware requirements

**Graph Launch Mode** ([`CUDAQ_DISPATCH_GRAPH_LAUNCH`{.docutils .literal
.notranslate}]{.pre}):

-   Enables complex multi-kernel workflows

-   Benefits from CUDA graph optimizations

-   Requires [`sm_90+`{.docutils .literal .notranslate}]{.pre} hardware
    (Hopper or later)

-   Higher setup overhead but can reduce per-invocation latency for
    complex pipelines
:::
:::

::: {#building-and-sending-an-rpc-message .section}
## Building and Sending an RPC Message[¶](#building-and-sending-an-rpc-message "Permalink to this heading"){.headerlink}

Adapted from [`test_realtime_decoding.cu`{.docutils .literal
.notranslate}]{.pre} (the actual test uses a library helper,
[`setup_mock_decode_function_table`{.docutils .literal
.notranslate}]{.pre}, that performs equivalent setup via
[`cudaMemcpy`{.docutils .literal .notranslate}]{.pre}):

Note: this host-side snippet emulates what the external device/FPGA
would do when populating RX slots in a HSB deployment.

::: {.highlight-cpp .notranslate}
::: highlight
    /// @brief Write detection events to RX buffer in RPC format.
    void write_rpc_request(std::size_t slot, const std::vector<uint8_t>& measurements) {
      uint8_t* slot_data = const_cast<uint8_t*>(rx_data_host_) + slot * slot_size_;

      // Write RPCHeader
      cudaq::realtime::RPCHeader* header =
          reinterpret_cast<cudaq::realtime::RPCHeader*>(slot_data);
      header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
      header->function_id = MOCK_DECODE_FUNCTION_ID;
      header->arg_len = static_cast<std::uint32_t>(measurements.size());
      header->request_id = static_cast<std::uint32_t>(slot);
      header->ptp_timestamp = 0;  // Set by FPGA in production; 0 for NIC-free tests

      // Write measurement data after header
      memcpy(slot_data + sizeof(cudaq::realtime::RPCHeader),
             measurements.data(), measurements.size());
    }
:::
:::
:::

::: {#reading-the-response .section}
## Reading the Response[¶](#reading-the-response "Permalink to this heading"){.headerlink}

Note: this host-side snippet emulates what the external device/FPGA
would do when consuming TX slots in a HSB deployment.

::: {.highlight-cpp .notranslate}
::: highlight
    /// @brief Read response from TX buffer.
    /// Responses are written by the dispatch kernel to the TX ring buffer; read from tx_data, not rx_data.
    bool read_rpc_response(std::size_t slot, uint8_t& correction,
                           std::int32_t* status_out = nullptr,
                           std::uint32_t* result_len_out = nullptr,
                           std::uint32_t* request_id_out = nullptr,
                           std::uint64_t* ptp_timestamp_out = nullptr) {
      __sync_synchronize();
      const uint8_t* slot_data = const_cast<uint8_t*>(tx_data_host_) + slot * slot_size_;

      // Read RPCResponse
      const cudaq::realtime::RPCResponse* response =
          reinterpret_cast<const cudaq::realtime::RPCResponse*>(slot_data);

      if (response->magic != cudaq::realtime::RPC_MAGIC_RESPONSE) {
        return false;
      }

      if (status_out)
        *status_out = response->status;
      if (result_len_out)
        *result_len_out = response->result_len;
      if (request_id_out)
        *request_id_out = response->request_id;
      if (ptp_timestamp_out)
        *ptp_timestamp_out = response->ptp_timestamp;

      if (response->status != 0) {
        return false;
      }

      // Read correction data after response header
      correction = *(slot_data + sizeof(cudaq::realtime::RPCResponse));
      return true;
    }
:::
:::
:::

::: {#schema-driven-argument-parsing .section}
## Schema-Driven Argument Parsing[¶](#schema-driven-argument-parsing "Permalink to this heading"){.headerlink}

The dispatcher uses the handler schema to interpret the
[`typeless`{.docutils .literal .notranslate}]{.pre} payload bytes. This
example shows conceptual parsing logic:

::: {.highlight-cpp .notranslate}
::: highlight
    __device__ void parse_args_from_payload(
        const uint8_t* payload,
        const cudaq_handler_schema_t& schema,
        void** arg_ptrs) {

      uint32_t offset = 0;

      for (uint8_t i = 0; i < schema.num_args; i++) {
        arg_ptrs[i] = const_cast<uint8_t*>(payload + offset);
        offset += schema.args[i].size_bytes;
      }
    }

    __device__ void dispatch_with_schema(
        uint8_t* slot_data,
        const cudaq_function_entry_t& entry) {

      RPCHeader* hdr = reinterpret_cast<RPCHeader*>(slot_data);
      uint8_t* payload = slot_data + sizeof(RPCHeader);

      // Parse arguments using schema
      void* arg_ptrs[8];
      parse_args_from_payload(payload, entry.schema, arg_ptrs);

      // Call handler with parsed arguments
      if (entry.dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL) {
        auto handler = reinterpret_cast<HandlerFn>(entry.handler.device_fn_ptr);
        handler(arg_ptrs, entry.schema.num_args, /* result buffer */);
      }
      // ... graph launch path uses same parsed args
    }
:::
:::

For multi-argument payloads, arguments are **concatenated in schema
order**:

::: {.highlight-text .notranslate}
::: highlight
    | RPCHeader | arg0_bytes | arg1_bytes | arg2_bytes | ... |
                 ^            ^            ^
                 offset=0     offset=16    offset=80
:::
:::

The schema specifies the size of each argument, allowing the dispatcher
to compute offsets.
:::

::: {#hsb-3-kernel-workflow-primary .section}
## HSB 3-Kernel Workflow (Primary)[¶](#hsb-3-kernel-workflow-primary "Permalink to this heading"){.headerlink}

See the 3-Kernel Architecture diagram above for the complete data flow.
The key integration points are:

**Ring buffer hand-off (RX → Dispatch)**:

::: {.highlight-cpp .notranslate}
::: highlight
    // HSB RX kernel sets this after writing detection event data
    rx_flags[slot] = device_ptr_to_slot_data;
:::
:::

**Ring buffer hand-off (Dispatch → TX)**:

::: {.highlight-cpp .notranslate}
::: highlight
    // Dispatch kernel sets this after writing RPCResponse
    tx_flags[slot] = device_ptr_to_slot_data;
:::
:::

**Latency path**: The critical path is:

1.  RDMA write completes → RX kernel signals → Dispatch polls and
    processes → TX kernel polls and sends → RDMA read completes

All three kernels are **persistent** (launched once, run indefinitely),
so there is no kernel launch overhead in the hot path.
:::

::: {#nic-free-testing-no-hsb-no-connectx-7 .section}
## NIC-Free Testing (No HSB / No ConnectX-7)[¶](#nic-free-testing-no-hsb-no-connectx-7 "Permalink to this heading"){.headerlink}

Emulate RX/TX with mapped host memory:

-   [`cuda-quantum`{.docutils .literal .notranslate}]{.pre} host API
    test:

    -   [`realtime/unittests/test_dispatch_kernel.cu`{.docutils .literal
        .notranslate}]{.pre}
:::

::: {#troubleshooting .section}
## Troubleshooting[¶](#troubleshooting "Permalink to this heading"){.headerlink}

-   **Timeout waiting for TX**: ensure the RX flag points to
    device-mapped memory.

-   **Invalid [`arg`{.docutils .literal .notranslate}]{.pre}**: check
    [`slot_size`{.docutils .literal .notranslate}]{.pre},
    [`num_slots`{.docutils .literal .notranslate}]{.pre}, function table
    pointers.

-   **CUDA errors**: verify [`device_id`{.docutils .literal
    .notranslate}]{.pre}, and that CUDA is initialized.
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](installation.html "Installation"){.btn .btn-neutral
.float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](protocol.html "CUDA-Q Realtime Messaging Protocol"){.btn
.btn-neutral .float-right accesskey="n" rel="next"}
:::

------------------------------------------------------------------------

::: {role="contentinfo"}
© Copyright 2026, NVIDIA Corporation & Affiliates.
:::

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme) provided by
[Read the Docs](https://readthedocs.org).
:::
:::
:::
:::
