::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-4290
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](using/quick_start.html){.reference .internal}
    -   [Install
        CUDA-Q](using/quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](using/quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q
        Academic](using/quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](using/basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q
        Kernel?](using/basics/kernel_intro.html){.reference .internal}
    -   [Building your first CUDA-Q
        Program](using/basics/build_kernel.html){.reference .internal}
    -   [Running your first CUDA-Q
        Program](using/basics/run_kernel.html){.reference .internal}
        -   [Sample](using/basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](using/basics/run_kernel.html#run){.reference
            .internal}
        -   [Observe](using/basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](using/basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](using/basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](using/basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
-   [Examples](using/examples/examples.html){.reference .internal}
    -   [Introduction](using/examples/introduction.html){.reference
        .internal}
    -   [Building
        Kernels](using/examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](using/examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](using/examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](using/examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](using/examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](using/examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](using/examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](using/examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](using/examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](using/examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](using/examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](using/examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](using/examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](using/examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring
        Kernels](using/examples/measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](using/examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing
        Kernels](using/examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](using/examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](using/examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](using/examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](using/examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](using/examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](using/examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](using/examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](using/examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](using/examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](using/examples/expectation_values.html){.reference
        .internal}
        -   [Parallelizing across Multiple
            Processors](using/examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](using/examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](using/examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](using/examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](using/examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](using/examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](using/examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends ([`remote-mqpu`{.code .docutils
            .literal
            .notranslate}]{.pre})](using/examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
            .internal}
    -   [Optimizers &
        Gradients](examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [CUDA-Q Optimizer
            Overview](examples/python/optimizers_gradients.html#CUDA-Q-Optimizer-Overview){.reference
            .internal}
            -   [Gradient-Free Optimizers (no gradients
                required):](examples/python/optimizers_gradients.html#Gradient-Free-Optimizers-(no-gradients-required):){.reference
                .internal}
            -   [Gradient-Based Optimizers (require
                gradients):](examples/python/optimizers_gradients.html#Gradient-Based-Optimizers-(require-gradients):){.reference
                .internal}
        -   [1. Built-in CUDA-Q Optimizers and
            Gradients](examples/python/optimizers_gradients.html#1.-Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
            -   [1.1 Adam Optimizer with Parameter
                Configuration](examples/python/optimizers_gradients.html#1.1-Adam-Optimizer-with-Parameter-Configuration){.reference
                .internal}
            -   [1.2 SGD (Stochastic Gradient Descent)
                Optimizer](examples/python/optimizers_gradients.html#1.2-SGD-(Stochastic-Gradient-Descent)-Optimizer){.reference
                .internal}
            -   [1.3 SPSA (Simultaneous Perturbation Stochastic
                Approximation)](examples/python/optimizers_gradients.html#1.3-SPSA-(Simultaneous-Perturbation-Stochastic-Approximation)){.reference
                .internal}
        -   [2. Third-Party
            Optimizers](examples/python/optimizers_gradients.html#2.-Third-Party-Optimizers){.reference
            .internal}
        -   [3. Parallel Parameter Shift
            Gradients](examples/python/optimizers_gradients.html#3.-Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](examples/python/noisy_simulations.html){.reference
        .internal}
    -   [Pre-Trajectory Sampling with Batch
        Execution](using/examples/ptsbe.html){.reference .internal}
        -   [Conceptual
            Overview](using/examples/ptsbe.html#conceptual-overview){.reference
            .internal}
        -   [When to Use
            PTSBE](using/examples/ptsbe.html#when-to-use-ptsbe){.reference
            .internal}
        -   [Quick
            Start](using/examples/ptsbe.html#quick-start){.reference
            .internal}
        -   [Usage
            Tutorial](using/examples/ptsbe.html#usage-tutorial){.reference
            .internal}
            -   [Controlling the Number of
                Trajectories](using/examples/ptsbe.html#controlling-the-number-of-trajectories){.reference
                .internal}
            -   [Choosing a Trajectory Sampling
                Strategy](using/examples/ptsbe.html#choosing-a-trajectory-sampling-strategy){.reference
                .internal}
            -   [Shot Allocation
                Strategies](using/examples/ptsbe.html#shot-allocation-strategies){.reference
                .internal}
            -   [Inspecting Execution
                Data](using/examples/ptsbe.html#inspecting-execution-data){.reference
                .internal}
    -   [Constructing
        Operators](using/examples/operators.html){.reference .internal}
        -   [Constructing Spin
            Operators](using/examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](using/examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](using/examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](using/examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](using/examples/hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](using/examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](using/examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](using/examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](using/examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](using/examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](using/examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](using/examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum Circuits,
            Inc.](using/examples/hardware_providers.html#quantum-circuits-inc){.reference
            .internal}
        -   [Quantum
            Machines](using/examples/hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](using/examples/hardware_providers.html#quera-computing){.reference
            .internal}
        -   [Scaleway](using/examples/hardware_providers.html#scaleway){.reference
            .internal}
        -   [TII](using/examples/hardware_providers.html#tii){.reference
            .internal}
    -   [When to Use sample vs.
        run](using/examples/sample_vs_run.html){.reference .internal}
        -   [Introduction](using/examples/sample_vs_run.html#introduction){.reference
            .internal}
        -   [Usage
            Guidelines](using/examples/sample_vs_run.html#usage-guidelines){.reference
            .internal}
        -   [What Is Supported with [`sample`{.docutils .literal
            .notranslate}]{.pre}](using/examples/sample_vs_run.html#what-is-supported-with-sample){.reference
            .internal}
        -   [What Is Not Supported with [`sample`{.docutils .literal
            .notranslate}]{.pre}](using/examples/sample_vs_run.html#what-is-not-supported-with-sample){.reference
            .internal}
        -   [How to
            Migrate](using/examples/sample_vs_run.html#how-to-migrate){.reference
            .internal}
            -   [Step 1: Add a return type to the
                kernel](using/examples/sample_vs_run.html#step-1-add-a-return-type-to-the-kernel){.reference
                .internal}
            -   [Step 2: Replace [`sample`{.docutils .literal
                .notranslate}]{.pre} with [`run`{.docutils .literal
                .notranslate}]{.pre}](using/examples/sample_vs_run.html#step-2-replace-sample-with-run){.reference
                .internal}
            -   [Step 3: Update result
                processing](using/examples/sample_vs_run.html#step-3-update-result-processing){.reference
                .internal}
        -   [Migration
            Examples](using/examples/sample_vs_run.html#migration-examples){.reference
            .internal}
            -   [Example 1: Simple conditional
                logic](using/examples/sample_vs_run.html#example-1-simple-conditional-logic){.reference
                .internal}
            -   [Example 2: Returning multiple measurement
                results](using/examples/sample_vs_run.html#example-2-returning-multiple-measurement-results){.reference
                .internal}
            -   [Example 3: Quantum
                teleportation](using/examples/sample_vs_run.html#example-3-quantum-teleportation){.reference
                .internal}
        -   [Additional
            Notes](using/examples/sample_vs_run.html#additional-notes){.reference
            .internal}
    -   [Dynamics
        Examples](using/examples/dynamics_examples.html){.reference
        .internal}
        -   [Python Examples (Jupyter
            Notebooks)](using/examples/dynamics_examples.html#python-examples-jupyter-notebooks){.reference
            .internal}
            -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
                Model)](examples/python/dynamics/dynamics_intro_1.html){.reference
                .internal}
            -   [Introduction to CUDA-Q Dynamics (Time Dependent
                Hamiltonians)](examples/python/dynamics/dynamics_intro_2.html){.reference
                .internal}
            -   [Superconducting
                Qubits](examples/python/dynamics/superconducting.html){.reference
                .internal}
            -   [Spin
                Qubits](examples/python/dynamics/spinqubits.html){.reference
                .internal}
            -   [Trapped Ion
                Qubits](examples/python/dynamics/iontrap.html){.reference
                .internal}
            -   [Control](examples/python/dynamics/control.html){.reference
                .internal}
        -   [C++
            Examples](using/examples/dynamics_examples.html#c-examples){.reference
            .internal}
            -   [Introduction: Single Qubit
                Dynamics](using/examples/dynamics_examples.html#introduction-single-qubit-dynamics){.reference
                .internal}
            -   [Introduction: Cavity QED (Jaynes-Cummings
                Model)](using/examples/dynamics_examples.html#introduction-cavity-qed-jaynes-cummings-model){.reference
                .internal}
            -   [Superconducting Qubits: Cross-Resonance
                Gate](using/examples/dynamics_examples.html#superconducting-qubits-cross-resonance-gate){.reference
                .internal}
            -   [Spin Qubits: Heisenberg Spin
                Chain](using/examples/dynamics_examples.html#spin-qubits-heisenberg-spin-chain){.reference
                .internal}
            -   [Control: Driven
                Qubit](using/examples/dynamics_examples.html#control-driven-qubit){.reference
                .internal}
            -   [State
                Batching](using/examples/dynamics_examples.html#state-batching){.reference
                .internal}
            -   [Numerical
                Integrators](using/examples/dynamics_examples.html#numerical-integrators){.reference
                .internal}
-   [Applications](using/applications.html){.reference .internal}
    -   [Max-Cut with QAOA](applications/python/qaoa.html){.reference
        .internal}
    -   [Molecular docking via
        DC-QAOA](applications/python/digitized_counterdiabatic_qaoa.html){.reference
        .internal}
        -   [Setting up the Molecular Docking
            Problem](applications/python/digitized_counterdiabatic_qaoa.html#Setting-up-the-Molecular-Docking-Problem){.reference
            .internal}
        -   [CUDA-Q
            Implementation](applications/python/digitized_counterdiabatic_qaoa.html#CUDA-Q-Implementation){.reference
            .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](applications/python/krylov.html){.reference .internal}
        -   [Setup](applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Quantum-Selected Configuration Interaction
        (QSCI)](applications/python/qsci.html){.reference .internal}
        -   [0. Problem
            definition](applications/python/qsci.html#0.-Problem-definition){.reference
            .internal}
        -   [1. Prepare an Approximate Quantum
            State](applications/python/qsci.html#1.-Prepare-an-Approximate-Quantum-State){.reference
            .internal}
        -   [2 Quantum Sampling to Select
            Configuration](applications/python/qsci.html#2-Quantum-Sampling-to-Select-Configuration){.reference
            .internal}
        -   [3. Classical Diagonalization on the Selected
            Subspace](applications/python/qsci.html#3.-Classical-Diagonalization-on-the-Selected-Subspace){.reference
            .internal}
        -   [5. Compare
            results](applications/python/qsci.html#5.-Compare-results){.reference
            .internal}
        -   [Reference](applications/python/qsci.html#Reference){.reference
            .internal}
    -   [Bernstein-Vazirani
        Algorithm](applications/python/bernstein_vazirani.html){.reference
        .internal}
        -   [Classical
            case](applications/python/bernstein_vazirani.html#Classical-case){.reference
            .internal}
        -   [Quantum
            case](applications/python/bernstein_vazirani.html#Quantum-case){.reference
            .internal}
        -   [Implementing in
            CUDA-Q](applications/python/bernstein_vazirani.html#Implementing-in-CUDA-Q){.reference
            .internal}
    -   [Cost
        Minimization](applications/python/cost_minimization.html){.reference
        .internal}
    -   [Deutsch's
        Algorithm](applications/python/deutsch_algorithm.html){.reference
        .internal}
        -   [XOR [\\(\\oplus\\)]{.math .notranslate
            .nohighlight}](applications/python/deutsch_algorithm.html#XOR-\oplus){.reference
            .internal}
        -   [Quantum
            oracles](applications/python/deutsch_algorithm.html#Quantum-oracles){.reference
            .internal}
        -   [Phase
            oracle](applications/python/deutsch_algorithm.html#Phase-oracle){.reference
            .internal}
        -   [Quantum
            parallelism](applications/python/deutsch_algorithm.html#Quantum-parallelism){.reference
            .internal}
        -   [Deutsch's
            Algorithm:](applications/python/deutsch_algorithm.html#Deutsch's-Algorithm:){.reference
            .internal}
    -   [Divisive Clustering With Coresets Using
        CUDA-Q](applications/python/divisive_clustering_coresets.html){.reference
        .internal}
        -   [Data
            preprocessing](applications/python/divisive_clustering_coresets.html#Data-preprocessing){.reference
            .internal}
        -   [Quantum
            functions](applications/python/divisive_clustering_coresets.html#Quantum-functions){.reference
            .internal}
        -   [Divisive Clustering
            Function](applications/python/divisive_clustering_coresets.html#Divisive-Clustering-Function){.reference
            .internal}
        -   [QAOA
            Implementation](applications/python/divisive_clustering_coresets.html#QAOA-Implementation){.reference
            .internal}
        -   [Scaling simulations with
            CUDA-Q](applications/python/divisive_clustering_coresets.html#Scaling-simulations-with-CUDA-Q){.reference
            .internal}
    -   [Hybrid Quantum Neural
        Networks](applications/python/hybrid_quantum_neural_networks.html){.reference
        .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using [`Sample`{.docutils .literal .notranslate}]{.pre} to
            perform the Hadamard
            test](applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. [`Compute`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`overlap`{.docutils .literal
                .notranslate}]{.pre}](applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum Fourier
        Transform](applications/python/quantum_fourier_transform.html){.reference
        .internal}
        -   [Quantum Fourier Transform
            revisited](applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform-revisited){.reference
            .internal}
    -   [Quantum
        Teleporation](applications/python/quantum_teleportation.html){.reference
        .internal}
        -   [Teleportation
            explained](applications/python/quantum_teleportation.html#Teleportation-explained){.reference
            .internal}
    -   [Quantum
        Volume](applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Compiling Unitaries Using Diffusion
        Models](applications/python/unitary_compilation_diffusion_models.html){.reference
        .internal}
        -   [Diffusion model
            pipeline](applications/python/unitary_compilation_diffusion_models.html#Diffusion-model-pipeline){.reference
            .internal}
        -   [Setup and load
            models](applications/python/unitary_compilation_diffusion_models.html#Setup-and-load-models){.reference
            .internal}
            -   [Load discrete
                model](applications/python/unitary_compilation_diffusion_models.html#Load-discrete-model){.reference
                .internal}
            -   [Load continuous
                model](applications/python/unitary_compilation_diffusion_models.html#Load-continuous-model){.reference
                .internal}
            -   [Create helper
                functions](applications/python/unitary_compilation_diffusion_models.html#Create-helper-functions){.reference
                .internal}
        -   [Unitary
            compilation](applications/python/unitary_compilation_diffusion_models.html#Unitary-compilation){.reference
            .internal}
            -   [Random
                unitary](applications/python/unitary_compilation_diffusion_models.html#Random-unitary){.reference
                .internal}
            -   [Discrete
                model](applications/python/unitary_compilation_diffusion_models.html#Discrete-model){.reference
                .internal}
            -   [Continuous
                model](applications/python/unitary_compilation_diffusion_models.html#Continuous-model){.reference
                .internal}
            -   [Quantum Fourier
                transform](applications/python/unitary_compilation_diffusion_models.html#Quantum-Fourier-transform){.reference
                .internal}
            -   [XXZ-Hamiltonian
                evolution](applications/python/unitary_compilation_diffusion_models.html#XXZ-Hamiltonian-evolution){.reference
                .internal}
        -   [Choosing the circuit you
            need](applications/python/unitary_compilation_diffusion_models.html#Choosing-the-circuit-you-need){.reference
            .internal}
    -   [VQE with gradients, active spaces, and gate
        fusion](applications/python/vqe_advanced.html){.reference
        .internal}
        -   [The Basics of
            VQE](applications/python/vqe_advanced.html#The-Basics-of-VQE){.reference
            .internal}
        -   [Installing/Loading Relevant
            Packages](applications/python/vqe_advanced.html#Installing/Loading-Relevant-Packages){.reference
            .internal}
        -   [Implementing VQE in
            CUDA-Q](applications/python/vqe_advanced.html#Implementing-VQE-in-CUDA-Q){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](applications/python/vqe_advanced.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
        -   [Using an Active
            Space](applications/python/vqe_advanced.html#Using-an-Active-Space){.reference
            .internal}
        -   [Gate Fusion for Larger
            Circuits](applications/python/vqe_advanced.html#Gate-Fusion-for-Larger-Circuits){.reference
            .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](applications/python/afqmc.html){.reference .internal}
        -   [Hamiltonian preparation for
            VQE](applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [ADAPT-QAOA
        algorithm](applications/python/adapt_qaoa.html){.reference
        .internal}
        -   [Simulation
            input:](applications/python/adapt_qaoa.html#Simulation-input:){.reference
            .internal}
        -   [The problem Hamiltonian [\\(H_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A_j\\)]{.math .notranslate
            .nohighlight}:](applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H_C,A_j\]\\)]{.math .notranslate
            .nohighlight}:](applications/python/adapt_qaoa.html#The-commutator-%5BH_C,A_j%5D:){.reference
            .internal}
        -   [Beginning of ADAPT-QAOA
            iteration:](applications/python/adapt_qaoa.html#Beginning-of-ADAPT-QAOA-iteration:){.reference
            .internal}
    -   [ADAPT-VQE
        algorithm](applications/python/adapt_vqe.html){.reference
        .internal}
        -   [Classical
            pre-processing](applications/python/adapt_vqe.html#Classical-pre-processing){.reference
            .internal}
        -   [Jordan
            Wigner:](applications/python/adapt_vqe.html#Jordan-Wigner:){.reference
            .internal}
        -   [UCCSD operator
            pool](applications/python/adapt_vqe.html#UCCSD-operator-pool){.reference
            .internal}
            -   [Single
                excitation](applications/python/adapt_vqe.html#Single-excitation){.reference
                .internal}
            -   [Double
                excitation](applications/python/adapt_vqe.html#Double-excitation){.reference
                .internal}
        -   [Commutator \[[\\(H\\)]{.math .notranslate .nohighlight},
            [\\(A_i\\)]{.math .notranslate
            .nohighlight}\]](applications/python/adapt_vqe.html#Commutator-%5BH,-A_i%5D){.reference
            .internal}
        -   [Reference
            State:](applications/python/adapt_vqe.html#Reference-State:){.reference
            .internal}
        -   [Quantum
            kernels:](applications/python/adapt_vqe.html#Quantum-kernels:){.reference
            .internal}
        -   [Beginning of
            ADAPT-VQE:](applications/python/adapt_vqe.html#Beginning-of-ADAPT-VQE:){.reference
            .internal}
    -   [Quantum edge
        detection](applications/python/edge_detection.html){.reference
        .internal}
        -   [Image](applications/python/edge_detection.html#Image){.reference
            .internal}
        -   [Quantum Probability Image Encoding
            (QPIE):](applications/python/edge_detection.html#Quantum-Probability-Image-Encoding-(QPIE):){.reference
            .internal}
            -   [Below we show how to encode an image using QPIE in
                cudaq.](applications/python/edge_detection.html#Below-we-show-how-to-encode-an-image-using-QPIE-in-cudaq.){.reference
                .internal}
        -   [Flexible Representation of Quantum Images
            (FRQI):](applications/python/edge_detection.html#Flexible-Representation-of-Quantum-Images-(FRQI):){.reference
            .internal}
            -   [Building the FRQI
                State:](applications/python/edge_detection.html#Building-the-FRQI-State:){.reference
                .internal}
        -   [Quantum Hadamard Edge Detection
            (QHED)](applications/python/edge_detection.html#Quantum-Hadamard-Edge-Detection-(QHED)){.reference
            .internal}
            -   [Post-processing](applications/python/edge_detection.html#Post-processing){.reference
                .internal}
    -   [Factoring Integers With Shor's
        Algorithm](applications/python/shors.html){.reference .internal}
        -   [Shor's
            algorithm](applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
            -   [(a) Generate the molecular Hamiltonian using Restricted
                Hartree Fock molecular
                orbitals](applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Restricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(b) Generate the molecular Hamiltonian using
                Unrestricted Hartree Fock molecular
                orbitals](applications/python/generate_fermionic_ham.html#(b)-Generate-the-molecular-Hamiltonian-using-Unrestricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(a) Generate the active space hamiltonian using RHF
                molecular
                orbitals.](applications/python/generate_fermionic_ham.html#(a)-Generate-the-active-space-hamiltonian-using-RHF-molecular-orbitals.){.reference
                .internal}
            -   [(b) Generate the active space Hamiltonian using the
                natural orbitals computed from MP2
                simulation](applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
                .internal}
            -   [(c) Generate the active space Hamiltonian computed from
                the CASSCF molecular
                orbitals](applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
                .internal}
            -   [(d) Generate the electronic Hamiltonian using
                ROHF](applications/python/generate_fermionic_ham.html#(d)-Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
            -   [(e) Generate electronic Hamiltonian using
                UHF](applications/python/generate_fermionic_ham.html#(e)-Generate-electronic-Hamiltonian-using-UHF){.reference
                .internal}
    -   [Grover's
        Algorithm](applications/python/grovers.html){.reference
        .internal}
        -   [Overview](applications/python/grovers.html#Overview){.reference
            .internal}
        -   [Problem](applications/python/grovers.html#Problem){.reference
            .internal}
        -   [Structure of Grover's
            Algorithm](applications/python/grovers.html#Structure-of-Grover's-Algorithm){.reference
            .internal}
            -   [Step 1:
                Preparation](applications/python/grovers.html#Step-1:-Preparation){.reference
                .internal}
            -   [Good and Bad
                States](applications/python/grovers.html#Good-and-Bad-States){.reference
                .internal}
            -   [Step 2: Oracle
                application](applications/python/grovers.html#Step-2:-Oracle-application){.reference
                .internal}
            -   [Step 3: Amplitude
                amplification](applications/python/grovers.html#Step-3:-Amplitude-amplification){.reference
                .internal}
            -   [Steps 4 and 5: Iteration and
                measurement](applications/python/grovers.html#Steps-4-and-5:-Iteration-and-measurement){.reference
                .internal}
    -   [Quantum
        PageRank](applications/python/quantum_pagerank.html){.reference
        .internal}
        -   [Problem
            Definition](applications/python/quantum_pagerank.html#Problem-Definition){.reference
            .internal}
        -   [Simulating Quantum PageRank by CUDA-Q
            dynamics](applications/python/quantum_pagerank.html#Simulating-Quantum-PageRank-by-CUDA-Q-dynamics){.reference
            .internal}
        -   [Breakdown of
            Terms](applications/python/quantum_pagerank.html#Breakdown-of-Terms){.reference
            .internal}
    -   [The UCCSD Wavefunction
        ansatz](applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [QM/MM simulation: VQE within a Polarizable Embedded
        Framework.](applications/python/qm_mm_pe.html){.reference
        .internal}
        -   [Key
            concepts:](applications/python/qm_mm_pe.html#Key-concepts:){.reference
            .internal}
        -   [PE-VQE-SCF Algorithm
            Steps](applications/python/qm_mm_pe.html#PE-VQE-SCF-Algorithm-Steps){.reference
            .internal}
            -   [Step 1: Initialize (Classical
                pre-processing)](applications/python/qm_mm_pe.html#Step-1:-Initialize-(Classical-pre-processing)){.reference
                .internal}
            -   [Step 2: Build the
                Hamiltonian](applications/python/qm_mm_pe.html#Step-2:-Build-the-Hamiltonian){.reference
                .internal}
            -   [Step 3: Run
                VQE](applications/python/qm_mm_pe.html#Step-3:-Run-VQE){.reference
                .internal}
            -   [Step 4: Update
                Environment](applications/python/qm_mm_pe.html#Step-4:-Update-Environment){.reference
                .internal}
            -   [Step 5: Self-Consistency
                Loop](applications/python/qm_mm_pe.html#Step-5:-Self-Consistency-Loop){.reference
                .internal}
            -   [Requirments:](applications/python/qm_mm_pe.html#Requirments:){.reference
                .internal}
            -   [Example 1: LiH with 2 water
                molecules.](applications/python/qm_mm_pe.html#Example-1:-LiH-with-2-water-molecules.){.reference
                .internal}
            -   [VQE, update environment, and scf
                loop.](applications/python/qm_mm_pe.html#VQE,-update-environment,-and-scf-loop.){.reference
                .internal}
            -   [Example 2: NH3 with 46 water molecule using active
                space.](applications/python/qm_mm_pe.html#Example-2:-NH3-with-46-water-molecule-using-active-space.){.reference
                .internal}
    -   [Sample-Based Krylov Quantum Diagonalization
        (SKQD)](applications/python/skqd.html){.reference .internal}
        -   [Why
            SKQD?](applications/python/skqd.html#Why-SKQD?){.reference
            .internal}
        -   [Understanding Krylov
            Subspaces](applications/python/skqd.html#Understanding-Krylov-Subspaces){.reference
            .internal}
            -   [What is a Krylov
                Subspace?](applications/python/skqd.html#What-is-a-Krylov-Subspace?){.reference
                .internal}
            -   [The SKQD
                Algorithm](applications/python/skqd.html#The-SKQD-Algorithm){.reference
                .internal}
        -   [Problem Setup: 22-Qubit Heisenberg
            Model](applications/python/skqd.html#Problem-Setup:-22-Qubit-Heisenberg-Model){.reference
            .internal}
        -   [Krylov State Generation via Repeated
            Evolution](applications/python/skqd.html#Krylov-State-Generation-via-Repeated-Evolution){.reference
            .internal}
        -   [Quantum Measurements and
            Sampling](applications/python/skqd.html#Quantum-Measurements-and-Sampling){.reference
            .internal}
            -   [The Sampling
                Process](applications/python/skqd.html#The-Sampling-Process){.reference
                .internal}
        -   [Classical Post-Processing and
            Diagonalization](applications/python/skqd.html#Classical-Post-Processing-and-Diagonalization){.reference
            .internal}
            -   [Matrix Construction
                Details](applications/python/skqd.html#Matrix-Construction-Details){.reference
                .internal}
            -   [Approach 1: GPU-Vectorized CSR Sparse
                Matrix](applications/python/skqd.html#Approach-1:-GPU-Vectorized-CSR-Sparse-Matrix){.reference
                .internal}
            -   [Approach 2: Matrix-Free Lanczos via
                [`distributed_eigsh`{.docutils .literal
                .notranslate}]{.pre}](applications/python/skqd.html#Approach-2:-Matrix-Free-Lanczos-via-distributed_eigsh){.reference
                .internal}
        -   [Results Analysis and
            Convergence](applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
        -   [Postprocessing Acceleration: CSR matrix approach, single
            GPU vs
            CPU](applications/python/skqd.html#Postprocessing-Acceleration:-CSR-matrix-approach,-single-GPU-vs-CPU){.reference
            .internal}
        -   [Postprocessing Scale-Up and Scale-Out: Linear Operator
            Approach, Multi-GPU
            Multi-Node](applications/python/skqd.html#Postprocessing-Scale-Up-and-Scale-Out:-Linear-Operator-Approach,-Multi-GPU-Multi-Node){.reference
            .internal}
            -   [Saving Hamiltonian
                Data](applications/python/skqd.html#Saving-Hamiltonian-Data){.reference
                .internal}
            -   [Running the Distributed
                Solver](applications/python/skqd.html#Running-the-Distributed-Solver){.reference
                .internal}
        -   [Summary](applications/python/skqd.html#Summary){.reference
            .internal}
    -   [Entanglement Accelerates Quantum
        Simulation](applications/python/entanglement_acc_hamiltonian_simulation.html){.reference
        .internal}
        -   [2. Model
            Definition](applications/python/entanglement_acc_hamiltonian_simulation.html#2.-Model-Definition){.reference
            .internal}
            -   [2.1 Initial product
                state](applications/python/entanglement_acc_hamiltonian_simulation.html#2.1-Initial-product-state){.reference
                .internal}
            -   [2.2 QIMF
                Hamiltonian](applications/python/entanglement_acc_hamiltonian_simulation.html#2.2-QIMF-Hamiltonian){.reference
                .internal}
            -   [2.3 First-Order Trotter Formula
                (PF1)](applications/python/entanglement_acc_hamiltonian_simulation.html#2.3-First-Order-Trotter-Formula-(PF1)){.reference
                .internal}
            -   [2.4 PF1 step for the QIMF
                partition](applications/python/entanglement_acc_hamiltonian_simulation.html#2.4-PF1-step-for-the-QIMF-partition){.reference
                .internal}
            -   [2.5 Hamiltonian
                helpers](applications/python/entanglement_acc_hamiltonian_simulation.html#2.5-Hamiltonian-helpers){.reference
                .internal}
        -   [3. Entanglement
            metrics](applications/python/entanglement_acc_hamiltonian_simulation.html#3.-Entanglement-metrics){.reference
            .internal}
        -   [4. Simulation
            workflow](applications/python/entanglement_acc_hamiltonian_simulation.html#4.-Simulation-workflow){.reference
            .internal}
            -   [4.1 Single-step Trotter
                error](applications/python/entanglement_acc_hamiltonian_simulation.html#4.1-Single-step-Trotter-error){.reference
                .internal}
            -   [4.2 Dual trajectory
                update](applications/python/entanglement_acc_hamiltonian_simulation.html#4.2-Dual-trajectory-update){.reference
                .internal}
        -   [5. Reproducing the paper's Figure
            1a](applications/python/entanglement_acc_hamiltonian_simulation.html#5.-Reproducing-the-paper’s-Figure-1a){.reference
            .internal}
            -   [5.1 Visualising the joint
                behaviour](applications/python/entanglement_acc_hamiltonian_simulation.html#5.1-Visualising-the-joint-behaviour){.reference
                .internal}
            -   [5.2 Interpreting the
                result](applications/python/entanglement_acc_hamiltonian_simulation.html#5.2-Interpreting-the-result){.reference
                .internal}
        -   [6. References and further
            reading](applications/python/entanglement_acc_hamiltonian_simulation.html#6.-References-and-further-reading){.reference
            .internal}
    -   [Pre-Trajectory Sampling with Batch Execution
        (PTSBE)](applications/python/ptsbe.html){.reference .internal}
        -   [Set up the
            environment](applications/python/ptsbe.html#Set-up-the-environment){.reference
            .internal}
        -   [Define the circuit and noise
            model](applications/python/ptsbe.html#Define-the-circuit-and-noise-model){.reference
            .internal}
            -   [Inline noise with [`apply_noise`{.docutils .literal
                .notranslate}]{.pre}](applications/python/ptsbe.html#Inline-noise-with-apply_noise){.reference
                .internal}
        -   [Run PTSBE
            sampling](applications/python/ptsbe.html#Run-PTSBE-sampling){.reference
            .internal}
            -   [Larger circuit for execution
                data](applications/python/ptsbe.html#Larger-circuit-for-execution-data){.reference
                .internal}
        -   [Inspecting trajectories with execution
            data](applications/python/ptsbe.html#Inspecting-trajectories-with-execution-data){.reference
            .internal}
        -   [Performance of PTSBE vs standard noisy
            sampling](applications/python/ptsbe.html#Performance-of-PTSBE-vs-standard-noisy-sampling){.reference
            .internal}
-   [Backends](using/backends/backends.html){.reference .internal}
    -   [Circuit Simulation](using/backends/simulators.html){.reference
        .internal}
        -   [State Vector
            Simulators](using/backends/sims/svsims.html){.reference
            .internal}
            -   [CPU](using/backends/sims/svsims.html#cpu){.reference
                .internal}
            -   [Single-GPU](using/backends/sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](using/backends/sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network
            Simulators](using/backends/sims/tnsims.html){.reference
            .internal}
            -   [Multi-GPU
                multi-node](using/backends/sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](using/backends/sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](using/backends/sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU
            Simulators](using/backends/sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](using/backends/sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](using/backends/sims/mqpusims.html#multi-qpu-other-backends){.reference
                .internal}
        -   [Noisy
            Simulators](using/backends/sims/noisy.html){.reference
            .internal}
            -   [Trajectory Noisy
                Simulation](using/backends/sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](using/backends/sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](using/backends/sims/noisy.html#stim){.reference
                .internal}
        -   [Photonics
            Simulators](using/backends/sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](using/backends/sims/photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware
        (QPUs)](using/backends/hardware.html){.reference .internal}
        -   [Ion Trap
            QPUs](using/backends/hardware/iontrap.html){.reference
            .internal}
            -   [IonQ](using/backends/hardware/iontrap.html#ionq){.reference
                .internal}
            -   [Quantinuum](using/backends/hardware/iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting
            QPUs](using/backends/hardware/superconducting.html){.reference
            .internal}
            -   [Anyon Technologies/Anyon
                Computing](using/backends/hardware/superconducting.html#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](using/backends/hardware/superconducting.html#iqm){.reference
                .internal}
            -   [OQC](using/backends/hardware/superconducting.html#oqc){.reference
                .internal}
            -   [Quantum Circuits,
                Inc.](using/backends/hardware/superconducting.html#quantum-circuits-inc){.reference
                .internal}
            -   [TII](using/backends/hardware/superconducting.html#tii){.reference
                .internal}
        -   [Neutral Atom
            QPUs](using/backends/hardware/neutralatom.html){.reference
            .internal}
            -   [Infleqtion](using/backends/hardware/neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](using/backends/hardware/neutralatom.html#pasqal){.reference
                .internal}
            -   [QuEra
                Computing](using/backends/hardware/neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic
            QPUs](using/backends/hardware/photonic.html){.reference
            .internal}
            -   [ORCA
                Computing](using/backends/hardware/photonic.html#orca-computing){.reference
                .internal}
        -   [Quantum Control
            Systems](using/backends/hardware/qcontrol.html){.reference
            .internal}
            -   [Quantum
                Machines](using/backends/hardware/qcontrol.html#quantum-machines){.reference
                .internal}
    -   [Dynamics
        Simulation](using/backends/dynamics_backends.html){.reference
        .internal}
    -   [Cloud](using/backends/cloud.html){.reference .internal}
        -   [Amazon Braket
            (braket)](using/backends/cloud/braket.html){.reference
            .internal}
            -   [Setting
                Credentials](using/backends/cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submitting](using/backends/cloud/braket.html#submitting){.reference
                .internal}
        -   [Scaleway QaaS
            (scaleway)](using/backends/cloud/scaleway.html){.reference
            .internal}
            -   [Setting
                Credentials](using/backends/cloud/scaleway.html#setting-credentials){.reference
                .internal}
            -   [Submitting](using/backends/cloud/scaleway.html#submitting){.reference
                .internal}
            -   [Manage your QPU
                session](using/backends/cloud/scaleway.html#manage-your-qpu-session){.reference
                .internal}
-   [Dynamics](using/dynamics.html){.reference .internal}
    -   [Quick Start](using/dynamics.html#quick-start){.reference
        .internal}
    -   [Operator](using/dynamics.html#operator){.reference .internal}
    -   [Time-Dependent
        Dynamics](using/dynamics.html#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](using/dynamics.html#super-operator-representation){.reference
        .internal}
    -   [Numerical
        Integrators](using/dynamics.html#numerical-integrators){.reference
        .internal}
    -   [Batch
        simulation](using/dynamics.html#batch-simulation){.reference
        .internal}
    -   [Multi-GPU Multi-Node
        Execution](using/dynamics.html#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](using/dynamics.html#examples){.reference .internal}
-   [Realtime](using/realtime.html){.reference .internal}
    -   [Installation](using/realtime/installation.html){.reference
        .internal}
        -   [Prerequisites](using/realtime/installation.html#prerequisites){.reference
            .internal}
        -   [Setup](using/realtime/installation.html#setup){.reference
            .internal}
        -   [Latency
            Measurement](using/realtime/installation.html#latency-measurement){.reference
            .internal}
    -   [Host API](using/realtime/host.html){.reference .internal}
        -   [What is
            HSB?](using/realtime/host.html#what-is-hsb){.reference
            .internal}
        -   [Transport
            Mechanisms](using/realtime/host.html#transport-mechanisms){.reference
            .internal}
            -   [Supported Transport
                Options](using/realtime/host.html#supported-transport-options){.reference
                .internal}
        -   [The 3-Kernel Architecture (HSB Example)
            {#three-kernel-architecture}](using/realtime/host.html#the-3-kernel-architecture-hsb-example-three-kernel-architecture){.reference
            .internal}
            -   [Data Flow
                Summary](using/realtime/host.html#data-flow-summary){.reference
                .internal}
            -   [Why 3
                Kernels?](using/realtime/host.html#why-3-kernels){.reference
                .internal}
        -   [Unified Dispatch
            Mode](using/realtime/host.html#unified-dispatch-mode){.reference
            .internal}
            -   [Architecture](using/realtime/host.html#architecture){.reference
                .internal}
            -   [Transport-Agnostic
                Design](using/realtime/host.html#transport-agnostic-design){.reference
                .internal}
            -   [When to Use Which
                Mode](using/realtime/host.html#when-to-use-which-mode){.reference
                .internal}
            -   [Host API
                Extensions](using/realtime/host.html#host-api-extensions){.reference
                .internal}
            -   [Wiring Example (Unified Mode with
                HSB)](using/realtime/host.html#wiring-example-unified-mode-with-hsb){.reference
                .internal}
        -   [What This API Does (In One
            Paragraph)](using/realtime/host.html#what-this-api-does-in-one-paragraph){.reference
            .internal}
        -   [Scope](using/realtime/host.html#scope){.reference
            .internal}
        -   [Terms and
            Components](using/realtime/host.html#terms-and-components){.reference
            .internal}
        -   [Schema Data
            Structures](using/realtime/host.html#schema-data-structures){.reference
            .internal}
            -   [Type
                Descriptors](using/realtime/host.html#type-descriptors){.reference
                .internal}
            -   [Handler
                Schema](using/realtime/host.html#handler-schema){.reference
                .internal}
        -   [RPC Messaging
            Protocol](using/realtime/host.html#rpc-messaging-protocol){.reference
            .internal}
        -   [Host API
            Overview](using/realtime/host.html#host-api-overview){.reference
            .internal}
        -   [Manager and Dispatcher
            Topology](using/realtime/host.html#manager-and-dispatcher-topology){.reference
            .internal}
        -   [Host API
            Functions](using/realtime/host.html#host-api-functions){.reference
            .internal}
            -   [Occupancy Query and Eager Module
                Loading](using/realtime/host.html#occupancy-query-and-eager-module-loading){.reference
                .internal}
            -   [Graph-Based Dispatch
                Functions](using/realtime/host.html#graph-based-dispatch-functions){.reference
                .internal}
            -   [Kernel Launch Helper
                Functions](using/realtime/host.html#kernel-launch-helper-functions){.reference
                .internal}
        -   [Memory Layout and Ring Buffer
            Wiring](using/realtime/host.html#memory-layout-and-ring-buffer-wiring){.reference
            .internal}
        -   [Step-by-Step: Wiring the Host API
            (Minimal)](using/realtime/host.html#step-by-step-wiring-the-host-api-minimal){.reference
            .internal}
        -   [Device Handler and Function
            ID](using/realtime/host.html#device-handler-and-function-id){.reference
            .internal}
            -   [Multi-Argument Handler
                Example](using/realtime/host.html#multi-argument-handler-example){.reference
                .internal}
        -   [CUDA Graph Dispatch
            Mode](using/realtime/host.html#cuda-graph-dispatch-mode){.reference
            .internal}
            -   [Requirements](using/realtime/host.html#requirements){.reference
                .internal}
            -   [Graph-Based Dispatch
                API](using/realtime/host.html#graph-based-dispatch-api){.reference
                .internal}
            -   [Graph Handler Setup
                Example](using/realtime/host.html#graph-handler-setup-example){.reference
                .internal}
            -   [Graph Capture and
                Instantiation](using/realtime/host.html#graph-capture-and-instantiation){.reference
                .internal}
            -   [When to Use Graph
                Dispatch](using/realtime/host.html#when-to-use-graph-dispatch){.reference
                .internal}
            -   [Graph vs Device Call
                Dispatch](using/realtime/host.html#graph-vs-device-call-dispatch){.reference
                .internal}
        -   [Building and Sending an RPC
            Message](using/realtime/host.html#building-and-sending-an-rpc-message){.reference
            .internal}
        -   [Reading the
            Response](using/realtime/host.html#reading-the-response){.reference
            .internal}
        -   [Schema-Driven Argument
            Parsing](using/realtime/host.html#schema-driven-argument-parsing){.reference
            .internal}
        -   [HSB 3-Kernel Workflow
            (Primary)](using/realtime/host.html#hsb-3-kernel-workflow-primary){.reference
            .internal}
        -   [NIC-Free Testing (No HSB / No
            ConnectX-7)](using/realtime/host.html#nic-free-testing-no-hsb-no-connectx-7){.reference
            .internal}
        -   [Troubleshooting](using/realtime/host.html#troubleshooting){.reference
            .internal}
    -   [Messaging Protocol](using/realtime/protocol.html){.reference
        .internal}
        -   [Scope](using/realtime/protocol.html#scope){.reference
            .internal}
        -   [RPC Header /
            Response](using/realtime/protocol.html#rpc-header-response){.reference
            .internal}
        -   [Request ID
            Semantics](using/realtime/protocol.html#request-id-semantics){.reference
            .internal}
        -   [[`PTP`{.docutils .literal .notranslate}]{.pre} Timestamp
            Semantics](using/realtime/protocol.html#ptp-timestamp-semantics){.reference
            .internal}
        -   [Function ID
            Semantics](using/realtime/protocol.html#function-id-semantics){.reference
            .internal}
        -   [Schema and Payload
            Interpretation](using/realtime/protocol.html#schema-and-payload-interpretation){.reference
            .internal}
            -   [Type
                System](using/realtime/protocol.html#type-system){.reference
                .internal}
        -   [Payload
            Encoding](using/realtime/protocol.html#payload-encoding){.reference
            .internal}
            -   [Single-Argument
                Payloads](using/realtime/protocol.html#single-argument-payloads){.reference
                .internal}
            -   [Multi-Argument
                Payloads](using/realtime/protocol.html#multi-argument-payloads){.reference
                .internal}
            -   [Size
                Constraints](using/realtime/protocol.html#size-constraints){.reference
                .internal}
            -   [Encoding
                Examples](using/realtime/protocol.html#encoding-examples){.reference
                .internal}
            -   [Bit-Packed Data
                Encoding](using/realtime/protocol.html#bit-packed-data-encoding){.reference
                .internal}
            -   [Multi-Bit Measurement
                Encoding](using/realtime/protocol.html#multi-bit-measurement-encoding){.reference
                .internal}
        -   [Response
            Encoding](using/realtime/protocol.html#response-encoding){.reference
            .internal}
            -   [Single-Result
                Response](using/realtime/protocol.html#single-result-response){.reference
                .internal}
            -   [Multi-Result
                Response](using/realtime/protocol.html#multi-result-response){.reference
                .internal}
            -   [Status
                Codes](using/realtime/protocol.html#status-codes){.reference
                .internal}
        -   [QEC-Specific Usage
            Example](using/realtime/protocol.html#qec-specific-usage-example){.reference
            .internal}
            -   [QEC
                Terminology](using/realtime/protocol.html#qec-terminology){.reference
                .internal}
            -   [QEC Decoder
                Handler](using/realtime/protocol.html#qec-decoder-handler){.reference
                .internal}
            -   [Decoding
                Rounds](using/realtime/protocol.html#decoding-rounds){.reference
                .internal}
-   [CUDA-QX](using/cudaqx/cudaqx.html){.reference .internal}
    -   [CUDA-Q
        Solvers](using/cudaqx/cudaqx.html#cuda-q-solvers){.reference
        .internal}
    -   [CUDA-Q QEC](using/cudaqx/cudaqx.html#cuda-q-qec){.reference
        .internal}
-   [Installation](using/install/install.html){.reference .internal}
    -   [Local
        Installation](using/install/local_installation.html){.reference
        .internal}
        -   [Introduction](using/install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](using/install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](using/install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](using/install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](using/install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](using/install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](using/install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](using/install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](using/install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](using/install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](using/install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](using/install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](using/install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](using/install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](using/install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](using/install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](using/install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](using/install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](using/install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](using/install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](using/install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](using/install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](using/install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
        -   [Next
            Steps](using/install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](using/install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](using/install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](using/install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](using/install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](using/install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](using/install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](using/install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](using/install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](using/install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](using/install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](using/install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](using/integration/integration.html){.reference
    .internal}
    -   [Downstream CMake
        Integration](using/integration/cmake_app.html){.reference
        .internal}
    -   [Combining CUDA with
        CUDA-Q](using/integration/cuda_gpu.html){.reference .internal}
    -   [Integrating with Third-Party
        Libraries](using/integration/libraries.html){.reference
        .internal}
        -   [Calling a CUDA-Q library from
            C++](using/integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](using/integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](using/integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](using/extending/extending.html){.reference .internal}
    -   [Add a new Hardware
        Backend](using/extending/backend.html){.reference .internal}
        -   [Overview](using/extending/backend.html#overview){.reference
            .internal}
        -   [Server Helper
            Implementation](using/extending/backend.html#server-helper-implementation){.reference
            .internal}
            -   [Directory
                Structure](using/extending/backend.html#directory-structure){.reference
                .internal}
            -   [Server Helper
                Class](using/extending/backend.html#server-helper-class){.reference
                .internal}
            -   [[`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](using/extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](using/extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent [`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](using/extending/backend.html#update-parent-cmakelists-txt){.reference
                .internal}
        -   [Testing](using/extending/backend.html#testing){.reference
            .internal}
            -   [Unit
                Tests](using/extending/backend.html#unit-tests){.reference
                .internal}
            -   [Mock
                Server](using/extending/backend.html#mock-server){.reference
                .internal}
            -   [Python
                Tests](using/extending/backend.html#python-tests){.reference
                .internal}
            -   [Integration
                Tests](using/extending/backend.html#integration-tests){.reference
                .internal}
        -   [Documentation](using/extending/backend.html#documentation){.reference
            .internal}
        -   [Example
            Usage](using/extending/backend.html#example-usage){.reference
            .internal}
        -   [Code
            Review](using/extending/backend.html#code-review){.reference
            .internal}
        -   [Maintaining a
            Backend](using/extending/backend.html#maintaining-a-backend){.reference
            .internal}
        -   [Conclusion](using/extending/backend.html#conclusion){.reference
            .internal}
    -   [Create a new NVQIR
        Simulator](using/extending/nvqir_simulator.html){.reference
        .internal}
        -   [[`CircuitSimulator`{.code .docutils .literal
            .notranslate}]{.pre}](using/extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](using/extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
    -   [Working with CUDA-Q
        IR](using/extending/cudaq_ir.html){.reference .internal}
    -   [Create an MLIR Pass for
        CUDA-Q](using/extending/mlir_pass.html){.reference .internal}
-   [Specifications](specification/index.html){.reference .internal}
    -   [Language Specification](specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](specification/cudaq/types.html){.reference .internal}
            -   [3.1. [`cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. [`cudaq::qubit`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. [`cudaq::spin_op`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on [`cudaq::qubit`{.code .docutils
                .literal
                .notranslate}]{.pre}](specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](specification/cudaq/kernels.html){.reference
            .internal}
        -   [7. Sub-circuit
            Synthesis](specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11. Platform](specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. [`cudaq::sample`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. [`cudaq::run`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. [`cudaq::observe`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. [`cudaq::optimizer`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. [`cudaq::gradient`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](specification/quake-dialect.html#motivation){.reference
            .internal}
-   [API Reference](api/api.html){.reference .internal}
    -   [C++ API](api/languages/cpp_api.html){.reference .internal}
        -   [Operators](api/languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](api/languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](api/languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](api/languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](api/languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](api/languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Platform](api/languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](api/languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](api/languages/cpp_api.html#namespaces){.reference
            .internal}
        -   [PTSBE](api/languages/cpp_api.html#ptsbe){.reference
            .internal}
            -   [Sampling
                Functions](api/languages/cpp_api.html#sampling-functions){.reference
                .internal}
            -   [Options](api/languages/cpp_api.html#options){.reference
                .internal}
            -   [Result
                Type](api/languages/cpp_api.html#result-type){.reference
                .internal}
            -   [Trajectory Sampling
                Strategies](api/languages/cpp_api.html#trajectory-sampling-strategies){.reference
                .internal}
            -   [Shot Allocation
                Strategy](api/languages/cpp_api.html#shot-allocation-strategy){.reference
                .internal}
            -   [Execution
                Data](api/languages/cpp_api.html#execution-data){.reference
                .internal}
            -   [Trajectory and Selection
                Types](api/languages/cpp_api.html#trajectory-and-selection-types){.reference
                .internal}
    -   [Python API](api/languages/python_api.html){.reference
        .internal}
        -   [Program
            Construction](api/languages/python_api.html#program-construction){.reference
            .internal}
            -   [[`make_kernel()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [[`PyKernel`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [[`Kernel`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [[`PyKernelDecorator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [[`kernel()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [[`sample_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [[`run()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [[`run_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [[`observe()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [[`observe_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [[`get_state()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [[`get_state_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [[`vqe()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [[`draw()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [[`translate()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [[`estimate_resources()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
        -   [Backend
            Configuration](api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [[`has_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [[`get_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [[`get_targets()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [[`set_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [[`reset_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [[`set_noise()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [[`unset_noise()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [[`register_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [[`unregister_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [[`cudaq.apply_noise()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [[`initialize_cudaq()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [[`num_available_gpus()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [[`set_random_seed()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [[`evolve()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [[`evolve_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [[`Schedule`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [[`BaseIntegrator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [[`InitialState`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [[`InitialStateType`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [[`IntermediateResultSave`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](api/languages/python_api.html#operators){.reference
            .internal}
            -   [[`OperatorSum`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [[`ProductOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [[`ElementaryOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [[`ScalarOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [[`RydbergHamiltonian`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [[`SuperOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [[`operators.define()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [[`operators.instantiate()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](api/languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](api/languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](api/languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](api/languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](api/languages/python_api.html#data-types){.reference
            .internal}
            -   [[`SimulationPrecision`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [[`Target`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [[`State`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [[`Tensor`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [[`QuakeValue`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [[`qubit`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [[`qreg`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [[`qvector`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [[`ComplexMatrix`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [[`SampleResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [[`AsyncSampleResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [[`ObserveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [[`AsyncObserveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [[`AsyncStateResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [[`OptimizationResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [[`EvolveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [[`AsyncEvolveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [[`Resources`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Resources){.reference
                .internal}
            -   [Optimizers](api/languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](api/languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](api/languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](api/languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [[`initialize()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [[`rank()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [[`num_ranks()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [[`all_gather()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [[`broadcast()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [[`is_initialized()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [[`finalize()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
        -   [PTSBE
            Submodule](api/languages/python_api.html#ptsbe-submodule){.reference
            .internal}
            -   [Sampling
                Functions](api/languages/python_api.html#sampling-functions){.reference
                .internal}
            -   [Result
                Type](api/languages/python_api.html#result-type){.reference
                .internal}
            -   [Trajectory Sampling
                Strategies](api/languages/python_api.html#trajectory-sampling-strategies){.reference
                .internal}
            -   [Shot Allocation
                Strategy](api/languages/python_api.html#shot-allocation-strategy){.reference
                .internal}
            -   [Execution
                Data](api/languages/python_api.html#execution-data){.reference
                .internal}
            -   [Trajectory and Selection
                Types](api/languages/python_api.html#trajectory-and-selection-types){.reference
                .internal}
    -   [Quantum Operations](api/default_ops.html){.reference .internal}
        -   [Unitary Operations on
            Qubits](api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [[`x`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#x){.reference
                .internal}
            -   [[`y`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#y){.reference
                .internal}
            -   [[`z`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#z){.reference
                .internal}
            -   [[`h`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#h){.reference
                .internal}
            -   [[`r1`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#r1){.reference
                .internal}
            -   [[`rx`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#rx){.reference
                .internal}
            -   [[`ry`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#ry){.reference
                .internal}
            -   [[`rz`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#rz){.reference
                .internal}
            -   [[`s`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#s){.reference
                .internal}
            -   [[`t`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#t){.reference
                .internal}
            -   [[`swap`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#swap){.reference
                .internal}
            -   [[`u3`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#u3){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#mz){.reference
                .internal}
            -   [[`mx`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#mx){.reference
                .internal}
            -   [[`my`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [[`create`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#create){.reference
                .internal}
            -   [[`annihilate`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#annihilate){.reference
                .internal}
            -   [[`phase_shift`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#phase-shift){.reference
                .internal}
            -   [[`beam_splitter`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](index.html)

::: wy-nav-content
::: rst-content
::: {role="navigation" aria-label="Page navigation"}
-   [](index.html){.icon .icon-home aria-label="Home"}
-   Index
-   

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
# Index

::: genindex-jumpbox
[**\_**](#_) \| [**A**](#A) \| [**B**](#B) \| [**C**](#C) \| [**D**](#D)
\| [**E**](#E) \| [**F**](#F) \| [**G**](#G) \| [**H**](#H) \|
[**I**](#I) \| [**K**](#K) \| [**L**](#L) \| [**M**](#M) \| [**N**](#N)
\| [**O**](#O) \| [**P**](#P) \| [**Q**](#Q) \| [**R**](#R) \|
[**S**](#S) \| [**T**](#T) \| [**U**](#U) \| [**V**](#V) \| [**W**](#W)
\| [**X**](#X) \| [**Y**](#Y) \| [**Z**](#Z)
:::

## \_ {#_}

+-----------------------------------+-----------------------------------+
| -   [\_\_add\_\_()                | -   [\_\_iter\_\_()               |
|     (cudaq.QuakeValue             |     (cudaq.SampleResult           |
|                                   |     m                             |
|   method)](api/languages/python_a | ethod)](api/languages/python_api. |
| pi.html#cudaq.QuakeValue.__add__) | html#cudaq.SampleResult.__iter__) |
| -   [\_\_call\_\_()               | -   [\_\_len\_\_()                |
|     (cudaq.PyKernelDecorator      |     (cudaq.SampleResult           |
|     method                        |                                   |
| )](api/languages/python_api.html# | method)](api/languages/python_api |
| cudaq.PyKernelDecorator.__call__) | .html#cudaq.SampleResult.__len__) |
| -   [\_\_getitem\_\_()            | -   [\_\_mul\_\_()                |
|     (cudaq.ComplexMatrix          |     (cudaq.QuakeValue             |
|     metho                         |                                   |
| d)](api/languages/python_api.html |   method)](api/languages/python_a |
| #cudaq.ComplexMatrix.__getitem__) | pi.html#cudaq.QuakeValue.__mul__) |
|     -   [(cudaq.KrausChannel      | -   [\_\_neg\_\_()                |
|         meth                      |     (cudaq.QuakeValue             |
| od)](api/languages/python_api.htm |                                   |
| l#cudaq.KrausChannel.__getitem__) |   method)](api/languages/python_a |
|     -   [(cudaq.QuakeValue        | pi.html#cudaq.QuakeValue.__neg__) |
|         me                        | -   [\_\_radd\_\_()               |
| thod)](api/languages/python_api.h |     (cudaq.QuakeValue             |
| tml#cudaq.QuakeValue.__getitem__) |                                   |
|     -   [(cudaq.SampleResult      |  method)](api/languages/python_ap |
|         meth                      | i.html#cudaq.QuakeValue.__radd__) |
| od)](api/languages/python_api.htm | -   [\_\_rmul\_\_()               |
| l#cudaq.SampleResult.__getitem__) |     (cudaq.QuakeValue             |
| -   [\_\_init\_\_()               |                                   |
|                                   |  method)](api/languages/python_ap |
|    (cudaq.AmplitudeDampingChannel | i.html#cudaq.QuakeValue.__rmul__) |
|     method)](api                  | -   [\_\_rsub\_\_()               |
| /languages/python_api.html#cudaq. |     (cudaq.QuakeValue             |
| AmplitudeDampingChannel.__init__) |                                   |
|     -   [(cudaq.BitFlipChannel    |  method)](api/languages/python_ap |
|         met                       | i.html#cudaq.QuakeValue.__rsub__) |
| hod)](api/languages/python_api.ht | -   [\_\_str\_\_()                |
| ml#cudaq.BitFlipChannel.__init__) |     (cudaq.ComplexMatrix          |
|                                   |     m                             |
| -   [(cudaq.DepolarizationChannel | ethod)](api/languages/python_api. |
|         method)](a                | html#cudaq.ComplexMatrix.__str__) |
| pi/languages/python_api.html#cuda |     -   [(cudaq.PyKernelDecorator |
| q.DepolarizationChannel.__init__) |         metho                     |
|     -   [(cudaq.NoiseModel        | d)](api/languages/python_api.html |
|                                   | #cudaq.PyKernelDecorator.__str__) |
|  method)](api/languages/python_ap | -   [\_\_sub\_\_()                |
| i.html#cudaq.NoiseModel.__init__) |     (cudaq.QuakeValue             |
|     -   [(c                       |                                   |
| udaq.operators.RydbergHamiltonian |   method)](api/languages/python_a |
|         method)](api/lang         | pi.html#cudaq.QuakeValue.__sub__) |
| uages/python_api.html#cudaq.opera |                                   |
| tors.RydbergHamiltonian.__init__) |                                   |
|     -   [(cudaq.PhaseFlipChannel  |                                   |
|         metho                     |                                   |
| d)](api/languages/python_api.html |                                   |
| #cudaq.PhaseFlipChannel.__init__) |                                   |
+-----------------------------------+-----------------------------------+

## A {#A}

+-----------------------------------+-----------------------------------+
| -   [Adam (class in               | -   [append() (cudaq.KrausChannel |
|     cudaq                         |                                   |
| .optimizers)](api/languages/pytho |  method)](api/languages/python_ap |
| n_api.html#cudaq.optimizers.Adam) | i.html#cudaq.KrausChannel.append) |
| -   [add_all_qubit_channel()      | -   [argument_count               |
|     (cudaq.NoiseModel             |     (cudaq.PyKernel               |
|     method)](api                  |     attrib                        |
| /languages/python_api.html#cudaq. | ute)](api/languages/python_api.ht |
| NoiseModel.add_all_qubit_channel) | ml#cudaq.PyKernel.argument_count) |
| -   [add_channel()                | -   [arguments (cudaq.PyKernel    |
|     (cudaq.NoiseModel             |     a                             |
|     me                            | ttribute)](api/languages/python_a |
| thod)](api/languages/python_api.h | pi.html#cudaq.PyKernel.arguments) |
| tml#cudaq.NoiseModel.add_channel) | -   [as_pauli()                   |
| -   [all_gather() (in module      |     (cudaq.o                      |
|                                   | perators.spin.SpinOperatorElement |
|    cudaq.mpi)](api/languages/pyth |     method)](api/languages/       |
| on_api.html#cudaq.mpi.all_gather) | python_api.html#cudaq.operators.s |
| -   [amplitude() (cudaq.State     | pin.SpinOperatorElement.as_pauli) |
|     method)](api/languages/pytho  | -   [AsyncEvolveResult (class in  |
| n_api.html#cudaq.State.amplitude) |     cudaq)](api/languages/python_ |
| -   [AmplitudeDampingChannel      | api.html#cudaq.AsyncEvolveResult) |
|     (class in                     | -   [AsyncObserveResult (class in |
|     cu                            |                                   |
| daq)](api/languages/python_api.ht |    cudaq)](api/languages/python_a |
| ml#cudaq.AmplitudeDampingChannel) | pi.html#cudaq.AsyncObserveResult) |
| -   [amplitudes() (cudaq.State    | -   [AsyncSampleResult (class in  |
|     method)](api/languages/python |     cudaq)](api/languages/python_ |
| _api.html#cudaq.State.amplitudes) | api.html#cudaq.AsyncSampleResult) |
| -   [annihilate() (in module      | -   [AsyncStateResult (class in   |
|     c                             |     cudaq)](api/languages/python  |
| udaq.boson)](api/languages/python | _api.html#cudaq.AsyncStateResult) |
| _api.html#cudaq.boson.annihilate) |                                   |
|     -   [(in module               |                                   |
|         cudaq                     |                                   |
| .fermion)](api/languages/python_a |                                   |
| pi.html#cudaq.fermion.annihilate) |                                   |
+-----------------------------------+-----------------------------------+

## B {#B}

+-----------------------------------+-----------------------------------+
| -   [BaseIntegrator (class in     | -   [bias_strength                |
|                                   |     (c                            |
| cudaq.dynamics.integrator)](api/l | udaq.ptsbe.ShotAllocationStrategy |
| anguages/python_api.html#cudaq.dy |     property)](api/languages      |
| namics.integrator.BaseIntegrator) | /python_api.html#cudaq.ptsbe.Shot |
| -   [batch_size                   | AllocationStrategy.bias_strength) |
|     (cudaq.optimizers.Adam        | -   [BitFlipChannel (class in     |
|     property                      |     cudaq)](api/languages/pyth    |
| )](api/languages/python_api.html# | on_api.html#cudaq.BitFlipChannel) |
| cudaq.optimizers.Adam.batch_size) | -   [BosonOperator (class in      |
|     -   [(cudaq.optimizers.SGD    |     cudaq.operators.boson)](      |
|         propert                   | api/languages/python_api.html#cud |
| y)](api/languages/python_api.html | aq.operators.boson.BosonOperator) |
| #cudaq.optimizers.SGD.batch_size) | -   [BosonOperatorElement (class  |
| -   [beta1 (cudaq.optimizers.Adam |     in                            |
|     pro                           |                                   |
| perty)](api/languages/python_api. |   cudaq.operators.boson)](api/lan |
| html#cudaq.optimizers.Adam.beta1) | guages/python_api.html#cudaq.oper |
| -   [beta2 (cudaq.optimizers.Adam | ators.boson.BosonOperatorElement) |
|     pro                           | -   [BosonOperatorTerm (class in  |
| perty)](api/languages/python_api. |     cudaq.operators.boson)](api/  |
| html#cudaq.optimizers.Adam.beta2) | languages/python_api.html#cudaq.o |
| -   [beta_reduction()             | perators.boson.BosonOperatorTerm) |
|     (cudaq.PyKernelDecorator      | -   [broadcast() (in module       |
|     method)](api                  |     cudaq.mpi)](api/languages/pyt |
| /languages/python_api.html#cudaq. | hon_api.html#cudaq.mpi.broadcast) |
| PyKernelDecorator.beta_reduction) |                                   |
+-----------------------------------+-----------------------------------+

## C {#C}

+-----------------------------------+-----------------------------------+
| -   [canonicalize()               | -   [cudaq::product_op::degrees   |
|     (cu                           |     (C++                          |
| daq.operators.boson.BosonOperator |     function)                     |
|     method)](api/languages        | ](api/languages/cpp_api.html#_CPP |
| /python_api.html#cudaq.operators. | v4NK5cudaq10product_op7degreesEv) |
| boson.BosonOperator.canonicalize) | -   [cudaq::product_op::dump (C++ |
|     -   [(cudaq.                  |     functi                        |
| operators.boson.BosonOperatorTerm | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4NK5cudaq10product_op4dumpEv) |
|        method)](api/languages/pyt | -   [cudaq::product_op::end (C++  |
| hon_api.html#cudaq.operators.boso |     funct                         |
| n.BosonOperatorTerm.canonicalize) | ion)](api/languages/cpp_api.html# |
|     -   [(cudaq.                  | _CPPv4NK5cudaq10product_op3endEv) |
| operators.fermion.FermionOperator | -   [c                            |
|                                   | udaq::product_op::get_coefficient |
|        method)](api/languages/pyt |     (C++                          |
| hon_api.html#cudaq.operators.ferm |     function)](api/lan            |
| ion.FermionOperator.canonicalize) | guages/cpp_api.html#_CPPv4NK5cuda |
|     -   [(cudaq.oper              | q10product_op15get_coefficientEv) |
| ators.fermion.FermionOperatorTerm | -                                 |
|                                   |   [cudaq::product_op::get_term_id |
|    method)](api/languages/python_ |     (C++                          |
| api.html#cudaq.operators.fermion. |     function)](api                |
| FermionOperatorTerm.canonicalize) | /languages/cpp_api.html#_CPPv4NK5 |
|     -                             | cudaq10product_op11get_term_idEv) |
|  [(cudaq.operators.MatrixOperator | -                                 |
|         method)](api/lang         |   [cudaq::product_op::is_identity |
| uages/python_api.html#cudaq.opera |     (C++                          |
| tors.MatrixOperator.canonicalize) |     function)](api                |
|     -   [(c                       | /languages/cpp_api.html#_CPPv4NK5 |
| udaq.operators.MatrixOperatorTerm | cudaq10product_op11is_identityEv) |
|         method)](api/language     | -   [cudaq::product_op::num_ops   |
| s/python_api.html#cudaq.operators |     (C++                          |
| .MatrixOperatorTerm.canonicalize) |     function)                     |
|     -   [(                        | ](api/languages/cpp_api.html#_CPP |
| cudaq.operators.spin.SpinOperator | v4NK5cudaq10product_op7num_opsEv) |
|         method)](api/languag      | -                                 |
| es/python_api.html#cudaq.operator |    [cudaq::product_op::operator\* |
| s.spin.SpinOperator.canonicalize) |     (C++                          |
|     -   [(cuda                    |     function)](api/languages/     |
| q.operators.spin.SpinOperatorTerm | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|         method)](api/languages/p  | oduct_opmlE10product_opI1TERK15sc |
| ython_api.html#cudaq.operators.sp | alar_operatorRK10product_opI1TE), |
| in.SpinOperatorTerm.canonicalize) |     [\[1\]](api/languages/        |
| -   [canonicalized() (in module   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     cuda                          | oduct_opmlE10product_opI1TERK15sc |
| q.boson)](api/languages/python_ap | alar_operatorRR10product_opI1TE), |
| i.html#cudaq.boson.canonicalized) |     [\[2\]](api/languages/        |
|     -   [(in module               | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|         cudaq.fe                  | oduct_opmlE10product_opI1TERR15sc |
| rmion)](api/languages/python_api. | alar_operatorRK10product_opI1TE), |
| html#cudaq.fermion.canonicalized) |     [\[3\]](api/languages/        |
|     -   [(in module               | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|                                   | oduct_opmlE10product_opI1TERR15sc |
|        cudaq.operators.custom)](a | alar_operatorRR10product_opI1TE), |
| pi/languages/python_api.html#cuda |     [\[4\]](api/                  |
| q.operators.custom.canonicalized) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(in module               | 5cudaq10product_opmlE6sum_opI1TER |
|         cu                        | K15scalar_operatorRK6sum_opI1TE), |
| daq.spin)](api/languages/python_a |     [\[5\]](api/                  |
| pi.html#cudaq.spin.canonicalized) | languages/cpp_api.html#_CPPv4I0EN |
| -   [captured_variables()         | 5cudaq10product_opmlE6sum_opI1TER |
|     (cudaq.PyKernelDecorator      | K15scalar_operatorRR6sum_opI1TE), |
|     method)](api/lan              |     [\[6\]](api/                  |
| guages/python_api.html#cudaq.PyKe | languages/cpp_api.html#_CPPv4I0EN |
| rnelDecorator.captured_variables) | 5cudaq10product_opmlE6sum_opI1TER |
| -   [CentralDifference (class in  | R15scalar_operatorRK6sum_opI1TE), |
|     cudaq.gradients)              |     [\[7\]](api/                  |
| ](api/languages/python_api.html#c | languages/cpp_api.html#_CPPv4I0EN |
| udaq.gradients.CentralDifference) | 5cudaq10product_opmlE6sum_opI1TER |
| -   [clear() (cudaq.Resources     | R15scalar_operatorRR6sum_opI1TE), |
|     method)](api/languages/pytho  |     [\[8\]](api/languages         |
| n_api.html#cudaq.Resources.clear) | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     -   [(cudaq.SampleResult      | duct_opmlERK6sum_opI9HandlerTyE), |
|                                   |     [\[9\]](api/languages/cpp_a   |
|   method)](api/languages/python_a | pi.html#_CPPv4NKR5cudaq10product_ |
| pi.html#cudaq.SampleResult.clear) | opmlERK10product_opI9HandlerTyE), |
| -   [COBYLA (class in             |     [\[10\]](api/language         |
|     cudaq.o                       | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ptimizers)](api/languages/python_ | roduct_opmlERK15scalar_operator), |
| api.html#cudaq.optimizers.COBYLA) |     [\[11\]](api/languages/cpp_a  |
| -   [coefficient                  | pi.html#_CPPv4NKR5cudaq10product_ |
|     (cudaq.                       | opmlERR10product_opI9HandlerTyE), |
| operators.boson.BosonOperatorTerm |     [\[12\]](api/language         |
|     property)](api/languages/py   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| thon_api.html#cudaq.operators.bos | roduct_opmlERR15scalar_operator), |
| on.BosonOperatorTerm.coefficient) |     [\[13\]](api/languages/cpp_   |
|     -   [(cudaq.oper              | api.html#_CPPv4NO5cudaq10product_ |
| ators.fermion.FermionOperatorTerm | opmlERK10product_opI9HandlerTyE), |
|                                   |     [\[14\]](api/languag          |
|   property)](api/languages/python | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _api.html#cudaq.operators.fermion | roduct_opmlERK15scalar_operator), |
| .FermionOperatorTerm.coefficient) |     [\[15\]](api/languages/cpp_   |
|     -   [(c                       | api.html#_CPPv4NO5cudaq10product_ |
| udaq.operators.MatrixOperatorTerm | opmlERR10product_opI9HandlerTyE), |
|         property)](api/languag    |     [\[16\]](api/langua           |
| es/python_api.html#cudaq.operator | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| s.MatrixOperatorTerm.coefficient) | product_opmlERR15scalar_operator) |
|     -   [(cuda                    | -                                 |
| q.operators.spin.SpinOperatorTerm |   [cudaq::product_op::operator\*= |
|         property)](api/languages/ |     (C++                          |
| python_api.html#cudaq.operators.s |     function)](api/languages/cpp  |
| pin.SpinOperatorTerm.coefficient) | _api.html#_CPPv4N5cudaq10product_ |
| -   [col_count                    | opmLERK10product_opI9HandlerTyE), |
|     (cudaq.KrausOperator          |     [\[1\]](api/langua            |
|     prope                         | ges/cpp_api.html#_CPPv4N5cudaq10p |
| rty)](api/languages/python_api.ht | roduct_opmLERK15scalar_operator), |
| ml#cudaq.KrausOperator.col_count) |     [\[2\]](api/languages/cp      |
| -   [compile()                    | p_api.html#_CPPv4N5cudaq10product |
|     (cudaq.PyKernelDecorator      | _opmLERR10product_opI9HandlerTyE) |
|     metho                         | -   [cudaq::product_op::operator+ |
| d)](api/languages/python_api.html |     (C++                          |
| #cudaq.PyKernelDecorator.compile) |     function)](api/langu          |
| -   [ComplexMatrix (class in      | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cudaq)](api/languages/pyt     | q10product_opplE6sum_opI1TERK15sc |
| hon_api.html#cudaq.ComplexMatrix) | alar_operatorRK10product_opI1TE), |
| -   [compute()                    |     [\[1\]](api/                  |
|     (                             | languages/cpp_api.html#_CPPv4I0EN |
| cudaq.gradients.CentralDifference | 5cudaq10product_opplE6sum_opI1TER |
|     method)](api/la               | K15scalar_operatorRK6sum_opI1TE), |
| nguages/python_api.html#cudaq.gra |     [\[2\]](api/langu             |
| dients.CentralDifference.compute) | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     -   [(                        | q10product_opplE6sum_opI1TERK15sc |
| cudaq.gradients.ForwardDifference | alar_operatorRR10product_opI1TE), |
|         method)](api/la           |     [\[3\]](api/                  |
| nguages/python_api.html#cudaq.gra | languages/cpp_api.html#_CPPv4I0EN |
| dients.ForwardDifference.compute) | 5cudaq10product_opplE6sum_opI1TER |
|     -                             | K15scalar_operatorRR6sum_opI1TE), |
|  [(cudaq.gradients.ParameterShift |     [\[4\]](api/langu             |
|         method)](api              | ages/cpp_api.html#_CPPv4I0EN5cuda |
| /languages/python_api.html#cudaq. | q10product_opplE6sum_opI1TERR15sc |
| gradients.ParameterShift.compute) | alar_operatorRK10product_opI1TE), |
| -   [const()                      |     [\[5\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|   (cudaq.operators.ScalarOperator | 5cudaq10product_opplE6sum_opI1TER |
|     class                         | R15scalar_operatorRK6sum_opI1TE), |
|     method)](a                    |     [\[6\]](api/langu             |
| pi/languages/python_api.html#cuda | ages/cpp_api.html#_CPPv4I0EN5cuda |
| q.operators.ScalarOperator.const) | q10product_opplE6sum_opI1TERR15sc |
| -   [copy()                       | alar_operatorRR10product_opI1TE), |
|     (cu                           |     [\[7\]](api/                  |
| daq.operators.boson.BosonOperator | languages/cpp_api.html#_CPPv4I0EN |
|     method)](api/l                | 5cudaq10product_opplE6sum_opI1TER |
| anguages/python_api.html#cudaq.op | R15scalar_operatorRR6sum_opI1TE), |
| erators.boson.BosonOperator.copy) |     [\[8\]](api/languages/cpp_a   |
|     -   [(cudaq.                  | pi.html#_CPPv4NKR5cudaq10product_ |
| operators.boson.BosonOperatorTerm | opplERK10product_opI9HandlerTyE), |
|         method)](api/langu        |     [\[9\]](api/language          |
| ages/python_api.html#cudaq.operat | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ors.boson.BosonOperatorTerm.copy) | roduct_opplERK15scalar_operator), |
|     -   [(cudaq.                  |     [\[10\]](api/languages/       |
| operators.fermion.FermionOperator | cpp_api.html#_CPPv4NKR5cudaq10pro |
|         method)](api/langu        | duct_opplERK6sum_opI9HandlerTyE), |
| ages/python_api.html#cudaq.operat |     [\[11\]](api/languages/cpp_a  |
| ors.fermion.FermionOperator.copy) | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [(cudaq.oper              | opplERR10product_opI9HandlerTyE), |
| ators.fermion.FermionOperatorTerm |     [\[12\]](api/language         |
|         method)](api/languages    | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| /python_api.html#cudaq.operators. | roduct_opplERR15scalar_operator), |
| fermion.FermionOperatorTerm.copy) |     [\[13\]](api/languages/       |
|     -                             | cpp_api.html#_CPPv4NKR5cudaq10pro |
|  [(cudaq.operators.MatrixOperator | duct_opplERR6sum_opI9HandlerTyE), |
|         method)](                 |     [\[                           |
| api/languages/python_api.html#cud | 14\]](api/languages/cpp_api.html# |
| aq.operators.MatrixOperator.copy) | _CPPv4NKR5cudaq10product_opplEv), |
|     -   [(c                       |     [\[15\]](api/languages/cpp_   |
| udaq.operators.MatrixOperatorTerm | api.html#_CPPv4NO5cudaq10product_ |
|         method)](api/             | opplERK10product_opI9HandlerTyE), |
| languages/python_api.html#cudaq.o |     [\[16\]](api/languag          |
| perators.MatrixOperatorTerm.copy) | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     -   [(                        | roduct_opplERK15scalar_operator), |
| cudaq.operators.spin.SpinOperator |     [\[17\]](api/languages        |
|         method)](api              | /cpp_api.html#_CPPv4NO5cudaq10pro |
| /languages/python_api.html#cudaq. | duct_opplERK6sum_opI9HandlerTyE), |
| operators.spin.SpinOperator.copy) |     [\[18\]](api/languages/cpp_   |
|     -   [(cuda                    | api.html#_CPPv4NO5cudaq10product_ |
| q.operators.spin.SpinOperatorTerm | opplERR10product_opI9HandlerTyE), |
|         method)](api/lan          |     [\[19\]](api/languag          |
| guages/python_api.html#cudaq.oper | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ators.spin.SpinOperatorTerm.copy) | roduct_opplERR15scalar_operator), |
| -   [count() (cudaq.Resources     |     [\[20\]](api/languages        |
|     method)](api/languages/pytho  | /cpp_api.html#_CPPv4NO5cudaq10pro |
| n_api.html#cudaq.Resources.count) | duct_opplERR6sum_opI9HandlerTyE), |
|     -   [(cudaq.SampleResult      |     [                             |
|                                   | \[21\]](api/languages/cpp_api.htm |
|   method)](api/languages/python_a | l#_CPPv4NO5cudaq10product_opplEv) |
| pi.html#cudaq.SampleResult.count) | -   [cudaq::product_op::operator- |
| -   [count_controls()             |     (C++                          |
|     (cudaq.Resources              |     function)](api/langu          |
|     meth                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
| od)](api/languages/python_api.htm | q10product_opmiE6sum_opI1TERK15sc |
| l#cudaq.Resources.count_controls) | alar_operatorRK10product_opI1TE), |
| -   [count_instructions()         |     [\[1\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|   (cudaq.ptsbe.PTSBEExecutionData | 5cudaq10product_opmiE6sum_opI1TER |
|     method)](api/languages/       | K15scalar_operatorRK6sum_opI1TE), |
| python_api.html#cudaq.ptsbe.PTSBE |     [\[2\]](api/langu             |
| ExecutionData.count_instructions) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [counts()                     | q10product_opmiE6sum_opI1TERK15sc |
|     (cudaq.ObserveResult          | alar_operatorRR10product_opI1TE), |
|                                   |     [\[3\]](api/                  |
| method)](api/languages/python_api | languages/cpp_api.html#_CPPv4I0EN |
| .html#cudaq.ObserveResult.counts) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [create() (in module          | K15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[4\]](api/langu             |
|    cudaq.boson)](api/languages/py | ages/cpp_api.html#_CPPv4I0EN5cuda |
| thon_api.html#cudaq.boson.create) | q10product_opmiE6sum_opI1TERR15sc |
|     -   [(in module               | alar_operatorRK10product_opI1TE), |
|         c                         |     [\[5\]](api/                  |
| udaq.fermion)](api/languages/pyth | languages/cpp_api.html#_CPPv4I0EN |
| on_api.html#cudaq.fermion.create) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [csr_spmatrix (C++            | R15scalar_operatorRK6sum_opI1TE), |
|     type)](api/languages/c        |     [\[6\]](api/langu             |
| pp_api.html#_CPPv412csr_spmatrix) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   cudaq                         | q10product_opmiE6sum_opI1TERR15sc |
|     -   [module](api/langua       | alar_operatorRR10product_opI1TE), |
| ges/python_api.html#module-cudaq) |     [\[7\]](api/                  |
| -   [cudaq (C++                   | languages/cpp_api.html#_CPPv4I0EN |
|     type)](api/lan                | 5cudaq10product_opmiE6sum_opI1TER |
| guages/cpp_api.html#_CPPv45cudaq) | R15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq.apply_noise() (in      |     [\[8\]](api/languages/cpp_a   |
|     module                        | pi.html#_CPPv4NKR5cudaq10product_ |
|     cudaq)](api/languages/python_ | opmiERK10product_opI9HandlerTyE), |
| api.html#cudaq.cudaq.apply_noise) |     [\[9\]](api/language          |
| -   cudaq.boson                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     -   [module](api/languages/py | roduct_opmiERK15scalar_operator), |
| thon_api.html#module-cudaq.boson) |     [\[10\]](api/languages/       |
| -   cudaq.fermion                 | cpp_api.html#_CPPv4NKR5cudaq10pro |
|                                   | duct_opmiERK6sum_opI9HandlerTyE), |
|   -   [module](api/languages/pyth |     [\[11\]](api/languages/cpp_a  |
| on_api.html#module-cudaq.fermion) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.operators.custom        | opmiERR10product_opI9HandlerTyE), |
|     -   [mo                       |     [\[12\]](api/language         |
| dule](api/languages/python_api.ht | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ml#module-cudaq.operators.custom) | roduct_opmiERR15scalar_operator), |
| -   cudaq.spin                    |     [\[13\]](api/languages/       |
|     -   [module](api/languages/p  | cpp_api.html#_CPPv4NKR5cudaq10pro |
| ython_api.html#module-cudaq.spin) | duct_opmiERR6sum_opI9HandlerTyE), |
| -   [cudaq::amplitude_damping     |     [\[                           |
|     (C++                          | 14\]](api/languages/cpp_api.html# |
|     cla                           | _CPPv4NKR5cudaq10product_opmiEv), |
| ss)](api/languages/cpp_api.html#_ |     [\[15\]](api/languages/cpp_   |
| CPPv4N5cudaq17amplitude_dampingE) | api.html#_CPPv4NO5cudaq10product_ |
| -                                 | opmiERK10product_opI9HandlerTyE), |
| [cudaq::amplitude_damping_channel |     [\[16\]](api/languag          |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     class)](api                   | roduct_opmiERK15scalar_operator), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[17\]](api/languages        |
| udaq25amplitude_damping_channelE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::amplitud              | duct_opmiERK6sum_opI9HandlerTyE), |
| e_damping_channel::num_parameters |     [\[18\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     member)](api/languages/cpp_a  | opmiERR10product_opI9HandlerTyE), |
| pi.html#_CPPv4N5cudaq25amplitude_ |     [\[19\]](api/languag          |
| damping_channel14num_parametersE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::ampli                 | roduct_opmiERR15scalar_operator), |
| tude_damping_channel::num_targets |     [\[20\]](api/languages        |
|     (C++                          | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     member)](api/languages/cp     | duct_opmiERR6sum_opI9HandlerTyE), |
| p_api.html#_CPPv4N5cudaq25amplitu |     [                             |
| de_damping_channel11num_targetsE) | \[21\]](api/languages/cpp_api.htm |
| -   [cudaq::AnalogRemoteRESTQPU   | l#_CPPv4NO5cudaq10product_opmiEv) |
|     (C++                          | -   [cudaq::product_op::operator/ |
|     class                         |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/language       |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::apply_noise (C++      | roduct_opdvERK15scalar_operator), |
|     function)](api/               |     [\[1\]](api/language          |
| languages/cpp_api.html#_CPPv4I0Dp | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| EN5cudaq11apply_noiseEvDpRR4Args) | roduct_opdvERR15scalar_operator), |
| -   [cudaq::async_result (C++     |     [\[2\]](api/languag           |
|     c                             | es/cpp_api.html#_CPPv4NO5cudaq10p |
| lass)](api/languages/cpp_api.html | roduct_opdvERK15scalar_operator), |
| #_CPPv4I0EN5cudaq12async_resultE) |     [\[3\]](api/langua            |
| -   [cudaq::async_result::get     | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     (C++                          | product_opdvERR15scalar_operator) |
|     functi                        | -                                 |
| on)](api/languages/cpp_api.html#_ |    [cudaq::product_op::operator/= |
| CPPv4N5cudaq12async_result3getEv) |     (C++                          |
| -   [cudaq::async_sample_result   |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     type                          | product_opdVERK15scalar_operator) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::product_op::operator= |
| Pv4N5cudaq19async_sample_resultE) |     (C++                          |
| -   [cudaq::BaseRemoteRESTQPU     |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4I0_NSt |
|     cla                           | 11enable_if_tIXaantNSt7is_sameI1T |
| ss)](api/languages/cpp_api.html#_ | 9HandlerTyE5valueENSt16is_constru |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | ctibleI9HandlerTy1TE5valueEEbEEEN |
| -                                 | 5cudaq10product_opaSER10product_o |
|    [cudaq::BaseRemoteSimulatorQPU | pI9HandlerTyERK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/languages/cpp     |
|     class)](                      | _api.html#_CPPv4N5cudaq10product_ |
| api/languages/cpp_api.html#_CPPv4 | opaSERK10product_opI9HandlerTyE), |
| N5cudaq22BaseRemoteSimulatorQPUE) |     [\[2\]](api/languages/cp      |
| -   [cudaq::bit_flip_channel (C++ | p_api.html#_CPPv4N5cudaq10product |
|     cl                            | _opaSERR10product_opI9HandlerTyE) |
| ass)](api/languages/cpp_api.html# | -                                 |
| _CPPv4N5cudaq16bit_flip_channelE) |    [cudaq::product_op::operator== |
| -   [cudaq:                       |     (C++                          |
| :bit_flip_channel::num_parameters |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq10product |
|     member)](api/langua           | _opeqERK10product_opI9HandlerTyE) |
| ges/cpp_api.html#_CPPv4N5cudaq16b | -                                 |
| it_flip_channel14num_parametersE) |  [cudaq::product_op::operator\[\] |
| -   [cud                          |     (C++                          |
| aq::bit_flip_channel::num_targets |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     member)](api/lan              | 5cudaq10product_opixENSt6size_tE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -                                 |
| 16bit_flip_channel11num_targetsE) |    [cudaq::product_op::product_op |
| -   [cudaq::boson_handler (C++    |     (C++                          |
|                                   |     function)](api/languages/c    |
|  class)](api/languages/cpp_api.ht | pp_api.html#_CPPv4I0_NSt11enable_ |
| ml#_CPPv4N5cudaq13boson_handlerE) | if_tIXaaNSt7is_sameI9HandlerTy14m |
| -   [cudaq::boson_op (C++         | atrix_handlerE5valueEaantNSt7is_s |
|     type)](api/languages/cpp_     | ameI1T9HandlerTyE5valueENSt16is_c |
| api.html#_CPPv4N5cudaq8boson_opE) | onstructibleI9HandlerTy1TE5valueE |
| -   [cudaq::boson_op_term (C++    | EbEEEN5cudaq10product_op10product |
|                                   | _opERK10product_opI1TERKN14matrix |
|   type)](api/languages/cpp_api.ht | _handler20commutation_behaviorE), |
| ml#_CPPv4N5cudaq13boson_op_termE) |                                   |
| -   [cudaq::CodeGenConfig (C++    |  [\[1\]](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| struct)](api/languages/cpp_api.ht | tNSt7is_sameI1T9HandlerTyE5valueE |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | NSt16is_constructibleI9HandlerTy1 |
| -   [cudaq::commutation_relations | TE5valueEEbEEEN5cudaq10product_op |
|     (C++                          | 10product_opERK10product_opI1TE), |
|     struct)]                      |                                   |
| (api/languages/cpp_api.html#_CPPv |   [\[2\]](api/languages/cpp_api.h |
| 4N5cudaq21commutation_relationsE) | tml#_CPPv4N5cudaq10product_op10pr |
| -   [cudaq::complex (C++          | oduct_opENSt6size_tENSt6size_tE), |
|     type)](api/languages/cpp      |     [\[3\]](api/languages/cp      |
| _api.html#_CPPv4N5cudaq7complexE) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq::complex_matrix (C++   | _op10product_opENSt7complexIdEE), |
|                                   |     [\[4\]](api/l                 |
| class)](api/languages/cpp_api.htm | anguages/cpp_api.html#_CPPv4N5cud |
| l#_CPPv4N5cudaq14complex_matrixE) | aq10product_op10product_opERK10pr |
| -                                 | oduct_opI9HandlerTyENSt6size_tE), |
|   [cudaq::complex_matrix::adjoint |     [\[5\]](api/l                 |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     function)](a                  | aq10product_op10product_opERR10pr |
| pi/languages/cpp_api.html#_CPPv4N | oduct_opI9HandlerTyENSt6size_tE), |
| 5cudaq14complex_matrix7adjointEv) |     [\[6\]](api/languages         |
| -   [cudaq::                      | /cpp_api.html#_CPPv4N5cudaq10prod |
| complex_matrix::diagonal_elements | uct_op10product_opERR9HandlerTy), |
|     (C++                          |     [\[7\]](ap                    |
|     function)](api/languages      | i/languages/cpp_api.html#_CPPv4N5 |
| /cpp_api.html#_CPPv4NK5cudaq14com | cudaq10product_op10product_opEd), |
| plex_matrix17diagonal_elementsEi) |     [\[8\]](a                     |
| -   [cudaq::complex_matrix::dump  | pi/languages/cpp_api.html#_CPPv4N |
|     (C++                          | 5cudaq10product_op10product_opEv) |
|     function)](api/language       | -   [cuda                         |
| s/cpp_api.html#_CPPv4NK5cudaq14co | q::product_op::to_diagonal_matrix |
| mplex_matrix4dumpERNSt7ostreamE), |     (C++                          |
|     [\[1\]]                       |     function)](api/               |
| (api/languages/cpp_api.html#_CPPv | languages/cpp_api.html#_CPPv4NK5c |
| 4NK5cudaq14complex_matrix4dumpEv) | udaq10product_op18to_diagonal_mat |
| -   [c                            | rixENSt13unordered_mapINSt6size_t |
| udaq::complex_matrix::eigenvalues | ENSt7int64_tEEERKNSt13unordered_m |
|     (C++                          | apINSt6stringENSt7complexIdEEEEb) |
|     function)](api/lan            | -   [cudaq::product_op::to_matrix |
| guages/cpp_api.html#_CPPv4NK5cuda |     (C++                          |
| q14complex_matrix11eigenvaluesEv) |     funct                         |
| -   [cu                           | ion)](api/languages/cpp_api.html# |
| daq::complex_matrix::eigenvectors | _CPPv4NK5cudaq10product_op9to_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     function)](api/lang           | ENSt7int64_tEEERKNSt13unordered_m |
| uages/cpp_api.html#_CPPv4NK5cudaq | apINSt6stringENSt7complexIdEEEEb) |
| 14complex_matrix12eigenvectorsEv) | -   [cu                           |
| -   [c                            | daq::product_op::to_sparse_matrix |
| udaq::complex_matrix::exponential |     (C++                          |
|     (C++                          |     function)](ap                 |
|     function)](api/la             | i/languages/cpp_api.html#_CPPv4NK |
| nguages/cpp_api.html#_CPPv4N5cuda | 5cudaq10product_op16to_sparse_mat |
| q14complex_matrix11exponentialEv) | rixENSt13unordered_mapINSt6size_t |
| -                                 | ENSt7int64_tEEERKNSt13unordered_m |
|  [cudaq::complex_matrix::identity | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -   [cudaq::product_op::to_string |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq14comp |     function)](                   |
| lex_matrix8identityEKNSt6size_tE) | api/languages/cpp_api.html#_CPPv4 |
| -                                 | NK5cudaq10product_op9to_stringEv) |
| [cudaq::complex_matrix::kronecker | -                                 |
|     (C++                          |  [cudaq::product_op::\~product_op |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4I00EN5cu |     fu                            |
| daq14complex_matrix9kroneckerE14c | nction)](api/languages/cpp_api.ht |
| omplex_matrix8Iterable8Iterable), | ml#_CPPv4N5cudaq10product_opD0Ev) |
|     [\[1\]](api/l                 | -   [cudaq::ptsbe (C++            |
| anguages/cpp_api.html#_CPPv4N5cud |     type)](api/languages/c        |
| aq14complex_matrix9kroneckerERK14 | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
| complex_matrixRK14complex_matrix) | -   [cudaq::p                     |
| -   [cudaq::c                     | tsbe::ConditionalSamplingStrategy |
| omplex_matrix::minimal_eigenvalue |     (C++                          |
|     (C++                          |     class)](api/languag           |
|     function)](api/languages/     | es/cpp_api.html#_CPPv4N5cudaq5pts |
| cpp_api.html#_CPPv4NK5cudaq14comp | be27ConditionalSamplingStrategyE) |
| lex_matrix18minimal_eigenvalueEv) | -   [cudaq::ptsbe::C              |
| -   [                             | onditionalSamplingStrategy::clone |
| cudaq::complex_matrix::operator() |     (C++                          |
|     (C++                          |                                   |
|     function)](api/languages/cpp  |    function)](api/languages/cpp_a |
| _api.html#_CPPv4N5cudaq14complex_ | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
| matrixclENSt6size_tENSt6size_tE), | ditionalSamplingStrategy5cloneEv) |
|     [\[1\]](api/languages/cpp     | -   [cuda                         |
| _api.html#_CPPv4NK5cudaq14complex | q::ptsbe::ConditionalSamplingStra |
| _matrixclENSt6size_tENSt6size_tE) | tegy::ConditionalSamplingStrategy |
| -   [                             |     (C++                          |
| cudaq::complex_matrix::operator\* |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     function)](api/langua         | ptsbe27ConditionalSamplingStrateg |
| ges/cpp_api.html#_CPPv4N5cudaq14c | y27ConditionalSamplingStrategyE19 |
| omplex_matrixmlEN14complex_matrix | TrajectoryPredicateNSt8uint64_tE) |
| 10value_typeERK14complex_matrix), | -                                 |
|     [\[1\]                        |   [cudaq::ptsbe::ConditionalSampl |
| ](api/languages/cpp_api.html#_CPP | ingStrategy::generateTrajectories |
| v4N5cudaq14complex_matrixmlERK14c |     (C++                          |
| omplex_matrixRK14complex_matrix), |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4NK5cudaq5pts |
|  [\[2\]](api/languages/cpp_api.ht | be27ConditionalSamplingStrategy20 |
| ml#_CPPv4N5cudaq14complex_matrixm | generateTrajectoriesENSt4spanIKN6 |
| lERK14complex_matrixRKNSt6vectorI | detail10NoisePointEEENSt6size_tE) |
| N14complex_matrix10value_typeEEE) | -   [cudaq::ptsbe::               |
| -                                 | ConditionalSamplingStrategy::name |
| [cudaq::complex_matrix::operator+ |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     function                      | api.html#_CPPv4NK5cudaq5ptsbe27Co |
| )](api/languages/cpp_api.html#_CP | nditionalSamplingStrategy4nameEv) |
| Pv4N5cudaq14complex_matrixplERK14 | -   [cudaq:                       |
| complex_matrixRK14complex_matrix) | :ptsbe::ConditionalSamplingStrate |
| -                                 | gy::\~ConditionalSamplingStrategy |
| [cudaq::complex_matrix::operator- |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     function                      | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| )](api/languages/cpp_api.html#_CP | 7ConditionalSamplingStrategyD0Ev) |
| Pv4N5cudaq14complex_matrixmiERK14 | -                                 |
| complex_matrixRK14complex_matrix) | [cudaq::ptsbe::detail::NoisePoint |
| -   [cu                           |     (C++                          |
| daq::complex_matrix::operator\[\] |     struct)](a                    |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|                                   | 5cudaq5ptsbe6detail10NoisePointE) |
|  function)](api/languages/cpp_api | -   [cudaq::p                     |
| .html#_CPPv4N5cudaq14complex_matr | tsbe::detail::NoisePoint::channel |
| ixixERKNSt6vectorINSt6size_tEEE), |     (C++                          |
|     [\[1\]](api/languages/cpp_api |     member)](api/langu            |
| .html#_CPPv4NK5cudaq14complex_mat | ages/cpp_api.html#_CPPv4N5cudaq5p |
| rixixERKNSt6vectorINSt6size_tEEE) | tsbe6detail10NoisePoint7channelE) |
| -   [cudaq::complex_matrix::power | -   [cudaq::ptsbe::det            |
|     (C++                          | ail::NoisePoint::circuit_location |
|     function)]                    |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     member)](api/languages/cpp_a  |
| 4N5cudaq14complex_matrix5powerEi) | pi.html#_CPPv4N5cudaq5ptsbe6detai |
| -                                 | l10NoisePoint16circuit_locationE) |
|  [cudaq::complex_matrix::set_zero | -   [cudaq::p                     |
|     (C++                          | tsbe::detail::NoisePoint::op_name |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     member)](api/langu            |
| cudaq14complex_matrix8set_zeroEv) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -                                 | tsbe6detail10NoisePoint7op_nameE) |
| [cudaq::complex_matrix::to_string | -   [cudaq::                      |
|     (C++                          | ptsbe::detail::NoisePoint::qubits |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     member)](api/lang             |
| udaq14complex_matrix9to_stringEv) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -   [                             | ptsbe6detail10NoisePoint6qubitsE) |
| cudaq::complex_matrix::value_type | -   [cudaq::                      |
|     (C++                          | ptsbe::ExhaustiveSamplingStrategy |
|     type)](api/                   |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     class)](api/langua            |
| daq14complex_matrix10value_typeE) | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| -   [cudaq::contrib (C++          | sbe26ExhaustiveSamplingStrategyE) |
|     type)](api/languages/cpp      | -   [cudaq::ptsbe::               |
| _api.html#_CPPv4N5cudaq7contribE) | ExhaustiveSamplingStrategy::clone |
| -   [cudaq::contrib::draw (C++    |     (C++                          |
|     function)                     |     function)](api/languages/cpp_ |
| ](api/languages/cpp_api.html#_CPP | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| v4I0DpEN5cudaq7contrib4drawENSt6s | haustiveSamplingStrategy5cloneEv) |
| tringERR13QuantumKernelDpRR4Args) | -   [cu                           |
| -                                 | daq::ptsbe::ExhaustiveSamplingStr |
| [cudaq::contrib::get_unitary_cmat | ategy::ExhaustiveSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](api/la             |
| p_api.html#_CPPv4I0DpEN5cudaq7con | nguages/cpp_api.html#_CPPv4N5cuda |
| trib16get_unitary_cmatE14complex_ | q5ptsbe26ExhaustiveSamplingStrate |
| matrixRR13QuantumKernelDpRR4Args) | gy26ExhaustiveSamplingStrategyEv) |
| -   [cudaq::CusvState (C++        | -                                 |
|                                   |    [cudaq::ptsbe::ExhaustiveSampl |
|    class)](api/languages/cpp_api. | ingStrategy::generateTrajectories |
| html#_CPPv4I0EN5cudaq9CusvStateE) |     (C++                          |
| -   [cudaq::depolarization1 (C++  |     function)](api/languag        |
|     c                             | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| lass)](api/languages/cpp_api.html | sbe26ExhaustiveSamplingStrategy20 |
| #_CPPv4N5cudaq15depolarization1E) | generateTrajectoriesENSt4spanIKN6 |
| -   [cudaq::depolarization2 (C++  | detail10NoisePointEEENSt6size_tE) |
|     c                             | -   [cudaq::ptsbe:                |
| lass)](api/languages/cpp_api.html | :ExhaustiveSamplingStrategy::name |
| #_CPPv4N5cudaq15depolarization2E) |     (C++                          |
| -   [cudaq:                       |     function)](api/languages/cpp  |
| :depolarization2::depolarization2 | _api.html#_CPPv4NK5cudaq5ptsbe26E |
|     (C++                          | xhaustiveSamplingStrategy4nameEv) |
|     function)](api/languages/cp   | -   [cuda                         |
| p_api.html#_CPPv4N5cudaq15depolar | q::ptsbe::ExhaustiveSamplingStrat |
| ization215depolarization2EK4real) | egy::\~ExhaustiveSamplingStrategy |
| -   [cudaq                        |     (C++                          |
| ::depolarization2::num_parameters |     function)](api/languages      |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     member)](api/langu            | 26ExhaustiveSamplingStrategyD0Ev) |
| ages/cpp_api.html#_CPPv4N5cudaq15 | -   [cuda                         |
| depolarization214num_parametersE) | q::ptsbe::OrderedSamplingStrategy |
| -   [cu                           |     (C++                          |
| daq::depolarization2::num_targets |     class)](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     member)](api/la               | 5ptsbe23OrderedSamplingStrategyE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::ptsb                  |
| q15depolarization211num_targetsE) | e::OrderedSamplingStrategy::clone |
| -                                 |     (C++                          |
|    [cudaq::depolarization_channel |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
|     class)](                      | 3OrderedSamplingStrategy5cloneEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::ptsbe::OrderedSampl   |
| N5cudaq22depolarization_channelE) | ingStrategy::generateTrajectories |
| -   [cudaq::depol                 |     (C++                          |
| arization_channel::num_parameters |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4NK5cudaq |
|     member)](api/languages/cp     | 5ptsbe23OrderedSamplingStrategy20 |
| p_api.html#_CPPv4N5cudaq22depolar | generateTrajectoriesENSt4spanIKN6 |
| ization_channel14num_parametersE) | detail10NoisePointEEENSt6size_tE) |
| -   [cudaq::de                    | -   [cudaq::pts                   |
| polarization_channel::num_targets | be::OrderedSamplingStrategy::name |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     function)](api/languages/     |
| /cpp_api.html#_CPPv4N5cudaq22depo | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| larization_channel11num_targetsE) | 23OrderedSamplingStrategy4nameEv) |
| -   [cudaq::details (C++          | -                                 |
|     type)](api/languages/cpp      |    [cudaq::ptsbe::OrderedSampling |
| _api.html#_CPPv4N5cudaq7detailsE) | Strategy::OrderedSamplingStrategy |
| -   [cudaq::details::future (C++  |     (C++                          |
|                                   |     function)](                   |
|  class)](api/languages/cpp_api.ht | api/languages/cpp_api.html#_CPPv4 |
| ml#_CPPv4N5cudaq7details6futureE) | N5cudaq5ptsbe23OrderedSamplingStr |
| -                                 | ategy23OrderedSamplingStrategyEv) |
|   [cudaq::details::future::future | -                                 |
|     (C++                          |  [cudaq::ptsbe::OrderedSamplingSt |
|     functio                       | rategy::\~OrderedSamplingStrategy |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq7details6future6future |     function)](api/langua         |
| ERNSt6vectorI3JobEERNSt6stringERN | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| St3mapINSt6stringENSt6stringEEE), | sbe23OrderedSamplingStrategyD0Ev) |
|     [\[1\]](api/lang              | -   [cudaq::pts                   |
| uages/cpp_api.html#_CPPv4N5cudaq7 | be::ProbabilisticSamplingStrategy |
| details6future6futureERR6future), |     (C++                          |
|     [\[2\]]                       |     class)](api/languages         |
| (api/languages/cpp_api.html#_CPPv | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| 4N5cudaq7details6future6futureEv) | 29ProbabilisticSamplingStrategyE) |
| -   [cu                           | -   [cudaq::ptsbe::Pro            |
| daq::details::kernel_builder_base | babilisticSamplingStrategy::clone |
|     (C++                          |     (C++                          |
|     class)](api/l                 |                                   |
| anguages/cpp_api.html#_CPPv4N5cud |  function)](api/languages/cpp_api |
| aq7details19kernel_builder_baseE) | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| -   [cudaq::details::             | bilisticSamplingStrategy5cloneEv) |
| kernel_builder_base::operator\<\< | -                                 |
|     (C++                          | [cudaq::ptsbe::ProbabilisticSampl |
|     function)](api/langua         | ingStrategy::generateTrajectories |
| ges/cpp_api.html#_CPPv4N5cudaq7de |     (C++                          |
| tails19kernel_builder_baselsERNSt |     function)](api/languages/     |
| 7ostreamERK19kernel_builder_base) | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| -   [                             | 29ProbabilisticSamplingStrategy20 |
| cudaq::details::KernelBuilderType | generateTrajectoriesENSt4spanIKN6 |
|     (C++                          | detail10NoisePointEEENSt6size_tE) |
|     class)](api                   | -   [cudaq::ptsbe::Pr             |
| /languages/cpp_api.html#_CPPv4N5c | obabilisticSamplingStrategy::name |
| udaq7details17KernelBuilderTypeE) |     (C++                          |
| -   [cudaq::d                     |                                   |
| etails::KernelBuilderType::create |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
|     function)                     | abilisticSamplingStrategy4nameEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::p                     |
| v4N5cudaq7details17KernelBuilderT | tsbe::ProbabilisticSamplingStrate |
| ype6createEPN4mlir11MLIRContextE) | gy::ProbabilisticSamplingStrategy |
| -   [cudaq::details::Ker          |     (C++                          |
| nelBuilderType::KernelBuilderType |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/lang           | 4N5cudaq5ptsbe29ProbabilisticSamp |
| uages/cpp_api.html#_CPPv4N5cudaq7 | lingStrategy29ProbabilisticSampli |
| details17KernelBuilderType17Kerne | ngStrategyENSt8optionalINSt8uint6 |
| lBuilderTypeERRNSt8functionIFN4ml | 4_tEEENSt8optionalINSt6size_tEEE) |
| ir4TypeEPN4mlir11MLIRContextEEEE) | -   [cudaq::pts                   |
| -   [cudaq::diag_matrix_callback  | be::ProbabilisticSamplingStrategy |
|     (C++                          | ::\~ProbabilisticSamplingStrategy |
|     class)                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languages/cp   |
| v4N5cudaq20diag_matrix_callbackE) | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| -   [cudaq::dyn (C++              | robabilisticSamplingStrategyD0Ev) |
|     member)](api/languages        | -                                 |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | [cudaq::ptsbe::PTSBEExecutionData |
| -   [cudaq::ExecutionContext (C++ |     (C++                          |
|     cl                            |     struct)](ap                   |
| ass)](api/languages/cpp_api.html# | i/languages/cpp_api.html#_CPPv4N5 |
| _CPPv4N5cudaq16ExecutionContextE) | cudaq5ptsbe18PTSBEExecutionDataE) |
| -   [cudaq                        | -   [cudaq::ptsbe::PTSBE          |
| ::ExecutionContext::amplitudeMaps | ExecutionData::count_instructions |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api/l              |
| ages/cpp_api.html#_CPPv4N5cudaq16 | anguages/cpp_api.html#_CPPv4NK5cu |
| ExecutionContext13amplitudeMapsE) | daq5ptsbe18PTSBEExecutionData18co |
| -   [c                            | unt_instructionsE20TraceInstructi |
| udaq::ExecutionContext::asyncExec | onTypeNSt8optionalINSt6stringEEE) |
|     (C++                          | -   [cudaq::ptsbe::P              |
|     member)](api/                 | TSBEExecutionData::get_trajectory |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq16ExecutionContext9asyncExecE) |     function                      |
| -   [cud                          | )](api/languages/cpp_api.html#_CP |
| aq::ExecutionContext::asyncResult | Pv4NK5cudaq5ptsbe18PTSBEExecution |
|     (C++                          | Data14get_trajectoryENSt6size_tE) |
|     member)](api/lan              | -   [cudaq::ptsbe:                |
| guages/cpp_api.html#_CPPv4N5cudaq | :PTSBEExecutionData::instructions |
| 16ExecutionContext11asyncResultE) |     (C++                          |
| -   [cudaq:                       |     member)](api/languages/cp     |
| :ExecutionContext::batchIteration | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     (C++                          | TSBEExecutionData12instructionsE) |
|     member)](api/langua           | -   [cudaq::ptsbe:                |
| ges/cpp_api.html#_CPPv4N5cudaq16E | :PTSBEExecutionData::trajectories |
| xecutionContext14batchIterationE) |     (C++                          |
| -   [cudaq::E                     |     member)](api/languages/cp     |
| xecutionContext::canHandleObserve | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     (C++                          | TSBEExecutionData12trajectoriesE) |
|     member)](api/language         | -   [cudaq::ptsbe::PTSBEOptions   |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     (C++                          |
| cutionContext16canHandleObserveE) |     struc                         |
| -   [cudaq::E                     | t)](api/languages/cpp_api.html#_C |
| xecutionContext::ExecutionContext | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
|     (C++                          | -   [cudaq::ptsbe::PTSB           |
|     func                          | EOptions::include_sequential_data |
| tion)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq16ExecutionContext1 |                                   |
| 6ExecutionContextERKNSt6stringE), |    member)](api/languages/cpp_api |
|     [\[1\]](api/languages/        | .html#_CPPv4N5cudaq5ptsbe12PTSBEO |
| cpp_api.html#_CPPv4N5cudaq16Execu | ptions23include_sequential_dataE) |
| tionContext16ExecutionContextERKN | -   [cudaq::ptsb                  |
| St6stringENSt6size_tENSt6size_tE) | e::PTSBEOptions::max_trajectories |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::expectationValue |     member)](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
|     member)](api/language         | 2PTSBEOptions16max_trajectoriesE) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::ptsbe::PT             |
| cutionContext16expectationValueE) | SBEOptions::return_execution_data |
| -   [cudaq::Execu                 |     (C++                          |
| tionContext::explicitMeasurements |     member)](api/languages/cpp_a  |
|     (C++                          | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
|     member)](api/languages/cp     | EOptions21return_execution_dataE) |
| p_api.html#_CPPv4N5cudaq16Executi | -   [cudaq::pts                   |
| onContext20explicitMeasurementsE) | be::PTSBEOptions::shot_allocation |
| -   [cuda                         |     (C++                          |
| q::ExecutionContext::futureResult |     member)](api/languages        |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     member)](api/lang             | 12PTSBEOptions15shot_allocationE) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cud                          |
| 6ExecutionContext12futureResultE) | aq::ptsbe::PTSBEOptions::strategy |
| -   [cudaq::ExecutionContext      |     (C++                          |
| ::hasConditionalsOnMeasureResults |     member)](api/l                |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     mem                           | aq5ptsbe12PTSBEOptions8strategyE) |
| ber)](api/languages/cpp_api.html# | -   [cudaq::ptsbe::PTSBETrace     |
| _CPPv4N5cudaq16ExecutionContext31 |     (C++                          |
| hasConditionalsOnMeasureResultsE) |     t                             |
| -   [cudaq::Executi               | ype)](api/languages/cpp_api.html# |
| onContext::invocationResultBuffer | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
|     (C++                          | -   [                             |
|     member)](api/languages/cpp_   | cudaq::ptsbe::PTSSamplingStrategy |
| api.html#_CPPv4N5cudaq16Execution |     (C++                          |
| Context22invocationResultBufferE) |     class)](api                   |
| -   [cu                           | /languages/cpp_api.html#_CPPv4N5c |
| daq::ExecutionContext::kernelName | udaq5ptsbe19PTSSamplingStrategyE) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/la               | ptsbe::PTSSamplingStrategy::clone |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10kernelNameE) |     function)](api/languag        |
| -   [cud                          | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| aq::ExecutionContext::kernelTrace | sbe19PTSSamplingStrategy5cloneEv) |
|     (C++                          | -   [cudaq::ptsbe::PTSSampl       |
|     member)](api/lan              | ingStrategy::generateTrajectories |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11kernelTraceE) |     function)](api/               |
| -   [cudaq:                       | languages/cpp_api.html#_CPPv4NK5c |
| :ExecutionContext::msm_dimensions | udaq5ptsbe19PTSSamplingStrategy20 |
|     (C++                          | generateTrajectoriesENSt4spanIKN6 |
|     member)](api/langua           | detail10NoisePointEEENSt6size_tE) |
| ges/cpp_api.html#_CPPv4N5cudaq16E | -   [cudaq:                       |
| xecutionContext14msm_dimensionsE) | :ptsbe::PTSSamplingStrategy::name |
| -   [cudaq::                      |     (C++                          |
| ExecutionContext::msm_prob_err_id |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq5p |
|     member)](api/languag          | tsbe19PTSSamplingStrategy4nameEv) |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | -   [cudaq::ptsbe::PTSSampli      |
| ecutionContext15msm_prob_err_idE) | ngStrategy::\~PTSSamplingStrategy |
| -   [cudaq::Ex                    |     (C++                          |
| ecutionContext::msm_probabilities |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4N5cuda |
|     member)](api/languages        | q5ptsbe19PTSSamplingStrategyD0Ev) |
| /cpp_api.html#_CPPv4N5cudaq16Exec | -   [cudaq::ptsbe::sample (C++    |
| utionContext17msm_probabilitiesE) |                                   |
| -                                 |  function)](api/languages/cpp_api |
|    [cudaq::ExecutionContext::name | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
|     (C++                          | mpleE13sample_resultRK14sample_op |
|     member)]                      | tionsRR13QuantumKernelDpRR4Args), |
| (api/languages/cpp_api.html#_CPPv |     [\[1\]](api                   |
| 4N5cudaq16ExecutionContext4nameE) | /languages/cpp_api.html#_CPPv4I0D |
| -   [cu                           | pEN5cudaq5ptsbe6sampleE13sample_r |
| daq::ExecutionContext::noiseModel | esultRKN5cudaq11noise_modelENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4Args) |
|     member)](api/la               | -   [cudaq::ptsbe::sample_async   |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10noiseModelE) |     function)](a                  |
| -   [cudaq::Exe                   | pi/languages/cpp_api.html#_CPPv4I |
| cutionContext::numberTrajectories | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
|     (C++                          | 9async_sample_resultRK14sample_op |
|     member)](api/languages/       | tionsRR13QuantumKernelDpRR4Args), |
| cpp_api.html#_CPPv4N5cudaq16Execu |     [\[1\]](api/languages/cp      |
| tionContext18numberTrajectoriesE) | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| -   [c                            | be12sample_asyncE19async_sample_r |
| udaq::ExecutionContext::optResult | esultRKN5cudaq11noise_modelENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4Args) |
|     member)](api/                 | -   [cudaq::ptsbe::sample_options |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq16ExecutionContext9optResultE) |     struct)                       |
| -   [cudaq::Execu                 | ](api/languages/cpp_api.html#_CPP |
| tionContext::overlapComputeStates | v4N5cudaq5ptsbe14sample_optionsE) |
|     (C++                          | -   [cudaq::ptsbe::sample_result  |
|     member)](api/languages/cp     |     (C++                          |
| p_api.html#_CPPv4N5cudaq16Executi |     class                         |
| onContext20overlapComputeStatesE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq                        | Pv4N5cudaq5ptsbe13sample_resultE) |
| ::ExecutionContext::overlapResult | -   [cudaq::pts                   |
|     (C++                          | be::sample_result::execution_data |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     function)](api/languages/c    |
| ExecutionContext13overlapResultE) | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| -                                 | 3sample_result14execution_dataEv) |
|   [cudaq::ExecutionContext::qpuId | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::has_execution_data |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |                                   |
| N5cudaq16ExecutionContext5qpuIdE) |    function)](api/languages/cpp_a |
| -   [cudaq                        | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| ::ExecutionContext::registerNames | ple_result18has_execution_dataEv) |
|     (C++                          | -   [cudaq::pt                    |
|     member)](api/langu            | sbe::sample_result::sample_result |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13registerNamesE) |     function)](api/l              |
| -   [cu                           | anguages/cpp_api.html#_CPPv4N5cud |
| daq::ExecutionContext::reorderIdx | aq5ptsbe13sample_result13sample_r |
|     (C++                          | esultERRN5cudaq13sample_resultE), |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda |  [\[1\]](api/languages/cpp_api.ht |
| q16ExecutionContext10reorderIdxE) | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| -                                 | sult13sample_resultERRN5cudaq13sa |
|  [cudaq::ExecutionContext::result | mple_resultE18PTSBEExecutionData) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     member)](a                    | sample_result::set_execution_data |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq16ExecutionContext6resultE) |     function)](api/               |
| -                                 | languages/cpp_api.html#_CPPv4N5cu |
|   [cudaq::ExecutionContext::shots | daq5ptsbe13sample_result18set_exe |
|     (C++                          | cution_dataE18PTSBEExecutionData) |
|     member)](                     | -   [cud                          |
| api/languages/cpp_api.html#_CPPv4 | aq::ptsbe::ShotAllocationStrategy |
| N5cudaq16ExecutionContext5shotsE) |     (C++                          |
| -   [cudaq::                      |     struct)](using                |
| ExecutionContext::simulationState | /examples/ptsbe.html#_CPPv4N5cuda |
|     (C++                          | q5ptsbe22ShotAllocationStrategyE) |
|     member)](api/languag          | -   [cudaq::ptsbe::ShotAllocatio  |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | nStrategy::ShotAllocationStrategy |
| ecutionContext15simulationStateE) |     (C++                          |
| -                                 |     function)                     |
|    [cudaq::ExecutionContext::spin | ](using/examples/ptsbe.html#_CPPv |
|     (C++                          | 4N5cudaq5ptsbe22ShotAllocationStr |
|     member)]                      | ategy22ShotAllocationStrategyE4Ty |
| (api/languages/cpp_api.html#_CPPv | pedNSt8optionalINSt8uint64_tEEE), |
| 4N5cudaq16ExecutionContext4spinE) |     [\[1\                         |
| -   [cudaq::                      | ]](using/examples/ptsbe.html#_CPP |
| ExecutionContext::totalIterations | v4N5cudaq5ptsbe22ShotAllocationSt |
|     (C++                          | rategy22ShotAllocationStrategyEv) |
|     member)](api/languag          | -   [cudaq::pt                    |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | sbe::ShotAllocationStrategy::Type |
| ecutionContext15totalIterationsE) |     (C++                          |
| -   [cudaq::Executio              |     enum)](using/exam             |
| nContext::warnedNamedMeasurements | ples/ptsbe.html#_CPPv4N5cudaq5pts |
|     (C++                          | be22ShotAllocationStrategy4TypeE) |
|     member)](api/languages/cpp_a  | -   [cudaq::ptsbe::ShotAllocatio  |
| pi.html#_CPPv4N5cudaq16ExecutionC | nStrategy::Type::HIGH_WEIGHT_BIAS |
| ontext23warnedNamedMeasurementsE) |     (C++                          |
| -   [cudaq::ExecutionResult (C++  |     enumerat                      |
|     st                            | or)](using/examples/ptsbe.html#_C |
| ruct)](api/languages/cpp_api.html | PPv4N5cudaq5ptsbe22ShotAllocation |
| #_CPPv4N5cudaq15ExecutionResultE) | Strategy4Type16HIGH_WEIGHT_BIASE) |
| -   [cud                          | -   [cudaq::ptsbe::ShotAllocati   |
| aq::ExecutionResult::appendResult | onStrategy::Type::LOW_WEIGHT_BIAS |
|     (C++                          |     (C++                          |
|     functio                       |     enumera                       |
| n)](api/languages/cpp_api.html#_C | tor)](using/examples/ptsbe.html#_ |
| PPv4N5cudaq15ExecutionResult12app | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| endResultENSt6stringENSt6size_tE) | nStrategy4Type15LOW_WEIGHT_BIASE) |
| -   [cu                           | -   [cudaq::ptsbe::ShotAlloc      |
| daq::ExecutionResult::deserialize | ationStrategy::Type::PROPORTIONAL |
|     (C++                          |     (C++                          |
|     function)                     |     enum                          |
| ](api/languages/cpp_api.html#_CPP | erator)](using/examples/ptsbe.htm |
| v4N5cudaq15ExecutionResult11deser | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| ializeERNSt6vectorINSt6size_tEEE) | tionStrategy4Type12PROPORTIONALE) |
| -   [cudaq:                       | -   [cudaq::ptsbe::Shot           |
| :ExecutionResult::ExecutionResult | AllocationStrategy::Type::UNIFORM |
|     (C++                          |     (C++                          |
|     functio                       |                                   |
| n)](api/languages/cpp_api.html#_C |   enumerator)](using/examples/pts |
| PPv4N5cudaq15ExecutionResult15Exe | be.html#_CPPv4N5cudaq5ptsbe22Shot |
| cutionResultE16CountsDictionary), | AllocationStrategy4Type7UNIFORME) |
|     [\[1\]](api/lan               | -                                 |
| guages/cpp_api.html#_CPPv4N5cudaq |   [cudaq::ptsbe::TraceInstruction |
| 15ExecutionResult15ExecutionResul |     (C++                          |
| tE16CountsDictionaryNSt6stringE), |     struct)](                     |
|     [\[2\                         | api/languages/cpp_api.html#_CPPv4 |
| ]](api/languages/cpp_api.html#_CP | N5cudaq5ptsbe16TraceInstructionE) |
| Pv4N5cudaq15ExecutionResult15Exec | -   [cudaq:                       |
| utionResultE16CountsDictionaryd), | :ptsbe::TraceInstruction::channel |
|                                   |     (C++                          |
|    [\[3\]](api/languages/cpp_api. |     member)](api/lang             |
| html#_CPPv4N5cudaq15ExecutionResu | uages/cpp_api.html#_CPPv4N5cudaq5 |
| lt15ExecutionResultENSt6stringE), | ptsbe16TraceInstruction7channelE) |
|     [\[4\                         | -   [cudaq::                      |
| ]](api/languages/cpp_api.html#_CP | ptsbe::TraceInstruction::controls |
| Pv4N5cudaq15ExecutionResult15Exec |     (C++                          |
| utionResultERK15ExecutionResult), |     member)](api/langu            |
|     [\[5\]](api/language          | ages/cpp_api.html#_CPPv4N5cudaq5p |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | tsbe16TraceInstruction8controlsE) |
| cutionResult15ExecutionResultEd), | -   [cud                          |
|     [\[6\]](api/languag           | aq::ptsbe::TraceInstruction::name |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |     (C++                          |
| ecutionResult15ExecutionResultEv) |     member)](api/l                |
| -   [                             | anguages/cpp_api.html#_CPPv4N5cud |
| cudaq::ExecutionResult::operator= | aq5ptsbe16TraceInstruction4nameE) |
|     (C++                          | -   [cudaq                        |
|     function)](api/languages/     | ::ptsbe::TraceInstruction::params |
| cpp_api.html#_CPPv4N5cudaq15Execu |     (C++                          |
| tionResultaSERK15ExecutionResult) |     member)](api/lan              |
| -   [c                            | guages/cpp_api.html#_CPPv4N5cudaq |
| udaq::ExecutionResult::operator== | 5ptsbe16TraceInstruction6paramsE) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/languages/c    | :ptsbe::TraceInstruction::targets |
| pp_api.html#_CPPv4NK5cudaq15Execu |     (C++                          |
| tionResulteqERK15ExecutionResult) |     member)](api/lang             |
| -   [cud                          | uages/cpp_api.html#_CPPv4N5cudaq5 |
| aq::ExecutionResult::registerName | ptsbe16TraceInstruction7targetsE) |
|     (C++                          | -   [cudaq::ptsbe::T              |
|     member)](api/lan              | raceInstruction::TraceInstruction |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult12registerNameE) |                                   |
| -   [cudaq                        |   function)](api/languages/cpp_ap |
| ::ExecutionResult::sequentialData | i.html#_CPPv4N5cudaq5ptsbe16Trace |
|     (C++                          | Instruction16TraceInstructionE20T |
|     member)](api/langu            | raceInstructionTypeNSt6stringENSt |
| ages/cpp_api.html#_CPPv4N5cudaq15 | 6vectorINSt6size_tEEENSt6vectorIN |
| ExecutionResult14sequentialDataE) | St6size_tEEENSt6vectorIdEENSt8opt |
| -   [                             | ionalIN5cudaq13kraus_channelEEE), |
| cudaq::ExecutionResult::serialize |     [\[1\]](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
|     function)](api/l              | eInstruction16TraceInstructionEv) |
| anguages/cpp_api.html#_CPPv4NK5cu | -   [cud                          |
| daq15ExecutionResult9serializeEv) | aq::ptsbe::TraceInstruction::type |
| -   [cudaq::fermion_handler (C++  |     (C++                          |
|     c                             |     member)](api/l                |
| lass)](api/languages/cpp_api.html | anguages/cpp_api.html#_CPPv4N5cud |
| #_CPPv4N5cudaq15fermion_handlerE) | aq5ptsbe16TraceInstruction4typeE) |
| -   [cudaq::fermion_op (C++       | -   [c                            |
|     type)](api/languages/cpp_api  | udaq::ptsbe::TraceInstructionType |
| .html#_CPPv4N5cudaq10fermion_opE) |     (C++                          |
| -   [cudaq::fermion_op_term (C++  |     enum)](api/                   |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
| type)](api/languages/cpp_api.html | daq5ptsbe20TraceInstructionTypeE) |
| #_CPPv4N5cudaq15fermion_op_termE) | -   [cudaq::                      |
| -   [cudaq::FermioniqBaseQPU (C++ | ptsbe::TraceInstructionType::Gate |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     enumerator)](api/langu        |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [cudaq::get_state (C++        | tsbe20TraceInstructionType4GateE) |
|                                   | -   [cudaq::ptsbe::               |
|    function)](api/languages/cpp_a | TraceInstructionType::Measurement |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     (C++                          |
| ateEDaRR13QuantumKernelDpRR4Args) |                                   |
| -   [cudaq::gradient (C++         |    enumerator)](api/languages/cpp |
|     class)](api/languages/cpp_    | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| api.html#_CPPv4N5cudaq8gradientE) | aceInstructionType11MeasurementE) |
| -   [cudaq::gradient::clone (C++  | -   [cudaq::p                     |
|     fun                           | tsbe::TraceInstructionType::Noise |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     enumerator)](api/langua       |
| -   [cudaq::gradient::compute     | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     (C++                          | sbe20TraceInstructionType5NoiseE) |
|     function)](api/language       | -   [                             |
| s/cpp_api.html#_CPPv4N5cudaq8grad | cudaq::ptsbe::TrajectoryPredicate |
| ient7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     type)](api                    |
|     [\[1\]](ap                    | /languages/cpp_api.html#_CPPv4N5c |
| i/languages/cpp_api.html#_CPPv4N5 | udaq5ptsbe19TrajectoryPredicateE) |
| cudaq8gradient7computeERKNSt6vect | -   [cudaq::QPU (C++              |
| orIdEERNSt6vectorIdEERK7spin_opd) |     class)](api/languages         |
| -   [cudaq::gradient::gradient    | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     (C++                          | -   [cudaq::QPU::beginExecution   |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4I00EN5cu |     function                      |
| daq8gradient8gradientER7KernelT), | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq3QPU14beginExecutionEv) |
|    [\[1\]](api/languages/cpp_api. | -   [cuda                         |
| html#_CPPv4I00EN5cudaq8gradient8g | q::QPU::configureExecutionContext |
| radientER7KernelTRR10ArgsMapper), |     (C++                          |
|     [\[2\                         |     funct                         |
| ]](api/languages/cpp_api.html#_CP | ion)](api/languages/cpp_api.html# |
| Pv4I00EN5cudaq8gradient8gradientE | _CPPv4NK5cudaq3QPU25configureExec |
| RR13QuantumKernelRR10ArgsMapper), | utionContextER16ExecutionContext) |
|     [\[3                          | -   [cudaq::QPU::endExecution     |
| \]](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq8gradient8gradientERRN |     functi                        |
| St8functionIFvNSt6vectorIdEEEEE), | on)](api/languages/cpp_api.html#_ |
|     [\[                           | CPPv4N5cudaq3QPU12endExecutionEv) |
| 4\]](api/languages/cpp_api.html#_ | -   [cudaq::QPU::enqueue (C++     |
| CPPv4N5cudaq8gradient8gradientEv) |     function)](ap                 |
| -   [cudaq::gradient::setArgs     | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq3QPU7enqueueER11QuantumTask) |
|     fu                            | -   [cud                          |
| nction)](api/languages/cpp_api.ht | aq::QPU::finalizeExecutionContext |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     (C++                          |
| tArgsEvR13QuantumKernelDpRR4Args) |     func                          |
| -   [cudaq::gradient::setKernel   | tion)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     function)](api/languages/c    | utionContextER16ExecutionContext) |
| pp_api.html#_CPPv4I0EN5cudaq8grad | -   [cudaq::QPU::getConnectivity  |
| ient9setKernelEvR13QuantumKernel) |     (C++                          |
| -   [cud                          |     function)                     |
| aq::gradients::central_difference | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq3QPU15getConnectivityEv) |
|     class)](api/la                | -                                 |
| nguages/cpp_api.html#_CPPv4N5cuda | [cudaq::QPU::getExecutionThreadId |
| q9gradients18central_differenceE) |     (C++                          |
| -   [cudaq::gra                   |     function)](api/               |
| dients::central_difference::clone | languages/cpp_api.html#_CPPv4NK5c |
|     (C++                          | udaq3QPU20getExecutionThreadIdEv) |
|     function)](api/languages      | -   [cudaq::QPU::getNumQubits     |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18central_difference5cloneEv) |     functi                        |
| -   [cudaq::gradi                 | on)](api/languages/cpp_api.html#_ |
| ents::central_difference::compute | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|     (C++                          | -   [                             |
|     function)](                   | cudaq::QPU::getRemoteCapabilities |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq9gradients18central_differ |     function)](api/l              |
| ence7computeERKNSt6vectorIdEERKNS | anguages/cpp_api.html#_CPPv4NK5cu |
| t8functionIFdNSt6vectorIdEEEEEd), | daq3QPU21getRemoteCapabilitiesEv) |
|                                   | -   [cudaq::QPU::isEmulated (C++  |
|   [\[1\]](api/languages/cpp_api.h |     func                          |
| tml#_CPPv4N5cudaq9gradients18cent | tion)](api/languages/cpp_api.html |
| ral_difference7computeERKNSt6vect | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::QPU::isSimulator (C++ |
| -   [cudaq::gradie                |     funct                         |
| nts::central_difference::gradient | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|     functio                       | -   [cudaq::QPU::launchKernel     |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4I00EN5cudaq9gradients18centra |     function)](api                |
| l_difference8gradientER7KernelT), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[1\]](api/langua            | udaq3QPU12launchKernelERKNSt6stri |
| ges/cpp_api.html#_CPPv4I00EN5cuda | ngE15KernelThunkTypePvNSt8uint64_ |
| q9gradients18central_difference8g | tENSt8uint64_tERKNSt6vectorIPvEE) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::QPU::onRandomSeedSet  |
|     [\[2\]](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4I00EN5cudaq9gradie |     function)](api/lang           |
| nts18central_difference8gradientE | uages/cpp_api.html#_CPPv4N5cudaq3 |
| RR13QuantumKernelRR10ArgsMapper), | QPU15onRandomSeedSetENSt6size_tE) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::QPU::QPU (C++         |
| _api.html#_CPPv4N5cudaq9gradients |     functio                       |
| 18central_difference8gradientERRN | n)](api/languages/cpp_api.html#_C |
| St8functionIFvNSt6vectorIdEEEEE), | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
|     [\[4\]](api/languages/cp      |                                   |
| p_api.html#_CPPv4N5cudaq9gradient |  [\[1\]](api/languages/cpp_api.ht |
| s18central_difference8gradientEv) | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| -   [cud                          |     [\[2\]](api/languages/cpp_    |
| aq::gradients::forward_difference | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
|     (C++                          | -   [cudaq::QPU::setId (C++       |
|     class)](api/la                |     function                      |
| nguages/cpp_api.html#_CPPv4N5cuda | )](api/languages/cpp_api.html#_CP |
| q9gradients18forward_differenceE) | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| -   [cudaq::gra                   | -   [cudaq::QPU::setShots (C++    |
| dients::forward_difference::clone |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     function)](api/languages      | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::                      |
| ents18forward_difference5cloneEv) | QPU::supportsExplicitMeasurements |
| -   [cudaq::gradi                 |     (C++                          |
| ents::forward_difference::compute |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq3QPU |
|     function)](                   | 28supportsExplicitMeasurementsEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::QPU::\~QPU (C++       |
| N5cudaq9gradients18forward_differ |     function)](api/languages/cp   |
| ence7computeERKNSt6vectorIdEERKNS | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::QPUState (C++         |
|                                   |     class)](api/languages/cpp_    |
|   [\[1\]](api/languages/cpp_api.h | api.html#_CPPv4N5cudaq8QPUStateE) |
| tml#_CPPv4N5cudaq9gradients18forw | -   [cudaq::qreg (C++             |
| ard_difference7computeERKNSt6vect |     class)](api/lan               |
| orIdEERNSt6vectorIdEERK7spin_opd) | guages/cpp_api.html#_CPPv4I_NSt6s |
| -   [cudaq::gradie                | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| nts::forward_difference::gradient | -   [cudaq::qreg::back (C++       |
|     (C++                          |     function)                     |
|     functio                       | ](api/languages/cpp_api.html#_CPP |
| n)](api/languages/cpp_api.html#_C | v4N5cudaq4qreg4backENSt6size_tE), |
| PPv4I00EN5cudaq9gradients18forwar |     [\[1\]](api/languages/cpp_ap  |
| d_difference8gradientER7KernelT), | i.html#_CPPv4N5cudaq4qreg4backEv) |
|     [\[1\]](api/langua            | -   [cudaq::qreg::begin (C++      |
| ges/cpp_api.html#_CPPv4I00EN5cuda |                                   |
| q9gradients18forward_difference8g |  function)](api/languages/cpp_api |
| radientER7KernelTRR10ArgsMapper), | .html#_CPPv4N5cudaq4qreg5beginEv) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::qreg::clear (C++      |
| api.html#_CPPv4I00EN5cudaq9gradie |                                   |
| nts18forward_difference8gradientE |  function)](api/languages/cpp_api |
| RR13QuantumKernelRR10ArgsMapper), | .html#_CPPv4N5cudaq4qreg5clearEv) |
|     [\[3\]](api/languages/cpp     | -   [cudaq::qreg::front (C++      |
| _api.html#_CPPv4N5cudaq9gradients |     function)]                    |
| 18forward_difference8gradientERRN | (api/languages/cpp_api.html#_CPPv |
| St8functionIFvNSt6vectorIdEEEEE), | 4N5cudaq4qreg5frontENSt6size_tE), |
|     [\[4\]](api/languages/cp      |     [\[1\]](api/languages/cpp_api |
| p_api.html#_CPPv4N5cudaq9gradient | .html#_CPPv4N5cudaq4qreg5frontEv) |
| s18forward_difference8gradientEv) | -   [cudaq::qreg::operator\[\]    |
| -   [                             |     (C++                          |
| cudaq::gradients::parameter_shift |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     class)](api                   | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::qreg::qreg (C++       |
| udaq9gradients15parameter_shiftE) |     function)                     |
| -   [cudaq::                      | ](api/languages/cpp_api.html#_CPP |
| gradients::parameter_shift::clone | v4N5cudaq4qreg4qregENSt6size_tE), |
|     (C++                          |     [\[1\]](api/languages/cpp_ap  |
|     function)](api/langua         | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | -   [cudaq::qreg::size (C++       |
| adients15parameter_shift5cloneEv) |                                   |
| -   [cudaq::gr                    |  function)](api/languages/cpp_api |
| adients::parameter_shift::compute | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|     (C++                          | -   [cudaq::qreg::slice (C++      |
|     function                      |     function)](api/langu          |
| )](api/languages/cpp_api.html#_CP | ages/cpp_api.html#_CPPv4N5cudaq4q |
| Pv4N5cudaq9gradients15parameter_s | reg5sliceENSt6size_tENSt6size_tE) |
| hift7computeERKNSt6vectorIdEERKNS | -   [cudaq::qreg::value_type (C++ |
| t8functionIFdNSt6vectorIdEEEEEd), |                                   |
|     [\[1\]](api/languages/cpp_ap  | type)](api/languages/cpp_api.html |
| i.html#_CPPv4N5cudaq9gradients15p | #_CPPv4N5cudaq4qreg10value_typeE) |
| arameter_shift7computeERKNSt6vect | -   [cudaq::qspan (C++            |
| orIdEERNSt6vectorIdEERK7spin_opd) |     class)](api/lang              |
| -   [cudaq::gra                   | uages/cpp_api.html#_CPPv4I_NSt6si |
| dients::parameter_shift::gradient | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|     (C++                          | -   [cudaq::QuakeValue (C++       |
|     func                          |     class)](api/languages/cpp_api |
| tion)](api/languages/cpp_api.html | .html#_CPPv4N5cudaq10QuakeValueE) |
| #_CPPv4I00EN5cudaq9gradients15par | -   [cudaq::Q                     |
| ameter_shift8gradientER7KernelT), | uakeValue::canValidateNumElements |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4I00EN5c |     function)](api/languages      |
| udaq9gradients15parameter_shift8g | /cpp_api.html#_CPPv4N5cudaq10Quak |
| radientER7KernelTRR10ArgsMapper), | eValue22canValidateNumElementsEv) |
|     [\[2\]](api/languages/c       | -                                 |
| pp_api.html#_CPPv4I00EN5cudaq9gra |  [cudaq::QuakeValue::constantSize |
| dients15parameter_shift8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api                |
|     [\[3\]](api/languages/        | /languages/cpp_api.html#_CPPv4N5c |
| cpp_api.html#_CPPv4N5cudaq9gradie | udaq10QuakeValue12constantSizeEv) |
| nts15parameter_shift8gradientERRN | -   [cudaq::QuakeValue::dump (C++ |
| St8functionIFvNSt6vectorIdEEEEE), |     function)](api/lan            |
|     [\[4\]](api/languages         | guages/cpp_api.html#_CPPv4N5cudaq |
| /cpp_api.html#_CPPv4N5cudaq9gradi | 10QuakeValue4dumpERNSt7ostreamE), |
| ents15parameter_shift8gradientEv) |     [\                            |
| -   [cudaq::kernel_builder (C++   | [1\]](api/languages/cpp_api.html# |
|     clas                          | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| s)](api/languages/cpp_api.html#_C | -   [cudaq                        |
| PPv4IDpEN5cudaq14kernel_builderE) | ::QuakeValue::getRequiredElements |
| -   [c                            |     (C++                          |
| udaq::kernel_builder::constantVal |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq10Q |
|     function)](api/la             | uakeValue19getRequiredElementsEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QuakeValue::getValue  |
| q14kernel_builder11constantValEd) |     (C++                          |
| -   [cu                           |     function)]                    |
| daq::kernel_builder::getArguments | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NK5cudaq10QuakeValue8getValueEv) |
|     function)](api/lan            | -   [cudaq::QuakeValue::inverse   |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 14kernel_builder12getArgumentsEv) |     function)                     |
| -   [cu                           | ](api/languages/cpp_api.html#_CPP |
| daq::kernel_builder::getNumParams | v4NK5cudaq10QuakeValue7inverseEv) |
|     (C++                          | -   [cudaq::QuakeValue::isStdVec  |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)                     |
| 14kernel_builder12getNumParamsEv) | ](api/languages/cpp_api.html#_CPP |
| -   [c                            | v4N5cudaq10QuakeValue8isStdVecEv) |
| udaq::kernel_builder::isArgStdVec | -                                 |
|     (C++                          |    [cudaq::QuakeValue::operator\* |
|     function)](api/languages/cp   |     (C++                          |
| p_api.html#_CPPv4N5cudaq14kernel_ |     function)](api                |
| builder11isArgStdVecENSt6size_tE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cuda                         | udaq10QuakeValuemlE10QuakeValue), |
| q::kernel_builder::kernel_builder |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     function)](api/languages/cpp_ | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| api.html#_CPPv4N5cudaq14kernel_bu | -   [cudaq::QuakeValue::operator+ |
| ilder14kernel_builderERNSt6vector |     (C++                          |
| IN7details17KernelBuilderTypeEEE) |     function)](api                |
| -   [cudaq::kernel_builder::name  | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValueplE10QuakeValue), |
|     function)                     |     [                             |
| ](api/languages/cpp_api.html#_CPP | \[1\]](api/languages/cpp_api.html |
| v4N5cudaq14kernel_builder4nameEv) | #_CPPv4N5cudaq10QuakeValueplEKd), |
| -                                 |                                   |
|    [cudaq::kernel_builder::qalloc | [\[2\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|     function)](api/language       | -   [cudaq::QuakeValue::operator- |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     (C++                          |
| nel_builder6qallocE10QuakeValue), |     function)](api                |
|     [\[1\]](api/language          | /languages/cpp_api.html#_CPPv4N5c |
| s/cpp_api.html#_CPPv4N5cudaq14ker | udaq10QuakeValuemiE10QuakeValue), |
| nel_builder6qallocEKNSt6size_tE), |     [                             |
|     [\[2                          | \[1\]](api/languages/cpp_api.html |
| \]](api/languages/cpp_api.html#_C | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| PPv4N5cudaq14kernel_builder6qallo |     [                             |
| cERNSt6vectorINSt7complexIdEEEE), | \[2\]](api/languages/cpp_api.html |
|     [\[3\]](                      | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| api/languages/cpp_api.html#_CPPv4 |                                   |
| N5cudaq14kernel_builder6qallocEv) | [\[3\]](api/languages/cpp_api.htm |
| -   [cudaq::kernel_builder::swap  | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
|     (C++                          | -   [cudaq::QuakeValue::operator/ |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |     function)](api                |
| 4kernel_builder4swapEvRK10QuakeVa | /languages/cpp_api.html#_CPPv4N5c |
| lueRK10QuakeValueRK10QuakeValue), | udaq10QuakeValuedvE10QuakeValue), |
|                                   |                                   |
| [\[1\]](api/languages/cpp_api.htm | [\[1\]](api/languages/cpp_api.htm |
| l#_CPPv4I00EN5cudaq14kernel_build | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| er4swapEvRKNSt6vectorI10QuakeValu | -                                 |
| eEERK10QuakeValueRK10QuakeValue), |  [cudaq::QuakeValue::operator\[\] |
|                                   |     (C++                          |
| [\[2\]](api/languages/cpp_api.htm |     function)](api                |
| l#_CPPv4N5cudaq14kernel_builder4s | /languages/cpp_api.html#_CPPv4N5c |
| wapERK10QuakeValueRK10QuakeValue) | udaq10QuakeValueixEKNSt6size_tE), |
| -   [cudaq::KernelExecutionTask   |     [\[1\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     type                          | daq10QuakeValueixERK10QuakeValue) |
| )](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq19KernelExecutionTaskE) |    [cudaq::QuakeValue::QuakeValue |
| -   [cudaq::KernelThunkResultType |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     struct)]                      | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| (api/languages/cpp_api.html#_CPPv | akeValue10QuakeValueERN4mlir20Imp |
| 4N5cudaq21KernelThunkResultTypeE) | licitLocOpBuilderEN4mlir5ValueE), |
| -   [cudaq::KernelThunkType (C++  |     [\[1\]                        |
|                                   | ](api/languages/cpp_api.html#_CPP |
| type)](api/languages/cpp_api.html | v4N5cudaq10QuakeValue10QuakeValue |
| #_CPPv4N5cudaq15KernelThunkTypeE) | ERN4mlir20ImplicitLocOpBuilderEd) |
| -   [cudaq::kraus_channel (C++    | -   [cudaq::QuakeValue::size (C++ |
|                                   |     funct                         |
|  class)](api/languages/cpp_api.ht | ion)](api/languages/cpp_api.html# |
| ml#_CPPv4N5cudaq13kraus_channelE) | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| -   [cudaq::kraus_channel::empty  | -   [cudaq::QuakeValue::slice     |
|     (C++                          |     (C++                          |
|     function)]                    |     function)](api/languages/cpp_ |
| (api/languages/cpp_api.html#_CPPv | api.html#_CPPv4N5cudaq10QuakeValu |
| 4NK5cudaq13kraus_channel5emptyEv) | e5sliceEKNSt6size_tEKNSt6size_tE) |
| -   [cudaq::kraus_c               | -   [cudaq::quantum_platform (C++ |
| hannel::generateUnitaryParameters |     cl                            |
|     (C++                          | ass)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq16quantum_platformE) |
|    function)](api/languages/cpp_a | -   [cudaq:                       |
| pi.html#_CPPv4N5cudaq13kraus_chan | :quantum_platform::beginExecution |
| nel25generateUnitaryParametersEv) |     (C++                          |
| -                                 |     function)](api/languag        |
|    [cudaq::kraus_channel::get_ops | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14beginExecutionEv) |
|     function)](a                  | -   [cudaq::quantum_pl            |
| pi/languages/cpp_api.html#_CPPv4N | atform::configureExecutionContext |
| K5cudaq13kraus_channel7get_opsEv) |     (C++                          |
| -   [cud                          |     function)](api/lang           |
| aq::kraus_channel::identity_flags | uages/cpp_api.html#_CPPv4NK5cudaq |
|     (C++                          | 16quantum_platform25configureExec |
|     member)](api/lan              | utionContextER16ExecutionContext) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cuda                         |
| 13kraus_channel14identity_flagsE) | q::quantum_platform::connectivity |
| -   [cud                          |     (C++                          |
| aq::kraus_channel::is_identity_op |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq16 |
|                                   | quantum_platform12connectivityEv) |
|    function)](api/languages/cpp_a | -   [cuda                         |
| pi.html#_CPPv4NK5cudaq13kraus_cha | q::quantum_platform::endExecution |
| nnel14is_identity_opENSt6size_tE) |     (C++                          |
| -   [cudaq::                      |     function)](api/langu          |
| kraus_channel::is_unitary_mixture | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     (C++                          | quantum_platform12endExecutionEv) |
|     function)](api/languages      | -   [cudaq::q                     |
| /cpp_api.html#_CPPv4NK5cudaq13kra | uantum_platform::enqueueAsyncTask |
| us_channel18is_unitary_mixtureEv) |     (C++                          |
| -   [cu                           |     function)](api/languages/     |
| daq::kraus_channel::kraus_channel | cpp_api.html#_CPPv4N5cudaq16quant |
|     (C++                          | um_platform16enqueueAsyncTaskEKNS |
|     function)](api/lang           | t6size_tER19KernelExecutionTask), |
| uages/cpp_api.html#_CPPv4IDpEN5cu |     [\[1\]](api/languag           |
| daq13kraus_channel13kraus_channel | es/cpp_api.html#_CPPv4N5cudaq16qu |
| EDpRRNSt16initializer_listI1TEE), | antum_platform16enqueueAsyncTaskE |
|                                   | KNSt6size_tERNSt8functionIFvvEEE) |
|  [\[1\]](api/languages/cpp_api.ht | -   [cudaq::quantum_p             |
| ml#_CPPv4N5cudaq13kraus_channel13 | latform::finalizeExecutionContext |
| kraus_channelERK13kraus_channel), |     (C++                          |
|     [\[2\]                        |     function)](api/languages/c    |
| ](api/languages/cpp_api.html#_CPP | pp_api.html#_CPPv4NK5cudaq16quant |
| v4N5cudaq13kraus_channel13kraus_c | um_platform24finalizeExecutionCon |
| hannelERKNSt6vectorI8kraus_opEE), | textERN5cudaq16ExecutionContextE) |
|     [\[3\]                        | -   [cudaq::qua                   |
| ](api/languages/cpp_api.html#_CPP | ntum_platform::get_codegen_config |
| v4N5cudaq13kraus_channel13kraus_c |     (C++                          |
| hannelERRNSt6vectorI8kraus_opEE), |     function)](api/languages/c    |
|     [\[4\]](api/lan               | pp_api.html#_CPPv4N5cudaq16quantu |
| guages/cpp_api.html#_CPPv4N5cudaq | m_platform18get_codegen_configEv) |
| 13kraus_channel13kraus_channelEv) | -   [cuda                         |
| -                                 | q::quantum_platform::get_exec_ctx |
| [cudaq::kraus_channel::noise_type |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     member)](api                  | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| /languages/cpp_api.html#_CPPv4N5c | quantum_platform12get_exec_ctxEv) |
| udaq13kraus_channel10noise_typeE) | -   [c                            |
| -                                 | udaq::quantum_platform::get_noise |
|   [cudaq::kraus_channel::op_names |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     member)](                     | pp_api.html#_CPPv4N5cudaq16quantu |
| api/languages/cpp_api.html#_CPPv4 | m_platform9get_noiseENSt6size_tE) |
| N5cudaq13kraus_channel8op_namesE) | -   [cudaq:                       |
| -                                 | :quantum_platform::get_num_qubits |
|  [cudaq::kraus_channel::operator= |     (C++                          |
|     (C++                          |                                   |
|     function)](api/langua         | function)](api/languages/cpp_api. |
| ges/cpp_api.html#_CPPv4N5cudaq13k | html#_CPPv4NK5cudaq16quantum_plat |
| raus_channelaSERK13kraus_channel) | form14get_num_qubitsENSt6size_tE) |
| -   [c                            | -   [cudaq::quantum_              |
| udaq::kraus_channel::operator\[\] | platform::get_remote_capabilities |
|     (C++                          |     (C++                          |
|     function)](api/l              |     function)                     |
| anguages/cpp_api.html#_CPPv4N5cud | ](api/languages/cpp_api.html#_CPP |
| aq13kraus_channelixEKNSt6size_tE) | v4NK5cudaq16quantum_platform23get |
| -                                 | _remote_capabilitiesENSt6size_tE) |
| [cudaq::kraus_channel::parameters | -   [cudaq::qua                   |
|     (C++                          | ntum_platform::get_runtime_target |
|     member)](api                  |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     function)](api/languages/cp   |
| udaq13kraus_channel10parametersE) | p_api.html#_CPPv4NK5cudaq16quantu |
| -   [cudaq::krau                  | m_platform18get_runtime_targetEv) |
| s_channel::populateDefaultOpNames | -   [cuda                         |
|     (C++                          | q::quantum_platform::getLogStream |
|     function)](api/languages/cp   |     (C++                          |
| p_api.html#_CPPv4N5cudaq13kraus_c |     function)](api/langu          |
| hannel22populateDefaultOpNamesEv) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [cu                           | quantum_platform12getLogStreamEv) |
| daq::kraus_channel::probabilities | -   [cud                          |
|     (C++                          | aq::quantum_platform::is_emulated |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q13kraus_channel13probabilitiesE) |    function)](api/languages/cpp_a |
| -                                 | pi.html#_CPPv4NK5cudaq16quantum_p |
|  [cudaq::kraus_channel::push_back | latform11is_emulatedENSt6size_tE) |
|     (C++                          | -   [c                            |
|     function)](api                | udaq::quantum_platform::is_remote |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel9push_backE8kr |     function)](api/languages/cp   |
| aus_opNSt8optionalINSt6stringEEE) | p_api.html#_CPPv4NK5cudaq16quantu |
| -   [cudaq::kraus_channel::size   | m_platform9is_remoteENSt6size_tE) |
|     (C++                          | -   [cuda                         |
|     function)                     | q::quantum_platform::is_simulator |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4NK5cudaq13kraus_channel4sizeEv) |                                   |
| -   [                             |   function)](api/languages/cpp_ap |
| cudaq::kraus_channel::unitary_ops | i.html#_CPPv4NK5cudaq16quantum_pl |
|     (C++                          | atform12is_simulatorENSt6size_tE) |
|     member)](api/                 | -   [c                            |
| languages/cpp_api.html#_CPPv4N5cu | udaq::quantum_platform::launchVQE |
| daq13kraus_channel11unitary_opsE) |     (C++                          |
| -   [cudaq::kraus_op (C++         |     function)](                   |
|     struct)](api/languages/cpp_   | api/languages/cpp_api.html#_CPPv4 |
| api.html#_CPPv4N5cudaq8kraus_opE) | N5cudaq16quantum_platform9launchV |
| -   [cudaq::kraus_op::adjoint     | QEEKNSt6stringEPKvPN5cudaq8gradie |
|     (C++                          | ntERKN5cudaq7spin_opERN5cudaq9opt |
|     functi                        | imizerEKiKNSt6size_tENSt6size_tE) |
| on)](api/languages/cpp_api.html#_ | -   [cudaq:                       |
| CPPv4NK5cudaq8kraus_op7adjointEv) | :quantum_platform::list_platforms |
| -   [cudaq::kraus_op::data (C++   |     (C++                          |
|                                   |     function)](api/languag        |
|  member)](api/languages/cpp_api.h | es/cpp_api.html#_CPPv4N5cudaq16qu |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | antum_platform14list_platformsEv) |
| -   [cudaq::kraus_op::kraus_op    | -                                 |
|     (C++                          |    [cudaq::quantum_platform::name |
|     func                          |     (C++                          |
| tion)](api/languages/cpp_api.html |     function)](a                  |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | pi/languages/cpp_api.html#_CPPv4N |
| opERRNSt16initializer_listI1TEE), | K5cudaq16quantum_platform4nameEv) |
|                                   | -   [                             |
|  [\[1\]](api/languages/cpp_api.ht | cudaq::quantum_platform::num_qpus |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o |     (C++                          |
| pENSt6vectorIN5cudaq7complexEEE), |     function)](api/l              |
|     [\[2\]](api/l                 | anguages/cpp_api.html#_CPPv4NK5cu |
| anguages/cpp_api.html#_CPPv4N5cud | daq16quantum_platform8num_qpusEv) |
| aq8kraus_op8kraus_opERK8kraus_op) | -   [cudaq::                      |
| -   [cudaq::kraus_op::nCols (C++  | quantum_platform::onRandomSeedSet |
|                                   |     (C++                          |
| member)](api/languages/cpp_api.ht |                                   |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | function)](api/languages/cpp_api. |
| -   [cudaq::kraus_op::nRows (C++  | html#_CPPv4N5cudaq16quantum_platf |
|                                   | orm15onRandomSeedSetENSt6size_tE) |
| member)](api/languages/cpp_api.ht | -   [cudaq:                       |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | :quantum_platform::reset_exec_ctx |
| -   [cudaq::kraus_op::operator=   |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)                     | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ](api/languages/cpp_api.html#_CPP | antum_platform14reset_exec_ctxEv) |
| v4N5cudaq8kraus_opaSERK8kraus_op) | -   [cud                          |
| -   [cudaq::kraus_op::precision   | aq::quantum_platform::reset_noise |
|     (C++                          |     (C++                          |
|     memb                          |     function)](api/languages/cpp_ |
| er)](api/languages/cpp_api.html#_ | api.html#_CPPv4N5cudaq16quantum_p |
| CPPv4N5cudaq8kraus_op9precisionE) | latform11reset_noiseENSt6size_tE) |
| -   [cudaq::KrausSelection (C++   | -   [cudaq:                       |
|     s                             | :quantum_platform::resetLogStream |
| truct)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14KrausSelectionE) |     function)](api/languag        |
| -   [cudaq:                       | es/cpp_api.html#_CPPv4N5cudaq16qu |
| :KrausSelection::circuit_location | antum_platform14resetLogStreamEv) |
|     (C++                          | -   [cuda                         |
|     member)](api/langua           | q::quantum_platform::set_exec_ctx |
| ges/cpp_api.html#_CPPv4N5cudaq14K |     (C++                          |
| rausSelection16circuit_locationE) |     funct                         |
| -                                 | ion)](api/languages/cpp_api.html# |
|  [cudaq::KrausSelection::is_error | _CPPv4N5cudaq16quantum_platform12 |
|     (C++                          | set_exec_ctxEP16ExecutionContext) |
|     member)](a                    | -   [c                            |
| pi/languages/cpp_api.html#_CPPv4N | udaq::quantum_platform::set_noise |
| 5cudaq14KrausSelection8is_errorE) |     (C++                          |
| -   [cudaq::Kra                   |     function                      |
| usSelection::kraus_operator_index | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq16quantum_platform9set_ |
|     member)](api/languages/       | noiseEPK11noise_modelNSt6size_tE) |
| cpp_api.html#_CPPv4N5cudaq14Kraus | -   [cuda                         |
| Selection20kraus_operator_indexE) | q::quantum_platform::setLogStream |
| -   [cuda                         |     (C++                          |
| q::KrausSelection::KrausSelection |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     function)](a                  | .html#_CPPv4N5cudaq16quantum_plat |
| pi/languages/cpp_api.html#_CPPv4N | form12setLogStreamERNSt7ostreamE) |
| 5cudaq14KrausSelection14KrausSele | -   [cudaq::quantum_platfor       |
| ctionENSt6size_tENSt6vectorINSt6s | m::supports_explicit_measurements |
| ize_tEEENSt6stringENSt6size_tEb), |     (C++                          |
|     [\[1\]](api/langu             |     function)](api/l              |
| ages/cpp_api.html#_CPPv4N5cudaq14 | anguages/cpp_api.html#_CPPv4NK5cu |
| KrausSelection14KrausSelectionEv) | daq16quantum_platform30supports_e |
| -                                 | xplicit_measurementsENSt6size_tE) |
|   [cudaq::KrausSelection::op_name | -   [cudaq::quantum_pla           |
|     (C++                          | tform::supports_task_distribution |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     fu                            |
| N5cudaq14KrausSelection7op_nameE) | nction)](api/languages/cpp_api.ht |
| -   [                             | ml#_CPPv4NK5cudaq16quantum_platfo |
| cudaq::KrausSelection::operator== | rm26supports_task_distributionEv) |
|     (C++                          | -   [cudaq::quantum               |
|     function)](api/languages      | _platform::with_execution_context |
| /cpp_api.html#_CPPv4NK5cudaq14Kra |     (C++                          |
| usSelectioneqERK14KrausSelection) |     function)                     |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|    [cudaq::KrausSelection::qubits | v4I0DpEN5cudaq16quantum_platform2 |
|     (C++                          | 2with_execution_contextEDaR16Exec |
|     member)]                      | utionContextRR8CallableDpRR4Args) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::QuantumTask (C++      |
| 4N5cudaq14KrausSelection6qubitsE) |     type)](api/languages/cpp_api. |
| -   [cudaq::KrausTrajectory (C++  | html#_CPPv4N5cudaq11QuantumTaskE) |
|     st                            | -   [cudaq::qubit (C++            |
| ruct)](api/languages/cpp_api.html |     type)](api/languages/c        |
| #_CPPv4N5cudaq15KrausTrajectoryE) | pp_api.html#_CPPv4N5cudaq5qubitE) |
| -                                 | -   [cudaq::QubitConnectivity     |
|  [cudaq::KrausTrajectory::builder |     (C++                          |
|     (C++                          |     ty                            |
|     function)](ap                 | pe)](api/languages/cpp_api.html#_ |
| i/languages/cpp_api.html#_CPPv4N5 | CPPv4N5cudaq17QubitConnectivityE) |
| cudaq15KrausTrajectory7builderEv) | -   [cudaq::QubitEdge (C++        |
| -   [cu                           |     type)](api/languages/cpp_a    |
| daq::KrausTrajectory::countErrors | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|     (C++                          | -   [cudaq::qudit (C++            |
|     function)](api/lang           |     clas                          |
| uages/cpp_api.html#_CPPv4NK5cudaq | s)](api/languages/cpp_api.html#_C |
| 15KrausTrajectory11countErrorsEv) | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| -   [                             | -   [cudaq::qudit::qudit (C++     |
| cudaq::KrausTrajectory::isOrdered |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     function)](api/l              | html#_CPPv4N5cudaq5qudit5quditEv) |
| anguages/cpp_api.html#_CPPv4NK5cu | -   [cudaq::qvector (C++          |
| daq15KrausTrajectory9isOrderedEv) |     class)                        |
| -   [cudaq::                      | ](api/languages/cpp_api.html#_CPP |
| KrausTrajectory::kraus_selections | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|     (C++                          | -   [cudaq::qvector::back (C++    |
|     member)](api/languag          |     function)](a                  |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | pi/languages/cpp_api.html#_CPPv4N |
| ausTrajectory16kraus_selectionsE) | 5cudaq7qvector4backENSt6size_tE), |
| -   [cudaq:                       |                                   |
| :KrausTrajectory::KrausTrajectory |   [\[1\]](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq7qvector4backEv) |
|     function                      | -   [cudaq::qvector::begin (C++   |
| )](api/languages/cpp_api.html#_CP |     fu                            |
| Pv4N5cudaq15KrausTrajectory15Krau | nction)](api/languages/cpp_api.ht |
| sTrajectoryENSt6size_tENSt6vector | ml#_CPPv4N5cudaq7qvector5beginEv) |
| I14KrausSelectionEEdNSt6size_tE), | -   [cudaq::qvector::clear (C++   |
|     [\[1\]](api/languag           |     fu                            |
| es/cpp_api.html#_CPPv4N5cudaq15Kr | nction)](api/languages/cpp_api.ht |
| ausTrajectory15KrausTrajectoryEv) | ml#_CPPv4N5cudaq7qvector5clearEv) |
| -   [cudaq::Kr                    | -   [cudaq::qvector::end (C++     |
| ausTrajectory::measurement_counts |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     member)](api/languages        | html#_CPPv4N5cudaq7qvector3endEv) |
| /cpp_api.html#_CPPv4N5cudaq15Krau | -   [cudaq::qvector::front (C++   |
| sTrajectory18measurement_countsE) |     function)](ap                 |
| -   [cud                          | i/languages/cpp_api.html#_CPPv4N5 |
| aq::KrausTrajectory::multiplicity | cudaq7qvector5frontENSt6size_tE), |
|     (C++                          |                                   |
|     member)](api/lan              |  [\[1\]](api/languages/cpp_api.ht |
| guages/cpp_api.html#_CPPv4N5cudaq | ml#_CPPv4N5cudaq7qvector5frontEv) |
| 15KrausTrajectory12multiplicityE) | -   [cudaq::qvector::operator=    |
| -   [                             |     (C++                          |
| cudaq::KrausTrajectory::num_shots |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](api                  | PPv4N5cudaq7qvectoraSERK7qvector) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::qvector::operator\[\] |
| udaq15KrausTrajectory9num_shotsE) |     (C++                          |
| -   [c                            |     function)                     |
| udaq::KrausTrajectory::operator== | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq7qvectorixEKNSt6size_tE) |
|     function)](api/languages/c    | -   [cudaq::qvector::qvector (C++ |
| pp_api.html#_CPPv4NK5cudaq15Kraus |     function)](api/               |
| TrajectoryeqERK15KrausTrajectory) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cu                           | daq7qvector7qvectorENSt6size_tE), |
| daq::KrausTrajectory::probability |     [\[1\]](a                     |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     member)](api/la               | 5cudaq7qvector7qvectorERK5state), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[2\]](api                   |
| q15KrausTrajectory11probabilityE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cuda                         | udaq7qvector7qvectorERK7qvector), |
| q::KrausTrajectory::trajectory_id |     [\[3\]](ap                    |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     member)](api/lang             | cudaq7qvector7qvectorERR7qvector) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq::qvector::size (C++    |
| 5KrausTrajectory13trajectory_idE) |     fu                            |
| -                                 | nction)](api/languages/cpp_api.ht |
|   [cudaq::KrausTrajectory::weight | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     (C++                          | -   [cudaq::qvector::slice (C++   |
|     member)](                     |     function)](api/language       |
| api/languages/cpp_api.html#_CPPv4 | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| N5cudaq15KrausTrajectory6weightE) | tor5sliceENSt6size_tENSt6size_tE) |
| -                                 | -   [cudaq::qvector::value_type   |
|    [cudaq::KrausTrajectoryBuilder |     (C++                          |
|     (C++                          |     typ                           |
|     class)](                      | e)](api/languages/cpp_api.html#_C |
| api/languages/cpp_api.html#_CPPv4 | PPv4N5cudaq7qvector10value_typeE) |
| N5cudaq22KrausTrajectoryBuilderE) | -   [cudaq::qview (C++            |
| -   [cud                          |     clas                          |
| aq::KrausTrajectoryBuilder::build | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     function)](api/lang           | -   [cudaq::qview::back (C++      |
| uages/cpp_api.html#_CPPv4NK5cudaq |     function)                     |
| 22KrausTrajectoryBuilder5buildEv) | ](api/languages/cpp_api.html#_CPP |
| -   [cud                          | v4N5cudaq5qview4backENSt6size_tE) |
| aq::KrausTrajectoryBuilder::setId | -   [cudaq::qview::begin (C++     |
|     (C++                          |                                   |
|     function)](api/languages/cpp  | function)](api/languages/cpp_api. |
| _api.html#_CPPv4N5cudaq22KrausTra | html#_CPPv4N5cudaq5qview5beginEv) |
| jectoryBuilder5setIdENSt6size_tE) | -   [cudaq::qview::end (C++       |
| -   [cudaq::Kraus                 |                                   |
| TrajectoryBuilder::setProbability |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4N5cudaq5qview3endEv) |
|     function)](api/languages/cpp  | -   [cudaq::qview::front (C++     |
| _api.html#_CPPv4N5cudaq22KrausTra |     function)](                   |
| jectoryBuilder14setProbabilityEd) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::Krau                  | N5cudaq5qview5frontENSt6size_tE), |
| sTrajectoryBuilder::setSelections |                                   |
|     (C++                          |    [\[1\]](api/languages/cpp_api. |
|     function)](api/languag        | html#_CPPv4N5cudaq5qview5frontEv) |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | -   [cudaq::qview::operator\[\]   |
| ausTrajectoryBuilder13setSelectio |     (C++                          |
| nsENSt6vectorI14KrausSelectionEE) |     functio                       |
| -   [cudaq::matrix_callback (C++  | n)](api/languages/cpp_api.html#_C |
|     c                             | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| lass)](api/languages/cpp_api.html | -   [cudaq::qview::qview (C++     |
| #_CPPv4N5cudaq15matrix_callbackE) |     functio                       |
| -   [cudaq::matrix_handler (C++   | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4I0EN5cudaq5qview5qviewERR1R), |
| class)](api/languages/cpp_api.htm |     [\[1                          |
| l#_CPPv4N5cudaq14matrix_handlerE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::mat                   | PPv4N5cudaq5qview5qviewERK5qview) |
| rix_handler::commutation_behavior | -   [cudaq::qview::size (C++      |
|     (C++                          |                                   |
|     struct)](api/languages/       | function)](api/languages/cpp_api. |
| cpp_api.html#_CPPv4N5cudaq14matri | html#_CPPv4NK5cudaq5qview4sizeEv) |
| x_handler20commutation_behaviorE) | -   [cudaq::qview::slice (C++     |
| -                                 |     function)](api/langua         |
|    [cudaq::matrix_handler::define | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|     (C++                          | iew5sliceENSt6size_tENSt6size_tE) |
|     function)](a                  | -   [cudaq::qview::value_type     |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14matrix_handler6defineENSt |     t                             |
| 6stringENSt6vectorINSt7int64_tEEE | ype)](api/languages/cpp_api.html# |
| RR15matrix_callbackRKNSt13unorder | _CPPv4N5cudaq5qview10value_typeE) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cudaq::range (C++            |
|                                   |     fun                           |
| [\[1\]](api/languages/cpp_api.htm | ction)](api/languages/cpp_api.htm |
| l#_CPPv4N5cudaq14matrix_handler6d | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| efineENSt6stringENSt6vectorINSt7i | orI11ElementTypeEE11ElementType), |
| nt64_tEEERR15matrix_callbackRR20d |     [\[1\]](api/languages/cpp_    |
| iag_matrix_callbackRKNSt13unorder | api.html#_CPPv4I0EN5cudaq5rangeEN |
| ed_mapINSt6stringENSt6stringEEE), | St6vectorI11ElementTypeEE11Elemen |
|     [\[2\]](                      | tType11ElementType11ElementType), |
| api/languages/cpp_api.html#_CPPv4 |     [                             |
| N5cudaq14matrix_handler6defineENS | \[2\]](api/languages/cpp_api.html |
| t6stringENSt6vectorINSt7int64_tEE | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| ERR15matrix_callbackRRNSt13unorde | -   [cudaq::real (C++             |
| red_mapINSt6stringENSt6stringEEE) |     type)](api/languages/         |
| -                                 | cpp_api.html#_CPPv4N5cudaq4realE) |
|   [cudaq::matrix_handler::degrees | -   [cudaq::registry (C++         |
|     (C++                          |     type)](api/languages/cpp_     |
|     function)](ap                 | api.html#_CPPv4N5cudaq8registryE) |
| i/languages/cpp_api.html#_CPPv4NK | -                                 |
| 5cudaq14matrix_handler7degreesEv) |  [cudaq::registry::RegisteredType |
| -                                 |     (C++                          |
|  [cudaq::matrix_handler::displace |     class)](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)](api/language       | 5cudaq8registry14RegisteredTypeE) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cudaq::RemoteCapabilities    |
| rix_handler8displaceENSt6size_tE) |     (C++                          |
| -   [cudaq::matrix                |     struc                         |
| _handler::get_expected_dimensions | t)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq18RemoteCapabilitiesE) |
|                                   | -   [cudaq::Remo                  |
|    function)](api/languages/cpp_a | teCapabilities::isRemoteSimulator |
| pi.html#_CPPv4NK5cudaq14matrix_ha |     (C++                          |
| ndler23get_expected_dimensionsEv) |     member)](api/languages/c      |
| -   [cudaq::matrix_ha             | pp_api.html#_CPPv4N5cudaq18Remote |
| ndler::get_parameter_descriptions | Capabilities17isRemoteSimulatorE) |
|     (C++                          | -   [cudaq::Remot                 |
|                                   | eCapabilities::RemoteCapabilities |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4NK5cudaq14matrix_handl |     function)](api/languages/cpp  |
| er26get_parameter_descriptionsEv) | _api.html#_CPPv4N5cudaq18RemoteCa |
| -   [c                            | pabilities18RemoteCapabilitiesEb) |
| udaq::matrix_handler::instantiate | -   [cudaq:                       |
|     (C++                          | :RemoteCapabilities::stateOverlap |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     member)](api/langua           |
| 5cudaq14matrix_handler11instantia | ges/cpp_api.html#_CPPv4N5cudaq18R |
| teENSt6stringERKNSt6vectorINSt6si | emoteCapabilities12stateOverlapE) |
| ze_tEEERK20commutation_behavior), | -                                 |
|     [\[1\]](                      |   [cudaq::RemoteCapabilities::vqe |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14matrix_handler11instanti |     member)](                     |
| ateENSt6stringERRNSt6vectorINSt6s | api/languages/cpp_api.html#_CPPv4 |
| ize_tEEERK20commutation_behavior) | N5cudaq18RemoteCapabilities3vqeE) |
| -   [cuda                         | -   [cudaq::RemoteSimulationState |
| q::matrix_handler::matrix_handler |     (C++                          |
|     (C++                          |     class)]                       |
|     function)](api/languag        | (api/languages/cpp_api.html#_CPPv |
| es/cpp_api.html#_CPPv4I0_NSt11ena | 4N5cudaq21RemoteSimulationStateE) |
| ble_if_tINSt12is_base_of_vI16oper | -   [cudaq::Resources (C++        |
| ator_handler1TEEbEEEN5cudaq14matr |     class)](api/languages/cpp_a   |
| ix_handler14matrix_handlerERK1T), | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     [\[1\]](ap                    | -   [cudaq::run (C++              |
| i/languages/cpp_api.html#_CPPv4I0 |     function)]                    |
| _NSt11enable_if_tINSt12is_base_of | (api/languages/cpp_api.html#_CPPv |
| _vI16operator_handler1TEEbEEEN5cu | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| daq14matrix_handler14matrix_handl | 5invoke_result_tINSt7decay_tI13Qu |
| erERK1TRK20commutation_behavior), | antumKernelEEDpNSt7decay_tI4ARGSE |
|     [\[2\]](api/languages/cpp_ap  | EEEEENSt6size_tERN5cudaq11noise_m |
| i.html#_CPPv4N5cudaq14matrix_hand | odelERR13QuantumKernelDpRR4ARGS), |
| ler14matrix_handlerENSt6size_tE), |     [\[1\]](api/langu             |
|     [\[3\]](api/                  | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| languages/cpp_api.html#_CPPv4N5cu | daq3runENSt6vectorINSt15invoke_re |
| daq14matrix_handler14matrix_handl | sult_tINSt7decay_tI13QuantumKerne |
| erENSt6stringERKNSt6vectorINSt6si | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| ze_tEEERK20commutation_behavior), | ize_tERR13QuantumKernelDpRR4ARGS) |
|     [\[4\]](api/                  | -   [cudaq::run_async (C++        |
| languages/cpp_api.html#_CPPv4N5cu |     functio                       |
| daq14matrix_handler14matrix_handl | n)](api/languages/cpp_api.html#_C |
| erENSt6stringERRNSt6vectorINSt6si | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| ze_tEEERK20commutation_behavior), | tureINSt6vectorINSt15invoke_resul |
|     [\                            | t_tINSt7decay_tI13QuantumKernelEE |
| [5\]](api/languages/cpp_api.html# | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| _CPPv4N5cudaq14matrix_handler14ma | ze_tENSt6size_tERN5cudaq11noise_m |
| trix_handlerERK14matrix_handler), | odelERR13QuantumKernelDpRR4ARGS), |
|     [                             |     [\[1\]](api/la                |
| \[6\]](api/languages/cpp_api.html | nguages/cpp_api.html#_CPPv4I0DpEN |
| #_CPPv4N5cudaq14matrix_handler14m | 5cudaq9run_asyncENSt6futureINSt6v |
| atrix_handlerERR14matrix_handler) | ectorINSt15invoke_result_tINSt7de |
| -                                 | cay_tI13QuantumKernelEEDpNSt7deca |
|  [cudaq::matrix_handler::momentum | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4ARGS) |
|     function)](api/language       | -   [cudaq::RuntimeTarget (C++    |
| s/cpp_api.html#_CPPv4N5cudaq14mat |                                   |
| rix_handler8momentumENSt6size_tE) | struct)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|    [cudaq::matrix_handler::number | -   [cudaq::sample (C++           |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/langua         | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| ges/cpp_api.html#_CPPv4N5cudaq14m | mpleE13sample_resultRK14sample_op |
| atrix_handler6numberENSt6size_tE) | tionsRR13QuantumKernelDpRR4Args), |
| -                                 |     [\[1\                         |
| [cudaq::matrix_handler::operator= | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     fun                           | esultRR13QuantumKernelDpRR4Args), |
| ction)](api/languages/cpp_api.htm |     [\                            |
| l#_CPPv4I0_NSt11enable_if_tIXaant | [2\]](api/languages/cpp_api.html# |
| NSt7is_sameI1T14matrix_handlerE5v | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| alueENSt12is_base_of_vI16operator | ize_tERR13QuantumKernelDpRR4Args) |
| _handler1TEEEbEEEN5cudaq14matrix_ | -   [cudaq::sample_options (C++   |
| handleraSER14matrix_handlerRK1T), |     s                             |
|     [\[1\]](api/languages         | truct)](api/languages/cpp_api.htm |
| /cpp_api.html#_CPPv4N5cudaq14matr | l#_CPPv4N5cudaq14sample_optionsE) |
| ix_handleraSERK14matrix_handler), | -   [cudaq::sample_result (C++    |
|     [\[2\]](api/language          |                                   |
| s/cpp_api.html#_CPPv4N5cudaq14mat |  class)](api/languages/cpp_api.ht |
| rix_handleraSERR14matrix_handler) | ml#_CPPv4N5cudaq13sample_resultE) |
| -   [                             | -   [cudaq::sample_result::append |
| cudaq::matrix_handler::operator== |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     function)](api/languages      | api.html#_CPPv4N5cudaq13sample_re |
| /cpp_api.html#_CPPv4NK5cudaq14mat | sult6appendERK15ExecutionResultb) |
| rix_handlereqERK14matrix_handler) | -   [cudaq::sample_result::begin  |
| -                                 |     (C++                          |
|    [cudaq::matrix_handler::parity |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/langua         | 4N5cudaq13sample_result5beginEv), |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     [\[1\]]                       |
| atrix_handler6parityENSt6size_tE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NK5cudaq13sample_result5beginEv) |
|  [cudaq::matrix_handler::position | -   [cudaq::sample_result::cbegin |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](                   |
| s/cpp_api.html#_CPPv4N5cudaq14mat | api/languages/cpp_api.html#_CPPv4 |
| rix_handler8positionENSt6size_tE) | NK5cudaq13sample_result6cbeginEv) |
| -   [cudaq::                      | -   [cudaq::sample_result::cend   |
| matrix_handler::remove_definition |     (C++                          |
|     (C++                          |     function)                     |
|     fu                            | ](api/languages/cpp_api.html#_CPP |
| nction)](api/languages/cpp_api.ht | v4NK5cudaq13sample_result4cendEv) |
| ml#_CPPv4N5cudaq14matrix_handler1 | -   [cudaq::sample_result::clear  |
| 7remove_definitionERKNSt6stringE) |     (C++                          |
| -                                 |     function)                     |
|   [cudaq::matrix_handler::squeeze | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq13sample_result5clearEv) |
|     function)](api/languag        | -   [cudaq::sample_result::count  |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     (C++                          |
| trix_handler7squeezeENSt6size_tE) |     function)](                   |
| -   [cudaq::m                     | api/languages/cpp_api.html#_CPPv4 |
| atrix_handler::to_diagonal_matrix | NK5cudaq13sample_result5countENSt |
|     (C++                          | 11string_viewEKNSt11string_viewE) |
|     function)](api/lang           | -   [                             |
| uages/cpp_api.html#_CPPv4NK5cudaq | cudaq::sample_result::deserialize |
| 14matrix_handler18to_diagonal_mat |     (C++                          |
| rixERNSt13unordered_mapINSt6size_ |     functio                       |
| tENSt7int64_tEEERKNSt13unordered_ | n)](api/languages/cpp_api.html#_C |
| mapINSt6stringENSt7complexIdEEEE) | PPv4N5cudaq13sample_result11deser |
| -                                 | ializeERNSt6vectorINSt6size_tEEE) |
| [cudaq::matrix_handler::to_matrix | -   [cudaq::sample_result::dump   |
|     (C++                          |     (C++                          |
|     function)                     |     function)](api/languag        |
| ](api/languages/cpp_api.html#_CPP | es/cpp_api.html#_CPPv4NK5cudaq13s |
| v4NK5cudaq14matrix_handler9to_mat | ample_result4dumpERNSt7ostreamE), |
| rixERNSt13unordered_mapINSt6size_ |     [\[1\]                        |
| tENSt7int64_tEEERKNSt13unordered_ | ](api/languages/cpp_api.html#_CPP |
| mapINSt6stringENSt7complexIdEEEE) | v4NK5cudaq13sample_result4dumpEv) |
| -                                 | -   [cudaq::sample_result::end    |
| [cudaq::matrix_handler::to_string |     (C++                          |
|     (C++                          |     function                      |
|     function)](api/               | )](api/languages/cpp_api.html#_CP |
| languages/cpp_api.html#_CPPv4NK5c | Pv4N5cudaq13sample_result3endEv), |
| udaq14matrix_handler9to_stringEb) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
| [cudaq::matrix_handler::unique_id | Pv4NK5cudaq13sample_result3endEv) |
|     (C++                          | -   [                             |
|     function)](api/               | cudaq::sample_result::expectation |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14matrix_handler9unique_idEv) |     f                             |
| -   [cudaq:                       | unction)](api/languages/cpp_api.h |
| :matrix_handler::\~matrix_handler | tml#_CPPv4NK5cudaq13sample_result |
|     (C++                          | 11expectationEKNSt11string_viewE) |
|     functi                        | -   [c                            |
| on)](api/languages/cpp_api.html#_ | udaq::sample_result::get_marginal |
| CPPv4N5cudaq14matrix_handlerD0Ev) |     (C++                          |
| -   [cudaq::matrix_op (C++        |     function)](api/languages/cpp_ |
|     type)](api/languages/cpp_a    | api.html#_CPPv4NK5cudaq13sample_r |
| pi.html#_CPPv4N5cudaq9matrix_opE) | esult12get_marginalERKNSt6vectorI |
| -   [cudaq::matrix_op_term (C++   | NSt6size_tEEEKNSt11string_viewE), |
|                                   |     [\[1\]](api/languages/cpp_    |
|  type)](api/languages/cpp_api.htm | api.html#_CPPv4NK5cudaq13sample_r |
| l#_CPPv4N5cudaq14matrix_op_termE) | esult12get_marginalERRKNSt6vector |
| -                                 | INSt6size_tEEEKNSt11string_viewE) |
|    [cudaq::mdiag_operator_handler | -   [cuda                         |
|     (C++                          | q::sample_result::get_total_shots |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/langua         |
| N5cudaq22mdiag_operator_handlerE) | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| -   [cudaq::mpi (C++              | sample_result15get_total_shotsEv) |
|     type)](api/languages          | -   [cuda                         |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | q::sample_result::has_even_parity |
| -   [cudaq::mpi::all_gather (C++  |     (C++                          |
|     fu                            |     fun                           |
| nction)](api/languages/cpp_api.ht | ction)](api/languages/cpp_api.htm |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | l#_CPPv4N5cudaq13sample_result15h |
| RNSt6vectorIdEERKNSt6vectorIdEE), | as_even_parityENSt11string_viewE) |
|                                   | -   [cuda                         |
|   [\[1\]](api/languages/cpp_api.h | q::sample_result::has_expectation |
| tml#_CPPv4N5cudaq3mpi10all_gather |     (C++                          |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     funct                         |
| -   [cudaq::mpi::all_reduce (C++  | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4NK5cudaq13sample_result15ha |
|  function)](api/languages/cpp_api | s_expectationEKNSt11string_viewE) |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | -   [cu                           |
| reduceE1TRK1TRK14BinaryFunction), | daq::sample_result::most_probable |
|     [\[1\]](api/langu             |     (C++                          |
| ages/cpp_api.html#_CPPv4I00EN5cud |     fun                           |
| aq3mpi10all_reduceE1TRK1TRK4Func) | ction)](api/languages/cpp_api.htm |
| -   [cudaq::mpi::broadcast (C++   | l#_CPPv4NK5cudaq13sample_result13 |
|     function)](api/               | most_probableEKNSt11string_viewE) |
| languages/cpp_api.html#_CPPv4N5cu | -                                 |
| daq3mpi9broadcastERNSt6stringEi), | [cudaq::sample_result::operator+= |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/langua         |
| q3mpi9broadcastERNSt6vectorIdEEi) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -   [cudaq::mpi::finalize (C++    | ample_resultpLERK13sample_result) |
|     f                             | -                                 |
| unction)](api/languages/cpp_api.h |  [cudaq::sample_result::operator= |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     (C++                          |
| -   [cudaq::mpi::initialize (C++  |     function)](api/langua         |
|     function                      | ges/cpp_api.html#_CPPv4N5cudaq13s |
| )](api/languages/cpp_api.html#_CP | ample_resultaSERR13sample_result) |
| Pv4N5cudaq3mpi10initializeEiPPc), | -                                 |
|     [                             | [cudaq::sample_result::operator== |
| \[1\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq3mpi10initializeEv) |     function)](api/languag        |
| -   [cudaq::mpi::is_initialized   | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     (C++                          | ample_resulteqERK13sample_result) |
|     function                      | -   [                             |
| )](api/languages/cpp_api.html#_CP | cudaq::sample_result::probability |
| Pv4N5cudaq3mpi14is_initializedEv) |     (C++                          |
| -   [cudaq::mpi::num_ranks (C++   |     function)](api/lan            |
|     fu                            | guages/cpp_api.html#_CPPv4NK5cuda |
| nction)](api/languages/cpp_api.ht | q13sample_result11probabilityENSt |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | 11string_viewEKNSt11string_viewE) |
| -   [cudaq::mpi::rank (C++        | -   [cud                          |
|                                   | aq::sample_result::register_names |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     function)](api/langu          |
| -   [cudaq::noise_model (C++      | ages/cpp_api.html#_CPPv4NK5cudaq1 |
|                                   | 3sample_result14register_namesEv) |
|    class)](api/languages/cpp_api. | -                                 |
| html#_CPPv4N5cudaq11noise_modelE) |    [cudaq::sample_result::reorder |
| -   [cudaq::n                     |     (C++                          |
| oise_model::add_all_qubit_channel |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     function)](api                | ample_result7reorderERKNSt6vector |
| /languages/cpp_api.html#_CPPv4IDp | INSt6size_tEEEKNSt11string_viewE) |
| EN5cudaq11noise_model21add_all_qu | -   [cu                           |
| bit_channelEvRK13kraus_channeli), | daq::sample_result::sample_result |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq11n |     func                          |
| oise_model21add_all_qubit_channel | tion)](api/languages/cpp_api.html |
| ERKNSt6stringERK13kraus_channeli) | #_CPPv4N5cudaq13sample_result13sa |
| -                                 | mple_resultERK15ExecutionResult), |
|  [cudaq::noise_model::add_channel |     [\[1\]](api/la                |
|     (C++                          | nguages/cpp_api.html#_CPPv4N5cuda |
|     funct                         | q13sample_result13sample_resultER |
| ion)](api/languages/cpp_api.html# | KNSt6vectorI15ExecutionResultEE), |
| _CPPv4IDpEN5cudaq11noise_model11a |                                   |
| dd_channelEvRK15PredicateFuncTy), |  [\[2\]](api/languages/cpp_api.ht |
|     [\[1\]](api/languages/cpp_    | ml#_CPPv4N5cudaq13sample_result13 |
| api.html#_CPPv4IDpEN5cudaq11noise | sample_resultERR13sample_result), |
| _model11add_channelEvRKNSt6vector |     [                             |
| INSt6size_tEEERK13kraus_channel), | \[3\]](api/languages/cpp_api.html |
|     [\[2\]](ap                    | #_CPPv4N5cudaq13sample_result13sa |
| i/languages/cpp_api.html#_CPPv4N5 | mple_resultERR15ExecutionResult), |
| cudaq11noise_model11add_channelER |     [\[4\]](api/lan               |
| KNSt6stringERK15PredicateFuncTy), | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 13sample_result13sample_resultEdR |
| [\[3\]](api/languages/cpp_api.htm | KNSt6vectorI15ExecutionResultEE), |
| l#_CPPv4N5cudaq11noise_model11add |     [\[5\]](api/lan               |
| _channelERKNSt6stringERKNSt6vecto | guages/cpp_api.html#_CPPv4N5cudaq |
| rINSt6size_tEEERK13kraus_channel) | 13sample_result13sample_resultEv) |
| -   [cudaq::noise_model::empty    | -                                 |
|     (C++                          |  [cudaq::sample_result::serialize |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api                |
| Pv4NK5cudaq11noise_model5emptyEv) | /languages/cpp_api.html#_CPPv4NK5 |
| -                                 | cudaq13sample_result9serializeEv) |
| [cudaq::noise_model::get_channels | -   [cudaq::sample_result::size   |
|     (C++                          |     (C++                          |
|     function)](api/l              |     function)](api/languages/c    |
| anguages/cpp_api.html#_CPPv4I0ENK | pp_api.html#_CPPv4NK5cudaq13sampl |
| 5cudaq11noise_model12get_channels | e_result4sizeEKNSt11string_viewE) |
| ENSt6vectorI13kraus_channelEERKNS | -   [cudaq::sample_result::to_map |
| t6vectorINSt6size_tEEERKNSt6vecto |     (C++                          |
| rINSt6size_tEEERKNSt6vectorIdEE), |     function)](api/languages/cpp  |
|     [\[1\]](api/languages/cpp_a   | _api.html#_CPPv4NK5cudaq13sample_ |
| pi.html#_CPPv4NK5cudaq11noise_mod | result6to_mapEKNSt11string_viewE) |
| el12get_channelsERKNSt6stringERKN | -   [cuda                         |
| St6vectorINSt6size_tEEERKNSt6vect | q::sample_result::\~sample_result |
| orINSt6size_tEEERKNSt6vectorIdEE) |     (C++                          |
| -                                 |     funct                         |
|  [cudaq::noise_model::noise_model | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq13sample_resultD0Ev) |
|     function)](api                | -   [cudaq::scalar_callback (C++  |
| /languages/cpp_api.html#_CPPv4N5c |     c                             |
| udaq11noise_model11noise_modelEv) | lass)](api/languages/cpp_api.html |
| -   [cu                           | #_CPPv4N5cudaq15scalar_callbackE) |
| daq::noise_model::PredicateFuncTy | -   [c                            |
|     (C++                          | udaq::scalar_callback::operator() |
|     type)](api/la                 |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/language       |
| q11noise_model15PredicateFuncTyE) | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| -   [cud                          | alar_callbackclERKNSt13unordered_ |
| aq::noise_model::register_channel | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [                             |
|     function)](api/languages      | cudaq::scalar_callback::operator= |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     (C++                          |
| noise_model16register_channelEvv) |     function)](api/languages/c    |
| -   [cudaq::                      | pp_api.html#_CPPv4N5cudaq15scalar |
| noise_model::requires_constructor | _callbackaSERK15scalar_callback), |
|     (C++                          |     [\[1\]](api/languages/        |
|     type)](api/languages/cp       | cpp_api.html#_CPPv4N5cudaq15scala |
| p_api.html#_CPPv4I0DpEN5cudaq11no | r_callbackaSERR15scalar_callback) |
| ise_model20requires_constructorE) | -   [cudaq:                       |
| -   [cudaq::noise_model_type (C++ | :scalar_callback::scalar_callback |
|     e                             |     (C++                          |
| num)](api/languages/cpp_api.html# |     function)](api/languag        |
| _CPPv4N5cudaq16noise_model_typeE) | es/cpp_api.html#_CPPv4I0_NSt11ena |
| -   [cudaq::no                    | ble_if_tINSt16is_invocable_r_vINS |
| ise_model_type::amplitude_damping | t7complexIdEE8CallableRKNSt13unor |
|     (C++                          | dered_mapINSt6stringENSt7complexI |
|     enumerator)](api/languages    | dEEEEEEbEEEN5cudaq15scalar_callba |
| /cpp_api.html#_CPPv4N5cudaq16nois | ck15scalar_callbackERR8Callable), |
| e_model_type17amplitude_dampingE) |     [\[1\                         |
| -   [cudaq::noise_mode            | ]](api/languages/cpp_api.html#_CP |
| l_type::amplitude_damping_channel | Pv4N5cudaq15scalar_callback15scal |
|     (C++                          | ar_callbackERK15scalar_callback), |
|     e                             |     [\[2                          |
| numerator)](api/languages/cpp_api | \]](api/languages/cpp_api.html#_C |
| .html#_CPPv4N5cudaq16noise_model_ | PPv4N5cudaq15scalar_callback15sca |
| type25amplitude_damping_channelE) | lar_callbackERR15scalar_callback) |
| -   [cudaq::n                     | -   [cudaq::scalar_operator (C++  |
| oise_model_type::bit_flip_channel |     c                             |
|     (C++                          | lass)](api/languages/cpp_api.html |
|     enumerator)](api/language     | #_CPPv4N5cudaq15scalar_operatorE) |
| s/cpp_api.html#_CPPv4N5cudaq16noi | -                                 |
| se_model_type16bit_flip_channelE) | [cudaq::scalar_operator::evaluate |
| -   [cudaq::                      |     (C++                          |
| noise_model_type::depolarization1 |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     enumerator)](api/languag      | pi.html#_CPPv4NK5cudaq15scalar_op |
| es/cpp_api.html#_CPPv4N5cudaq16no | erator8evaluateERKNSt13unordered_ |
| ise_model_type15depolarization1E) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::                      | -   [cudaq::scalar_ope            |
| noise_model_type::depolarization2 | rator::get_parameter_descriptions |
|     (C++                          |     (C++                          |
|     enumerator)](api/languag      |     f                             |
| es/cpp_api.html#_CPPv4N5cudaq16no | unction)](api/languages/cpp_api.h |
| ise_model_type15depolarization2E) | tml#_CPPv4NK5cudaq15scalar_operat |
| -   [cudaq::noise_m               | or26get_parameter_descriptionsEv) |
| odel_type::depolarization_channel | -   [cu                           |
|     (C++                          | daq::scalar_operator::is_constant |
|                                   |     (C++                          |
|   enumerator)](api/languages/cpp_ |     function)](api/lang           |
| api.html#_CPPv4N5cudaq16noise_mod | uages/cpp_api.html#_CPPv4NK5cudaq |
| el_type22depolarization_channelE) | 15scalar_operator11is_constantEv) |
| -                                 | -   [c                            |
|  [cudaq::noise_model_type::pauli1 | udaq::scalar_operator::operator\* |
|     (C++                          |     (C++                          |
|     enumerator)](a                |     function                      |
| pi/languages/cpp_api.html#_CPPv4N | )](api/languages/cpp_api.html#_CP |
| 5cudaq16noise_model_type6pauli1E) | Pv4N5cudaq15scalar_operatormlENSt |
| -                                 | 7complexIdEERK15scalar_operator), |
|  [cudaq::noise_model_type::pauli2 |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](a                | Pv4N5cudaq15scalar_operatormlENSt |
| pi/languages/cpp_api.html#_CPPv4N | 7complexIdEERR15scalar_operator), |
| 5cudaq16noise_model_type6pauli2E) |     [\[2\]](api/languages/cp      |
| -   [cudaq                        | p_api.html#_CPPv4N5cudaq15scalar_ |
| ::noise_model_type::phase_damping | operatormlEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     enumerator)](api/langu        | p_api.html#_CPPv4N5cudaq15scalar_ |
| ages/cpp_api.html#_CPPv4N5cudaq16 | operatormlEdRR15scalar_operator), |
| noise_model_type13phase_dampingE) |     [\[4\]](api/languages         |
| -   [cudaq::noi                   | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| se_model_type::phase_flip_channel | alar_operatormlENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     enumerator)](api/languages/   | _api.html#_CPPv4NKR5cudaq15scalar |
| cpp_api.html#_CPPv4N5cudaq16noise | _operatormlERK15scalar_operator), |
| _model_type18phase_flip_channelE) |     [\[6\]]                       |
| -                                 | (api/languages/cpp_api.html#_CPPv |
| [cudaq::noise_model_type::unknown | 4NKR5cudaq15scalar_operatormlEd), |
|     (C++                          |     [\[7\]](api/language          |
|     enumerator)](ap               | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| i/languages/cpp_api.html#_CPPv4N5 | alar_operatormlENSt7complexIdEE), |
| cudaq16noise_model_type7unknownE) |     [\[8\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4NO5cudaq15scalar |
| [cudaq::noise_model_type::x_error | _operatormlERK15scalar_operator), |
|     (C++                          |     [\[9\                         |
|     enumerator)](ap               | ]](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4NO5cudaq15scalar_operatormlEd) |
| cudaq16noise_model_type7x_errorE) | -   [cu                           |
| -                                 | daq::scalar_operator::operator\*= |
| [cudaq::noise_model_type::y_error |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](ap               | es/cpp_api.html#_CPPv4N5cudaq15sc |
| i/languages/cpp_api.html#_CPPv4N5 | alar_operatormLENSt7complexIdEE), |
| cudaq16noise_model_type7y_errorE) |     [\[1\]](api/languages/c       |
| -                                 | pp_api.html#_CPPv4N5cudaq15scalar |
| [cudaq::noise_model_type::z_error | _operatormLERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     enumerator)](ap               | \]](api/languages/cpp_api.html#_C |
| i/languages/cpp_api.html#_CPPv4N5 | PPv4N5cudaq15scalar_operatormLEd) |
| cudaq16noise_model_type7z_errorE) | -   [                             |
| -   [cudaq::num_available_gpus    | cudaq::scalar_operator::operator+ |
|     (C++                          |     (C++                          |
|     function                      |     function                      |
| )](api/languages/cpp_api.html#_CP | )](api/languages/cpp_api.html#_CP |
| Pv4N5cudaq18num_available_gpusEv) | Pv4N5cudaq15scalar_operatorplENSt |
| -   [cudaq::observe (C++          | 7complexIdEERK15scalar_operator), |
|     function)]                    |     [\[1\                         |
| (api/languages/cpp_api.html#_CPPv | ]](api/languages/cpp_api.html#_CP |
| 4I00DpEN5cudaq7observeENSt6vector | Pv4N5cudaq15scalar_operatorplENSt |
| I14observe_resultEERR13QuantumKer | 7complexIdEERR15scalar_operator), |
| nelRK15SpinOpContainerDpRR4Args), |     [\[2\]](api/languages/cp      |
|     [\[1\]](api/languages/cpp_ap  | p_api.html#_CPPv4N5cudaq15scalar_ |
| i.html#_CPPv4I0DpEN5cudaq7observe | operatorplEdRK15scalar_operator), |
| E14observe_resultNSt6size_tERR13Q |     [\[3\]](api/languages/cp      |
| uantumKernelRK7spin_opDpRR4Args), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[                           | operatorplEdRR15scalar_operator), |
| 2\]](api/languages/cpp_api.html#_ |     [\[4\]](api/languages         |
| CPPv4I0DpEN5cudaq7observeE14obser | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ve_resultRK15observe_optionsRR13Q | alar_operatorplENSt7complexIdEE), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[5\]](api/languages/cpp     |
|     [\[3\]](api/lang              | _api.html#_CPPv4NKR5cudaq15scalar |
| uages/cpp_api.html#_CPPv4I0DpEN5c | _operatorplERK15scalar_operator), |
| udaq7observeE14observe_resultRR13 |     [\[6\]]                       |
| QuantumKernelRK7spin_opDpRR4Args) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::observe_options (C++  | 4NKR5cudaq15scalar_operatorplEd), |
|     st                            |     [\[7\]]                       |
| ruct)](api/languages/cpp_api.html | (api/languages/cpp_api.html#_CPPv |
| #_CPPv4N5cudaq15observe_optionsE) | 4NKR5cudaq15scalar_operatorplEv), |
| -   [cudaq::observe_result (C++   |     [\[8\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| class)](api/languages/cpp_api.htm | alar_operatorplENSt7complexIdEE), |
| l#_CPPv4N5cudaq14observe_resultE) |     [\[9\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4NO5cudaq15scalar |
|    [cudaq::observe_result::counts | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[10\]                       |
|     function)](api/languages/c    | ](api/languages/cpp_api.html#_CPP |
| pp_api.html#_CPPv4N5cudaq14observ | v4NO5cudaq15scalar_operatorplEd), |
| e_result6countsERK12spin_op_term) |     [\[11\                        |
| -   [cudaq::observe_result::dump  | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatorplEv) |
|     function)                     | -   [c                            |
| ](api/languages/cpp_api.html#_CPP | udaq::scalar_operator::operator+= |
| v4N5cudaq14observe_result4dumpEv) |     (C++                          |
| -   [c                            |     function)](api/languag        |
| udaq::observe_result::expectation | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatorpLENSt7complexIdEE), |
|                                   |     [\[1\]](api/languages/c       |
| function)](api/languages/cpp_api. | pp_api.html#_CPPv4N5cudaq15scalar |
| html#_CPPv4N5cudaq14observe_resul | _operatorpLERK15scalar_operator), |
| t11expectationERK12spin_op_term), |     [\[2                          |
|     [\[1\]](api/la                | \]](api/languages/cpp_api.html#_C |
| nguages/cpp_api.html#_CPPv4N5cuda | PPv4N5cudaq15scalar_operatorpLEd) |
| q14observe_result11expectationEv) | -   [                             |
| -   [cuda                         | cudaq::scalar_operator::operator- |
| q::observe_result::id_coefficient |     (C++                          |
|     (C++                          |     function                      |
|     function)](api/langu          | )](api/languages/cpp_api.html#_CP |
| ages/cpp_api.html#_CPPv4N5cudaq14 | Pv4N5cudaq15scalar_operatormiENSt |
| observe_result14id_coefficientEv) | 7complexIdEERK15scalar_operator), |
| -   [cuda                         |     [\[1\                         |
| q::observe_result::observe_result | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormiENSt |
|                                   | 7complexIdEERR15scalar_operator), |
|   function)](api/languages/cpp_ap |     [\[2\]](api/languages/cp      |
| i.html#_CPPv4N5cudaq14observe_res | p_api.html#_CPPv4N5cudaq15scalar_ |
| ult14observe_resultEdRK7spin_op), | operatormiEdRK15scalar_operator), |
|     [\[1\]](a                     |     [\[3\]](api/languages/cp      |
| pi/languages/cpp_api.html#_CPPv4N | p_api.html#_CPPv4N5cudaq15scalar_ |
| 5cudaq14observe_result14observe_r | operatormiEdRR15scalar_operator), |
| esultEdRK7spin_op13sample_result) |     [\[4\]](api/languages         |
| -                                 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|  [cudaq::observe_result::operator | alar_operatormiENSt7complexIdEE), |
|     double (C++                   |     [\[5\]](api/languages/cpp     |
|     functio                       | _api.html#_CPPv4NKR5cudaq15scalar |
| n)](api/languages/cpp_api.html#_C | _operatormiERK15scalar_operator), |
| PPv4N5cudaq14observe_resultcvdEv) |     [\[6\]]                       |
| -                                 | (api/languages/cpp_api.html#_CPPv |
|  [cudaq::observe_result::raw_data | 4NKR5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[7\]]                       |
|     function)](ap                 | (api/languages/cpp_api.html#_CPPv |
| i/languages/cpp_api.html#_CPPv4N5 | 4NKR5cudaq15scalar_operatormiEv), |
| cudaq14observe_result8raw_dataEv) |     [\[8\]](api/language          |
| -   [cudaq::operator_handler (C++ | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     cl                            | alar_operatormiENSt7complexIdEE), |
| ass)](api/languages/cpp_api.html# |     [\[9\]](api/languages/cp      |
| _CPPv4N5cudaq16operator_handlerE) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq::optimizable_function  | _operatormiERK15scalar_operator), |
|     (C++                          |     [\[10\]                       |
|     class)                        | ](api/languages/cpp_api.html#_CPP |
| ](api/languages/cpp_api.html#_CPP | v4NO5cudaq15scalar_operatormiEd), |
| v4N5cudaq20optimizable_functionE) |     [\[11\                        |
| -   [cudaq::optimization_result   | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatormiEv) |
|     type                          | -   [c                            |
| )](api/languages/cpp_api.html#_CP | udaq::scalar_operator::operator-= |
| Pv4N5cudaq19optimization_resultE) |     (C++                          |
| -   [cudaq::optimizer (C++        |     function)](api/languag        |
|     class)](api/languages/cpp_a   | es/cpp_api.html#_CPPv4N5cudaq15sc |
| pi.html#_CPPv4N5cudaq9optimizerE) | alar_operatormIENSt7complexIdEE), |
| -   [cudaq::optimizer::optimize   |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|                                   | _operatormIERK15scalar_operator), |
|  function)](api/languages/cpp_api |     [\[2                          |
| .html#_CPPv4N5cudaq9optimizer8opt | \]](api/languages/cpp_api.html#_C |
| imizeEKiRR20optimizable_function) | PPv4N5cudaq15scalar_operatormIEd) |
| -   [cu                           | -   [                             |
| daq::optimizer::requiresGradients | cudaq::scalar_operator::operator/ |
|     (C++                          |     (C++                          |
|     function)](api/la             |     function                      |
| nguages/cpp_api.html#_CPPv4N5cuda | )](api/languages/cpp_api.html#_CP |
| q9optimizer17requiresGradientsEv) | Pv4N5cudaq15scalar_operatordvENSt |
| -   [cudaq::orca (C++             | 7complexIdEERK15scalar_operator), |
|     type)](api/languages/         |     [\[1\                         |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::orca::sample (C++     | Pv4N5cudaq15scalar_operatordvENSt |
|     function)](api/languages/c    | 7complexIdEERR15scalar_operator), |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     [\[2\]](api/languages/cp      |
| mpleERNSt6vectorINSt6size_tEEERNS | p_api.html#_CPPv4N5cudaq15scalar_ |
| t6vectorINSt6size_tEEERNSt6vector | operatordvEdRK15scalar_operator), |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     [\[3\]](api/languages/cp      |
|     [\[1\]]                       | p_api.html#_CPPv4N5cudaq15scalar_ |
| (api/languages/cpp_api.html#_CPPv | operatordvEdRR15scalar_operator), |
| 4N5cudaq4orca6sampleERNSt6vectorI |     [\[4\]](api/languages         |
| NSt6size_tEEERNSt6vectorINSt6size | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | alar_operatordvENSt7complexIdEE), |
| -   [cudaq::orca::sample_async    |     [\[5\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4NKR5cudaq15scalar |
|                                   | _operatordvERK15scalar_operator), |
| function)](api/languages/cpp_api. |     [\[6\]]                       |
| html#_CPPv4N5cudaq4orca12sample_a | (api/languages/cpp_api.html#_CPPv |
| syncERNSt6vectorINSt6size_tEEERNS | 4NKR5cudaq15scalar_operatordvEd), |
| t6vectorINSt6size_tEEERNSt6vector |     [\[7\]](api/language          |
| IdEERNSt6vectorIdEEiNSt6size_tE), | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     [\[1\]](api/la                | alar_operatordvENSt7complexIdEE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[8\]](api/languages/cp      |
| q4orca12sample_asyncERNSt6vectorI | p_api.html#_CPPv4NO5cudaq15scalar |
| NSt6size_tEEERNSt6vectorINSt6size | _operatordvERK15scalar_operator), |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     [\[9\                         |
| -   [cudaq::OrcaRemoteRESTQPU     | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatordvEd) |
|     cla                           | -   [c                            |
| ss)](api/languages/cpp_api.html#_ | udaq::scalar_operator::operator/= |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) |     (C++                          |
| -   [cudaq::pauli1 (C++           |     function)](api/languag        |
|     class)](api/languages/cp      | es/cpp_api.html#_CPPv4N5cudaq15sc |
| p_api.html#_CPPv4N5cudaq6pauli1E) | alar_operatordVENSt7complexIdEE), |
| -                                 |     [\[1\]](api/languages/c       |
|    [cudaq::pauli1::num_parameters | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatordVERK15scalar_operator), |
|     member)]                      |     [\[2                          |
| (api/languages/cpp_api.html#_CPPv | \]](api/languages/cpp_api.html#_C |
| 4N5cudaq6pauli114num_parametersE) | PPv4N5cudaq15scalar_operatordVEd) |
| -   [cudaq::pauli1::num_targets   | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator= |
|     membe                         |     (C++                          |
| r)](api/languages/cpp_api.html#_C |     function)](api/languages/c    |
| PPv4N5cudaq6pauli111num_targetsE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::pauli1::pauli1 (C++   | _operatoraSERK15scalar_operator), |
|     function)](api/languages/cpp_ |     [\[1\]](api/languages/        |
| api.html#_CPPv4N5cudaq6pauli16pau | cpp_api.html#_CPPv4N5cudaq15scala |
| li1ERKNSt6vectorIN5cudaq4realEEE) | r_operatoraSERR15scalar_operator) |
| -   [cudaq::pauli2 (C++           | -   [c                            |
|     class)](api/languages/cp      | udaq::scalar_operator::operator== |
| p_api.html#_CPPv4N5cudaq6pauli2E) |     (C++                          |
| -                                 |     function)](api/languages/c    |
|    [cudaq::pauli2::num_parameters | pp_api.html#_CPPv4NK5cudaq15scala |
|     (C++                          | r_operatoreqERK15scalar_operator) |
|     member)]                      | -   [cudaq:                       |
| (api/languages/cpp_api.html#_CPPv | :scalar_operator::scalar_operator |
| 4N5cudaq6pauli214num_parametersE) |     (C++                          |
| -   [cudaq::pauli2::num_targets   |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|     membe                         | #_CPPv4N5cudaq15scalar_operator15 |
| r)](api/languages/cpp_api.html#_C | scalar_operatorENSt7complexIdEE), |
| PPv4N5cudaq6pauli211num_targetsE) |     [\[1\]](api/langu             |
| -   [cudaq::pauli2::pauli2 (C++   | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     function)](api/languages/cpp_ | scalar_operator15scalar_operatorE |
| api.html#_CPPv4N5cudaq6pauli26pau | RK15scalar_callbackRRNSt13unorder |
| li2ERKNSt6vectorIN5cudaq4realEEE) | ed_mapINSt6stringENSt6stringEEE), |
| -   [cudaq::phase_damping (C++    |     [\[2\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
|  class)](api/languages/cpp_api.ht | Pv4N5cudaq15scalar_operator15scal |
| ml#_CPPv4N5cudaq13phase_dampingE) | ar_operatorERK15scalar_operator), |
| -   [cud                          |     [\[3\]](api/langu             |
| aq::phase_damping::num_parameters | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     (C++                          | scalar_operator15scalar_operatorE |
|     member)](api/lan              | RR15scalar_callbackRRNSt13unorder |
| guages/cpp_api.html#_CPPv4N5cudaq | ed_mapINSt6stringENSt6stringEEE), |
| 13phase_damping14num_parametersE) |     [\[4\                         |
| -   [                             | ]](api/languages/cpp_api.html#_CP |
| cudaq::phase_damping::num_targets | Pv4N5cudaq15scalar_operator15scal |
|     (C++                          | ar_operatorERR15scalar_operator), |
|     member)](api/                 |     [\[5\]](api/language          |
| languages/cpp_api.html#_CPPv4N5cu | s/cpp_api.html#_CPPv4N5cudaq15sca |
| daq13phase_damping11num_targetsE) | lar_operator15scalar_operatorEd), |
| -   [cudaq::phase_flip_channel    |     [\[6\]](api/languag           |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     clas                          | alar_operator15scalar_operatorEv) |
| s)](api/languages/cpp_api.html#_C | -   [                             |
| PPv4N5cudaq18phase_flip_channelE) | cudaq::scalar_operator::to_matrix |
| -   [cudaq::p                     |     (C++                          |
| hase_flip_channel::num_parameters |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     member)](api/language         | i.html#_CPPv4NK5cudaq15scalar_ope |
| s/cpp_api.html#_CPPv4N5cudaq18pha | rator9to_matrixERKNSt13unordered_ |
| se_flip_channel14num_parametersE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq                        | -   [                             |
| ::phase_flip_channel::num_targets | cudaq::scalar_operator::to_string |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api/l              |
| ages/cpp_api.html#_CPPv4N5cudaq18 | anguages/cpp_api.html#_CPPv4NK5cu |
| phase_flip_channel11num_targetsE) | daq15scalar_operator9to_stringEv) |
| -   [cudaq::product_op (C++       | -   [cudaq::s                     |
|                                   | calar_operator::\~scalar_operator |
|  class)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4I0EN5cudaq10product_opE) |     functio                       |
| -   [cudaq::product_op::begin     | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq15scalar_operatorD0Ev) |
|     functio                       | -   [cudaq::set_noise (C++        |
| n)](api/languages/cpp_api.html#_C |     function)](api/langu          |
| PPv4NK5cudaq10product_op5beginEv) | ages/cpp_api.html#_CPPv4N5cudaq9s |
| -                                 | et_noiseERKN5cudaq11noise_modelE) |
|  [cudaq::product_op::canonicalize | -   [cudaq::set_random_seed (C++  |
|     (C++                          |     function)](api/               |
|     func                          | languages/cpp_api.html#_CPPv4N5cu |
| tion)](api/languages/cpp_api.html | daq15set_random_seedENSt6size_tE) |
| #_CPPv4N5cudaq10product_op12canon | -   [cudaq::simulation_precision  |
| icalizeERKNSt3setINSt6size_tEEE), |     (C++                          |
|     [\[1\]](api                   |     enum)                         |
| /languages/cpp_api.html#_CPPv4N5c | ](api/languages/cpp_api.html#_CPP |
| udaq10product_op12canonicalizeEv) | v4N5cudaq20simulation_precisionE) |
| -   [                             | -   [                             |
| cudaq::product_op::const_iterator | cudaq::simulation_precision::fp32 |
|     (C++                          |     (C++                          |
|     struct)](api/                 |     enumerator)](api              |
| languages/cpp_api.html#_CPPv4N5cu | /languages/cpp_api.html#_CPPv4N5c |
| daq10product_op14const_iteratorE) | udaq20simulation_precision4fp32E) |
| -   [cudaq::product_o             | -   [                             |
| p::const_iterator::const_iterator | cudaq::simulation_precision::fp64 |
|     (C++                          |     (C++                          |
|     fu                            |     enumerator)](api              |
| nction)](api/languages/cpp_api.ht | /languages/cpp_api.html#_CPPv4N5c |
| ml#_CPPv4N5cudaq10product_op14con | udaq20simulation_precision4fp64E) |
| st_iterator14const_iteratorEPK10p | -   [cudaq::SimulationState (C++  |
| roduct_opI9HandlerTyENSt6size_tE) |     c                             |
| -   [cudaq::produ                 | lass)](api/languages/cpp_api.html |
| ct_op::const_iterator::operator!= | #_CPPv4N5cudaq15SimulationStateE) |
|     (C++                          | -   [c                            |
|     fun                           | udaq::SimulationState::HostBuffer |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4NK5cudaq10product_op14con |     struct)](api/l                |
| st_iteratorneERK14const_iterator) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::produ                 | aq15SimulationState10HostBufferE) |
| ct_op::const_iterator::operator\* | -   [                             |
|     (C++                          | cudaq::SimulationState::precision |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4NK5cudaq |     enum)](api                    |
| 10product_op14const_iteratormlEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::produ                 | udaq15SimulationState9precisionE) |
| ct_op::const_iterator::operator++ | -   [cudaq:                       |
|     (C++                          | :SimulationState::precision::fp32 |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     enumerator)](api/lang         |
| 0product_op14const_iteratorppEi), | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     [\[1\]](api/lan               | 5SimulationState9precision4fp32E) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq:                       |
| 10product_op14const_iteratorppEv) | :SimulationState::precision::fp64 |
| -   [cudaq::produc                |     (C++                          |
| t_op::const_iterator::operator\-- |     enumerator)](api/lang         |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     function)](api/lang           | 5SimulationState9precision4fp64E) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -                                 |
| 0product_op14const_iteratormmEi), |   [cudaq::SimulationState::Tensor |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     struct)](                     |
| 10product_op14const_iteratormmEv) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::produc                | N5cudaq15SimulationState6TensorE) |
| t_op::const_iterator::operator-\> | -   [cudaq::spin_handler (C++     |
|     (C++                          |                                   |
|     function)](api/lan            |   class)](api/languages/cpp_api.h |
| guages/cpp_api.html#_CPPv4N5cudaq | tml#_CPPv4N5cudaq12spin_handlerE) |
| 10product_op14const_iteratorptEv) | -   [cudaq:                       |
| -   [cudaq::produ                 | :spin_handler::to_diagonal_matrix |
| ct_op::const_iterator::operator== |     (C++                          |
|     (C++                          |     function)](api/la             |
|     fun                           | nguages/cpp_api.html#_CPPv4NK5cud |
| ction)](api/languages/cpp_api.htm | aq12spin_handler18to_diagonal_mat |
| l#_CPPv4NK5cudaq10product_op14con | rixERNSt13unordered_mapINSt6size_ |
| st_iteratoreqERK14const_iterator) | tENSt7int64_tEEERKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -                                 |
|                                   |   [cudaq::spin_handler::to_matrix |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq12spin_handler9to_matri |
|                                   | xERKNSt6stringENSt7complexIdEEb), |
|                                   |     [\[1                          |
|                                   | \]](api/languages/cpp_api.html#_C |
|                                   | PPv4NK5cudaq12spin_handler9to_mat |
|                                   | rixERNSt13unordered_mapINSt6size_ |
|                                   | tENSt7int64_tEEERKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [cuda                         |
|                                   | q::spin_handler::to_sparse_matrix |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq12spin_handler16to_sparse_matr |
|                                   | ixERKNSt6stringENSt7complexIdEEb) |
|                                   | -                                 |
|                                   |   [cudaq::spin_handler::to_string |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq12spin_handler9to_stringEb) |
|                                   | -                                 |
|                                   |   [cudaq::spin_handler::unique_id |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq12spin_handler9unique_idEv) |
|                                   | -   [cudaq::spin_op (C++          |
|                                   |     type)](api/languages/cpp      |
|                                   | _api.html#_CPPv4N5cudaq7spin_opE) |
|                                   | -   [cudaq::spin_op_term (C++     |
|                                   |                                   |
|                                   |    type)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq12spin_op_termE) |
|                                   | -   [cudaq::state (C++            |
|                                   |     class)](api/languages/c       |
|                                   | pp_api.html#_CPPv4N5cudaq5stateE) |
|                                   | -   [cudaq::state::amplitude (C++ |
|                                   |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq5 |
|                                   | state9amplitudeERKNSt6vectorIiEE) |
|                                   | -   [cudaq::state::amplitudes     |
|                                   |     (C++                          |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq5state10amplitud |
|                                   | esERKNSt6vectorINSt6vectorIiEEEE) |
|                                   | -   [cudaq::state::dump (C++      |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq5state4dumpERNSt7ostreamE), |
|                                   |                                   |
|                                   |    [\[1\]](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq5state4dumpEv) |
|                                   | -   [cudaq::state::from_data (C++ |
|                                   |     function)](api/la             |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q5state9from_dataERK10state_data) |
|                                   | -   [cudaq::state::get_num_qubits |
|                                   |     (C++                          |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq5state14get_num_qubitsEv) |
|                                   | -                                 |
|                                   |    [cudaq::state::get_num_tensors |
|                                   |     (C++                          |
|                                   |     function)](a                  |
|                                   | pi/languages/cpp_api.html#_CPPv4N |
|                                   | K5cudaq5state15get_num_tensorsEv) |
|                                   | -   [cudaq::state::get_precision  |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NK5cudaq5state13get_precisionEv) |
|                                   | -   [cudaq::state::get_tensor     |
|                                   |     (C++                          |
|                                   |     function)](api/la             |
|                                   | nguages/cpp_api.html#_CPPv4NK5cud |
|                                   | aq5state10get_tensorENSt6size_tE) |
|                                   | -   [cudaq::state::get_tensors    |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq5state11get_tensorsEv) |
|                                   | -   [cudaq::state::is_on_gpu (C++ |
|                                   |     funct                         |
|                                   | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4NK5cudaq5state9is_on_gpuEv) |
|                                   | -   [cudaq::state::operator()     |
|                                   |     (C++                          |
|                                   |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 5stateclENSt6size_tENSt6size_tE), |
|                                   |     [\[1\]](                      |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq5stateclERKNSt16initializ |
|                                   | er_listINSt6size_tEEENSt6size_tE) |
|                                   | -   [cudaq::state::operator= (C++ |
|                                   |     fun                           |
|                                   | ction)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq5stateaSERR5state) |
|                                   | -   [cudaq::state::operator\[\]   |
|                                   |     (C++                          |
|                                   |     functio                       |
|                                   | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4NK5cudaq5stateixENSt6size_tE) |
|                                   | -   [cudaq::state::overlap (C++   |
|                                   |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq5state7overlapERK5state) |
|                                   | -   [cudaq::state::state (C++     |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 5state5stateEP15SimulationState), |
|                                   |     [\[1\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq5state5stateERK5state), |
|                                   |     [\[2\]](api/languages/cpp_    |
|                                   | api.html#_CPPv4N5cudaq5state5stat |
|                                   | eERKNSt6vectorINSt7complexIdEEEE) |
|                                   | -   [cudaq::state::to_host (C++   |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | I0ENK5cudaq5state7to_hostEvPNSt7c |
|                                   | omplexI10ScalarTypeEENSt6size_tE) |
|                                   | -   [cudaq::state::toHostBuffer   |
|                                   |     (C++                          |
|                                   |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 5state12toHostBufferENSt6size_tE) |
|                                   | -   [cudaq::state::\~state (C++   |
|                                   |     function)](api/languages/cpp_ |
|                                   | api.html#_CPPv4N5cudaq5stateD0Ev) |
|                                   | -   [cudaq::state_data (C++       |
|                                   |     type)](api/languages/cpp_api  |
|                                   | .html#_CPPv4N5cudaq10state_dataE) |
|                                   | -   [cudaq::sum_op (C++           |
|                                   |     class)](api/languages/cpp_a   |
|                                   | pi.html#_CPPv4I0EN5cudaq6sum_opE) |
|                                   | -   [cudaq::sum_op::begin (C++    |
|                                   |     fu                            |
|                                   | nction)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4NK5cudaq6sum_op5beginEv) |
|                                   | -   [cudaq::sum_op::canonicalize  |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |  function)](api/languages/cpp_api |
|                                   | .html#_CPPv4N5cudaq6sum_op12canon |
|                                   | icalizeERKNSt3setINSt6size_tEEE), |
|                                   |     [\[1\]                        |
|                                   | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq6sum_op12canonicalizeEv) |
|                                   | -                                 |
|                                   |    [cudaq::sum_op::const_iterator |
|                                   |     (C++                          |
|                                   |     struct)]                      |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4N5cudaq6sum_op14const_iteratorE) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator!= |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq6sum_op14con |
|                                   | st_iteratorneERK14const_iterator) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator\* |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratormlEv) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator++ |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratorppEv) |
|                                   | -   [cudaq::su                    |
|                                   | m_op::const_iterator::operator-\> |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratorptEv) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator== |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq6sum_op14con |
|                                   | st_iteratoreqERK14const_iterator) |
|                                   | -   [cudaq::sum_op::degrees (C++  |
|                                   |     func                          |
|                                   | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4NK5cudaq6sum_op7degreesEv) |
|                                   | -                                 |
|                                   |  [cudaq::sum_op::distribute_terms |
|                                   |     (C++                          |
|                                   |     function)](api/languages      |
|                                   | /cpp_api.html#_CPPv4NK5cudaq6sum_ |
|                                   | op16distribute_termsENSt6size_tE) |
|                                   | -   [cudaq::sum_op::dump (C++     |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4NK5cudaq6sum_op4dumpEv) |
|                                   | -   [cudaq::sum_op::empty (C++    |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq6sum_op5emptyEv) |
|                                   | -   [cudaq::sum_op::end (C++      |
|                                   |                                   |
|                                   | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq6sum_op3endEv) |
|                                   | -   [cudaq::sum_op::identity (C++ |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq6sum_op8identityENSt6size_tE), |
|                                   |     [                             |
|                                   | \[1\]](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq6sum_op8identityEv) |
|                                   | -   [cudaq::sum_op::num_terms     |
|                                   |     (C++                          |
|                                   |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4NK5cudaq6sum_op9num_termsEv) |
|                                   | -   [cudaq::sum_op::operator\*    |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opmlE6sum_opI1TER |
|                                   | K15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[1\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opmlE6sum_opI1TER |
|                                   | K15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[2\]](api/languages         |
|                                   | /cpp_api.html#_CPPv4NK5cudaq6sum_ |
|                                   | opmlERK10product_opI9HandlerTyE), |
|                                   |     [\[3\]](api/lang              |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 6sum_opmlERK6sum_opI9HandlerTyE), |
|                                   |     [\[4\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opmlERK15scalar_operator), |
|                                   |     [\[5\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4NO5cu |
|                                   | daq6sum_opmlERK15scalar_operator) |
|                                   | -   [cudaq::sum_op::operator\*=   |
|                                   |     (C++                          |
|                                   |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opmLERK10product_opI9HandlerTyE), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_opmLERK15scalar_operator), |
|                                   |     [\[2\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_opmLERK6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator+     |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERK15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | K15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[2\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[3\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | K15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[4\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERR15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[5\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | R15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[6\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERR15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[7\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | R15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[8\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opplERK10product_opI9HandlerTyE), |
|                                   |     [\[9\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opplERK15scalar_operator), |
|                                   |     [\[10\]](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opplERK6sum_opI9HandlerTyE), |
|                                   |     [\[11\]](api/languages/       |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|                                   |     [\[12\]](api/lan              |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opplERR15scalar_operator), |
|                                   |     [\[13\]](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opplERR6sum_opI9HandlerTyE), |
|                                   |                                   |
|                                   |   [\[14\]](api/languages/cpp_api. |
|                                   | html#_CPPv4NKR5cudaq6sum_opplEv), |
|                                   |     [\[15\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opplERK10product_opI9HandlerTyE), |
|                                   |     [\[16\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opplERK15scalar_operator), |
|                                   |     [\[17\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opplERK6sum_opI9HandlerTyE), |
|                                   |     [\[18\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|                                   |     [\[19\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opplERR15scalar_operator), |
|                                   |     [\[20\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opplERR6sum_opI9HandlerTyE), |
|                                   |     [\[21\]](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NO5cudaq6sum_opplEv) |
|                                   | -   [cudaq::sum_op::operator+=    |
|                                   |     (C++                          |
|                                   |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | oppLERK10product_opI9HandlerTyE), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_oppLERK15scalar_operator), |
|                                   |     [\[2\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 6sum_oppLERK6sum_opI9HandlerTyE), |
|                                   |     [\[3\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | oppLERR10product_opI9HandlerTyE), |
|                                   |     [\[4\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_oppLERR15scalar_operator), |
|                                   |     [\[5\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_oppLERR6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator-     |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERK15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[2\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERR15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[3\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opmiE6sum_opI1TER |
|                                   | R15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[4\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERR15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[5\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opmiERK10product_opI9HandlerTyE), |
|                                   |     [\[6\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opmiERK15scalar_operator), |
|                                   |     [\[7\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opmiERK6sum_opI9HandlerTyE), |
|                                   |     [\[8\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opmiERR10product_opI9HandlerTyE), |
|                                   |     [\[9\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opmiERR15scalar_operator), |
|                                   |     [\[10\]](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opmiERR6sum_opI9HandlerTyE), |
|                                   |                                   |
|                                   |   [\[11\]](api/languages/cpp_api. |
|                                   | html#_CPPv4NKR5cudaq6sum_opmiEv), |
|                                   |     [\[12\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opmiERK10product_opI9HandlerTyE), |
|                                   |     [\[13\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opmiERK15scalar_operator), |
|                                   |     [\[14\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opmiERK6sum_opI9HandlerTyE), |
|                                   |     [\[15\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opmiERR10product_opI9HandlerTyE), |
|                                   |     [\[16\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opmiERR15scalar_operator), |
|                                   |     [\[17\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opmiERR6sum_opI9HandlerTyE), |
|                                   |     [\[18\]](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NO5cudaq6sum_opmiEv) |
|                                   | -   [cudaq::sum_op::operator-=    |
|                                   |     (C++                          |
|                                   |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opmIERK10product_opI9HandlerTyE), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_opmIERK15scalar_operator), |
|                                   |     [\[2\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 6sum_opmIERK6sum_opI9HandlerTyE), |
|                                   |     [\[3\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opmIERR10product_opI9HandlerTyE), |
|                                   |     [\[4\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_opmIERR15scalar_operator), |
|                                   |     [\[5\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_opmIERR6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator/     |
|                                   |     (C++                          |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opdvERK15scalar_operator), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4NO5cu |
|                                   | daq6sum_opdvERK15scalar_operator) |
|                                   | -   [cudaq::sum_op::operator/=    |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq6sum_opdVERK15scalar_operator) |
|                                   | -   [cudaq::sum_op::operator=     |
|                                   |     (C++                          |
|                                   |     functio                       |
|                                   | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4I0_NSt11enable_if_tIXaantNSt7 |
|                                   | is_sameI1T9HandlerTyE5valueENSt16 |
|                                   | is_constructibleI9HandlerTy1TE5va |
|                                   | lueEEbEEEN5cudaq6sum_opaSER6sum_o |
|                                   | pI9HandlerTyERK10product_opI1TE), |
|                                   |                                   |
|                                   |  [\[1\]](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4I0_NSt11enable_if_tIXaan |
|                                   | tNSt7is_sameI1T9HandlerTyE5valueE |
|                                   | NSt16is_constructibleI9HandlerTy1 |
|                                   | TE5valueEEbEEEN5cudaq6sum_opaSER6 |
|                                   | sum_opI9HandlerTyERK6sum_opI1TE), |
|                                   |     [\[2\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opaSERK10product_opI9HandlerTyE), |
|                                   |     [\[3\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 6sum_opaSERK6sum_opI9HandlerTyE), |
|                                   |     [\[4\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opaSERR10product_opI9HandlerTyE), |
|                                   |     [\[5\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_opaSERR6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator==    |
|                                   |     (C++                          |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4NK5cuda |
|                                   | q6sum_opeqERK6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator\[\]  |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq6sum_opixENSt6size_tE) |
|                                   | -   [cudaq::sum_op::sum_op (C++   |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | I0_NSt11enable_if_tIXaaNSt7is_sam |
|                                   | eI9HandlerTy14matrix_handlerE5val |
|                                   | ueEaantNSt7is_sameI1T9HandlerTyE5 |
|                                   | valueENSt16is_constructibleI9Hand |
|                                   | lerTy1TE5valueEEbEEEN5cudaq6sum_o |
|                                   | p6sum_opERK6sum_opI1TERKN14matrix |
|                                   | _handler20commutation_behaviorE), |
|                                   |     [\[1\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4I0_NSt11e |
|                                   | nable_if_tIXaantNSt7is_sameI1T9Ha |
|                                   | ndlerTyE5valueENSt16is_constructi |
|                                   | bleI9HandlerTy1TE5valueEEbEEEN5cu |
|                                   | daq6sum_op6sum_opERK6sum_opI1TE), |
|                                   |     [\[2\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4IDp_NSt |
|                                   | 11enable_if_tIXaaNSt11conjunction |
|                                   | IDpNSt7is_sameI10product_opI9Hand |
|                                   | lerTyE4ArgsEEE5valueEsZ4ArgsEbEEE |
|                                   | N5cudaq6sum_op6sum_opEDpRR4Args), |
|                                   |     [\[3\]](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq6sum_op6su |
|                                   | m_opERK10product_opI9HandlerTyE), |
|                                   |     [\[4\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | op6sum_opERK6sum_opI9HandlerTyE), |
|                                   |     [\[5\]](api/languag           |
|                                   | es/cpp_api.html#_CPPv4N5cudaq6sum |
|                                   | _op6sum_opERR6sum_opI9HandlerTyE) |
|                                   | -   [                             |
|                                   | cudaq::sum_op::to_diagonal_matrix |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NK5cudaq6sum_op18to_diagonal_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -   [cudaq::sum_op::to_matrix     |
|                                   |     (C++                          |
|                                   |                                   |
|                                   | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq6sum_op9to_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -                                 |
|                                   |  [cudaq::sum_op::to_sparse_matrix |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq6sum_op16to_sparse_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -   [cudaq::sum_op::to_string     |
|                                   |     (C++                          |
|                                   |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4NK5cudaq6sum_op9to_stringEv) |
|                                   | -   [cudaq::sum_op::trim (C++     |
|                                   |     function)](api/l              |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_op4trimEdRKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [cudaq::sum_op::\~sum_op (C++ |
|                                   |                                   |
|                                   |    function)](api/languages/cpp_a |
|                                   | pi.html#_CPPv4N5cudaq6sum_opD0Ev) |
|                                   | -   [cudaq::tensor (C++           |
|                                   |     type)](api/languages/cp       |
|                                   | p_api.html#_CPPv4N5cudaq6tensorE) |
|                                   | -   [cudaq::TensorStateData (C++  |
|                                   |                                   |
|                                   | type)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq15TensorStateDataE) |
|                                   | -   [cudaq::Trace (C++            |
|                                   |     class)](api/languages/c       |
|                                   | pp_api.html#_CPPv4N5cudaq5TraceE) |
|                                   | -   [cudaq::unset_noise (C++      |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq11unset_noiseEv) |
|                                   | -   [cudaq::x_error (C++          |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7x_errorE) |
|                                   | -   [cudaq::y_error (C++          |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7y_errorE) |
|                                   | -                                 |
|                                   |   [cudaq::y_error::num_parameters |
|                                   |     (C++                          |
|                                   |     member)](                     |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq7y_error14num_parametersE) |
|                                   | -   [cudaq::y_error::num_targets  |
|                                   |     (C++                          |
|                                   |     member                        |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq7y_error11num_targetsE) |
|                                   | -   [cudaq::z_error (C++          |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7z_errorE) |
+-----------------------------------+-----------------------------------+

## D {#D}

+-----------------------------------+-----------------------------------+
| -   [define() (cudaq.operators    | -   [description (cudaq.Target    |
|     method)](api/languages/python |                                   |
| _api.html#cudaq.operators.define) | property)](api/languages/python_a |
|     -   [(cuda                    | pi.html#cudaq.Target.description) |
| q.operators.MatrixOperatorElement | -   [deserialize()                |
|         class                     |     (cudaq.SampleResult           |
|         method)](api/langu        |     meth                          |
| ages/python_api.html#cudaq.operat | od)](api/languages/python_api.htm |
| ors.MatrixOperatorElement.define) | l#cudaq.SampleResult.deserialize) |
|     -   [(in module               | -   [displace() (in module        |
|         cudaq.operators.cus       |     cudaq.operators.custo         |
| tom)](api/languages/python_api.ht | m)](api/languages/python_api.html |
| ml#cudaq.operators.custom.define) | #cudaq.operators.custom.displace) |
| -   [degrees                      | -   [distribute_terms()           |
|     (cu                           |     (cu                           |
| daq.operators.boson.BosonOperator | daq.operators.boson.BosonOperator |
|     property)](api/lang           |     method)](api/languages/pyt    |
| uages/python_api.html#cudaq.opera | hon_api.html#cudaq.operators.boso |
| tors.boson.BosonOperator.degrees) | n.BosonOperator.distribute_terms) |
|     -   [(cudaq.ope               |     -   [(cudaq.                  |
| rators.boson.BosonOperatorElement | operators.fermion.FermionOperator |
|                                   |                                   |
|        property)](api/languages/p |    method)](api/languages/python_ |
| ython_api.html#cudaq.operators.bo | api.html#cudaq.operators.fermion. |
| son.BosonOperatorElement.degrees) | FermionOperator.distribute_terms) |
|     -   [(cudaq.                  |     -                             |
| operators.boson.BosonOperatorTerm |  [(cudaq.operators.MatrixOperator |
|         property)](api/language   |         method)](api/language     |
| s/python_api.html#cudaq.operators | s/python_api.html#cudaq.operators |
| .boson.BosonOperatorTerm.degrees) | .MatrixOperator.distribute_terms) |
|     -   [(cudaq.                  |     -   [(                        |
| operators.fermion.FermionOperator | cudaq.operators.spin.SpinOperator |
|         property)](api/language   |         method)](api/languages/p  |
| s/python_api.html#cudaq.operators | ython_api.html#cudaq.operators.sp |
| .fermion.FermionOperator.degrees) | in.SpinOperator.distribute_terms) |
|     -   [(cudaq.operato           |     -   [(cuda                    |
| rs.fermion.FermionOperatorElement | q.operators.spin.SpinOperatorTerm |
|                                   |                                   |
|    property)](api/languages/pytho |      method)](api/languages/pytho |
| n_api.html#cudaq.operators.fermio | n_api.html#cudaq.operators.spin.S |
| n.FermionOperatorElement.degrees) | pinOperatorTerm.distribute_terms) |
|     -   [(cudaq.oper              | -   [draw() (in module            |
| ators.fermion.FermionOperatorTerm |     cudaq)](api/lang              |
|                                   | uages/python_api.html#cudaq.draw) |
|       property)](api/languages/py | -   [dump() (cudaq.ComplexMatrix  |
| thon_api.html#cudaq.operators.fer |                                   |
| mion.FermionOperatorTerm.degrees) |   method)](api/languages/python_a |
|     -                             | pi.html#cudaq.ComplexMatrix.dump) |
|  [(cudaq.operators.MatrixOperator |     -   [(cudaq.ObserveResult     |
|         property)](api            |                                   |
| /languages/python_api.html#cudaq. |   method)](api/languages/python_a |
| operators.MatrixOperator.degrees) | pi.html#cudaq.ObserveResult.dump) |
|     -   [(cuda                    |     -   [(cu                      |
| q.operators.MatrixOperatorElement | daq.operators.boson.BosonOperator |
|         property)](api/langua     |         method)](api/l            |
| ges/python_api.html#cudaq.operato | anguages/python_api.html#cudaq.op |
| rs.MatrixOperatorElement.degrees) | erators.boson.BosonOperator.dump) |
|     -   [(c                       |     -   [(cudaq.                  |
| udaq.operators.MatrixOperatorTerm | operators.boson.BosonOperatorTerm |
|         property)](api/lan        |         method)](api/langu        |
| guages/python_api.html#cudaq.oper | ages/python_api.html#cudaq.operat |
| ators.MatrixOperatorTerm.degrees) | ors.boson.BosonOperatorTerm.dump) |
|     -   [(                        |     -   [(cudaq.                  |
| cudaq.operators.spin.SpinOperator | operators.fermion.FermionOperator |
|         property)](api/la         |         method)](api/langu        |
| nguages/python_api.html#cudaq.ope | ages/python_api.html#cudaq.operat |
| rators.spin.SpinOperator.degrees) | ors.fermion.FermionOperator.dump) |
|     -   [(cudaq.o                 |     -   [(cudaq.oper              |
| perators.spin.SpinOperatorElement | ators.fermion.FermionOperatorTerm |
|         property)](api/languages  |         method)](api/languages    |
| /python_api.html#cudaq.operators. | /python_api.html#cudaq.operators. |
| spin.SpinOperatorElement.degrees) | fermion.FermionOperatorTerm.dump) |
|     -   [(cuda                    |     -                             |
| q.operators.spin.SpinOperatorTerm |  [(cudaq.operators.MatrixOperator |
|         property)](api/langua     |         method)](                 |
| ges/python_api.html#cudaq.operato | api/languages/python_api.html#cud |
| rs.spin.SpinOperatorTerm.degrees) | aq.operators.MatrixOperator.dump) |
| -                                 |     -   [(c                       |
|  [delete_cache_execution_engine() | udaq.operators.MatrixOperatorTerm |
|     (cudaq.PyKernelDecorator      |         method)](api/             |
|     method)](api/languages/pyth   | languages/python_api.html#cudaq.o |
| on_api.html#cudaq.PyKernelDecorat | perators.MatrixOperatorTerm.dump) |
| or.delete_cache_execution_engine) |     -   [(                        |
| -   [Depolarization1 (class in    | cudaq.operators.spin.SpinOperator |
|     cudaq)](api/languages/pytho   |         method)](api              |
| n_api.html#cudaq.Depolarization1) | /languages/python_api.html#cudaq. |
| -   [Depolarization2 (class in    | operators.spin.SpinOperator.dump) |
|     cudaq)](api/languages/pytho   |     -   [(cuda                    |
| n_api.html#cudaq.Depolarization2) | q.operators.spin.SpinOperatorTerm |
| -   [DepolarizationChannel (class |         method)](api/lan          |
|     in                            | guages/python_api.html#cudaq.oper |
|                                   | ators.spin.SpinOperatorTerm.dump) |
| cudaq)](api/languages/python_api. |     -   [(cudaq.Resources         |
| html#cudaq.DepolarizationChannel) |                                   |
| -   [depth (cudaq.Resources       |       method)](api/languages/pyth |
|                                   | on_api.html#cudaq.Resources.dump) |
|    property)](api/languages/pytho |     -   [(cudaq.SampleResult      |
| n_api.html#cudaq.Resources.depth) |                                   |
| -   [depth_2q (cudaq.Resources    |    method)](api/languages/python_ |
|                                   | api.html#cudaq.SampleResult.dump) |
| property)](api/languages/python_a |     -   [(cudaq.State             |
| pi.html#cudaq.Resources.depth_2q) |         method)](api/languages/   |
|                                   | python_api.html#cudaq.State.dump) |
+-----------------------------------+-----------------------------------+

## E {#E}

+-----------------------------------+-----------------------------------+
| -   [ElementaryOperator (in       | -   [evaluate()                   |
|     module                        |                                   |
|     cudaq.operators)]             |   (cudaq.operators.ScalarOperator |
| (api/languages/python_api.html#cu |     method)](api/                 |
| daq.operators.ElementaryOperator) | languages/python_api.html#cudaq.o |
| -   [empty()                      | perators.ScalarOperator.evaluate) |
|     (cu                           | -   [evaluate_coefficient()       |
| daq.operators.boson.BosonOperator |     (cudaq.                       |
|     static                        | operators.boson.BosonOperatorTerm |
|     method)](api/la               |     m                             |
| nguages/python_api.html#cudaq.ope | ethod)](api/languages/python_api. |
| rators.boson.BosonOperator.empty) | html#cudaq.operators.boson.BosonO |
|     -   [(cudaq.                  | peratorTerm.evaluate_coefficient) |
| operators.fermion.FermionOperator |     -   [(cudaq.oper              |
|         static                    | ators.fermion.FermionOperatorTerm |
|         method)](api/langua       |         metho                     |
| ges/python_api.html#cudaq.operato | d)](api/languages/python_api.html |
| rs.fermion.FermionOperator.empty) | #cudaq.operators.fermion.FermionO |
|     -                             | peratorTerm.evaluate_coefficient) |
|  [(cudaq.operators.MatrixOperator |     -   [(c                       |
|         static                    | udaq.operators.MatrixOperatorTerm |
|         method)](a                |                                   |
| pi/languages/python_api.html#cuda |     method)](api/languages/python |
| q.operators.MatrixOperator.empty) | _api.html#cudaq.operators.MatrixO |
|     -   [(                        | peratorTerm.evaluate_coefficient) |
| cudaq.operators.spin.SpinOperator |     -   [(cuda                    |
|         static                    | q.operators.spin.SpinOperatorTerm |
|         method)](api/             |                                   |
| languages/python_api.html#cudaq.o |  method)](api/languages/python_ap |
| perators.spin.SpinOperator.empty) | i.html#cudaq.operators.spin.SpinO |
|     -   [(in module               | peratorTerm.evaluate_coefficient) |
|                                   | -   [evolve() (in module          |
|     cudaq.boson)](api/languages/p |     cudaq)](api/langua            |
| ython_api.html#cudaq.boson.empty) | ges/python_api.html#cudaq.evolve) |
|     -   [(in module               | -   [evolve_async() (in module    |
|                                   |     cudaq)](api/languages/py      |
| cudaq.fermion)](api/languages/pyt | thon_api.html#cudaq.evolve_async) |
| hon_api.html#cudaq.fermion.empty) | -   [EvolveResult (class in       |
|     -   [(in module               |     cudaq)](api/languages/py      |
|         cudaq.operators.cu        | thon_api.html#cudaq.EvolveResult) |
| stom)](api/languages/python_api.h | -   [ExhaustiveSamplingStrategy   |
| tml#cudaq.operators.custom.empty) |     (class in                     |
|     -   [(in module               |     cudaq.ptsbe)](api             |
|                                   | /languages/python_api.html#cudaq. |
|       cudaq.spin)](api/languages/ | ptsbe.ExhaustiveSamplingStrategy) |
| python_api.html#cudaq.spin.empty) | -   [expectation()                |
| -   [empty_op()                   |     (cudaq.ObserveResult          |
|     (                             |     metho                         |
| cudaq.operators.spin.SpinOperator | d)](api/languages/python_api.html |
|     static                        | #cudaq.ObserveResult.expectation) |
|     method)](api/lan              |     -   [(cudaq.SampleResult      |
| guages/python_api.html#cudaq.oper |         meth                      |
| ators.spin.SpinOperator.empty_op) | od)](api/languages/python_api.htm |
| -   [enable_return_to_log()       | l#cudaq.SampleResult.expectation) |
|     (cudaq.PyKernelDecorator      | -   [expectation_values()         |
|     method)](api/langu            |     (cudaq.EvolveResult           |
| ages/python_api.html#cudaq.PyKern |     method)](ap                   |
| elDecorator.enable_return_to_log) | i/languages/python_api.html#cudaq |
| -   [epsilon                      | .EvolveResult.expectation_values) |
|     (cudaq.optimizers.Adam        | -   [expectation_z()              |
|     prope                         |     (cudaq.SampleResult           |
| rty)](api/languages/python_api.ht |     method                        |
| ml#cudaq.optimizers.Adam.epsilon) | )](api/languages/python_api.html# |
| -   [estimate_resources() (in     | cudaq.SampleResult.expectation_z) |
|     module                        | -   [expected_dimensions          |
|                                   |     (cuda                         |
|    cudaq)](api/languages/python_a | q.operators.MatrixOperatorElement |
| pi.html#cudaq.estimate_resources) |                                   |
|                                   | property)](api/languages/python_a |
|                                   | pi.html#cudaq.operators.MatrixOpe |
|                                   | ratorElement.expected_dimensions) |
+-----------------------------------+-----------------------------------+

## F {#F}

+-----------------------------------+-----------------------------------+
| -   [f_tol (cudaq.optimizers.Adam | -   [from_json()                  |
|     pro                           |     (                             |
| perty)](api/languages/python_api. | cudaq.gradients.CentralDifference |
| html#cudaq.optimizers.Adam.f_tol) |     static                        |
|     -   [(cudaq.optimizers.SGD    |     method)](api/lang             |
|         pr                        | uages/python_api.html#cudaq.gradi |
| operty)](api/languages/python_api | ents.CentralDifference.from_json) |
| .html#cudaq.optimizers.SGD.f_tol) |     -   [(                        |
| -   [FermionOperator (class in    | cudaq.gradients.ForwardDifference |
|                                   |         static                    |
|    cudaq.operators.fermion)](api/ |         method)](api/lang         |
| languages/python_api.html#cudaq.o | uages/python_api.html#cudaq.gradi |
| perators.fermion.FermionOperator) | ents.ForwardDifference.from_json) |
| -   [FermionOperatorElement       |     -                             |
|     (class in                     |  [(cudaq.gradients.ParameterShift |
|     cuda                          |         static                    |
| q.operators.fermion)](api/languag |         method)](api/l            |
| es/python_api.html#cudaq.operator | anguages/python_api.html#cudaq.gr |
| s.fermion.FermionOperatorElement) | adients.ParameterShift.from_json) |
| -   [FermionOperatorTerm (class   |     -   [(                        |
|     in                            | cudaq.operators.spin.SpinOperator |
|     c                             |         static                    |
| udaq.operators.fermion)](api/lang |         method)](api/lang         |
| uages/python_api.html#cudaq.opera | uages/python_api.html#cudaq.opera |
| tors.fermion.FermionOperatorTerm) | tors.spin.SpinOperator.from_json) |
| -   [final_expectation_values()   |     -   [(cuda                    |
|     (cudaq.EvolveResult           | q.operators.spin.SpinOperatorTerm |
|     method)](api/lang             |         static                    |
| uages/python_api.html#cudaq.Evolv |         method)](api/language     |
| eResult.final_expectation_values) | s/python_api.html#cudaq.operators |
| -   [final_state()                | .spin.SpinOperatorTerm.from_json) |
|     (cudaq.EvolveResult           |     -   [(cudaq.optimizers.Adam   |
|     meth                          |         static                    |
| od)](api/languages/python_api.htm |         metho                     |
| l#cudaq.EvolveResult.final_state) | d)](api/languages/python_api.html |
| -   [finalize() (in module        | #cudaq.optimizers.Adam.from_json) |
|     cudaq.mpi)](api/languages/py  |     -   [(cudaq.optimizers.COBYLA |
| thon_api.html#cudaq.mpi.finalize) |         static                    |
| -   [for_each_pauli()             |         method)                   |
|     (                             | ](api/languages/python_api.html#c |
| cudaq.operators.spin.SpinOperator | udaq.optimizers.COBYLA.from_json) |
|     method)](api/languages        |     -   [                         |
| /python_api.html#cudaq.operators. | (cudaq.optimizers.GradientDescent |
| spin.SpinOperator.for_each_pauli) |         static                    |
|     -   [(cuda                    |         method)](api/lan          |
| q.operators.spin.SpinOperatorTerm | guages/python_api.html#cudaq.opti |
|                                   | mizers.GradientDescent.from_json) |
|        method)](api/languages/pyt |     -   [(cudaq.optimizers.LBFGS  |
| hon_api.html#cudaq.operators.spin |         static                    |
| .SpinOperatorTerm.for_each_pauli) |         method                    |
| -   [for_each_term()              | )](api/languages/python_api.html# |
|     (                             | cudaq.optimizers.LBFGS.from_json) |
| cudaq.operators.spin.SpinOperator |                                   |
|     method)](api/language         | -   [(cudaq.optimizers.NelderMead |
| s/python_api.html#cudaq.operators |         static                    |
| .spin.SpinOperator.for_each_term) |         method)](ap               |
| -   [ForwardDifference (class in  | i/languages/python_api.html#cudaq |
|     cudaq.gradients)              | .optimizers.NelderMead.from_json) |
| ](api/languages/python_api.html#c |     -   [(cudaq.optimizers.SGD    |
| udaq.gradients.ForwardDifference) |         static                    |
| -   [from_data() (cudaq.State     |         meth                      |
|     static                        | od)](api/languages/python_api.htm |
|     method)](api/languages/pytho  | l#cudaq.optimizers.SGD.from_json) |
| n_api.html#cudaq.State.from_data) |     -   [(cudaq.optimizers.SPSA   |
|                                   |         static                    |
|                                   |         metho                     |
|                                   | d)](api/languages/python_api.html |
|                                   | #cudaq.optimizers.SPSA.from_json) |
|                                   |     -   [(cudaq.PyKernelDecorator |
|                                   |         static                    |
|                                   |         method)                   |
|                                   | ](api/languages/python_api.html#c |
|                                   | udaq.PyKernelDecorator.from_json) |
|                                   | -   [from_word()                  |
|                                   |     (                             |
|                                   | cudaq.operators.spin.SpinOperator |
|                                   |     static                        |
|                                   |     method)](api/lang             |
|                                   | uages/python_api.html#cudaq.opera |
|                                   | tors.spin.SpinOperator.from_word) |
+-----------------------------------+-----------------------------------+

## G {#G}

+-----------------------------------+-----------------------------------+
| -   [gamma (cudaq.optimizers.SPSA | -   [get_register_counts()        |
|     pro                           |     (cudaq.SampleResult           |
| perty)](api/languages/python_api. |     method)](api                  |
| html#cudaq.optimizers.SPSA.gamma) | /languages/python_api.html#cudaq. |
| -   [get()                        | SampleResult.get_register_counts) |
|     (cudaq.AsyncEvolveResult      | -   [get_sequential_data()        |
|     m                             |     (cudaq.SampleResult           |
| ethod)](api/languages/python_api. |     method)](api                  |
| html#cudaq.AsyncEvolveResult.get) | /languages/python_api.html#cudaq. |
|                                   | SampleResult.get_sequential_data) |
|    -   [(cudaq.AsyncObserveResult | -   [get_spin()                   |
|         me                        |     (cudaq.ObserveResult          |
| thod)](api/languages/python_api.h |     me                            |
| tml#cudaq.AsyncObserveResult.get) | thod)](api/languages/python_api.h |
|     -   [(cudaq.AsyncStateResult  | tml#cudaq.ObserveResult.get_spin) |
|                                   | -   [get_state() (in module       |
| method)](api/languages/python_api |     cudaq)](api/languages         |
| .html#cudaq.AsyncStateResult.get) | /python_api.html#cudaq.get_state) |
| -   [get_binary_symplectic_form() | -   [get_state_async() (in module |
|     (cuda                         |     cudaq)](api/languages/pytho   |
| q.operators.spin.SpinOperatorTerm | n_api.html#cudaq.get_state_async) |
|     metho                         | -   [get_state_refval()           |
| d)](api/languages/python_api.html |     (cudaq.State                  |
| #cudaq.operators.spin.SpinOperato |     me                            |
| rTerm.get_binary_symplectic_form) | thod)](api/languages/python_api.h |
| -   [get_channels()               | tml#cudaq.State.get_state_refval) |
|     (cudaq.NoiseModel             | -   [get_target() (in module      |
|     met                           |     cudaq)](api/languages/        |
| hod)](api/languages/python_api.ht | python_api.html#cudaq.get_target) |
| ml#cudaq.NoiseModel.get_channels) | -   [get_targets() (in module     |
| -   [get_coefficient()            |     cudaq)](api/languages/p       |
|     (                             | ython_api.html#cudaq.get_targets) |
| cudaq.operators.spin.SpinOperator | -   [get_term_count()             |
|     method)](api/languages/       |     (                             |
| python_api.html#cudaq.operators.s | cudaq.operators.spin.SpinOperator |
| pin.SpinOperator.get_coefficient) |     method)](api/languages        |
|     -   [(cuda                    | /python_api.html#cudaq.operators. |
| q.operators.spin.SpinOperatorTerm | spin.SpinOperator.get_term_count) |
|                                   | -   [get_total_shots()            |
|       method)](api/languages/pyth |     (cudaq.SampleResult           |
| on_api.html#cudaq.operators.spin. |     method)]                      |
| SpinOperatorTerm.get_coefficient) | (api/languages/python_api.html#cu |
| -   [get_marginal_counts()        | daq.SampleResult.get_total_shots) |
|     (cudaq.SampleResult           | -   [get_trajectory()             |
|     method)](api                  |                                   |
| /languages/python_api.html#cudaq. |   (cudaq.ptsbe.PTSBEExecutionData |
| SampleResult.get_marginal_counts) |     method)](api/langua           |
| -   [get_ops()                    | ges/python_api.html#cudaq.ptsbe.P |
|     (cudaq.KrausChannel           | TSBEExecutionData.get_trajectory) |
|                                   | -   [getTensor() (cudaq.State     |
| method)](api/languages/python_api |     method)](api/languages/pytho  |
| .html#cudaq.KrausChannel.get_ops) | n_api.html#cudaq.State.getTensor) |
| -   [get_pauli_word()             | -   [getTensors() (cudaq.State    |
|     (cuda                         |     method)](api/languages/python |
| q.operators.spin.SpinOperatorTerm | _api.html#cudaq.State.getTensors) |
|     method)](api/languages/pyt    | -   [gradient (class in           |
| hon_api.html#cudaq.operators.spin |     cudaq.g                       |
| .SpinOperatorTerm.get_pauli_word) | radients)](api/languages/python_a |
| -   [get_precision()              | pi.html#cudaq.gradients.gradient) |
|     (cudaq.Target                 | -   [GradientDescent (class in    |
|                                   |     cudaq.optimizers              |
| method)](api/languages/python_api | )](api/languages/python_api.html# |
| .html#cudaq.Target.get_precision) | cudaq.optimizers.GradientDescent) |
| -   [get_qubit_count()            |                                   |
|     (                             |                                   |
| cudaq.operators.spin.SpinOperator |                                   |
|     method)](api/languages/       |                                   |
| python_api.html#cudaq.operators.s |                                   |
| pin.SpinOperator.get_qubit_count) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|                                   |                                   |
|       method)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.spin. |                                   |
| SpinOperatorTerm.get_qubit_count) |                                   |
| -   [get_raw_data()               |                                   |
|     (                             |                                   |
| cudaq.operators.spin.SpinOperator |                                   |
|     method)](api/languag          |                                   |
| es/python_api.html#cudaq.operator |                                   |
| s.spin.SpinOperator.get_raw_data) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         method)](api/languages/p  |                                   |
| ython_api.html#cudaq.operators.sp |                                   |
| in.SpinOperatorTerm.get_raw_data) |                                   |
+-----------------------------------+-----------------------------------+

## H {#H}

+-----------------------------------+-----------------------------------+
| -   [has_execution_data()         | -   [has_target() (in module      |
|                                   |     cudaq)](api/languages/        |
|    (cudaq.ptsbe.PTSBESampleResult | python_api.html#cudaq.has_target) |
|     method)](api/languages        |                                   |
| /python_api.html#cudaq.ptsbe.PTSB |                                   |
| ESampleResult.has_execution_data) |                                   |
+-----------------------------------+-----------------------------------+

## I {#I}

+-----------------------------------+-----------------------------------+
| -   [i() (in module               | -   [initialize() (in module      |
|     cudaq.spin)](api/langua       |                                   |
| ges/python_api.html#cudaq.spin.i) |    cudaq.mpi)](api/languages/pyth |
| -   [id                           | on_api.html#cudaq.mpi.initialize) |
|     (cuda                         | -   [initialize_cudaq() (in       |
| q.operators.MatrixOperatorElement |     module                        |
|     property)](api/l              |     cudaq)](api/languages/python  |
| anguages/python_api.html#cudaq.op | _api.html#cudaq.initialize_cudaq) |
| erators.MatrixOperatorElement.id) | -   [InitialState (in module      |
| -   [identities() (in module      |     cudaq.dynamics.helpers)](     |
|     c                             | api/languages/python_api.html#cud |
| udaq.boson)](api/languages/python | aq.dynamics.helpers.InitialState) |
| _api.html#cudaq.boson.identities) | -   [InitialStateType (class in   |
|     -   [(in module               |     cudaq)](api/languages/python  |
|         cudaq                     | _api.html#cudaq.InitialStateType) |
| .fermion)](api/languages/python_a | -   [instantiate()                |
| pi.html#cudaq.fermion.identities) |     (cudaq.operators              |
|     -   [(in module               |     m                             |
|         cudaq.operators.custom)   | ethod)](api/languages/python_api. |
| ](api/languages/python_api.html#c | html#cudaq.operators.instantiate) |
| udaq.operators.custom.identities) |     -   [(in module               |
|     -   [(in module               |         cudaq.operators.custom)]  |
|                                   | (api/languages/python_api.html#cu |
|  cudaq.spin)](api/languages/pytho | daq.operators.custom.instantiate) |
| n_api.html#cudaq.spin.identities) | -   [intermediate_states()        |
| -   [identity()                   |     (cudaq.EvolveResult           |
|     (cu                           |     method)](api                  |
| daq.operators.boson.BosonOperator | /languages/python_api.html#cudaq. |
|     static                        | EvolveResult.intermediate_states) |
|     method)](api/langu            | -   [IntermediateResultSave       |
| ages/python_api.html#cudaq.operat |     (class in                     |
| ors.boson.BosonOperator.identity) |     c                             |
|     -   [(cudaq.                  | udaq)](api/languages/python_api.h |
| operators.fermion.FermionOperator | tml#cudaq.IntermediateResultSave) |
|         static                    | -   [is_compiled()                |
|         method)](api/languages    |     (cudaq.PyKernelDecorator      |
| /python_api.html#cudaq.operators. |     method)](                     |
| fermion.FermionOperator.identity) | api/languages/python_api.html#cud |
|     -                             | aq.PyKernelDecorator.is_compiled) |
|  [(cudaq.operators.MatrixOperator | -   [is_constant()                |
|         static                    |                                   |
|         method)](api/             |   (cudaq.operators.ScalarOperator |
| languages/python_api.html#cudaq.o |     method)](api/lan              |
| perators.MatrixOperator.identity) | guages/python_api.html#cudaq.oper |
|     -   [(                        | ators.ScalarOperator.is_constant) |
| cudaq.operators.spin.SpinOperator | -   [is_emulated() (cudaq.Target  |
|         static                    |                                   |
|         method)](api/lan          |   method)](api/languages/python_a |
| guages/python_api.html#cudaq.oper | pi.html#cudaq.Target.is_emulated) |
| ators.spin.SpinOperator.identity) | -   [is_identity()                |
|     -   [(in module               |     (cudaq.                       |
|                                   | operators.boson.BosonOperatorTerm |
|  cudaq.boson)](api/languages/pyth |     method)](api/languages/py     |
| on_api.html#cudaq.boson.identity) | thon_api.html#cudaq.operators.bos |
|     -   [(in module               | on.BosonOperatorTerm.is_identity) |
|         cud                       |     -   [(cudaq.oper              |
| aq.fermion)](api/languages/python | ators.fermion.FermionOperatorTerm |
| _api.html#cudaq.fermion.identity) |                                   |
|     -   [(in module               |     method)](api/languages/python |
|                                   | _api.html#cudaq.operators.fermion |
|    cudaq.spin)](api/languages/pyt | .FermionOperatorTerm.is_identity) |
| hon_api.html#cudaq.spin.identity) |     -   [(c                       |
| -   [initial_parameters           | udaq.operators.MatrixOperatorTerm |
|     (cudaq.optimizers.Adam        |         method)](api/languag      |
|     property)](api/l              | es/python_api.html#cudaq.operator |
| anguages/python_api.html#cudaq.op | s.MatrixOperatorTerm.is_identity) |
| timizers.Adam.initial_parameters) |     -   [(                        |
|     -   [(cudaq.optimizers.COBYLA | cudaq.operators.spin.SpinOperator |
|         property)](api/lan        |         method)](api/langua       |
| guages/python_api.html#cudaq.opti | ges/python_api.html#cudaq.operato |
| mizers.COBYLA.initial_parameters) | rs.spin.SpinOperator.is_identity) |
|     -   [                         |     -   [(cuda                    |
| (cudaq.optimizers.GradientDescent | q.operators.spin.SpinOperatorTerm |
|                                   |         method)](api/languages/   |
|       property)](api/languages/py | python_api.html#cudaq.operators.s |
| thon_api.html#cudaq.optimizers.Gr | pin.SpinOperatorTerm.is_identity) |
| adientDescent.initial_parameters) | -   [is_initialized() (in module  |
|     -   [(cudaq.optimizers.LBFGS  |     c                             |
|         property)](api/la         | udaq.mpi)](api/languages/python_a |
| nguages/python_api.html#cudaq.opt | pi.html#cudaq.mpi.is_initialized) |
| imizers.LBFGS.initial_parameters) | -   [is_on_gpu() (cudaq.State     |
|                                   |     method)](api/languages/pytho  |
| -   [(cudaq.optimizers.NelderMead | n_api.html#cudaq.State.is_on_gpu) |
|         property)](api/languag    | -   [is_remote() (cudaq.Target    |
| es/python_api.html#cudaq.optimize |     method)](api/languages/python |
| rs.NelderMead.initial_parameters) | _api.html#cudaq.Target.is_remote) |
|     -   [(cudaq.optimizers.SGD    | -   [is_remote_simulator()        |
|         property)](api/           |     (cudaq.Target                 |
| languages/python_api.html#cudaq.o |     method                        |
| ptimizers.SGD.initial_parameters) | )](api/languages/python_api.html# |
|     -   [(cudaq.optimizers.SPSA   | cudaq.Target.is_remote_simulator) |
|         property)](api/l          | -   [items() (cudaq.SampleResult  |
| anguages/python_api.html#cudaq.op |                                   |
| timizers.SPSA.initial_parameters) |   method)](api/languages/python_a |
|                                   | pi.html#cudaq.SampleResult.items) |
+-----------------------------------+-----------------------------------+

## K {#K}

+-----------------------------------+-----------------------------------+
| -   [Kernel (in module            | -   [KrausOperator (class in      |
|     cudaq)](api/langua            |     cudaq)](api/languages/pyt     |
| ges/python_api.html#cudaq.Kernel) | hon_api.html#cudaq.KrausOperator) |
| -   [kernel() (in module          | -   [KrausSelection (class in     |
|     cudaq)](api/langua            |     cudaq                         |
| ges/python_api.html#cudaq.kernel) | .ptsbe)](api/languages/python_api |
| -   [KrausChannel (class in       | .html#cudaq.ptsbe.KrausSelection) |
|     cudaq)](api/languages/py      | -   [KrausTrajectory (class in    |
| thon_api.html#cudaq.KrausChannel) |     cudaq.                        |
|                                   | ptsbe)](api/languages/python_api. |
|                                   | html#cudaq.ptsbe.KrausTrajectory) |
+-----------------------------------+-----------------------------------+

## L {#L}

+-----------------------------------------------------------------------+
| -   [launch_args_required() (cudaq.PyKernelDecorator                  |
|     method)](api/la                                                   |
| nguages/python_api.html#cudaq.PyKernelDecorator.launch_args_required) |
| -   [LBFGS (class in                                                  |
|     cud                                                               |
| aq.optimizers)](api/languages/python_api.html#cudaq.optimizers.LBFGS) |
| -   [left_multiply() (cudaq.SuperOperator static                      |
|     meth                                                              |
| od)](api/languages/python_api.html#cudaq.SuperOperator.left_multiply) |
| -   [left_right_multiply() (cudaq.SuperOperator static                |
|     method)](a                                                        |
| pi/languages/python_api.html#cudaq.SuperOperator.left_right_multiply) |
| -   [lower_bounds (cudaq.optimizers.Adam                              |
|     propert                                                           |
| y)](api/languages/python_api.html#cudaq.optimizers.Adam.lower_bounds) |
|     -   [(cudaq.optimizers.COBYLA                                     |
|         property)                                                     |
| ](api/languages/python_api.html#cudaq.optimizers.COBYLA.lower_bounds) |
|     -   [(cudaq.optimizers.GradientDescent                            |
|         property)](api/lan                                            |
| guages/python_api.html#cudaq.optimizers.GradientDescent.lower_bounds) |
|     -   [(cudaq.optimizers.LBFGS                                      |
|         property                                                      |
| )](api/languages/python_api.html#cudaq.optimizers.LBFGS.lower_bounds) |
|     -   [(cudaq.optimizers.NelderMead                                 |
|         property)](ap                                                 |
| i/languages/python_api.html#cudaq.optimizers.NelderMead.lower_bounds) |
|     -   [(cudaq.optimizers.SGD                                        |
|         proper                                                        |
| ty)](api/languages/python_api.html#cudaq.optimizers.SGD.lower_bounds) |
|     -   [(cudaq.optimizers.SPSA                                       |
|         propert                                                       |
| y)](api/languages/python_api.html#cudaq.optimizers.SPSA.lower_bounds) |
+-----------------------------------------------------------------------+

## M {#M}

+-----------------------------------+-----------------------------------+
| -   [make_kernel() (in module     | -   [merge_quake_source()         |
|     cudaq)](api/languages/p       |     (cudaq.PyKernelDecorator      |
| ython_api.html#cudaq.make_kernel) |     method)](api/lan              |
| -   [MatrixOperator (class in     | guages/python_api.html#cudaq.PyKe |
|     cudaq.operato                 | rnelDecorator.merge_quake_source) |
| rs)](api/languages/python_api.htm | -   [min_degree                   |
| l#cudaq.operators.MatrixOperator) |     (cu                           |
| -   [MatrixOperatorElement (class | daq.operators.boson.BosonOperator |
|     in                            |     property)](api/languag        |
|     cudaq.operators)](ap          | es/python_api.html#cudaq.operator |
| i/languages/python_api.html#cudaq | s.boson.BosonOperator.min_degree) |
| .operators.MatrixOperatorElement) |     -   [(cudaq.                  |
| -   [MatrixOperatorTerm (class in | operators.boson.BosonOperatorTerm |
|     cudaq.operators)]             |                                   |
| (api/languages/python_api.html#cu |        property)](api/languages/p |
| daq.operators.MatrixOperatorTerm) | ython_api.html#cudaq.operators.bo |
| -   [max_degree                   | son.BosonOperatorTerm.min_degree) |
|     (cu                           |     -   [(cudaq.                  |
| daq.operators.boson.BosonOperator | operators.fermion.FermionOperator |
|     property)](api/languag        |                                   |
| es/python_api.html#cudaq.operator |        property)](api/languages/p |
| s.boson.BosonOperator.max_degree) | ython_api.html#cudaq.operators.fe |
|     -   [(cudaq.                  | rmion.FermionOperator.min_degree) |
| operators.boson.BosonOperatorTerm |     -   [(cudaq.oper              |
|                                   | ators.fermion.FermionOperatorTerm |
|        property)](api/languages/p |                                   |
| ython_api.html#cudaq.operators.bo |    property)](api/languages/pytho |
| son.BosonOperatorTerm.max_degree) | n_api.html#cudaq.operators.fermio |
|     -   [(cudaq.                  | n.FermionOperatorTerm.min_degree) |
| operators.fermion.FermionOperator |     -                             |
|                                   |  [(cudaq.operators.MatrixOperator |
|        property)](api/languages/p |         property)](api/la         |
| ython_api.html#cudaq.operators.fe | nguages/python_api.html#cudaq.ope |
| rmion.FermionOperator.max_degree) | rators.MatrixOperator.min_degree) |
|     -   [(cudaq.oper              |     -   [(c                       |
| ators.fermion.FermionOperatorTerm | udaq.operators.MatrixOperatorTerm |
|                                   |         property)](api/langua     |
|    property)](api/languages/pytho | ges/python_api.html#cudaq.operato |
| n_api.html#cudaq.operators.fermio | rs.MatrixOperatorTerm.min_degree) |
| n.FermionOperatorTerm.max_degree) |     -   [(                        |
|     -                             | cudaq.operators.spin.SpinOperator |
|  [(cudaq.operators.MatrixOperator |         property)](api/langu      |
|         property)](api/la         | ages/python_api.html#cudaq.operat |
| nguages/python_api.html#cudaq.ope | ors.spin.SpinOperator.min_degree) |
| rators.MatrixOperator.max_degree) |     -   [(cuda                    |
|     -   [(c                       | q.operators.spin.SpinOperatorTerm |
| udaq.operators.MatrixOperatorTerm |         property)](api/languages  |
|         property)](api/langua     | /python_api.html#cudaq.operators. |
| ges/python_api.html#cudaq.operato | spin.SpinOperatorTerm.min_degree) |
| rs.MatrixOperatorTerm.max_degree) | -   [minimal_eigenvalue()         |
|     -   [(                        |     (cudaq.ComplexMatrix          |
| cudaq.operators.spin.SpinOperator |     method)](api                  |
|         property)](api/langu      | /languages/python_api.html#cudaq. |
| ages/python_api.html#cudaq.operat | ComplexMatrix.minimal_eigenvalue) |
| ors.spin.SpinOperator.max_degree) | -   [minus() (in module           |
|     -   [(cuda                    |     cudaq.spin)](api/languages/   |
| q.operators.spin.SpinOperatorTerm | python_api.html#cudaq.spin.minus) |
|         property)](api/languages  | -   module                        |
| /python_api.html#cudaq.operators. |     -   [cudaq](api/langua        |
| spin.SpinOperatorTerm.max_degree) | ges/python_api.html#module-cudaq) |
| -   [max_iterations               |     -                             |
|     (cudaq.optimizers.Adam        |    [cudaq.boson](api/languages/py |
|     property)](a                  | thon_api.html#module-cudaq.boson) |
| pi/languages/python_api.html#cuda |     -   [                         |
| q.optimizers.Adam.max_iterations) | cudaq.fermion](api/languages/pyth |
|     -   [(cudaq.optimizers.COBYLA | on_api.html#module-cudaq.fermion) |
|         property)](api            |     -   [cudaq.operators.cu       |
| /languages/python_api.html#cudaq. | stom](api/languages/python_api.ht |
| optimizers.COBYLA.max_iterations) | ml#module-cudaq.operators.custom) |
|     -   [                         |                                   |
| (cudaq.optimizers.GradientDescent |  -   [cudaq.spin](api/languages/p |
|         property)](api/language   | ython_api.html#module-cudaq.spin) |
| s/python_api.html#cudaq.optimizer | -   [momentum() (in module        |
| s.GradientDescent.max_iterations) |                                   |
|     -   [(cudaq.optimizers.LBFGS  |  cudaq.boson)](api/languages/pyth |
|         property)](ap             | on_api.html#cudaq.boson.momentum) |
| i/languages/python_api.html#cudaq |     -   [(in module               |
| .optimizers.LBFGS.max_iterations) |         cudaq.operators.custo     |
|                                   | m)](api/languages/python_api.html |
| -   [(cudaq.optimizers.NelderMead | #cudaq.operators.custom.momentum) |
|         property)](api/lan        | -   [most_probable()              |
| guages/python_api.html#cudaq.opti |     (cudaq.SampleResult           |
| mizers.NelderMead.max_iterations) |     method                        |
|     -   [(cudaq.optimizers.SGD    | )](api/languages/python_api.html# |
|         property)](               | cudaq.SampleResult.most_probable) |
| api/languages/python_api.html#cud | -   [multiplicity                 |
| aq.optimizers.SGD.max_iterations) |     (cudaq.ptsbe.KrausTrajectory  |
|     -   [(cudaq.optimizers.SPSA   |     property)](api/l              |
|         property)](a              | anguages/python_api.html#cudaq.pt |
| pi/languages/python_api.html#cuda | sbe.KrausTrajectory.multiplicity) |
| q.optimizers.SPSA.max_iterations) |                                   |
| -   [mdiag_sparse_matrix (C++     |                                   |
|     type)](api/languages/cpp_api. |                                   |
| html#_CPPv419mdiag_sparse_matrix) |                                   |
| -   [merge_kernel()               |                                   |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](a                    |                                   |
| pi/languages/python_api.html#cuda |                                   |
| q.PyKernelDecorator.merge_kernel) |                                   |
+-----------------------------------+-----------------------------------+

## N {#N}

+-----------------------------------+-----------------------------------+
| -   [name                         | -   [num_columns()                |
|                                   |     (cudaq.ComplexMatrix          |
|   (cudaq.ptsbe.ShotAllocationType |     metho                         |
|     property)](                   | d)](api/languages/python_api.html |
| api/languages/python_api.html#cud | #cudaq.ComplexMatrix.num_columns) |
| aq.ptsbe.ShotAllocationType.name) | -   [num_qpus() (cudaq.Target     |
|     -   [                         |     method)](api/languages/pytho  |
| (cudaq.ptsbe.TraceInstructionType | n_api.html#cudaq.Target.num_qpus) |
|         property)](ap             | -   [num_qubits (cudaq.Resources  |
| i/languages/python_api.html#cudaq |     pr                            |
| .ptsbe.TraceInstructionType.name) | operty)](api/languages/python_api |
|     -   [(cudaq.PyKernel          | .html#cudaq.Resources.num_qubits) |
|                                   | -   [num_qubits() (cudaq.State    |
|     attribute)](api/languages/pyt |     method)](api/languages/python |
| hon_api.html#cudaq.PyKernel.name) | _api.html#cudaq.State.num_qubits) |
|                                   | -   [num_ranks() (in module       |
|   -   [(cudaq.SimulationPrecision |     cudaq.mpi)](api/languages/pyt |
|         proper                    | hon_api.html#cudaq.mpi.num_ranks) |
| ty)](api/languages/python_api.htm | -   [num_rows()                   |
| l#cudaq.SimulationPrecision.name) |     (cudaq.ComplexMatrix          |
|     -   [(cudaq.spin.Pauli        |     me                            |
|                                   | thod)](api/languages/python_api.h |
|    property)](api/languages/pytho | tml#cudaq.ComplexMatrix.num_rows) |
| n_api.html#cudaq.spin.Pauli.name) | -   [number() (in module          |
|     -   [(cudaq.Target            |                                   |
|                                   |    cudaq.boson)](api/languages/py |
|        property)](api/languages/p | thon_api.html#cudaq.boson.number) |
| ython_api.html#cudaq.Target.name) |     -   [(in module               |
| -   [name()                       |         c                         |
|                                   | udaq.fermion)](api/languages/pyth |
|  (cudaq.ptsbe.PTSSamplingStrategy | on_api.html#cudaq.fermion.number) |
|     method)](a                    |     -   [(in module               |
| pi/languages/python_api.html#cuda |         cudaq.operators.cus       |
| q.ptsbe.PTSSamplingStrategy.name) | tom)](api/languages/python_api.ht |
| -   [NelderMead (class in         | ml#cudaq.operators.custom.number) |
|     cudaq.optim                   | -   [nvqir::MPSSimulationState    |
| izers)](api/languages/python_api. |     (C++                          |
| html#cudaq.optimizers.NelderMead) |     class)]                       |
| -   [NoiseModel (class in         | (api/languages/cpp_api.html#_CPPv |
|     cudaq)](api/languages/        | 4I0EN5nvqir18MPSSimulationStateE) |
| python_api.html#cudaq.NoiseModel) | -                                 |
| -   [num_available_gpus() (in     |  [nvqir::TensorNetSimulationState |
|     module                        |     (C++                          |
|                                   |     class)](api/l                 |
|    cudaq)](api/languages/python_a | anguages/cpp_api.html#_CPPv4I0EN5 |
| pi.html#cudaq.num_available_gpus) | nvqir24TensorNetSimulationStateE) |
+-----------------------------------+-----------------------------------+

## O {#O}

+-----------------------------------+-----------------------------------+
| -   [observe() (in module         | -   [OptimizationResult (class in |
|     cudaq)](api/languag           |                                   |
| es/python_api.html#cudaq.observe) |    cudaq)](api/languages/python_a |
| -   [observe_async() (in module   | pi.html#cudaq.OptimizationResult) |
|     cudaq)](api/languages/pyt     | -   [OrderedSamplingStrategy      |
| hon_api.html#cudaq.observe_async) |     (class in                     |
| -   [ObserveResult (class in      |     cudaq.ptsbe)](                |
|     cudaq)](api/languages/pyt     | api/languages/python_api.html#cud |
| hon_api.html#cudaq.ObserveResult) | aq.ptsbe.OrderedSamplingStrategy) |
| -   [OperatorSum (in module       | -   [overlap() (cudaq.State       |
|     cudaq.oper                    |     method)](api/languages/pyt    |
| ators)](api/languages/python_api. | hon_api.html#cudaq.State.overlap) |
| html#cudaq.operators.OperatorSum) |                                   |
| -   [ops_count                    |                                   |
|     (cudaq.                       |                                   |
| operators.boson.BosonOperatorTerm |                                   |
|     property)](api/languages/     |                                   |
| python_api.html#cudaq.operators.b |                                   |
| oson.BosonOperatorTerm.ops_count) |                                   |
|     -   [(cudaq.oper              |                                   |
| ators.fermion.FermionOperatorTerm |                                   |
|                                   |                                   |
|     property)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.fermi |                                   |
| on.FermionOperatorTerm.ops_count) |                                   |
|     -   [(c                       |                                   |
| udaq.operators.MatrixOperatorTerm |                                   |
|         property)](api/langu      |                                   |
| ages/python_api.html#cudaq.operat |                                   |
| ors.MatrixOperatorTerm.ops_count) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         property)](api/language   |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .spin.SpinOperatorTerm.ops_count) |                                   |
+-----------------------------------+-----------------------------------+

## P {#P}

+-----------------------------------+-----------------------------------+
| -   [parameters                   | -   [PhaseDamping (class in       |
|     (cu                           |     cudaq)](api/languages/py      |
| daq.operators.boson.BosonOperator | thon_api.html#cudaq.PhaseDamping) |
|     property)](api/languag        | -   [PhaseFlipChannel (class in   |
| es/python_api.html#cudaq.operator |     cudaq)](api/languages/python  |
| s.boson.BosonOperator.parameters) | _api.html#cudaq.PhaseFlipChannel) |
|     -   [(cudaq.                  | -   [platform (cudaq.Target       |
| operators.boson.BosonOperatorTerm |                                   |
|                                   |    property)](api/languages/pytho |
|        property)](api/languages/p | n_api.html#cudaq.Target.platform) |
| ython_api.html#cudaq.operators.bo | -   [plus() (in module            |
| son.BosonOperatorTerm.parameters) |     cudaq.spin)](api/languages    |
|     -   [(cudaq.                  | /python_api.html#cudaq.spin.plus) |
| operators.fermion.FermionOperator | -   [position() (in module        |
|                                   |                                   |
|        property)](api/languages/p |  cudaq.boson)](api/languages/pyth |
| ython_api.html#cudaq.operators.fe | on_api.html#cudaq.boson.position) |
| rmion.FermionOperator.parameters) |     -   [(in module               |
|     -   [(cudaq.oper              |         cudaq.operators.custo     |
| ators.fermion.FermionOperatorTerm | m)](api/languages/python_api.html |
|                                   | #cudaq.operators.custom.position) |
|    property)](api/languages/pytho | -   [prepare_call()               |
| n_api.html#cudaq.operators.fermio |     (cudaq.PyKernelDecorator      |
| n.FermionOperatorTerm.parameters) |     method)](a                    |
|     -                             | pi/languages/python_api.html#cuda |
|  [(cudaq.operators.MatrixOperator | q.PyKernelDecorator.prepare_call) |
|         property)](api/la         | -                                 |
| nguages/python_api.html#cudaq.ope |    [ProbabilisticSamplingStrategy |
| rators.MatrixOperator.parameters) |     (class in                     |
|     -   [(cuda                    |     cudaq.ptsbe)](api/la          |
| q.operators.MatrixOperatorElement | nguages/python_api.html#cudaq.pts |
|         property)](api/languages  | be.ProbabilisticSamplingStrategy) |
| /python_api.html#cudaq.operators. | -   [probability()                |
| MatrixOperatorElement.parameters) |     (cudaq.SampleResult           |
|     -   [(c                       |     meth                          |
| udaq.operators.MatrixOperatorTerm | od)](api/languages/python_api.htm |
|         property)](api/langua     | l#cudaq.SampleResult.probability) |
| ges/python_api.html#cudaq.operato | -   [process_call_arguments()     |
| rs.MatrixOperatorTerm.parameters) |     (cudaq.PyKernelDecorator      |
|     -                             |     method)](api/languag          |
|  [(cudaq.operators.ScalarOperator | es/python_api.html#cudaq.PyKernel |
|         property)](api/la         | Decorator.process_call_arguments) |
| nguages/python_api.html#cudaq.ope | -   [ProductOperator (in module   |
| rators.ScalarOperator.parameters) |     cudaq.operator                |
|     -   [(                        | s)](api/languages/python_api.html |
| cudaq.operators.spin.SpinOperator | #cudaq.operators.ProductOperator) |
|         property)](api/langu      | -   [ptsbe_execution_data         |
| ages/python_api.html#cudaq.operat |                                   |
| ors.spin.SpinOperator.parameters) |    (cudaq.ptsbe.PTSBESampleResult |
|     -   [(cuda                    |     property)](api/languages/p    |
| q.operators.spin.SpinOperatorTerm | ython_api.html#cudaq.ptsbe.PTSBES |
|         property)](api/languages  | ampleResult.ptsbe_execution_data) |
| /python_api.html#cudaq.operators. | -   [PTSBEExecutionData (class in |
| spin.SpinOperatorTerm.parameters) |     cudaq.pts                     |
| -   [ParameterShift (class in     | be)](api/languages/python_api.htm |
|     cudaq.gradien                 | l#cudaq.ptsbe.PTSBEExecutionData) |
| ts)](api/languages/python_api.htm | -   [PTSBESampleResult (class in  |
| l#cudaq.gradients.ParameterShift) |     cudaq.pt                      |
| -   [parity() (in module          | sbe)](api/languages/python_api.ht |
|     cudaq.operators.cus           | ml#cudaq.ptsbe.PTSBESampleResult) |
| tom)](api/languages/python_api.ht | -   [PTSSamplingStrategy (class   |
| ml#cudaq.operators.custom.parity) |     in                            |
| -   [Pauli (class in              |     cudaq.ptsb                    |
|     cudaq.spin)](api/languages/   | e)](api/languages/python_api.html |
| python_api.html#cudaq.spin.Pauli) | #cudaq.ptsbe.PTSSamplingStrategy) |
| -   [Pauli1 (class in             | -   [PyKernel (class in           |
|     cudaq)](api/langua            |     cudaq)](api/language          |
| ges/python_api.html#cudaq.Pauli1) | s/python_api.html#cudaq.PyKernel) |
| -   [Pauli2 (class in             | -   [PyKernelDecorator (class in  |
|     cudaq)](api/langua            |     cudaq)](api/languages/python_ |
| ges/python_api.html#cudaq.Pauli2) | api.html#cudaq.PyKernelDecorator) |
| -   [per_qubit_depth              |                                   |
|     (cudaq.Resources              |                                   |
|     propert                       |                                   |
| y)](api/languages/python_api.html |                                   |
| #cudaq.Resources.per_qubit_depth) |                                   |
| -   [per_qubit_depth_2q           |                                   |
|     (cudaq.Resources              |                                   |
|     property)]                    |                                   |
| (api/languages/python_api.html#cu |                                   |
| daq.Resources.per_qubit_depth_2q) |                                   |
+-----------------------------------+-----------------------------------+

## Q {#Q}

+-----------------------------------+-----------------------------------+
| -   [qkeModule                    | -   [qubit (class in              |
|     (cudaq.PyKernelDecorator      |     cudaq)](api/langu             |
|     property)                     | ages/python_api.html#cudaq.qubit) |
| ](api/languages/python_api.html#c | -   [qubit_count                  |
| udaq.PyKernelDecorator.qkeModule) |     (                             |
| -   [qreg (in module              | cudaq.operators.spin.SpinOperator |
|     cudaq)](api/lang              |     property)](api/langua         |
| uages/python_api.html#cudaq.qreg) | ges/python_api.html#cudaq.operato |
| -   [QuakeValue (class in         | rs.spin.SpinOperator.qubit_count) |
|     cudaq)](api/languages/        |     -   [(cuda                    |
| python_api.html#cudaq.QuakeValue) | q.operators.spin.SpinOperatorTerm |
|                                   |         property)](api/languages/ |
|                                   | python_api.html#cudaq.operators.s |
|                                   | pin.SpinOperatorTerm.qubit_count) |
|                                   | -   [qvector (class in            |
|                                   |     cudaq)](api/languag           |
|                                   | es/python_api.html#cudaq.qvector) |
+-----------------------------------+-----------------------------------+

## R {#R}

+-----------------------------------+-----------------------------------+
| -   [random()                     | -   [Resources (class in          |
|     (                             |     cudaq)](api/languages         |
| cudaq.operators.spin.SpinOperator | /python_api.html#cudaq.Resources) |
|     static                        | -   [right_multiply()             |
|     method)](api/l                |     (cudaq.SuperOperator static   |
| anguages/python_api.html#cudaq.op |     method)]                      |
| erators.spin.SpinOperator.random) | (api/languages/python_api.html#cu |
| -   [rank() (in module            | daq.SuperOperator.right_multiply) |
|     cudaq.mpi)](api/language      | -   [row_count                    |
| s/python_api.html#cudaq.mpi.rank) |     (cudaq.KrausOperator          |
| -   [register_names               |     prope                         |
|     (cudaq.SampleResult           | rty)](api/languages/python_api.ht |
|     attribute)                    | ml#cudaq.KrausOperator.row_count) |
| ](api/languages/python_api.html#c | -   [run() (in module             |
| udaq.SampleResult.register_names) |     cudaq)](api/lan               |
| -                                 | guages/python_api.html#cudaq.run) |
|   [register_set_target_callback() | -   [run_async() (in module       |
|     (in module                    |     cudaq)](api/languages         |
|     cudaq)]                       | /python_api.html#cudaq.run_async) |
| (api/languages/python_api.html#cu | -   [RydbergHamiltonian (class in |
| daq.register_set_target_callback) |     cudaq.operators)]             |
| -   [reset_target() (in module    | (api/languages/python_api.html#cu |
|     cudaq)](api/languages/py      | daq.operators.RydbergHamiltonian) |
| thon_api.html#cudaq.reset_target) |                                   |
| -   [resolve_captured_arguments() |                                   |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](api/languages/p      |                                   |
| ython_api.html#cudaq.PyKernelDeco |                                   |
| rator.resolve_captured_arguments) |                                   |
+-----------------------------------+-----------------------------------+

## S {#S}

+-----------------------------------+-----------------------------------+
| -   [sample() (in module          | -   [ShotAllocationType (class in |
|     cudaq)](api/langua            |     cudaq.pts                     |
| ges/python_api.html#cudaq.sample) | be)](api/languages/python_api.htm |
|     -   [(in module               | l#cudaq.ptsbe.ShotAllocationType) |
|                                   | -   [signatureWithCallables()     |
|      cudaq.orca)](api/languages/p |     (cudaq.PyKernelDecorator      |
| ython_api.html#cudaq.orca.sample) |     method)](api/languag          |
|     -   [(in module               | es/python_api.html#cudaq.PyKernel |
|                                   | Decorator.signatureWithCallables) |
|    cudaq.ptsbe)](api/languages/py | -   [SimulationPrecision (class   |
| thon_api.html#cudaq.ptsbe.sample) |     in                            |
| -   [sample_async() (in module    |                                   |
|     cudaq)](api/languages/py      |   cudaq)](api/languages/python_ap |
| thon_api.html#cudaq.sample_async) | i.html#cudaq.SimulationPrecision) |
|     -   [(in module               | -   [simulator (cudaq.Target      |
|         cud                       |                                   |
| aq.ptsbe)](api/languages/python_a |   property)](api/languages/python |
| pi.html#cudaq.ptsbe.sample_async) | _api.html#cudaq.Target.simulator) |
| -   [SampleResult (class in       | -   [slice() (cudaq.QuakeValue    |
|     cudaq)](api/languages/py      |     method)](api/languages/python |
| thon_api.html#cudaq.SampleResult) | _api.html#cudaq.QuakeValue.slice) |
| -   [ScalarOperator (class in     | -   [SpinOperator (class in       |
|     cudaq.operato                 |     cudaq.operators.spin)         |
| rs)](api/languages/python_api.htm | ](api/languages/python_api.html#c |
| l#cudaq.operators.ScalarOperator) | udaq.operators.spin.SpinOperator) |
| -   [Schedule (class in           | -   [SpinOperatorElement (class   |
|     cudaq)](api/language          |     in                            |
| s/python_api.html#cudaq.Schedule) |     cudaq.operators.spin)](api/l  |
| -   [serialize()                  | anguages/python_api.html#cudaq.op |
|     (                             | erators.spin.SpinOperatorElement) |
| cudaq.operators.spin.SpinOperator | -   [SpinOperatorTerm (class in   |
|     method)](api/lang             |     cudaq.operators.spin)](ap     |
| uages/python_api.html#cudaq.opera | i/languages/python_api.html#cudaq |
| tors.spin.SpinOperator.serialize) | .operators.spin.SpinOperatorTerm) |
|     -   [(cuda                    | -   [SPSA (class in               |
| q.operators.spin.SpinOperatorTerm |     cudaq                         |
|         method)](api/language     | .optimizers)](api/languages/pytho |
| s/python_api.html#cudaq.operators | n_api.html#cudaq.optimizers.SPSA) |
| .spin.SpinOperatorTerm.serialize) | -   [squeeze() (in module         |
|     -   [(cudaq.SampleResult      |     cudaq.operators.cust          |
|         me                        | om)](api/languages/python_api.htm |
| thod)](api/languages/python_api.h | l#cudaq.operators.custom.squeeze) |
| tml#cudaq.SampleResult.serialize) | -   [State (class in              |
| -   [set_noise() (in module       |     cudaq)](api/langu             |
|     cudaq)](api/languages         | ages/python_api.html#cudaq.State) |
| /python_api.html#cudaq.set_noise) | -   [step_size                    |
| -   [set_random_seed() (in module |     (cudaq.optimizers.Adam        |
|     cudaq)](api/languages/pytho   |     propert                       |
| n_api.html#cudaq.set_random_seed) | y)](api/languages/python_api.html |
| -   [set_target() (in module      | #cudaq.optimizers.Adam.step_size) |
|     cudaq)](api/languages/        |     -   [(cudaq.optimizers.SGD    |
| python_api.html#cudaq.set_target) |         proper                    |
| -   [SGD (class in                | ty)](api/languages/python_api.htm |
|     cuda                          | l#cudaq.optimizers.SGD.step_size) |
| q.optimizers)](api/languages/pyth |     -   [(cudaq.optimizers.SPSA   |
| on_api.html#cudaq.optimizers.SGD) |         propert                   |
| -   [ShotAllocationStrategy       | y)](api/languages/python_api.html |
|     (class in                     | #cudaq.optimizers.SPSA.step_size) |
|     cudaq.ptsbe)]                 | -   [SuperOperator (class in      |
| (api/languages/python_api.html#cu |     cudaq)](api/languages/pyt     |
| daq.ptsbe.ShotAllocationStrategy) | hon_api.html#cudaq.SuperOperator) |
|                                   | -   [supports_compilation()       |
|                                   |     (cudaq.PyKernelDecorator      |
|                                   |     method)](api/langu            |
|                                   | ages/python_api.html#cudaq.PyKern |
|                                   | elDecorator.supports_compilation) |
+-----------------------------------+-----------------------------------+

## T {#T}

+-----------------------------------+-----------------------------------+
| -   [Target (class in             | -   [to_numpy()                   |
|     cudaq)](api/langua            |     (cudaq.ComplexMatrix          |
| ges/python_api.html#cudaq.Target) |     me                            |
| -   [target                       | thod)](api/languages/python_api.h |
|     (cudaq.ope                    | tml#cudaq.ComplexMatrix.to_numpy) |
| rators.boson.BosonOperatorElement | -   [to_sparse_matrix()           |
|     property)](api/languages/     |     (cu                           |
| python_api.html#cudaq.operators.b | daq.operators.boson.BosonOperator |
| oson.BosonOperatorElement.target) |     method)](api/languages/pyt    |
|     -   [(cudaq.operato           | hon_api.html#cudaq.operators.boso |
| rs.fermion.FermionOperatorElement | n.BosonOperator.to_sparse_matrix) |
|                                   |     -   [(cudaq.                  |
|     property)](api/languages/pyth | operators.boson.BosonOperatorTerm |
| on_api.html#cudaq.operators.fermi |                                   |
| on.FermionOperatorElement.target) |    method)](api/languages/python_ |
|     -   [(cudaq.o                 | api.html#cudaq.operators.boson.Bo |
| perators.spin.SpinOperatorElement | sonOperatorTerm.to_sparse_matrix) |
|         property)](api/language   |     -   [(cudaq.                  |
| s/python_api.html#cudaq.operators | operators.fermion.FermionOperator |
| .spin.SpinOperatorElement.target) |                                   |
| -   [Tensor (class in             |    method)](api/languages/python_ |
|     cudaq)](api/langua            | api.html#cudaq.operators.fermion. |
| ges/python_api.html#cudaq.Tensor) | FermionOperator.to_sparse_matrix) |
| -   [term_count                   |     -   [(cudaq.oper              |
|     (cu                           | ators.fermion.FermionOperatorTerm |
| daq.operators.boson.BosonOperator |         m                         |
|     property)](api/languag        | ethod)](api/languages/python_api. |
| es/python_api.html#cudaq.operator | html#cudaq.operators.fermion.Ferm |
| s.boson.BosonOperator.term_count) | ionOperatorTerm.to_sparse_matrix) |
|     -   [(cudaq.                  |     -   [(                        |
| operators.fermion.FermionOperator | cudaq.operators.spin.SpinOperator |
|                                   |         method)](api/languages/p  |
|        property)](api/languages/p | ython_api.html#cudaq.operators.sp |
| ython_api.html#cudaq.operators.fe | in.SpinOperator.to_sparse_matrix) |
| rmion.FermionOperator.term_count) |     -   [(cuda                    |
|     -                             | q.operators.spin.SpinOperatorTerm |
|  [(cudaq.operators.MatrixOperator |                                   |
|         property)](api/la         |      method)](api/languages/pytho |
| nguages/python_api.html#cudaq.ope | n_api.html#cudaq.operators.spin.S |
| rators.MatrixOperator.term_count) | pinOperatorTerm.to_sparse_matrix) |
|     -   [(                        | -   [to_string()                  |
| cudaq.operators.spin.SpinOperator |     (cudaq.ope                    |
|         property)](api/langu      | rators.boson.BosonOperatorElement |
| ages/python_api.html#cudaq.operat |     method)](api/languages/pyt    |
| ors.spin.SpinOperator.term_count) | hon_api.html#cudaq.operators.boso |
|     -   [(cuda                    | n.BosonOperatorElement.to_string) |
| q.operators.spin.SpinOperatorTerm |     -   [(cudaq.operato           |
|         property)](api/languages  | rs.fermion.FermionOperatorElement |
| /python_api.html#cudaq.operators. |                                   |
| spin.SpinOperatorTerm.term_count) |    method)](api/languages/python_ |
| -   [term_id                      | api.html#cudaq.operators.fermion. |
|     (cudaq.                       | FermionOperatorElement.to_string) |
| operators.boson.BosonOperatorTerm |     -   [(cuda                    |
|     property)](api/language       | q.operators.MatrixOperatorElement |
| s/python_api.html#cudaq.operators |         method)](api/language     |
| .boson.BosonOperatorTerm.term_id) | s/python_api.html#cudaq.operators |
|     -   [(cudaq.oper              | .MatrixOperatorElement.to_string) |
| ators.fermion.FermionOperatorTerm |     -   [(                        |
|                                   | cudaq.operators.spin.SpinOperator |
|       property)](api/languages/py |         method)](api/lang         |
| thon_api.html#cudaq.operators.fer | uages/python_api.html#cudaq.opera |
| mion.FermionOperatorTerm.term_id) | tors.spin.SpinOperator.to_string) |
|     -   [(c                       |     -   [(cudaq.o                 |
| udaq.operators.MatrixOperatorTerm | perators.spin.SpinOperatorElement |
|         property)](api/lan        |         method)](api/languages/p  |
| guages/python_api.html#cudaq.oper | ython_api.html#cudaq.operators.sp |
| ators.MatrixOperatorTerm.term_id) | in.SpinOperatorElement.to_string) |
|     -   [(cuda                    |     -   [(cuda                    |
| q.operators.spin.SpinOperatorTerm | q.operators.spin.SpinOperatorTerm |
|         property)](api/langua     |         method)](api/language     |
| ges/python_api.html#cudaq.operato | s/python_api.html#cudaq.operators |
| rs.spin.SpinOperatorTerm.term_id) | .spin.SpinOperatorTerm.to_string) |
| -   [to_dict() (cudaq.Resources   | -   [TraceInstruction (class in   |
|                                   |     cudaq.p                       |
|    method)](api/languages/python_ | tsbe)](api/languages/python_api.h |
| api.html#cudaq.Resources.to_dict) | tml#cudaq.ptsbe.TraceInstruction) |
| -   [to_json()                    | -   [TraceInstructionType (class  |
|     (                             |     in                            |
| cudaq.gradients.CentralDifference |     cudaq.ptsbe                   |
|     method)](api/la               | )](api/languages/python_api.html# |
| nguages/python_api.html#cudaq.gra | cudaq.ptsbe.TraceInstructionType) |
| dients.CentralDifference.to_json) | -   [translate() (in module       |
|     -   [(                        |     cudaq)](api/languages         |
| cudaq.gradients.ForwardDifference | /python_api.html#cudaq.translate) |
|         method)](api/la           | -   [trim()                       |
| nguages/python_api.html#cudaq.gra |     (cu                           |
| dients.ForwardDifference.to_json) | daq.operators.boson.BosonOperator |
|     -                             |     method)](api/l                |
|  [(cudaq.gradients.ParameterShift | anguages/python_api.html#cudaq.op |
|         method)](api              | erators.boson.BosonOperator.trim) |
| /languages/python_api.html#cudaq. |     -   [(cudaq.                  |
| gradients.ParameterShift.to_json) | operators.fermion.FermionOperator |
|     -   [(                        |         method)](api/langu        |
| cudaq.operators.spin.SpinOperator | ages/python_api.html#cudaq.operat |
|         method)](api/la           | ors.fermion.FermionOperator.trim) |
| nguages/python_api.html#cudaq.ope |     -                             |
| rators.spin.SpinOperator.to_json) |  [(cudaq.operators.MatrixOperator |
|     -   [(cuda                    |         method)](                 |
| q.operators.spin.SpinOperatorTerm | api/languages/python_api.html#cud |
|         method)](api/langua       | aq.operators.MatrixOperator.trim) |
| ges/python_api.html#cudaq.operato |     -   [(                        |
| rs.spin.SpinOperatorTerm.to_json) | cudaq.operators.spin.SpinOperator |
|     -   [(cudaq.optimizers.Adam   |         method)](api              |
|         met                       | /languages/python_api.html#cudaq. |
| hod)](api/languages/python_api.ht | operators.spin.SpinOperator.trim) |
| ml#cudaq.optimizers.Adam.to_json) | -   [two_qubit_gate_count         |
|     -   [(cudaq.optimizers.COBYLA |     (cudaq.Resources              |
|         metho                     |     property)](a                  |
| d)](api/languages/python_api.html | pi/languages/python_api.html#cuda |
| #cudaq.optimizers.COBYLA.to_json) | q.Resources.two_qubit_gate_count) |
|     -   [                         | -   [type                         |
| (cudaq.optimizers.GradientDescent |     (c                            |
|         method)](api/l            | udaq.ptsbe.ShotAllocationStrategy |
| anguages/python_api.html#cudaq.op |     property)](api/               |
| timizers.GradientDescent.to_json) | languages/python_api.html#cudaq.p |
|     -   [(cudaq.optimizers.LBFGS  | tsbe.ShotAllocationStrategy.type) |
|         meth                      | -   [type_to_str()                |
| od)](api/languages/python_api.htm |     (cudaq.PyKernelDecorator      |
| l#cudaq.optimizers.LBFGS.to_json) |     static                        |
|                                   |     method)](                     |
| -   [(cudaq.optimizers.NelderMead | api/languages/python_api.html#cud |
|         method)](                 | aq.PyKernelDecorator.type_to_str) |
| api/languages/python_api.html#cud |                                   |
| aq.optimizers.NelderMead.to_json) |                                   |
|     -   [(cudaq.optimizers.SGD    |                                   |
|         me                        |                                   |
| thod)](api/languages/python_api.h |                                   |
| tml#cudaq.optimizers.SGD.to_json) |                                   |
|     -   [(cudaq.optimizers.SPSA   |                                   |
|         met                       |                                   |
| hod)](api/languages/python_api.ht |                                   |
| ml#cudaq.optimizers.SPSA.to_json) |                                   |
|     -   [(cudaq.PyKernelDecorator |                                   |
|         metho                     |                                   |
| d)](api/languages/python_api.html |                                   |
| #cudaq.PyKernelDecorator.to_json) |                                   |
| -   [to_matrix()                  |                                   |
|     (cu                           |                                   |
| daq.operators.boson.BosonOperator |                                   |
|     method)](api/langua           |                                   |
| ges/python_api.html#cudaq.operato |                                   |
| rs.boson.BosonOperator.to_matrix) |                                   |
|     -   [(cudaq.ope               |                                   |
| rators.boson.BosonOperatorElement |                                   |
|                                   |                                   |
|        method)](api/languages/pyt |                                   |
| hon_api.html#cudaq.operators.boso |                                   |
| n.BosonOperatorElement.to_matrix) |                                   |
|     -   [(cudaq.                  |                                   |
| operators.boson.BosonOperatorTerm |                                   |
|         method)](api/languages/   |                                   |
| python_api.html#cudaq.operators.b |                                   |
| oson.BosonOperatorTerm.to_matrix) |                                   |
|     -   [(cudaq.                  |                                   |
| operators.fermion.FermionOperator |                                   |
|         method)](api/languages/   |                                   |
| python_api.html#cudaq.operators.f |                                   |
| ermion.FermionOperator.to_matrix) |                                   |
|     -   [(cudaq.operato           |                                   |
| rs.fermion.FermionOperatorElement |                                   |
|                                   |                                   |
|    method)](api/languages/python_ |                                   |
| api.html#cudaq.operators.fermion. |                                   |
| FermionOperatorElement.to_matrix) |                                   |
|     -   [(cudaq.oper              |                                   |
| ators.fermion.FermionOperatorTerm |                                   |
|                                   |                                   |
|       method)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.fermi |                                   |
| on.FermionOperatorTerm.to_matrix) |                                   |
|     -                             |                                   |
|  [(cudaq.operators.MatrixOperator |                                   |
|         method)](api/l            |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| erators.MatrixOperator.to_matrix) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.MatrixOperatorElement |                                   |
|         method)](api/language     |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .MatrixOperatorElement.to_matrix) |                                   |
|     -   [(c                       |                                   |
| udaq.operators.MatrixOperatorTerm |                                   |
|         method)](api/langu        |                                   |
| ages/python_api.html#cudaq.operat |                                   |
| ors.MatrixOperatorTerm.to_matrix) |                                   |
|     -                             |                                   |
|  [(cudaq.operators.ScalarOperator |                                   |
|         method)](api/l            |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| erators.ScalarOperator.to_matrix) |                                   |
|     -   [(                        |                                   |
| cudaq.operators.spin.SpinOperator |                                   |
|         method)](api/lang         |                                   |
| uages/python_api.html#cudaq.opera |                                   |
| tors.spin.SpinOperator.to_matrix) |                                   |
|     -   [(cudaq.o                 |                                   |
| perators.spin.SpinOperatorElement |                                   |
|         method)](api/languages/p  |                                   |
| ython_api.html#cudaq.operators.sp |                                   |
| in.SpinOperatorElement.to_matrix) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         method)](api/language     |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .spin.SpinOperatorTerm.to_matrix) |                                   |
+-----------------------------------+-----------------------------------+

## U {#U}

+-----------------------------------------------------------------------+
| -   [unregister_set_target_callback() (in module                      |
|     cudaq)                                                            |
| ](api/languages/python_api.html#cudaq.unregister_set_target_callback) |
| -   [unset_noise() (in module                                         |
|     cudaq)](api/languages/python_api.html#cudaq.unset_noise)          |
| -   [upper_bounds (cudaq.optimizers.Adam                              |
|     propert                                                           |
| y)](api/languages/python_api.html#cudaq.optimizers.Adam.upper_bounds) |
|     -   [(cudaq.optimizers.COBYLA                                     |
|         property)                                                     |
| ](api/languages/python_api.html#cudaq.optimizers.COBYLA.upper_bounds) |
|     -   [(cudaq.optimizers.GradientDescent                            |
|         property)](api/lan                                            |
| guages/python_api.html#cudaq.optimizers.GradientDescent.upper_bounds) |
|     -   [(cudaq.optimizers.LBFGS                                      |
|         property                                                      |
| )](api/languages/python_api.html#cudaq.optimizers.LBFGS.upper_bounds) |
|     -   [(cudaq.optimizers.NelderMead                                 |
|         property)](ap                                                 |
| i/languages/python_api.html#cudaq.optimizers.NelderMead.upper_bounds) |
|     -   [(cudaq.optimizers.SGD                                        |
|         proper                                                        |
| ty)](api/languages/python_api.html#cudaq.optimizers.SGD.upper_bounds) |
|     -   [(cudaq.optimizers.SPSA                                       |
|         propert                                                       |
| y)](api/languages/python_api.html#cudaq.optimizers.SPSA.upper_bounds) |
+-----------------------------------------------------------------------+

## V {#V}

+-----------------------------------+-----------------------------------+
| -   [values() (cudaq.SampleResult | -   [vqe() (in module             |
|                                   |     cudaq)](api/lan               |
|  method)](api/languages/python_ap | guages/python_api.html#cudaq.vqe) |
| i.html#cudaq.SampleResult.values) |                                   |
+-----------------------------------+-----------------------------------+

## W {#W}

+-----------------------------------------------------------------------+
| -   [weight (cudaq.ptsbe.KrausTrajectory                              |
|     propert                                                           |
| y)](api/languages/python_api.html#cudaq.ptsbe.KrausTrajectory.weight) |
+-----------------------------------------------------------------------+

## X {#X}

+-----------------------------------+-----------------------------------+
| -   [x() (in module               | -   [XError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.x) | ges/python_api.html#cudaq.XError) |
+-----------------------------------+-----------------------------------+

## Y {#Y}

+-----------------------------------+-----------------------------------+
| -   [y() (in module               | -   [YError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.y) | ges/python_api.html#cudaq.YError) |
+-----------------------------------+-----------------------------------+

## Z {#Z}

+-----------------------------------+-----------------------------------+
| -   [z() (in module               | -   [ZError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.z) | ges/python_api.html#cudaq.ZError) |
+-----------------------------------+-----------------------------------+
:::
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
