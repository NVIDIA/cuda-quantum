::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../index.html){.icon .icon-home}

::: version
pr-4418
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
-   [Examples](examples.html){.reference .internal}
    -   [Introduction](introduction.html){.reference .internal}
    -   [Building Kernels](building_kernels.html){.reference .internal}
        -   [Defining
            Kernels](building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum Operations](quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring Kernels](measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
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
    -   [Executing Kernels](executing_kernels.html){.reference
        .internal}
        -   [Sample](executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](executing_kernels.html#run){.reference .internal}
            -   [Return Custom Data
                Types](executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get State](executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](expectation_values.html){.reference .internal}
        -   [Parallelizing across Multiple
            Processors](expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU Workflows](multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends ([`remote-mqpu`{.code .docutils
            .literal
            .notranslate}]{.pre})](multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
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
        Execution](ptsbe.html){.reference .internal}
        -   [Conceptual
            Overview](ptsbe.html#conceptual-overview){.reference
            .internal}
        -   [When to Use PTSBE](ptsbe.html#when-to-use-ptsbe){.reference
            .internal}
        -   [Quick Start](ptsbe.html#quick-start){.reference .internal}
        -   [Usage Tutorial](ptsbe.html#usage-tutorial){.reference
            .internal}
            -   [Controlling the Number of
                Trajectories](ptsbe.html#controlling-the-number-of-trajectories){.reference
                .internal}
            -   [Choosing a Trajectory Sampling
                Strategy](ptsbe.html#choosing-a-trajectory-sampling-strategy){.reference
                .internal}
            -   [Shot Allocation
                Strategies](ptsbe.html#shot-allocation-strategies){.reference
                .internal}
            -   [Inspecting Execution
                Data](ptsbe.html#inspecting-execution-data){.reference
                .internal}
    -   [Constructing Operators](operators.html){.reference .internal}
        -   [Constructing Spin
            Operators](operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](../../examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](../../examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](hardware_providers.html){.reference .internal}
        -   [Amazon
            Braket](hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](hardware_providers.html#ionq){.reference .internal}
        -   [IQM](hardware_providers.html#iqm){.reference .internal}
        -   [OQC](hardware_providers.html#oqc){.reference .internal}
        -   [ORCA
            Computing](hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum Circuits,
            Inc.](hardware_providers.html#quantum-circuits-inc){.reference
            .internal}
        -   [Quantum
            Machines](hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](hardware_providers.html#quera-computing){.reference
            .internal}
        -   [Scaleway](hardware_providers.html#scaleway){.reference
            .internal}
        -   [TII](hardware_providers.html#tii){.reference .internal}
    -   [When to Use sample vs. run](sample_vs_run.html){.reference
        .internal}
        -   [Introduction](sample_vs_run.html#introduction){.reference
            .internal}
        -   [Usage
            Guidelines](sample_vs_run.html#usage-guidelines){.reference
            .internal}
        -   [What Is Supported with [`sample`{.docutils .literal
            .notranslate}]{.pre}](sample_vs_run.html#what-is-supported-with-sample){.reference
            .internal}
        -   [What Is Not Supported with [`sample`{.docutils .literal
            .notranslate}]{.pre}](sample_vs_run.html#what-is-not-supported-with-sample){.reference
            .internal}
        -   [How to
            Migrate](sample_vs_run.html#how-to-migrate){.reference
            .internal}
            -   [Step 1: Add a return type to the
                kernel](sample_vs_run.html#step-1-add-a-return-type-to-the-kernel){.reference
                .internal}
            -   [Step 2: Replace [`sample`{.docutils .literal
                .notranslate}]{.pre} with [`run`{.docutils .literal
                .notranslate}]{.pre}](sample_vs_run.html#step-2-replace-sample-with-run){.reference
                .internal}
            -   [Step 3: Update result
                processing](sample_vs_run.html#step-3-update-result-processing){.reference
                .internal}
        -   [Migration
            Examples](sample_vs_run.html#migration-examples){.reference
            .internal}
            -   [Example 1: Simple conditional
                logic](sample_vs_run.html#example-1-simple-conditional-logic){.reference
                .internal}
            -   [Example 2: Returning multiple measurement
                results](sample_vs_run.html#example-2-returning-multiple-measurement-results){.reference
                .internal}
            -   [Example 3: Quantum
                teleportation](sample_vs_run.html#example-3-quantum-teleportation){.reference
                .internal}
        -   [Additional
            Notes](sample_vs_run.html#additional-notes){.reference
            .internal}
    -   [Dynamics Examples](#){.current .reference .internal}
        -   [Python Examples (Jupyter
            Notebooks)](#python-examples-jupyter-notebooks){.reference
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
        -   [C++ Examples](#c-examples){.reference .internal}
            -   [Introduction: Single Qubit
                Dynamics](#introduction-single-qubit-dynamics){.reference
                .internal}
            -   [Introduction: Cavity QED (Jaynes-Cummings
                Model)](#introduction-cavity-qed-jaynes-cummings-model){.reference
                .internal}
            -   [Superconducting Qubits: Cross-Resonance
                Gate](#superconducting-qubits-cross-resonance-gate){.reference
                .internal}
            -   [Spin Qubits: Heisenberg Spin
                Chain](#spin-qubits-heisenberg-spin-chain){.reference
                .internal}
            -   [Control: Driven
                Qubit](#control-driven-qubit){.reference .internal}
            -   [State Batching](#state-batching){.reference .internal}
            -   [Numerical
                Integrators](#numerical-integrators){.reference
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
    -   [Installation](../realtime/installation.html){.reference
        .internal}
        -   [Prerequisites](../realtime/installation.html#prerequisites){.reference
            .internal}
        -   [Setup](../realtime/installation.html#setup){.reference
            .internal}
        -   [Latency
            Measurement](../realtime/installation.html#latency-measurement){.reference
            .internal}
    -   [Host API](../realtime/host.html){.reference .internal}
        -   [What is HSB?](../realtime/host.html#what-is-hsb){.reference
            .internal}
        -   [Transport
            Mechanisms](../realtime/host.html#transport-mechanisms){.reference
            .internal}
            -   [Supported Transport
                Options](../realtime/host.html#supported-transport-options){.reference
                .internal}
        -   [The 3-Kernel Architecture (HSB Example)
            {#three-kernel-architecture}](../realtime/host.html#the-3-kernel-architecture-hsb-example-three-kernel-architecture){.reference
            .internal}
            -   [Data Flow
                Summary](../realtime/host.html#data-flow-summary){.reference
                .internal}
            -   [Why 3
                Kernels?](../realtime/host.html#why-3-kernels){.reference
                .internal}
        -   [Unified Dispatch
            Mode](../realtime/host.html#unified-dispatch-mode){.reference
            .internal}
            -   [Architecture](../realtime/host.html#architecture){.reference
                .internal}
            -   [Transport-Agnostic
                Design](../realtime/host.html#transport-agnostic-design){.reference
                .internal}
            -   [When to Use Which
                Mode](../realtime/host.html#when-to-use-which-mode){.reference
                .internal}
            -   [Host API
                Extensions](../realtime/host.html#host-api-extensions){.reference
                .internal}
            -   [Wiring Example (Unified Mode with
                HSB)](../realtime/host.html#wiring-example-unified-mode-with-hsb){.reference
                .internal}
        -   [What This API Does (In One
            Paragraph)](../realtime/host.html#what-this-api-does-in-one-paragraph){.reference
            .internal}
        -   [Scope](../realtime/host.html#scope){.reference .internal}
        -   [Terms and
            Components](../realtime/host.html#terms-and-components){.reference
            .internal}
        -   [Schema Data
            Structures](../realtime/host.html#schema-data-structures){.reference
            .internal}
            -   [Type
                Descriptors](../realtime/host.html#type-descriptors){.reference
                .internal}
            -   [Handler
                Schema](../realtime/host.html#handler-schema){.reference
                .internal}
        -   [RPC Messaging
            Protocol](../realtime/host.html#rpc-messaging-protocol){.reference
            .internal}
        -   [Host API
            Overview](../realtime/host.html#host-api-overview){.reference
            .internal}
        -   [Manager and Dispatcher
            Topology](../realtime/host.html#manager-and-dispatcher-topology){.reference
            .internal}
        -   [Host API
            Functions](../realtime/host.html#host-api-functions){.reference
            .internal}
            -   [Occupancy Query and Eager Module
                Loading](../realtime/host.html#occupancy-query-and-eager-module-loading){.reference
                .internal}
            -   [Graph-Based Dispatch
                Functions](../realtime/host.html#graph-based-dispatch-functions){.reference
                .internal}
            -   [Kernel Launch Helper
                Functions](../realtime/host.html#kernel-launch-helper-functions){.reference
                .internal}
        -   [Memory Layout and Ring Buffer
            Wiring](../realtime/host.html#memory-layout-and-ring-buffer-wiring){.reference
            .internal}
        -   [Step-by-Step: Wiring the Host API
            (Minimal)](../realtime/host.html#step-by-step-wiring-the-host-api-minimal){.reference
            .internal}
        -   [Device Handler and Function
            ID](../realtime/host.html#device-handler-and-function-id){.reference
            .internal}
            -   [Multi-Argument Handler
                Example](../realtime/host.html#multi-argument-handler-example){.reference
                .internal}
        -   [CUDA Graph Dispatch
            Mode](../realtime/host.html#cuda-graph-dispatch-mode){.reference
            .internal}
            -   [Requirements](../realtime/host.html#requirements){.reference
                .internal}
            -   [Graph-Based Dispatch
                API](../realtime/host.html#graph-based-dispatch-api){.reference
                .internal}
            -   [Graph Handler Setup
                Example](../realtime/host.html#graph-handler-setup-example){.reference
                .internal}
            -   [Graph Capture and
                Instantiation](../realtime/host.html#graph-capture-and-instantiation){.reference
                .internal}
            -   [When to Use Graph
                Dispatch](../realtime/host.html#when-to-use-graph-dispatch){.reference
                .internal}
            -   [Graph vs Device Call
                Dispatch](../realtime/host.html#graph-vs-device-call-dispatch){.reference
                .internal}
        -   [Building and Sending an RPC
            Message](../realtime/host.html#building-and-sending-an-rpc-message){.reference
            .internal}
        -   [Reading the
            Response](../realtime/host.html#reading-the-response){.reference
            .internal}
        -   [Schema-Driven Argument
            Parsing](../realtime/host.html#schema-driven-argument-parsing){.reference
            .internal}
        -   [HSB 3-Kernel Workflow
            (Primary)](../realtime/host.html#hsb-3-kernel-workflow-primary){.reference
            .internal}
        -   [NIC-Free Testing (No HSB / No
            ConnectX-7)](../realtime/host.html#nic-free-testing-no-hsb-no-connectx-7){.reference
            .internal}
        -   [Troubleshooting](../realtime/host.html#troubleshooting){.reference
            .internal}
    -   [Messaging Protocol](../realtime/protocol.html){.reference
        .internal}
        -   [Scope](../realtime/protocol.html#scope){.reference
            .internal}
        -   [RPC Header /
            Response](../realtime/protocol.html#rpc-header-response){.reference
            .internal}
        -   [Request ID
            Semantics](../realtime/protocol.html#request-id-semantics){.reference
            .internal}
        -   [[`PTP`{.docutils .literal .notranslate}]{.pre} Timestamp
            Semantics](../realtime/protocol.html#ptp-timestamp-semantics){.reference
            .internal}
        -   [Function ID
            Semantics](../realtime/protocol.html#function-id-semantics){.reference
            .internal}
        -   [Schema and Payload
            Interpretation](../realtime/protocol.html#schema-and-payload-interpretation){.reference
            .internal}
            -   [Type
                System](../realtime/protocol.html#type-system){.reference
                .internal}
        -   [Payload
            Encoding](../realtime/protocol.html#payload-encoding){.reference
            .internal}
            -   [Single-Argument
                Payloads](../realtime/protocol.html#single-argument-payloads){.reference
                .internal}
            -   [Multi-Argument
                Payloads](../realtime/protocol.html#multi-argument-payloads){.reference
                .internal}
            -   [Size
                Constraints](../realtime/protocol.html#size-constraints){.reference
                .internal}
            -   [Encoding
                Examples](../realtime/protocol.html#encoding-examples){.reference
                .internal}
            -   [Bit-Packed Data
                Encoding](../realtime/protocol.html#bit-packed-data-encoding){.reference
                .internal}
            -   [Multi-Bit Measurement
                Encoding](../realtime/protocol.html#multi-bit-measurement-encoding){.reference
                .internal}
        -   [Response
            Encoding](../realtime/protocol.html#response-encoding){.reference
            .internal}
            -   [Single-Result
                Response](../realtime/protocol.html#single-result-response){.reference
                .internal}
            -   [Multi-Result
                Response](../realtime/protocol.html#multi-result-response){.reference
                .internal}
            -   [Status
                Codes](../realtime/protocol.html#status-codes){.reference
                .internal}
        -   [QEC-Specific Usage
            Example](../realtime/protocol.html#qec-specific-usage-example){.reference
            .internal}
            -   [QEC
                Terminology](../realtime/protocol.html#qec-terminology){.reference
                .internal}
            -   [QEC Decoder
                Handler](../realtime/protocol.html#qec-decoder-handler){.reference
                .internal}
            -   [Decoding
                Rounds](../realtime/protocol.html#decoding-rounds){.reference
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
-   [CUDA-Q by Example](examples.html)
-   CUDA-Q Dynamics
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](sample_vs_run.html "When to Use sample vs. run"){.btn
.btn-neutral .float-left accesskey="p"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](../../examples/python/dynamics/dynamics_intro_1.html "Introduction to CUDA-Q Dynamics (Jaynes-Cummings Model)"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#cuda-q-dynamics .section}
# CUDA-Q Dynamics[¶](#cuda-q-dynamics "Permalink to this heading"){.headerlink}

This section contains examples for CUDA-Q Dynamics in both Python and
C++. For a conceptual overview of the [`evolve`{.docutils .literal
.notranslate}]{.pre} API, see the [[Dynamics Simulation]{.std
.std-ref}](../dynamics.html#dynamics){.reference .internal} page.

::: {#python-examples-jupyter-notebooks .section}
## Python Examples (Jupyter Notebooks)[¶](#python-examples-jupyter-notebooks "Permalink to this heading"){.headerlink}

The notebooks below contain groups of examples using CUDA-Q Dynamics.
The first two notebooks provide an introduction to CUDA-Q Dynamics
appropriate for new users.

Download the notebooks below
[here](https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/python/dynamics){.reference
.external}.

::: {.toctree-wrapper .compound}
-   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
    Model)](../../examples/python/dynamics/dynamics_intro_1.html){.reference
    .internal}
    -   [Why dynamics simulations vs. circuit
        simulations?](../../examples/python/dynamics/dynamics_intro_1.html#Why-dynamics-simulations-vs.-circuit-simulations?){.reference
        .internal}
    -   [Functionality](../../examples/python/dynamics/dynamics_intro_1.html#Functionality){.reference
        .internal}
    -   [Performance](../../examples/python/dynamics/dynamics_intro_1.html#Performance){.reference
        .internal}
    -   [Section 1 - Simulating the Jaynes-Cummings
        Hamiltonian](../../examples/python/dynamics/dynamics_intro_1.html#Section-1---Simulating-the-Jaynes-Cummings-Hamiltonian){.reference
        .internal}
    -   [Exercise 1 - Simulating a many-photon Jaynes-Cummings
        Hamiltonian](../../examples/python/dynamics/dynamics_intro_1.html#Exercise-1---Simulating-a-many-photon-Jaynes-Cummings-Hamiltonian){.reference
        .internal}
    -   [Section 2 - Simulating open quantum systems with the
        [`collapse_operators`{.docutils .literal
        .notranslate}]{.pre}](../../examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
        .internal}
    -   [Exercise 2 - Adding additional jump operators [\\(L_i\\)]{.math
        .notranslate
        .nohighlight}](../../examples/python/dynamics/dynamics_intro_1.html#Exercise-2---Adding-additional-jump-operators-L_i){.reference
        .internal}
    -   [Section 3 - Many qubits coupled to the
        resonator](../../examples/python/dynamics/dynamics_intro_1.html#Section-3---Many-qubits-coupled-to-the-resonator){.reference
        .internal}
-   [Introduction to CUDA-Q Dynamics (Time Dependent
    Hamiltonians)](../../examples/python/dynamics/dynamics_intro_2.html){.reference
    .internal}
    -   [The Landau-Zener
        model](../../examples/python/dynamics/dynamics_intro_2.html#The-Landau-Zener-model){.reference
        .internal}
    -   [Section 1 - Implementing time dependent
        terms](../../examples/python/dynamics/dynamics_intro_2.html#Section-1---Implementing-time-dependent-terms){.reference
        .internal}
    -   [Section 2 - Implementing custom
        operators](../../examples/python/dynamics/dynamics_intro_2.html#Section-2---Implementing-custom-operators){.reference
        .internal}
    -   [Section 3 - Heisenberg Model with a time-varying magnetic
        field](../../examples/python/dynamics/dynamics_intro_2.html#Section-3---Heisenberg-Model-with-a-time-varying-magnetic-field){.reference
        .internal}
    -   [Exercise 1 - Define a time-varying magnetic
        field](../../examples/python/dynamics/dynamics_intro_2.html#Exercise-1---Define-a-time-varying-magnetic-field){.reference
        .internal}
    -   [Exercise 2
        (Optional)](../../examples/python/dynamics/dynamics_intro_2.html#Exercise-2-(Optional)){.reference
        .internal}
-   [Superconducting
    Qubits](../../examples/python/dynamics/superconducting.html){.reference
    .internal}
    -   [Cavity
        QED](../../examples/python/dynamics/superconducting.html#Cavity-QED){.reference
        .internal}
    -   [Cross
        Resonance](../../examples/python/dynamics/superconducting.html#Cross-Resonance){.reference
        .internal}
    -   [Transmon
        Resonator](../../examples/python/dynamics/superconducting.html#Transmon-Resonator){.reference
        .internal}
-   [Spin
    Qubits](../../examples/python/dynamics/spinqubits.html){.reference
    .internal}
    -   [Silicon Spin
        Qubit](../../examples/python/dynamics/spinqubits.html#Silicon-Spin-Qubit){.reference
        .internal}
    -   [Heisenberg
        Model](../../examples/python/dynamics/spinqubits.html#Heisenberg-Model){.reference
        .internal}
-   [Trapped Ion
    Qubits](../../examples/python/dynamics/iontrap.html){.reference
    .internal}
    -   [GHZ
        state](../../examples/python/dynamics/iontrap.html#GHZ-state){.reference
        .internal}
-   [Control](../../examples/python/dynamics/control.html){.reference
    .internal}
    -   [Gate
        Calibration](../../examples/python/dynamics/control.html#Gate-Calibration){.reference
        .internal}
    -   [Pulse](../../examples/python/dynamics/control.html#Pulse){.reference
        .internal}
    -   [Qubit
        Control](../../examples/python/dynamics/control.html#Qubit-Control){.reference
        .internal}
    -   [Qubit
        Dynamics](../../examples/python/dynamics/control.html#Qubit-Dynamics){.reference
        .internal}
    -   [Landau-Zenner](../../examples/python/dynamics/control.html#Landau-Zenner){.reference
        .internal}
:::
:::

::: {#c-examples .section}
## C++ Examples[¶](#c-examples "Permalink to this heading"){.headerlink}

The following C++ examples demonstrate core CUDA-Q Dynamics
capabilities. Each example can be compiled and run with:

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target dynamics <example>.cpp -o a.out && ./a.out
:::
:::

The source files are available in the [CUDA-Q
repository](https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/cpp/dynamics){.reference
.external}.

::: {#introduction-single-qubit-dynamics .section}
### Introduction: Single Qubit Dynamics[¶](#introduction-single-qubit-dynamics "Permalink to this heading"){.headerlink}

This example demonstrates the basic workflow for time-evolving a single
qubit under a transverse field Hamiltonian, with and without dissipation
(collapse operators).

::: {.highlight-cpp .notranslate}
::: highlight
    /*******************************************************************************
     * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
     * All rights reserved.                                                        *
     *                                                                             *
     * This source code and the accompanying materials are made available under    *
     * the terms of the Apache License 2.0 which accompanies this distribution.    *
     ******************************************************************************/

    // Compile and run with:
    // ```
    // nvq++ --target dynamics qubit_dynamics.cpp -o a.out && ./a.out
    // ```

    #include "cudaq/algorithms/evolve.h"
    #include "cudaq/algorithms/integrator.h"
    #include "cudaq/operators.h"
    #include "export_csv_helper.h"
    #include <cudaq.h>

    int main() {
      // Qubit `hamiltonian`: 2 * pi * 0.1 * sigma_x
      // Physically, this represents a qubit (a two-level system) driven by a weak
      // transverse field along the x-axis.
      auto hamiltonian = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);

      // Dimensions: one subsystem of dimension 2 (a two-level system).
      const cudaq::dimension_map dimensions = {{0, 2}};

      // Initial state: ground state
      std::vector<std::complex<double>> initial_state_vec = {1.0, 0.0};
      auto psi0 = cudaq::state::from_data(initial_state_vec);

      // Create a schedule of time steps from 0 to 10 with 101 points
      std::vector<double> steps = cudaq::linspace(0.0, 10.0, 101);
      cudaq::schedule schedule(steps);

      // Runge-`Kutta` integrator with a time step of 0.01 and order 4
      cudaq::integrators::runge_kutta integrator(4, 0.01);

      // Run the simulation without collapse operators (ideal evolution)
      auto evolve_result =
          cudaq::evolve(hamiltonian, dimensions, schedule, psi0, integrator, {},
                        {cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
                        cudaq::IntermediateResultSave::ExpectationValue);

      constexpr double decay_rate = 0.05;
      auto collapse_operator = std::sqrt(decay_rate) * cudaq::spin_op::x(0);

      // Evolve with collapse operators
      cudaq::evolve_result evolve_result_decay = cudaq::evolve(
          hamiltonian, dimensions, schedule, psi0, integrator, {collapse_operator},
          {cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
          cudaq::IntermediateResultSave::ExpectationValue);

      // Lambda to extract expectation values for a given observable index
      auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
        std::vector<double> expectations;

        auto all_exps = result.expectation_values.value();
        for (auto exp_vals : all_exps) {
          expectations.push_back((double)exp_vals[idx]);
        }
        return expectations;
      };

      auto ideal_result0 = get_expectation(0, evolve_result);
      auto ideal_result1 = get_expectation(1, evolve_result);
      auto decay_result0 = get_expectation(0, evolve_result_decay);
      auto decay_result1 = get_expectation(1, evolve_result_decay);

      export_csv("qubit_dynamics_ideal_result.csv", "time", steps, "sigma_y",
                 ideal_result0, "sigma_z", ideal_result1);
      export_csv("qubit_dynamics_decay_result.csv", "time", steps, "sigma_y",
                 decay_result0, "sigma_z", decay_result1);

      std::cout << "Results exported to qubit_dynamics_ideal_result.csv and "
                   "qubit_dynamics_decay_result.csv"
                << std::endl;

      return 0;
    }
:::
:::
:::

::: {#introduction-cavity-qed-jaynes-cummings-model .section}
### Introduction: Cavity QED (Jaynes-Cummings Model)[¶](#introduction-cavity-qed-jaynes-cummings-model "Permalink to this heading"){.headerlink}

This example simulates a two-level atom coupled to a single-mode optical
cavity, known as the Jaynes-Cummings model. It demonstrates how to set
up composite quantum systems with different subsystem dimensions.

::: {.highlight-cpp .notranslate}
::: highlight
    /*******************************************************************************
     * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
     * All rights reserved.                                                        *
     *                                                                             *
     * This source code and the accompanying materials are made available under    *
     * the terms of the Apache License 2.0 which accompanies this distribution.    *
     ******************************************************************************/

    // Compile and run with:
    // ```
    // nvq++ --target dynamics cavity_qed.cpp -o a.out && ./a.out
    // ```

    #include "cudaq/algorithms/evolve.h"
    #include "cudaq/algorithms/integrator.h"
    #include "cudaq/operators.h"
    #include "export_csv_helper.h"
    #include <cudaq.h>

    int main() {

      // Dimension of our composite quantum system:
      // subsystem 0 (atom) has 2 levels (ground and excited states).
      // subsystem 1 (cavity) has 10 levels (Fock states, representing photon number
      // states).
      cudaq::dimension_map dimensions{{0, 2}, {1, 10}};

      // For the cavity subsystem 1
      // We create the annihilation (a) and creation (a+) operators.
      // These operators lower and raise the photon number, respectively.
      auto a = cudaq::boson_op::annihilate(1);
      auto a_dag = cudaq::boson_op::create(1);

      // For the atom subsystem 0
      // We create the annihilation (`sm`) and creation (`sm_dag`) operators.
      // These operators lower and raise the excitation number, respectively.
      auto sm = cudaq::boson_op::annihilate(0);
      auto sm_dag = cudaq::boson_op::create(0);

      // Number operators
      // These operators count the number of excitations.
      // For the atom (`subsytem` 0) and the cavity (`subsystem` 1) they give the
      // population in each subsystem.
      auto atom_occ_op = cudaq::matrix_op::number(0);
      auto cavity_occ_op = cudaq::matrix_op::number(1);

      // Hamiltonian
      // The `hamiltonian` models the dynamics of the atom-cavity (cavity QED)
      // system It has 3 parts:
      // 1. Cavity energy: 2 * pi * photon_number_operator -> energy proportional to
      // the number of photons.
      // 2. Atomic energy: 2 * pi * atom_number_operator -> energy proportional to
      // the atomic excitation.
      // 3. Atomic-cavity interaction: 2 * pi * 0.25 * (`sm` * a_dag + `sm_dag` * a)
      // -> represents the exchange of energy between the atom and the cavity. It is
      // analogous to the Jaynes-Cummings model in cavity QED.
      auto hamiltonian = (2 * M_PI * cavity_occ_op) + (2 * M_PI * atom_occ_op) +
                         (2 * M_PI * 0.25 * (sm * a_dag + sm_dag * a));

      // Build the initial state
      // Atom (sub-system 0) in ground state.
      // Cavity (sub-system 1) has 5 photons (Fock space).
      // The overall Hilbert space is 2 * 10 = 20.
      const int num_photons = 5;
      std::vector<std::complex<double>> initial_state_vec(20, 0.0);
      // The index is chosen such that the atom is in the ground state while the
      // cavity is in the Fock state with 5 photons.
      initial_state_vec[dimensions[0] * num_photons] = 1;

      // Define a time evolution schedule
      // We define a time grid from 0 to 10 (in arbitrary time units) with 201 time
      // steps. This schedule is used by the integrator to simulate the dynamics.
      const int num_steps = 201;
      std::vector<double> steps = cudaq::linspace(0.0, 10.0, num_steps);
      cudaq::schedule schedule(steps);

      // Create a CUDA quantum state
      // The initial state is converted into a quantum state object for evolution.
      auto rho0 = cudaq::state::from_data(initial_state_vec);

      // Numerical integrator
      // Here we choose a Runge-`Kutta` method for time evolution.
      // `dt` defines the time step for the numerical integration, and order 4
      // indicates a 4`th` order method.
      cudaq::integrators::runge_kutta integrator(4, 0.01);

      // Evolve without collapse operators
      // This evolution is ideal (closed system) dynamics governed solely by the
      // Hamiltonian. The expectation values of the observables (cavity photon
      // number and atom excitation probability) are recorded.
      cudaq::evolve_result evolve_result =
          cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator, {},
                        {cavity_occ_op, atom_occ_op},
                        cudaq::IntermediateResultSave::ExpectationValue);

      // Adding dissipation
      // To simulate a realistic scenario, we introduce decay (dissipation).
      // Here, the collapse operator represents photon loss from the cavity.
      constexpr double decay_rate = 0.1;
      auto collapse_operator = std::sqrt(decay_rate) * a;
      // Evolve with the collapse operator to incorporate the effect of decay.
      cudaq::evolve_result evolve_result_decay =
          cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator,
                        {collapse_operator}, {cavity_occ_op, atom_occ_op},
                        cudaq::IntermediateResultSave::ExpectationValue);

      // Lambda to extract expectation values for a given observable index
      // Here, index 0 corresponds to the cavity photon number and index 1
      // corresponds to the atom excitation probability.
      auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
        std::vector<double> expectations;

        auto all_exps = result.expectation_values.value();
        for (auto exp_vals : all_exps) {
          expectations.push_back((double)exp_vals[idx]);
        }
        return expectations;
      };

      // Retrieve expectation values from both the ideal and decaying `evolutions`.
      auto ideal_result0 = get_expectation(0, evolve_result);
      auto ideal_result1 = get_expectation(1, evolve_result);
      auto decay_result0 = get_expectation(0, evolve_result_decay);
      auto decay_result1 = get_expectation(1, evolve_result_decay);

      // Export the results to `CSV` files
      // "cavity_`qed`_ideal_result.`csv`" contains the ideal evolution results.
      // "cavity_`qed`_decay_result.`csv`" contains the evolution results with
      // cavity decay.
      export_csv("cavity_qed_ideal_result.csv", "time", steps,
                 "cavity_photon_number", ideal_result0,
                 "atom_excitation_probability", ideal_result1);
      export_csv("cavity_qed_decay_result.csv", "time", steps,
                 "cavity_photon_number", decay_result0,
                 "atom_excitation_probability", decay_result1);

      std::cout << "Simulation complete. The results are saved in "
                   "cavity_qed_ideal_result.csv "
                   "and cavity_qed_decay_result.csv files."
                << std::endl;
      return 0;
    }
:::
:::
:::

::: {#superconducting-qubits-cross-resonance-gate .section}
### Superconducting Qubits: Cross-Resonance Gate[¶](#superconducting-qubits-cross-resonance-gate "Permalink to this heading"){.headerlink}

This example simulates the cross-resonance interaction between two
coupled superconducting qubits, a key primitive for entangling gates in
superconducting hardware. It demonstrates time-dependent Hamiltonians
and batched state evolution.

::: {.highlight-cpp .notranslate}
::: highlight
    /*******************************************************************************
     * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
     * All rights reserved.                                                        *
     *                                                                             *
     * This source code and the accompanying materials are made available under    *
     * the terms of the Apache License 2.0 which accompanies this distribution.    *
     ******************************************************************************/

    // Compile and run with:
    // ```
    // nvq++ --target dynamics cavity_qed.cpp -o a.out && ./a.out
    // ```

    #include "cudaq/algorithms/evolve.h"
    #include "cudaq/algorithms/integrator.h"
    #include "cudaq/operators.h"
    #include "export_csv_helper.h"
    #include <cudaq.h>

    int main() {

      // `delta` represents the detuning between the two qubits.
      // In physical terms, detuning is the energy difference (or frequency offset)
      // between qubit levels. Detuning term (in angular frequency units).
      double delta = 100 * 2 * M_PI;
      // `J` is the static coupling strength between the two qubits.
      // This terms facilitates energy exchange between the qubits, effectively
      // coupling their dynamics.
      double J = 7 * 2 * M_PI;
      // `m_12` models spurious electromagnetic `crosstalk`.
      // `Crosstalk` is an unwanted interaction , here represented as a fraction of
      // the drive strength applied to the second qubit.
      double m_12 = 0.2;
      // `Omega` is the drive strength applied to the qubits.
      // A driving field can induce transitions between qubit states.
      double Omega = 20 * 2 * M_PI;

      // For a spin-1/2 system, the raising operator S^+ and lowering operator S^-
      // are defined as: S^+ = 0.5 * (X + `iY`) and S^- = 0.5 * (X - `iY`) These
      // operators allow transitions between the spin states (|0> and |1>).
      auto spin_plus = [](int degree) {
        return 0.5 * (cudaq::spin_op::x(degree) +
                      std::complex<double>(0.0, 1.0) * cudaq::spin_op::y(degree));
      };

      auto spin_minus = [](int degree) {
        return 0.5 * (cudaq::spin_op::x(degree) -
                      std::complex<double>(0.0, 1.0) * cudaq::spin_op::y(degree));
      };

      // The Hamiltonian describes the energy and dynamics of our 2-qubit system.
      // It consist of several parts:
      // 1. Detuning term for qubit 0: (delta / 2) * Z. This sets the energy
      // splitting for qubit 0.
      // 2. Exchange interaction: J * (S^-_1 * S^+_0 + S^+_1 * S^-_0). This couples
      // the two qubits, enabling excitation transfer.
      // 3. Drive on qubit 0: Omega * X. A control field that drives transition in
      // qubit 0.
      // 4. `Crosstalk` drive on qubit 1: m_12 * Omega * X. A reduces drive on qubit
      // 1 due to electromagnetic `crosstalk`.
      auto hamiltonian =
          (delta / 2.0) * cudaq::spin_op::z(0) +
          J * (spin_minus(1) * spin_plus(0) + spin_plus(1) * spin_minus(0)) +
          Omega * cudaq::spin_op::x(0) + m_12 * Omega * cudaq::spin_op::x(1);

      // Each qubit is a 2-level system (dimension 2).
      // The composite system (two qubits) has a total Hilbert space dimension of 2
      // * 2 = 4.
      cudaq::dimension_map dimensions{{0, 2}, {1, 2}};

      // Build the initial state
      // psi_00 represents the state |00> (both qubits in the ground state).
      // psi_10 represents the state |10> (first qubit excited, second qubit in the
      // ground state).
      std::vector<std::complex<double>> psi00_data = {
          {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
      std::vector<std::complex<double>> psi10_data = {
          {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

      // Two initial state vectors for the 2-qubit system (dimension 4)
      auto psi_00 = cudaq::state::from_data(psi00_data);
      auto psi_10 = cudaq::state::from_data(psi10_data);

      // Create a list of time steps for the simulation.
      // Here we use 1001 points linearly spaced between time 0 and 1.
      // This schedule will be used to integrate the time evolution of the system.
      const int num_steps = 1001;
      std::vector<double> steps = cudaq::linspace(0.0, 1.0, num_steps);
      cudaq::schedule schedule(steps);

      // Use Runge-`Kutta` integrator (4`th` order) to solve the time-dependent
      // evolution. `dt` is the integration time step, and `order` sets the accuracy
      // of the integrator method.
      cudaq::integrators::runge_kutta integrator(4, 0.0001);

      // The observables are the spin components along the x, y, and z directions
      // for both qubits. These observables will be measured during the evolution.
      auto observables = {cudaq::spin_op::x(0), cudaq::spin_op::y(0),
                          cudaq::spin_op::z(0), cudaq::spin_op::x(1),
                          cudaq::spin_op::y(1), cudaq::spin_op::z(1)};

      // Evolution with 2 initial states
      // We evolve the system under the defined Hamiltonian for both initial states
      // simultaneously. No collapsed operators are provided (closed system
      // evolution). The evolution returns expectation values for all defined
      // observables at each time step.
      auto evolution_results = cudaq::evolve(
          hamiltonian, dimensions, schedule, {psi_00, psi_10}, integrator, {},
          observables, cudaq::IntermediateResultSave::ExpectationValue);

      // Retrieve the evolution result corresponding to each initial state.
      auto &evolution_result_00 = evolution_results[0];
      auto &evolution_result_10 = evolution_results[1];

      // Lambda to extract expectation values for a given observable index
      auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
        std::vector<double> expectations;

        auto all_exps = result.expectation_values.value();
        for (auto exp_vals : all_exps) {
          expectations.push_back((double)exp_vals[idx]);
        }
        return expectations;
      };

      // For the two `evolutions`, extract the six observable trajectories.
      // For the |00> initial state, we extract the expectation trajectories for
      // each observable.
      auto result_00_0 = get_expectation(0, evolution_result_00);
      auto result_00_1 = get_expectation(1, evolution_result_00);
      auto result_00_2 = get_expectation(2, evolution_result_00);
      auto result_00_3 = get_expectation(3, evolution_result_00);
      auto result_00_4 = get_expectation(4, evolution_result_00);
      auto result_00_5 = get_expectation(5, evolution_result_00);

      // Similarly, for the |10> initial state:
      auto result_10_0 = get_expectation(0, evolution_result_10);
      auto result_10_1 = get_expectation(1, evolution_result_10);
      auto result_10_2 = get_expectation(2, evolution_result_10);
      auto result_10_3 = get_expectation(3, evolution_result_10);
      auto result_10_4 = get_expectation(4, evolution_result_10);
      auto result_10_5 = get_expectation(5, evolution_result_10);

      // Export the results to a `CSV` file
      // Export the Z-component of qubit 1's expectation values for both initial
      // states. The `CSV` file "cross_resonance_z.`csv`" will have time versus (Z1)
      // data for both |00> and |10> initial conditions.
      export_csv("cross_resonance_z.csv", "time", steps, "<Z1>_00", result_00_5,
                 "<Z1>_10", result_10_5);
      // Export the Y-component of qubit 1's expectation values for both initial
      // states. The `CSV` file "cross_resonance_y.`csv`" will have time versus (Y1)
      // data.
      export_csv("cross_resonance_y.csv", "time", steps, "<Y1>_00", result_00_4,
                 "<Y1>_10", result_10_4);

      std::cout
          << "Simulation complete. The results are saved in cross_resonance_z.csv "
             "and cross_resonance_y.csv files."
          << std::endl;
      return 0;
    }
:::
:::
:::

::: {#spin-qubits-heisenberg-spin-chain .section}
### Spin Qubits: Heisenberg Spin Chain[¶](#spin-qubits-heisenberg-spin-chain "Permalink to this heading"){.headerlink}

This example simulates the time evolution of a Heisenberg spin chain, a
canonical model for studying quantum magnetism and entanglement dynamics
in spin qubit systems.

::: {.highlight-cpp .notranslate}
::: highlight
    /*******************************************************************************
     * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
     * All rights reserved.                                                        *
     *                                                                             *
     * This source code and the accompanying materials are made available under    *
     * the terms of the Apache License 2.0 which accompanies this distribution.    *
     ******************************************************************************/

    // Compile and run with:
    // ```
    // nvq++ --target dynamics heisenberg_model.cpp -o a.out && ./a.out
    // ```

    #include "cudaq/algorithms/evolve.h"
    #include "cudaq/algorithms/integrator.h"
    #include "cudaq/operators.h"
    #include "export_csv_helper.h"
    #include <cudaq.h>

    int main() {

      // Set up a 9-spin chain, where each spin is a two-level system.
      const int num_spins = 9;
      cudaq::dimension_map dimensions;
      for (int i = 0; i < num_spins; i++) {
        dimensions[i] = 2; // Each spin (site) has dimension 2.
      }

      // Initial state
      // Prepare an initial state where the spins are arranged in a staggered
      // configuration. Even indices get the value '0' and odd indices get '1'. For
      // example, for 9 spins: spins: 0 1 0 1 0 1 0 1 0
      std::string spin_state;
      for (int i = 0; i < num_spins; i++) {
        spin_state.push_back((i % 2 == 0) ? '0' : '1');
      }

      // Convert the binary string to an integer index
      // In the Hilbert space of 9 spins (size 2^9 = 512), this index corresponds to
      // the state |0 1 0 1 0 1 0 1 0>
      int initial_state_index = std::stoi(spin_state, nullptr, 2);

      // Build the staggered magnetization operator
      // The staggered magnetization operator is used to measure antiferromagnetic
      // order. It is defined as a sum over all spins of the Z operator, alternating
      // in sign. For even sites, we add `sz`; for odd sites, we subtract `sz`.
      auto staggered_magnetization_t = cudaq::spin_op::empty();
      for (int i = 0; i < num_spins; i++) {
        auto sz = cudaq::spin_op::z(i);
        if (i % 2 == 0) {
          staggered_magnetization_t += sz;
        } else {
          staggered_magnetization_t -= sz;
        }
      }

      // Normalize the number of spins so that the observable is intensive.
      auto stagged_magnetization_op =
          (1 / static_cast<double>(num_spins)) * staggered_magnetization_t;

      // Each entry will associate a value of g (the `anisotropy` in the Z coupling)
      // with its corresponding time-series of expectation values of the staggered
      // magnetization.
      std::vector<std::pair<double, std::vector<double>>> observe_results;

      // Simulate the dynamics over 1000 time steps spanning from time 0 to 5.
      const int num_steps = 1000;
      std::vector<double> steps = cudaq::linspace(0.0, 5.0, num_steps);

      // For three different values of g, which sets the strength of the Z-Z
      // interaction: g = 0.0 (isotropic in the `XY` plane), 0.25, and 4.0 (strongly
      // `anisotropy`).
      std::vector<double> g_values = {0.0, 0.25, 4.0};

      // Initial state vector
      // For a 9-spin system, the Hilbert space dimension is 2^9 = 512.
      // Initialize the state as a vector with all zeros except for a 1 at the
      // index corresponding to our staggered state.
      const int state_size = 1 << num_spins;
      std::vector<std::complex<double>> psi0_data(state_size, {0.0, 0.0});
      psi0_data[initial_state_index] = {1.0, 0.0};

      // We construct a list of Hamiltonian operators for each value of g.
      // All simulations will be batched together in a single call to `evolve`.
      std::vector<cudaq::sum_op<cudaq::matrix_handler>> batched_hamiltonians;
      std::vector<cudaq::state> initial_states;
      for (auto g : g_values) {
        // Set the coupling strengths:
        // `Jx` and `Jy` are set to 1.0 (coupling along X and Y axes), while `Jz` is
        // set to the current g value (coupling along the Z axis).
        double Jx = 1.0, Jy = 1.0, Jz = g;

        // The Hamiltonian is built from the nearest-neighbor interactions:
        // H = H + `Jx` * `Sx`_i * `Sx`_{i+1}
        // H = H + `Jy` * `Sy`_i * `Sy`_{i+1}
        // H = H + `Jz` * `Sz`_i * `Sz`_{i+1}
        // This is a form of the `anisotropic` Heisenberg (or `XYZ`) model.
        auto hamiltonian = cudaq::spin_op::empty();
        for (int i = 0; i < num_spins - 1; i++) {
          hamiltonian =
              hamiltonian + Jx * cudaq::spin_op::x(i) * cudaq::spin_op::x(i + 1);
          hamiltonian =
              hamiltonian + Jy * cudaq::spin_op::y(i) * cudaq::spin_op::y(i + 1);
          hamiltonian =
              hamiltonian + Jz * cudaq::spin_op::z(i) * cudaq::spin_op::z(i + 1);
        }

        // Add the Hamiltonian to the batch.
        batched_hamiltonians.emplace_back(hamiltonian);
        // Initial states for each simulation.
        initial_states.emplace_back(cudaq::state::from_data(psi0_data));
      }

      // The schedule is built using the time steps array.
      cudaq::schedule schedule(steps);

      // Use a Runge-`Kutta` integrator (4`th` order) with a small time step `dt`
      // = 0.001.
      cudaq::integrators::runge_kutta integrator(4, 0.001);

      // Evolve the initial state psi0 under the list of Hamiltonian operators,
      // using the specified schedule and integrator. No collapse operators are
      // included (closed system evolution). Measure the expectation value of the
      // staggered magnetization operator at each time step.
      auto evolve_results =
          cudaq::evolve(batched_hamiltonians, dimensions, schedule, initial_states,
                        integrator, {}, {stagged_magnetization_op},
                        cudaq::IntermediateResultSave::ExpectationValue);

      if (evolve_results.size() != g_values.size()) {
        std::cerr << "Unexpected number of results. Expected " << g_values.size()
                  << "; got " << evolve_results.size() << std::endl;
        return 1;
      }

      // Lambda to extract expectation values for a given observable index
      auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
        std::vector<double> expectations;

        auto all_exps = result.expectation_values.value();
        for (auto exp_vals : all_exps) {
          expectations.push_back((double)exp_vals[idx]);
        }
        return expectations;
      };

      for (std::size_t i = 0; i < g_values.size(); ++i) {
        observe_results.push_back(
            {g_values[i], get_expectation(0, evolve_results[i])});
      }

      // The `CSV` file "`heisenberg`_model.`csv`" will contain column with:
      //    - The time steps
      //    - The expectation values of the staggered magnetization for each g value
      //    (labeled g_0, g_0.25, g_4).
      export_csv("heisenberg_model_result.csv", "time", steps, "g_0",
                 observe_results[0].second, "g_0.25", observe_results[1].second,
                 "g_4", observe_results[2].second);

      std::cout << "Simulation complete. The results are saved in "
                   "heisenberg_model_result.csv file."
                << std::endl;
      return 0;
    }
:::
:::
:::

::: {#control-driven-qubit .section}
### Control: Driven Qubit[¶](#control-driven-qubit "Permalink to this heading"){.headerlink}

This example demonstrates qubit control via a time-dependent driving
Hamiltonian. It shows how to construct schedules with named time
parameters and time-dependent coefficient callbacks for modelling
control pulses.

::: {.highlight-cpp .notranslate}
::: highlight
    /*******************************************************************************
     * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
     * All rights reserved.                                                        *
     *                                                                             *
     * This source code and the accompanying materials are made available under    *
     * the terms of the Apache License 2.0 which accompanies this distribution.    *
     ******************************************************************************/

    // Compile and run with:
    // ```
    // nvq++ --target dynamics qubit_control.cpp -o a.out && ./a.out
    // ```

    #include "cudaq/algorithms/evolve.h"
    #include "cudaq/algorithms/integrator.h"
    #include "cudaq/operators.h"
    #include "export_csv_helper.h"
    #include <cudaq.h>

    int main() {
      // Qubit resonant frequency (energy splitting along Z).
      double omega_z = 10.0 * 2 * M_PI;
      // Transverse driving term (amplitude of the drive along the X-axis).
      double omega_x = 2 * M_PI;
      // Driving frequency, chosen to be slightly off-resonance (0.99 of omega_z).
      double omega_drive = 0.99 * omega_z;

      // The lambda function acts as a callback that returns a modulation factor for
      // the drive. It extracts the time `t` from the provided parameters and
      // computes cos(omega_drive * t).
      auto mod_func =
          [omega_drive](
              const cudaq::parameter_map &params) -> std::complex<double> {
        auto it = params.find("t");
        if (it != params.end()) {
          double t = it->second.real();
          const auto result = std::cos(omega_drive * t);
          return result;
        }
        throw std::runtime_error("Cannot find the time parameter.");
      };

      // The Hamiltonian consists of two terms:
      // 1. A static term: 0.5 * omega_z * `Sz`_0, representing the `qubit's`
      // intrinsic energy splitting.
      // 2. A time-dependent driving term: omega_x * cos(omega_drive * t) * `Sx`_0,
      // which induces rotations about the X-axis. The scalar_operator(mod_`func`)
      // allows the drive term to vary in time according to mod_`func`.
      auto hamiltonian = 0.5 * omega_z * cudaq::spin_op::z(0) +
                         mod_func * cudaq::spin_op::x(0) * omega_x;

      // A single qubit with dimension 2.
      cudaq::dimension_map dimensions = {{0, 2}};

      // The qubit starts in the |0> state, represented by the vector [1, 0].
      std::vector<std::complex<double>> initial_state_vec = {1.0, 0.0};
      auto psi0 = cudaq::state::from_data(initial_state_vec);

      // Set the final simulation time such that t_final = pi / omega_x, which
      // relates to a specific qubit rotation.
      double t_final = M_PI / omega_x;
      // Define the integration time step `dt` as a small fraction of the drive
      // period.
      double dt = 2.0 * M_PI / omega_drive / 100;
      // Compute the number of steps required for the simulation
      int num_steps = static_cast<int>(std::ceil(t_final / dt)) + 1;
      // Create a schedule with time steps from 0 to t_final.
      std::vector<double> steps = cudaq::linspace(0.0, t_final, num_steps);
      // The schedule carries the time parameter `labelled` `t`, which is used by
      // mod_`func`.
      cudaq::schedule schedule(steps, {"t"});

      // A default Runge-`Kutta` integrator (4`th` order) with time step `dt`
      // depending on the schedule.
      cudaq::integrators::runge_kutta integrator;

      // Measure the expectation values of the `qubit's` spin components along the
      // X, Y, and Z directions.
      auto observables = {cudaq::spin_op::x(0), cudaq::spin_op::y(0),
                          cudaq::spin_op::z(0)};

      // Simulation without decoherence
      // Evolve the system under the Hamiltonian, using the specified schedule and
      // integrator. No collapse operators are included (closed system evolution).
      auto evolve_result = cudaq::evolve(
          hamiltonian, dimensions, schedule, psi0, integrator, {}, observables,
          cudaq::IntermediateResultSave::ExpectationValue);

      // Simulation with decoherence
      // Introduce `dephasing` (decoherence) through a collapse operator.
      // Here, gamma_`sz` = 1.0 is the `dephasing` rate, and the collapse operator
      // is `sqrt`(gamma_`sz`) * `Sz`_0 which simulates decoherence in the energy
      // basis (Z-basis `dephasing`).
      double gamma_sz = 1.0;
      auto evolve_result_decay =
          cudaq::evolve(hamiltonian, dimensions, schedule, psi0, integrator,
                        {std::sqrt(gamma_sz) * cudaq::spin_op::z(0)}, observables,
                        cudaq::IntermediateResultSave::ExpectationValue);

      // Lambda to extract expectation values for a given observable index
      auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
        std::vector<double> expectations;

        auto all_exps = result.expectation_values.value();
        for (auto exp_vals : all_exps) {
          expectations.push_back(exp_vals[idx].expectation());
        }
        return expectations;
      };

      // For the ideal evolution
      auto ideal_result_x = get_expectation(0, evolve_result);
      auto ideal_result_y = get_expectation(1, evolve_result);
      auto ideal_result_z = get_expectation(2, evolve_result);

      // For the decoherence evolution
      auto decoherence_result_x = get_expectation(0, evolve_result_decay);
      auto decoherence_result_y = get_expectation(1, evolve_result_decay);
      auto decoherence_result_z = get_expectation(2, evolve_result_decay);

      // Export the results to a `CSV` file
      export_csv("qubit_control_ideal_result.csv", "t", steps, "sigma_x",
                 ideal_result_x, "sigma_y", ideal_result_y, "sigma_z",
                 ideal_result_z);
      export_csv("qubit_control_decoherence_result.csv", "t", steps, "sigma_x",
                 decoherence_result_x, "sigma_y", decoherence_result_y, "sigma_z",
                 decoherence_result_z);

      std::cout << "Results exported to qubit_control_ideal_result.csv and "
                   "qubit_control_decoherence_result.csv"
                << std::endl;

      return 0;
    }
:::
:::
:::

::: {#state-batching .section}
### State Batching[¶](#state-batching "Permalink to this heading"){.headerlink}

Batching multiple initial states in a single [`evolve`{.docutils
.literal .notranslate}]{.pre} call enables efficient process tomography
and parallel parameter sweeps. This example evolves four SIC-POVM states
under the same Hamiltonian simultaneously.

::: {.highlight-cpp .notranslate}
::: highlight
    /*******************************************************************************
     * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
     * All rights reserved.                                                        *
     *                                                                             *
     * This source code and the accompanying materials are made available under    *
     * the terms of the Apache License 2.0 which accompanies this distribution.    *
     ******************************************************************************/

    // Compile and run with:
    // ```
    // nvq++ --target dynamics qubit_dynamics.cpp -o a.out && ./a.out
    // ```

    #include "cudaq/algorithms/evolve.h"
    #include "cudaq/algorithms/integrator.h"
    #include "cudaq/operators.h"
    #include "export_csv_helper.h"
    #include <cudaq.h>

    int main() {
      // Qubit `hamiltonian`: 2 * pi * 0.1 * sigma_x
      // Physically, this represents a qubit (a two-level system) driven by a weak
      // transverse field along the x-axis.
      auto hamiltonian = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);

      // Dimensions: one subsystem of dimension 2 (a two-level system).
      const cudaq::dimension_map dimensions = {{0, 2}};

      // Initial state: ground state
      std::vector<std::complex<double>> initial_state_zero = {1.0, 0.0};
      std::vector<std::complex<double>> initial_state_one = {0.0, 1.0};

      auto psi0 = cudaq::state::from_data(initial_state_zero);
      auto psi1 = cudaq::state::from_data(initial_state_one);

      // Create a schedule of time steps from 0 to 10 with 101 points
      std::vector<double> steps = cudaq::linspace(0.0, 10.0, 101);
      cudaq::schedule schedule(steps);

      // Runge-`Kutta` integrator with a time step of 0.01 and order 4
      cudaq::integrators::runge_kutta integrator(4, 0.01);

      // Run the simulation without collapse operators (ideal evolution)
      auto evolve_results =
          cudaq::evolve(hamiltonian, dimensions, schedule, {psi0, psi1}, integrator,
                        {}, {cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
                        cudaq::IntermediateResultSave::ExpectationValue);

      // Lambda to extract expectation values for a given observable index
      auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
        std::vector<double> expectations;

        auto all_exps = result.expectation_values.value();
        for (auto exp_vals : all_exps) {
          expectations.push_back((double)exp_vals[idx]);
        }
        return expectations;
      };

      auto result_state0_y = get_expectation(0, evolve_results[0]);
      auto result_state0_z = get_expectation(1, evolve_results[0]);
      auto result_state1_y = get_expectation(0, evolve_results[1]);
      auto result_state1_z = get_expectation(1, evolve_results[1]);

      export_csv("qubit_dynamics_state_0.csv", "time", steps, "sigma_y",
                 result_state0_y, "sigma_z", result_state0_z);
      export_csv("qubit_dynamics_state_1.csv", "time", steps, "sigma_y",
                 result_state1_y, "sigma_z", result_state1_z);

      std::cout << "Results exported to qubit_dynamics_state_0.csv and "
                   "qubit_dynamics_state_1.csv"
                << std::endl;

      return 0;
    }
:::
:::
:::

::: {#numerical-integrators .section}
### Numerical Integrators[¶](#numerical-integrators "Permalink to this heading"){.headerlink}

CUDA-Q provides three numerical integrators for the
[`dynamics`{.docutils .literal .notranslate}]{.pre} target.

The following example shows how to use these integrators on the same
single-qubit problem:

::: {.highlight-cpp .notranslate}
::: highlight
      // Explicit 4th-order Runge-Kutta method (the default integrator).
      // Arguments: order (1, 2, or 4) and optional max sub-step size.
      cudaq::integrators::runge_kutta rk_integrator(/*order=*/4,
                                                    /*max_step_size=*/0.01);
      auto rk_result = cudaq::evolve(
          hamiltonian, dimensions, schedule, psi0, rk_integrator, {}, observables,
          cudaq::IntermediateResultSave::ExpectationValue);
:::
:::

The Crank-Nicolson integrator:

::: {.highlight-cpp .notranslate}
::: highlight
      // Implicit Crank-Nicolson predictor-corrector method.
      // Well-suited for stiff systems or when energy conservation is important.
      // Arguments: number of corrector iterations (default: 2) and optional max
      // sub-step size.
      cudaq::integrators::crank_nicolson cn_integrator(/*num_corrector_steps=*/2,
                                                       /*max_step_size=*/0.01);
      auto cn_result = cudaq::evolve(
          hamiltonian, dimensions, schedule, psi0, cn_integrator, {}, observables,
          cudaq::IntermediateResultSave::ExpectationValue);
:::
:::

The Magnus expansion integrator:

::: {.highlight-cpp .notranslate}
::: highlight
      // Magnus expansion integrator.
      // Uses a finite Taylor series truncation to approximate the matrix
      // exponential, approximating unitary evolution. Suitable for smooth,
      // oscillatory
      // Hamiltonians. Arguments: maximum number of Taylor terms (default: 10) and
      // optional max sub-step size.
      cudaq::integrators::magnus_expansion magnus_integrator(
          /*num_taylor_terms=*/10, /*max_step_size=*/0.01);
      auto magnus_result = cudaq::evolve(
          hamiltonian, dimensions, schedule, psi0, magnus_integrator, {},
          observables, cudaq::IntermediateResultSave::ExpectationValue);
:::
:::
:::
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](sample_vs_run.html "When to Use sample vs. run"){.btn
.btn-neutral .float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](../../examples/python/dynamics/dynamics_intro_1.html "Introduction to CUDA-Q Dynamics (Jaynes-Cummings Model)"){.btn
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
