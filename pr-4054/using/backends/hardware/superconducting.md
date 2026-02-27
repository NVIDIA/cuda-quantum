::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../../index.html){.icon .icon-home}

::: version
pr-4054
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](../../quick_start.html){.reference .internal}
    -   [Install
        CUDA-Q](../../quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](../../quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q
        Academic](../../quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](../../basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q
        Kernel?](../../basics/kernel_intro.html){.reference .internal}
    -   [Building your first CUDA-Q
        Program](../../basics/build_kernel.html){.reference .internal}
    -   [Running your first CUDA-Q
        Program](../../basics/run_kernel.html){.reference .internal}
        -   [Sample](../../basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](../../basics/run_kernel.html#run){.reference
            .internal}
        -   [Observe](../../basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](../../basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](../../basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](../../basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
-   [Examples](../../examples/examples.html){.reference .internal}
    -   [Introduction](../../examples/introduction.html){.reference
        .internal}
    -   [Building
        Kernels](../../examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](../../examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](../../examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](../../examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](../../examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](../../examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](../../examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](../../examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](../../examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](../../examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](../../examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](../../examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](../../examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](../../examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring
        Kernels](../../examples/measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](../../examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](../../../examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](../../../examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](../../../examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing
        Kernels](../../examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](../../examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](../../examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](../../examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](../../examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](../../examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](../../examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](../../examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](../../examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](../../examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](../../examples/expectation_values.html){.reference
        .internal}
        -   [Parallelizing across Multiple
            Processors](../../examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](../../examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](../../examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../../examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../../examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](../../examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](../../examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends ([`remote-mqpu`{.code .docutils
            .literal
            .notranslate}]{.pre})](../../examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
            .internal}
    -   [Optimizers &
        Gradients](../../../examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [CUDA-Q Optimizer
            Overview](../../../examples/python/optimizers_gradients.html#CUDA-Q-Optimizer-Overview){.reference
            .internal}
            -   [Gradient-Free Optimizers (no gradients
                required):](../../../examples/python/optimizers_gradients.html#Gradient-Free-Optimizers-(no-gradients-required):){.reference
                .internal}
            -   [Gradient-Based Optimizers (require
                gradients):](../../../examples/python/optimizers_gradients.html#Gradient-Based-Optimizers-(require-gradients):){.reference
                .internal}
        -   [1. Built-in CUDA-Q Optimizers and
            Gradients](../../../examples/python/optimizers_gradients.html#1.-Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
            -   [1.1 Adam Optimizer with Parameter
                Configuration](../../../examples/python/optimizers_gradients.html#1.1-Adam-Optimizer-with-Parameter-Configuration){.reference
                .internal}
            -   [1.2 SGD (Stochastic Gradient Descent)
                Optimizer](../../../examples/python/optimizers_gradients.html#1.2-SGD-(Stochastic-Gradient-Descent)-Optimizer){.reference
                .internal}
            -   [1.3 SPSA (Simultaneous Perturbation Stochastic
                Approximation)](../../../examples/python/optimizers_gradients.html#1.3-SPSA-(Simultaneous-Perturbation-Stochastic-Approximation)){.reference
                .internal}
        -   [2. Third-Party
            Optimizers](../../../examples/python/optimizers_gradients.html#2.-Third-Party-Optimizers){.reference
            .internal}
        -   [3. Parallel Parameter Shift
            Gradients](../../../examples/python/optimizers_gradients.html#3.-Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](../../../examples/python/noisy_simulations.html){.reference
        .internal}
    -   [PTSBE End-to-End
        Workflow](../../../examples/python/ptsbe_end_to_end_workflow.html){.reference
        .internal}
        -   [1. Set up the
            environment](../../../examples/python/ptsbe_end_to_end_workflow.html#1.-Set-up-the-environment){.reference
            .internal}
        -   [2. Define the circuit and noise
            model](../../../examples/python/ptsbe_end_to_end_workflow.html#2.-Define-the-circuit-and-noise-model){.reference
            .internal}
        -   [3. Run PTSBE
            sampling](../../../examples/python/ptsbe_end_to_end_workflow.html#3.-Run-PTSBE-sampling){.reference
            .internal}
        -   [4. Compare with standard (density-matrix)
            sampling](../../../examples/python/ptsbe_end_to_end_workflow.html#4.-Compare-with-standard-(density-matrix)-sampling){.reference
            .internal}
        -   [5. Return execution
            data](../../../examples/python/ptsbe_end_to_end_workflow.html#5.-Return-execution-data){.reference
            .internal}
        -   [6. Two API
            options:](../../../examples/python/ptsbe_end_to_end_workflow.html#6.-Two-API-options:){.reference
            .internal}
    -   [Constructing
        Operators](../../examples/operators.html){.reference .internal}
        -   [Constructing Spin
            Operators](../../examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](../../examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](../../../examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](../../../examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](../../examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](../../examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](../../examples/hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](../../examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](../../examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](../../examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](../../examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](../../examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](../../examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](../../examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum Circuits,
            Inc.](../../examples/hardware_providers.html#quantum-circuits-inc){.reference
            .internal}
        -   [Quantum
            Machines](../../examples/hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](../../examples/hardware_providers.html#quera-computing){.reference
            .internal}
    -   [Dynamics
        Examples](../../examples/dynamics_examples.html){.reference
        .internal}
        -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
            Model)](../../../examples/python/dynamics/dynamics_intro_1.html){.reference
            .internal}
            -   [Why dynamics simulations vs. circuit
                simulations?](../../../examples/python/dynamics/dynamics_intro_1.html#Why-dynamics-simulations-vs.-circuit-simulations?){.reference
                .internal}
            -   [Functionality](../../../examples/python/dynamics/dynamics_intro_1.html#Functionality){.reference
                .internal}
            -   [Performance](../../../examples/python/dynamics/dynamics_intro_1.html#Performance){.reference
                .internal}
            -   [Section 1 - Simulating the Jaynes-Cummings
                Hamiltonian](../../../examples/python/dynamics/dynamics_intro_1.html#Section-1---Simulating-the-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Exercise 1 - Simulating a many-photon Jaynes-Cummings
                Hamiltonian](../../../examples/python/dynamics/dynamics_intro_1.html#Exercise-1---Simulating-a-many-photon-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Section 2 - Simulating open quantum systems with the
                [`collapse_operators`{.docutils .literal
                .notranslate}]{.pre}](../../../examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
                .internal}
            -   [Exercise 2 - Adding additional jump operators
                [\\(L_i\\)]{.math .notranslate
                .nohighlight}](../../../examples/python/dynamics/dynamics_intro_1.html#Exercise-2---Adding-additional-jump-operators-L_i){.reference
                .internal}
            -   [Section 3 - Many qubits coupled to the
                resonator](../../../examples/python/dynamics/dynamics_intro_1.html#Section-3---Many-qubits-coupled-to-the-resonator){.reference
                .internal}
        -   [Introduction to CUDA-Q Dynamics (Time Dependent
            Hamiltonians)](../../../examples/python/dynamics/dynamics_intro_2.html){.reference
            .internal}
            -   [The Landau-Zener
                model](../../../examples/python/dynamics/dynamics_intro_2.html#The-Landau-Zener-model){.reference
                .internal}
            -   [Section 1 - Implementing time dependent
                terms](../../../examples/python/dynamics/dynamics_intro_2.html#Section-1---Implementing-time-dependent-terms){.reference
                .internal}
            -   [Section 2 - Implementing custom
                operators](../../../examples/python/dynamics/dynamics_intro_2.html#Section-2---Implementing-custom-operators){.reference
                .internal}
            -   [Section 3 - Heisenberg Model with a time-varying
                magnetic
                field](../../../examples/python/dynamics/dynamics_intro_2.html#Section-3---Heisenberg-Model-with-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 1 - Define a time-varying magnetic
                field](../../../examples/python/dynamics/dynamics_intro_2.html#Exercise-1---Define-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 2
                (Optional)](../../../examples/python/dynamics/dynamics_intro_2.html#Exercise-2-(Optional)){.reference
                .internal}
        -   [Superconducting
            Qubits](../../../examples/python/dynamics/superconducting.html){.reference
            .internal}
            -   [Cavity
                QED](../../../examples/python/dynamics/superconducting.html#Cavity-QED){.reference
                .internal}
            -   [Cross
                Resonance](../../../examples/python/dynamics/superconducting.html#Cross-Resonance){.reference
                .internal}
            -   [Transmon
                Resonator](../../../examples/python/dynamics/superconducting.html#Transmon-Resonator){.reference
                .internal}
        -   [Spin
            Qubits](../../../examples/python/dynamics/spinqubits.html){.reference
            .internal}
            -   [Silicon Spin
                Qubit](../../../examples/python/dynamics/spinqubits.html#Silicon-Spin-Qubit){.reference
                .internal}
            -   [Heisenberg
                Model](../../../examples/python/dynamics/spinqubits.html#Heisenberg-Model){.reference
                .internal}
        -   [Trapped Ion
            Qubits](../../../examples/python/dynamics/iontrap.html){.reference
            .internal}
            -   [GHZ
                state](../../../examples/python/dynamics/iontrap.html#GHZ-state){.reference
                .internal}
        -   [Control](../../../examples/python/dynamics/control.html){.reference
            .internal}
            -   [Gate
                Calibration](../../../examples/python/dynamics/control.html#Gate-Calibration){.reference
                .internal}
            -   [Pulse](../../../examples/python/dynamics/control.html#Pulse){.reference
                .internal}
            -   [Qubit
                Control](../../../examples/python/dynamics/control.html#Qubit-Control){.reference
                .internal}
            -   [Qubit
                Dynamics](../../../examples/python/dynamics/control.html#Qubit-Dynamics){.reference
                .internal}
            -   [Landau-Zenner](../../../examples/python/dynamics/control.html#Landau-Zenner){.reference
                .internal}
-   [Applications](../../applications.html){.reference .internal}
    -   [Max-Cut with
        QAOA](../../../applications/python/qaoa.html){.reference
        .internal}
    -   [Molecular docking via
        DC-QAOA](../../../applications/python/digitized_counterdiabatic_qaoa.html){.reference
        .internal}
        -   [Setting up the Molecular Docking
            Problem](../../../applications/python/digitized_counterdiabatic_qaoa.html#Setting-up-the-Molecular-Docking-Problem){.reference
            .internal}
        -   [CUDA-Q
            Implementation](../../../applications/python/digitized_counterdiabatic_qaoa.html#CUDA-Q-Implementation){.reference
            .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](../../../applications/python/krylov.html){.reference
        .internal}
        -   [Setup](../../../applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](../../../applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](../../../applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Quantum-Selected Configuration Interaction
        (QSCI)](../../../applications/python/qsci.html){.reference
        .internal}
        -   [0. Problem
            definition](../../../applications/python/qsci.html#0.-Problem-definition){.reference
            .internal}
        -   [1. Prepare an Approximate Quantum
            State](../../../applications/python/qsci.html#1.-Prepare-an-Approximate-Quantum-State){.reference
            .internal}
        -   [2 Quantum Sampling to Select
            Configuration](../../../applications/python/qsci.html#2-Quantum-Sampling-to-Select-Configuration){.reference
            .internal}
        -   [3. Classical Diagonalization on the Selected
            Subspace](../../../applications/python/qsci.html#3.-Classical-Diagonalization-on-the-Selected-Subspace){.reference
            .internal}
        -   [5. Compuare
            results](../../../applications/python/qsci.html#5.-Compuare-results){.reference
            .internal}
        -   [Reference](../../../applications/python/qsci.html#Reference){.reference
            .internal}
    -   [Bernstein-Vazirani
        Algorithm](../../../applications/python/bernstein_vazirani.html){.reference
        .internal}
        -   [Classical
            case](../../../applications/python/bernstein_vazirani.html#Classical-case){.reference
            .internal}
        -   [Quantum
            case](../../../applications/python/bernstein_vazirani.html#Quantum-case){.reference
            .internal}
        -   [Implementing in
            CUDA-Q](../../../applications/python/bernstein_vazirani.html#Implementing-in-CUDA-Q){.reference
            .internal}
    -   [Cost
        Minimization](../../../applications/python/cost_minimization.html){.reference
        .internal}
    -   [Deutsch's
        Algorithm](../../../applications/python/deutsch_algorithm.html){.reference
        .internal}
        -   [XOR [\\(\\oplus\\)]{.math .notranslate
            .nohighlight}](../../../applications/python/deutsch_algorithm.html#XOR-\oplus){.reference
            .internal}
        -   [Quantum
            oracles](../../../applications/python/deutsch_algorithm.html#Quantum-oracles){.reference
            .internal}
        -   [Phase
            oracle](../../../applications/python/deutsch_algorithm.html#Phase-oracle){.reference
            .internal}
        -   [Quantum
            parallelism](../../../applications/python/deutsch_algorithm.html#Quantum-parallelism){.reference
            .internal}
        -   [Deutsch's
            Algorithm:](../../../applications/python/deutsch_algorithm.html#Deutsch's-Algorithm:){.reference
            .internal}
    -   [Divisive Clustering With Coresets Using
        CUDA-Q](../../../applications/python/divisive_clustering_coresets.html){.reference
        .internal}
        -   [Data
            preprocessing](../../../applications/python/divisive_clustering_coresets.html#Data-preprocessing){.reference
            .internal}
        -   [Quantum
            functions](../../../applications/python/divisive_clustering_coresets.html#Quantum-functions){.reference
            .internal}
        -   [Divisive Clustering
            Function](../../../applications/python/divisive_clustering_coresets.html#Divisive-Clustering-Function){.reference
            .internal}
        -   [QAOA
            Implementation](../../../applications/python/divisive_clustering_coresets.html#QAOA-Implementation){.reference
            .internal}
        -   [Scaling simulations with
            CUDA-Q](../../../applications/python/divisive_clustering_coresets.html#Scaling-simulations-with-CUDA-Q){.reference
            .internal}
    -   [Hybrid Quantum Neural
        Networks](../../../applications/python/hybrid_quantum_neural_networks.html){.reference
        .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](../../../applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](../../../applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using [`Sample`{.docutils .literal .notranslate}]{.pre} to
            perform the Hadamard
            test](../../../applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](../../../applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](../../../applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Anderson Impurity Model ground state solver on Infleqtion's
        Sqale](../../../applications/python/logical_aim_sqale.html){.reference
        .internal}
        -   [Performing logical Variational Quantum Eigensolver (VQE)
            with
            CUDA-QX](../../../applications/python/logical_aim_sqale.html#Performing-logical-Variational-Quantum-Eigensolver-(VQE)-with-CUDA-QX){.reference
            .internal}
        -   [Constructing circuits in the [`[[4,2,2]]`{.docutils
            .literal .notranslate}]{.pre}
            encoding](../../../applications/python/logical_aim_sqale.html#Constructing-circuits-in-the-%5B%5B4,2,2%5D%5D-encoding){.reference
            .internal}
        -   [Setting up submission and decoding
            workflow](../../../applications/python/logical_aim_sqale.html#Setting-up-submission-and-decoding-workflow){.reference
            .internal}
        -   [Running a CUDA-Q noisy
            simulation](../../../applications/python/logical_aim_sqale.html#Running-a-CUDA-Q-noisy-simulation){.reference
            .internal}
        -   [Running logical AIM on Infleqtion's
            hardware](../../../applications/python/logical_aim_sqale.html#Running-logical-AIM-on-Infleqtion's-hardware){.reference
            .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](../../../applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](../../../applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](../../../applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](../../../applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](../../../applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](../../../applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](../../../applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](../../../applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. [`Compute`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`overlap`{.docutils .literal
                .notranslate}]{.pre}](../../../applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](../../../applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](../../../applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](../../../applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](../../../applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](../../../applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](../../../applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](../../../applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](../../../applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](../../../applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum Fourier
        Transform](../../../applications/python/quantum_fourier_transform.html){.reference
        .internal}
        -   [Quantum Fourier Transform
            revisited](../../../applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform-revisited){.reference
            .internal}
    -   [Quantum
        Teleporation](../../../applications/python/quantum_teleportation.html){.reference
        .internal}
        -   [Teleportation
            explained](../../../applications/python/quantum_teleportation.html#Teleportation-explained){.reference
            .internal}
    -   [Quantum
        Volume](../../../applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](../../../applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](../../../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](../../../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](../../../applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Compiling Unitaries Using Diffusion
        Models](../../../applications/python/unitary_compilation_diffusion_models.html){.reference
        .internal}
        -   [Diffusion model
            pipeline](../../../applications/python/unitary_compilation_diffusion_models.html#Diffusion-model-pipeline){.reference
            .internal}
        -   [Setup and load
            models](../../../applications/python/unitary_compilation_diffusion_models.html#Setup-and-load-models){.reference
            .internal}
            -   [Load discrete
                model](../../../applications/python/unitary_compilation_diffusion_models.html#Load-discrete-model){.reference
                .internal}
            -   [Load continuous
                model](../../../applications/python/unitary_compilation_diffusion_models.html#Load-continuous-model){.reference
                .internal}
            -   [Create helper
                functions](../../../applications/python/unitary_compilation_diffusion_models.html#Create-helper-functions){.reference
                .internal}
        -   [Unitary
            compilation](../../../applications/python/unitary_compilation_diffusion_models.html#Unitary-compilation){.reference
            .internal}
            -   [Random
                unitary](../../../applications/python/unitary_compilation_diffusion_models.html#Random-unitary){.reference
                .internal}
            -   [Discrete
                model](../../../applications/python/unitary_compilation_diffusion_models.html#Discrete-model){.reference
                .internal}
            -   [Continuous
                model](../../../applications/python/unitary_compilation_diffusion_models.html#Continuous-model){.reference
                .internal}
            -   [Quantum Fourier
                transform](../../../applications/python/unitary_compilation_diffusion_models.html#Quantum-Fourier-transform){.reference
                .internal}
            -   [XXZ-Hamiltonian
                evolution](../../../applications/python/unitary_compilation_diffusion_models.html#XXZ-Hamiltonian-evolution){.reference
                .internal}
        -   [Choosing the circuit you
            need](../../../applications/python/unitary_compilation_diffusion_models.html#Choosing-the-circuit-you-need){.reference
            .internal}
    -   [VQE with gradients, active spaces, and gate
        fusion](../../../applications/python/vqe_advanced.html){.reference
        .internal}
        -   [The Basics of
            VQE](../../../applications/python/vqe_advanced.html#The-Basics-of-VQE){.reference
            .internal}
        -   [Installing/Loading Relevant
            Packages](../../../applications/python/vqe_advanced.html#Installing/Loading-Relevant-Packages){.reference
            .internal}
        -   [Implementing VQE in
            CUDA-Q](../../../applications/python/vqe_advanced.html#Implementing-VQE-in-CUDA-Q){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../../../applications/python/vqe_advanced.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
        -   [Using an Active
            Space](../../../applications/python/vqe_advanced.html#Using-an-Active-Space){.reference
            .internal}
        -   [Gate Fusion for Larger
            Circuits](../../../applications/python/vqe_advanced.html#Gate-Fusion-for-Larger-Circuits){.reference
            .internal}
    -   [Quantum
        Transformer](../../../applications/python/quantum_transformer.html){.reference
        .internal}
        -   [Installation](../../../applications/python/quantum_transformer.html#Installation){.reference
            .internal}
        -   [Algorithm and
            Example](../../../applications/python/quantum_transformer.html#Algorithm-and-Example){.reference
            .internal}
            -   [Creating the self-attention
                circuits](../../../applications/python/quantum_transformer.html#Creating-the-self-attention-circuits){.reference
                .internal}
        -   [Usage](../../../applications/python/quantum_transformer.html#Usage){.reference
            .internal}
            -   [Model
                Training](../../../applications/python/quantum_transformer.html#Model-Training){.reference
                .internal}
            -   [Generating
                Molecules](../../../applications/python/quantum_transformer.html#Generating-Molecules){.reference
                .internal}
            -   [Attention
                Maps](../../../applications/python/quantum_transformer.html#Attention-Maps){.reference
                .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](../../../applications/python/afqmc.html){.reference
        .internal}
        -   [Hamiltonian preparation for
            VQE](../../../applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](../../../applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](../../../applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](../../../applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](../../../applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](../../../applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [ADAPT-QAOA
        algorithm](../../../applications/python/adapt_qaoa.html){.reference
        .internal}
        -   [Simulation
            input:](../../../applications/python/adapt_qaoa.html#Simulation-input:){.reference
            .internal}
        -   [The problem Hamiltonian [\\(H_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](../../../applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A_j\\)]{.math .notranslate
            .nohighlight}:](../../../applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H_C,A_j\]\\)]{.math .notranslate
            .nohighlight}:](../../../applications/python/adapt_qaoa.html#The-commutator-%5BH_C,A_j%5D:){.reference
            .internal}
        -   [Beginning of ADAPT-QAOA
            iteration:](../../../applications/python/adapt_qaoa.html#Beginning-of-ADAPT-QAOA-iteration:){.reference
            .internal}
    -   [ADAPT-VQE
        algorithm](../../../applications/python/adapt_vqe.html){.reference
        .internal}
        -   [Classical
            pre-processing](../../../applications/python/adapt_vqe.html#Classical-pre-processing){.reference
            .internal}
        -   [Jordan
            Wigner:](../../../applications/python/adapt_vqe.html#Jordan-Wigner:){.reference
            .internal}
        -   [UCCSD operator
            pool](../../../applications/python/adapt_vqe.html#UCCSD-operator-pool){.reference
            .internal}
            -   [Single
                excitation](../../../applications/python/adapt_vqe.html#Single-excitation){.reference
                .internal}
            -   [Double
                excitation](../../../applications/python/adapt_vqe.html#Double-excitation){.reference
                .internal}
        -   [Commutator \[[\\(H\\)]{.math .notranslate .nohighlight},
            [\\(A_i\\)]{.math .notranslate
            .nohighlight}\]](../../../applications/python/adapt_vqe.html#Commutator-%5BH,-A_i%5D){.reference
            .internal}
        -   [Reference
            State:](../../../applications/python/adapt_vqe.html#Reference-State:){.reference
            .internal}
        -   [Quantum
            kernels:](../../../applications/python/adapt_vqe.html#Quantum-kernels:){.reference
            .internal}
        -   [Beginning of
            ADAPT-VQE:](../../../applications/python/adapt_vqe.html#Beginning-of-ADAPT-VQE:){.reference
            .internal}
    -   [Quantum edge
        detection](../../../applications/python/edge_detection.html){.reference
        .internal}
        -   [Image](../../../applications/python/edge_detection.html#Image){.reference
            .internal}
        -   [Quantum Probability Image Encoding
            (QPIE):](../../../applications/python/edge_detection.html#Quantum-Probability-Image-Encoding-(QPIE):){.reference
            .internal}
            -   [Below we show how to encode an image using QPIE in
                cudaq.](../../../applications/python/edge_detection.html#Below-we-show-how-to-encode-an-image-using-QPIE-in-cudaq.){.reference
                .internal}
        -   [Flexible Representation of Quantum Images
            (FRQI):](../../../applications/python/edge_detection.html#Flexible-Representation-of-Quantum-Images-(FRQI):){.reference
            .internal}
            -   [Building the FRQI
                State:](../../../applications/python/edge_detection.html#Building-the-FRQI-State:){.reference
                .internal}
        -   [Quantum Hadamard Edge Detection
            (QHED)](../../../applications/python/edge_detection.html#Quantum-Hadamard-Edge-Detection-(QHED)){.reference
            .internal}
            -   [Post-processing](../../../applications/python/edge_detection.html#Post-processing){.reference
                .internal}
    -   [Factoring Integers With Shor's
        Algorithm](../../../applications/python/shors.html){.reference
        .internal}
        -   [Shor's
            algorithm](../../../applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](../../../applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](../../../applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](../../../applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](../../../applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](../../../applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](../../../applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](../../../applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
            -   [(a) Generate the molecular Hamiltonian using Restricted
                Hartree Fock molecular
                orbitals](../../../applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Restricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(b) Generate the molecular Hamiltonian using
                Unrestricted Hartree Fock molecular
                orbitals](../../../applications/python/generate_fermionic_ham.html#(b)-Generate-the-molecular-Hamiltonian-using-Unrestricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(a) Generate the active space hamiltonian using RHF
                molecular
                orbitals.](../../../applications/python/generate_fermionic_ham.html#(a)-Generate-the-active-space-hamiltonian-using-RHF-molecular-orbitals.){.reference
                .internal}
            -   [(b) Generate the active space Hamiltonian using the
                natural orbitals computed from MP2
                simulation](../../../applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
                .internal}
            -   [(c) Generate the active space Hamiltonian computed from
                the CASSCF molecular
                orbitals](../../../applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
                .internal}
            -   [(d) Generate the electronic Hamiltonian using
                ROHF](../../../applications/python/generate_fermionic_ham.html#(d)-Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
            -   [(e) Generate electronic Hamiltonian using
                UHF](../../../applications/python/generate_fermionic_ham.html#(e)-Generate-electronic-Hamiltonian-using-UHF){.reference
                .internal}
    -   [Grover's
        Algorithm](../../../applications/python/grovers.html){.reference
        .internal}
        -   [Overview](../../../applications/python/grovers.html#Overview){.reference
            .internal}
        -   [Problem](../../../applications/python/grovers.html#Problem){.reference
            .internal}
        -   [Structure of Grover's
            Algorithm](../../../applications/python/grovers.html#Structure-of-Grover's-Algorithm){.reference
            .internal}
            -   [Step 1:
                Preparation](../../../applications/python/grovers.html#Step-1:-Preparation){.reference
                .internal}
            -   [Good and Bad
                States](../../../applications/python/grovers.html#Good-and-Bad-States){.reference
                .internal}
            -   [Step 2: Oracle
                application](../../../applications/python/grovers.html#Step-2:-Oracle-application){.reference
                .internal}
            -   [Step 3: Amplitude
                amplification](../../../applications/python/grovers.html#Step-3:-Amplitude-amplification){.reference
                .internal}
            -   [Steps 4 and 5: Iteration and
                measurement](../../../applications/python/grovers.html#Steps-4-and-5:-Iteration-and-measurement){.reference
                .internal}
    -   [Quantum
        PageRank](../../../applications/python/quantum_pagerank.html){.reference
        .internal}
        -   [Problem
            Definition](../../../applications/python/quantum_pagerank.html#Problem-Definition){.reference
            .internal}
        -   [Simulating Quantum PageRank by CUDA-Q
            dynamics](../../../applications/python/quantum_pagerank.html#Simulating-Quantum-PageRank-by-CUDA-Q-dynamics){.reference
            .internal}
        -   [Breakdown of
            Terms](../../../applications/python/quantum_pagerank.html#Breakdown-of-Terms){.reference
            .internal}
    -   [The UCCSD Wavefunction
        ansatz](../../../applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](../../../applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](../../../applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](../../../applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](../../../applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](../../../applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](../../../applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [QM/MM simulation: VQE within a Polarizable Embedded
        Framework.](../../../applications/python/qm_mm_pe.html){.reference
        .internal}
        -   [Key
            concepts:](../../../applications/python/qm_mm_pe.html#Key-concepts:){.reference
            .internal}
        -   [PE-VQE-SCF Algorithm
            Steps](../../../applications/python/qm_mm_pe.html#PE-VQE-SCF-Algorithm-Steps){.reference
            .internal}
            -   [Step 1: Initialize (Classical
                pre-processing)](../../../applications/python/qm_mm_pe.html#Step-1:-Initialize-(Classical-pre-processing)){.reference
                .internal}
            -   [Step 2: Build the
                Hamiltonian](../../../applications/python/qm_mm_pe.html#Step-2:-Build-the-Hamiltonian){.reference
                .internal}
            -   [Step 3: Run
                VQE](../../../applications/python/qm_mm_pe.html#Step-3:-Run-VQE){.reference
                .internal}
            -   [Step 4: Update
                Environment](../../../applications/python/qm_mm_pe.html#Step-4:-Update-Environment){.reference
                .internal}
            -   [Step 5: Self-Consistency
                Loop](../../../applications/python/qm_mm_pe.html#Step-5:-Self-Consistency-Loop){.reference
                .internal}
            -   [Requirments:](../../../applications/python/qm_mm_pe.html#Requirments:){.reference
                .internal}
            -   [Example 1: LiH with 2 water
                molecules.](../../../applications/python/qm_mm_pe.html#Example-1:-LiH-with-2-water-molecules.){.reference
                .internal}
            -   [VQE, update environment, and scf
                loop.](../../../applications/python/qm_mm_pe.html#VQE,-update-environment,-and-scf-loop.){.reference
                .internal}
            -   [Example 2: NH3 with 46 water molecule using active
                space.](../../../applications/python/qm_mm_pe.html#Example-2:-NH3-with-46-water-molecule-using-active-space.){.reference
                .internal}
    -   [Sample-Based Krylov Quantum Diagonalization
        (SKQD)](../../../applications/python/skqd.html){.reference
        .internal}
        -   [Why
            SKQD?](../../../applications/python/skqd.html#Why-SKQD?){.reference
            .internal}
        -   [Understanding Krylov
            Subspaces](../../../applications/python/skqd.html#Understanding-Krylov-Subspaces){.reference
            .internal}
            -   [What is a Krylov
                Subspace?](../../../applications/python/skqd.html#What-is-a-Krylov-Subspace?){.reference
                .internal}
            -   [The SKQD
                Algorithm](../../../applications/python/skqd.html#The-SKQD-Algorithm){.reference
                .internal}
        -   [Problem Setup: 22-Qubit Heisenberg
            Model](../../../applications/python/skqd.html#Problem-Setup:-22-Qubit-Heisenberg-Model){.reference
            .internal}
        -   [Krylov State Generation via Repeated
            Evolution](../../../applications/python/skqd.html#Krylov-State-Generation-via-Repeated-Evolution){.reference
            .internal}
        -   [Quantum Measurements and
            Sampling](../../../applications/python/skqd.html#Quantum-Measurements-and-Sampling){.reference
            .internal}
            -   [The Sampling
                Process](../../../applications/python/skqd.html#The-Sampling-Process){.reference
                .internal}
        -   [Classical Post-Processing and
            Diagonalization](../../../applications/python/skqd.html#Classical-Post-Processing-and-Diagonalization){.reference
            .internal}
            -   [The SKQD Algorithm: Matrix Construction
                Details](../../../applications/python/skqd.html#The-SKQD-Algorithm:-Matrix-Construction-Details){.reference
                .internal}
        -   [Results Analysis and
            Convergence](../../../applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](../../../applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
        -   [GPU Acceleration for
            Postprocessing](../../../applications/python/skqd.html#GPU-Acceleration-for-Postprocessing){.reference
            .internal}
    -   [Entanglement Accelerates Quantum
        Simulation](../../../applications/python/entanglement_acc_hamiltonian_simulation.html){.reference
        .internal}
        -   [2. Model
            Definition](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.-Model-Definition){.reference
            .internal}
            -   [2.1 Initial product
                state](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.1-Initial-product-state){.reference
                .internal}
            -   [2.2 QIMF
                Hamiltonian](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.2-QIMF-Hamiltonian){.reference
                .internal}
            -   [2.3 First-Order Trotter Formula
                (PF1)](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.3-First-Order-Trotter-Formula-(PF1)){.reference
                .internal}
            -   [2.4 PF1 step for the QIMF
                partition](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.4-PF1-step-for-the-QIMF-partition){.reference
                .internal}
            -   [2.5 Hamiltonian
                helpers](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.5-Hamiltonian-helpers){.reference
                .internal}
        -   [3. Entanglement
            metrics](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#3.-Entanglement-metrics){.reference
            .internal}
        -   [4. Simulation
            workflow](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#4.-Simulation-workflow){.reference
            .internal}
            -   [4.1 Single-step Trotter
                error](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#4.1-Single-step-Trotter-error){.reference
                .internal}
            -   [4.2 Dual trajectory
                update](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#4.2-Dual-trajectory-update){.reference
                .internal}
        -   [5. Reproducing the paper's Figure
            1a](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#5.-Reproducing-the-paper’s-Figure-1a){.reference
            .internal}
            -   [5.1 Visualising the joint
                behaviour](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#5.1-Visualising-the-joint-behaviour){.reference
                .internal}
            -   [5.2 Interpreting the
                result](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#5.2-Interpreting-the-result){.reference
                .internal}
        -   [6. References and further
            reading](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#6.-References-and-further-reading){.reference
            .internal}
-   [Backends](../backends.html){.reference .internal}
    -   [Circuit Simulation](../simulators.html){.reference .internal}
        -   [State Vector Simulators](../sims/svsims.html){.reference
            .internal}
            -   [CPU](../sims/svsims.html#cpu){.reference .internal}
            -   [Single-GPU](../sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](../sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network Simulators](../sims/tnsims.html){.reference
            .internal}
            -   [Multi-GPU
                multi-node](../sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](../sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](../sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU Simulators](../sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](../sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](../sims/mqpusims.html#multi-qpu-other-backends){.reference
                .internal}
        -   [Noisy Simulators](../sims/noisy.html){.reference .internal}
            -   [Trajectory Noisy
                Simulation](../sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](../sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](../sims/noisy.html#stim){.reference .internal}
        -   [Photonics Simulators](../sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](../sims/photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware (QPUs)](../hardware.html){.reference
        .internal}
        -   [Ion Trap QPUs](iontrap.html){.reference .internal}
            -   [IonQ](iontrap.html#ionq){.reference .internal}
            -   [Quantinuum](iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting QPUs](#){.current .reference .internal}
            -   [Anyon Technologies/Anyon
                Computing](#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](#iqm){.reference .internal}
            -   [OQC](#oqc){.reference .internal}
            -   [Quantum Circuits,
                Inc.](#quantum-circuits-inc){.reference .internal}
        -   [Neutral Atom QPUs](neutralatom.html){.reference .internal}
            -   [Infleqtion](neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](neutralatom.html#pasqal){.reference .internal}
            -   [QuEra
                Computing](neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic QPUs](photonic.html){.reference .internal}
            -   [ORCA
                Computing](photonic.html#orca-computing){.reference
                .internal}
        -   [Quantum Control Systems](qcontrol.html){.reference
            .internal}
            -   [Quantum
                Machines](qcontrol.html#quantum-machines){.reference
                .internal}
    -   [Dynamics Simulation](../dynamics_backends.html){.reference
        .internal}
    -   [Cloud](../cloud.html){.reference .internal}
        -   [Amazon Braket (braket)](../cloud/braket.html){.reference
            .internal}
            -   [Setting
                Credentials](../cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submitting](../cloud/braket.html#submitting){.reference
                .internal}
-   [Dynamics](../../dynamics.html){.reference .internal}
    -   [Quick Start](../../dynamics.html#quick-start){.reference
        .internal}
    -   [Operator](../../dynamics.html#operator){.reference .internal}
    -   [Time-Dependent
        Dynamics](../../dynamics.html#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](../../dynamics.html#super-operator-representation){.reference
        .internal}
    -   [Numerical
        Integrators](../../dynamics.html#numerical-integrators){.reference
        .internal}
    -   [Batch
        simulation](../../dynamics.html#batch-simulation){.reference
        .internal}
    -   [Multi-GPU Multi-Node
        Execution](../../dynamics.html#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](../../dynamics.html#examples){.reference .internal}
-   [CUDA-QX](../../cudaqx/cudaqx.html){.reference .internal}
    -   [CUDA-Q
        Solvers](../../cudaqx/cudaqx.html#cuda-q-solvers){.reference
        .internal}
    -   [CUDA-Q QEC](../../cudaqx/cudaqx.html#cuda-q-qec){.reference
        .internal}
-   [Installation](../../install/install.html){.reference .internal}
    -   [Local
        Installation](../../install/local_installation.html){.reference
        .internal}
        -   [Introduction](../../install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](../../install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](../../install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](../../install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](../../install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](../../install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](../../install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](../../install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](../../install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](../../install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](../../install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](../../install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](../../install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](../../install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](../../install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](../../install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](../../install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](../../install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](../../install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](../../install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](../../install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](../../install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](../../install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
        -   [Next
            Steps](../../install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](../../install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](../../install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](../../install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](../../install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](../../install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](../../install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](../../install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](../../install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](../../install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](../../install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](../../install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](../../integration/integration.html){.reference
    .internal}
    -   [Downstream CMake
        Integration](../../integration/cmake_app.html){.reference
        .internal}
    -   [Combining CUDA with
        CUDA-Q](../../integration/cuda_gpu.html){.reference .internal}
    -   [Integrating with Third-Party
        Libraries](../../integration/libraries.html){.reference
        .internal}
        -   [Calling a CUDA-Q library from
            C++](../../integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](../../integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](../../integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](../../extending/extending.html){.reference .internal}
    -   [Add a new Hardware
        Backend](../../extending/backend.html){.reference .internal}
        -   [Overview](../../extending/backend.html#overview){.reference
            .internal}
        -   [Server Helper
            Implementation](../../extending/backend.html#server-helper-implementation){.reference
            .internal}
            -   [Directory
                Structure](../../extending/backend.html#directory-structure){.reference
                .internal}
            -   [Server Helper
                Class](../../extending/backend.html#server-helper-class){.reference
                .internal}
            -   [[`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](../../extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](../../extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent [`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](../../extending/backend.html#update-parent-cmakelists-txt){.reference
                .internal}
        -   [Testing](../../extending/backend.html#testing){.reference
            .internal}
            -   [Unit
                Tests](../../extending/backend.html#unit-tests){.reference
                .internal}
            -   [Mock
                Server](../../extending/backend.html#mock-server){.reference
                .internal}
            -   [Python
                Tests](../../extending/backend.html#python-tests){.reference
                .internal}
            -   [Integration
                Tests](../../extending/backend.html#integration-tests){.reference
                .internal}
        -   [Documentation](../../extending/backend.html#documentation){.reference
            .internal}
        -   [Example
            Usage](../../extending/backend.html#example-usage){.reference
            .internal}
        -   [Code
            Review](../../extending/backend.html#code-review){.reference
            .internal}
        -   [Maintaining a
            Backend](../../extending/backend.html#maintaining-a-backend){.reference
            .internal}
        -   [Conclusion](../../extending/backend.html#conclusion){.reference
            .internal}
    -   [Create a new NVQIR
        Simulator](../../extending/nvqir_simulator.html){.reference
        .internal}
        -   [[`CircuitSimulator`{.code .docutils .literal
            .notranslate}]{.pre}](../../extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](../../extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
    -   [Working with CUDA-Q
        IR](../../extending/cudaq_ir.html){.reference .internal}
    -   [Create an MLIR Pass for
        CUDA-Q](../../extending/mlir_pass.html){.reference .internal}
-   [Specifications](../../../specification/index.html){.reference
    .internal}
    -   [Language
        Specification](../../../specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](../../../specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](../../../specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](../../../specification/cudaq/types.html){.reference
            .internal}
            -   [3.1. [`cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. [`cudaq::qubit`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](../../../specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](../../../specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. [`cudaq::spin_op`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](../../../specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on [`cudaq::qubit`{.code .docutils
                .literal
                .notranslate}]{.pre}](../../../specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](../../../specification/cudaq/kernels.html){.reference
            .internal}
        -   [7. Sub-circuit
            Synthesis](../../../specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](../../../specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](../../../specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](../../../specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](../../../specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11.
            Platform](../../../specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](../../../specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. [`cudaq::sample`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. [`cudaq::run`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. [`cudaq::observe`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. [`cudaq::optimizer`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../../../specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. [`cudaq::gradient`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../../../specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](../../../specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](../../../specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](../../../specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](../../../specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](../../../specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](../../../specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](../../../specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](../../../specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](../../../specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](../../../specification/quake-dialect.html#motivation){.reference
            .internal}
-   [API Reference](../../../api/api.html){.reference .internal}
    -   [C++ API](../../../api/languages/cpp_api.html){.reference
        .internal}
        -   [Operators](../../../api/languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](../../../api/languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](../../../api/languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](../../../api/languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](../../../api/languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](../../../api/languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Platform](../../../api/languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](../../../api/languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](../../../api/languages/cpp_api.html#namespaces){.reference
            .internal}
    -   [Python API](../../../api/languages/python_api.html){.reference
        .internal}
        -   [Program
            Construction](../../../api/languages/python_api.html#program-construction){.reference
            .internal}
            -   [[`make_kernel()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [[`PyKernel`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [[`Kernel`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [[`PyKernelDecorator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [[`kernel()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](../../../api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [[`sample_async()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [[`run()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [[`run_async()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [[`observe()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [[`observe_async()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [[`get_state()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [[`get_state_async()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [[`vqe()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [[`draw()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [[`translate()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [[`estimate_resources()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
        -   [Backend
            Configuration](../../../api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [[`has_target()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [[`get_target()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [[`get_targets()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [[`set_target()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [[`reset_target()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [[`set_noise()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [[`unset_noise()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [[`register_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [[`unregister_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [[`cudaq.apply_noise()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [[`initialize_cudaq()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [[`num_available_gpus()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [[`set_random_seed()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](../../../api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [[`evolve()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [[`evolve_async()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [[`Schedule`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [[`BaseIntegrator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [[`InitialState`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [[`InitialStateType`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [[`IntermediateResultSave`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](../../../api/languages/python_api.html#operators){.reference
            .internal}
            -   [[`OperatorSum`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [[`ProductOperator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [[`ElementaryOperator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [[`ScalarOperator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [[`RydbergHamiltonian`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [[`SuperOperator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [[`operators.define()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [[`operators.instantiate()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](../../../api/languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](../../../api/languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](../../../api/languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](../../../api/languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](../../../api/languages/python_api.html#data-types){.reference
            .internal}
            -   [[`SimulationPrecision`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [[`Target`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [[`State`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [[`Tensor`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [[`QuakeValue`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [[`qubit`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [[`qreg`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [[`qvector`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [[`ComplexMatrix`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [[`SampleResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [[`AsyncSampleResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [[`ObserveResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [[`AsyncObserveResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [[`AsyncStateResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [[`OptimizationResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [[`EvolveResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [[`AsyncEvolveResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [[`Resources`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.Resources){.reference
                .internal}
            -   [Optimizers](../../../api/languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](../../../api/languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](../../../api/languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](../../../api/languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [[`initialize()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [[`rank()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [[`num_ranks()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [[`all_gather()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [[`broadcast()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [[`is_initialized()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [[`finalize()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](../../../api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
    -   [Quantum Operations](../../../api/default_ops.html){.reference
        .internal}
        -   [Unitary Operations on
            Qubits](../../../api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [[`x`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#x){.reference
                .internal}
            -   [[`y`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#y){.reference
                .internal}
            -   [[`z`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#z){.reference
                .internal}
            -   [[`h`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#h){.reference
                .internal}
            -   [[`r1`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#r1){.reference
                .internal}
            -   [[`rx`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#rx){.reference
                .internal}
            -   [[`ry`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#ry){.reference
                .internal}
            -   [[`rz`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#rz){.reference
                .internal}
            -   [[`s`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#s){.reference
                .internal}
            -   [[`t`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#t){.reference
                .internal}
            -   [[`swap`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#swap){.reference
                .internal}
            -   [[`u3`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#u3){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](../../../api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](../../../api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#mz){.reference
                .internal}
            -   [[`mx`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#mx){.reference
                .internal}
            -   [[`my`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](../../../api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](../../../api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [[`create`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#create){.reference
                .internal}
            -   [[`annihilate`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#annihilate){.reference
                .internal}
            -   [[`phase_shift`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#phase-shift){.reference
                .internal}
            -   [[`beam_splitter`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](../../../versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](../../../index.html)

::: wy-nav-content
::: rst-content
::: {role="navigation" aria-label="Page navigation"}
-   [](../../../index.html){.icon .icon-home aria-label="Home"}
-   [CUDA-Q Backends](../backends.html)
-   [Quantum Hardware (QPU)](../hardware.html)
-   Superconducting
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](iontrap.html "Ion Trap"){.btn .btn-neutral .float-left
accesskey="p"} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](backend_iqm.html "IQM Backend Advanced Use Cases"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#superconducting .section}
# Superconducting[¶](#superconducting "Permalink to this heading"){.headerlink}

::: {#anyon-technologies-anyon-computing .section}
## Anyon Technologies/Anyon Computing[¶](#anyon-technologies-anyon-computing "Permalink to this heading"){.headerlink}

::: {#setting-credentials .section}
[]{#anyon-backend}

### Setting Credentials[¶](#setting-credentials "Permalink to this heading"){.headerlink}

Programmers of CUDA-Q may access the Anyon API from either C++ or
Python. Anyon requires a credential configuration file with username and
password. The configuration file can be generated as follows, replacing
the [`<username>`{.docutils .literal .notranslate}]{.pre} and
[`<password>`{.docutils .literal .notranslate}]{.pre} in the first line
with your Anyon Technologies account details. The credential in the file
will be used by CUDA-Q to login to Anyon quantum services and will be
updated by CUDA-Q with an obtained API token and refresh token. Note,
the credential line will be deleted in the updated configuration file.

::: {.highlight-bash .notranslate}
::: highlight
    echo 'credentials: {"username":"<username>","password":"<password>"}' > $HOME/.anyon_config
:::
:::

Users can also login and get the keys manually using the following
commands:

::: {.highlight-bash .notranslate}
::: highlight
    # You may need to run: `apt-get update && apt-get install curl jq`
    curl -X POST --user "<username>:<password>"  -H "Content-Type: application/json" \
    https://api.anyon.cloud:5000/login > credentials.json
    id_token=`cat credentials.json | jq -r '."id_token"'`
    refresh_token=`cat credentials.json | jq -r '."refresh_token"'`
    echo "key: $id_token" > ~/.anyon_config
    echo "refresh: $refresh_token" >> ~/.anyon_config
:::
:::

The path to the configuration can be specified as an environment
variable:

::: {.highlight-bash .notranslate}
::: highlight
    export CUDAQ_ANYON_CREDENTIALS=$HOME/.anyon_config
:::
:::
:::

::: {#submitting .section}
### Submitting[¶](#submitting "Permalink to this heading"){.headerlink}

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
The target to which quantum kernels are submitted can be controlled with
the [`cudaq.set_target()`{.docutils .literal .notranslate}]{.pre}
function.

To execute your kernels using Anyon Technologies backends, specify which
machine to submit quantum kernels to by setting the [`machine`{.code
.docutils .literal .notranslate}]{.pre} parameter of the target. If
[`machine`{.code .docutils .literal .notranslate}]{.pre} is not
specified, the default machine will be [`telegraph-8q`{.docutils
.literal .notranslate}]{.pre}.

::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target('anyon', machine='telegraph-8q')
:::
:::

As shown above, [`telegraph-8q`{.docutils .literal .notranslate}]{.pre}
is an example of a physical QPU.

To emulate the Anyon Technologies machine locally, without submitting
through the cloud, you can also set the [`emulate`{.docutils .literal
.notranslate}]{.pre} flag to [`True`{.docutils .literal
.notranslate}]{.pre}. This will emit any target specific compiler
warnings and diagnostics, before running a noise free emulation.

::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target('anyon', emulate=True)
:::
:::

The number of shots for a kernel execution can be set through the
[`shots_count`{.docutils .literal .notranslate}]{.pre} argument to
[`cudaq.sample`{.docutils .literal .notranslate}]{.pre} or
[`cudaq.observe`{.docutils .literal .notranslate}]{.pre}. By default,
the [`shots_count`{.docutils .literal .notranslate}]{.pre} is set to
1000.

::: {.highlight-python .notranslate}
::: highlight
    cudaq.sample(kernel, shots_count=10000)
:::
:::
:::

C++

::: {.tab-content .docutils}
To target quantum kernel code for execution in the Anyon Technologies
backends, pass the flag [`--target`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`anyon`{.docutils .literal .notranslate}]{.pre} to the
[`nvq++`{.docutils .literal .notranslate}]{.pre} compiler. CUDA-Q will
authenticate via the Anyon Technologies REST API using the credential in
your configuration file.

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target anyon --<backend-type> <machine> src.cpp ...
:::
:::

To execute your kernels using Anyon Technologies backends, pass the
[`--anyon-machine`{.docutils .literal .notranslate}]{.pre} flag to the
[`nvq++`{.docutils .literal .notranslate}]{.pre} compiler as the
[`--<backend-type>`{.docutils .literal .notranslate}]{.pre} to specify
which machine to submit quantum kernels to:

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target anyon --anyon-machine telegraph-8q src.cpp ...
:::
:::

where [`telegraph-8q`{.docutils .literal .notranslate}]{.pre} is an
example of a physical QPU (Architecture: Telegraph, Qubit Count: 8).

Currently, [`telegraph-8q`{.docutils .literal .notranslate}]{.pre} and
[`berkeley-25q`{.docutils .literal .notranslate}]{.pre} are available
for access over CUDA-Q.

To emulate the Anyon Technologies machine locally, without submitting
through the cloud, you can also pass the [`--emulate`{.docutils .literal
.notranslate}]{.pre} flag as the [`--<backend-type>`{.docutils .literal
.notranslate}]{.pre} to [`nvq++`{.docutils .literal
.notranslate}]{.pre}. This will emit any target specific compiler
warnings and diagnostics, before running a noise free emulation.

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target anyon --emulate src.cpp
:::
:::
:::
:::

To see a complete example, take a look at [[Anyon examples]{.std
.std-ref}](../../examples/hardware_providers.html#anyon-examples){.reference
.internal}.
:::
:::

::: {#iqm .section}
## IQM[¶](#iqm "Permalink to this heading"){.headerlink}

[IQM Resonance](https://meetiqm.com/products/iqm-resonance/){.reference
.external} offers access to various different IQM quantum computers. The
machines available there will be constantly extended as development
progresses. Programmers of CUDA-Q may use IQM Resonance with either C++
or Python.

With this version it is no longer necessary to define the target QPU
architecture in the code or at compile time. The IQM backend integration
now contacts at runtime the configured IQM server and fetches the active
dynamic quantum architecture of the QPU. This is then used as input to
transpile the quantum kernel code just-in-time for the target QPU
topology. By setting the environment variable
[`IQM_SERVER_URL`{.docutils .literal .notranslate}]{.pre} the target
server can be selected just before executing the program. As result the
python script or the compiled C++ program can be executed on different
QPUs without recompilation or code changes.

Please find also more documentation after logging in to the IQM
Resonance portal.

::: {#id1 .section}
### Setting Credentials[¶](#id1 "Permalink to this heading"){.headerlink}

Create a free account on the [IQM Resonance
portal](https://meetiqm.com/products/iqm-resonance/){.reference
.external} and log-in. Navigate to the account profile (top right).
There generate an "API Token" and copy the generated token-string. Set
the environment variable [`IQM_TOKEN`{.docutils .literal
.notranslate}]{.pre} to contain the value of the token-string. The IQM
backend integration will use this as authorization token at the IQM
server.
:::

::: {#id2 .section}
### Submitting[¶](#id2 "Permalink to this heading"){.headerlink}

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
The target to which quantum kernels are submitted can be controlled with
the [`cudaq.set_target()`{.docutils .literal .notranslate}]{.pre}
function.

::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target("iqm", url="https://<IQM Server>/")
:::
:::

Please note that setting the environment variable
[`IQM_SERVER_URL`{.docutils .literal .notranslate}]{.pre} takes
precedence over the URL configured in the code.
:::

C++

::: {.tab-content .docutils}
To target quantum kernel code for execution on an IQM Server, pass the
[`--target`{.docutils .literal .notranslate}]{.pre}` `{.docutils
.literal .notranslate}[`iqm`{.docutils .literal .notranslate}]{.pre}
option to the [`nvq++`{.docutils .literal .notranslate}]{.pre} compiler.

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target iqm src.cpp
:::
:::

Once the binary for an IQM QPU is compiled, it can be executed against
any IQM Server by setting the environment variable
[`IQM_SERVER_URL`{.docutils .literal .notranslate}]{.pre} as shown here:

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target iqm src.cpp -o program
    IQM_SERVER_URL="https://demo.qc.iqm.fi/" ./program
:::
:::
:::
:::

To see a complete example for using IQM server backends, take a look at
[[IQM examples]{.std
.std-ref}](../../examples/hardware_providers.html#iqm-examples){.reference
.internal}.
:::

::: {#advanced-use-cases .section}
### Advanced use cases[¶](#advanced-use-cases "Permalink to this heading"){.headerlink}

The IQM backend integration offers more options for advanced use cases.
Please find these here:

::: {.toctree-wrapper .compound}
-   [IQM backend advanced use cases](backend_iqm.html){.reference
    .internal}
    -   [Emulation Mode](backend_iqm.html#emulation-mode){.reference
        .internal}
    -   [Setting the Number of
        Shots](backend_iqm.html#setting-the-number-of-shots){.reference
        .internal}
    -   [Using Credentials Saved in a
        File](backend_iqm.html#using-credentials-saved-in-a-file){.reference
        .internal}
:::
:::
:::

::: {#oqc .section}
## OQC[¶](#oqc "Permalink to this heading"){.headerlink}

[Oxford Quantum Circuits](https://oxfordquantumcircuits.com/){.reference
.external} (OQC) is currently providing CUDA-Q integration for multiple
Quantum Processing Unit types. The 8 qubit ring topology Lucy device and
the 32 qubit Kagome lattice topology Toshiko device are both supported
via machine options described below.

::: {#id3 .section}
### Setting Credentials[¶](#id3 "Permalink to this heading"){.headerlink}

In order to use the OQC devices you will need to register. Registration
is achieved by contacting
[`oqc_qcaas_support@oxfordquantumcircuits.com`{.code .docutils .literal
.notranslate}]{.pre}.

Once registered you will be able to authenticate with your
[`email`{.docutils .literal .notranslate}]{.pre} and
[`password`{.docutils .literal .notranslate}]{.pre}

There are three environment variables that the OQC target will look for
during configuration:

1.  [`OQC_URL`{.docutils .literal .notranslate}]{.pre}

2.  [`OQC_EMAIL`{.docutils .literal .notranslate}]{.pre}

3.  [`OQC_PASSWORD`{.docutils .literal .notranslate}]{.pre} - is
    mandatory
:::

::: {#id4 .section}
### Submitting[¶](#id4 "Permalink to this heading"){.headerlink}

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
To set which OQC URL, set the [`url`{.code .docutils .literal
.notranslate}]{.pre} parameter. To set which OQC email, set the
[`email`{.code .docutils .literal .notranslate}]{.pre} parameter. To set
which OQC machine, set the [`machine`{.code .docutils .literal
.notranslate}]{.pre} parameter.

::: {.highlight-python .notranslate}
::: highlight
    import os
    import cudaq
    os.environ['OQC_PASSWORD'] = password
    cudaq.set_target("oqc", url=url, machine="lucy")
:::
:::

You can then execute a kernel against the platform using the OQC Lucy
device

To emulate the OQC device locally, without submitting through the OQC
QCaaS services, you can set the [`emulate`{.docutils .literal
.notranslate}]{.pre} flag to [`True`{.docutils .literal
.notranslate}]{.pre}. This will emit any target specific compiler
warnings and diagnostics, before running a noise free emulation.

::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target("oqc", emulate=True)
:::
:::
:::

C++

::: {.tab-content .docutils}
To target quantum kernel code for execution on the OQC platform, provide
the flag [`--target`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal .notranslate}[`oqc`{.docutils
.literal .notranslate}]{.pre} to the [`nvq++`{.docutils .literal
.notranslate}]{.pre} compiler.

Users may provide their [`email`{.code .docutils .literal
.notranslate}]{.pre} and [`url`{.code .docutils .literal
.notranslate}]{.pre} as extra arguments

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target oqc --oqc-email <email> --oqc-url <url> src.cpp -o executable
:::
:::

Where both environment variables and extra arguments are supplied,
precedent is given to the extra arguments. To run the output, provide
the runtime loaded variables and invoke the pre-built executable

::: {.highlight-bash .notranslate}
::: highlight
    OQC_PASSWORD=<password> ./executable
:::
:::

To emulate the OQC device locally, without submitting through the OQC
QCaaS services, you can pass the [`--emulate`{.docutils .literal
.notranslate}]{.pre} flag to [`nvq++`{.docutils .literal
.notranslate}]{.pre}. This will emit any target specific compiler
warnings and diagnostics, before running a noise free emulation.

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --emulate --target oqc src.cpp -o executable
:::
:::

::: {.admonition .note}
Note

The oqc target supports a [`--oqc-machine`{.docutils .literal
.notranslate}]{.pre} option. The default is the 8 qubit Lucy device. You
can set this to be either [`toshiko`{.docutils .literal
.notranslate}]{.pre} or [`lucy`{.docutils .literal .notranslate}]{.pre}
via this flag.
:::
:::
:::

::: {.admonition .note}
Note

The OQC quantum assembly toolchain (qat) which is used to compile and
execute instructions can be found on github as
[oqc-community/qat](https://github.com/oqc-community/qat){.reference
.external}
:::

To see a complete example, take a look at [[OQC examples]{.std
.std-ref}](../../examples/hardware_providers.html#oqc-examples){.reference
.internal}.
:::
:::

::: {#quantum-circuits-inc .section}
## Quantum Circuits, Inc.[¶](#quantum-circuits-inc "Permalink to this heading"){.headerlink}

Quantum Circuits offers users the ability to execute CUDA-Q programs on
its [Seeker QPU](https://quantumcircuits.com/product/#seeker){.reference
.external} and simulate them using its simulator,
[AquSim](https://quantumcircuits.com/product/#simulator){.reference
.external}. The Seeker is the first dual-rail qubit QPU available over
the cloud today, and through CUDA-Q users have access to its universal
gate set, high fidelity operations, and fast throughput. Upcoming
releases of CUDA-Q will continue to evolve these capabilities to include
real-time control flow and access to an expanded collection of
actionable data enabled by the Quantum Circuits error aware technology.

AquSim models error detection and real-time control of Quantum Circuits'
Dual-Rail Cavity Qubit systems, and uses a Monte Carlo approach to do so
on a shot-by-shot basis. The supported features include all of the
single and two-qubit gates offered by CUDA-Q. AquSim additionally
supports real-time conditional logic enabled by feed-forward capability.
Noise modeling is offered, effectively enabling users to emulate the
execution of programs on the Seeker QPU and thereby providing a powerful
application prototyping tool to be leveraged in advance of execution on
hardware.

With C++ and Python programming supported, users are able to prototype,
test and explore quantum applications in CUDA-Q on the Seeker and
AquSim. Users who wish to get started with running CUDA-Q with Quantum
Circuits should visit our
[Explore](https://quantumcircuits.com/explore/){.reference .external}
page to learn more about the Quantum Circuits Select Quantum Release
Program.

::: {#installation-getting-started .section}
### Installation & Getting Started[¶](#installation-getting-started "Permalink to this heading"){.headerlink}

Until CUDA-Q release 0.13.0 is available, the integration with Quantum
Circuits will be supported through the [nightly build Docker
images](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nightly/containers/cuda-quantum/tags){.reference
.external}.

Instructions on how to install and get started with CUDA-Q using Docker
can be found [[here]{.std
.std-ref}](../../install/local_installation.html#install-docker-image){.reference
.internal}.

You may present your user token to Quantum Circuits via CUDA-Q by
setting an environment variable named [`QCI_AUTH_TOKEN`{.code .docutils
.literal .notranslate}]{.pre} before running your CUDA-Q program.

For example:

::: {.highlight-bash .notranslate}
::: highlight
    export QCI_AUTH_TOKEN="example-token"
:::
:::

Tokens are provided as part of the Strategic Quantum Release Program.
Please visit our
[Explore](https://quantumcircuits.com/explore/){.reference .external}
page to learn more.
:::

::: {#using-cuda-q-with-quantum-circuits .section}
### Using CUDA-Q with Quantum Circuits[¶](#using-cuda-q-with-quantum-circuits "Permalink to this heading"){.headerlink}

Quantum Circuits' Seeker system detects errors in real-time and returns
not just 0s and 1s as the measurement outcomes, but unique results
tagged as -1, which indicate that an erasure was detected on the
Dual-Rail Cavity Qubit. AquSim emulates this execution as well, enabling
users to model error aware programs in advance of execution on the QPU.
While -1 data is not yet available via the CUDA-Q API, the user still
has insight into these dynamics through the number of shots that are
collected in a given run.
:::

::: {#yield .section}
### Yield[¶](#yield "Permalink to this heading"){.headerlink}

Quantum Circuits architecture can detect errors in measurements. The
target will return to the user the outcome from every measurement for
every shot, regardless of whether errors were detected. However, the
data from a shot in which any of the measurements had an error detected
will:

-   Every **RESULT** where an error is detected will be [`-1`{.docutils
    .literal .notranslate}]{.pre} (instead of [`0`{.docutils .literal
    .notranslate}]{.pre} or [`1`{.docutils .literal
    .notranslate}]{.pre}).

-   The shot will be marked with an **exit code** of [`1`{.docutils
    .literal .notranslate}]{.pre} (instead of [`0`{.docutils .literal
    .notranslate}]{.pre}).

-   It will be **excluded** from the histogram.

Apart from an ideal simulation, most jobs will include at least some
shots for which errors were detected.

The shots that have no errors detected are referred to as
**post-selected** and will have an exit code of [`0`{.docutils .literal
.notranslate}]{.pre}. The **yield** represents the fraction of executed
shots that are not rejected due to detected errors:

::: {.math .notranslate .nohighlight}
\\\[\\text{yield} = \\frac{\\text{number of post-selected
shots}}{\\text{number of shots executed}}\\\]
:::

The yield depends on the number of qubits and the depth of the circuit.
:::

::: {#options .section}
### Options[¶](#options "Permalink to this heading"){.headerlink}

**machine**

:   This is a string option with 2 supported values.

    -   **Seeker**

        -   Name of the QPU supported by Quantum Circuits.

        -   Supports up to **8 qubit** programs and the
            [`base_profile`{.docutils .literal .notranslate}]{.pre}.

        -   Regardless of whether the method is [`execute`{.docutils
            .literal .notranslate}]{.pre} or [`simulate`{.docutils
            .literal .notranslate}]{.pre}, the program will be **fully
            compiled** for strict validation of suitability to run on
            the QPU.

    -   **AquSim**

        -   This "machine" is not associated with a specific QPU and not
            strictly validated.

        -   Supports up to **25 qubits**, a **square grid coupling
            map**, and the [`adaptive_profile`{.docutils .literal
            .notranslate}]{.pre}.

**method**

:   This is a string option with 2 supported values.

    -   **execute**

        -   If [`machine="Seeker"`{.docutils .literal
            .notranslate}]{.pre}, the program will run on the QPU
            (depending on availability).

        -   Not supported if [`machine="AquSim"`{.docutils .literal
            .notranslate}]{.pre}.

    -   **simulate**

        -   The program will be run in [`AquSim`{.docutils .literal
            .notranslate}]{.pre}.

**noisy**

:   This boolean option is only supported for
    [`method="simulate"`{.docutils .literal .notranslate}]{.pre}.

    -   **True**

        -   [`AquSim`{.docutils .literal .notranslate}]{.pre} will
            simulate noise and error detection using a **Dual-Rail
            statevector-based noise model** on a transpiled program.

    -   **False**

        -   An **ideal simulation**.

**repeat_until_shots_requested**

:   This is a boolean option.

    -   **True**

        -   The machine will return as many post-selected shots as were
            requested (unless an upper limit of shots executed is
            encountered first).

        -   The **execution time is proportional to 1 / yield**.

    -   **False**

        -   The machine will execute **exactly the number of shots
            requested**, regardless of how many errors are detected.

        -   The execution time does **not depend on yield**.
:::

::: {#id5 .section}
### Submitting[¶](#id5 "Permalink to this heading"){.headerlink}

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
To set the target to Quantum Circuits, add the following to your Python
program:

::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target('qci')
    [... your Python here]
:::
:::

To run on AquSim, simply execute the script using your Python
interpreter.

To specify which Quantum Circuits machine to use, set the
[`machine`{.code .docutils .literal .notranslate}]{.pre} parameter:

::: {.highlight-python .notranslate}
::: highlight
    # The default machine is AquSim
    cudaq.set_target('qci', machine='AquSim')
    # or
    cudaq.set_target('qci', machine='Seeker')
:::
:::

You can control the execution method using the [`method`{.code .docutils
.literal .notranslate}]{.pre} parameter:

::: {.highlight-python .notranslate}
::: highlight
    # For simulation (default)
    cudaq.set_target('Seeker', method='simulate')
    # For hardware execution
    cudaq.set_target('Seeker', method='execute')
:::
:::

For noisy simulation, you can enable the [`noisy`{.code .docutils
.literal .notranslate}]{.pre} parameter:

::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target('qci', noisy=True)
:::
:::

When collecting shots, you can ensure the requested number of shots are
obtained by enabling the [`repeat_until_shots_requested`{.code .docutils
.literal .notranslate}]{.pre} parameter:

::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target('qci', repeat_until_shots_requested=True)
:::
:::
:::

C++

::: {.tab-content .docutils}
When executing programs in C++, they must first be compiled using the
CUDA-Q nvq++ compiler, and then submitted to run on the Seeker or
AquSim.

Note that your token is fetched from your environment at run time, not
at compile time.

In the example below, the compilation step shows two flags being passed
to the nvq++ compiler: the Quantum Circuits target [`--target`{.code
.docutils .literal .notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`qci`{.code .docutils .literal .notranslate}]{.pre}, and
the output file [`-o`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`example.x`{.code .docutils .literal .notranslate}]{.pre}.
The second line executes the program against AquSim. Here are the shell
commands in full:

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ example.cpp --target qci -o example.x
    ./example.x
:::
:::

To specify which Quantum Circuits machine to use, pass the
[`--qci-machine`{.docutils .literal .notranslate}]{.pre} flag:

::: {.highlight-bash .notranslate}
::: highlight
    # The default machine is AquSim
    nvq++ --target qci --qci-machine AquSim src.cpp -o example.x
    # or
    nvq++ --target qci --qci-machine Seeker src.cpp -o example.x
:::
:::

You can control the execution method using the [`--qci-method`{.docutils
.literal .notranslate}]{.pre} flag:

::: {.highlight-bash .notranslate}
::: highlight
    # For simulation (default)
    nvq++ --target qci --qci-machine Seeker --qci-method simulate src.cpp -o example.x
    # For hardware execution
    nvq++ --target qci --qci-machine Seeker --qci-method execute src.cpp -o example.x
:::
:::

For noisy simulation, you can set the [`--qci-noisy`{.docutils .literal
.notranslate}]{.pre} argument to [`true`{.code .docutils .literal
.notranslate}]{.pre}:

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target qci --qci-noisy true src.cpp -o example.x
:::
:::

When collecting shots, you can ensure the requested number of shots are
obtained with the [`--qci-repeat_until_shots_requested`{.docutils
.literal .notranslate}]{.pre} argument:

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target qci --qci-repeat_until_shots_requested true src.cpp -o example.x
:::
:::
:::
:::

::: {.admonition .note}
Note

By default, only successful shots are presented to the user and may be
fewer than the requested number. Enabling
[`repeat_until_shots_requested`{.code .docutils .literal
.notranslate}]{.pre} ensures the full requested shot count is collected,
at the cost of increased execution time.
:::

To see a complete example of using Quantum Circuits' backends, please
take a look at the [[Quantum Circuits examples]{.std
.std-ref}](../../examples/hardware_providers.html#quantum-circuits-examples){.reference
.internal}.

::: {.admonition .note}
Note

In local emulation mode ([`emulate`{.docutils .literal
.notranslate}]{.pre} flag set to [`True`{.docutils .literal
.notranslate}]{.pre}), the program will be executed on the [[default
simulator]{.std
.std-ref}](../sims/svsims.html#default-simulator){.reference .internal}.
The environment variable [`CUDAQ_DEFAULT_SIMULATOR`{.docutils .literal
.notranslate}]{.pre} can be used to change the emulation simulator.

For example, the simulation floating point accuracy and/or the
simulation capabilities (e.g., maximum number of qubits, supported
quantum gates), depend on the selected simulator.

Any environment variables must be set prior to setting the target or
running "[`import`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`cudaq`{.code .docutils .literal .notranslate}]{.pre}".
:::
:::
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](iontrap.html "Ion Trap"){.btn .btn-neutral .float-left
accesskey="p" rel="prev"} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](backend_iqm.html "IQM Backend Advanced Use Cases"){.btn
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
