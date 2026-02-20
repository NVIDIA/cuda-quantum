::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../index.html){.icon .icon-home}

::: version
pr-4013
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](../using/quick_start.html){.reference .internal}
    -   [Install
        CUDA-Q](../using/quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](../using/quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q
        Academic](../using/quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](../using/basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q
        Kernel?](../using/basics/kernel_intro.html){.reference
        .internal}
    -   [Building your first CUDA-Q
        Program](../using/basics/build_kernel.html){.reference
        .internal}
    -   [Running your first CUDA-Q
        Program](../using/basics/run_kernel.html){.reference .internal}
        -   [Sample](../using/basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](../using/basics/run_kernel.html#run){.reference
            .internal}
        -   [Observe](../using/basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](../using/basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](../using/basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](../using/basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
-   [Examples](../using/examples/examples.html){.reference .internal}
    -   [Introduction](../using/examples/introduction.html){.reference
        .internal}
    -   [Building
        Kernels](../using/examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](../using/examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](../using/examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](../using/examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](../using/examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](../using/examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](../using/examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](../using/examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](../using/examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](../using/examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](../using/examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](../using/examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](../using/examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](../using/examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring
        Kernels](../using/examples/measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](../using/examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](../examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](../examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](../examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing
        Kernels](../using/examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](../using/examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](../using/examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](../using/examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](../using/examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](../using/examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](../using/examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](../using/examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](../using/examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](../using/examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](../using/examples/expectation_values.html){.reference
        .internal}
        -   [Parallelizing across Multiple
            Processors](../using/examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](../using/examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](../using/examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../using/examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../using/examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](../using/examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](../using/examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends ([`remote-mqpu`{.code .docutils
            .literal
            .notranslate}]{.pre})](../using/examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
            .internal}
    -   [Optimizers &
        Gradients](../examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [CUDA-Q Optimizer
            Overview](../examples/python/optimizers_gradients.html#CUDA-Q-Optimizer-Overview){.reference
            .internal}
            -   [Gradient-Free Optimizers (no gradients
                required):](../examples/python/optimizers_gradients.html#Gradient-Free-Optimizers-(no-gradients-required):){.reference
                .internal}
            -   [Gradient-Based Optimizers (require
                gradients):](../examples/python/optimizers_gradients.html#Gradient-Based-Optimizers-(require-gradients):){.reference
                .internal}
        -   [1. Built-in CUDA-Q Optimizers and
            Gradients](../examples/python/optimizers_gradients.html#1.-Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
            -   [1.1 Adam Optimizer with Parameter
                Configuration](../examples/python/optimizers_gradients.html#1.1-Adam-Optimizer-with-Parameter-Configuration){.reference
                .internal}
            -   [1.2 SGD (Stochastic Gradient Descent)
                Optimizer](../examples/python/optimizers_gradients.html#1.2-SGD-(Stochastic-Gradient-Descent)-Optimizer){.reference
                .internal}
            -   [1.3 SPSA (Simultaneous Perturbation Stochastic
                Approximation)](../examples/python/optimizers_gradients.html#1.3-SPSA-(Simultaneous-Perturbation-Stochastic-Approximation)){.reference
                .internal}
        -   [2. Third-Party
            Optimizers](../examples/python/optimizers_gradients.html#2.-Third-Party-Optimizers){.reference
            .internal}
        -   [3. Parallel Parameter Shift
            Gradients](../examples/python/optimizers_gradients.html#3.-Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](../examples/python/noisy_simulations.html){.reference
        .internal}
    -   [PTSBE End-to-End
        Workflow](../examples/python/ptsbe_end_to_end_workflow.html){.reference
        .internal}
        -   [1. Set up the
            environment](../examples/python/ptsbe_end_to_end_workflow.html#1.-Set-up-the-environment){.reference
            .internal}
        -   [2. Define the circuit and noise
            model](../examples/python/ptsbe_end_to_end_workflow.html#2.-Define-the-circuit-and-noise-model){.reference
            .internal}
        -   [3. Run PTSBE
            sampling](../examples/python/ptsbe_end_to_end_workflow.html#3.-Run-PTSBE-sampling){.reference
            .internal}
        -   [4. Compare with standard (density-matrix)
            sampling](../examples/python/ptsbe_end_to_end_workflow.html#4.-Compare-with-standard-(density-matrix)-sampling){.reference
            .internal}
        -   [5. Return execution
            data](../examples/python/ptsbe_end_to_end_workflow.html#5.-Return-execution-data){.reference
            .internal}
        -   [6. Two API
            options:](../examples/python/ptsbe_end_to_end_workflow.html#6.-Two-API-options:){.reference
            .internal}
    -   [PTSBE Accuracy
        Validation](../examples/python/ptsbe_accuracy_validation.html){.reference
        .internal}
    -   [Constructing
        Operators](../using/examples/operators.html){.reference
        .internal}
        -   [Constructing Spin
            Operators](../using/examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](../using/examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](../examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](../examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](../using/examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](../using/examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](../using/examples/hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](../using/examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](../using/examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](../using/examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](../using/examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](../using/examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](../using/examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](../using/examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum Circuits,
            Inc.](../using/examples/hardware_providers.html#quantum-circuits-inc){.reference
            .internal}
        -   [Quantum
            Machines](../using/examples/hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](../using/examples/hardware_providers.html#quera-computing){.reference
            .internal}
    -   [Dynamics
        Examples](../using/examples/dynamics_examples.html){.reference
        .internal}
        -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
            Model)](../examples/python/dynamics/dynamics_intro_1.html){.reference
            .internal}
            -   [Why dynamics simulations vs. circuit
                simulations?](../examples/python/dynamics/dynamics_intro_1.html#Why-dynamics-simulations-vs.-circuit-simulations?){.reference
                .internal}
            -   [Functionality](../examples/python/dynamics/dynamics_intro_1.html#Functionality){.reference
                .internal}
            -   [Performance](../examples/python/dynamics/dynamics_intro_1.html#Performance){.reference
                .internal}
            -   [Section 1 - Simulating the Jaynes-Cummings
                Hamiltonian](../examples/python/dynamics/dynamics_intro_1.html#Section-1---Simulating-the-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Exercise 1 - Simulating a many-photon Jaynes-Cummings
                Hamiltonian](../examples/python/dynamics/dynamics_intro_1.html#Exercise-1---Simulating-a-many-photon-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Section 2 - Simulating open quantum systems with the
                [`collapse_operators`{.docutils .literal
                .notranslate}]{.pre}](../examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
                .internal}
            -   [Exercise 2 - Adding additional jump operators
                [\\(L_i\\)]{.math .notranslate
                .nohighlight}](../examples/python/dynamics/dynamics_intro_1.html#Exercise-2---Adding-additional-jump-operators-L_i){.reference
                .internal}
            -   [Section 3 - Many qubits coupled to the
                resonator](../examples/python/dynamics/dynamics_intro_1.html#Section-3---Many-qubits-coupled-to-the-resonator){.reference
                .internal}
        -   [Introduction to CUDA-Q Dynamics (Time Dependent
            Hamiltonians)](../examples/python/dynamics/dynamics_intro_2.html){.reference
            .internal}
            -   [The Landau-Zener
                model](../examples/python/dynamics/dynamics_intro_2.html#The-Landau-Zener-model){.reference
                .internal}
            -   [Section 1 - Implementing time dependent
                terms](../examples/python/dynamics/dynamics_intro_2.html#Section-1---Implementing-time-dependent-terms){.reference
                .internal}
            -   [Section 2 - Implementing custom
                operators](../examples/python/dynamics/dynamics_intro_2.html#Section-2---Implementing-custom-operators){.reference
                .internal}
            -   [Section 3 - Heisenberg Model with a time-varying
                magnetic
                field](../examples/python/dynamics/dynamics_intro_2.html#Section-3---Heisenberg-Model-with-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 1 - Define a time-varying magnetic
                field](../examples/python/dynamics/dynamics_intro_2.html#Exercise-1---Define-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 2
                (Optional)](../examples/python/dynamics/dynamics_intro_2.html#Exercise-2-(Optional)){.reference
                .internal}
        -   [Superconducting
            Qubits](../examples/python/dynamics/superconducting.html){.reference
            .internal}
            -   [Cavity
                QED](../examples/python/dynamics/superconducting.html#Cavity-QED){.reference
                .internal}
            -   [Cross
                Resonance](../examples/python/dynamics/superconducting.html#Cross-Resonance){.reference
                .internal}
            -   [Transmon
                Resonator](../examples/python/dynamics/superconducting.html#Transmon-Resonator){.reference
                .internal}
        -   [Spin
            Qubits](../examples/python/dynamics/spinqubits.html){.reference
            .internal}
            -   [Silicon Spin
                Qubit](../examples/python/dynamics/spinqubits.html#Silicon-Spin-Qubit){.reference
                .internal}
            -   [Heisenberg
                Model](../examples/python/dynamics/spinqubits.html#Heisenberg-Model){.reference
                .internal}
        -   [Trapped Ion
            Qubits](../examples/python/dynamics/iontrap.html){.reference
            .internal}
            -   [GHZ
                state](../examples/python/dynamics/iontrap.html#GHZ-state){.reference
                .internal}
        -   [Control](../examples/python/dynamics/control.html){.reference
            .internal}
            -   [Gate
                Calibration](../examples/python/dynamics/control.html#Gate-Calibration){.reference
                .internal}
            -   [Pulse](../examples/python/dynamics/control.html#Pulse){.reference
                .internal}
            -   [Qubit
                Control](../examples/python/dynamics/control.html#Qubit-Control){.reference
                .internal}
            -   [Qubit
                Dynamics](../examples/python/dynamics/control.html#Qubit-Dynamics){.reference
                .internal}
            -   [Landau-Zenner](../examples/python/dynamics/control.html#Landau-Zenner){.reference
                .internal}
-   [Applications](../using/applications.html){.reference .internal}
    -   [Max-Cut with QAOA](../applications/python/qaoa.html){.reference
        .internal}
    -   [Molecular docking via
        DC-QAOA](../applications/python/digitized_counterdiabatic_qaoa.html){.reference
        .internal}
        -   [Setting up the Molecular Docking
            Problem](../applications/python/digitized_counterdiabatic_qaoa.html#Setting-up-the-Molecular-Docking-Problem){.reference
            .internal}
        -   [CUDA-Q
            Implementation](../applications/python/digitized_counterdiabatic_qaoa.html#CUDA-Q-Implementation){.reference
            .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](../applications/python/krylov.html){.reference
        .internal}
        -   [Setup](../applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](../applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](../applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Quantum-Selected Configuration Interaction
        (QSCI)](../applications/python/qsci.html){.reference .internal}
        -   [0. Problem
            definition](../applications/python/qsci.html#0.-Problem-definition){.reference
            .internal}
        -   [1. Prepare an Approximate Quantum
            State](../applications/python/qsci.html#1.-Prepare-an-Approximate-Quantum-State){.reference
            .internal}
        -   [2 Quantum Sampling to Select
            Configuration](../applications/python/qsci.html#2-Quantum-Sampling-to-Select-Configuration){.reference
            .internal}
        -   [3. Classical Diagonalization on the Selected
            Subspace](../applications/python/qsci.html#3.-Classical-Diagonalization-on-the-Selected-Subspace){.reference
            .internal}
        -   [5. Compuare
            results](../applications/python/qsci.html#5.-Compuare-results){.reference
            .internal}
        -   [Reference](../applications/python/qsci.html#Reference){.reference
            .internal}
    -   [Bernstein-Vazirani
        Algorithm](../applications/python/bernstein_vazirani.html){.reference
        .internal}
        -   [Classical
            case](../applications/python/bernstein_vazirani.html#Classical-case){.reference
            .internal}
        -   [Quantum
            case](../applications/python/bernstein_vazirani.html#Quantum-case){.reference
            .internal}
        -   [Implementing in
            CUDA-Q](../applications/python/bernstein_vazirani.html#Implementing-in-CUDA-Q){.reference
            .internal}
    -   [Cost
        Minimization](../applications/python/cost_minimization.html){.reference
        .internal}
    -   [Deutsch's
        Algorithm](../applications/python/deutsch_algorithm.html){.reference
        .internal}
        -   [XOR [\\(\\oplus\\)]{.math .notranslate
            .nohighlight}](../applications/python/deutsch_algorithm.html#XOR-\oplus){.reference
            .internal}
        -   [Quantum
            oracles](../applications/python/deutsch_algorithm.html#Quantum-oracles){.reference
            .internal}
        -   [Phase
            oracle](../applications/python/deutsch_algorithm.html#Phase-oracle){.reference
            .internal}
        -   [Quantum
            parallelism](../applications/python/deutsch_algorithm.html#Quantum-parallelism){.reference
            .internal}
        -   [Deutsch's
            Algorithm:](../applications/python/deutsch_algorithm.html#Deutsch's-Algorithm:){.reference
            .internal}
    -   [Divisive Clustering With Coresets Using
        CUDA-Q](../applications/python/divisive_clustering_coresets.html){.reference
        .internal}
        -   [Data
            preprocessing](../applications/python/divisive_clustering_coresets.html#Data-preprocessing){.reference
            .internal}
        -   [Quantum
            functions](../applications/python/divisive_clustering_coresets.html#Quantum-functions){.reference
            .internal}
        -   [Divisive Clustering
            Function](../applications/python/divisive_clustering_coresets.html#Divisive-Clustering-Function){.reference
            .internal}
        -   [QAOA
            Implementation](../applications/python/divisive_clustering_coresets.html#QAOA-Implementation){.reference
            .internal}
        -   [Scaling simulations with
            CUDA-Q](../applications/python/divisive_clustering_coresets.html#Scaling-simulations-with-CUDA-Q){.reference
            .internal}
    -   [Hybrid Quantum Neural
        Networks](../applications/python/hybrid_quantum_neural_networks.html){.reference
        .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](../applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](../applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using [`Sample`{.docutils .literal .notranslate}]{.pre} to
            perform the Hadamard
            test](../applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](../applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](../applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Anderson Impurity Model ground state solver on Infleqtion's
        Sqale](../applications/python/logical_aim_sqale.html){.reference
        .internal}
        -   [Performing logical Variational Quantum Eigensolver (VQE)
            with
            CUDA-QX](../applications/python/logical_aim_sqale.html#Performing-logical-Variational-Quantum-Eigensolver-(VQE)-with-CUDA-QX){.reference
            .internal}
        -   [Constructing circuits in the [`[[4,2,2]]`{.docutils
            .literal .notranslate}]{.pre}
            encoding](../applications/python/logical_aim_sqale.html#Constructing-circuits-in-the-%5B%5B4,2,2%5D%5D-encoding){.reference
            .internal}
        -   [Setting up submission and decoding
            workflow](../applications/python/logical_aim_sqale.html#Setting-up-submission-and-decoding-workflow){.reference
            .internal}
        -   [Running a CUDA-Q noisy
            simulation](../applications/python/logical_aim_sqale.html#Running-a-CUDA-Q-noisy-simulation){.reference
            .internal}
        -   [Running logical AIM on Infleqtion's
            hardware](../applications/python/logical_aim_sqale.html#Running-logical-AIM-on-Infleqtion's-hardware){.reference
            .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](../applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](../applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](../applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](../applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](../applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](../applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](../applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](../applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. [`Compute`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`overlap`{.docutils .literal
                .notranslate}]{.pre}](../applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](../applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](../applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](../applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](../applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](../applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](../applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](../applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](../applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](../applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum Fourier
        Transform](../applications/python/quantum_fourier_transform.html){.reference
        .internal}
        -   [Quantum Fourier Transform
            revisited](../applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform-revisited){.reference
            .internal}
    -   [Quantum
        Teleporation](../applications/python/quantum_teleportation.html){.reference
        .internal}
        -   [Teleportation
            explained](../applications/python/quantum_teleportation.html#Teleportation-explained){.reference
            .internal}
    -   [Quantum
        Volume](../applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](../applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](../applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Compiling Unitaries Using Diffusion
        Models](../applications/python/unitary_compilation_diffusion_models.html){.reference
        .internal}
        -   [Diffusion model
            pipeline](../applications/python/unitary_compilation_diffusion_models.html#Diffusion-model-pipeline){.reference
            .internal}
        -   [Setup and load
            models](../applications/python/unitary_compilation_diffusion_models.html#Setup-and-load-models){.reference
            .internal}
            -   [Load discrete
                model](../applications/python/unitary_compilation_diffusion_models.html#Load-discrete-model){.reference
                .internal}
            -   [Load continuous
                model](../applications/python/unitary_compilation_diffusion_models.html#Load-continuous-model){.reference
                .internal}
            -   [Create helper
                functions](../applications/python/unitary_compilation_diffusion_models.html#Create-helper-functions){.reference
                .internal}
        -   [Unitary
            compilation](../applications/python/unitary_compilation_diffusion_models.html#Unitary-compilation){.reference
            .internal}
            -   [Random
                unitary](../applications/python/unitary_compilation_diffusion_models.html#Random-unitary){.reference
                .internal}
            -   [Discrete
                model](../applications/python/unitary_compilation_diffusion_models.html#Discrete-model){.reference
                .internal}
            -   [Continuous
                model](../applications/python/unitary_compilation_diffusion_models.html#Continuous-model){.reference
                .internal}
            -   [Quantum Fourier
                transform](../applications/python/unitary_compilation_diffusion_models.html#Quantum-Fourier-transform){.reference
                .internal}
            -   [XXZ-Hamiltonian
                evolution](../applications/python/unitary_compilation_diffusion_models.html#XXZ-Hamiltonian-evolution){.reference
                .internal}
        -   [Choosing the circuit you
            need](../applications/python/unitary_compilation_diffusion_models.html#Choosing-the-circuit-you-need){.reference
            .internal}
    -   [VQE with gradients, active spaces, and gate
        fusion](../applications/python/vqe_advanced.html){.reference
        .internal}
        -   [The Basics of
            VQE](../applications/python/vqe_advanced.html#The-Basics-of-VQE){.reference
            .internal}
        -   [Installing/Loading Relevant
            Packages](../applications/python/vqe_advanced.html#Installing/Loading-Relevant-Packages){.reference
            .internal}
        -   [Implementing VQE in
            CUDA-Q](../applications/python/vqe_advanced.html#Implementing-VQE-in-CUDA-Q){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../applications/python/vqe_advanced.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
        -   [Using an Active
            Space](../applications/python/vqe_advanced.html#Using-an-Active-Space){.reference
            .internal}
        -   [Gate Fusion for Larger
            Circuits](../applications/python/vqe_advanced.html#Gate-Fusion-for-Larger-Circuits){.reference
            .internal}
    -   [Quantum
        Transformer](../applications/python/quantum_transformer.html){.reference
        .internal}
        -   [Installation](../applications/python/quantum_transformer.html#Installation){.reference
            .internal}
        -   [Algorithm and
            Example](../applications/python/quantum_transformer.html#Algorithm-and-Example){.reference
            .internal}
            -   [Creating the self-attention
                circuits](../applications/python/quantum_transformer.html#Creating-the-self-attention-circuits){.reference
                .internal}
        -   [Usage](../applications/python/quantum_transformer.html#Usage){.reference
            .internal}
            -   [Model
                Training](../applications/python/quantum_transformer.html#Model-Training){.reference
                .internal}
            -   [Generating
                Molecules](../applications/python/quantum_transformer.html#Generating-Molecules){.reference
                .internal}
            -   [Attention
                Maps](../applications/python/quantum_transformer.html#Attention-Maps){.reference
                .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](../applications/python/afqmc.html){.reference .internal}
        -   [Hamiltonian preparation for
            VQE](../applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](../applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](../applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](../applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](../applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](../applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [ADAPT-QAOA
        algorithm](../applications/python/adapt_qaoa.html){.reference
        .internal}
        -   [Simulation
            input:](../applications/python/adapt_qaoa.html#Simulation-input:){.reference
            .internal}
        -   [The problem Hamiltonian [\\(H_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](../applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A_j\\)]{.math .notranslate
            .nohighlight}:](../applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H_C,A_j\]\\)]{.math .notranslate
            .nohighlight}:](../applications/python/adapt_qaoa.html#The-commutator-%5BH_C,A_j%5D:){.reference
            .internal}
        -   [Beginning of ADAPT-QAOA
            iteration:](../applications/python/adapt_qaoa.html#Beginning-of-ADAPT-QAOA-iteration:){.reference
            .internal}
    -   [ADAPT-VQE
        algorithm](../applications/python/adapt_vqe.html){.reference
        .internal}
        -   [Classical
            pre-processing](../applications/python/adapt_vqe.html#Classical-pre-processing){.reference
            .internal}
        -   [Jordan
            Wigner:](../applications/python/adapt_vqe.html#Jordan-Wigner:){.reference
            .internal}
        -   [UCCSD operator
            pool](../applications/python/adapt_vqe.html#UCCSD-operator-pool){.reference
            .internal}
            -   [Single
                excitation](../applications/python/adapt_vqe.html#Single-excitation){.reference
                .internal}
            -   [Double
                excitation](../applications/python/adapt_vqe.html#Double-excitation){.reference
                .internal}
        -   [Commutator \[[\\(H\\)]{.math .notranslate .nohighlight},
            [\\(A_i\\)]{.math .notranslate
            .nohighlight}\]](../applications/python/adapt_vqe.html#Commutator-%5BH,-A_i%5D){.reference
            .internal}
        -   [Reference
            State:](../applications/python/adapt_vqe.html#Reference-State:){.reference
            .internal}
        -   [Quantum
            kernels:](../applications/python/adapt_vqe.html#Quantum-kernels:){.reference
            .internal}
        -   [Beginning of
            ADAPT-VQE:](../applications/python/adapt_vqe.html#Beginning-of-ADAPT-VQE:){.reference
            .internal}
    -   [Quantum edge
        detection](../applications/python/edge_detection.html){.reference
        .internal}
        -   [Image](../applications/python/edge_detection.html#Image){.reference
            .internal}
        -   [Quantum Probability Image Encoding
            (QPIE):](../applications/python/edge_detection.html#Quantum-Probability-Image-Encoding-(QPIE):){.reference
            .internal}
            -   [Below we show how to encode an image using QPIE in
                cudaq.](../applications/python/edge_detection.html#Below-we-show-how-to-encode-an-image-using-QPIE-in-cudaq.){.reference
                .internal}
        -   [Flexible Representation of Quantum Images
            (FRQI):](../applications/python/edge_detection.html#Flexible-Representation-of-Quantum-Images-(FRQI):){.reference
            .internal}
            -   [Building the FRQI
                State:](../applications/python/edge_detection.html#Building-the-FRQI-State:){.reference
                .internal}
        -   [Quantum Hadamard Edge Detection
            (QHED)](../applications/python/edge_detection.html#Quantum-Hadamard-Edge-Detection-(QHED)){.reference
            .internal}
            -   [Post-processing](../applications/python/edge_detection.html#Post-processing){.reference
                .internal}
    -   [Factoring Integers With Shor's
        Algorithm](../applications/python/shors.html){.reference
        .internal}
        -   [Shor's
            algorithm](../applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](../applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](../applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](../applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](../applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](../applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](../applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](../applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
            -   [(a) Generate the molecular Hamiltonian using Restricted
                Hartree Fock molecular
                orbitals](../applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Restricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(b) Generate the molecular Hamiltonian using
                Unrestricted Hartree Fock molecular
                orbitals](../applications/python/generate_fermionic_ham.html#(b)-Generate-the-molecular-Hamiltonian-using-Unrestricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(a) Generate the active space hamiltonian using RHF
                molecular
                orbitals.](../applications/python/generate_fermionic_ham.html#(a)-Generate-the-active-space-hamiltonian-using-RHF-molecular-orbitals.){.reference
                .internal}
            -   [(b) Generate the active space Hamiltonian using the
                natural orbitals computed from MP2
                simulation](../applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
                .internal}
            -   [(c) Generate the active space Hamiltonian computed from
                the CASSCF molecular
                orbitals](../applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
                .internal}
            -   [(d) Generate the electronic Hamiltonian using
                ROHF](../applications/python/generate_fermionic_ham.html#(d)-Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
            -   [(e) Generate electronic Hamiltonian using
                UHF](../applications/python/generate_fermionic_ham.html#(e)-Generate-electronic-Hamiltonian-using-UHF){.reference
                .internal}
    -   [Grover's
        Algorithm](../applications/python/grovers.html){.reference
        .internal}
        -   [Overview](../applications/python/grovers.html#Overview){.reference
            .internal}
        -   [Problem](../applications/python/grovers.html#Problem){.reference
            .internal}
        -   [Structure of Grover's
            Algorithm](../applications/python/grovers.html#Structure-of-Grover's-Algorithm){.reference
            .internal}
            -   [Step 1:
                Preparation](../applications/python/grovers.html#Step-1:-Preparation){.reference
                .internal}
            -   [Good and Bad
                States](../applications/python/grovers.html#Good-and-Bad-States){.reference
                .internal}
            -   [Step 2: Oracle
                application](../applications/python/grovers.html#Step-2:-Oracle-application){.reference
                .internal}
            -   [Step 3: Amplitude
                amplification](../applications/python/grovers.html#Step-3:-Amplitude-amplification){.reference
                .internal}
            -   [Steps 4 and 5: Iteration and
                measurement](../applications/python/grovers.html#Steps-4-and-5:-Iteration-and-measurement){.reference
                .internal}
    -   [Quantum
        PageRank](../applications/python/quantum_pagerank.html){.reference
        .internal}
        -   [Problem
            Definition](../applications/python/quantum_pagerank.html#Problem-Definition){.reference
            .internal}
        -   [Simulating Quantum PageRank by CUDA-Q
            dynamics](../applications/python/quantum_pagerank.html#Simulating-Quantum-PageRank-by-CUDA-Q-dynamics){.reference
            .internal}
        -   [Breakdown of
            Terms](../applications/python/quantum_pagerank.html#Breakdown-of-Terms){.reference
            .internal}
    -   [The UCCSD Wavefunction
        ansatz](../applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](../applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](../applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](../applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](../applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](../applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](../applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [QM/MM simulation: VQE within a Polarizable Embedded
        Framework.](../applications/python/qm_mm_pe.html){.reference
        .internal}
        -   [Key
            concepts:](../applications/python/qm_mm_pe.html#Key-concepts:){.reference
            .internal}
        -   [PE-VQE-SCF Algorithm
            Steps](../applications/python/qm_mm_pe.html#PE-VQE-SCF-Algorithm-Steps){.reference
            .internal}
            -   [Step 1: Initialize (Classical
                pre-processing)](../applications/python/qm_mm_pe.html#Step-1:-Initialize-(Classical-pre-processing)){.reference
                .internal}
            -   [Step 2: Build the
                Hamiltonian](../applications/python/qm_mm_pe.html#Step-2:-Build-the-Hamiltonian){.reference
                .internal}
            -   [Step 3: Run
                VQE](../applications/python/qm_mm_pe.html#Step-3:-Run-VQE){.reference
                .internal}
            -   [Step 4: Update
                Environment](../applications/python/qm_mm_pe.html#Step-4:-Update-Environment){.reference
                .internal}
            -   [Step 5: Self-Consistency
                Loop](../applications/python/qm_mm_pe.html#Step-5:-Self-Consistency-Loop){.reference
                .internal}
            -   [Requirments:](../applications/python/qm_mm_pe.html#Requirments:){.reference
                .internal}
            -   [Example 1: LiH with 2 water
                molecules.](../applications/python/qm_mm_pe.html#Example-1:-LiH-with-2-water-molecules.){.reference
                .internal}
            -   [VQE, update environment, and scf
                loop.](../applications/python/qm_mm_pe.html#VQE,-update-environment,-and-scf-loop.){.reference
                .internal}
            -   [Example 2: NH3 with 46 water molecule using active
                space.](../applications/python/qm_mm_pe.html#Example-2:-NH3-with-46-water-molecule-using-active-space.){.reference
                .internal}
    -   [Sample-Based Krylov Quantum Diagonalization
        (SKQD)](../applications/python/skqd.html){.reference .internal}
        -   [Why
            SKQD?](../applications/python/skqd.html#Why-SKQD?){.reference
            .internal}
        -   [Understanding Krylov
            Subspaces](../applications/python/skqd.html#Understanding-Krylov-Subspaces){.reference
            .internal}
            -   [What is a Krylov
                Subspace?](../applications/python/skqd.html#What-is-a-Krylov-Subspace?){.reference
                .internal}
            -   [The SKQD
                Algorithm](../applications/python/skqd.html#The-SKQD-Algorithm){.reference
                .internal}
        -   [Problem Setup: 22-Qubit Heisenberg
            Model](../applications/python/skqd.html#Problem-Setup:-22-Qubit-Heisenberg-Model){.reference
            .internal}
        -   [Krylov State Generation via Repeated
            Evolution](../applications/python/skqd.html#Krylov-State-Generation-via-Repeated-Evolution){.reference
            .internal}
        -   [Quantum Measurements and
            Sampling](../applications/python/skqd.html#Quantum-Measurements-and-Sampling){.reference
            .internal}
            -   [The Sampling
                Process](../applications/python/skqd.html#The-Sampling-Process){.reference
                .internal}
        -   [Classical Post-Processing and
            Diagonalization](../applications/python/skqd.html#Classical-Post-Processing-and-Diagonalization){.reference
            .internal}
            -   [The SKQD Algorithm: Matrix Construction
                Details](../applications/python/skqd.html#The-SKQD-Algorithm:-Matrix-Construction-Details){.reference
                .internal}
        -   [Results Analysis and
            Convergence](../applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](../applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
        -   [GPU Acceleration for
            Postprocessing](../applications/python/skqd.html#GPU-Acceleration-for-Postprocessing){.reference
            .internal}
    -   [Entanglement Accelerates Quantum
        Simulation](../applications/python/entanglement_acc_hamiltonian_simulation.html){.reference
        .internal}
        -   [2. Model
            Definition](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.-Model-Definition){.reference
            .internal}
            -   [2.1 Initial product
                state](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.1-Initial-product-state){.reference
                .internal}
            -   [2.2 QIMF
                Hamiltonian](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.2-QIMF-Hamiltonian){.reference
                .internal}
            -   [2.3 First-Order Trotter Formula
                (PF1)](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.3-First-Order-Trotter-Formula-(PF1)){.reference
                .internal}
            -   [2.4 PF1 step for the QIMF
                partition](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.4-PF1-step-for-the-QIMF-partition){.reference
                .internal}
            -   [2.5 Hamiltonian
                helpers](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.5-Hamiltonian-helpers){.reference
                .internal}
        -   [3. Entanglement
            metrics](../applications/python/entanglement_acc_hamiltonian_simulation.html#3.-Entanglement-metrics){.reference
            .internal}
        -   [4. Simulation
            workflow](../applications/python/entanglement_acc_hamiltonian_simulation.html#4.-Simulation-workflow){.reference
            .internal}
            -   [4.1 Single-step Trotter
                error](../applications/python/entanglement_acc_hamiltonian_simulation.html#4.1-Single-step-Trotter-error){.reference
                .internal}
            -   [4.2 Dual trajectory
                update](../applications/python/entanglement_acc_hamiltonian_simulation.html#4.2-Dual-trajectory-update){.reference
                .internal}
        -   [5. Reproducing the paper's Figure
            1a](../applications/python/entanglement_acc_hamiltonian_simulation.html#5.-Reproducing-the-paper’s-Figure-1a){.reference
            .internal}
            -   [5.1 Visualising the joint
                behaviour](../applications/python/entanglement_acc_hamiltonian_simulation.html#5.1-Visualising-the-joint-behaviour){.reference
                .internal}
            -   [5.2 Interpreting the
                result](../applications/python/entanglement_acc_hamiltonian_simulation.html#5.2-Interpreting-the-result){.reference
                .internal}
        -   [6. References and further
            reading](../applications/python/entanglement_acc_hamiltonian_simulation.html#6.-References-and-further-reading){.reference
            .internal}
-   [Backends](../using/backends/backends.html){.reference .internal}
    -   [Circuit
        Simulation](../using/backends/simulators.html){.reference
        .internal}
        -   [State Vector
            Simulators](../using/backends/sims/svsims.html){.reference
            .internal}
            -   [CPU](../using/backends/sims/svsims.html#cpu){.reference
                .internal}
            -   [Single-GPU](../using/backends/sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](../using/backends/sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network
            Simulators](../using/backends/sims/tnsims.html){.reference
            .internal}
            -   [Multi-GPU
                multi-node](../using/backends/sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](../using/backends/sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](../using/backends/sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU
            Simulators](../using/backends/sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](../using/backends/sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](../using/backends/sims/mqpusims.html#multi-qpu-other-backends){.reference
                .internal}
        -   [Noisy
            Simulators](../using/backends/sims/noisy.html){.reference
            .internal}
            -   [Trajectory Noisy
                Simulation](../using/backends/sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](../using/backends/sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](../using/backends/sims/noisy.html#stim){.reference
                .internal}
        -   [Photonics
            Simulators](../using/backends/sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](../using/backends/sims/photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware
        (QPUs)](../using/backends/hardware.html){.reference .internal}
        -   [Ion Trap
            QPUs](../using/backends/hardware/iontrap.html){.reference
            .internal}
            -   [IonQ](../using/backends/hardware/iontrap.html#ionq){.reference
                .internal}
            -   [Quantinuum](../using/backends/hardware/iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting
            QPUs](../using/backends/hardware/superconducting.html){.reference
            .internal}
            -   [Anyon Technologies/Anyon
                Computing](../using/backends/hardware/superconducting.html#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](../using/backends/hardware/superconducting.html#iqm){.reference
                .internal}
            -   [OQC](../using/backends/hardware/superconducting.html#oqc){.reference
                .internal}
            -   [Quantum Circuits,
                Inc.](../using/backends/hardware/superconducting.html#quantum-circuits-inc){.reference
                .internal}
        -   [Neutral Atom
            QPUs](../using/backends/hardware/neutralatom.html){.reference
            .internal}
            -   [Infleqtion](../using/backends/hardware/neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](../using/backends/hardware/neutralatom.html#pasqal){.reference
                .internal}
            -   [QuEra
                Computing](../using/backends/hardware/neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic
            QPUs](../using/backends/hardware/photonic.html){.reference
            .internal}
            -   [ORCA
                Computing](../using/backends/hardware/photonic.html#orca-computing){.reference
                .internal}
        -   [Quantum Control
            Systems](../using/backends/hardware/qcontrol.html){.reference
            .internal}
            -   [Quantum
                Machines](../using/backends/hardware/qcontrol.html#quantum-machines){.reference
                .internal}
    -   [Dynamics
        Simulation](../using/backends/dynamics_backends.html){.reference
        .internal}
    -   [Cloud](../using/backends/cloud.html){.reference .internal}
        -   [Amazon Braket
            (braket)](../using/backends/cloud/braket.html){.reference
            .internal}
            -   [Setting
                Credentials](../using/backends/cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submission from
                C++](../using/backends/cloud/braket.html#submission-from-c){.reference
                .internal}
            -   [Submission from
                Python](../using/backends/cloud/braket.html#submission-from-python){.reference
                .internal}
-   [Dynamics](../using/dynamics.html){.reference .internal}
    -   [Quick Start](../using/dynamics.html#quick-start){.reference
        .internal}
    -   [Operator](../using/dynamics.html#operator){.reference
        .internal}
    -   [Time-Dependent
        Dynamics](../using/dynamics.html#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](../using/dynamics.html#super-operator-representation){.reference
        .internal}
    -   [Numerical
        Integrators](../using/dynamics.html#numerical-integrators){.reference
        .internal}
    -   [Batch
        simulation](../using/dynamics.html#batch-simulation){.reference
        .internal}
    -   [Multi-GPU Multi-Node
        Execution](../using/dynamics.html#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](../using/dynamics.html#examples){.reference
        .internal}
-   [CUDA-QX](../using/cudaqx/cudaqx.html){.reference .internal}
    -   [CUDA-Q
        Solvers](../using/cudaqx/cudaqx.html#cuda-q-solvers){.reference
        .internal}
    -   [CUDA-Q QEC](../using/cudaqx/cudaqx.html#cuda-q-qec){.reference
        .internal}
-   [Installation](../using/install/install.html){.reference .internal}
    -   [Local
        Installation](../using/install/local_installation.html){.reference
        .internal}
        -   [Introduction](../using/install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](../using/install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](../using/install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](../using/install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](../using/install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](../using/install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](../using/install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](../using/install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](../using/install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](../using/install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](../using/install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](../using/install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](../using/install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](../using/install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](../using/install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](../using/install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](../using/install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](../using/install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](../using/install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](../using/install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](../using/install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](../using/install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](../using/install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
        -   [Next
            Steps](../using/install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](../using/install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](../using/install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](../using/install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](../using/install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](../using/install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](../using/install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](../using/install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](../using/install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](../using/install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](../using/install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](../using/install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](../using/integration/integration.html){.reference
    .internal}
    -   [Downstream CMake
        Integration](../using/integration/cmake_app.html){.reference
        .internal}
    -   [Combining CUDA with
        CUDA-Q](../using/integration/cuda_gpu.html){.reference
        .internal}
    -   [Integrating with Third-Party
        Libraries](../using/integration/libraries.html){.reference
        .internal}
        -   [Calling a CUDA-Q library from
            C++](../using/integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](../using/integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](../using/integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](../using/extending/extending.html){.reference .internal}
    -   [Add a new Hardware
        Backend](../using/extending/backend.html){.reference .internal}
        -   [Overview](../using/extending/backend.html#overview){.reference
            .internal}
        -   [Server Helper
            Implementation](../using/extending/backend.html#server-helper-implementation){.reference
            .internal}
            -   [Directory
                Structure](../using/extending/backend.html#directory-structure){.reference
                .internal}
            -   [Server Helper
                Class](../using/extending/backend.html#server-helper-class){.reference
                .internal}
            -   [[`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](../using/extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](../using/extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent [`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](../using/extending/backend.html#update-parent-cmakelists-txt){.reference
                .internal}
        -   [Testing](../using/extending/backend.html#testing){.reference
            .internal}
            -   [Unit
                Tests](../using/extending/backend.html#unit-tests){.reference
                .internal}
            -   [Mock
                Server](../using/extending/backend.html#mock-server){.reference
                .internal}
            -   [Python
                Tests](../using/extending/backend.html#python-tests){.reference
                .internal}
            -   [Integration
                Tests](../using/extending/backend.html#integration-tests){.reference
                .internal}
        -   [Documentation](../using/extending/backend.html#documentation){.reference
            .internal}
        -   [Example
            Usage](../using/extending/backend.html#example-usage){.reference
            .internal}
        -   [Code
            Review](../using/extending/backend.html#code-review){.reference
            .internal}
        -   [Maintaining a
            Backend](../using/extending/backend.html#maintaining-a-backend){.reference
            .internal}
        -   [Conclusion](../using/extending/backend.html#conclusion){.reference
            .internal}
    -   [Create a new NVQIR
        Simulator](../using/extending/nvqir_simulator.html){.reference
        .internal}
        -   [[`CircuitSimulator`{.code .docutils .literal
            .notranslate}]{.pre}](../using/extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](../using/extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
    -   [Working with CUDA-Q
        IR](../using/extending/cudaq_ir.html){.reference .internal}
    -   [Create an MLIR Pass for
        CUDA-Q](../using/extending/mlir_pass.html){.reference .internal}
-   [Specifications](../specification/index.html){.reference .internal}
    -   [Language Specification](../specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](../specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](../specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](../specification/cudaq/types.html){.reference
            .internal}
            -   [3.1. [`cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. [`cudaq::qubit`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](../specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](../specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. [`cudaq::spin_op`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](../specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on [`cudaq::qubit`{.code .docutils
                .literal
                .notranslate}]{.pre}](../specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](../specification/cudaq/kernels.html){.reference
            .internal}
        -   [7. Sub-circuit
            Synthesis](../specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](../specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](../specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](../specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](../specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11.
            Platform](../specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](../specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. [`cudaq::sample`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. [`cudaq::run`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. [`cudaq::observe`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. [`cudaq::optimizer`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. [`cudaq::gradient`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](../specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](../specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](../specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](../specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](../specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](../specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](../specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](../specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](../specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](../specification/quake-dialect.html#motivation){.reference
            .internal}
-   [API Reference](api.html){.reference .internal}
    -   [C++ API](languages/cpp_api.html){.reference .internal}
        -   [Operators](languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Platform](languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](languages/cpp_api.html#namespaces){.reference
            .internal}
    -   [Python API](languages/python_api.html){.reference .internal}
        -   [Program
            Construction](languages/python_api.html#program-construction){.reference
            .internal}
            -   [[`make_kernel()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [[`PyKernel`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [[`Kernel`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [[`PyKernelDecorator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [[`kernel()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [[`sample_async()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [[`run()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [[`run_async()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [[`observe()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [[`observe_async()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [[`get_state()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [[`get_state_async()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [[`vqe()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [[`draw()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [[`translate()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [[`estimate_resources()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
        -   [Backend
            Configuration](languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [[`has_target()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [[`get_target()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [[`get_targets()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [[`set_target()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [[`reset_target()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [[`set_noise()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [[`unset_noise()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [[`register_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [[`unregister_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [[`cudaq.apply_noise()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [[`initialize_cudaq()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [[`num_available_gpus()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [[`set_random_seed()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](languages/python_api.html#dynamics){.reference
            .internal}
            -   [[`evolve()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [[`evolve_async()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [[`Schedule`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [[`BaseIntegrator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [[`InitialState`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [[`InitialStateType`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [[`IntermediateResultSave`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](languages/python_api.html#operators){.reference
            .internal}
            -   [[`OperatorSum`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [[`ProductOperator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [[`ElementaryOperator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [[`ScalarOperator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [[`RydbergHamiltonian`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [[`SuperOperator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [[`operators.define()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [[`operators.instantiate()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](languages/python_api.html#data-types){.reference
            .internal}
            -   [[`SimulationPrecision`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [[`Target`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [[`State`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [[`Tensor`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [[`QuakeValue`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [[`qubit`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [[`qreg`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [[`qvector`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [[`ComplexMatrix`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [[`SampleResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [[`AsyncSampleResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [[`ObserveResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [[`AsyncObserveResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [[`AsyncStateResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [[`OptimizationResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [[`EvolveResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [[`AsyncEvolveResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [[`Resources`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.Resources){.reference
                .internal}
            -   [Optimizers](languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [[`initialize()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [[`rank()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [[`num_ranks()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [[`all_gather()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [[`broadcast()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [[`is_initialized()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [[`finalize()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
    -   [Quantum Operations](default_ops.html){.reference .internal}
        -   [Unitary Operations on
            Qubits](default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [[`x`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#x){.reference
                .internal}
            -   [[`y`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#y){.reference
                .internal}
            -   [[`z`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#z){.reference
                .internal}
            -   [[`h`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#h){.reference
                .internal}
            -   [[`r1`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#r1){.reference
                .internal}
            -   [[`rx`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#rx){.reference
                .internal}
            -   [[`ry`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#ry){.reference
                .internal}
            -   [[`rz`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#rz){.reference
                .internal}
            -   [[`s`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#s){.reference
                .internal}
            -   [[`t`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#t){.reference
                .internal}
            -   [[`swap`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#swap){.reference
                .internal}
            -   [[`u3`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#u3){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#mz){.reference
                .internal}
            -   [[`mx`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#mx){.reference
                .internal}
            -   [[`my`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [[`create`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#create){.reference
                .internal}
            -   [[`annihilate`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#annihilate){.reference
                .internal}
            -   [[`phase_shift`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#phase-shift){.reference
                .internal}
            -   [[`beam_splitter`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#beam-splitter){.reference
                .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](default_ops.html#id1){.reference
                .internal}
    -   [PTSBE API](#){.current .reference .internal}
        -   [Python API --- [`cudaq.ptsbe`{.docutils .literal
            .notranslate}]{.pre}](#python-api-cudaq-ptsbe){.reference
            .internal}
            -   [Sampling Functions](#sampling-functions){.reference
                .internal}
            -   [Result Type](#result-type){.reference .internal}
            -   [Trajectory Sampling
                Strategies](#trajectory-sampling-strategies){.reference
                .internal}
            -   [Shot Allocation
                Strategy](#shot-allocation-strategy){.reference
                .internal}
            -   [Execution Data](#execution-data){.reference .internal}
        -   [C++ API --- [`cudaq::ptsbe`{.docutils .literal
            .notranslate}]{.pre}](#c-api-cudaq-ptsbe){.reference
            .internal}
            -   [Sampling Functions](#id1){.reference .internal}
            -   [Options](#options){.reference .internal}
            -   [Result Type](#id2){.reference .internal}
            -   [Trajectory Sampling Strategies](#id3){.reference
                .internal}
            -   [Shot Allocation Strategy](#id4){.reference .internal}
            -   [Execution Data](#id5){.reference .internal}
            -   [Trajectory and Selection
                Types](#trajectory-and-selection-types){.reference
                .internal}
-   [User Guide](../using/user_guide.html){.reference .internal}
    -   [PTSBE](../using/ptsbe.html){.reference .internal}
        -   [Conceptual
            Overview](../using/ptsbe.html#conceptual-overview){.reference
            .internal}
        -   [When to Use
            PTSBE](../using/ptsbe.html#when-to-use-ptsbe){.reference
            .internal}
        -   [Quick Start](../using/ptsbe.html#quick-start){.reference
            .internal}
        -   [Usage
            Tutorial](../using/ptsbe.html#usage-tutorial){.reference
            .internal}
            -   [Controlling the Number of
                Trajectories](../using/ptsbe.html#controlling-the-number-of-trajectories){.reference
                .internal}
            -   [Choosing a Trajectory Sampling
                Strategy](../using/ptsbe.html#choosing-a-trajectory-sampling-strategy){.reference
                .internal}
            -   [Shot Allocation
                Strategies](../using/ptsbe.html#shot-allocation-strategies){.reference
                .internal}
            -   [Inspecting Execution
                Data](../using/ptsbe.html#inspecting-execution-data){.reference
                .internal}
            -   [Asynchronous
                Execution](../using/ptsbe.html#asynchronous-execution){.reference
                .internal}
        -   [Trajectory vs Shot
            Trade-offs](../using/ptsbe.html#trajectory-vs-shot-trade-offs){.reference
            .internal}
        -   [Backend
            Requirements](../using/ptsbe.html#backend-requirements){.reference
            .internal}
        -   [Related
            Approaches](../using/ptsbe.html#related-approaches){.reference
            .internal}
        -   [References](../using/ptsbe.html#references){.reference
            .internal}
-   [Other Versions](../versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](../index.html)

::: wy-nav-content
::: rst-content
::: {role="navigation" aria-label="Page navigation"}
-   [](../index.html){.icon .icon-home aria-label="Home"}
-   [Code documentation](api.html)
-   PTSBE API Reference
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](default_ops.html "Quantum Operations"){.btn .btn-neutral
.float-left accesskey="p"} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](../using/user_guide.html "User Guide"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#ptsbe-api-reference .section}
# PTSBE API Reference[¶](#ptsbe-api-reference "Permalink to this heading"){.headerlink}

This page documents the public API for Pre-Trajectory Sampling with
Batch Execution (PTSBE). For a conceptual overview and usage tutorial
see [[Noisy Simulation with
PTSBE]{.doc}](../using/ptsbe.html){.reference .internal}.

Contents

-   [Python API --- [`cudaq.ptsbe`{.docutils .literal
    .notranslate}]{.pre}](#python-api-cudaq-ptsbe){#id6 .reference
    .internal}

    -   [Sampling Functions](#sampling-functions){#id7 .reference
        .internal}

    -   [Result Type](#result-type){#id8 .reference .internal}

    -   [Trajectory Sampling
        Strategies](#trajectory-sampling-strategies){#id9 .reference
        .internal}

    -   [Shot Allocation Strategy](#shot-allocation-strategy){#id10
        .reference .internal}

    -   [Execution Data](#execution-data){#id11 .reference .internal}

-   [C++ API --- [`cudaq::ptsbe`{.docutils .literal
    .notranslate}]{.pre}](#c-api-cudaq-ptsbe){#id12 .reference
    .internal}

    -   [Sampling Functions](#id1){#id13 .reference .internal}

    -   [Options](#options){#id14 .reference .internal}

    -   [Result Type](#id2){#id15 .reference .internal}

    -   [Trajectory Sampling Strategies](#id3){#id16 .reference
        .internal}

    -   [Shot Allocation Strategy](#id4){#id17 .reference .internal}

    -   [Execution Data](#id5){#id18 .reference .internal}

    -   [Trajectory and Selection
        Types](#trajectory-and-selection-types){#id19 .reference
        .internal}

------------------------------------------------------------------------

::: {#python-api-cudaq-ptsbe .section}
## [Python API --- [`cudaq.ptsbe`{.docutils .literal .notranslate}]{.pre}](#id6){.toc-backref role="doc-backlink"}[¶](#python-api-cudaq-ptsbe "Permalink to this heading"){.headerlink}

::: {#sampling-functions .section}
### [Sampling Functions](#id7){.toc-backref role="doc-backlink"}[¶](#sampling-functions "Permalink to this heading"){.headerlink}

[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[sample]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[kernel]{.pre}]{.n}*, *[[\*]{.pre}]{.o}[[args]{.pre}]{.n}*, *[[shots_count]{.pre}]{.n}[[=]{.pre}]{.o}[[1000]{.pre}]{.default_value}*, *[[noise_model]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[max_trajectories]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[sampling_strategy]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[shot_allocation]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[return_execution_data]{.pre}]{.n}[[=]{.pre}]{.o}[[False]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#cudaq.ptsbe.sample "Permalink to this definition"){.headerlink}

:   Sample a quantum kernel using Pre-Trajectory Sampling with Batch
    Execution.

    Pre-samples *T* unique noise trajectories from the circuit's noise
    model and batches circuit executions by unique trajectory. Each
    trajectory is simulated as a pure-state circuit; results are merged
    into a single [[`SampleResult`{.xref .py .py-class .docutils
    .literal
    .notranslate}]{.pre}](languages/python_api.html#cudaq.SampleResult "cudaq.SampleResult"){.reference
    .internal}.

    When any argument is a list (broadcast mode), the kernel is executed
    for each element of the list and a list of results is returned.

    Parameters[:]{.colon}

    :   -   **kernel** -- The quantum kernel to execute. Must be a
            static circuit with no mid-circuit measurements or
            measurement-dependent conditional logic.

        -   **args** -- Positional arguments forwarded to the kernel.

        -   **shots_count**
            ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference
            .external}) -- Total number of measurement shots to
            distribute across all trajectories. Default:
            [`1000`{.docutils .literal .notranslate}]{.pre}.

        -   **noise_model** ([[`cudaq.NoiseModel`{.xref .py .py-class
            .docutils .literal
            .notranslate}]{.pre}](languages/python_api.html#cudaq.NoiseModel "cudaq.NoiseModel"){.reference
            .internal} or [`None`{.docutils .literal
            .notranslate}]{.pre}) -- Noise model describing gate-level
            error channels. Noise can also be injected inside the kernel
            via [`cudaq.apply_noise()`{.docutils .literal
            .notranslate}]{.pre}; both can be combined. Default:
            [`None`{.docutils .literal .notranslate}]{.pre} (no noise).

        -   **max_trajectories** (int or [`None`{.docutils .literal
            .notranslate}]{.pre}) -- Maximum number of unique
            trajectories to generate. [`None`{.docutils .literal
            .notranslate}]{.pre} defaults to [`shots_count`{.docutils
            .literal .notranslate}]{.pre}. Setting an explicit limit
            (e.g. 500) enables trajectory reuse and is strongly
            recommended for large shot counts.

        -   **sampling_strategy** ([[`PTSSamplingStrategy`{.xref .py
            .py-class .docutils .literal
            .notranslate}]{.pre}](#cudaq.ptsbe.PTSSamplingStrategy "cudaq.ptsbe.PTSSamplingStrategy"){.reference
            .internal} or [`None`{.docutils .literal
            .notranslate}]{.pre}) -- Strategy used to select
            trajectories from the noise space. [`None`{.docutils
            .literal .notranslate}]{.pre} uses the default
            [[`ProbabilisticSamplingStrategy`{.xref .py .py-class
            .docutils .literal
            .notranslate}]{.pre}](#cudaq.ptsbe.ProbabilisticSamplingStrategy "cudaq.ptsbe.ProbabilisticSamplingStrategy"){.reference
            .internal}.

        -   **shot_allocation** ([[`ShotAllocationStrategy`{.xref .py
            .py-class .docutils .literal
            .notranslate}]{.pre}](#cudaq.ptsbe.ShotAllocationStrategy "cudaq.ptsbe.ShotAllocationStrategy"){.reference
            .internal} or [`None`{.docutils .literal
            .notranslate}]{.pre}) -- Strategy used to distribute shots
            across the selected trajectories. [`None`{.docutils .literal
            .notranslate}]{.pre} uses [[`PROPORTIONAL`{.xref .py
            .py-attr .docutils .literal
            .notranslate}]{.pre}](#cudaq.ptsbe.ShotAllocationStrategy.Type.PROPORTIONAL "cudaq.ptsbe.ShotAllocationStrategy.Type.PROPORTIONAL"){.reference
            .internal}.

        -   **return_execution_data**
            ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"){.reference
            .external}) -- When [`True`{.docutils .literal
            .notranslate}]{.pre}, attaches the full execution trace
            (circuit instructions, trajectory specifications, and
            per-trajectory measurement counts) to the returned result.
            Default: [`False`{.docutils .literal .notranslate}]{.pre}.

    Returns[:]{.colon}

    :   Aggregated measurement outcomes. In broadcast mode, a list of
        results is returned.

    Return type[:]{.colon}

    :   [[`cudaq.ptsbe.PTSBESampleResult`{.xref .py .py-class .docutils
        .literal
        .notranslate}]{.pre}](#cudaq.ptsbe.PTSBESampleResult "cudaq.ptsbe.PTSBESampleResult"){.reference
        .internal}

    Raises[:]{.colon}

    :   [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.14)"){.reference
        .external} -- If the kernel contains mid-circuit measurements,
        conditional feedback, unsupported noise channels, or invalid
        arguments.

    ::: {.highlight-python .notranslate}
    ::: highlight
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
        noise.add_channel("h", [0], cudaq.DepolarizationChannel(0.01))

        result = ptsbe.sample(bell, shots_count=10_000,
                              noise_model=noise, max_trajectories=200)
        print(result)
    :::
    :::

```{=html}
<!-- -->
```

[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[sample_async]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[kernel]{.pre}]{.n}*, *[[\*]{.pre}]{.o}[[args]{.pre}]{.n}*, *[[shots_count]{.pre}]{.n}[[=]{.pre}]{.o}[[1000]{.pre}]{.default_value}*, *[[noise_model]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[max_trajectories]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[sampling_strategy]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[shot_allocation]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*, *[[return_execution_data]{.pre}]{.n}[[=]{.pre}]{.o}[[False]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#cudaq.ptsbe.sample_async "Permalink to this definition"){.headerlink}

:   Asynchronous variant of [[`sample()`{.xref .py .py-func .docutils
    .literal
    .notranslate}]{.pre}](#cudaq.ptsbe.sample "cudaq.ptsbe.sample"){.reference
    .internal}. Submits the job without blocking and returns a future.

    All parameters are identical to [[`sample()`{.xref .py .py-func
    .docutils .literal
    .notranslate}]{.pre}](#cudaq.ptsbe.sample "cudaq.ptsbe.sample"){.reference
    .internal}.

    Returns[:]{.colon}

    :   A future whose [`.get()`{.docutils .literal .notranslate}]{.pre}
        method returns the [[`PTSBESampleResult`{.xref .py .py-class
        .docutils .literal
        .notranslate}]{.pre}](#cudaq.ptsbe.PTSBESampleResult "cudaq.ptsbe.PTSBESampleResult"){.reference
        .internal}.

    Return type[:]{.colon}

    :   [[`AsyncSampleResult`{.xref .py .py-class .docutils .literal
        .notranslate}]{.pre}](languages/python_api.html#cudaq.AsyncSampleResult "cudaq.AsyncSampleResult"){.reference
        .internal}

    ::: {.highlight-python .notranslate}
    ::: highlight
        future = ptsbe.sample_async(bell, shots_count=10_000, noise_model=noise)
        # ... do other work ...
        result = future.get()
    :::
    :::
:::

------------------------------------------------------------------------

::: {#result-type .section}
### [Result Type](#id8){.toc-backref role="doc-backlink"}[¶](#result-type "Permalink to this heading"){.headerlink}

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[PTSBESampleResult]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.PTSBESampleResult "Permalink to this definition"){.headerlink}

:   Extends [[`cudaq.SampleResult`{.xref .py .py-class .docutils
    .literal
    .notranslate}]{.pre}](languages/python_api.html#cudaq.SampleResult "cudaq.SampleResult"){.reference
    .internal} with an optional [[`PTSBEExecutionData`{.xref .py
    .py-class .docutils .literal
    .notranslate}]{.pre}](#cudaq.ptsbe.PTSBEExecutionData "cudaq.ptsbe.PTSBEExecutionData"){.reference
    .internal} payload produced when
    [`return_execution_data=True`{.docutils .literal
    .notranslate}]{.pre}.

    [[has_execution_data]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren} [[→]{.sig-return-icon} [[[bool]{.pre}](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"){.reference .external}]{.sig-return-typehint}]{.sig-return}[¶](#cudaq.ptsbe.PTSBESampleResult.has_execution_data "Permalink to this definition"){.headerlink}

    :   Return [`True`{.docutils .literal .notranslate}]{.pre} if
        execution data is attached to this result.

    [[execution_data]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren} [[→]{.sig-return-icon} [[[PTSBEExecutionData]{.pre}](#cudaq.ptsbe.PTSBEExecutionData "cudaq.ptsbe.PTSBEExecutionData"){.reference .internal}]{.sig-return-typehint}]{.sig-return}[¶](#cudaq.ptsbe.PTSBESampleResult.execution_data "Permalink to this definition"){.headerlink}

    :   Return the attached [[`PTSBEExecutionData`{.xref .py .py-class
        .docutils .literal
        .notranslate}]{.pre}](#cudaq.ptsbe.PTSBEExecutionData "cudaq.ptsbe.PTSBEExecutionData"){.reference
        .internal}.

        Raises[:]{.colon}

        :   [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.14)"){.reference
            .external} -- If no execution data is available. Check
            [[`has_execution_data()`{.xref .py .py-meth .docutils
            .literal
            .notranslate}]{.pre}](#cudaq.ptsbe.PTSBESampleResult.has_execution_data "cudaq.ptsbe.PTSBESampleResult.has_execution_data"){.reference
            .internal} first.
:::

------------------------------------------------------------------------

::: {#trajectory-sampling-strategies .section}
### [Trajectory Sampling Strategies](#id9){.toc-backref role="doc-backlink"}[¶](#trajectory-sampling-strategies "Permalink to this heading"){.headerlink}

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[PTSSamplingStrategy]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.PTSSamplingStrategy "Permalink to this definition"){.headerlink}

:   Abstract base class for trajectory sampling strategies. Subclass and
    implement [[`generate_trajectories()`{.xref .py .py-meth .docutils
    .literal
    .notranslate}]{.pre}](#cudaq.ptsbe.PTSSamplingStrategy.generate_trajectories "cudaq.ptsbe.PTSSamplingStrategy.generate_trajectories"){.reference
    .internal} to define a custom strategy.

    [[generate_trajectories]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[noise_points]{.pre}]{.n}*, *[[max_trajectories]{.pre}]{.n}*[)]{.sig-paren} [[→]{.sig-return-icon} [[[list]{.pre}](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)"){.reference .external}[[\[]{.pre}]{.p}[[KrausTrajectory]{.pre}](#cudaq.ptsbe.KrausTrajectory "cudaq.ptsbe.KrausTrajectory"){.reference .internal}[[\]]{.pre}]{.p}]{.sig-return-typehint}]{.sig-return}[¶](#cudaq.ptsbe.PTSSamplingStrategy.generate_trajectories "Permalink to this definition"){.headerlink}

    :   Generate up to *max_trajectories* unique trajectories from the
        given noise points.

        Parameters[:]{.colon}

        :   -   **noise_points** -- Noise site information extracted
                from the circuit.

            -   **max_trajectories**
                ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference
                .external}) -- Upper bound on the number of
                trajectories.

        Returns[:]{.colon}

        :   List of unique [[`KrausTrajectory`{.xref .py .py-class
            .docutils .literal
            .notranslate}]{.pre}](#cudaq.ptsbe.KrausTrajectory "cudaq.ptsbe.KrausTrajectory"){.reference
            .internal} objects.

    [[name]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren} [[→]{.sig-return-icon} [[[str]{.pre}](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"){.reference .external}]{.sig-return-typehint}]{.sig-return}[¶](#cudaq.ptsbe.PTSSamplingStrategy.name "Permalink to this definition"){.headerlink}

    :   Return a human-readable name for this strategy.

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[ProbabilisticSamplingStrategy]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[seed]{.pre}]{.n}[[=]{.pre}]{.o}[[0]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#cudaq.ptsbe.ProbabilisticSamplingStrategy "Permalink to this definition"){.headerlink}

:   Randomly samples unique trajectories weighted by their occurrence
    probability. Produces a representative cross-section of the noise
    space. Duplicate trajectories are discarded.

    Parameters[:]{.colon}

    :   **seed**
        ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference
        .external}) -- Random seed for reproducibility. [`0`{.docutils
        .literal .notranslate}]{.pre} uses the global CUDA-Q seed if
        set, otherwise a random device seed.

    ::: {.highlight-python .notranslate}
    ::: highlight
        strategy = ptsbe.ProbabilisticSamplingStrategy(seed=42)
        result = ptsbe.sample(bell, shots_count=10_000,
                              noise_model=noise,
                              sampling_strategy=strategy)
    :::
    :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[OrderedSamplingStrategy]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.OrderedSamplingStrategy "Permalink to this definition"){.headerlink}

:   Selects the top-*T* trajectories sorted by probability in descending
    order. Ensures the highest-probability noise realizations are always
    represented. Best when the noise space is dominated by a small
    number of likely error patterns.

    ::: {.highlight-python .notranslate}
    ::: highlight
        result = ptsbe.sample(bell, shots_count=10_000,
                              noise_model=noise,
                              max_trajectories=100,
                              sampling_strategy=ptsbe.OrderedSamplingStrategy())
    :::
    :::

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[ExhaustiveSamplingStrategy]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.ExhaustiveSamplingStrategy "Permalink to this definition"){.headerlink}

:   Enumerates every possible trajectory in lexicographic order.
    Produces a complete representation of the noise space. Only
    practical when the noise space is small (few noise sites and low
    Kraus operator count).

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[ConditionalSamplingStrategy]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[predicate]{.pre}]{.n}*, *[[seed]{.pre}]{.n}[[=]{.pre}]{.o}[[0]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#cudaq.ptsbe.ConditionalSamplingStrategy "Permalink to this definition"){.headerlink}

:   Samples trajectories that satisfy a user-supplied predicate
    function. Useful for targeted studies such as restricting to
    single-qubit error events or trajectories below a probability
    threshold.

    Parameters[:]{.colon}

    :   -   **predicate** -- A callable [`(KrausTrajectory)`{.docutils
            .literal .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`->`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`bool`{.docutils .literal .notranslate}]{.pre}
            that returns [`True`{.docutils .literal .notranslate}]{.pre}
            for trajectories to include.

        -   **seed**
            ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference
            .external}) -- Random seed. [`0`{.docutils .literal
            .notranslate}]{.pre} uses the global CUDA-Q seed.

    ::: {.highlight-python .notranslate}
    ::: highlight
        # Keep only trajectories with at most one error
        strategy = ptsbe.ConditionalSamplingStrategy(
            predicate=lambda traj: traj.count_errors() <= 1,
            seed=42,
        )
        result = ptsbe.sample(bell, shots_count=10_000,
                              noise_model=noise,
                              sampling_strategy=strategy)
    :::
    :::
:::

------------------------------------------------------------------------

::: {#shot-allocation-strategy .section}
### [Shot Allocation Strategy](#id10){.toc-backref role="doc-backlink"}[¶](#shot-allocation-strategy "Permalink to this heading"){.headerlink}

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[ShotAllocationStrategy]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[type]{.pre}]{.n}[[=]{.pre}]{.o}[[ShotAllocationStrategy.Type.PROPORTIONAL]{.pre}]{.default_value}*, *[[bias_strength]{.pre}]{.n}[[=]{.pre}]{.o}[[2.0]{.pre}]{.default_value}*, *[[seed]{.pre}]{.n}[[=]{.pre}]{.o}[[0]{.pre}]{.default_value}*[)]{.sig-paren}[¶](#cudaq.ptsbe.ShotAllocationStrategy "Permalink to this definition"){.headerlink}

:   Controls how the total shot count is distributed across the selected
    trajectories after trajectory sampling.

    Parameters[:]{.colon}

    :   -   **type** ([[`Type`{.xref .py .py-class .docutils .literal
            .notranslate}]{.pre}](#cudaq.ptsbe.ShotAllocationStrategy.Type "cudaq.ptsbe.ShotAllocationStrategy.Type"){.reference
            .internal}) -- Allocation strategy type.

        -   **bias_strength**
            ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"){.reference
            .external}) -- Exponent used by the biased strategies.
            Higher values produce stronger bias. Default:
            [`2.0`{.docutils .literal .notranslate}]{.pre}.

        -   **seed**
            ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference
            .external}) -- Random seed used by probabilistic allocation
            (PROPORTIONAL and biased strategies). [`0`{.docutils
            .literal .notranslate}]{.pre} uses the global CUDA-Q seed.

    *[class]{.pre}[ ]{.w}*[[Type]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.ShotAllocationStrategy.Type "Permalink to this definition"){.headerlink}

    :   

        [[PROPORTIONAL]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.ShotAllocationStrategy.Type.PROPORTIONAL "Permalink to this definition"){.headerlink}

        :   *(default)* Shots are allocated via multinomial sampling
            weighted by trajectory probability. The total is always
            exactly [`shots_count`{.docutils .literal
            .notranslate}]{.pre} and every trajectory with non-zero
            probability receives a fair share.

        [[UNIFORM]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.ShotAllocationStrategy.Type.UNIFORM "Permalink to this definition"){.headerlink}

        :   Equal shots per trajectory regardless of probability.

        [[LOW_WEIGHT_BIAS]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.ShotAllocationStrategy.Type.LOW_WEIGHT_BIAS "Permalink to this definition"){.headerlink}

        :   Biases more shots toward trajectories with fewer errors
            (lower Kraus weight). Weight formula: [`(1`{.docutils
            .literal .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`+`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`error_count)^(-bias_strength)`{.docutils
            .literal .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`*`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`probability`{.docutils .literal
            .notranslate}]{.pre}.

        [[HIGH_WEIGHT_BIAS]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.ShotAllocationStrategy.Type.HIGH_WEIGHT_BIAS "Permalink to this definition"){.headerlink}

        :   Biases more shots toward trajectories with more errors.
            Weight formula: [`(1`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`+`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`error_count)^(+bias_strength)`{.docutils
            .literal .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`*`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`probability`{.docutils .literal
            .notranslate}]{.pre}.

    ::: {.highlight-python .notranslate}
    ::: highlight
        alloc = ptsbe.ShotAllocationStrategy(
            ptsbe.ShotAllocationStrategy.Type.LOW_WEIGHT_BIAS,
            bias_strength=3.0,
        )
        result = ptsbe.sample(bell, shots_count=10_000,
                              noise_model=noise,
                              shot_allocation=alloc)
    :::
    :::
:::

------------------------------------------------------------------------

::: {#execution-data .section}
### [Execution Data](#id11){.toc-backref role="doc-backlink"}[¶](#execution-data "Permalink to this heading"){.headerlink}

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[PTSBEExecutionData]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.PTSBEExecutionData "Permalink to this definition"){.headerlink}

:   Container for the full PTSBE execution trace. Returned by
    [[`execution_data()`{.xref .py .py-meth .docutils .literal
    .notranslate}]{.pre}](#cudaq.ptsbe.PTSBESampleResult.execution_data "cudaq.ptsbe.PTSBESampleResult.execution_data"){.reference
    .internal} when [`return_execution_data=True`{.docutils .literal
    .notranslate}]{.pre}.

    [[instructions]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[list]{.pre}](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)"){.reference .external}[[\[]{.pre}]{.p}[[TraceInstruction]{.pre}](#cudaq.ptsbe.TraceInstruction "cudaq.ptsbe.TraceInstruction"){.reference .internal}[[\]]{.pre}]{.p}*[¶](#cudaq.ptsbe.PTSBEExecutionData.instructions "Permalink to this definition"){.headerlink}

    :   Ordered list of circuit operations: gates, noise channel
        locations, and terminal measurements.

    [[trajectories]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[list]{.pre}](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)"){.reference .external}[[\[]{.pre}]{.p}[[KrausTrajectory]{.pre}](#cudaq.ptsbe.KrausTrajectory "cudaq.ptsbe.KrausTrajectory"){.reference .internal}[[\]]{.pre}]{.p}*[¶](#cudaq.ptsbe.PTSBEExecutionData.trajectories "Permalink to this definition"){.headerlink}

    :   The trajectories that were sampled and executed.

    [[count_instructions]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[type]{.pre}]{.n}*, *[[name]{.pre}]{.n}[[=]{.pre}]{.o}[[None]{.pre}]{.default_value}*[)]{.sig-paren} [[→]{.sig-return-icon} [[[int]{.pre}](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference .external}]{.sig-return-typehint}]{.sig-return}[¶](#cudaq.ptsbe.PTSBEExecutionData.count_instructions "Permalink to this definition"){.headerlink}

    :   Count instructions of the given [[`TraceInstructionType`{.xref
        .py .py-class .docutils .literal
        .notranslate}]{.pre}](#cudaq.ptsbe.TraceInstructionType "cudaq.ptsbe.TraceInstructionType"){.reference
        .internal}, optionally filtered by operation name.

    [[get_trajectory]{.pre}]{.sig-name .descname}[(]{.sig-paren}*[[trajectory_id]{.pre}]{.n}*[)]{.sig-paren}[¶](#cudaq.ptsbe.PTSBEExecutionData.get_trajectory "Permalink to this definition"){.headerlink}

    :   Look up a trajectory by its ID. Returns [`None`{.docutils
        .literal .notranslate}]{.pre} if not found.

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[TraceInstruction]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.TraceInstruction "Permalink to this definition"){.headerlink}

:   A single operation in the PTSBE execution trace.

    [[type]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[TraceInstructionType]{.pre}](#cudaq.ptsbe.TraceInstructionType "cudaq.ptsbe.TraceInstructionType"){.reference .internal}*[¶](#cudaq.ptsbe.TraceInstruction.type "Permalink to this definition"){.headerlink}

    :   Whether this instruction is a [`Gate`{.docutils .literal
        .notranslate}]{.pre}, [`Noise`{.docutils .literal
        .notranslate}]{.pre}, or [`Measurement`{.docutils .literal
        .notranslate}]{.pre} (see [[`TraceInstructionType`{.xref .py
        .py-class .docutils .literal
        .notranslate}]{.pre}](#cudaq.ptsbe.TraceInstructionType "cudaq.ptsbe.TraceInstructionType"){.reference
        .internal}).

    [[name]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[str]{.pre}](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"){.reference .external}*[¶](#cudaq.ptsbe.TraceInstruction.name "Permalink to this definition"){.headerlink}

    :   Operation name (e.g. [`"h"`{.docutils .literal
        .notranslate}]{.pre}, [`"cx"`{.docutils .literal
        .notranslate}]{.pre}, [`"depolarizing"`{.docutils .literal
        .notranslate}]{.pre}, [`"mz"`{.docutils .literal
        .notranslate}]{.pre}).

    [[targets]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[list]{.pre}](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)"){.reference .external}[[\[]{.pre}]{.p}[[int]{.pre}](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference .external}[[\]]{.pre}]{.p}*[¶](#cudaq.ptsbe.TraceInstruction.targets "Permalink to this definition"){.headerlink}

    :   Target qubit indices.

    [[controls]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[list]{.pre}](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)"){.reference .external}[[\[]{.pre}]{.p}[[int]{.pre}](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference .external}[[\]]{.pre}]{.p}*[¶](#cudaq.ptsbe.TraceInstruction.controls "Permalink to this definition"){.headerlink}

    :   Control qubit indices. Empty for non-controlled operations.

    [[params]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[list]{.pre}](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)"){.reference .external}[[\[]{.pre}]{.p}[[float]{.pre}](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"){.reference .external}[[\]]{.pre}]{.p}*[¶](#cudaq.ptsbe.TraceInstruction.params "Permalink to this definition"){.headerlink}

    :   Gate rotation angles or noise channel parameters.

    [[channel]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.TraceInstruction.channel "Permalink to this definition"){.headerlink}

    :   The noise channel ([`cudaq.KrausChannel`{.docutils .literal
        .notranslate}]{.pre}), or [`None`{.docutils .literal
        .notranslate}]{.pre}. Populated only for [`Noise`{.docutils
        .literal .notranslate}]{.pre} instructions.

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[TraceInstructionType]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.TraceInstructionType "Permalink to this definition"){.headerlink}

:   Discriminator enum for [[`TraceInstruction`{.xref .py .py-class
    .docutils .literal
    .notranslate}]{.pre}](#cudaq.ptsbe.TraceInstruction "cudaq.ptsbe.TraceInstruction"){.reference
    .internal} entries.

    [[Gate]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.TraceInstructionType.Gate "Permalink to this definition"){.headerlink}

    :   A unitary quantum gate (H, X, CNOT, RX, ...).

    [[Noise]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.TraceInstructionType.Noise "Permalink to this definition"){.headerlink}

    :   A noise channel injection point.

    [[Measurement]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.TraceInstructionType.Measurement "Permalink to this definition"){.headerlink}

    :   A terminal measurement operation.

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[KrausTrajectory]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.KrausTrajectory "Permalink to this definition"){.headerlink}

:   One complete assignment of Kraus operators across all noise sites in
    the circuit.

    [[trajectory_id]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[int]{.pre}](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference .external}*[¶](#cudaq.ptsbe.KrausTrajectory.trajectory_id "Permalink to this definition"){.headerlink}

    :   Unique identifier assigned during trajectory sampling.

    [[probability]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[float]{.pre}](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"){.reference .external}*[¶](#cudaq.ptsbe.KrausTrajectory.probability "Permalink to this definition"){.headerlink}

    :   Product of the probabilities of the selected Kraus operators at
        each noise site.

    [[num_shots]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[int]{.pre}](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference .external}*[¶](#cudaq.ptsbe.KrausTrajectory.num_shots "Permalink to this definition"){.headerlink}

    :   Number of measurement shots allocated to this trajectory.

    [[multiplicity]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[int]{.pre}](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference .external}*[¶](#cudaq.ptsbe.KrausTrajectory.multiplicity "Permalink to this definition"){.headerlink}

    :   Number of times this trajectory was drawn before deduplication.

    [[kraus_selections]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[list]{.pre}](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)"){.reference .external}[[\[]{.pre}]{.p}[[KrausSelection]{.pre}](#cudaq.ptsbe.KrausSelection "cudaq.ptsbe.KrausSelection"){.reference .internal}[[\]]{.pre}]{.p}*[¶](#cudaq.ptsbe.KrausTrajectory.kraus_selections "Permalink to this definition"){.headerlink}

    :   Ordered list of Kraus operator choices, one per noise site.

    [[count_errors]{.pre}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren} [[→]{.sig-return-icon} [[[int]{.pre}](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference .external}]{.sig-return-typehint}]{.sig-return}[¶](#cudaq.ptsbe.KrausTrajectory.count_errors "Permalink to this definition"){.headerlink}

    :   Return the number of non-identity Kraus operators in this
        trajectory (the *error weight*).

```{=html}
<!-- -->
```

*[class]{.pre}[ ]{.w}*[[cudaq.ptsbe.]{.pre}]{.sig-prename .descclassname}[[KrausSelection]{.pre}]{.sig-name .descname}[¶](#cudaq.ptsbe.KrausSelection "Permalink to this definition"){.headerlink}

:   The choice of a specific Kraus operator at one noise site.

    [[circuit_location]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[int]{.pre}](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference .external}*[¶](#cudaq.ptsbe.KrausSelection.circuit_location "Permalink to this definition"){.headerlink}

    :   Index of the noise site in the circuit's instruction sequence.

    [[qubits]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[list]{.pre}](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)"){.reference .external}[[\[]{.pre}]{.p}[[int]{.pre}](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference .external}[[\]]{.pre}]{.p}*[¶](#cudaq.ptsbe.KrausSelection.qubits "Permalink to this definition"){.headerlink}

    :   Qubits affected by this noise operation.

    [[op_name]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[str]{.pre}](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"){.reference .external}*[¶](#cudaq.ptsbe.KrausSelection.op_name "Permalink to this definition"){.headerlink}

    :   Name of the gate after which this noise occurs (e.g.
        [`"h"`{.docutils .literal .notranslate}]{.pre}).

    [[kraus_operator_index]{.pre}]{.sig-name .descname}*[[:]{.pre}]{.p}[ ]{.w}[[int]{.pre}](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"){.reference .external}*[¶](#cudaq.ptsbe.KrausSelection.kraus_operator_index "Permalink to this definition"){.headerlink}

    :   Index of the selected Kraus operator. [`0`{.docutils .literal
        .notranslate}]{.pre} is the identity (no error); values ≥ 1
        represent actual error operators.
:::
:::

------------------------------------------------------------------------

::: {#c-api-cudaq-ptsbe .section}
## [C++ API --- [`cudaq::ptsbe`{.docutils .literal .notranslate}]{.pre}](#id12){.toc-backref role="doc-backlink"}[¶](#c-api-cudaq-ptsbe "Permalink to this heading"){.headerlink}

::: {#id1 .section}
### [Sampling Functions](#id13){.toc-backref role="doc-backlink"}[¶](#id1 "Permalink to this heading"){.headerlink}

[[template]{.pre}]{.k}[[\<]{.pre}]{.p}[[typename]{.pre}]{.k}[ ]{.w}[[[QuantumKernel]{.pre}]{.n}]{.sig-name .descname}[[,]{.pre}]{.p}[ ]{.w}[[typename]{.pre}]{.k}[ ]{.w}[[\...]{.pre}]{.p}[[[Args]{.pre}]{.n}]{.sig-name .descname}[[\>]{.pre}]{.p}\
[[[sample_result]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe13sample_resultE "cudaq::ptsbe::sample_result"){.reference .internal}[ ]{.w}[[[sample]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[[const]{.pre}]{.k}[ ]{.w}[[[sample_options]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe14sample_optionsE "cudaq::ptsbe::sample_options"){.reference .internal}[ ]{.w}[[&]{.pre}]{.p}[[options]{.pre}]{.n .sig-param}, [[[QuantumKernel]{.pre}]{.n}](#_CPPv4I0DpEN5cudaq5ptsbe6sampleE13sample_resultRK14sample_optionsRR13QuantumKernelDpRR4Args "cudaq::ptsbe::sample::QuantumKernel"){.reference .internal}[ ]{.w}[[&]{.pre}]{.p}[[&]{.pre}]{.p}[[kernel]{.pre}]{.n .sig-param}, [[[Args]{.pre}]{.n}](#_CPPv4I0DpEN5cudaq5ptsbe6sampleE13sample_resultRK14sample_optionsRR13QuantumKernelDpRR4Args "cudaq::ptsbe::sample::Args"){.reference .internal}[[&]{.pre}]{.p}[[&]{.pre}]{.p}[[\...]{.pre}]{.p}[ ]{.w}[[args]{.pre}]{.n .sig-param}[)]{.sig-paren}[¶](#_CPPv4I0DpEN5cudaq5ptsbe6sampleE13sample_resultRK14sample_optionsRR13QuantumKernelDpRR4Args "Permalink to this definition"){.headerlink}\

:   Sample a quantum kernel using PTSBE.

    Template Parameters[:]{.colon}

    :   -   **QuantumKernel** -- A CUDA-Q kernel callable.

        -   **Args** -- Kernel argument types.

    Parameters[:]{.colon}

    :   -   **options** -- Execution options (shots, noise model, PTSBE
            configuration).

        -   **kernel** -- The kernel to execute.

        -   **args** -- Arguments forwarded to the kernel.

    Returns[:]{.colon}

    :   Aggregated [`sample_result`{.docutils .literal
        .notranslate}]{.pre}.

    ::: {.highlight-cpp .notranslate}
    ::: highlight
        #include "cudaq/ptsbe/PTSBESample.h"

        cudaq::ptsbe::sample_options opts;
        opts.shots           = 10'000;
        opts.noise           = noise_model;
        opts.ptsbe.max_trajectories = 200;

        auto result = cudaq::ptsbe::sample(opts, bell);
        result.dump();
    :::
    :::

```{=html}
<!-- -->
```

[[template]{.pre}]{.k}[[\<]{.pre}]{.p}[[typename]{.pre}]{.k}[ ]{.w}[[[QuantumKernel]{.pre}]{.n}]{.sig-name .descname}[[,]{.pre}]{.p}[ ]{.w}[[typename]{.pre}]{.k}[ ]{.w}[[\...]{.pre}]{.p}[[[Args]{.pre}]{.n}]{.sig-name .descname}[[\>]{.pre}]{.p}\
[[[cudaq]{.pre}]{.n}](languages/cpp_api.html#_CPPv45cudaq "cudaq"){.reference .internal}[[::]{.pre}]{.p}[[[async_sample_result]{.pre}]{.n}](languages/cpp_api.html#_CPPv4N5cudaq19async_sample_resultE "cudaq::async_sample_result"){.reference .internal}[ ]{.w}[[[sample_async]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[[const]{.pre}]{.k}[ ]{.w}[[[sample_options]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe14sample_optionsE "cudaq::ptsbe::sample_options"){.reference .internal}[ ]{.w}[[&]{.pre}]{.p}[[options]{.pre}]{.n .sig-param}, [[[QuantumKernel]{.pre}]{.n}](#_CPPv4I0DpEN5cudaq5ptsbe12sample_asyncEN5cudaq19async_sample_resultERK14sample_optionsRR13QuantumKernelDpRR4ArgsNSt6size_tE "cudaq::ptsbe::sample_async::QuantumKernel"){.reference .internal}[ ]{.w}[[&]{.pre}]{.p}[[&]{.pre}]{.p}[[kernel]{.pre}]{.n .sig-param}, [[[Args]{.pre}]{.n}](#_CPPv4I0DpEN5cudaq5ptsbe12sample_asyncEN5cudaq19async_sample_resultERK14sample_optionsRR13QuantumKernelDpRR4ArgsNSt6size_tE "cudaq::ptsbe::sample_async::Args"){.reference .internal}[[&]{.pre}]{.p}[[&]{.pre}]{.p}[[\...]{.pre}]{.p}[ ]{.w}[[args]{.pre}]{.n .sig-param}, [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[ ]{.w}[[qpu_id]{.pre}]{.n .sig-param}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[)]{.sig-paren}[¶](#_CPPv4I0DpEN5cudaq5ptsbe12sample_asyncEN5cudaq19async_sample_resultERK14sample_optionsRR13QuantumKernelDpRR4ArgsNSt6size_tE "Permalink to this definition"){.headerlink}\

:   Asynchronous variant of [[`sample()`{.xref .cpp .cpp-func .docutils
    .literal
    .notranslate}]{.pre}](#_CPPv4I0DpEN5cudaq5ptsbe6sampleE13sample_resultRK14sample_optionsRR13QuantumKernelDpRR4Args "cudaq::ptsbe::sample"){.reference
    .internal}. Returns a [`std::future<sample_result>`{.docutils
    .literal .notranslate}]{.pre}.

    ::: {.highlight-cpp .notranslate}
    ::: highlight
        auto future = cudaq::ptsbe::sample_async(opts, bell);
        auto result = future.get();
    :::
    :::
:::

------------------------------------------------------------------------

::: {#options .section}
### [Options](#id14){.toc-backref role="doc-backlink"}[¶](#options "Permalink to this heading"){.headerlink}

[[struct]{.pre}]{.k}[ ]{.w}[[[sample_options]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe14sample_optionsE "Permalink to this definition"){.headerlink}\

:   Top-level options passed to [[`sample()`{.xref .cpp .cpp-func
    .docutils .literal
    .notranslate}]{.pre}](#_CPPv4I0DpEN5cudaq5ptsbe6sampleE13sample_resultRK14sample_optionsRR13QuantumKernelDpRR4Args "cudaq::ptsbe::sample"){.reference
    .internal}.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[ ]{.w}[[[shots]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[1000]{.pre}]{.m}[¶](#_CPPv4N5cudaq5ptsbe14sample_options5shotsE "Permalink to this definition"){.headerlink}\

    :   Total number of measurement shots.

    [[[cudaq]{.pre}]{.n}](languages/cpp_api.html#_CPPv45cudaq "cudaq"){.reference .internal}[[::]{.pre}]{.p}[[[noise_model]{.pre}]{.n}](languages/cpp_api.html#_CPPv4N5cudaq11noise_modelE "cudaq::noise_model"){.reference .internal}[ ]{.w}[[[noise]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe14sample_options5noiseE "Permalink to this definition"){.headerlink}\

    :   Noise model describing gate-level error channels.

    [[[PTSBEOptions]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe12PTSBEOptionsE "cudaq::ptsbe::PTSBEOptions"){.reference .internal}[ ]{.w}[[[ptsbe]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe14sample_options5ptsbeE "Permalink to this definition"){.headerlink}\

    :   PTSBE-specific configuration (trajectories, strategy,
        allocation).

```{=html}
<!-- -->
```

[[struct]{.pre}]{.k}[ ]{.w}[[[PTSBEOptions]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe12PTSBEOptionsE "Permalink to this definition"){.headerlink}\

:   PTSBE-specific execution configuration.

    [[bool]{.pre}]{.kt}[ ]{.w}[[[return_execution_data]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[false]{.pre}]{.k}[¶](#_CPPv4N5cudaq5ptsbe12PTSBEOptions21return_execution_dataE "Permalink to this definition"){.headerlink}\

    :   When [`true`{.docutils .literal .notranslate}]{.pre}, the
        returned result contains a [[`PTSBEExecutionData`{.xref .cpp
        .cpp-struct .docutils .literal
        .notranslate}]{.pre}](#_CPPv4N5cudaq5ptsbe18PTSBEExecutionDataE "cudaq::ptsbe::PTSBEExecutionData"){.reference
        .internal} payload with the circuit trace, trajectory details,
        and per-trajectory measurement counts.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[optional]{.pre}]{.n}[[\<]{.pre}]{.p}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[[\>]{.pre}]{.p}[ ]{.w}[[[max_trajectories]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[nullopt]{.pre}]{.n}[¶](#_CPPv4N5cudaq5ptsbe12PTSBEOptions16max_trajectoriesE "Permalink to this definition"){.headerlink}\

    :   Maximum number of unique trajectories to generate.
        [`std::nullopt`{.docutils .literal .notranslate}]{.pre} defaults
        to the shot count.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[shared_ptr]{.pre}]{.n}[[\<]{.pre}]{.p}[[[PTSSamplingStrategy]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe19PTSSamplingStrategyE "cudaq::ptsbe::PTSSamplingStrategy"){.reference .internal}[[\>]{.pre}]{.p}[ ]{.w}[[[strategy]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[nullptr]{.pre}]{.k}[¶](#_CPPv4N5cudaq5ptsbe12PTSBEOptions8strategyE "Permalink to this definition"){.headerlink}\

    :   Trajectory sampling strategy. [`nullptr`{.docutils .literal
        .notranslate}]{.pre} uses the default
        [[`ProbabilisticSamplingStrategy`{.xref .cpp .cpp-class
        .docutils .literal
        .notranslate}]{.pre}](#_CPPv4N5cudaq5ptsbe29ProbabilisticSamplingStrategyE "cudaq::ptsbe::ProbabilisticSamplingStrategy"){.reference
        .internal}.

    [[[ShotAllocationStrategy]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategyE "cudaq::ptsbe::ShotAllocationStrategy"){.reference .internal}[ ]{.w}[[[shot_allocation]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe12PTSBEOptions15shot_allocationE "Permalink to this definition"){.headerlink}\

    :   Shot allocation strategy. Defaults to
        [[`ShotAllocationStrategy::Type::PROPORTIONAL`{.xref .cpp
        .cpp-enumerator .docutils .literal
        .notranslate}]{.pre}](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4Type12PROPORTIONALE "cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL"){.reference
        .internal}.
:::

------------------------------------------------------------------------

::: {#id2 .section}
### [Result Type](#id15){.toc-backref role="doc-backlink"}[¶](#id2 "Permalink to this heading"){.headerlink}

[[class]{.pre}]{.k}[ ]{.w}[[[sample_result]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[:]{.pre}]{.p}[ ]{.w}[[public]{.pre}]{.k}[ ]{.w}[[[cudaq]{.pre}]{.n}](languages/cpp_api.html#_CPPv45cudaq "cudaq"){.reference .internal}[[::]{.pre}]{.p}[[[sample_result]{.pre}]{.n}](languages/cpp_api.html#_CPPv4N5cudaq13sample_resultE "cudaq::sample_result"){.reference .internal}[¶](#_CPPv4N5cudaq5ptsbe13sample_resultE "Permalink to this definition"){.headerlink}\

:   Extends [[`cudaq::sample_result`{.xref .cpp .cpp-class .docutils
    .literal
    .notranslate}]{.pre}](languages/cpp_api.html#_CPPv4N5cudaq13sample_resultE "cudaq::sample_result"){.reference
    .internal} with an optional [[`PTSBEExecutionData`{.xref .cpp
    .cpp-struct .docutils .literal
    .notranslate}]{.pre}](#_CPPv4N5cudaq5ptsbe18PTSBEExecutionDataE "cudaq::ptsbe::PTSBEExecutionData"){.reference
    .internal} payload.

    [[bool]{.pre}]{.kt}[ ]{.w}[[[has_execution_data]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[ ]{.w}[[const]{.pre}]{.k}[¶](#_CPPv4NK5cudaq5ptsbe13sample_result18has_execution_dataEv "Permalink to this definition"){.headerlink}\

    :   Return [`true`{.docutils .literal .notranslate}]{.pre} if
        execution data is attached.

    [[const]{.pre}]{.k}[ ]{.w}[[[PTSBEExecutionData]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe18PTSBEExecutionDataE "cudaq::ptsbe::PTSBEExecutionData"){.reference .internal}[ ]{.w}[[&]{.pre}]{.p}[[[execution_data]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[ ]{.w}[[const]{.pre}]{.k}[¶](#_CPPv4NK5cudaq5ptsbe13sample_result14execution_dataEv "Permalink to this definition"){.headerlink}\

    :   Return the attached execution data.

        Throws[:]{.colon}

        :   [[std]{.n}[::]{.p}[runtime_error]{.n}]{.cpp-expr .sig
            .sig-inline .cpp} -- If no execution data is available.

    [[void]{.pre}]{.kt}[ ]{.w}[[[set_execution_data]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[[[PTSBEExecutionData]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe18PTSBEExecutionDataE "cudaq::ptsbe::PTSBEExecutionData"){.reference .internal}[ ]{.w}[[data]{.pre}]{.n .sig-param}[)]{.sig-paren}[¶](#_CPPv4N5cudaq5ptsbe13sample_result18set_execution_dataE18PTSBEExecutionData "Permalink to this definition"){.headerlink}\

    :   Attach execution data to this result.
:::

------------------------------------------------------------------------

::: {#id3 .section}
### [Trajectory Sampling Strategies](#id16){.toc-backref role="doc-backlink"}[¶](#id3 "Permalink to this heading"){.headerlink}

[[class]{.pre}]{.k}[ ]{.w}[[[PTSSamplingStrategy]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe19PTSSamplingStrategyE "Permalink to this definition"){.headerlink}\

:   Abstract base class for trajectory sampling strategies.

    [[virtual]{.pre}]{.k}[ ]{.w}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[vector]{.pre}]{.n}[[\<]{.pre}]{.p}[[[cudaq]{.pre}]{.n}](languages/cpp_api.html#_CPPv45cudaq "cudaq"){.reference .internal}[[::]{.pre}]{.p}[[[KrausTrajectory]{.pre}]{.n}](#_CPPv4N5cudaq15KrausTrajectoryE "cudaq::KrausTrajectory"){.reference .internal}[[\>]{.pre}]{.p}[ ]{.w}[[[generateTrajectories]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[span]{.pre}]{.n}[[\<]{.pre}]{.p}[[const]{.pre}]{.k}[ ]{.w}[[[cudaq]{.pre}]{.n}](languages/cpp_api.html#_CPPv45cudaq "cudaq"){.reference .internal}[[::]{.pre}]{.p}[[[KrausTrajectory]{.pre}]{.n}](#_CPPv4N5cudaq15KrausTrajectoryE "cudaq::KrausTrajectory"){.reference .internal}[[\>]{.pre}]{.p}[ ]{.w}[[noise_points]{.pre}]{.n .sig-param}, [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[ ]{.w}[[max_trajectories]{.pre}]{.n .sig-param}[)]{.sig-paren}[ ]{.w}[[const]{.pre}]{.k}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[¶](#_CPPv4NK5cudaq5ptsbe19PTSSamplingStrategy20generateTrajectoriesENSt4spanIKN5cudaq15KrausTrajectoryEEENSt6size_tE "Permalink to this definition"){.headerlink}\

    :   Generate up to *max_trajectories* unique trajectories from the
        noise space.

    [[virtual]{.pre}]{.k}[ ]{.w}[[const]{.pre}]{.k}[ ]{.w}[[char]{.pre}]{.kt}[ ]{.w}[[\*]{.pre}]{.p}[[[name]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[ ]{.w}[[const]{.pre}]{.k}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[¶](#_CPPv4NK5cudaq5ptsbe19PTSSamplingStrategy4nameEv "Permalink to this definition"){.headerlink}\

    :   Return the strategy name.

    [[virtual]{.pre}]{.k}[ ]{.w}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[unique_ptr]{.pre}]{.n}[[\<]{.pre}]{.p}[[[PTSSamplingStrategy]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe19PTSSamplingStrategyE "cudaq::ptsbe::PTSSamplingStrategy"){.reference .internal}[[\>]{.pre}]{.p}[ ]{.w}[[[clone]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[ ]{.w}[[const]{.pre}]{.k}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[¶](#_CPPv4NK5cudaq5ptsbe19PTSSamplingStrategy5cloneEv "Permalink to this definition"){.headerlink}\

    :   Return a deep copy of this strategy.

```{=html}
<!-- -->
```

[[class]{.pre}]{.k}[ ]{.w}[[[ProbabilisticSamplingStrategy]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[:]{.pre}]{.p}[ ]{.w}[[public]{.pre}]{.k}[ ]{.w}[[[PTSSamplingStrategy]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe19PTSSamplingStrategyE "cudaq::ptsbe::PTSSamplingStrategy"){.reference .internal}[¶](#_CPPv4N5cudaq5ptsbe29ProbabilisticSamplingStrategyE "Permalink to this definition"){.headerlink}\

:   Randomly samples unique trajectories weighted by probability.

    [[explicit]{.pre}]{.k}[ ]{.w}[[[ProbabilisticSamplingStrategy]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[uint64_t]{.pre}]{.n}[ ]{.w}[[seed]{.pre}]{.n .sig-param}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[)]{.sig-paren}[¶](#_CPPv4N5cudaq5ptsbe29ProbabilisticSamplingStrategy29ProbabilisticSamplingStrategyENSt8uint64_tE "Permalink to this definition"){.headerlink}\

    :   

        Parameters[:]{.colon}

        :   **seed** -- Random seed. [`0`{.docutils .literal
            .notranslate}]{.pre} uses the global CUDA-Q seed if set,
            otherwise [`std::random_device`{.docutils .literal
            .notranslate}]{.pre}.

    ::: {.highlight-cpp .notranslate}
    ::: highlight
        #include "cudaq/ptsbe/strategies/ProbabilisticSamplingStrategy.h"

        opts.ptsbe.strategy =
            std::make_shared<cudaq::ptsbe::ProbabilisticSamplingStrategy>(/*seed=*/42);
    :::
    :::

```{=html}
<!-- -->
```

[[class]{.pre}]{.k}[ ]{.w}[[[OrderedSamplingStrategy]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[:]{.pre}]{.p}[ ]{.w}[[public]{.pre}]{.k}[ ]{.w}[[[PTSSamplingStrategy]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe19PTSSamplingStrategyE "cudaq::ptsbe::PTSSamplingStrategy"){.reference .internal}[¶](#_CPPv4N5cudaq5ptsbe23OrderedSamplingStrategyE "Permalink to this definition"){.headerlink}\

:   Selects the top-*T* trajectories by probability (descending order).

    ::: {.highlight-cpp .notranslate}
    ::: highlight
        #include "cudaq/ptsbe/strategies/OrderedSamplingStrategy.h"

        opts.ptsbe.max_trajectories = 100;
        opts.ptsbe.strategy =
            std::make_shared<cudaq::ptsbe::OrderedSamplingStrategy>();
    :::
    :::

```{=html}
<!-- -->
```

[[class]{.pre}]{.k}[ ]{.w}[[[ExhaustiveSamplingStrategy]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[:]{.pre}]{.p}[ ]{.w}[[public]{.pre}]{.k}[ ]{.w}[[[PTSSamplingStrategy]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe19PTSSamplingStrategyE "cudaq::ptsbe::PTSSamplingStrategy"){.reference .internal}[¶](#_CPPv4N5cudaq5ptsbe26ExhaustiveSamplingStrategyE "Permalink to this definition"){.headerlink}\

:   Enumerates every possible trajectory in lexicographic order.

```{=html}
<!-- -->
```

[[class]{.pre}]{.k}[ ]{.w}[[[ConditionalSamplingStrategy]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[:]{.pre}]{.p}[ ]{.w}[[public]{.pre}]{.k}[ ]{.w}[[[PTSSamplingStrategy]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe19PTSSamplingStrategyE "cudaq::ptsbe::PTSSamplingStrategy"){.reference .internal}[¶](#_CPPv4N5cudaq5ptsbe27ConditionalSamplingStrategyE "Permalink to this definition"){.headerlink}\

:   Samples trajectories that satisfy a user-supplied predicate.

    [[using]{.pre}]{.k}[ ]{.w}[[[TrajectoryPredicate]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[function]{.pre}]{.n}[[\<]{.pre}]{.p}[[bool]{.pre}]{.kt}[[(]{.pre}]{.p}[[const]{.pre}]{.k}[ ]{.w}[[[cudaq]{.pre}]{.n}](languages/cpp_api.html#_CPPv45cudaq "cudaq"){.reference .internal}[[::]{.pre}]{.p}[[[KrausTrajectory]{.pre}]{.n}](#_CPPv4N5cudaq15KrausTrajectoryE "cudaq::KrausTrajectory"){.reference .internal}[[&]{.pre}]{.p}[[)]{.pre}]{.p}[[\>]{.pre}]{.p}[¶](#_CPPv4N5cudaq5ptsbe27ConditionalSamplingStrategy19TrajectoryPredicateE "Permalink to this definition"){.headerlink}\

    :   

    [[explicit]{.pre}]{.k}[ ]{.w}[[[ConditionalSamplingStrategy]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[[[TrajectoryPredicate]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe27ConditionalSamplingStrategy19TrajectoryPredicateE "cudaq::ptsbe::ConditionalSamplingStrategy::TrajectoryPredicate"){.reference .internal}[ ]{.w}[[predicate]{.pre}]{.n .sig-param}, [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[uint64_t]{.pre}]{.n}[ ]{.w}[[seed]{.pre}]{.n .sig-param}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[)]{.sig-paren}[¶](#_CPPv4N5cudaq5ptsbe27ConditionalSamplingStrategy27ConditionalSamplingStrategyE19TrajectoryPredicateNSt8uint64_tE "Permalink to this definition"){.headerlink}\

    :   

        Parameters[:]{.colon}

        :   -   **predicate** -- Returns [`true`{.docutils .literal
                .notranslate}]{.pre} for trajectories to include.

            -   **seed** -- Random seed. [`0`{.docutils .literal
                .notranslate}]{.pre} uses the global CUDA-Q seed.

    ::: {.highlight-cpp .notranslate}
    ::: highlight
        #include "cudaq/ptsbe/strategies/ConditionalSamplingStrategy.h"

        // Only single-error trajectories
        opts.ptsbe.strategy =
            std::make_shared<cudaq::ptsbe::ConditionalSamplingStrategy>(
                [](const cudaq::KrausTrajectory &t) {
                  return t.countErrors() <= 1;
                });
    :::
    :::
:::

------------------------------------------------------------------------

::: {#id4 .section}
### [Shot Allocation Strategy](#id17){.toc-backref role="doc-backlink"}[¶](#id4 "Permalink to this heading"){.headerlink}

[[struct]{.pre}]{.k}[ ]{.w}[[[ShotAllocationStrategy]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategyE "Permalink to this definition"){.headerlink}\

:   Controls how shots are distributed across selected trajectories.

    [[enum]{.pre}]{.k}[ ]{.w}[[class]{.pre}]{.k}[ ]{.w}[[[Type]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4TypeE "Permalink to this definition"){.headerlink}\

    :   

        [[enumerator]{.pre}]{.k}[ ]{.w}[[[PROPORTIONAL]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4Type12PROPORTIONALE "Permalink to this definition"){.headerlink}\

        :   *(default)* Multinomial sampling weighted by trajectory
            probability. Total is always exactly
            [`total_shots`{.docutils .literal .notranslate}]{.pre}.

        [[enumerator]{.pre}]{.k}[ ]{.w}[[[UNIFORM]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4Type7UNIFORME "Permalink to this definition"){.headerlink}\

        :   Equal shots per trajectory.

        [[enumerator]{.pre}]{.k}[ ]{.w}[[[LOW_WEIGHT_BIAS]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4Type15LOW_WEIGHT_BIASE "Permalink to this definition"){.headerlink}\

        :   More shots to low-error trajectories. Weight:
            [`(1`{.docutils .literal .notranslate}]{.pre}` `{.docutils
            .literal .notranslate}[`+`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`error_count)^(-bias_strength)`{.docutils
            .literal .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`*`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`probability`{.docutils .literal
            .notranslate}]{.pre}.

        [[enumerator]{.pre}]{.k}[ ]{.w}[[[HIGH_WEIGHT_BIAS]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4Type16HIGH_WEIGHT_BIASE "Permalink to this definition"){.headerlink}\

        :   More shots to high-error trajectories. Weight:
            [`(1`{.docutils .literal .notranslate}]{.pre}` `{.docutils
            .literal .notranslate}[`+`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`error_count)^(+bias_strength)`{.docutils
            .literal .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`*`{.docutils .literal
            .notranslate}]{.pre}` `{.docutils .literal
            .notranslate}[`probability`{.docutils .literal
            .notranslate}]{.pre}.

    [[[Type]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4TypeE "cudaq::ptsbe::ShotAllocationStrategy::Type"){.reference .internal}[ ]{.w}[[[type]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[[Type]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4TypeE "cudaq::ptsbe::ShotAllocationStrategy::Type"){.reference .internal}[[::]{.pre}]{.p}[[[PROPORTIONAL]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4Type12PROPORTIONALE "cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL"){.reference .internal}[¶](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4typeE "Permalink to this definition"){.headerlink}\

    :   

    [[double]{.pre}]{.kt}[ ]{.w}[[[bias_strength]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[2.0]{.pre}]{.m}[¶](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy13bias_strengthE "Permalink to this definition"){.headerlink}\

    :   Exponent for the biased strategies.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[uint64_t]{.pre}]{.n}[ ]{.w}[[[seed]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[¶](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4seedE "Permalink to this definition"){.headerlink}\

    :   Random seed for the multinomial draw. [`0`{.docutils .literal
        .notranslate}]{.pre} uses the global CUDA-Q seed.

    [[explicit]{.pre}]{.k}[ ]{.w}[[[ShotAllocationStrategy]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[[[Type]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy4TypeE "cudaq::ptsbe::ShotAllocationStrategy::Type"){.reference .internal}[ ]{.w}[[t]{.pre}]{.n .sig-param}, [[double]{.pre}]{.kt}[ ]{.w}[[bias]{.pre}]{.n .sig-param}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[2.0]{.pre}]{.m}, [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[uint64_t]{.pre}]{.n}[ ]{.w}[[seed]{.pre}]{.n .sig-param}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[)]{.sig-paren}[¶](#_CPPv4N5cudaq5ptsbe22ShotAllocationStrategy22ShotAllocationStrategyE4TypedNSt8uint64_tE "Permalink to this definition"){.headerlink}\

    :   

    ::: {.highlight-cpp .notranslate}
    ::: highlight
        #include "cudaq/ptsbe/ShotAllocationStrategy.h"

        opts.ptsbe.shot_allocation = cudaq::ptsbe::ShotAllocationStrategy(
            cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS,
            /*bias=*/3.0);
    :::
    :::
:::

------------------------------------------------------------------------

::: {#id5 .section}
### [Execution Data](#id18){.toc-backref role="doc-backlink"}[¶](#id5 "Permalink to this heading"){.headerlink}

[[struct]{.pre}]{.k}[ ]{.w}[[[PTSBEExecutionData]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe18PTSBEExecutionDataE "Permalink to this definition"){.headerlink}\

:   Full execution trace attached to the result when
    [`return_execution_data`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`=`{.docutils .literal
    .notranslate}]{.pre}` `{.docutils .literal
    .notranslate}[`true`{.docutils .literal .notranslate}]{.pre}.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[vector]{.pre}]{.n}[[\<]{.pre}]{.p}[[[TraceInstruction]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe16TraceInstructionE "cudaq::ptsbe::TraceInstruction"){.reference .internal}[[\>]{.pre}]{.p}[ ]{.w}[[[instructions]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe18PTSBEExecutionData12instructionsE "Permalink to this definition"){.headerlink}\

    :   Ordered circuit operations ([`PTSBETrace`{.docutils .literal
        .notranslate}]{.pre}, alias for
        [`std::vector<TraceInstruction>`{.docutils .literal
        .notranslate}]{.pre}).

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[vector]{.pre}]{.n}[[\<]{.pre}]{.p}[[[cudaq]{.pre}]{.n}](languages/cpp_api.html#_CPPv45cudaq "cudaq"){.reference .internal}[[::]{.pre}]{.p}[[[KrausTrajectory]{.pre}]{.n}](#_CPPv4N5cudaq15KrausTrajectoryE "cudaq::KrausTrajectory"){.reference .internal}[[\>]{.pre}]{.p}[ ]{.w}[[[trajectories]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe18PTSBEExecutionData12trajectoriesE "Permalink to this definition"){.headerlink}\

    :   Trajectories that were sampled and executed.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[ ]{.w}[[[count_instructions]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[[[TraceInstructionType]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe20TraceInstructionTypeE "cudaq::ptsbe::TraceInstructionType"){.reference .internal}[ ]{.w}[[type]{.pre}]{.n .sig-param}, [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[optional]{.pre}]{.n}[[\<]{.pre}]{.p}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[string]{.pre}]{.n}[[\>]{.pre}]{.p}[ ]{.w}[[name]{.pre}]{.n .sig-param}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[nullopt]{.pre}]{.n}[)]{.sig-paren}[ ]{.w}[[const]{.pre}]{.k}[¶](#_CPPv4NK5cudaq5ptsbe18PTSBEExecutionData18count_instructionsE20TraceInstructionTypeNSt8optionalINSt6stringEEE "Permalink to this definition"){.headerlink}\

    :   Count instructions of the given type, optionally filtered by
        name.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[optional]{.pre}]{.n}[[\<]{.pre}]{.p}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[reference_wrapper]{.pre}]{.n}[[\<]{.pre}]{.p}[[const]{.pre}]{.k}[ ]{.w}[[[cudaq]{.pre}]{.n}](languages/cpp_api.html#_CPPv45cudaq "cudaq"){.reference .internal}[[::]{.pre}]{.p}[[[KrausTrajectory]{.pre}]{.n}](#_CPPv4N5cudaq15KrausTrajectoryE "cudaq::KrausTrajectory"){.reference .internal}[[\>]{.pre}]{.p}[[\>]{.pre}]{.p}[ ]{.w}[[[get_trajectory]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[ ]{.w}[[trajectory_id]{.pre}]{.n .sig-param}[)]{.sig-paren}[ ]{.w}[[const]{.pre}]{.k}[¶](#_CPPv4NK5cudaq5ptsbe18PTSBEExecutionData14get_trajectoryENSt6size_tE "Permalink to this definition"){.headerlink}\

    :   Look up a trajectory by ID. Returns [`std::nullopt`{.docutils
        .literal .notranslate}]{.pre} if not found.

```{=html}
<!-- -->
```

[[struct]{.pre}]{.k}[ ]{.w}[[[TraceInstruction]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe16TraceInstructionE "Permalink to this definition"){.headerlink}\

:   A single operation in the PTSBE execution trace.

    [[[TraceInstructionType]{.pre}]{.n}](#_CPPv4N5cudaq5ptsbe20TraceInstructionTypeE "cudaq::ptsbe::TraceInstructionType"){.reference .internal}[ ]{.w}[[[type]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe16TraceInstruction4typeE "Permalink to this definition"){.headerlink}\

    :   

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[string]{.pre}]{.n}[ ]{.w}[[[name]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe16TraceInstruction4nameE "Permalink to this definition"){.headerlink}\

    :   Operation name (e.g. [`"h"`{.docutils .literal
        .notranslate}]{.pre}, [`"depolarizing"`{.docutils .literal
        .notranslate}]{.pre}, [`"mz"`{.docutils .literal
        .notranslate}]{.pre}).

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[vector]{.pre}]{.n}[[\<]{.pre}]{.p}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[[\>]{.pre}]{.p}[ ]{.w}[[[targets]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe16TraceInstruction7targetsE "Permalink to this definition"){.headerlink}\

    :   

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[vector]{.pre}]{.n}[[\<]{.pre}]{.p}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[[\>]{.pre}]{.p}[ ]{.w}[[[controls]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe16TraceInstruction8controlsE "Permalink to this definition"){.headerlink}\

    :   

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[vector]{.pre}]{.n}[[\<]{.pre}]{.p}[[double]{.pre}]{.kt}[[\>]{.pre}]{.p}[ ]{.w}[[[params]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe16TraceInstruction6paramsE "Permalink to this definition"){.headerlink}\

    :   

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[optional]{.pre}]{.n}[[\<]{.pre}]{.p}[[[cudaq]{.pre}]{.n}](languages/cpp_api.html#_CPPv45cudaq "cudaq"){.reference .internal}[[::]{.pre}]{.p}[[[kraus_channel]{.pre}]{.n}](languages/cpp_api.html#_CPPv4N5cudaq13kraus_channelE "cudaq::kraus_channel"){.reference .internal}[[\>]{.pre}]{.p}[ ]{.w}[[[channel]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe16TraceInstruction7channelE "Permalink to this definition"){.headerlink}\

    :   Populated only for [`Noise`{.docutils .literal
        .notranslate}]{.pre} instructions.

```{=html}
<!-- -->
```

[[enum]{.pre}]{.k}[ ]{.w}[[class]{.pre}]{.k}[ ]{.w}[[[TraceInstructionType]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe20TraceInstructionTypeE "Permalink to this definition"){.headerlink}\

:   

    [[enumerator]{.pre}]{.k}[ ]{.w}[[[Gate]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe20TraceInstructionType4GateE "Permalink to this definition"){.headerlink}\

    :   

    [[enumerator]{.pre}]{.k}[ ]{.w}[[[Noise]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe20TraceInstructionType5NoiseE "Permalink to this definition"){.headerlink}\

    :   

    [[enumerator]{.pre}]{.k}[ ]{.w}[[[Measurement]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq5ptsbe20TraceInstructionType11MeasurementE "Permalink to this definition"){.headerlink}\

    :   
:::

------------------------------------------------------------------------

::: {#trajectory-and-selection-types .section}
### [Trajectory and Selection Types](#id19){.toc-backref role="doc-backlink"}[¶](#trajectory-and-selection-types "Permalink to this heading"){.headerlink}

[[struct]{.pre}]{.k}[ ]{.w}[[[KrausTrajectory]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq15KrausTrajectoryE "Permalink to this definition"){.headerlink}\

:   One complete assignment of Kraus operators across all circuit noise
    sites.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[ ]{.w}[[[trajectory_id]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[¶](#_CPPv4N5cudaq15KrausTrajectory13trajectory_idE "Permalink to this definition"){.headerlink}\

    :   

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[vector]{.pre}]{.n}[[\<]{.pre}]{.p}[[[KrausSelection]{.pre}]{.n}](#_CPPv4N5cudaq14KrausSelectionE "cudaq::KrausSelection"){.reference .internal}[[\>]{.pre}]{.p}[ ]{.w}[[[kraus_selections]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq15KrausTrajectory16kraus_selectionsE "Permalink to this definition"){.headerlink}\

    :   Ordered by [`circuit_location`{.docutils .literal
        .notranslate}]{.pre} (ascending).

    [[double]{.pre}]{.kt}[ ]{.w}[[[probability]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0.0]{.pre}]{.m}[¶](#_CPPv4N5cudaq15KrausTrajectory11probabilityE "Permalink to this definition"){.headerlink}\

    :   Product of the selected Kraus operator probabilities at each
        site.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[ ]{.w}[[[num_shots]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[¶](#_CPPv4N5cudaq15KrausTrajectory9num_shotsE "Permalink to this definition"){.headerlink}\

    :   Shots allocated to this trajectory.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[ ]{.w}[[[multiplicity]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[1]{.pre}]{.m}[¶](#_CPPv4N5cudaq15KrausTrajectory12multiplicityE "Permalink to this definition"){.headerlink}\

    :   Draw count before deduplication.

    [[CountsDictionary]{.pre}]{.n}[ ]{.w}[[[measurement_counts]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq15KrausTrajectory18measurement_countsE "Permalink to this definition"){.headerlink}\

    :   Per-trajectory measurement outcomes (populated after execution).

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[ ]{.w}[[[countErrors]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[ ]{.w}[[const]{.pre}]{.k}[¶](#_CPPv4NK5cudaq15KrausTrajectory11countErrorsEv "Permalink to this definition"){.headerlink}\

    :   Return the number of non-identity Kraus selections (error
        weight).

    [[bool]{.pre}]{.kt}[ ]{.w}[[[isOrdered]{.pre}]{.n}]{.sig-name .descname}[(]{.sig-paren}[)]{.sig-paren}[ ]{.w}[[const]{.pre}]{.k}[¶](#_CPPv4NK5cudaq15KrausTrajectory9isOrderedEv "Permalink to this definition"){.headerlink}\

    :   Return [`true`{.docutils .literal .notranslate}]{.pre} if
        [`kraus_selections`{.docutils .literal .notranslate}]{.pre} are
        sorted by [`circuit_location`{.docutils .literal
        .notranslate}]{.pre}.

```{=html}
<!-- -->
```

[[struct]{.pre}]{.k}[ ]{.w}[[[KrausSelection]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq14KrausSelectionE "Permalink to this definition"){.headerlink}\

:   The choice of a specific Kraus operator at one noise site.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[ ]{.w}[[[circuit_location]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[¶](#_CPPv4N5cudaq14KrausSelection16circuit_locationE "Permalink to this definition"){.headerlink}\

    :   Index of the noise site in the circuit instruction sequence.

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[vector]{.pre}]{.n}[[\<]{.pre}]{.p}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[[\>]{.pre}]{.p}[ ]{.w}[[[qubits]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq14KrausSelection6qubitsE "Permalink to this definition"){.headerlink}\

    :   

    [[std]{.pre}]{.n}[[::]{.pre}]{.p}[[string]{.pre}]{.n}[ ]{.w}[[[op_name]{.pre}]{.n}]{.sig-name .descname}[¶](#_CPPv4N5cudaq14KrausSelection7op_nameE "Permalink to this definition"){.headerlink}\

    :   Gate name after which this noise occurs (e.g. [`"h"`{.docutils
        .literal .notranslate}]{.pre}).

    [[[KrausOperatorType]{.pre}]{.n}](#_CPPv4N5cudaq17KrausOperatorTypeE "cudaq::KrausOperatorType"){.reference .internal}[ ]{.w}[[[kraus_operator_index]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[[KrausOperatorType]{.pre}]{.n}](#_CPPv4N5cudaq17KrausOperatorTypeE "cudaq::KrausOperatorType"){.reference .internal}[[::]{.pre}]{.p}[[[IDENTITY]{.pre}]{.n}](#_CPPv4N5cudaq17KrausOperatorType8IDENTITYE "cudaq::KrausOperatorType::IDENTITY"){.reference .internal}[¶](#_CPPv4N5cudaq14KrausSelection20kraus_operator_indexE "Permalink to this definition"){.headerlink}\

    :   Selected Kraus operator index. [`IDENTITY`{.docutils .literal
        .notranslate}]{.pre} (0) means no error.

```{=html}
<!-- -->
```

[[enum]{.pre}]{.k}[ ]{.w}[[class]{.pre}]{.k}[ ]{.w}[[[KrausOperatorType]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[:]{.pre}]{.p}[ ]{.w}[[std]{.pre}]{.n}[[::]{.pre}]{.p}[[size_t]{.pre}]{.n}[¶](#_CPPv4N5cudaq17KrausOperatorTypeE "Permalink to this definition"){.headerlink}\

:   

    [[enumerator]{.pre}]{.k}[ ]{.w}[[[IDENTITY]{.pre}]{.n}]{.sig-name .descname}[ ]{.w}[[=]{.pre}]{.p}[ ]{.w}[[0]{.pre}]{.m}[¶](#_CPPv4N5cudaq17KrausOperatorType8IDENTITYE "Permalink to this definition"){.headerlink}\

    :   The identity (no-error) Kraus operator.

    Values ≥ 1 correspond to actual error operators from the noise
    channel, indexed in the order they appear in the
    [[`cudaq::kraus_channel`{.xref .cpp .cpp-class .docutils .literal
    .notranslate}]{.pre}](languages/cpp_api.html#_CPPv4N5cudaq13kraus_channelE "cudaq::kraus_channel"){.reference
    .internal}.
:::
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](default_ops.html "Quantum Operations"){.btn .btn-neutral
.float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](../using/user_guide.html "User Guide"){.btn
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
