::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-4013
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
        Execution](using/ptsbe.html){.reference .internal}
        -   [PTSBE User Guide](using/ptsbe_user_guide.html){.reference
            .internal}
            -   [Conceptual
                Overview](using/ptsbe_user_guide.html#conceptual-overview){.reference
                .internal}
            -   [When to Use
                PTSBE](using/ptsbe_user_guide.html#when-to-use-ptsbe){.reference
                .internal}
            -   [Quick
                Start](using/ptsbe_user_guide.html#quick-start){.reference
                .internal}
            -   [Usage
                Tutorial](using/ptsbe_user_guide.html#usage-tutorial){.reference
                .internal}
            -   [Trajectory vs Shot
                Trade-offs](using/ptsbe_user_guide.html#trajectory-vs-shot-trade-offs){.reference
                .internal}
            -   [Backend
                Requirements](using/ptsbe_user_guide.html#backend-requirements){.reference
                .internal}
            -   [References](using/ptsbe_user_guide.html#references){.reference
                .internal}
        -   [PTSBE End-to-End
            Workflow](examples/python/ptsbe_end_to_end_workflow.html){.reference
            .internal}
            -   [1. Set up the
                environment](examples/python/ptsbe_end_to_end_workflow.html#1.-Set-up-the-environment){.reference
                .internal}
            -   [2. Define the circuit and noise
                model](examples/python/ptsbe_end_to_end_workflow.html#2.-Define-the-circuit-and-noise-model){.reference
                .internal}
            -   [3. Run PTSBE
                sampling](examples/python/ptsbe_end_to_end_workflow.html#3.-Run-PTSBE-sampling){.reference
                .internal}
            -   [4. Compare with standard (density-matrix)
                sampling](examples/python/ptsbe_end_to_end_workflow.html#4.-Compare-with-standard-(density-matrix)-sampling){.reference
                .internal}
            -   [5. Return execution
                data](examples/python/ptsbe_end_to_end_workflow.html#5.-Return-execution-data){.reference
                .internal}
            -   [6. Two API
                options:](examples/python/ptsbe_end_to_end_workflow.html#6.-Two-API-options:){.reference
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
    -   [Dynamics
        Examples](using/examples/dynamics_examples.html){.reference
        .internal}
        -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
            Model)](examples/python/dynamics/dynamics_intro_1.html){.reference
            .internal}
            -   [Why dynamics simulations vs. circuit
                simulations?](examples/python/dynamics/dynamics_intro_1.html#Why-dynamics-simulations-vs.-circuit-simulations?){.reference
                .internal}
            -   [Functionality](examples/python/dynamics/dynamics_intro_1.html#Functionality){.reference
                .internal}
            -   [Performance](examples/python/dynamics/dynamics_intro_1.html#Performance){.reference
                .internal}
            -   [Section 1 - Simulating the Jaynes-Cummings
                Hamiltonian](examples/python/dynamics/dynamics_intro_1.html#Section-1---Simulating-the-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Exercise 1 - Simulating a many-photon Jaynes-Cummings
                Hamiltonian](examples/python/dynamics/dynamics_intro_1.html#Exercise-1---Simulating-a-many-photon-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Section 2 - Simulating open quantum systems with the
                [`collapse_operators`{.docutils .literal
                .notranslate}]{.pre}](examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
                .internal}
            -   [Exercise 2 - Adding additional jump operators
                [\\(L_i\\)]{.math .notranslate
                .nohighlight}](examples/python/dynamics/dynamics_intro_1.html#Exercise-2---Adding-additional-jump-operators-L_i){.reference
                .internal}
            -   [Section 3 - Many qubits coupled to the
                resonator](examples/python/dynamics/dynamics_intro_1.html#Section-3---Many-qubits-coupled-to-the-resonator){.reference
                .internal}
        -   [Introduction to CUDA-Q Dynamics (Time Dependent
            Hamiltonians)](examples/python/dynamics/dynamics_intro_2.html){.reference
            .internal}
            -   [The Landau-Zener
                model](examples/python/dynamics/dynamics_intro_2.html#The-Landau-Zener-model){.reference
                .internal}
            -   [Section 1 - Implementing time dependent
                terms](examples/python/dynamics/dynamics_intro_2.html#Section-1---Implementing-time-dependent-terms){.reference
                .internal}
            -   [Section 2 - Implementing custom
                operators](examples/python/dynamics/dynamics_intro_2.html#Section-2---Implementing-custom-operators){.reference
                .internal}
            -   [Section 3 - Heisenberg Model with a time-varying
                magnetic
                field](examples/python/dynamics/dynamics_intro_2.html#Section-3---Heisenberg-Model-with-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 1 - Define a time-varying magnetic
                field](examples/python/dynamics/dynamics_intro_2.html#Exercise-1---Define-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 2
                (Optional)](examples/python/dynamics/dynamics_intro_2.html#Exercise-2-(Optional)){.reference
                .internal}
        -   [Superconducting
            Qubits](examples/python/dynamics/superconducting.html){.reference
            .internal}
            -   [Cavity
                QED](examples/python/dynamics/superconducting.html#Cavity-QED){.reference
                .internal}
            -   [Cross
                Resonance](examples/python/dynamics/superconducting.html#Cross-Resonance){.reference
                .internal}
            -   [Transmon
                Resonator](examples/python/dynamics/superconducting.html#Transmon-Resonator){.reference
                .internal}
        -   [Spin
            Qubits](examples/python/dynamics/spinqubits.html){.reference
            .internal}
            -   [Silicon Spin
                Qubit](examples/python/dynamics/spinqubits.html#Silicon-Spin-Qubit){.reference
                .internal}
            -   [Heisenberg
                Model](examples/python/dynamics/spinqubits.html#Heisenberg-Model){.reference
                .internal}
        -   [Trapped Ion
            Qubits](examples/python/dynamics/iontrap.html){.reference
            .internal}
            -   [GHZ
                state](examples/python/dynamics/iontrap.html#GHZ-state){.reference
                .internal}
        -   [Control](examples/python/dynamics/control.html){.reference
            .internal}
            -   [Gate
                Calibration](examples/python/dynamics/control.html#Gate-Calibration){.reference
                .internal}
            -   [Pulse](examples/python/dynamics/control.html#Pulse){.reference
                .internal}
            -   [Qubit
                Control](examples/python/dynamics/control.html#Qubit-Control){.reference
                .internal}
            -   [Qubit
                Dynamics](examples/python/dynamics/control.html#Qubit-Dynamics){.reference
                .internal}
            -   [Landau-Zenner](examples/python/dynamics/control.html#Landau-Zenner){.reference
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
        -   [5. Compuare
            results](applications/python/qsci.html#5.-Compuare-results){.reference
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
    -   [Anderson Impurity Model ground state solver on Infleqtion's
        Sqale](applications/python/logical_aim_sqale.html){.reference
        .internal}
        -   [Performing logical Variational Quantum Eigensolver (VQE)
            with
            CUDA-QX](applications/python/logical_aim_sqale.html#Performing-logical-Variational-Quantum-Eigensolver-(VQE)-with-CUDA-QX){.reference
            .internal}
        -   [Constructing circuits in the [`[[4,2,2]]`{.docutils
            .literal .notranslate}]{.pre}
            encoding](applications/python/logical_aim_sqale.html#Constructing-circuits-in-the-%5B%5B4,2,2%5D%5D-encoding){.reference
            .internal}
        -   [Setting up submission and decoding
            workflow](applications/python/logical_aim_sqale.html#Setting-up-submission-and-decoding-workflow){.reference
            .internal}
        -   [Running a CUDA-Q noisy
            simulation](applications/python/logical_aim_sqale.html#Running-a-CUDA-Q-noisy-simulation){.reference
            .internal}
        -   [Running logical AIM on Infleqtion's
            hardware](applications/python/logical_aim_sqale.html#Running-logical-AIM-on-Infleqtion's-hardware){.reference
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
    -   [Quantum
        Transformer](applications/python/quantum_transformer.html){.reference
        .internal}
        -   [Installation](applications/python/quantum_transformer.html#Installation){.reference
            .internal}
        -   [Algorithm and
            Example](applications/python/quantum_transformer.html#Algorithm-and-Example){.reference
            .internal}
            -   [Creating the self-attention
                circuits](applications/python/quantum_transformer.html#Creating-the-self-attention-circuits){.reference
                .internal}
        -   [Usage](applications/python/quantum_transformer.html#Usage){.reference
            .internal}
            -   [Model
                Training](applications/python/quantum_transformer.html#Model-Training){.reference
                .internal}
            -   [Generating
                Molecules](applications/python/quantum_transformer.html#Generating-Molecules){.reference
                .internal}
            -   [Attention
                Maps](applications/python/quantum_transformer.html#Attention-Maps){.reference
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
            -   [The SKQD Algorithm: Matrix Construction
                Details](applications/python/skqd.html#The-SKQD-Algorithm:-Matrix-Construction-Details){.reference
                .internal}
        -   [Results Analysis and
            Convergence](applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
        -   [GPU Acceleration for
            Postprocessing](applications/python/skqd.html#GPU-Acceleration-for-Postprocessing){.reference
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
[**S**](#S) \| [**T**](#T) \| [**U**](#U) \| [**V**](#V) \| [**X**](#X)
\| [**Y**](#Y) \| [**Z**](#Z)
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
| -   [BaseIntegrator (class in     | -   [beta_reduction()             |
|                                   |     (cudaq.PyKernelDecorator      |
| cudaq.dynamics.integrator)](api/l |     method)](api                  |
| anguages/python_api.html#cudaq.dy | /languages/python_api.html#cudaq. |
| namics.integrator.BaseIntegrator) | PyKernelDecorator.beta_reduction) |
| -   [batch_size                   | -   [BitFlipChannel (class in     |
|     (cudaq.optimizers.Adam        |     cudaq)](api/languages/pyth    |
|     property                      | on_api.html#cudaq.BitFlipChannel) |
| )](api/languages/python_api.html# | -   [BosonOperator (class in      |
| cudaq.optimizers.Adam.batch_size) |     cudaq.operators.boson)](      |
|     -   [(cudaq.optimizers.SGD    | api/languages/python_api.html#cud |
|         propert                   | aq.operators.boson.BosonOperator) |
| y)](api/languages/python_api.html | -   [BosonOperatorElement (class  |
| #cudaq.optimizers.SGD.batch_size) |     in                            |
| -   [beta1 (cudaq.optimizers.Adam |                                   |
|     pro                           |   cudaq.operators.boson)](api/lan |
| perty)](api/languages/python_api. | guages/python_api.html#cudaq.oper |
| html#cudaq.optimizers.Adam.beta1) | ators.boson.BosonOperatorElement) |
| -   [beta2 (cudaq.optimizers.Adam | -   [BosonOperatorTerm (class in  |
|     pro                           |     cudaq.operators.boson)](api/  |
| perty)](api/languages/python_api. | languages/python_api.html#cudaq.o |
| html#cudaq.optimizers.Adam.beta2) | perators.boson.BosonOperatorTerm) |
|                                   | -   [broadcast() (in module       |
|                                   |     cudaq.mpi)](api/languages/pyt |
|                                   | hon_api.html#cudaq.mpi.broadcast) |
+-----------------------------------+-----------------------------------+

## C {#C}

+-----------------------------------+-----------------------------------+
| -   [canonicalize()               | -   [cudaq::phase_damping (C++    |
|     (cu                           |                                   |
| daq.operators.boson.BosonOperator |  class)](api/languages/cpp_api.ht |
|     method)](api/languages        | ml#_CPPv4N5cudaq13phase_dampingE) |
| /python_api.html#cudaq.operators. | -   [cud                          |
| boson.BosonOperator.canonicalize) | aq::phase_damping::num_parameters |
|     -   [(cudaq.                  |     (C++                          |
| operators.boson.BosonOperatorTerm |     member)](api/lan              |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|        method)](api/languages/pyt | 13phase_damping14num_parametersE) |
| hon_api.html#cudaq.operators.boso | -   [                             |
| n.BosonOperatorTerm.canonicalize) | cudaq::phase_damping::num_targets |
|     -   [(cudaq.                  |     (C++                          |
| operators.fermion.FermionOperator |     member)](api/                 |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|        method)](api/languages/pyt | daq13phase_damping11num_targetsE) |
| hon_api.html#cudaq.operators.ferm | -   [cudaq::phase_flip_channel    |
| ion.FermionOperator.canonicalize) |     (C++                          |
|     -   [(cudaq.oper              |     clas                          |
| ators.fermion.FermionOperatorTerm | s)](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq18phase_flip_channelE) |
|    method)](api/languages/python_ | -   [cudaq::p                     |
| api.html#cudaq.operators.fermion. | hase_flip_channel::num_parameters |
| FermionOperatorTerm.canonicalize) |     (C++                          |
|     -                             |     member)](api/language         |
|  [(cudaq.operators.MatrixOperator | s/cpp_api.html#_CPPv4N5cudaq18pha |
|         method)](api/lang         | se_flip_channel14num_parametersE) |
| uages/python_api.html#cudaq.opera | -   [cudaq                        |
| tors.MatrixOperator.canonicalize) | ::phase_flip_channel::num_targets |
|     -   [(c                       |     (C++                          |
| udaq.operators.MatrixOperatorTerm |     member)](api/langu            |
|         method)](api/language     | ages/cpp_api.html#_CPPv4N5cudaq18 |
| s/python_api.html#cudaq.operators | phase_flip_channel11num_targetsE) |
| .MatrixOperatorTerm.canonicalize) | -   [cudaq::product_op (C++       |
|     -   [(                        |                                   |
| cudaq.operators.spin.SpinOperator |  class)](api/languages/cpp_api.ht |
|         method)](api/languag      | ml#_CPPv4I0EN5cudaq10product_opE) |
| es/python_api.html#cudaq.operator | -   [cudaq::product_op::begin     |
| s.spin.SpinOperator.canonicalize) |     (C++                          |
|     -   [(cuda                    |     functio                       |
| q.operators.spin.SpinOperatorTerm | n)](api/languages/cpp_api.html#_C |
|         method)](api/languages/p  | PPv4NK5cudaq10product_op5beginEv) |
| ython_api.html#cudaq.operators.sp | -                                 |
| in.SpinOperatorTerm.canonicalize) |  [cudaq::product_op::canonicalize |
| -   [canonicalized() (in module   |     (C++                          |
|     cuda                          |     func                          |
| q.boson)](api/languages/python_ap | tion)](api/languages/cpp_api.html |
| i.html#cudaq.boson.canonicalized) | #_CPPv4N5cudaq10product_op12canon |
|     -   [(in module               | icalizeERKNSt3setINSt6size_tEEE), |
|         cudaq.fe                  |     [\[1\]](api                   |
| rmion)](api/languages/python_api. | /languages/cpp_api.html#_CPPv4N5c |
| html#cudaq.fermion.canonicalized) | udaq10product_op12canonicalizeEv) |
|     -   [(in module               | -   [                             |
|                                   | cudaq::product_op::const_iterator |
|        cudaq.operators.custom)](a |     (C++                          |
| pi/languages/python_api.html#cuda |     struct)](api/                 |
| q.operators.custom.canonicalized) | languages/cpp_api.html#_CPPv4N5cu |
|     -   [(in module               | daq10product_op14const_iteratorE) |
|         cu                        | -   [cudaq::product_o             |
| daq.spin)](api/languages/python_a | p::const_iterator::const_iterator |
| pi.html#cudaq.spin.canonicalized) |     (C++                          |
| -   [captured_variables()         |     fu                            |
|     (cudaq.PyKernelDecorator      | nction)](api/languages/cpp_api.ht |
|     method)](api/lan              | ml#_CPPv4N5cudaq10product_op14con |
| guages/python_api.html#cudaq.PyKe | st_iterator14const_iteratorEPK10p |
| rnelDecorator.captured_variables) | roduct_opI9HandlerTyENSt6size_tE) |
| -   [CentralDifference (class in  | -   [cudaq::produ                 |
|     cudaq.gradients)              | ct_op::const_iterator::operator!= |
| ](api/languages/python_api.html#c |     (C++                          |
| udaq.gradients.CentralDifference) |     fun                           |
| -   [channel                      | ction)](api/languages/cpp_api.htm |
|     (cudaq.operators.cu           | l#_CPPv4NK5cudaq10product_op14con |
| stom.cudaq.ptsbe.TraceInstruction | st_iteratorneERK14const_iterator) |
|     att                           | -   [cudaq::produ                 |
| ribute)](api/languages/python_api | ct_op::const_iterator::operator\* |
| .html#cudaq.operators.custom.cuda |     (C++                          |
| q.ptsbe.TraceInstruction.channel) |     function)](api/lang           |
| -   [circuit_location             | uages/cpp_api.html#_CPPv4NK5cudaq |
|     (cudaq.operators.             | 10product_op14const_iteratormlEv) |
| custom.cudaq.ptsbe.KrausSelection | -   [cudaq::produ                 |
|     attribute)                    | ct_op::const_iterator::operator++ |
| ](api/languages/python_api.html#c |     (C++                          |
| udaq.operators.custom.cudaq.ptsbe |     function)](api/lang           |
| .KrausSelection.circuit_location) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [clear() (cudaq.Resources     | 0product_op14const_iteratorppEi), |
|     method)](api/languages/pytho  |     [\[1\]](api/lan               |
| n_api.html#cudaq.Resources.clear) | guages/cpp_api.html#_CPPv4N5cudaq |
|     -   [(cudaq.SampleResult      | 10product_op14const_iteratorppEv) |
|                                   | -   [cudaq::produc                |
|   method)](api/languages/python_a | t_op::const_iterator::operator\-- |
| pi.html#cudaq.SampleResult.clear) |     (C++                          |
| -   [COBYLA (class in             |     function)](api/lang           |
|     cudaq.o                       | uages/cpp_api.html#_CPPv4N5cudaq1 |
| ptimizers)](api/languages/python_ | 0product_op14const_iteratormmEi), |
| api.html#cudaq.optimizers.COBYLA) |     [\[1\]](api/lan               |
| -   [coefficient                  | guages/cpp_api.html#_CPPv4N5cudaq |
|     (cudaq.                       | 10product_op14const_iteratormmEv) |
| operators.boson.BosonOperatorTerm | -   [cudaq::produc                |
|     property)](api/languages/py   | t_op::const_iterator::operator-\> |
| thon_api.html#cudaq.operators.bos |     (C++                          |
| on.BosonOperatorTerm.coefficient) |     function)](api/lan            |
|     -   [(cudaq.oper              | guages/cpp_api.html#_CPPv4N5cudaq |
| ators.fermion.FermionOperatorTerm | 10product_op14const_iteratorptEv) |
|                                   | -   [cudaq::produ                 |
|   property)](api/languages/python | ct_op::const_iterator::operator== |
| _api.html#cudaq.operators.fermion |     (C++                          |
| .FermionOperatorTerm.coefficient) |     fun                           |
|     -   [(c                       | ction)](api/languages/cpp_api.htm |
| udaq.operators.MatrixOperatorTerm | l#_CPPv4NK5cudaq10product_op14con |
|         property)](api/languag    | st_iteratoreqERK14const_iterator) |
| es/python_api.html#cudaq.operator | -   [cudaq::product_op::degrees   |
| s.MatrixOperatorTerm.coefficient) |     (C++                          |
|     -   [(cuda                    |     function)                     |
| q.operators.spin.SpinOperatorTerm | ](api/languages/cpp_api.html#_CPP |
|         property)](api/languages/ | v4NK5cudaq10product_op7degreesEv) |
| python_api.html#cudaq.operators.s | -   [cudaq::product_op::dump (C++ |
| pin.SpinOperatorTerm.coefficient) |     functi                        |
| -   [col_count                    | on)](api/languages/cpp_api.html#_ |
|     (cudaq.KrausOperator          | CPPv4NK5cudaq10product_op4dumpEv) |
|     prope                         | -   [cudaq::product_op::end (C++  |
| rty)](api/languages/python_api.ht |     funct                         |
| ml#cudaq.KrausOperator.col_count) | ion)](api/languages/cpp_api.html# |
| -   [compile()                    | _CPPv4NK5cudaq10product_op3endEv) |
|     (cudaq.PyKernelDecorator      | -   [c                            |
|     metho                         | udaq::product_op::get_coefficient |
| d)](api/languages/python_api.html |     (C++                          |
| #cudaq.PyKernelDecorator.compile) |     function)](api/lan            |
| -   [ComplexMatrix (class in      | guages/cpp_api.html#_CPPv4NK5cuda |
|     cudaq)](api/languages/pyt     | q10product_op15get_coefficientEv) |
| hon_api.html#cudaq.ComplexMatrix) | -                                 |
| -   [compute()                    |   [cudaq::product_op::get_term_id |
|     (                             |     (C++                          |
| cudaq.gradients.CentralDifference |     function)](api                |
|     method)](api/la               | /languages/cpp_api.html#_CPPv4NK5 |
| nguages/python_api.html#cudaq.gra | cudaq10product_op11get_term_idEv) |
| dients.CentralDifference.compute) | -                                 |
|     -   [(                        |   [cudaq::product_op::is_identity |
| cudaq.gradients.ForwardDifference |     (C++                          |
|         method)](api/la           |     function)](api                |
| nguages/python_api.html#cudaq.gra | /languages/cpp_api.html#_CPPv4NK5 |
| dients.ForwardDifference.compute) | cudaq10product_op11is_identityEv) |
|     -                             | -   [cudaq::product_op::num_ops   |
|  [(cudaq.gradients.ParameterShift |     (C++                          |
|         method)](api              |     function)                     |
| /languages/python_api.html#cudaq. | ](api/languages/cpp_api.html#_CPP |
| gradients.ParameterShift.compute) | v4NK5cudaq10product_op7num_opsEv) |
| -   [const()                      | -                                 |
|                                   |    [cudaq::product_op::operator\* |
|   (cudaq.operators.ScalarOperator |     (C++                          |
|     class                         |     function)](api/languages/     |
|     method)](a                    | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| pi/languages/python_api.html#cuda | oduct_opmlE10product_opI1TERK15sc |
| q.operators.ScalarOperator.const) | alar_operatorRK10product_opI1TE), |
| -   [controls                     |     [\[1\]](api/languages/        |
|     (cudaq.operators.cu           | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| stom.cudaq.ptsbe.TraceInstruction | oduct_opmlE10product_opI1TERK15sc |
|     attr                          | alar_operatorRR10product_opI1TE), |
| ibute)](api/languages/python_api. |     [\[2\]](api/languages/        |
| html#cudaq.operators.custom.cudaq | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| .ptsbe.TraceInstruction.controls) | oduct_opmlE10product_opI1TERR15sc |
| -   [copy()                       | alar_operatorRK10product_opI1TE), |
|     (cu                           |     [\[3\]](api/languages/        |
| daq.operators.boson.BosonOperator | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     method)](api/l                | oduct_opmlE10product_opI1TERR15sc |
| anguages/python_api.html#cudaq.op | alar_operatorRR10product_opI1TE), |
| erators.boson.BosonOperator.copy) |     [\[4\]](api/                  |
|     -   [(cudaq.                  | languages/cpp_api.html#_CPPv4I0EN |
| operators.boson.BosonOperatorTerm | 5cudaq10product_opmlE6sum_opI1TER |
|         method)](api/langu        | K15scalar_operatorRK6sum_opI1TE), |
| ages/python_api.html#cudaq.operat |     [\[5\]](api/                  |
| ors.boson.BosonOperatorTerm.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(cudaq.                  | 5cudaq10product_opmlE6sum_opI1TER |
| operators.fermion.FermionOperator | K15scalar_operatorRR6sum_opI1TE), |
|         method)](api/langu        |     [\[6\]](api/                  |
| ages/python_api.html#cudaq.operat | languages/cpp_api.html#_CPPv4I0EN |
| ors.fermion.FermionOperator.copy) | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [(cudaq.oper              | R15scalar_operatorRK6sum_opI1TE), |
| ators.fermion.FermionOperatorTerm |     [\[7\]](api/                  |
|         method)](api/languages    | languages/cpp_api.html#_CPPv4I0EN |
| /python_api.html#cudaq.operators. | 5cudaq10product_opmlE6sum_opI1TER |
| fermion.FermionOperatorTerm.copy) | R15scalar_operatorRR6sum_opI1TE), |
|     -                             |     [\[8\]](api/languages         |
|  [(cudaq.operators.MatrixOperator | /cpp_api.html#_CPPv4NK5cudaq10pro |
|         method)](                 | duct_opmlERK6sum_opI9HandlerTyE), |
| api/languages/python_api.html#cud |     [\[9\]](api/languages/cpp_a   |
| aq.operators.MatrixOperator.copy) | pi.html#_CPPv4NKR5cudaq10product_ |
|     -   [(c                       | opmlERK10product_opI9HandlerTyE), |
| udaq.operators.MatrixOperatorTerm |     [\[10\]](api/language         |
|         method)](api/             | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| languages/python_api.html#cudaq.o | roduct_opmlERK15scalar_operator), |
| perators.MatrixOperatorTerm.copy) |     [\[11\]](api/languages/cpp_a  |
|     -   [(                        | pi.html#_CPPv4NKR5cudaq10product_ |
| cudaq.operators.spin.SpinOperator | opmlERR10product_opI9HandlerTyE), |
|         method)](api              |     [\[12\]](api/language         |
| /languages/python_api.html#cudaq. | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| operators.spin.SpinOperator.copy) | roduct_opmlERR15scalar_operator), |
|     -   [(cuda                    |     [\[13\]](api/languages/cpp_   |
| q.operators.spin.SpinOperatorTerm | api.html#_CPPv4NO5cudaq10product_ |
|         method)](api/lan          | opmlERK10product_opI9HandlerTyE), |
| guages/python_api.html#cudaq.oper |     [\[14\]](api/languag          |
| ators.spin.SpinOperatorTerm.copy) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [count() (cudaq.Resources     | roduct_opmlERK15scalar_operator), |
|     method)](api/languages/pytho  |     [\[15\]](api/languages/cpp_   |
| n_api.html#cudaq.Resources.count) | api.html#_CPPv4NO5cudaq10product_ |
|     -   [(cudaq.SampleResult      | opmlERR10product_opI9HandlerTyE), |
|                                   |     [\[16\]](api/langua           |
|   method)](api/languages/python_a | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| pi.html#cudaq.SampleResult.count) | product_opmlERR15scalar_operator) |
| -   [count_controls()             | -                                 |
|     (cudaq.Resources              |   [cudaq::product_op::operator\*= |
|     meth                          |     (C++                          |
| od)](api/languages/python_api.htm |     function)](api/languages/cpp  |
| l#cudaq.Resources.count_controls) | _api.html#_CPPv4N5cudaq10product_ |
| -   [count_errors()               | opmLERK10product_opI9HandlerTyE), |
|     (cudaq.operators.c            |     [\[1\]](api/langua            |
| ustom.cudaq.ptsbe.KrausTrajectory | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     meth                          | roduct_opmLERK15scalar_operator), |
| od)](api/languages/python_api.htm |     [\[2\]](api/languages/cp      |
| l#cudaq.operators.custom.cudaq.pt | p_api.html#_CPPv4N5cudaq10product |
| sbe.KrausTrajectory.count_errors) | _opmLERR10product_opI9HandlerTyE) |
| -   [count_instructions()         | -   [cudaq::product_op::operator+ |
|     (cudaq.operators.cust         |     (C++                          |
| om.cudaq.ptsbe.PTSBEExecutionData |     function)](api/langu          |
|     method)](api/                 | ages/cpp_api.html#_CPPv4I0EN5cuda |
| languages/python_api.html#cudaq.o | q10product_opplE6sum_opI1TERK15sc |
| perators.custom.cudaq.ptsbe.PTSBE | alar_operatorRK10product_opI1TE), |
| ExecutionData.count_instructions) |     [\[1\]](api/                  |
| -   [counts()                     | languages/cpp_api.html#_CPPv4I0EN |
|     (cudaq.ObserveResult          | 5cudaq10product_opplE6sum_opI1TER |
|                                   | K15scalar_operatorRK6sum_opI1TE), |
| method)](api/languages/python_api |     [\[2\]](api/langu             |
| .html#cudaq.ObserveResult.counts) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [create() (in module          | q10product_opplE6sum_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|    cudaq.boson)](api/languages/py |     [\[3\]](api/                  |
| thon_api.html#cudaq.boson.create) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(in module               | 5cudaq10product_opplE6sum_opI1TER |
|         c                         | K15scalar_operatorRR6sum_opI1TE), |
| udaq.fermion)](api/languages/pyth |     [\[4\]](api/langu             |
| on_api.html#cudaq.fermion.create) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [csr_spmatrix (C++            | q10product_opplE6sum_opI1TERR15sc |
|     type)](api/languages/c        | alar_operatorRK10product_opI1TE), |
| pp_api.html#_CPPv412csr_spmatrix) |     [\[5\]](api/                  |
| -   cudaq                         | languages/cpp_api.html#_CPPv4I0EN |
|     -   [module](api/langua       | 5cudaq10product_opplE6sum_opI1TER |
| ges/python_api.html#module-cudaq) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq (C++                   |     [\[6\]](api/langu             |
|     type)](api/lan                | ages/cpp_api.html#_CPPv4I0EN5cuda |
| guages/cpp_api.html#_CPPv45cudaq) | q10product_opplE6sum_opI1TERR15sc |
| -   [cudaq.apply_noise() (in      | alar_operatorRR10product_opI1TE), |
|     module                        |     [\[7\]](api/                  |
|     cudaq)](api/languages/python_ | languages/cpp_api.html#_CPPv4I0EN |
| api.html#cudaq.cudaq.apply_noise) | 5cudaq10product_opplE6sum_opI1TER |
| -   cudaq.boson                   | R15scalar_operatorRR6sum_opI1TE), |
|     -   [module](api/languages/py |     [\[8\]](api/languages/cpp_a   |
| thon_api.html#module-cudaq.boson) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.fermion                 | opplERK10product_opI9HandlerTyE), |
|                                   |     [\[9\]](api/language          |
|   -   [module](api/languages/pyth | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| on_api.html#module-cudaq.fermion) | roduct_opplERK15scalar_operator), |
| -   cudaq.operators.custom        |     [\[10\]](api/languages/       |
|     -   [mo                       | cpp_api.html#_CPPv4NKR5cudaq10pro |
| dule](api/languages/python_api.ht | duct_opplERK6sum_opI9HandlerTyE), |
| ml#module-cudaq.operators.custom) |     [\[11\]](api/languages/cpp_a  |
| -   [cudaq.                       | pi.html#_CPPv4NKR5cudaq10product_ |
| ptsbe.ConditionalSamplingStrategy | opplERR10product_opI9HandlerTyE), |
|     (class in                     |     [\[12\]](api/language         |
|     cudaq.operators.cus           | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| tom)](api/languages/python_api.ht | roduct_opplERR15scalar_operator), |
| ml#cudaq.operators.custom.cudaq.p |     [\[13\]](api/languages/       |
| tsbe.ConditionalSamplingStrategy) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq                        | duct_opplERR6sum_opI9HandlerTyE), |
| .ptsbe.ExhaustiveSamplingStrategy |     [\[                           |
|     (class in                     | 14\]](api/languages/cpp_api.html# |
|     cudaq.operators.cu            | _CPPv4NKR5cudaq10product_opplEv), |
| stom)](api/languages/python_api.h |     [\[15\]](api/languages/cpp_   |
| tml#cudaq.operators.custom.cudaq. | api.html#_CPPv4NO5cudaq10product_ |
| ptsbe.ExhaustiveSamplingStrategy) | opplERK10product_opI9HandlerTyE), |
| -   [cudaq.ptsbe.KrausSelection   |     [\[16\]](api/languag          |
|     (class in                     | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     cudaq.                        | roduct_opplERK15scalar_operator), |
| operators.custom)](api/languages/ |     [\[17\]](api/languages        |
| python_api.html#cudaq.operators.c | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ustom.cudaq.ptsbe.KrausSelection) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cudaq.ptsbe.KrausTrajectory  |     [\[18\]](api/languages/cpp_   |
|     (class in                     | api.html#_CPPv4NO5cudaq10product_ |
|     cudaq.o                       | opplERR10product_opI9HandlerTyE), |
| perators.custom)](api/languages/p |     [\[19\]](api/languag          |
| ython_api.html#cudaq.operators.cu | es/cpp_api.html#_CPPv4NO5cudaq10p |
| stom.cudaq.ptsbe.KrausTrajectory) | roduct_opplERR15scalar_operator), |
| -   [cu                           |     [\[20\]](api/languages        |
| daq.ptsbe.OrderedSamplingStrategy | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (class in                     | duct_opplERR6sum_opI9HandlerTyE), |
|     cudaq.operators               |     [                             |
| .custom)](api/languages/python_ap | \[21\]](api/languages/cpp_api.htm |
| i.html#cudaq.operators.custom.cud | l#_CPPv4NO5cudaq10product_opplEv) |
| aq.ptsbe.OrderedSamplingStrategy) | -   [cudaq::product_op::operator- |
| -   [cudaq.pt                     |     (C++                          |
| sbe.ProbabilisticSamplingStrategy |     function)](api/langu          |
|     (class in                     | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cudaq.operators.custo         | q10product_opmiE6sum_opI1TERK15sc |
| m)](api/languages/python_api.html | alar_operatorRK10product_opI1TE), |
| #cudaq.operators.custom.cudaq.pts |     [\[1\]](api/                  |
| be.ProbabilisticSamplingStrategy) | languages/cpp_api.html#_CPPv4I0EN |
| -                                 | 5cudaq10product_opmiE6sum_opI1TER |
|   [cudaq.ptsbe.PTSBEExecutionData | K15scalar_operatorRK6sum_opI1TE), |
|     (class in                     |     [\[2\]](api/langu             |
|     cudaq.oper                    | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ators.custom)](api/languages/pyth | q10product_opmiE6sum_opI1TERK15sc |
| on_api.html#cudaq.operators.custo | alar_operatorRR10product_opI1TE), |
| m.cudaq.ptsbe.PTSBEExecutionData) |     [\[3\]](api/                  |
| -                                 | languages/cpp_api.html#_CPPv4I0EN |
|    [cudaq.ptsbe.PTSBESampleResult | 5cudaq10product_opmiE6sum_opI1TER |
|     (class in                     | K15scalar_operatorRR6sum_opI1TE), |
|     cudaq.ope                     |     [\[4\]](api/langu             |
| rators.custom)](api/languages/pyt | ages/cpp_api.html#_CPPv4I0EN5cuda |
| hon_api.html#cudaq.operators.cust | q10product_opmiE6sum_opI1TERR15sc |
| om.cudaq.ptsbe.PTSBESampleResult) | alar_operatorRK10product_opI1TE), |
| -                                 |     [\[5\]](api/                  |
|  [cudaq.ptsbe.PTSSamplingStrategy | languages/cpp_api.html#_CPPv4I0EN |
|     (class in                     | 5cudaq10product_opmiE6sum_opI1TER |
|     cudaq.opera                   | R15scalar_operatorRK6sum_opI1TE), |
| tors.custom)](api/languages/pytho |     [\[6\]](api/langu             |
| n_api.html#cudaq.operators.custom | ages/cpp_api.html#_CPPv4I0EN5cuda |
| .cudaq.ptsbe.PTSSamplingStrategy) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq.ptsbe.sample() (in     | alar_operatorRR10product_opI1TE), |
|     module                        |     [\[7\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|   cudaq.operators.custom)](api/la | 5cudaq10product_opmiE6sum_opI1TER |
| nguages/python_api.html#cudaq.ope | R15scalar_operatorRR6sum_opI1TE), |
| rators.custom.cudaq.ptsbe.sample) |     [\[8\]](api/languages/cpp_a   |
| -   [cudaq.ptsbe.sample_async()   | pi.html#_CPPv4NKR5cudaq10product_ |
|     (in module                    | opmiERK10product_opI9HandlerTyE), |
|     cuda                          |     [\[9\]](api/language          |
| q.operators.custom)](api/language | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| s/python_api.html#cudaq.operators | roduct_opmiERK15scalar_operator), |
| .custom.cudaq.ptsbe.sample_async) |     [\[10\]](api/languages/       |
| -   [c                            | cpp_api.html#_CPPv4NKR5cudaq10pro |
| udaq.ptsbe.ShotAllocationStrategy | duct_opmiERK6sum_opI9HandlerTyE), |
|     (class in                     |     [\[11\]](api/languages/cpp_a  |
|     cudaq.operator                | pi.html#_CPPv4NKR5cudaq10product_ |
| s.custom)](api/languages/python_a | opmiERR10product_opI9HandlerTyE), |
| pi.html#cudaq.operators.custom.cu |     [\[12\]](api/language         |
| daq.ptsbe.ShotAllocationStrategy) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq.                       | roduct_opmiERR15scalar_operator), |
| ptsbe.ShotAllocationStrategy.Type |     [\[13\]](api/languages/       |
|     (class in                     | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     cudaq.operators.cus           | duct_opmiERR6sum_opI9HandlerTyE), |
| tom)](api/languages/python_api.ht |     [\[                           |
| ml#cudaq.operators.custom.cudaq.p | 14\]](api/languages/cpp_api.html# |
| tsbe.ShotAllocationStrategy.Type) | _CPPv4NKR5cudaq10product_opmiEv), |
| -   [cudaq.ptsbe.TraceInstruction |     [\[15\]](api/languages/cpp_   |
|     (class in                     | api.html#_CPPv4NO5cudaq10product_ |
|     cudaq.op                      | opmiERK10product_opI9HandlerTyE), |
| erators.custom)](api/languages/py |     [\[16\]](api/languag          |
| thon_api.html#cudaq.operators.cus | es/cpp_api.html#_CPPv4NO5cudaq10p |
| tom.cudaq.ptsbe.TraceInstruction) | roduct_opmiERK15scalar_operator), |
| -                                 |     [\[17\]](api/languages        |
| [cudaq.ptsbe.TraceInstructionType | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (class in                     | duct_opmiERK6sum_opI9HandlerTyE), |
|     cudaq.operat                  |     [\[18\]](api/languages/cpp_   |
| ors.custom)](api/languages/python | api.html#_CPPv4NO5cudaq10product_ |
| _api.html#cudaq.operators.custom. | opmiERR10product_opI9HandlerTyE), |
| cudaq.ptsbe.TraceInstructionType) |     [\[19\]](api/languag          |
| -   cudaq.spin                    | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     -   [module](api/languages/p  | roduct_opmiERR15scalar_operator), |
| ython_api.html#module-cudaq.spin) |     [\[20\]](api/languages        |
| -   [cudaq::amplitude_damping     | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERR6sum_opI9HandlerTyE), |
|     cla                           |     [                             |
| ss)](api/languages/cpp_api.html#_ | \[21\]](api/languages/cpp_api.htm |
| CPPv4N5cudaq17amplitude_dampingE) | l#_CPPv4NO5cudaq10product_opmiEv) |
| -                                 | -   [cudaq::product_op::operator/ |
| [cudaq::amplitude_damping_channel |     (C++                          |
|     (C++                          |     function)](api/language       |
|     class)](api                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| /languages/cpp_api.html#_CPPv4N5c | roduct_opdvERK15scalar_operator), |
| udaq25amplitude_damping_channelE) |     [\[1\]](api/language          |
| -   [cudaq::amplitud              | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| e_damping_channel::num_parameters | roduct_opdvERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languag           |
|     member)](api/languages/cpp_a  | es/cpp_api.html#_CPPv4NO5cudaq10p |
| pi.html#_CPPv4N5cudaq25amplitude_ | roduct_opdvERK15scalar_operator), |
| damping_channel14num_parametersE) |     [\[3\]](api/langua            |
| -   [cudaq::ampli                 | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| tude_damping_channel::num_targets | product_opdvERR15scalar_operator) |
|     (C++                          | -                                 |
|     member)](api/languages/cp     |    [cudaq::product_op::operator/= |
| p_api.html#_CPPv4N5cudaq25amplitu |     (C++                          |
| de_damping_channel11num_targetsE) |     function)](api/langu          |
| -   [cudaq::AnalogRemoteRESTQPU   | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     (C++                          | product_opdVERK15scalar_operator) |
|     class                         | -   [cudaq::product_op::operator= |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) |     function)](api/la             |
| -   [cudaq::apply_noise (C++      | nguages/cpp_api.html#_CPPv4I0_NSt |
|     function)](api/               | 11enable_if_tIXaantNSt7is_sameI1T |
| languages/cpp_api.html#_CPPv4I0Dp | 9HandlerTyE5valueENSt16is_constru |
| EN5cudaq11apply_noiseEvDpRR4Args) | ctibleI9HandlerTy1TE5valueEEbEEEN |
| -   [cudaq::async_result (C++     | 5cudaq10product_opaSER10product_o |
|     c                             | pI9HandlerTyERK10product_opI1TE), |
| lass)](api/languages/cpp_api.html |     [\[1\]](api/languages/cpp     |
| #_CPPv4I0EN5cudaq12async_resultE) | _api.html#_CPPv4N5cudaq10product_ |
| -   [cudaq::async_result::get     | opaSERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     functi                        | p_api.html#_CPPv4N5cudaq10product |
| on)](api/languages/cpp_api.html#_ | _opaSERR10product_opI9HandlerTyE) |
| CPPv4N5cudaq12async_result3getEv) | -                                 |
| -   [cudaq::async_sample_result   |    [cudaq::product_op::operator== |
|     (C++                          |     (C++                          |
|     type                          |     function)](api/languages/cpp  |
| )](api/languages/cpp_api.html#_CP | _api.html#_CPPv4NK5cudaq10product |
| Pv4N5cudaq19async_sample_resultE) | _opeqERK10product_opI9HandlerTyE) |
| -   [cudaq::BaseRemoteRESTQPU     | -                                 |
|     (C++                          |  [cudaq::product_op::operator\[\] |
|     cla                           |     (C++                          |
| ss)](api/languages/cpp_api.html#_ |     function)](ap                 |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | i/languages/cpp_api.html#_CPPv4NK |
| -                                 | 5cudaq10product_opixENSt6size_tE) |
|    [cudaq::BaseRemoteSimulatorQPU | -                                 |
|     (C++                          |    [cudaq::product_op::product_op |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languages/c    |
| N5cudaq22BaseRemoteSimulatorQPUE) | pp_api.html#_CPPv4I0_NSt11enable_ |
| -   [cudaq::bit_flip_channel (C++ | if_tIXaaNSt7is_sameI9HandlerTy14m |
|     cl                            | atrix_handlerE5valueEaantNSt7is_s |
| ass)](api/languages/cpp_api.html# | ameI1T9HandlerTyE5valueENSt16is_c |
| _CPPv4N5cudaq16bit_flip_channelE) | onstructibleI9HandlerTy1TE5valueE |
| -   [cudaq:                       | EbEEEN5cudaq10product_op10product |
| :bit_flip_channel::num_parameters | _opERK10product_opI1TERKN14matrix |
|     (C++                          | _handler20commutation_behaviorE), |
|     member)](api/langua           |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq16b |  [\[1\]](api/languages/cpp_api.ht |
| it_flip_channel14num_parametersE) | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| -   [cud                          | tNSt7is_sameI1T9HandlerTyE5valueE |
| aq::bit_flip_channel::num_targets | NSt16is_constructibleI9HandlerTy1 |
|     (C++                          | TE5valueEEbEEEN5cudaq10product_op |
|     member)](api/lan              | 10product_opERK10product_opI1TE), |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 16bit_flip_channel11num_targetsE) |   [\[2\]](api/languages/cpp_api.h |
| -   [cudaq::boson_handler (C++    | tml#_CPPv4N5cudaq10product_op10pr |
|                                   | oduct_opENSt6size_tENSt6size_tE), |
|  class)](api/languages/cpp_api.ht |     [\[3\]](api/languages/cp      |
| ml#_CPPv4N5cudaq13boson_handlerE) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq::boson_op (C++         | _op10product_opENSt7complexIdEE), |
|     type)](api/languages/cpp_     |     [\[4\]](api/l                 |
| api.html#_CPPv4N5cudaq8boson_opE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::boson_op_term (C++    | aq10product_op10product_opERK10pr |
|                                   | oduct_opI9HandlerTyENSt6size_tE), |
|   type)](api/languages/cpp_api.ht |     [\[5\]](api/l                 |
| ml#_CPPv4N5cudaq13boson_op_termE) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::CodeGenConfig (C++    | aq10product_op10product_opERR10pr |
|                                   | oduct_opI9HandlerTyENSt6size_tE), |
| struct)](api/languages/cpp_api.ht |     [\[6\]](api/languages         |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | /cpp_api.html#_CPPv4N5cudaq10prod |
| -   [cudaq::commutation_relations | uct_op10product_opERR9HandlerTy), |
|     (C++                          |     [\[7\]](ap                    |
|     struct)]                      | i/languages/cpp_api.html#_CPPv4N5 |
| (api/languages/cpp_api.html#_CPPv | cudaq10product_op10product_opEd), |
| 4N5cudaq21commutation_relationsE) |     [\[8\]](a                     |
| -   [cudaq::complex (C++          | pi/languages/cpp_api.html#_CPPv4N |
|     type)](api/languages/cpp      | 5cudaq10product_op10product_opEv) |
| _api.html#_CPPv4N5cudaq7complexE) | -   [cuda                         |
| -   [cudaq::complex_matrix (C++   | q::product_op::to_diagonal_matrix |
|                                   |     (C++                          |
| class)](api/languages/cpp_api.htm |     function)](api/               |
| l#_CPPv4N5cudaq14complex_matrixE) | languages/cpp_api.html#_CPPv4NK5c |
| -                                 | udaq10product_op18to_diagonal_mat |
|   [cudaq::complex_matrix::adjoint | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](a                  | apINSt6stringENSt7complexIdEEEEb) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::product_op::to_matrix |
| 5cudaq14complex_matrix7adjointEv) |     (C++                          |
| -   [cudaq::                      |     funct                         |
| complex_matrix::diagonal_elements | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NK5cudaq10product_op9to_mat |
|     function)](api/languages      | rixENSt13unordered_mapINSt6size_t |
| /cpp_api.html#_CPPv4NK5cudaq14com | ENSt7int64_tEEERKNSt13unordered_m |
| plex_matrix17diagonal_elementsEi) | apINSt6stringENSt7complexIdEEEEb) |
| -   [cudaq::complex_matrix::dump  | -   [cu                           |
|     (C++                          | daq::product_op::to_sparse_matrix |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4NK5cudaq14co |     function)](ap                 |
| mplex_matrix4dumpERNSt7ostreamE), | i/languages/cpp_api.html#_CPPv4NK |
|     [\[1\]]                       | 5cudaq10product_op16to_sparse_mat |
| (api/languages/cpp_api.html#_CPPv | rixENSt13unordered_mapINSt6size_t |
| 4NK5cudaq14complex_matrix4dumpEv) | ENSt7int64_tEEERKNSt13unordered_m |
| -   [c                            | apINSt6stringENSt7complexIdEEEEb) |
| udaq::complex_matrix::eigenvalues | -   [cudaq::product_op::to_string |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function)](                   |
| guages/cpp_api.html#_CPPv4NK5cuda | api/languages/cpp_api.html#_CPPv4 |
| q14complex_matrix11eigenvaluesEv) | NK5cudaq10product_op9to_stringEv) |
| -   [cu                           | -                                 |
| daq::complex_matrix::eigenvectors |  [cudaq::product_op::\~product_op |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     fu                            |
| uages/cpp_api.html#_CPPv4NK5cudaq | nction)](api/languages/cpp_api.ht |
| 14complex_matrix12eigenvectorsEv) | ml#_CPPv4N5cudaq10product_opD0Ev) |
| -   [c                            | -   [cudaq::p                     |
| udaq::complex_matrix::exponential | tsbe::ConditionalSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/la             |     class)](api/languag           |
| nguages/cpp_api.html#_CPPv4N5cuda | es/cpp_api.html#_CPPv4N5cudaq5pts |
| q14complex_matrix11exponentialEv) | be27ConditionalSamplingStrategyE) |
| -                                 | -   [cuda                         |
|  [cudaq::complex_matrix::identity | q::ptsbe::ConditionalSamplingStra |
|     (C++                          | tegy::ConditionalSamplingStrategy |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq14comp |     function)](api/lang           |
| lex_matrix8identityEKNSt6size_tE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -                                 | ptsbe27ConditionalSamplingStrateg |
| [cudaq::complex_matrix::kronecker | y27ConditionalSamplingStrategyE19 |
|     (C++                          | TrajectoryPredicateNSt8uint64_tE) |
|     function)](api/lang           | -                                 |
| uages/cpp_api.html#_CPPv4I00EN5cu |    [cudaq::ptsbe::ConditionalSamp |
| daq14complex_matrix9kroneckerE14c | lingStrategy::TrajectoryPredicate |
| omplex_matrix8Iterable8Iterable), |     (C++                          |
|     [\[1\]](api/l                 |     type)]                        |
| anguages/cpp_api.html#_CPPv4N5cud | (api/languages/cpp_api.html#_CPPv |
| aq14complex_matrix9kroneckerERK14 | 4N5cudaq5ptsbe27ConditionalSampli |
| complex_matrixRK14complex_matrix) | ngStrategy19TrajectoryPredicateE) |
| -   [cudaq::c                     | -   [cudaq::                      |
| omplex_matrix::minimal_eigenvalue | ptsbe::ExhaustiveSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/languages/     |     class)](api/langua            |
| cpp_api.html#_CPPv4NK5cudaq14comp | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| lex_matrix18minimal_eigenvalueEv) | sbe26ExhaustiveSamplingStrategyE) |
| -   [                             | -   [cuda                         |
| cudaq::complex_matrix::operator() | q::ptsbe::OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/languages/cpp  |     class)](api/lan               |
| _api.html#_CPPv4N5cudaq14complex_ | guages/cpp_api.html#_CPPv4N5cudaq |
| matrixclENSt6size_tENSt6size_tE), | 5ptsbe23OrderedSamplingStrategyE) |
|     [\[1\]](api/languages/cpp     | -   [cudaq::pts                   |
| _api.html#_CPPv4NK5cudaq14complex | be::ProbabilisticSamplingStrategy |
| _matrixclENSt6size_tENSt6size_tE) |     (C++                          |
| -   [                             |     class)](api/languages         |
| cudaq::complex_matrix::operator\* | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     (C++                          | 29ProbabilisticSamplingStrategyE) |
|     function)](api/langua         | -   [cudaq::p                     |
| ges/cpp_api.html#_CPPv4N5cudaq14c | tsbe::ProbabilisticSamplingStrate |
| omplex_matrixmlEN14complex_matrix | gy::ProbabilisticSamplingStrategy |
| 10value_typeERK14complex_matrix), |     (C++                          |
|     [\[1\]                        |     fu                            |
| ](api/languages/cpp_api.html#_CPP | nction)](api/languages/cpp_api.ht |
| v4N5cudaq14complex_matrixmlERK14c | ml#_CPPv4N5cudaq5ptsbe29Probabili |
| omplex_matrixRK14complex_matrix), | sticSamplingStrategy29Probabilist |
|                                   | icSamplingStrategyENSt8uint64_tE) |
|  [\[2\]](api/languages/cpp_api.ht | -                                 |
| ml#_CPPv4N5cudaq14complex_matrixm | [cudaq::ptsbe::PTSBEExecutionData |
| lERK14complex_matrixRKNSt6vectorI |     (C++                          |
| N14complex_matrix10value_typeEEE) |     struct)](ap                   |
| -                                 | i/languages/cpp_api.html#_CPPv4N5 |
| [cudaq::complex_matrix::operator+ | cudaq5ptsbe18PTSBEExecutionDataE) |
|     (C++                          | -   [cudaq::ptsbe::PTSBE          |
|     function                      | ExecutionData::count_instructions |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq14complex_matrixplERK14 |     function)](api/l              |
| complex_matrixRK14complex_matrix) | anguages/cpp_api.html#_CPPv4NK5cu |
| -                                 | daq5ptsbe18PTSBEExecutionData18co |
| [cudaq::complex_matrix::operator- | unt_instructionsE20TraceInstructi |
|     (C++                          | onTypeNSt8optionalINSt6stringEEE) |
|     function                      | -   [cudaq::ptsbe::P              |
| )](api/languages/cpp_api.html#_CP | TSBEExecutionData::get_trajectory |
| Pv4N5cudaq14complex_matrixmiERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |     function                      |
| -   [cu                           | )](api/languages/cpp_api.html#_CP |
| daq::complex_matrix::operator\[\] | Pv4NK5cudaq5ptsbe18PTSBEExecution |
|     (C++                          | Data14get_trajectoryENSt6size_tE) |
|                                   | -   [cudaq::ptsbe:                |
|  function)](api/languages/cpp_api | :PTSBEExecutionData::instructions |
| .html#_CPPv4N5cudaq14complex_matr |     (C++                          |
| ixixERKNSt6vectorINSt6size_tEEE), |     member)](api/languages/cp     |
|     [\[1\]](api/languages/cpp_api | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| .html#_CPPv4NK5cudaq14complex_mat | TSBEExecutionData12instructionsE) |
| rixixERKNSt6vectorINSt6size_tEEE) | -   [cudaq::ptsbe:                |
| -   [cudaq::complex_matrix::power | :PTSBEExecutionData::trajectories |
|     (C++                          |     (C++                          |
|     function)]                    |     member)](api/languages/cp     |
| (api/languages/cpp_api.html#_CPPv | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| 4N5cudaq14complex_matrix5powerEi) | TSBEExecutionData12trajectoriesE) |
| -                                 | -   [cudaq::ptsbe::PTSBEOptions   |
|  [cudaq::complex_matrix::set_zero |     (C++                          |
|     (C++                          |     struc                         |
|     function)](ap                 | t)](api/languages/cpp_api.html#_C |
| i/languages/cpp_api.html#_CPPv4N5 | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| cudaq14complex_matrix8set_zeroEv) | -   [cudaq::ptsb                  |
| -                                 | e::PTSBEOptions::max_trajectories |
| [cudaq::complex_matrix::to_string |     (C++                          |
|     (C++                          |     member)](api/languages/       |
|     function)](api/               | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
| languages/cpp_api.html#_CPPv4NK5c | 2PTSBEOptions16max_trajectoriesE) |
| udaq14complex_matrix9to_stringEv) | -   [cudaq::ptsbe::PT             |
| -   [                             | SBEOptions::return_execution_data |
| cudaq::complex_matrix::value_type |     (C++                          |
|     (C++                          |     member)](api/languages/cpp_a  |
|     type)](api/                   | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| languages/cpp_api.html#_CPPv4N5cu | EOptions21return_execution_dataE) |
| daq14complex_matrix10value_typeE) | -   [cudaq::pts                   |
| -   [cudaq::contrib (C++          | be::PTSBEOptions::shot_allocation |
|     type)](api/languages/cpp      |     (C++                          |
| _api.html#_CPPv4N5cudaq7contribE) |     member)](api/languages        |
| -   [cudaq::contrib::draw (C++    | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     function)                     | 12PTSBEOptions15shot_allocationE) |
| ](api/languages/cpp_api.html#_CPP | -   [cud                          |
| v4I0DpEN5cudaq7contrib4drawENSt6s | aq::ptsbe::PTSBEOptions::strategy |
| tringERR13QuantumKernelDpRR4Args) |     (C++                          |
| -                                 |     member)](api/l                |
| [cudaq::contrib::get_unitary_cmat | anguages/cpp_api.html#_CPPv4N5cud |
|     (C++                          | aq5ptsbe12PTSBEOptions8strategyE) |
|     function)](api/languages/cp   | -   [                             |
| p_api.html#_CPPv4I0DpEN5cudaq7con | cudaq::ptsbe::PTSSamplingStrategy |
| trib16get_unitary_cmatE14complex_ |     (C++                          |
| matrixRR13QuantumKernelDpRR4Args) |     class)](api                   |
| -   [cudaq::CusvState (C++        | /languages/cpp_api.html#_CPPv4N5c |
|                                   | udaq5ptsbe19PTSSamplingStrategyE) |
|    class)](api/languages/cpp_api. | -   [cudaq::                      |
| html#_CPPv4I0EN5cudaq9CusvStateE) | ptsbe::PTSSamplingStrategy::clone |
| -   [cudaq::depolarization1 (C++  |     (C++                          |
|     c                             |     function)](api/languag        |
| lass)](api/languages/cpp_api.html | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| #_CPPv4N5cudaq15depolarization1E) | sbe19PTSSamplingStrategy5cloneEv) |
| -   [cudaq::depolarization2 (C++  | -   [cudaq::ptsbe::PTSSampl       |
|     c                             | ingStrategy::generateTrajectories |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15depolarization2E) |     function)](api/lang           |
| -   [cudaq:                       | uages/cpp_api.html#_CPPv4NK5cudaq |
| :depolarization2::depolarization2 | 5ptsbe19PTSSamplingStrategy20gene |
|     (C++                          | rateTrajectoriesENSt4spanIKN5cuda |
|     function)](api/languages/cp   | q15KrausTrajectoryEEENSt6size_tE) |
| p_api.html#_CPPv4N5cudaq15depolar | -   [cudaq:                       |
| ization215depolarization2EK4real) | :ptsbe::PTSSamplingStrategy::name |
| -   [cudaq                        |     (C++                          |
| ::depolarization2::num_parameters |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4NK5cudaq5p |
|     member)](api/langu            | tsbe19PTSSamplingStrategy4nameEv) |
| ages/cpp_api.html#_CPPv4N5cudaq15 | -   [cudaq::ptsbe::sample (C++    |
| depolarization214num_parametersE) |                                   |
| -   [cu                           |   function)](api/languages/cpp_ap |
| daq::depolarization2::num_targets | i.html#_CPPv4I0DpEN5cudaq5ptsbe6s |
|     (C++                          | ampleE13sample_resultRK14sample_o |
|     member)](api/la               | ptionsRR13QuantumKernelDpRR4Args) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::ptsbe::sample_async   |
| q15depolarization211num_targetsE) |     (C++                          |
| -                                 |                                   |
|    [cudaq::depolarization_channel |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4I0DpEN5cudaq5ptsbe1 |
|     class)](                      | 2sample_asyncEN5cudaq19async_samp |
| api/languages/cpp_api.html#_CPPv4 | le_resultERK14sample_optionsRR13Q |
| N5cudaq22depolarization_channelE) | uantumKernelDpRR4ArgsNSt6size_tE) |
| -   [cudaq::depol                 | -   [cudaq::ptsbe::sample_options |
| arization_channel::num_parameters |     (C++                          |
|     (C++                          |     struct)                       |
|     member)](api/languages/cp     | ](api/languages/cpp_api.html#_CPP |
| p_api.html#_CPPv4N5cudaq22depolar | v4N5cudaq5ptsbe14sample_optionsE) |
| ization_channel14num_parametersE) | -   [cu                           |
| -   [cudaq::de                    | daq::ptsbe::sample_options::noise |
| polarization_channel::num_targets |     (C++                          |
|     (C++                          |     member)](api/                 |
|     member)](api/languages        | languages/cpp_api.html#_CPPv4N5cu |
| /cpp_api.html#_CPPv4N5cudaq22depo | daq5ptsbe14sample_options5noiseE) |
| larization_channel11num_targetsE) | -   [cu                           |
| -   [cudaq::details (C++          | daq::ptsbe::sample_options::ptsbe |
|     type)](api/languages/cpp      |     (C++                          |
| _api.html#_CPPv4N5cudaq7detailsE) |     member)](api/                 |
| -   [cudaq::details::future (C++  | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq5ptsbe14sample_options5ptsbeE) |
|  class)](api/languages/cpp_api.ht | -   [cu                           |
| ml#_CPPv4N5cudaq7details6futureE) | daq::ptsbe::sample_options::shots |
| -                                 |     (C++                          |
|   [cudaq::details::future::future |     member)](api/                 |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     functio                       | daq5ptsbe14sample_options5shotsE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::ptsbe::sample_result  |
| PPv4N5cudaq7details6future6future |     (C++                          |
| ERNSt6vectorI3JobEERNSt6stringERN |     class                         |
| St3mapINSt6stringENSt6stringEEE), | )](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/lang              | Pv4N5cudaq5ptsbe13sample_resultE) |
| uages/cpp_api.html#_CPPv4N5cudaq7 | -   [cudaq::pts                   |
| details6future6futureERR6future), | be::sample_result::execution_data |
|     [\[2\]]                       |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/languages/c    |
| 4N5cudaq7details6future6futureEv) | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| -   [cu                           | 3sample_result14execution_dataEv) |
| daq::details::kernel_builder_base | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::has_execution_data |
|     class)](api/l                 |     (C++                          |
| anguages/cpp_api.html#_CPPv4N5cud |                                   |
| aq7details19kernel_builder_baseE) |    function)](api/languages/cpp_a |
| -   [cudaq::details::             | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| kernel_builder_base::operator\<\< | ple_result18has_execution_dataEv) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     function)](api/langua         | sample_result::set_execution_data |
| ges/cpp_api.html#_CPPv4N5cudaq7de |     (C++                          |
| tails19kernel_builder_baselsERNSt |     function)](api/               |
| 7ostreamERK19kernel_builder_base) | languages/cpp_api.html#_CPPv4N5cu |
| -   [                             | daq5ptsbe13sample_result18set_exe |
| cudaq::details::KernelBuilderType | cution_dataE18PTSBEExecutionData) |
|     (C++                          | -   [cud                          |
|     class)](api                   | aq::ptsbe::ShotAllocationStrategy |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq7details17KernelBuilderTypeE) |     struct)](api/la               |
| -   [cudaq::d                     | nguages/cpp_api.html#_CPPv4N5cuda |
| etails::KernelBuilderType::create | q5ptsbe22ShotAllocationStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::Shot           |
|     function)                     | AllocationStrategy::bias_strength |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq7details17KernelBuilderT |                                   |
| ype6createEPN4mlir11MLIRContextE) |    member)](api/languages/cpp_api |
| -   [cudaq::details::Ker          | .html#_CPPv4N5cudaq5ptsbe22ShotAl |
| nelBuilderType::KernelBuilderType | locationStrategy13bias_strengthE) |
|     (C++                          | -   [cudaq::pt                    |
|     function)](api/lang           | sbe::ShotAllocationStrategy::seed |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     (C++                          |
| details17KernelBuilderType17Kerne |     member)](api/languag          |
| lBuilderTypeERRNSt8functionIFN4ml | es/cpp_api.html#_CPPv4N5cudaq5pts |
| ir4TypeEPN4mlir11MLIRContextEEEE) | be22ShotAllocationStrategy4seedE) |
| -   [cudaq::diag_matrix_callback  | -   [cudaq::ptsbe::ShotAllocatio  |
|     (C++                          | nStrategy::ShotAllocationStrategy |
|     class)                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languages/cp   |
| v4N5cudaq20diag_matrix_callbackE) | p_api.html#_CPPv4N5cudaq5ptsbe22S |
| -   [cudaq::dyn (C++              | hotAllocationStrategy22ShotAlloca |
|     member)](api/languages        | tionStrategyE4TypedNSt8uint64_tE) |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | -   [cudaq::pt                    |
| -   [cudaq::ExecutionContext (C++ | sbe::ShotAllocationStrategy::Type |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     enum)](api/languag            |
| _CPPv4N5cudaq16ExecutionContextE) | es/cpp_api.html#_CPPv4N5cudaq5pts |
| -   [cudaq                        | be22ShotAllocationStrategy4TypeE) |
| ::ExecutionContext::amplitudeMaps | -   [cudaq::pt                    |
|     (C++                          | sbe::ShotAllocationStrategy::type |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     member)](api/languag          |
| ExecutionContext13amplitudeMapsE) | es/cpp_api.html#_CPPv4N5cudaq5pts |
| -   [c                            | be22ShotAllocationStrategy4typeE) |
| udaq::ExecutionContext::asyncExec | -   [cudaq::ptsbe::ShotAllocatio  |
|     (C++                          | nStrategy::Type::HIGH_WEIGHT_BIAS |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     enumerato                     |
| daq16ExecutionContext9asyncExecE) | r)](api/languages/cpp_api.html#_C |
| -   [cud                          | PPv4N5cudaq5ptsbe22ShotAllocation |
| aq::ExecutionContext::asyncResult | Strategy4Type16HIGH_WEIGHT_BIASE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAllocati   |
|     member)](api/lan              | onStrategy::Type::LOW_WEIGHT_BIAS |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11asyncResultE) |     enumerat                      |
| -   [cudaq:                       | or)](api/languages/cpp_api.html#_ |
| :ExecutionContext::batchIteration | CPPv4N5cudaq5ptsbe22ShotAllocatio |
|     (C++                          | nStrategy4Type15LOW_WEIGHT_BIASE) |
|     member)](api/langua           | -   [cudaq::ptsbe::ShotAlloc      |
| ges/cpp_api.html#_CPPv4N5cudaq16E | ationStrategy::Type::PROPORTIONAL |
| xecutionContext14batchIterationE) |     (C++                          |
| -   [cudaq::E                     |     enume                         |
| xecutionContext::canHandleObserve | rator)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
|     member)](api/language         | tionStrategy4Type12PROPORTIONALE) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::ptsbe::Shot           |
| cutionContext16canHandleObserveE) | AllocationStrategy::Type::UNIFORM |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::ExecutionContext |                                   |
|     (C++                          |  enumerator)](api/languages/cpp_a |
|     func                          | pi.html#_CPPv4N5cudaq5ptsbe22Shot |
| tion)](api/languages/cpp_api.html | AllocationStrategy4Type7UNIFORME) |
| #_CPPv4N5cudaq16ExecutionContext1 | -                                 |
| 6ExecutionContextERKNSt6stringE), |   [cudaq::ptsbe::TraceInstruction |
|     [\[1\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16Execu |     struct)](                     |
| tionContext16ExecutionContextERKN | api/languages/cpp_api.html#_CPPv4 |
| St6stringENSt6size_tENSt6size_tE) | N5cudaq5ptsbe16TraceInstructionE) |
| -   [cudaq::E                     | -   [cudaq:                       |
| xecutionContext::expectationValue | :ptsbe::TraceInstruction::channel |
|     (C++                          |     (C++                          |
|     member)](api/language         |     member)](api/lang             |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | uages/cpp_api.html#_CPPv4N5cudaq5 |
| cutionContext16expectationValueE) | ptsbe16TraceInstruction7channelE) |
| -   [cudaq::Execu                 | -   [cudaq::                      |
| tionContext::explicitMeasurements | ptsbe::TraceInstruction::controls |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     member)](api/langu            |
| p_api.html#_CPPv4N5cudaq16Executi | ages/cpp_api.html#_CPPv4N5cudaq5p |
| onContext20explicitMeasurementsE) | tsbe16TraceInstruction8controlsE) |
| -   [cuda                         | -   [cud                          |
| q::ExecutionContext::futureResult | aq::ptsbe::TraceInstruction::name |
|     (C++                          |     (C++                          |
|     member)](api/lang             |     member)](api/l                |
| uages/cpp_api.html#_CPPv4N5cudaq1 | anguages/cpp_api.html#_CPPv4N5cud |
| 6ExecutionContext12futureResultE) | aq5ptsbe16TraceInstruction4nameE) |
| -   [cudaq::ExecutionContext      | -   [cudaq                        |
| ::hasConditionalsOnMeasureResults | ::ptsbe::TraceInstruction::params |
|     (C++                          |     (C++                          |
|     mem                           |     member)](api/lan              |
| ber)](api/languages/cpp_api.html# | guages/cpp_api.html#_CPPv4N5cudaq |
| _CPPv4N5cudaq16ExecutionContext31 | 5ptsbe16TraceInstruction6paramsE) |
| hasConditionalsOnMeasureResultsE) | -   [cudaq:                       |
| -   [cudaq::Executi               | :ptsbe::TraceInstruction::targets |
| onContext::invocationResultBuffer |     (C++                          |
|     (C++                          |     member)](api/lang             |
|     member)](api/languages/cpp_   | uages/cpp_api.html#_CPPv4N5cudaq5 |
| api.html#_CPPv4N5cudaq16Execution | ptsbe16TraceInstruction7targetsE) |
| Context22invocationResultBufferE) | -   [cud                          |
| -   [cu                           | aq::ptsbe::TraceInstruction::type |
| daq::ExecutionContext::kernelName |     (C++                          |
|     (C++                          |     member)](api/l                |
|     member)](api/la               | anguages/cpp_api.html#_CPPv4N5cud |
| nguages/cpp_api.html#_CPPv4N5cuda | aq5ptsbe16TraceInstruction4typeE) |
| q16ExecutionContext10kernelNameE) | -   [c                            |
| -   [cud                          | udaq::ptsbe::TraceInstructionType |
| aq::ExecutionContext::kernelTrace |     (C++                          |
|     (C++                          |     enum)](api/                   |
|     member)](api/lan              | languages/cpp_api.html#_CPPv4N5cu |
| guages/cpp_api.html#_CPPv4N5cudaq | daq5ptsbe20TraceInstructionTypeE) |
| 16ExecutionContext11kernelTraceE) | -   [cudaq::                      |
| -   [cudaq:                       | ptsbe::TraceInstructionType::Gate |
| :ExecutionContext::msm_dimensions |     (C++                          |
|     (C++                          |     enumerator)](api/langu        |
|     member)](api/langua           | ages/cpp_api.html#_CPPv4N5cudaq5p |
| ges/cpp_api.html#_CPPv4N5cudaq16E | tsbe20TraceInstructionType4GateE) |
| xecutionContext14msm_dimensionsE) | -   [cudaq::ptsbe::               |
| -   [cudaq::                      | TraceInstructionType::Measurement |
| ExecutionContext::msm_prob_err_id |     (C++                          |
|     (C++                          |                                   |
|     member)](api/languag          |    enumerator)](api/languages/cpp |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| ecutionContext15msm_prob_err_idE) | aceInstructionType11MeasurementE) |
| -   [cudaq::Ex                    | -   [cudaq::p                     |
| ecutionContext::msm_probabilities | tsbe::TraceInstructionType::Noise |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     enumerator)](api/langua       |
| /cpp_api.html#_CPPv4N5cudaq16Exec | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| utionContext17msm_probabilitiesE) | sbe20TraceInstructionType5NoiseE) |
| -                                 | -   [cudaq::QPU (C++              |
|    [cudaq::ExecutionContext::name |     class)](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     member)]                      | -   [cudaq::QPU::beginExecution   |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq16ExecutionContext4nameE) |     function                      |
| -   [cu                           | )](api/languages/cpp_api.html#_CP |
| daq::ExecutionContext::noiseModel | Pv4N5cudaq3QPU14beginExecutionEv) |
|     (C++                          | -   [cuda                         |
|     member)](api/la               | q::QPU::configureExecutionContext |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10noiseModelE) |     funct                         |
| -   [cudaq::Exe                   | ion)](api/languages/cpp_api.html# |
| cutionContext::numberTrajectories | _CPPv4NK5cudaq3QPU25configureExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     member)](api/languages/       | -   [cudaq::QPU::endExecution     |
| cpp_api.html#_CPPv4N5cudaq16Execu |     (C++                          |
| tionContext18numberTrajectoriesE) |     functi                        |
| -   [c                            | on)](api/languages/cpp_api.html#_ |
| udaq::ExecutionContext::optResult | CPPv4N5cudaq3QPU12endExecutionEv) |
|     (C++                          | -   [cudaq::QPU::enqueue (C++     |
|     member)](api/                 |     function)](ap                 |
| languages/cpp_api.html#_CPPv4N5cu | i/languages/cpp_api.html#_CPPv4N5 |
| daq16ExecutionContext9optResultE) | cudaq3QPU7enqueueER11QuantumTask) |
| -   [cudaq::Execu                 | -   [cud                          |
| tionContext::overlapComputeStates | aq::QPU::finalizeExecutionContext |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     func                          |
| p_api.html#_CPPv4N5cudaq16Executi | tion)](api/languages/cpp_api.html |
| onContext20overlapComputeStatesE) | #_CPPv4NK5cudaq3QPU24finalizeExec |
| -   [cudaq                        | utionContextER16ExecutionContext) |
| ::ExecutionContext::overlapResult | -   [cudaq::QPU::getConnectivity  |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)                     |
| ages/cpp_api.html#_CPPv4N5cudaq16 | ](api/languages/cpp_api.html#_CPP |
| ExecutionContext13overlapResultE) | v4N5cudaq3QPU15getConnectivityEv) |
| -                                 | -                                 |
|   [cudaq::ExecutionContext::qpuId | [cudaq::QPU::getExecutionThreadId |
|     (C++                          |     (C++                          |
|     member)](                     |     function)](api/               |
| api/languages/cpp_api.html#_CPPv4 | languages/cpp_api.html#_CPPv4NK5c |
| N5cudaq16ExecutionContext5qpuIdE) | udaq3QPU20getExecutionThreadIdEv) |
| -   [cudaq                        | -   [cudaq::QPU::getNumQubits     |
| ::ExecutionContext::registerNames |     (C++                          |
|     (C++                          |     functi                        |
|     member)](api/langu            | on)](api/languages/cpp_api.html#_ |
| ages/cpp_api.html#_CPPv4N5cudaq16 | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| ExecutionContext13registerNamesE) | -   [                             |
| -   [cu                           | cudaq::QPU::getRemoteCapabilities |
| daq::ExecutionContext::reorderIdx |     (C++                          |
|     (C++                          |     function)](api/l              |
|     member)](api/la               | anguages/cpp_api.html#_CPPv4NK5cu |
| nguages/cpp_api.html#_CPPv4N5cuda | daq3QPU21getRemoteCapabilitiesEv) |
| q16ExecutionContext10reorderIdxE) | -   [cudaq::QPU::isEmulated (C++  |
| -                                 |     func                          |
|  [cudaq::ExecutionContext::result | tion)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     member)](a                    | -   [cudaq::QPU::isSimulator (C++ |
| pi/languages/cpp_api.html#_CPPv4N |     funct                         |
| 5cudaq16ExecutionContext6resultE) | ion)](api/languages/cpp_api.html# |
| -                                 | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|   [cudaq::ExecutionContext::shots | -   [cudaq::QPU::launchKernel     |
|     (C++                          |     (C++                          |
|     member)](                     |     function)](api/               |
| api/languages/cpp_api.html#_CPPv4 | languages/cpp_api.html#_CPPv4N5cu |
| N5cudaq16ExecutionContext5shotsE) | daq3QPU12launchKernelERKNSt6strin |
| -   [cudaq::                      | gE15KernelThunkTypePvNSt8uint64_t |
| ExecutionContext::simulationState | ENSt8uint64_tERKNSt6vectorIPvEE), |
|     (C++                          |                                   |
|     member)](api/languag          |  [\[1\]](api/languages/cpp_api.ht |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | ml#_CPPv4N5cudaq3QPU12launchKerne |
| ecutionContext15simulationStateE) | lERKNSt6stringERKNSt6vectorIPvEE) |
| -                                 | -   [cudaq::QPU::onRandomSeedSet  |
|    [cudaq::ExecutionContext::spin |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     member)]                      | uages/cpp_api.html#_CPPv4N5cudaq3 |
| (api/languages/cpp_api.html#_CPPv | QPU15onRandomSeedSetENSt6size_tE) |
| 4N5cudaq16ExecutionContext4spinE) | -   [cudaq::QPU::QPU (C++         |
| -   [cudaq::                      |     functio                       |
| ExecutionContext::totalIterations | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
|     member)](api/languag          |                                   |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |  [\[1\]](api/languages/cpp_api.ht |
| ecutionContext15totalIterationsE) | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| -   [cudaq::Executio              |     [\[2\]](api/languages/cpp_    |
| nContext::warnedNamedMeasurements | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
|     (C++                          | -   [cudaq::QPU::setId (C++       |
|     member)](api/languages/cpp_a  |     function                      |
| pi.html#_CPPv4N5cudaq16ExecutionC | )](api/languages/cpp_api.html#_CP |
| ontext23warnedNamedMeasurementsE) | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| -   [cudaq::ExecutionResult (C++  | -   [cudaq::QPU::setShots (C++    |
|     st                            |     f                             |
| ruct)](api/languages/cpp_api.html | unction)](api/languages/cpp_api.h |
| #_CPPv4N5cudaq15ExecutionResultE) | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| -   [cud                          | -   [cudaq::                      |
| aq::ExecutionResult::appendResult | QPU::supportsExplicitMeasurements |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api/languag        |
| n)](api/languages/cpp_api.html#_C | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| PPv4N5cudaq15ExecutionResult12app | 28supportsExplicitMeasurementsEv) |
| endResultENSt6stringENSt6size_tE) | -   [cudaq::QPU::\~QPU (C++       |
| -   [cu                           |     function)](api/languages/cp   |
| daq::ExecutionResult::deserialize | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
|     (C++                          | -   [cudaq::QPUState (C++         |
|     function)                     |     class)](api/languages/cpp_    |
| ](api/languages/cpp_api.html#_CPP | api.html#_CPPv4N5cudaq8QPUStateE) |
| v4N5cudaq15ExecutionResult11deser | -   [cudaq::qreg (C++             |
| ializeERNSt6vectorINSt6size_tEEE) |     class)](api/lan               |
| -   [cudaq:                       | guages/cpp_api.html#_CPPv4I_NSt6s |
| :ExecutionResult::ExecutionResult | ize_tE_NSt6size_tEEN5cudaq4qregE) |
|     (C++                          | -   [cudaq::qreg::back (C++       |
|     functio                       |     function)                     |
| n)](api/languages/cpp_api.html#_C | ](api/languages/cpp_api.html#_CPP |
| PPv4N5cudaq15ExecutionResult15Exe | v4N5cudaq4qreg4backENSt6size_tE), |
| cutionResultE16CountsDictionary), |     [\[1\]](api/languages/cpp_ap  |
|     [\[1\]](api/lan               | i.html#_CPPv4N5cudaq4qreg4backEv) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::qreg::begin (C++      |
| 15ExecutionResult15ExecutionResul |                                   |
| tE16CountsDictionaryNSt6stringE), |  function)](api/languages/cpp_api |
|     [\[2\                         | .html#_CPPv4N5cudaq4qreg5beginEv) |
| ]](api/languages/cpp_api.html#_CP | -   [cudaq::qreg::clear (C++      |
| Pv4N5cudaq15ExecutionResult15Exec |                                   |
| utionResultE16CountsDictionaryd), |  function)](api/languages/cpp_api |
|                                   | .html#_CPPv4N5cudaq4qreg5clearEv) |
|    [\[3\]](api/languages/cpp_api. | -   [cudaq::qreg::front (C++      |
| html#_CPPv4N5cudaq15ExecutionResu |     function)]                    |
| lt15ExecutionResultENSt6stringE), | (api/languages/cpp_api.html#_CPPv |
|     [\[4\                         | 4N5cudaq4qreg5frontENSt6size_tE), |
| ]](api/languages/cpp_api.html#_CP |     [\[1\]](api/languages/cpp_api |
| Pv4N5cudaq15ExecutionResult15Exec | .html#_CPPv4N5cudaq4qreg5frontEv) |
| utionResultERK15ExecutionResult), | -   [cudaq::qreg::operator\[\]    |
|     [\[5\]](api/language          |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     functi                        |
| cutionResult15ExecutionResultEd), | on)](api/languages/cpp_api.html#_ |
|     [\[6\]](api/languag           | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | -   [cudaq::qreg::qreg (C++       |
| ecutionResult15ExecutionResultEv) |     function)                     |
| -   [                             | ](api/languages/cpp_api.html#_CPP |
| cudaq::ExecutionResult::operator= | v4N5cudaq4qreg4qregENSt6size_tE), |
|     (C++                          |     [\[1\]](api/languages/cpp_ap  |
|     function)](api/languages/     | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| cpp_api.html#_CPPv4N5cudaq15Execu | -   [cudaq::qreg::size (C++       |
| tionResultaSERK15ExecutionResult) |                                   |
| -   [c                            |  function)](api/languages/cpp_api |
| udaq::ExecutionResult::operator== | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|     (C++                          | -   [cudaq::qreg::slice (C++      |
|     function)](api/languages/c    |     function)](api/langu          |
| pp_api.html#_CPPv4NK5cudaq15Execu | ages/cpp_api.html#_CPPv4N5cudaq4q |
| tionResulteqERK15ExecutionResult) | reg5sliceENSt6size_tENSt6size_tE) |
| -   [cud                          | -   [cudaq::qreg::value_type (C++ |
| aq::ExecutionResult::registerName |                                   |
|     (C++                          | type)](api/languages/cpp_api.html |
|     member)](api/lan              | #_CPPv4N5cudaq4qreg10value_typeE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::qspan (C++            |
| 15ExecutionResult12registerNameE) |     class)](api/lang              |
| -   [cudaq                        | uages/cpp_api.html#_CPPv4I_NSt6si |
| ::ExecutionResult::sequentialData | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|     (C++                          | -   [cudaq::QuakeValue (C++       |
|     member)](api/langu            |     class)](api/languages/cpp_api |
| ages/cpp_api.html#_CPPv4N5cudaq15 | .html#_CPPv4N5cudaq10QuakeValueE) |
| ExecutionResult14sequentialDataE) | -   [cudaq::Q                     |
| -   [                             | uakeValue::canValidateNumElements |
| cudaq::ExecutionResult::serialize |     (C++                          |
|     (C++                          |     function)](api/languages      |
|     function)](api/l              | /cpp_api.html#_CPPv4N5cudaq10Quak |
| anguages/cpp_api.html#_CPPv4NK5cu | eValue22canValidateNumElementsEv) |
| daq15ExecutionResult9serializeEv) | -                                 |
| -   [cudaq::fermion_handler (C++  |  [cudaq::QuakeValue::constantSize |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     function)](api                |
| #_CPPv4N5cudaq15fermion_handlerE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::fermion_op (C++       | udaq10QuakeValue12constantSizeEv) |
|     type)](api/languages/cpp_api  | -   [cudaq::QuakeValue::dump (C++ |
| .html#_CPPv4N5cudaq10fermion_opE) |     function)](api/lan            |
| -   [cudaq::fermion_op_term (C++  | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 10QuakeValue4dumpERNSt7ostreamE), |
| type)](api/languages/cpp_api.html |     [\                            |
| #_CPPv4N5cudaq15fermion_op_termE) | [1\]](api/languages/cpp_api.html# |
| -   [cudaq::FermioniqBaseQPU (C++ | _CPPv4N5cudaq10QuakeValue4dumpEv) |
|     cl                            | -   [cudaq                        |
| ass)](api/languages/cpp_api.html# | ::QuakeValue::getRequiredElements |
| _CPPv4N5cudaq16FermioniqBaseQPUE) |     (C++                          |
| -   [cudaq::get_state (C++        |     function)](api/langua         |
|                                   | ges/cpp_api.html#_CPPv4N5cudaq10Q |
|    function)](api/languages/cpp_a | uakeValue19getRequiredElementsEv) |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | -   [cudaq::QuakeValue::getValue  |
| ateEDaRR13QuantumKernelDpRR4Args) |     (C++                          |
| -   [cudaq::gradient (C++         |     function)]                    |
|     class)](api/languages/cpp_    | (api/languages/cpp_api.html#_CPPv |
| api.html#_CPPv4N5cudaq8gradientE) | 4NK5cudaq10QuakeValue8getValueEv) |
| -   [cudaq::gradient::clone (C++  | -   [cudaq::QuakeValue::inverse   |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     function)                     |
| l#_CPPv4N5cudaq8gradient5cloneEv) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::gradient::compute     | v4NK5cudaq10QuakeValue7inverseEv) |
|     (C++                          | -   [cudaq::QuakeValue::isStdVec  |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     function)                     |
| ient7computeERKNSt6vectorIdEERKNS | ](api/languages/cpp_api.html#_CPP |
| t8functionIFdNSt6vectorIdEEEEEd), | v4N5cudaq10QuakeValue8isStdVecEv) |
|     [\[1\]](ap                    | -                                 |
| i/languages/cpp_api.html#_CPPv4N5 |    [cudaq::QuakeValue::operator\* |
| cudaq8gradient7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |     function)](api                |
| -   [cudaq::gradient::gradient    | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuemlE10QuakeValue), |
|     function)](api/lang           |                                   |
| uages/cpp_api.html#_CPPv4I00EN5cu | [\[1\]](api/languages/cpp_api.htm |
| daq8gradient8gradientER7KernelT), | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
|                                   | -   [cudaq::QuakeValue::operator+ |
|    [\[1\]](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4I00EN5cudaq8gradient8g |     function)](api                |
| radientER7KernelTRR10ArgsMapper), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[2\                         | udaq10QuakeValueplE10QuakeValue), |
| ]](api/languages/cpp_api.html#_CP |     [                             |
| Pv4I00EN5cudaq8gradient8gradientE | \[1\]](api/languages/cpp_api.html |
| RR13QuantumKernelRR10ArgsMapper), | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     [\[3                          |                                   |
| \]](api/languages/cpp_api.html#_C | [\[2\]](api/languages/cpp_api.htm |
| PPv4N5cudaq8gradient8gradientERRN | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [cudaq::QuakeValue::operator- |
|     [\[                           |     (C++                          |
| 4\]](api/languages/cpp_api.html#_ |     function)](api                |
| CPPv4N5cudaq8gradient8gradientEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gradient::setArgs     | udaq10QuakeValuemiE10QuakeValue), |
|     (C++                          |     [                             |
|     fu                            | \[1\]](api/languages/cpp_api.html |
| nction)](api/languages/cpp_api.ht | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     [                             |
| tArgsEvR13QuantumKernelDpRR4Args) | \[2\]](api/languages/cpp_api.html |
| -   [cudaq::gradient::setKernel   | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     (C++                          |                                   |
|     function)](api/languages/c    | [\[3\]](api/languages/cpp_api.htm |
| pp_api.html#_CPPv4I0EN5cudaq8grad | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| ient9setKernelEvR13QuantumKernel) | -   [cudaq::QuakeValue::operator/ |
| -   [cud                          |     (C++                          |
| aq::gradients::central_difference |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     class)](api/la                | udaq10QuakeValuedvE10QuakeValue), |
| nguages/cpp_api.html#_CPPv4N5cuda |                                   |
| q9gradients18central_differenceE) | [\[1\]](api/languages/cpp_api.htm |
| -   [cudaq::gra                   | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| dients::central_difference::clone | -                                 |
|     (C++                          |  [cudaq::QuakeValue::operator\[\] |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     function)](api                |
| ents18central_difference5cloneEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gradi                 | udaq10QuakeValueixEKNSt6size_tE), |
| ents::central_difference::compute |     [\[1\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     function)](                   | daq10QuakeValueixERK10QuakeValue) |
| api/languages/cpp_api.html#_CPPv4 | -                                 |
| N5cudaq9gradients18central_differ |    [cudaq::QuakeValue::QuakeValue |
| ence7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)](api/languag        |
|                                   | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|   [\[1\]](api/languages/cpp_api.h | akeValue10QuakeValueERN4mlir20Imp |
| tml#_CPPv4N5cudaq9gradients18cent | licitLocOpBuilderEN4mlir5ValueE), |
| ral_difference7computeERKNSt6vect |     [\[1\]                        |
| orIdEERNSt6vectorIdEERK7spin_opd) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::gradie                | v4N5cudaq10QuakeValue10QuakeValue |
| nts::central_difference::gradient | ERN4mlir20ImplicitLocOpBuilderEd) |
|     (C++                          | -   [cudaq::QuakeValue::size (C++ |
|     functio                       |     funct                         |
| n)](api/languages/cpp_api.html#_C | ion)](api/languages/cpp_api.html# |
| PPv4I00EN5cudaq9gradients18centra | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| l_difference8gradientER7KernelT), | -   [cudaq::QuakeValue::slice     |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     function)](api/languages/cpp_ |
| q9gradients18central_difference8g | api.html#_CPPv4N5cudaq10QuakeValu |
| radientER7KernelTRR10ArgsMapper), | e5sliceEKNSt6size_tEKNSt6size_tE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::quantum_platform (C++ |
| api.html#_CPPv4I00EN5cudaq9gradie |     cl                            |
| nts18central_difference8gradientE | ass)](api/languages/cpp_api.html# |
| RR13QuantumKernelRR10ArgsMapper), | _CPPv4N5cudaq16quantum_platformE) |
|     [\[3\]](api/languages/cpp     | -   [cudaq:                       |
| _api.html#_CPPv4N5cudaq9gradients | :quantum_platform::beginExecution |
| 18central_difference8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)](api/languag        |
|     [\[4\]](api/languages/cp      | es/cpp_api.html#_CPPv4N5cudaq16qu |
| p_api.html#_CPPv4N5cudaq9gradient | antum_platform14beginExecutionEv) |
| s18central_difference8gradientEv) | -   [cudaq::quantum_pl            |
| -   [cud                          | atform::configureExecutionContext |
| aq::gradients::forward_difference |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     class)](api/la                | uages/cpp_api.html#_CPPv4NK5cudaq |
| nguages/cpp_api.html#_CPPv4N5cuda | 16quantum_platform25configureExec |
| q9gradients18forward_differenceE) | utionContextER16ExecutionContext) |
| -   [cudaq::gra                   | -   [cuda                         |
| dients::forward_difference::clone | q::quantum_platform::connectivity |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/langu          |
| /cpp_api.html#_CPPv4N5cudaq9gradi | ages/cpp_api.html#_CPPv4N5cudaq16 |
| ents18forward_difference5cloneEv) | quantum_platform12connectivityEv) |
| -   [cudaq::gradi                 | -   [cuda                         |
| ents::forward_difference::compute | q::quantum_platform::endExecution |
|     (C++                          |     (C++                          |
|     function)](                   |     function)](api/langu          |
| api/languages/cpp_api.html#_CPPv4 | ages/cpp_api.html#_CPPv4N5cudaq16 |
| N5cudaq9gradients18forward_differ | quantum_platform12endExecutionEv) |
| ence7computeERKNSt6vectorIdEERKNS | -   [cudaq::q                     |
| t8functionIFdNSt6vectorIdEEEEEd), | uantum_platform::enqueueAsyncTask |
|                                   |     (C++                          |
|   [\[1\]](api/languages/cpp_api.h |     function)](api/languages/     |
| tml#_CPPv4N5cudaq9gradients18forw | cpp_api.html#_CPPv4N5cudaq16quant |
| ard_difference7computeERKNSt6vect | um_platform16enqueueAsyncTaskEKNS |
| orIdEERNSt6vectorIdEERK7spin_opd) | t6size_tER19KernelExecutionTask), |
| -   [cudaq::gradie                |     [\[1\]](api/languag           |
| nts::forward_difference::gradient | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform16enqueueAsyncTaskE |
|     functio                       | KNSt6size_tERNSt8functionIFvvEEE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::quantum_p             |
| PPv4I00EN5cudaq9gradients18forwar | latform::finalizeExecutionContext |
| d_difference8gradientER7KernelT), |     (C++                          |
|     [\[1\]](api/langua            |     function)](api/languages/c    |
| ges/cpp_api.html#_CPPv4I00EN5cuda | pp_api.html#_CPPv4NK5cudaq16quant |
| q9gradients18forward_difference8g | um_platform24finalizeExecutionCon |
| radientER7KernelTRR10ArgsMapper), | textERN5cudaq16ExecutionContextE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::qua                   |
| api.html#_CPPv4I00EN5cudaq9gradie | ntum_platform::get_codegen_config |
| nts18forward_difference8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/languages/c    |
|     [\[3\]](api/languages/cpp     | pp_api.html#_CPPv4N5cudaq16quantu |
| _api.html#_CPPv4N5cudaq9gradients | m_platform18get_codegen_configEv) |
| 18forward_difference8gradientERRN | -   [cuda                         |
| St8functionIFvNSt6vectorIdEEEEE), | q::quantum_platform::get_exec_ctx |
|     [\[4\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq9gradient |     function)](api/langua         |
| s18forward_difference8gradientEv) | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| -   [                             | quantum_platform12get_exec_ctxEv) |
| cudaq::gradients::parameter_shift | -   [c                            |
|     (C++                          | udaq::quantum_platform::get_noise |
|     class)](api                   |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     function)](api/languages/c    |
| udaq9gradients15parameter_shiftE) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [cudaq::                      | m_platform9get_noiseENSt6size_tE) |
| gradients::parameter_shift::clone | -   [cudaq:                       |
|     (C++                          | :quantum_platform::get_num_qubits |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |                                   |
| adients15parameter_shift5cloneEv) | function)](api/languages/cpp_api. |
| -   [cudaq::gr                    | html#_CPPv4NK5cudaq16quantum_plat |
| adients::parameter_shift::compute | form14get_num_qubitsENSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_              |
|     function                      | platform::get_remote_capabilities |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq9gradients15parameter_s |     function)                     |
| hift7computeERKNSt6vectorIdEERKNS | ](api/languages/cpp_api.html#_CPP |
| t8functionIFdNSt6vectorIdEEEEEd), | v4NK5cudaq16quantum_platform23get |
|     [\[1\]](api/languages/cpp_ap  | _remote_capabilitiesENSt6size_tE) |
| i.html#_CPPv4N5cudaq9gradients15p | -   [cudaq::qua                   |
| arameter_shift7computeERKNSt6vect | ntum_platform::get_runtime_target |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gra                   |     function)](api/languages/cp   |
| dients::parameter_shift::gradient | p_api.html#_CPPv4NK5cudaq16quantu |
|     (C++                          | m_platform18get_runtime_targetEv) |
|     func                          | -   [cuda                         |
| tion)](api/languages/cpp_api.html | q::quantum_platform::getLogStream |
| #_CPPv4I00EN5cudaq9gradients15par |     (C++                          |
| ameter_shift8gradientER7KernelT), |     function)](api/langu          |
|     [\[1\]](api/lan               | ages/cpp_api.html#_CPPv4N5cudaq16 |
| guages/cpp_api.html#_CPPv4I00EN5c | quantum_platform12getLogStreamEv) |
| udaq9gradients15parameter_shift8g | -   [cud                          |
| radientER7KernelTRR10ArgsMapper), | aq::quantum_platform::is_emulated |
|     [\[2\]](api/languages/c       |     (C++                          |
| pp_api.html#_CPPv4I00EN5cudaq9gra |                                   |
| dients15parameter_shift8gradientE |    function)](api/languages/cpp_a |
| RR13QuantumKernelRR10ArgsMapper), | pi.html#_CPPv4NK5cudaq16quantum_p |
|     [\[3\]](api/languages/        | latform11is_emulatedENSt6size_tE) |
| cpp_api.html#_CPPv4N5cudaq9gradie | -   [c                            |
| nts15parameter_shift8gradientERRN | udaq::quantum_platform::is_remote |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[4\]](api/languages         |     function)](api/languages/cp   |
| /cpp_api.html#_CPPv4N5cudaq9gradi | p_api.html#_CPPv4NK5cudaq16quantu |
| ents15parameter_shift8gradientEv) | m_platform9is_remoteENSt6size_tE) |
| -   [cudaq::kernel_builder (C++   | -   [cuda                         |
|     clas                          | q::quantum_platform::is_simulator |
| s)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4IDpEN5cudaq14kernel_builderE) |                                   |
| -   [c                            |   function)](api/languages/cpp_ap |
| udaq::kernel_builder::constantVal | i.html#_CPPv4NK5cudaq16quantum_pl |
|     (C++                          | atform12is_simulatorENSt6size_tE) |
|     function)](api/la             | -   [c                            |
| nguages/cpp_api.html#_CPPv4N5cuda | udaq::quantum_platform::launchVQE |
| q14kernel_builder11constantValEd) |     (C++                          |
| -   [cu                           |     function)](                   |
| daq::kernel_builder::getArguments | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq16quantum_platform9launchV |
|     function)](api/lan            | QEEKNSt6stringEPKvPN5cudaq8gradie |
| guages/cpp_api.html#_CPPv4N5cudaq | ntERKN5cudaq7spin_opERN5cudaq9opt |
| 14kernel_builder12getArgumentsEv) | imizerEKiKNSt6size_tENSt6size_tE) |
| -   [cu                           | -   [cudaq:                       |
| daq::kernel_builder::getNumParams | :quantum_platform::list_platforms |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function)](api/languag        |
| guages/cpp_api.html#_CPPv4N5cudaq | es/cpp_api.html#_CPPv4N5cudaq16qu |
| 14kernel_builder12getNumParamsEv) | antum_platform14list_platformsEv) |
| -   [c                            | -                                 |
| udaq::kernel_builder::isArgStdVec |    [cudaq::quantum_platform::name |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     function)](a                  |
| p_api.html#_CPPv4N5cudaq14kernel_ | pi/languages/cpp_api.html#_CPPv4N |
| builder11isArgStdVecENSt6size_tE) | K5cudaq16quantum_platform4nameEv) |
| -   [cuda                         | -   [                             |
| q::kernel_builder::kernel_builder | cudaq::quantum_platform::num_qpus |
|     (C++                          |     (C++                          |
|     function)](api/languages/cpp_ |     function)](api/l              |
| api.html#_CPPv4N5cudaq14kernel_bu | anguages/cpp_api.html#_CPPv4NK5cu |
| ilder14kernel_builderERNSt6vector | daq16quantum_platform8num_qpusEv) |
| IN7details17KernelBuilderTypeEEE) | -   [cudaq::                      |
| -   [cudaq::kernel_builder::name  | quantum_platform::onRandomSeedSet |
|     (C++                          |     (C++                          |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP | function)](api/languages/cpp_api. |
| v4N5cudaq14kernel_builder4nameEv) | html#_CPPv4N5cudaq16quantum_platf |
| -                                 | orm15onRandomSeedSetENSt6size_tE) |
|    [cudaq::kernel_builder::qalloc | -   [cudaq:                       |
|     (C++                          | :quantum_platform::reset_exec_ctx |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14ker |     function)](api/languag        |
| nel_builder6qallocE10QuakeValue), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     [\[1\]](api/language          | antum_platform14reset_exec_ctxEv) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cud                          |
| nel_builder6qallocEKNSt6size_tE), | aq::quantum_platform::reset_noise |
|     [\[2                          |     (C++                          |
| \]](api/languages/cpp_api.html#_C |     function)](api/languages/cpp_ |
| PPv4N5cudaq14kernel_builder6qallo | api.html#_CPPv4N5cudaq16quantum_p |
| cERNSt6vectorINSt7complexIdEEEE), | latform11reset_noiseENSt6size_tE) |
|     [\[3\]](                      | -   [cudaq:                       |
| api/languages/cpp_api.html#_CPPv4 | :quantum_platform::resetLogStream |
| N5cudaq14kernel_builder6qallocEv) |     (C++                          |
| -   [cudaq::kernel_builder::swap  |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     function)](api/language       | antum_platform14resetLogStreamEv) |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | -   [cuda                         |
| 4kernel_builder4swapEvRK10QuakeVa | q::quantum_platform::set_exec_ctx |
| lueRK10QuakeValueRK10QuakeValue), |     (C++                          |
|                                   |     funct                         |
| [\[1\]](api/languages/cpp_api.htm | ion)](api/languages/cpp_api.html# |
| l#_CPPv4I00EN5cudaq14kernel_build | _CPPv4N5cudaq16quantum_platform12 |
| er4swapEvRKNSt6vectorI10QuakeValu | set_exec_ctxEP16ExecutionContext) |
| eEERK10QuakeValueRK10QuakeValue), | -   [c                            |
|                                   | udaq::quantum_platform::set_noise |
| [\[2\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14kernel_builder4s |     function                      |
| wapERK10QuakeValueRK10QuakeValue) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::KernelExecutionTask   | Pv4N5cudaq16quantum_platform9set_ |
|     (C++                          | noiseEPK11noise_modelNSt6size_tE) |
|     type                          | -   [cuda                         |
| )](api/languages/cpp_api.html#_CP | q::quantum_platform::setLogStream |
| Pv4N5cudaq19KernelExecutionTaskE) |     (C++                          |
| -   [cudaq::KernelThunkResultType |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     struct)]                      | .html#_CPPv4N5cudaq16quantum_plat |
| (api/languages/cpp_api.html#_CPPv | form12setLogStreamERNSt7ostreamE) |
| 4N5cudaq21KernelThunkResultTypeE) | -   [cudaq::quantum_platfor       |
| -   [cudaq::KernelThunkType (C++  | m::supports_explicit_measurements |
|                                   |     (C++                          |
| type)](api/languages/cpp_api.html |     function)](api/l              |
| #_CPPv4N5cudaq15KernelThunkTypeE) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cudaq::kraus_channel (C++    | daq16quantum_platform30supports_e |
|                                   | xplicit_measurementsENSt6size_tE) |
|  class)](api/languages/cpp_api.ht | -   [cudaq::quantum_pla           |
| ml#_CPPv4N5cudaq13kraus_channelE) | tform::supports_task_distribution |
| -   [cudaq::kraus_channel::empty  |     (C++                          |
|     (C++                          |     fu                            |
|     function)]                    | nction)](api/languages/cpp_api.ht |
| (api/languages/cpp_api.html#_CPPv | ml#_CPPv4NK5cudaq16quantum_platfo |
| 4NK5cudaq13kraus_channel5emptyEv) | rm26supports_task_distributionEv) |
| -   [cudaq::kraus_c               | -   [cudaq::quantum               |
| hannel::generateUnitaryParameters | _platform::with_execution_context |
|     (C++                          |     (C++                          |
|                                   |     function)                     |
|    function)](api/languages/cpp_a | ](api/languages/cpp_api.html#_CPP |
| pi.html#_CPPv4N5cudaq13kraus_chan | v4I0DpEN5cudaq16quantum_platform2 |
| nel25generateUnitaryParametersEv) | 2with_execution_contextEDaR16Exec |
| -                                 | utionContextRR8CallableDpRR4Args) |
|    [cudaq::kraus_channel::get_ops | -   [cudaq::QuantumTask (C++      |
|     (C++                          |     type)](api/languages/cpp_api. |
|     function)](a                  | html#_CPPv4N5cudaq11QuantumTaskE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::qubit (C++            |
| K5cudaq13kraus_channel7get_opsEv) |     type)](api/languages/c        |
| -   [cud                          | pp_api.html#_CPPv4N5cudaq5qubitE) |
| aq::kraus_channel::identity_flags | -   [cudaq::QubitConnectivity     |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     ty                            |
| guages/cpp_api.html#_CPPv4N5cudaq | pe)](api/languages/cpp_api.html#_ |
| 13kraus_channel14identity_flagsE) | CPPv4N5cudaq17QubitConnectivityE) |
| -   [cud                          | -   [cudaq::QubitEdge (C++        |
| aq::kraus_channel::is_identity_op |     type)](api/languages/cpp_a    |
|     (C++                          | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|                                   | -   [cudaq::qudit (C++            |
|    function)](api/languages/cpp_a |     clas                          |
| pi.html#_CPPv4NK5cudaq13kraus_cha | s)](api/languages/cpp_api.html#_C |
| nnel14is_identity_opENSt6size_tE) | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| -   [cudaq::                      | -   [cudaq::qudit::qudit (C++     |
| kraus_channel::is_unitary_mixture |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     function)](api/languages      | html#_CPPv4N5cudaq5qudit5quditEv) |
| /cpp_api.html#_CPPv4NK5cudaq13kra | -   [cudaq::qvector (C++          |
| us_channel18is_unitary_mixtureEv) |     class)                        |
| -   [cu                           | ](api/languages/cpp_api.html#_CPP |
| daq::kraus_channel::kraus_channel | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|     (C++                          | -   [cudaq::qvector::back (C++    |
|     function)](api/lang           |     function)](a                  |
| uages/cpp_api.html#_CPPv4IDpEN5cu | pi/languages/cpp_api.html#_CPPv4N |
| daq13kraus_channel13kraus_channel | 5cudaq7qvector4backENSt6size_tE), |
| EDpRRNSt16initializer_listI1TEE), |                                   |
|                                   |   [\[1\]](api/languages/cpp_api.h |
|  [\[1\]](api/languages/cpp_api.ht | tml#_CPPv4N5cudaq7qvector4backEv) |
| ml#_CPPv4N5cudaq13kraus_channel13 | -   [cudaq::qvector::begin (C++   |
| kraus_channelERK13kraus_channel), |     fu                            |
|     [\[2\]                        | nction)](api/languages/cpp_api.ht |
| ](api/languages/cpp_api.html#_CPP | ml#_CPPv4N5cudaq7qvector5beginEv) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::qvector::clear (C++   |
| hannelERKNSt6vectorI8kraus_opEE), |     fu                            |
|     [\[3\]                        | nction)](api/languages/cpp_api.ht |
| ](api/languages/cpp_api.html#_CPP | ml#_CPPv4N5cudaq7qvector5clearEv) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::qvector::end (C++     |
| hannelERRNSt6vectorI8kraus_opEE), |                                   |
|     [\[4\]](api/lan               | function)](api/languages/cpp_api. |
| guages/cpp_api.html#_CPPv4N5cudaq | html#_CPPv4N5cudaq7qvector3endEv) |
| 13kraus_channel13kraus_channelEv) | -   [cudaq::qvector::front (C++   |
| -                                 |     function)](ap                 |
| [cudaq::kraus_channel::noise_type | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq7qvector5frontENSt6size_tE), |
|     member)](api                  |                                   |
| /languages/cpp_api.html#_CPPv4N5c |  [\[1\]](api/languages/cpp_api.ht |
| udaq13kraus_channel10noise_typeE) | ml#_CPPv4N5cudaq7qvector5frontEv) |
| -                                 | -   [cudaq::qvector::operator=    |
|   [cudaq::kraus_channel::op_names |     (C++                          |
|     (C++                          |     functio                       |
|     member)](                     | n)](api/languages/cpp_api.html#_C |
| api/languages/cpp_api.html#_CPPv4 | PPv4N5cudaq7qvectoraSERK7qvector) |
| N5cudaq13kraus_channel8op_namesE) | -   [cudaq::qvector::operator\[\] |
| -                                 |     (C++                          |
|  [cudaq::kraus_channel::operator= |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/langua         | v4N5cudaq7qvectorixEKNSt6size_tE) |
| ges/cpp_api.html#_CPPv4N5cudaq13k | -   [cudaq::qvector::qvector (C++ |
| raus_channelaSERK13kraus_channel) |     function)](api/               |
| -   [c                            | languages/cpp_api.html#_CPPv4N5cu |
| udaq::kraus_channel::operator\[\] | daq7qvector7qvectorENSt6size_tE), |
|     (C++                          |     [\[1\]](a                     |
|     function)](api/l              | pi/languages/cpp_api.html#_CPPv4N |
| anguages/cpp_api.html#_CPPv4N5cud | 5cudaq7qvector7qvectorERK5state), |
| aq13kraus_channelixEKNSt6size_tE) |     [\[2\]](api                   |
| -                                 | /languages/cpp_api.html#_CPPv4N5c |
| [cudaq::kraus_channel::parameters | udaq7qvector7qvectorERK7qvector), |
|     (C++                          |     [\[3\]](api/languages/cpp     |
|     member)](api                  | _api.html#_CPPv4N5cudaq7qvector7q |
| /languages/cpp_api.html#_CPPv4N5c | vectorERKNSt6vectorI7complexEEb), |
| udaq13kraus_channel10parametersE) |     [\[4\]](ap                    |
| -   [cudaq::krau                  | i/languages/cpp_api.html#_CPPv4N5 |
| s_channel::populateDefaultOpNames | cudaq7qvector7qvectorERR7qvector) |
|     (C++                          | -   [cudaq::qvector::size (C++    |
|     function)](api/languages/cp   |     fu                            |
| p_api.html#_CPPv4N5cudaq13kraus_c | nction)](api/languages/cpp_api.ht |
| hannel22populateDefaultOpNamesEv) | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| -   [cu                           | -   [cudaq::qvector::slice (C++   |
| daq::kraus_channel::probabilities |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq7qvec |
|     member)](api/la               | tor5sliceENSt6size_tENSt6size_tE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::qvector::value_type   |
| q13kraus_channel13probabilitiesE) |     (C++                          |
| -                                 |     typ                           |
|  [cudaq::kraus_channel::push_back | e)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq7qvector10value_typeE) |
|     function)](api                | -   [cudaq::qview (C++            |
| /languages/cpp_api.html#_CPPv4N5c |     clas                          |
| udaq13kraus_channel9push_backE8kr | s)](api/languages/cpp_api.html#_C |
| aus_opNSt8optionalINSt6stringEEE) | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| -   [cudaq::kraus_channel::size   | -   [cudaq::qview::back (C++      |
|     (C++                          |     function)                     |
|     function)                     | ](api/languages/cpp_api.html#_CPP |
| ](api/languages/cpp_api.html#_CPP | v4N5cudaq5qview4backENSt6size_tE) |
| v4NK5cudaq13kraus_channel4sizeEv) | -   [cudaq::qview::begin (C++     |
| -   [                             |                                   |
| cudaq::kraus_channel::unitary_ops | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq5qview5beginEv) |
|     member)](api/                 | -   [cudaq::qview::end (C++       |
| languages/cpp_api.html#_CPPv4N5cu |                                   |
| daq13kraus_channel11unitary_opsE) |   function)](api/languages/cpp_ap |
| -   [cudaq::kraus_op (C++         | i.html#_CPPv4N5cudaq5qview3endEv) |
|     struct)](api/languages/cpp_   | -   [cudaq::qview::front (C++     |
| api.html#_CPPv4N5cudaq8kraus_opE) |     function)](                   |
| -   [cudaq::kraus_op::adjoint     | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq5qview5frontENSt6size_tE), |
|     functi                        |                                   |
| on)](api/languages/cpp_api.html#_ |    [\[1\]](api/languages/cpp_api. |
| CPPv4NK5cudaq8kraus_op7adjointEv) | html#_CPPv4N5cudaq5qview5frontEv) |
| -   [cudaq::kraus_op::data (C++   | -   [cudaq::qview::operator\[\]   |
|                                   |     (C++                          |
|  member)](api/languages/cpp_api.h |     functio                       |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::kraus_op::kraus_op    | PPv4N5cudaq5qviewixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qview::qview (C++     |
|     func                          |     functio                       |
| tion)](api/languages/cpp_api.html | n)](api/languages/cpp_api.html#_C |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | PPv4I0EN5cudaq5qview5qviewERR1R), |
| opERRNSt16initializer_listI1TEE), |     [\[1                          |
|                                   | \]](api/languages/cpp_api.html#_C |
|  [\[1\]](api/languages/cpp_api.ht | PPv4N5cudaq5qview5qviewERK5qview) |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | -   [cudaq::qview::size (C++      |
| pENSt6vectorIN5cudaq7complexEEE), |                                   |
|     [\[2\]](api/l                 | function)](api/languages/cpp_api. |
| anguages/cpp_api.html#_CPPv4N5cud | html#_CPPv4NK5cudaq5qview4sizeEv) |
| aq8kraus_op8kraus_opERK8kraus_op) | -   [cudaq::qview::slice (C++     |
| -   [cudaq::kraus_op::nCols (C++  |     function)](api/langua         |
|                                   | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| member)](api/languages/cpp_api.ht | iew5sliceENSt6size_tENSt6size_tE) |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | -   [cudaq::qview::value_type     |
| -   [cudaq::kraus_op::nRows (C++  |     (C++                          |
|                                   |     t                             |
| member)](api/languages/cpp_api.ht | ype)](api/languages/cpp_api.html# |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | _CPPv4N5cudaq5qview10value_typeE) |
| -   [cudaq::kraus_op::operator=   | -   [cudaq::range (C++            |
|     (C++                          |     fun                           |
|     function)                     | ction)](api/languages/cpp_api.htm |
| ](api/languages/cpp_api.html#_CPP | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| v4N5cudaq8kraus_opaSERK8kraus_op) | orI11ElementTypeEE11ElementType), |
| -   [cudaq::kraus_op::precision   |     [\[1\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4I0EN5cudaq5rangeEN |
|     memb                          | St6vectorI11ElementTypeEE11Elemen |
| er)](api/languages/cpp_api.html#_ | tType11ElementType11ElementType), |
| CPPv4N5cudaq8kraus_op9precisionE) |     [                             |
| -   [cudaq::KrausOperatorType     | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq5rangeENSt6size_tE) |
|     en                            | -   [cudaq::real (C++             |
| um)](api/languages/cpp_api.html#_ |     type)](api/languages/         |
| CPPv4N5cudaq17KrausOperatorTypeE) | cpp_api.html#_CPPv4N5cudaq4realE) |
| -   [c                            | -   [cudaq::registry (C++         |
| udaq::KrausOperatorType::IDENTITY |     type)](api/languages/cpp_     |
|     (C++                          | api.html#_CPPv4N5cudaq8registryE) |
|     enumerator)](api/             | -                                 |
| languages/cpp_api.html#_CPPv4N5cu |  [cudaq::registry::RegisteredType |
| daq17KrausOperatorType8IDENTITYE) |     (C++                          |
| -   [cudaq::KrausSelection (C++   |     class)](api/                  |
|     s                             | languages/cpp_api.html#_CPPv4I0EN |
| truct)](api/languages/cpp_api.htm | 5cudaq8registry14RegisteredTypeE) |
| l#_CPPv4N5cudaq14KrausSelectionE) | -   [cudaq::RemoteCapabilities    |
| -   [cudaq:                       |     (C++                          |
| :KrausSelection::circuit_location |     struc                         |
|     (C++                          | t)](api/languages/cpp_api.html#_C |
|     member)](api/langua           | PPv4N5cudaq18RemoteCapabilitiesE) |
| ges/cpp_api.html#_CPPv4N5cudaq14K | -   [cudaq::Remo                  |
| rausSelection16circuit_locationE) | teCapabilities::isRemoteSimulator |
| -   [cudaq::Kra                   |     (C++                          |
| usSelection::kraus_operator_index |     member)](api/languages/c      |
|     (C++                          | pp_api.html#_CPPv4N5cudaq18Remote |
|     member)](api/languages/       | Capabilities17isRemoteSimulatorE) |
| cpp_api.html#_CPPv4N5cudaq14Kraus | -   [cudaq::Remot                 |
| Selection20kraus_operator_indexE) | eCapabilities::RemoteCapabilities |
| -                                 |     (C++                          |
|   [cudaq::KrausSelection::op_name |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4N5cudaq18RemoteCa |
|     member)](                     | pabilities18RemoteCapabilitiesEb) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq:                       |
| N5cudaq14KrausSelection7op_nameE) | :RemoteCapabilities::stateOverlap |
| -                                 |     (C++                          |
|    [cudaq::KrausSelection::qubits |     member)](api/langua           |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq18R |
|     member)]                      | emoteCapabilities12stateOverlapE) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq14KrausSelection6qubitsE) |   [cudaq::RemoteCapabilities::vqe |
| -   [cudaq::KrausTrajectory (C++  |     (C++                          |
|     st                            |     member)](                     |
| ruct)](api/languages/cpp_api.html | api/languages/cpp_api.html#_CPPv4 |
| #_CPPv4N5cudaq15KrausTrajectoryE) | N5cudaq18RemoteCapabilities3vqeE) |
| -   [cu                           | -   [cudaq::RemoteSimulationState |
| daq::KrausTrajectory::countErrors |     (C++                          |
|     (C++                          |     class)]                       |
|     function)](api/lang           | (api/languages/cpp_api.html#_CPPv |
| uages/cpp_api.html#_CPPv4NK5cudaq | 4N5cudaq21RemoteSimulationStateE) |
| 15KrausTrajectory11countErrorsEv) | -   [cudaq::Resources (C++        |
| -   [                             |     class)](api/languages/cpp_a   |
| cudaq::KrausTrajectory::isOrdered | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     (C++                          | -   [cudaq::run (C++              |
|     function)](api/l              |     function)]                    |
| anguages/cpp_api.html#_CPPv4NK5cu | (api/languages/cpp_api.html#_CPPv |
| daq15KrausTrajectory9isOrderedEv) | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| -   [cudaq::                      | 5invoke_result_tINSt7decay_tI13Qu |
| KrausTrajectory::kraus_selections | antumKernelEEDpNSt7decay_tI4ARGSE |
|     (C++                          | EEEEENSt6size_tERN5cudaq11noise_m |
|     member)](api/languag          | odelERR13QuantumKernelDpRR4ARGS), |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |     [\[1\]](api/langu             |
| ausTrajectory16kraus_selectionsE) | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| -   [cudaq::Kr                    | daq3runENSt6vectorINSt15invoke_re |
| ausTrajectory::measurement_counts | sult_tINSt7decay_tI13QuantumKerne |
|     (C++                          | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
|     member)](api/languages        | ize_tERR13QuantumKernelDpRR4ARGS) |
| /cpp_api.html#_CPPv4N5cudaq15Krau | -   [cudaq::run_async (C++        |
| sTrajectory18measurement_countsE) |     functio                       |
| -   [cud                          | n)](api/languages/cpp_api.html#_C |
| aq::KrausTrajectory::multiplicity | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     (C++                          | tureINSt6vectorINSt15invoke_resul |
|     member)](api/lan              | t_tINSt7decay_tI13QuantumKernelEE |
| guages/cpp_api.html#_CPPv4N5cudaq | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| 15KrausTrajectory12multiplicityE) | ze_tENSt6size_tERN5cudaq11noise_m |
| -   [                             | odelERR13QuantumKernelDpRR4ARGS), |
| cudaq::KrausTrajectory::num_shots |     [\[1\]](api/la                |
|     (C++                          | nguages/cpp_api.html#_CPPv4I0DpEN |
|     member)](api                  | 5cudaq9run_asyncENSt6futureINSt6v |
| /languages/cpp_api.html#_CPPv4N5c | ectorINSt15invoke_result_tINSt7de |
| udaq15KrausTrajectory9num_shotsE) | cay_tI13QuantumKernelEEDpNSt7deca |
| -   [cu                           | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| daq::KrausTrajectory::probability | ize_tERR13QuantumKernelDpRR4ARGS) |
|     (C++                          | -   [cudaq::RuntimeTarget (C++    |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda | struct)](api/languages/cpp_api.ht |
| q15KrausTrajectory11probabilityE) | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| -   [cuda                         | -   [cudaq::sample (C++           |
| q::KrausTrajectory::trajectory_id |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     member)](api/lang             | mpleE13sample_resultRK14sample_op |
| uages/cpp_api.html#_CPPv4N5cudaq1 | tionsRR13QuantumKernelDpRR4Args), |
| 5KrausTrajectory13trajectory_idE) |     [\[1\                         |
| -   [cudaq::matrix_callback (C++  | ]](api/languages/cpp_api.html#_CP |
|     c                             | Pv4I0DpEN5cudaq6sampleE13sample_r |
| lass)](api/languages/cpp_api.html | esultRR13QuantumKernelDpRR4Args), |
| #_CPPv4N5cudaq15matrix_callbackE) |     [\                            |
| -   [cudaq::matrix_handler (C++   | [2\]](api/languages/cpp_api.html# |
|                                   | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| class)](api/languages/cpp_api.htm | ize_tERR13QuantumKernelDpRR4Args) |
| l#_CPPv4N5cudaq14matrix_handlerE) | -   [cudaq::sample_options (C++   |
| -   [cudaq::mat                   |     s                             |
| rix_handler::commutation_behavior | truct)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq14sample_optionsE) |
|     struct)](api/languages/       | -   [cudaq::sample_result (C++    |
| cpp_api.html#_CPPv4N5cudaq14matri |                                   |
| x_handler20commutation_behaviorE) |  class)](api/languages/cpp_api.ht |
| -                                 | ml#_CPPv4N5cudaq13sample_resultE) |
|    [cudaq::matrix_handler::define | -   [cudaq::sample_result::append |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)](api/languages/cpp_ |
| pi/languages/cpp_api.html#_CPPv4N | api.html#_CPPv4N5cudaq13sample_re |
| 5cudaq14matrix_handler6defineENSt | sult6appendERK15ExecutionResultb) |
| 6stringENSt6vectorINSt7int64_tEEE | -   [cudaq::sample_result::begin  |
| RR15matrix_callbackRKNSt13unorder |     (C++                          |
| ed_mapINSt6stringENSt6stringEEE), |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
| [\[1\]](api/languages/cpp_api.htm | 4N5cudaq13sample_result5beginEv), |
| l#_CPPv4N5cudaq14matrix_handler6d |     [\[1\]]                       |
| efineENSt6stringENSt6vectorINSt7i | (api/languages/cpp_api.html#_CPPv |
| nt64_tEEERR15matrix_callbackRR20d | 4NK5cudaq13sample_result5beginEv) |
| iag_matrix_callbackRKNSt13unorder | -   [cudaq::sample_result::cbegin |
| ed_mapINSt6stringENSt6stringEEE), |     (C++                          |
|     [\[2\]](                      |     function)](                   |
| api/languages/cpp_api.html#_CPPv4 | api/languages/cpp_api.html#_CPPv4 |
| N5cudaq14matrix_handler6defineENS | NK5cudaq13sample_result6cbeginEv) |
| t6stringENSt6vectorINSt7int64_tEE | -   [cudaq::sample_result::cend   |
| ERR15matrix_callbackRRNSt13unorde |     (C++                          |
| red_mapINSt6stringENSt6stringEEE) |     function)                     |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|   [cudaq::matrix_handler::degrees | v4NK5cudaq13sample_result4cendEv) |
|     (C++                          | -   [cudaq::sample_result::clear  |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4NK |     function)                     |
| 5cudaq14matrix_handler7degreesEv) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4N5cudaq13sample_result5clearEv) |
|  [cudaq::matrix_handler::displace | -   [cudaq::sample_result::count  |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](                   |
| s/cpp_api.html#_CPPv4N5cudaq14mat | api/languages/cpp_api.html#_CPPv4 |
| rix_handler8displaceENSt6size_tE) | NK5cudaq13sample_result5countENSt |
| -   [cudaq::matrix                | 11string_viewEKNSt11string_viewE) |
| _handler::get_expected_dimensions | -   [                             |
|     (C++                          | cudaq::sample_result::deserialize |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     functio                       |
| pi.html#_CPPv4NK5cudaq14matrix_ha | n)](api/languages/cpp_api.html#_C |
| ndler23get_expected_dimensionsEv) | PPv4N5cudaq13sample_result11deser |
| -   [cudaq::matrix_ha             | ializeERNSt6vectorINSt6size_tEEE) |
| ndler::get_parameter_descriptions | -   [cudaq::sample_result::dump   |
|     (C++                          |     (C++                          |
|                                   |     function)](api/languag        |
| function)](api/languages/cpp_api. | es/cpp_api.html#_CPPv4NK5cudaq13s |
| html#_CPPv4NK5cudaq14matrix_handl | ample_result4dumpERNSt7ostreamE), |
| er26get_parameter_descriptionsEv) |     [\[1\]                        |
| -   [c                            | ](api/languages/cpp_api.html#_CPP |
| udaq::matrix_handler::instantiate | v4NK5cudaq13sample_result4dumpEv) |
|     (C++                          | -   [cudaq::sample_result::end    |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function                      |
| 5cudaq14matrix_handler11instantia | )](api/languages/cpp_api.html#_CP |
| teENSt6stringERKNSt6vectorINSt6si | Pv4N5cudaq13sample_result3endEv), |
| ze_tEEERK20commutation_behavior), |     [\[1\                         |
|     [\[1\]](                      | ]](api/languages/cpp_api.html#_CP |
| api/languages/cpp_api.html#_CPPv4 | Pv4NK5cudaq13sample_result3endEv) |
| N5cudaq14matrix_handler11instanti | -   [                             |
| ateENSt6stringERRNSt6vectorINSt6s | cudaq::sample_result::expectation |
| ize_tEEERK20commutation_behavior) |     (C++                          |
| -   [cuda                         |     f                             |
| q::matrix_handler::matrix_handler | unction)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4NK5cudaq13sample_result |
|     function)](api/languag        | 11expectationEKNSt11string_viewE) |
| es/cpp_api.html#_CPPv4I0_NSt11ena | -   [c                            |
| ble_if_tINSt12is_base_of_vI16oper | udaq::sample_result::get_marginal |
| ator_handler1TEEbEEEN5cudaq14matr |     (C++                          |
| ix_handler14matrix_handlerERK1T), |     function)](api/languages/cpp_ |
|     [\[1\]](ap                    | api.html#_CPPv4NK5cudaq13sample_r |
| i/languages/cpp_api.html#_CPPv4I0 | esult12get_marginalERKNSt6vectorI |
| _NSt11enable_if_tINSt12is_base_of | NSt6size_tEEEKNSt11string_viewE), |
| _vI16operator_handler1TEEbEEEN5cu |     [\[1\]](api/languages/cpp_    |
| daq14matrix_handler14matrix_handl | api.html#_CPPv4NK5cudaq13sample_r |
| erERK1TRK20commutation_behavior), | esult12get_marginalERRKNSt6vector |
|     [\[2\]](api/languages/cpp_ap  | INSt6size_tEEEKNSt11string_viewE) |
| i.html#_CPPv4N5cudaq14matrix_hand | -   [cuda                         |
| ler14matrix_handlerENSt6size_tE), | q::sample_result::get_total_shots |
|     [\[3\]](api/                  |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/langua         |
| daq14matrix_handler14matrix_handl | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| erENSt6stringERKNSt6vectorINSt6si | sample_result15get_total_shotsEv) |
| ze_tEEERK20commutation_behavior), | -   [cuda                         |
|     [\[4\]](api/                  | q::sample_result::has_even_parity |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq14matrix_handler14matrix_handl |     fun                           |
| erENSt6stringERRNSt6vectorINSt6si | ction)](api/languages/cpp_api.htm |
| ze_tEEERK20commutation_behavior), | l#_CPPv4N5cudaq13sample_result15h |
|     [\                            | as_even_parityENSt11string_viewE) |
| [5\]](api/languages/cpp_api.html# | -   [cuda                         |
| _CPPv4N5cudaq14matrix_handler14ma | q::sample_result::has_expectation |
| trix_handlerERK14matrix_handler), |     (C++                          |
|     [                             |     funct                         |
| \[6\]](api/languages/cpp_api.html | ion)](api/languages/cpp_api.html# |
| #_CPPv4N5cudaq14matrix_handler14m | _CPPv4NK5cudaq13sample_result15ha |
| atrix_handlerERR14matrix_handler) | s_expectationEKNSt11string_viewE) |
| -                                 | -   [cu                           |
|  [cudaq::matrix_handler::momentum | daq::sample_result::most_probable |
|     (C++                          |     (C++                          |
|     function)](api/language       |     fun                           |
| s/cpp_api.html#_CPPv4N5cudaq14mat | ction)](api/languages/cpp_api.htm |
| rix_handler8momentumENSt6size_tE) | l#_CPPv4NK5cudaq13sample_result13 |
| -                                 | most_probableEKNSt11string_viewE) |
|    [cudaq::matrix_handler::number | -                                 |
|     (C++                          | [cudaq::sample_result::operator+= |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     function)](api/langua         |
| atrix_handler6numberENSt6size_tE) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -                                 | ample_resultpLERK13sample_result) |
| [cudaq::matrix_handler::operator= | -                                 |
|     (C++                          |  [cudaq::sample_result::operator= |
|     fun                           |     (C++                          |
| ction)](api/languages/cpp_api.htm |     function)](api/langua         |
| l#_CPPv4I0_NSt11enable_if_tIXaant | ges/cpp_api.html#_CPPv4N5cudaq13s |
| NSt7is_sameI1T14matrix_handlerE5v | ample_resultaSERR13sample_result) |
| alueENSt12is_base_of_vI16operator | -                                 |
| _handler1TEEEbEEEN5cudaq14matrix_ | [cudaq::sample_result::operator== |
| handleraSER14matrix_handlerRK1T), |     (C++                          |
|     [\[1\]](api/languages         |     function)](api/languag        |
| /cpp_api.html#_CPPv4N5cudaq14matr | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ix_handleraSERK14matrix_handler), | ample_resulteqERK13sample_result) |
|     [\[2\]](api/language          | -   [                             |
| s/cpp_api.html#_CPPv4N5cudaq14mat | cudaq::sample_result::probability |
| rix_handleraSERR14matrix_handler) |     (C++                          |
| -   [                             |     function)](api/lan            |
| cudaq::matrix_handler::operator== | guages/cpp_api.html#_CPPv4NK5cuda |
|     (C++                          | q13sample_result11probabilityENSt |
|     function)](api/languages      | 11string_viewEKNSt11string_viewE) |
| /cpp_api.html#_CPPv4NK5cudaq14mat | -   [cud                          |
| rix_handlereqERK14matrix_handler) | aq::sample_result::register_names |
| -                                 |     (C++                          |
|    [cudaq::matrix_handler::parity |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4NK5cudaq1 |
|     function)](api/langua         | 3sample_result14register_namesEv) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -                                 |
| atrix_handler6parityENSt6size_tE) |    [cudaq::sample_result::reorder |
| -                                 |     (C++                          |
|  [cudaq::matrix_handler::position |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     function)](api/language       | ample_result7reorderERKNSt6vector |
| s/cpp_api.html#_CPPv4N5cudaq14mat | INSt6size_tEEEKNSt11string_viewE) |
| rix_handler8positionENSt6size_tE) | -   [cu                           |
| -   [cudaq::                      | daq::sample_result::sample_result |
| matrix_handler::remove_definition |     (C++                          |
|     (C++                          |     func                          |
|     fu                            | tion)](api/languages/cpp_api.html |
| nction)](api/languages/cpp_api.ht | #_CPPv4N5cudaq13sample_result13sa |
| ml#_CPPv4N5cudaq14matrix_handler1 | mple_resultERK15ExecutionResult), |
| 7remove_definitionERKNSt6stringE) |     [\[1\]](api/la                |
| -                                 | nguages/cpp_api.html#_CPPv4N5cuda |
|   [cudaq::matrix_handler::squeeze | q13sample_result13sample_resultER |
|     (C++                          | KNSt6vectorI15ExecutionResultEE), |
|     function)](api/languag        |                                   |
| es/cpp_api.html#_CPPv4N5cudaq14ma |  [\[2\]](api/languages/cpp_api.ht |
| trix_handler7squeezeENSt6size_tE) | ml#_CPPv4N5cudaq13sample_result13 |
| -   [cudaq::m                     | sample_resultERR13sample_result), |
| atrix_handler::to_diagonal_matrix |     [                             |
|     (C++                          | \[3\]](api/languages/cpp_api.html |
|     function)](api/lang           | #_CPPv4N5cudaq13sample_result13sa |
| uages/cpp_api.html#_CPPv4NK5cudaq | mple_resultERR15ExecutionResult), |
| 14matrix_handler18to_diagonal_mat |     [\[4\]](api/lan               |
| rixERNSt13unordered_mapINSt6size_ | guages/cpp_api.html#_CPPv4N5cudaq |
| tENSt7int64_tEEERKNSt13unordered_ | 13sample_result13sample_resultEdR |
| mapINSt6stringENSt7complexIdEEEE) | KNSt6vectorI15ExecutionResultEE), |
| -                                 |     [\[5\]](api/lan               |
| [cudaq::matrix_handler::to_matrix | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 13sample_result13sample_resultEv) |
|     function)                     | -                                 |
| ](api/languages/cpp_api.html#_CPP |  [cudaq::sample_result::serialize |
| v4NK5cudaq14matrix_handler9to_mat |     (C++                          |
| rixERNSt13unordered_mapINSt6size_ |     function)](api                |
| tENSt7int64_tEEERKNSt13unordered_ | /languages/cpp_api.html#_CPPv4NK5 |
| mapINSt6stringENSt7complexIdEEEE) | cudaq13sample_result9serializeEv) |
| -                                 | -   [cudaq::sample_result::size   |
| [cudaq::matrix_handler::to_string |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/               | pp_api.html#_CPPv4NK5cudaq13sampl |
| languages/cpp_api.html#_CPPv4NK5c | e_result4sizeEKNSt11string_viewE) |
| udaq14matrix_handler9to_stringEb) | -   [cudaq::sample_result::to_map |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::unique_id |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq13sample_ |
|     function)](api/               | result6to_mapEKNSt11string_viewE) |
| languages/cpp_api.html#_CPPv4NK5c | -   [cuda                         |
| udaq14matrix_handler9unique_idEv) | q::sample_result::\~sample_result |
| -   [cudaq:                       |     (C++                          |
| :matrix_handler::\~matrix_handler |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     functi                        | _CPPv4N5cudaq13sample_resultD0Ev) |
| on)](api/languages/cpp_api.html#_ | -   [cudaq::scalar_callback (C++  |
| CPPv4N5cudaq14matrix_handlerD0Ev) |     c                             |
| -   [cudaq::matrix_op (C++        | lass)](api/languages/cpp_api.html |
|     type)](api/languages/cpp_a    | #_CPPv4N5cudaq15scalar_callbackE) |
| pi.html#_CPPv4N5cudaq9matrix_opE) | -   [c                            |
| -   [cudaq::matrix_op_term (C++   | udaq::scalar_callback::operator() |
|                                   |     (C++                          |
|  type)](api/languages/cpp_api.htm |     function)](api/language       |
| l#_CPPv4N5cudaq14matrix_op_termE) | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| -                                 | alar_callbackclERKNSt13unordered_ |
|    [cudaq::mdiag_operator_handler | mapINSt6stringENSt7complexIdEEEE) |
|     (C++                          | -   [                             |
|     class)](                      | cudaq::scalar_callback::operator= |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22mdiag_operator_handlerE) |     function)](api/languages/c    |
| -   [cudaq::mpi (C++              | pp_api.html#_CPPv4N5cudaq15scalar |
|     type)](api/languages          | _callbackaSERK15scalar_callback), |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) |     [\[1\]](api/languages/        |
| -   [cudaq::mpi::all_gather (C++  | cpp_api.html#_CPPv4N5cudaq15scala |
|     fu                            | r_callbackaSERR15scalar_callback) |
| nction)](api/languages/cpp_api.ht | -   [cudaq:                       |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | :scalar_callback::scalar_callback |
| RNSt6vectorIdEERKNSt6vectorIdEE), |     (C++                          |
|                                   |     function)](api/languag        |
|   [\[1\]](api/languages/cpp_api.h | es/cpp_api.html#_CPPv4I0_NSt11ena |
| tml#_CPPv4N5cudaq3mpi10all_gather | ble_if_tINSt16is_invocable_r_vINS |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | t7complexIdEE8CallableRKNSt13unor |
| -   [cudaq::mpi::all_reduce (C++  | dered_mapINSt6stringENSt7complexI |
|                                   | dEEEEEEbEEEN5cudaq15scalar_callba |
|  function)](api/languages/cpp_api | ck15scalar_callbackERR8Callable), |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     [\[1\                         |
| reduceE1TRK1TRK14BinaryFunction), | ]](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/langu             | Pv4N5cudaq15scalar_callback15scal |
| ages/cpp_api.html#_CPPv4I00EN5cud | ar_callbackERK15scalar_callback), |
| aq3mpi10all_reduceE1TRK1TRK4Func) |     [\[2                          |
| -   [cudaq::mpi::broadcast (C++   | \]](api/languages/cpp_api.html#_C |
|     function)](api/               | PPv4N5cudaq15scalar_callback15sca |
| languages/cpp_api.html#_CPPv4N5cu | lar_callbackERR15scalar_callback) |
| daq3mpi9broadcastERNSt6stringEi), | -   [cudaq::scalar_operator (C++  |
|     [\[1\]](api/la                |     c                             |
| nguages/cpp_api.html#_CPPv4N5cuda | lass)](api/languages/cpp_api.html |
| q3mpi9broadcastERNSt6vectorIdEEi) | #_CPPv4N5cudaq15scalar_operatorE) |
| -   [cudaq::mpi::finalize (C++    | -                                 |
|     f                             | [cudaq::scalar_operator::evaluate |
| unction)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |                                   |
| -   [cudaq::mpi::initialize (C++  |    function)](api/languages/cpp_a |
|     function                      | pi.html#_CPPv4NK5cudaq15scalar_op |
| )](api/languages/cpp_api.html#_CP | erator8evaluateERKNSt13unordered_ |
| Pv4N5cudaq3mpi10initializeEiPPc), | mapINSt6stringENSt7complexIdEEEE) |
|     [                             | -   [cudaq::scalar_ope            |
| \[1\]](api/languages/cpp_api.html | rator::get_parameter_descriptions |
| #_CPPv4N5cudaq3mpi10initializeEv) |     (C++                          |
| -   [cudaq::mpi::is_initialized   |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     function                      | tml#_CPPv4NK5cudaq15scalar_operat |
| )](api/languages/cpp_api.html#_CP | or26get_parameter_descriptionsEv) |
| Pv4N5cudaq3mpi14is_initializedEv) | -   [cu                           |
| -   [cudaq::mpi::num_ranks (C++   | daq::scalar_operator::is_constant |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/lang           |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [cudaq::mpi::rank (C++        | 15scalar_operator11is_constantEv) |
|                                   | -   [c                            |
|    function)](api/languages/cpp_a | udaq::scalar_operator::operator\* |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     (C++                          |
| -   [cudaq::noise_model (C++      |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|    class)](api/languages/cpp_api. | Pv4N5cudaq15scalar_operatormlENSt |
| html#_CPPv4N5cudaq11noise_modelE) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::n                     |     [\[1\                         |
| oise_model::add_all_qubit_channel | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormlENSt |
|     function)](api                | 7complexIdEERR15scalar_operator), |
| /languages/cpp_api.html#_CPPv4IDp |     [\[2\]](api/languages/cp      |
| EN5cudaq11noise_model21add_all_qu | p_api.html#_CPPv4N5cudaq15scalar_ |
| bit_channelEvRK13kraus_channeli), | operatormlEdRK15scalar_operator), |
|     [\[1\]](api/langua            |     [\[3\]](api/languages/cp      |
| ges/cpp_api.html#_CPPv4N5cudaq11n | p_api.html#_CPPv4N5cudaq15scalar_ |
| oise_model21add_all_qubit_channel | operatormlEdRR15scalar_operator), |
| ERKNSt6stringERK13kraus_channeli) |     [\[4\]](api/languages         |
| -                                 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|  [cudaq::noise_model::add_channel | alar_operatormlENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     funct                         | _api.html#_CPPv4NKR5cudaq15scalar |
| ion)](api/languages/cpp_api.html# | _operatormlERK15scalar_operator), |
| _CPPv4IDpEN5cudaq11noise_model11a |     [\[6\]]                       |
| dd_channelEvRK15PredicateFuncTy), | (api/languages/cpp_api.html#_CPPv |
|     [\[1\]](api/languages/cpp_    | 4NKR5cudaq15scalar_operatormlEd), |
| api.html#_CPPv4IDpEN5cudaq11noise |     [\[7\]](api/language          |
| _model11add_channelEvRKNSt6vector | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| INSt6size_tEEERK13kraus_channel), | alar_operatormlENSt7complexIdEE), |
|     [\[2\]](ap                    |     [\[8\]](api/languages/cp      |
| i/languages/cpp_api.html#_CPPv4N5 | p_api.html#_CPPv4NO5cudaq15scalar |
| cudaq11noise_model11add_channelER | _operatormlERK15scalar_operator), |
| KNSt6stringERK15PredicateFuncTy), |     [\[9\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
| [\[3\]](api/languages/cpp_api.htm | Pv4NO5cudaq15scalar_operatormlEd) |
| l#_CPPv4N5cudaq11noise_model11add | -   [cu                           |
| _channelERKNSt6stringERKNSt6vecto | daq::scalar_operator::operator\*= |
| rINSt6size_tEEERK13kraus_channel) |     (C++                          |
| -   [cudaq::noise_model::empty    |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     function                      | alar_operatormLENSt7complexIdEE), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]](api/languages/c       |
| Pv4NK5cudaq11noise_model5emptyEv) | pp_api.html#_CPPv4N5cudaq15scalar |
| -                                 | _operatormLERK15scalar_operator), |
| [cudaq::noise_model::get_channels |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     function)](api/l              | PPv4N5cudaq15scalar_operatormLEd) |
| anguages/cpp_api.html#_CPPv4I0ENK | -   [                             |
| 5cudaq11noise_model12get_channels | cudaq::scalar_operator::operator+ |
| ENSt6vectorI13kraus_channelEERKNS |     (C++                          |
| t6vectorINSt6size_tEEERKNSt6vecto |     function                      |
| rINSt6size_tEEERKNSt6vectorIdEE), | )](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/languages/cpp_a   | Pv4N5cudaq15scalar_operatorplENSt |
| pi.html#_CPPv4NK5cudaq11noise_mod | 7complexIdEERK15scalar_operator), |
| el12get_channelsERKNSt6stringERKN |     [\[1\                         |
| St6vectorINSt6size_tEEERKNSt6vect | ]](api/languages/cpp_api.html#_CP |
| orINSt6size_tEEERKNSt6vectorIdEE) | Pv4N5cudaq15scalar_operatorplENSt |
| -                                 | 7complexIdEERR15scalar_operator), |
|  [cudaq::noise_model::noise_model |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     function)](api                | operatorplEdRK15scalar_operator), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[3\]](api/languages/cp      |
| udaq11noise_model11noise_modelEv) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cu                           | operatorplEdRR15scalar_operator), |
| daq::noise_model::PredicateFuncTy |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     type)](api/la                 | alar_operatorplENSt7complexIdEE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[5\]](api/languages/cpp     |
| q11noise_model15PredicateFuncTyE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cud                          | _operatorplERK15scalar_operator), |
| aq::noise_model::register_channel |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function)](api/languages      | 4NKR5cudaq15scalar_operatorplEd), |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     [\[7\]]                       |
| noise_model16register_channelEvv) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::                      | 4NKR5cudaq15scalar_operatorplEv), |
| noise_model::requires_constructor |     [\[8\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     type)](api/languages/cp       | alar_operatorplENSt7complexIdEE), |
| p_api.html#_CPPv4I0DpEN5cudaq11no |     [\[9\]](api/languages/cp      |
| ise_model20requires_constructorE) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cudaq::noise_model_type (C++ | _operatorplERK15scalar_operator), |
|     e                             |     [\[10\]                       |
| num)](api/languages/cpp_api.html# | ](api/languages/cpp_api.html#_CPP |
| _CPPv4N5cudaq16noise_model_typeE) | v4NO5cudaq15scalar_operatorplEd), |
| -   [cudaq::no                    |     [\[11\                        |
| ise_model_type::amplitude_damping | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatorplEv) |
|     enumerator)](api/languages    | -   [c                            |
| /cpp_api.html#_CPPv4N5cudaq16nois | udaq::scalar_operator::operator+= |
| e_model_type17amplitude_dampingE) |     (C++                          |
| -   [cudaq::noise_mode            |     function)](api/languag        |
| l_type::amplitude_damping_channel | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatorpLENSt7complexIdEE), |
|     e                             |     [\[1\]](api/languages/c       |
| numerator)](api/languages/cpp_api | pp_api.html#_CPPv4N5cudaq15scalar |
| .html#_CPPv4N5cudaq16noise_model_ | _operatorpLERK15scalar_operator), |
| type25amplitude_damping_channelE) |     [\[2                          |
| -   [cudaq::n                     | \]](api/languages/cpp_api.html#_C |
| oise_model_type::bit_flip_channel | PPv4N5cudaq15scalar_operatorpLEd) |
|     (C++                          | -   [                             |
|     enumerator)](api/language     | cudaq::scalar_operator::operator- |
| s/cpp_api.html#_CPPv4N5cudaq16noi |     (C++                          |
| se_model_type16bit_flip_channelE) |     function                      |
| -   [cudaq::                      | )](api/languages/cpp_api.html#_CP |
| noise_model_type::depolarization1 | Pv4N5cudaq15scalar_operatormiENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     enumerator)](api/languag      |     [\[1\                         |
| es/cpp_api.html#_CPPv4N5cudaq16no | ]](api/languages/cpp_api.html#_CP |
| ise_model_type15depolarization1E) | Pv4N5cudaq15scalar_operatormiENSt |
| -   [cudaq::                      | 7complexIdEERR15scalar_operator), |
| noise_model_type::depolarization2 |     [\[2\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     enumerator)](api/languag      | operatormiEdRK15scalar_operator), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[3\]](api/languages/cp      |
| ise_model_type15depolarization2E) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::noise_m               | operatormiEdRR15scalar_operator), |
| odel_type::depolarization_channel |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|                                   | alar_operatormiENSt7complexIdEE), |
|   enumerator)](api/languages/cpp_ |     [\[5\]](api/languages/cpp     |
| api.html#_CPPv4N5cudaq16noise_mod | _api.html#_CPPv4NKR5cudaq15scalar |
| el_type22depolarization_channelE) | _operatormiERK15scalar_operator), |
| -                                 |     [\[6\]]                       |
|  [cudaq::noise_model_type::pauli1 | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEd), |
|     enumerator)](a                |     [\[7\]]                       |
| pi/languages/cpp_api.html#_CPPv4N | (api/languages/cpp_api.html#_CPPv |
| 5cudaq16noise_model_type6pauli1E) | 4NKR5cudaq15scalar_operatormiEv), |
| -                                 |     [\[8\]](api/language          |
|  [cudaq::noise_model_type::pauli2 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     enumerator)](a                |     [\[9\]](api/languages/cp      |
| pi/languages/cpp_api.html#_CPPv4N | p_api.html#_CPPv4NO5cudaq15scalar |
| 5cudaq16noise_model_type6pauli2E) | _operatormiERK15scalar_operator), |
| -   [cudaq                        |     [\[10\]                       |
| ::noise_model_type::phase_damping | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NO5cudaq15scalar_operatormiEd), |
|     enumerator)](api/langu        |     [\[11\                        |
| ages/cpp_api.html#_CPPv4N5cudaq16 | ]](api/languages/cpp_api.html#_CP |
| noise_model_type13phase_dampingE) | Pv4NO5cudaq15scalar_operatormiEv) |
| -   [cudaq::noi                   | -   [c                            |
| se_model_type::phase_flip_channel | udaq::scalar_operator::operator-= |
|     (C++                          |     (C++                          |
|     enumerator)](api/languages/   |     function)](api/languag        |
| cpp_api.html#_CPPv4N5cudaq16noise | es/cpp_api.html#_CPPv4N5cudaq15sc |
| _model_type18phase_flip_channelE) | alar_operatormIENSt7complexIdEE), |
| -                                 |     [\[1\]](api/languages/c       |
| [cudaq::noise_model_type::unknown | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatormIERK15scalar_operator), |
|     enumerator)](ap               |     [\[2                          |
| i/languages/cpp_api.html#_CPPv4N5 | \]](api/languages/cpp_api.html#_C |
| cudaq16noise_model_type7unknownE) | PPv4N5cudaq15scalar_operatormIEd) |
| -                                 | -   [                             |
| [cudaq::noise_model_type::x_error | cudaq::scalar_operator::operator/ |
|     (C++                          |     (C++                          |
|     enumerator)](ap               |     function                      |
| i/languages/cpp_api.html#_CPPv4N5 | )](api/languages/cpp_api.html#_CP |
| cudaq16noise_model_type7x_errorE) | Pv4N5cudaq15scalar_operatordvENSt |
| -                                 | 7complexIdEERK15scalar_operator), |
| [cudaq::noise_model_type::y_error |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](ap               | Pv4N5cudaq15scalar_operatordvENSt |
| i/languages/cpp_api.html#_CPPv4N5 | 7complexIdEERR15scalar_operator), |
| cudaq16noise_model_type7y_errorE) |     [\[2\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4N5cudaq15scalar_ |
| [cudaq::noise_model_type::z_error | operatordvEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     enumerator)](ap               | p_api.html#_CPPv4N5cudaq15scalar_ |
| i/languages/cpp_api.html#_CPPv4N5 | operatordvEdRR15scalar_operator), |
| cudaq16noise_model_type7z_errorE) |     [\[4\]](api/languages         |
| -   [cudaq::num_available_gpus    | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatordvENSt7complexIdEE), |
|     function                      |     [\[5\]](api/languages/cpp     |
| )](api/languages/cpp_api.html#_CP | _api.html#_CPPv4NKR5cudaq15scalar |
| Pv4N5cudaq18num_available_gpusEv) | _operatordvERK15scalar_operator), |
| -   [cudaq::observe (C++          |     [\[6\]]                       |
|     function)]                    | (api/languages/cpp_api.html#_CPPv |
| (api/languages/cpp_api.html#_CPPv | 4NKR5cudaq15scalar_operatordvEd), |
| 4I00DpEN5cudaq7observeENSt6vector |     [\[7\]](api/language          |
| I14observe_resultEERR13QuantumKer | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| nelRK15SpinOpContainerDpRR4Args), | alar_operatordvENSt7complexIdEE), |
|     [\[1\]](api/languages/cpp_ap  |     [\[8\]](api/languages/cp      |
| i.html#_CPPv4I0DpEN5cudaq7observe | p_api.html#_CPPv4NO5cudaq15scalar |
| E14observe_resultNSt6size_tERR13Q | _operatordvERK15scalar_operator), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[9\                         |
|     [\[                           | ]](api/languages/cpp_api.html#_CP |
| 2\]](api/languages/cpp_api.html#_ | Pv4NO5cudaq15scalar_operatordvEd) |
| CPPv4I0DpEN5cudaq7observeE14obser | -   [c                            |
| ve_resultRK15observe_optionsRR13Q | udaq::scalar_operator::operator/= |
| uantumKernelRK7spin_opDpRR4Args), |     (C++                          |
|     [\[3\]](api/lang              |     function)](api/languag        |
| uages/cpp_api.html#_CPPv4I0DpEN5c | es/cpp_api.html#_CPPv4N5cudaq15sc |
| udaq7observeE14observe_resultRR13 | alar_operatordVENSt7complexIdEE), |
| QuantumKernelRK7spin_opDpRR4Args) |     [\[1\]](api/languages/c       |
| -   [cudaq::observe_options (C++  | pp_api.html#_CPPv4N5cudaq15scalar |
|     st                            | _operatordVERK15scalar_operator), |
| ruct)](api/languages/cpp_api.html |     [\[2                          |
| #_CPPv4N5cudaq15observe_optionsE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::observe_result (C++   | PPv4N5cudaq15scalar_operatordVEd) |
|                                   | -   [                             |
| class)](api/languages/cpp_api.htm | cudaq::scalar_operator::operator= |
| l#_CPPv4N5cudaq14observe_resultE) |     (C++                          |
| -                                 |     function)](api/languages/c    |
|    [cudaq::observe_result::counts | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatoraSERK15scalar_operator), |
|     function)](api/languages/c    |     [\[1\]](api/languages/        |
| pp_api.html#_CPPv4N5cudaq14observ | cpp_api.html#_CPPv4N5cudaq15scala |
| e_result6countsERK12spin_op_term) | r_operatoraSERR15scalar_operator) |
| -   [cudaq::observe_result::dump  | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator== |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languages/c    |
| v4N5cudaq14observe_result4dumpEv) | pp_api.html#_CPPv4NK5cudaq15scala |
| -   [c                            | r_operatoreqERK15scalar_operator) |
| udaq::observe_result::expectation | -   [cudaq:                       |
|     (C++                          | :scalar_operator::scalar_operator |
|                                   |     (C++                          |
| function)](api/languages/cpp_api. |     func                          |
| html#_CPPv4N5cudaq14observe_resul | tion)](api/languages/cpp_api.html |
| t11expectationERK12spin_op_term), | #_CPPv4N5cudaq15scalar_operator15 |
|     [\[1\]](api/la                | scalar_operatorENSt7complexIdEE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[1\]](api/langu             |
| q14observe_result11expectationEv) | ages/cpp_api.html#_CPPv4N5cudaq15 |
| -   [cuda                         | scalar_operator15scalar_operatorE |
| q::observe_result::id_coefficient | RK15scalar_callbackRRNSt13unorder |
|     (C++                          | ed_mapINSt6stringENSt6stringEEE), |
|     function)](api/langu          |     [\[2\                         |
| ages/cpp_api.html#_CPPv4N5cudaq14 | ]](api/languages/cpp_api.html#_CP |
| observe_result14id_coefficientEv) | Pv4N5cudaq15scalar_operator15scal |
| -   [cuda                         | ar_operatorERK15scalar_operator), |
| q::observe_result::observe_result |     [\[3\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|                                   | scalar_operator15scalar_operatorE |
|   function)](api/languages/cpp_ap | RR15scalar_callbackRRNSt13unorder |
| i.html#_CPPv4N5cudaq14observe_res | ed_mapINSt6stringENSt6stringEEE), |
| ult14observe_resultEdRK7spin_op), |     [\[4\                         |
|     [\[1\]](a                     | ]](api/languages/cpp_api.html#_CP |
| pi/languages/cpp_api.html#_CPPv4N | Pv4N5cudaq15scalar_operator15scal |
| 5cudaq14observe_result14observe_r | ar_operatorERR15scalar_operator), |
| esultEdRK7spin_op13sample_result) |     [\[5\]](api/language          |
| -                                 | s/cpp_api.html#_CPPv4N5cudaq15sca |
|  [cudaq::observe_result::operator | lar_operator15scalar_operatorEd), |
|     double (C++                   |     [\[6\]](api/languag           |
|     functio                       | es/cpp_api.html#_CPPv4N5cudaq15sc |
| n)](api/languages/cpp_api.html#_C | alar_operator15scalar_operatorEv) |
| PPv4N5cudaq14observe_resultcvdEv) | -   [                             |
| -                                 | cudaq::scalar_operator::to_matrix |
|  [cudaq::observe_result::raw_data |     (C++                          |
|     (C++                          |                                   |
|     function)](ap                 |   function)](api/languages/cpp_ap |
| i/languages/cpp_api.html#_CPPv4N5 | i.html#_CPPv4NK5cudaq15scalar_ope |
| cudaq14observe_result8raw_dataEv) | rator9to_matrixERKNSt13unordered_ |
| -   [cudaq::operator_handler (C++ | mapINSt6stringENSt7complexIdEEEE) |
|     cl                            | -   [                             |
| ass)](api/languages/cpp_api.html# | cudaq::scalar_operator::to_string |
| _CPPv4N5cudaq16operator_handlerE) |     (C++                          |
| -   [cudaq::optimizable_function  |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|     class)                        | daq15scalar_operator9to_stringEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::s                     |
| v4N5cudaq20optimizable_functionE) | calar_operator::\~scalar_operator |
| -   [cudaq::optimization_result   |     (C++                          |
|     (C++                          |     functio                       |
|     type                          | n)](api/languages/cpp_api.html#_C |
| )](api/languages/cpp_api.html#_CP | PPv4N5cudaq15scalar_operatorD0Ev) |
| Pv4N5cudaq19optimization_resultE) | -   [cudaq::set_noise (C++        |
| -   [cudaq::optimizer (C++        |     function)](api/langu          |
|     class)](api/languages/cpp_a   | ages/cpp_api.html#_CPPv4N5cudaq9s |
| pi.html#_CPPv4N5cudaq9optimizerE) | et_noiseERKN5cudaq11noise_modelE) |
| -   [cudaq::optimizer::optimize   | -   [cudaq::set_random_seed (C++  |
|     (C++                          |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|  function)](api/languages/cpp_api | daq15set_random_seedENSt6size_tE) |
| .html#_CPPv4N5cudaq9optimizer8opt | -   [cudaq::simulation_precision  |
| imizeEKiRR20optimizable_function) |     (C++                          |
| -   [cu                           |     enum)                         |
| daq::optimizer::requiresGradients | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq20simulation_precisionE) |
|     function)](api/la             | -   [                             |
| nguages/cpp_api.html#_CPPv4N5cuda | cudaq::simulation_precision::fp32 |
| q9optimizer17requiresGradientsEv) |     (C++                          |
| -   [cudaq::orca (C++             |     enumerator)](api              |
|     type)](api/languages/         | /languages/cpp_api.html#_CPPv4N5c |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | udaq20simulation_precision4fp32E) |
| -   [cudaq::orca::sample (C++     | -   [                             |
|     function)](api/languages/c    | cudaq::simulation_precision::fp64 |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     (C++                          |
| mpleERNSt6vectorINSt6size_tEEERNS |     enumerator)](api              |
| t6vectorINSt6size_tEEERNSt6vector | /languages/cpp_api.html#_CPPv4N5c |
| IdEERNSt6vectorIdEEiNSt6size_tE), | udaq20simulation_precision4fp64E) |
|     [\[1\]]                       | -   [cudaq::SimulationState (C++  |
| (api/languages/cpp_api.html#_CPPv |     c                             |
| 4N5cudaq4orca6sampleERNSt6vectorI | lass)](api/languages/cpp_api.html |
| NSt6size_tEEERNSt6vectorINSt6size | #_CPPv4N5cudaq15SimulationStateE) |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | -   [                             |
| -   [cudaq::orca::sample_async    | cudaq::SimulationState::precision |
|     (C++                          |     (C++                          |
|                                   |     enum)](api                    |
| function)](api/languages/cpp_api. | /languages/cpp_api.html#_CPPv4N5c |
| html#_CPPv4N5cudaq4orca12sample_a | udaq15SimulationState9precisionE) |
| syncERNSt6vectorINSt6size_tEEERNS | -   [cudaq:                       |
| t6vectorINSt6size_tEEERNSt6vector | :SimulationState::precision::fp32 |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     (C++                          |
|     [\[1\]](api/la                |     enumerator)](api/lang         |
| nguages/cpp_api.html#_CPPv4N5cuda | uages/cpp_api.html#_CPPv4N5cudaq1 |
| q4orca12sample_asyncERNSt6vectorI | 5SimulationState9precision4fp32E) |
| NSt6size_tEEERNSt6vectorINSt6size | -   [cudaq:                       |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | :SimulationState::precision::fp64 |
| -   [cudaq::OrcaRemoteRESTQPU     |     (C++                          |
|     (C++                          |     enumerator)](api/lang         |
|     cla                           | uages/cpp_api.html#_CPPv4N5cudaq1 |
| ss)](api/languages/cpp_api.html#_ | 5SimulationState9precision4fp64E) |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | -                                 |
| -   [cudaq::pauli1 (C++           |   [cudaq::SimulationState::Tensor |
|     class)](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq6pauli1E) |     struct)](                     |
| -                                 | api/languages/cpp_api.html#_CPPv4 |
|    [cudaq::pauli1::num_parameters | N5cudaq15SimulationState6TensorE) |
|     (C++                          | -   [cudaq::spin_handler (C++     |
|     member)]                      |                                   |
| (api/languages/cpp_api.html#_CPPv |   class)](api/languages/cpp_api.h |
| 4N5cudaq6pauli114num_parametersE) | tml#_CPPv4N5cudaq12spin_handlerE) |
| -   [cudaq::pauli1::num_targets   | -   [cudaq:                       |
|     (C++                          | :spin_handler::to_diagonal_matrix |
|     membe                         |     (C++                          |
| r)](api/languages/cpp_api.html#_C |     function)](api/la             |
| PPv4N5cudaq6pauli111num_targetsE) | nguages/cpp_api.html#_CPPv4NK5cud |
| -   [cudaq::pauli1::pauli1 (C++   | aq12spin_handler18to_diagonal_mat |
|     function)](api/languages/cpp_ | rixERNSt13unordered_mapINSt6size_ |
| api.html#_CPPv4N5cudaq6pauli16pau | tENSt7int64_tEEERKNSt13unordered_ |
| li1ERKNSt6vectorIN5cudaq4realEEE) | mapINSt6stringENSt7complexIdEEEE) |
| -   [cudaq::pauli2 (C++           | -                                 |
|     class)](api/languages/cp      |   [cudaq::spin_handler::to_matrix |
| p_api.html#_CPPv4N5cudaq6pauli2E) |     (C++                          |
| -                                 |     function                      |
|    [cudaq::pauli2::num_parameters | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq12spin_handler9to_matri |
|     member)]                      | xERKNSt6stringENSt7complexIdEEb), |
| (api/languages/cpp_api.html#_CPPv |     [\[1                          |
| 4N5cudaq6pauli214num_parametersE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::pauli2::num_targets   | PPv4NK5cudaq12spin_handler9to_mat |
|     (C++                          | rixERNSt13unordered_mapINSt6size_ |
|     membe                         | tENSt7int64_tEEERKNSt13unordered_ |
| r)](api/languages/cpp_api.html#_C | mapINSt6stringENSt7complexIdEEEE) |
| PPv4N5cudaq6pauli211num_targetsE) | -   [cuda                         |
| -   [cudaq::pauli2::pauli2 (C++   | q::spin_handler::to_sparse_matrix |
|     function)](api/languages/cpp_ |     (C++                          |
| api.html#_CPPv4N5cudaq6pauli26pau |     function)](api/               |
| li2ERKNSt6vectorIN5cudaq4realEEE) | languages/cpp_api.html#_CPPv4N5cu |
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
| -   [define() (cudaq.operators    | -   [deserialize()                |
|     method)](api/languages/python |     (cudaq.SampleResult           |
| _api.html#cudaq.operators.define) |     meth                          |
|     -   [(cuda                    | od)](api/languages/python_api.htm |
| q.operators.MatrixOperatorElement | l#cudaq.SampleResult.deserialize) |
|         class                     | -   [displace() (in module        |
|         method)](api/langu        |     cudaq.operators.custo         |
| ages/python_api.html#cudaq.operat | m)](api/languages/python_api.html |
| ors.MatrixOperatorElement.define) | #cudaq.operators.custom.displace) |
|     -   [(in module               | -   [distribute_terms()           |
|         cudaq.operators.cus       |     (cu                           |
| tom)](api/languages/python_api.ht | daq.operators.boson.BosonOperator |
| ml#cudaq.operators.custom.define) |     method)](api/languages/pyt    |
| -   [degrees                      | hon_api.html#cudaq.operators.boso |
|     (cu                           | n.BosonOperator.distribute_terms) |
| daq.operators.boson.BosonOperator |     -   [(cudaq.                  |
|     property)](api/lang           | operators.fermion.FermionOperator |
| uages/python_api.html#cudaq.opera |                                   |
| tors.boson.BosonOperator.degrees) |    method)](api/languages/python_ |
|     -   [(cudaq.ope               | api.html#cudaq.operators.fermion. |
| rators.boson.BosonOperatorElement | FermionOperator.distribute_terms) |
|                                   |     -                             |
|        property)](api/languages/p |  [(cudaq.operators.MatrixOperator |
| ython_api.html#cudaq.operators.bo |         method)](api/language     |
| son.BosonOperatorElement.degrees) | s/python_api.html#cudaq.operators |
|     -   [(cudaq.                  | .MatrixOperator.distribute_terms) |
| operators.boson.BosonOperatorTerm |     -   [(                        |
|         property)](api/language   | cudaq.operators.spin.SpinOperator |
| s/python_api.html#cudaq.operators |         method)](api/languages/p  |
| .boson.BosonOperatorTerm.degrees) | ython_api.html#cudaq.operators.sp |
|     -   [(cudaq.                  | in.SpinOperator.distribute_terms) |
| operators.fermion.FermionOperator |     -   [(cuda                    |
|         property)](api/language   | q.operators.spin.SpinOperatorTerm |
| s/python_api.html#cudaq.operators |                                   |
| .fermion.FermionOperator.degrees) |      method)](api/languages/pytho |
|     -   [(cudaq.operato           | n_api.html#cudaq.operators.spin.S |
| rs.fermion.FermionOperatorElement | pinOperatorTerm.distribute_terms) |
|                                   | -   [draw() (in module            |
|    property)](api/languages/pytho |     cudaq)](api/lang              |
| n_api.html#cudaq.operators.fermio | uages/python_api.html#cudaq.draw) |
| n.FermionOperatorElement.degrees) | -   [dump() (cudaq.ComplexMatrix  |
|     -   [(cudaq.oper              |                                   |
| ators.fermion.FermionOperatorTerm |   method)](api/languages/python_a |
|                                   | pi.html#cudaq.ComplexMatrix.dump) |
|       property)](api/languages/py |     -   [(cudaq.ObserveResult     |
| thon_api.html#cudaq.operators.fer |                                   |
| mion.FermionOperatorTerm.degrees) |   method)](api/languages/python_a |
|     -                             | pi.html#cudaq.ObserveResult.dump) |
|  [(cudaq.operators.MatrixOperator |     -   [(cu                      |
|         property)](api            | daq.operators.boson.BosonOperator |
| /languages/python_api.html#cudaq. |         method)](api/l            |
| operators.MatrixOperator.degrees) | anguages/python_api.html#cudaq.op |
|     -   [(cuda                    | erators.boson.BosonOperator.dump) |
| q.operators.MatrixOperatorElement |     -   [(cudaq.                  |
|         property)](api/langua     | operators.boson.BosonOperatorTerm |
| ges/python_api.html#cudaq.operato |         method)](api/langu        |
| rs.MatrixOperatorElement.degrees) | ages/python_api.html#cudaq.operat |
|     -   [(c                       | ors.boson.BosonOperatorTerm.dump) |
| udaq.operators.MatrixOperatorTerm |     -   [(cudaq.                  |
|         property)](api/lan        | operators.fermion.FermionOperator |
| guages/python_api.html#cudaq.oper |         method)](api/langu        |
| ators.MatrixOperatorTerm.degrees) | ages/python_api.html#cudaq.operat |
|     -   [(                        | ors.fermion.FermionOperator.dump) |
| cudaq.operators.spin.SpinOperator |     -   [(cudaq.oper              |
|         property)](api/la         | ators.fermion.FermionOperatorTerm |
| nguages/python_api.html#cudaq.ope |         method)](api/languages    |
| rators.spin.SpinOperator.degrees) | /python_api.html#cudaq.operators. |
|     -   [(cudaq.o                 | fermion.FermionOperatorTerm.dump) |
| perators.spin.SpinOperatorElement |     -                             |
|         property)](api/languages  |  [(cudaq.operators.MatrixOperator |
| /python_api.html#cudaq.operators. |         method)](                 |
| spin.SpinOperatorElement.degrees) | api/languages/python_api.html#cud |
|     -   [(cuda                    | aq.operators.MatrixOperator.dump) |
| q.operators.spin.SpinOperatorTerm |     -   [(c                       |
|         property)](api/langua     | udaq.operators.MatrixOperatorTerm |
| ges/python_api.html#cudaq.operato |         method)](api/             |
| rs.spin.SpinOperatorTerm.degrees) | languages/python_api.html#cudaq.o |
| -                                 | perators.MatrixOperatorTerm.dump) |
|  [delete_cache_execution_engine() |     -   [(                        |
|     (cudaq.PyKernelDecorator      | cudaq.operators.spin.SpinOperator |
|     method)](api/languages/pyth   |         method)](api              |
| on_api.html#cudaq.PyKernelDecorat | /languages/python_api.html#cudaq. |
| or.delete_cache_execution_engine) | operators.spin.SpinOperator.dump) |
| -   [Depolarization1 (class in    |     -   [(cuda                    |
|     cudaq)](api/languages/pytho   | q.operators.spin.SpinOperatorTerm |
| n_api.html#cudaq.Depolarization1) |         method)](api/lan          |
| -   [Depolarization2 (class in    | guages/python_api.html#cudaq.oper |
|     cudaq)](api/languages/pytho   | ators.spin.SpinOperatorTerm.dump) |
| n_api.html#cudaq.Depolarization2) |     -   [(cudaq.Resources         |
| -   [DepolarizationChannel (class |                                   |
|     in                            |       method)](api/languages/pyth |
|                                   | on_api.html#cudaq.Resources.dump) |
| cudaq)](api/languages/python_api. |     -   [(cudaq.SampleResult      |
| html#cudaq.DepolarizationChannel) |                                   |
| -   [description (cudaq.Target    |    method)](api/languages/python_ |
|                                   | api.html#cudaq.SampleResult.dump) |
| property)](api/languages/python_a |     -   [(cudaq.State             |
| pi.html#cudaq.Target.description) |         method)](api/languages/   |
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
| stom)](api/languages/python_api.h | -   [execution_data()             |
| tml#cudaq.operators.custom.empty) |     (cudaq.operators.cus          |
|     -   [(in module               | tom.cudaq.ptsbe.PTSBESampleResult |
|                                   |     method)]                      |
|       cudaq.spin)](api/languages/ | (api/languages/python_api.html#cu |
| python_api.html#cudaq.spin.empty) | daq.operators.custom.cudaq.ptsbe. |
| -   [empty_op()                   | PTSBESampleResult.execution_data) |
|     (                             | -   [expectation()                |
| cudaq.operators.spin.SpinOperator |     (cudaq.ObserveResult          |
|     static                        |     metho                         |
|     method)](api/lan              | d)](api/languages/python_api.html |
| guages/python_api.html#cudaq.oper | #cudaq.ObserveResult.expectation) |
| ators.spin.SpinOperator.empty_op) |     -   [(cudaq.SampleResult      |
| -   [enable_return_to_log()       |         meth                      |
|     (cudaq.PyKernelDecorator      | od)](api/languages/python_api.htm |
|     method)](api/langu            | l#cudaq.SampleResult.expectation) |
| ages/python_api.html#cudaq.PyKern | -   [expectation_values()         |
| elDecorator.enable_return_to_log) |     (cudaq.EvolveResult           |
| -   [epsilon                      |     method)](ap                   |
|     (cudaq.optimizers.Adam        | i/languages/python_api.html#cudaq |
|     prope                         | .EvolveResult.expectation_values) |
| rty)](api/languages/python_api.ht | -   [expectation_z()              |
| ml#cudaq.optimizers.Adam.epsilon) |     (cudaq.SampleResult           |
| -   [estimate_resources() (in     |     method                        |
|     module                        | )](api/languages/python_api.html# |
|                                   | cudaq.SampleResult.expectation_z) |
|    cudaq)](api/languages/python_a | -   [expected_dimensions          |
| pi.html#cudaq.estimate_resources) |     (cuda                         |
|                                   | q.operators.MatrixOperatorElement |
|                                   |                                   |
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
| -   [gamma (cudaq.optimizers.SPSA | -   [get_raw_data()               |
|     pro                           |     (                             |
| perty)](api/languages/python_api. | cudaq.operators.spin.SpinOperator |
| html#cudaq.optimizers.SPSA.gamma) |     method)](api/languag          |
| -   [Gate                         | es/python_api.html#cudaq.operator |
|     (cudaq.operators.custom       | s.spin.SpinOperator.get_raw_data) |
| .cudaq.ptsbe.TraceInstructionType |     -   [(cuda                    |
|     attr                          | q.operators.spin.SpinOperatorTerm |
| ibute)](api/languages/python_api. |         method)](api/languages/p  |
| html#cudaq.operators.custom.cudaq | ython_api.html#cudaq.operators.sp |
| .ptsbe.TraceInstructionType.Gate) | in.SpinOperatorTerm.get_raw_data) |
| -   [generate_trajectories()      | -   [get_register_counts()        |
|     (cudaq.operators.custo        |     (cudaq.SampleResult           |
| m.cudaq.ptsbe.PTSSamplingStrategy |     method)](api                  |
|     method)](api/lang             | /languages/python_api.html#cudaq. |
| uages/python_api.html#cudaq.opera | SampleResult.get_register_counts) |
| tors.custom.cudaq.ptsbe.PTSSampli | -   [get_sequential_data()        |
| ngStrategy.generate_trajectories) |     (cudaq.SampleResult           |
| -   [get()                        |     method)](api                  |
|     (cudaq.AsyncEvolveResult      | /languages/python_api.html#cudaq. |
|     m                             | SampleResult.get_sequential_data) |
| ethod)](api/languages/python_api. | -   [get_spin()                   |
| html#cudaq.AsyncEvolveResult.get) |     (cudaq.ObserveResult          |
|                                   |     me                            |
|    -   [(cudaq.AsyncObserveResult | thod)](api/languages/python_api.h |
|         me                        | tml#cudaq.ObserveResult.get_spin) |
| thod)](api/languages/python_api.h | -   [get_state() (in module       |
| tml#cudaq.AsyncObserveResult.get) |     cudaq)](api/languages         |
|     -   [(cudaq.AsyncStateResult  | /python_api.html#cudaq.get_state) |
|                                   | -   [get_state_async() (in module |
| method)](api/languages/python_api |     cudaq)](api/languages/pytho   |
| .html#cudaq.AsyncStateResult.get) | n_api.html#cudaq.get_state_async) |
| -   [get_binary_symplectic_form() | -   [get_state_refval()           |
|     (cuda                         |     (cudaq.State                  |
| q.operators.spin.SpinOperatorTerm |     me                            |
|     metho                         | thod)](api/languages/python_api.h |
| d)](api/languages/python_api.html | tml#cudaq.State.get_state_refval) |
| #cudaq.operators.spin.SpinOperato | -   [get_target() (in module      |
| rTerm.get_binary_symplectic_form) |     cudaq)](api/languages/        |
| -   [get_channels()               | python_api.html#cudaq.get_target) |
|     (cudaq.NoiseModel             | -   [get_targets() (in module     |
|     met                           |     cudaq)](api/languages/p       |
| hod)](api/languages/python_api.ht | ython_api.html#cudaq.get_targets) |
| ml#cudaq.NoiseModel.get_channels) | -   [get_term_count()             |
| -   [get_coefficient()            |     (                             |
|     (                             | cudaq.operators.spin.SpinOperator |
| cudaq.operators.spin.SpinOperator |     method)](api/languages        |
|     method)](api/languages/       | /python_api.html#cudaq.operators. |
| python_api.html#cudaq.operators.s | spin.SpinOperator.get_term_count) |
| pin.SpinOperator.get_coefficient) | -   [get_total_shots()            |
|     -   [(cuda                    |     (cudaq.SampleResult           |
| q.operators.spin.SpinOperatorTerm |     method)]                      |
|                                   | (api/languages/python_api.html#cu |
|       method)](api/languages/pyth | daq.SampleResult.get_total_shots) |
| on_api.html#cudaq.operators.spin. | -   [get_trajectory()             |
| SpinOperatorTerm.get_coefficient) |     (cudaq.operators.cust         |
| -   [get_marginal_counts()        | om.cudaq.ptsbe.PTSBEExecutionData |
|     (cudaq.SampleResult           |     method)](                     |
|     method)](api                  | api/languages/python_api.html#cud |
| /languages/python_api.html#cudaq. | aq.operators.custom.cudaq.ptsbe.P |
| SampleResult.get_marginal_counts) | TSBEExecutionData.get_trajectory) |
| -   [get_ops()                    | -   [getTensor() (cudaq.State     |
|     (cudaq.KrausChannel           |     method)](api/languages/pytho  |
|                                   | n_api.html#cudaq.State.getTensor) |
| method)](api/languages/python_api | -   [getTensors() (cudaq.State    |
| .html#cudaq.KrausChannel.get_ops) |     method)](api/languages/python |
| -   [get_pauli_word()             | _api.html#cudaq.State.getTensors) |
|     (cuda                         | -   [gradient (class in           |
| q.operators.spin.SpinOperatorTerm |     cudaq.g                       |
|     method)](api/languages/pyt    | radients)](api/languages/python_a |
| hon_api.html#cudaq.operators.spin | pi.html#cudaq.gradients.gradient) |
| .SpinOperatorTerm.get_pauli_word) | -   [GradientDescent (class in    |
| -   [get_precision()              |     cudaq.optimizers              |
|     (cudaq.Target                 | )](api/languages/python_api.html# |
|                                   | cudaq.optimizers.GradientDescent) |
| method)](api/languages/python_api |                                   |
| .html#cudaq.Target.get_precision) |                                   |
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
+-----------------------------------+-----------------------------------+

## H {#H}

+-----------------------------------+-----------------------------------+
| -   [handle_call_arguments()      | -   [has_target() (in module      |
|     (cudaq.PyKernelDecorator      |     cudaq)](api/languages/        |
|     method)](api/langua           | python_api.html#cudaq.has_target) |
| ges/python_api.html#cudaq.PyKerne | -   [HIGH_WEIGHT_BIAS             |
| lDecorator.handle_call_arguments) |                                   |
| -   [has_execution_data()         |    (cudaq.operators.custom.cudaq. |
|     (cudaq.operators.cus          | ptsbe.ShotAllocationStrategy.Type |
| tom.cudaq.ptsbe.PTSBESampleResult |     attribute)](api/languag       |
|     method)](api                  | es/python_api.html#cudaq.operator |
| /languages/python_api.html#cudaq. | s.custom.cudaq.ptsbe.ShotAllocati |
| operators.custom.cudaq.ptsbe.PTSB | onStrategy.Type.HIGH_WEIGHT_BIAS) |
| ESampleResult.has_execution_data) |                                   |
+-----------------------------------+-----------------------------------+

## I {#I}

+-----------------------------------+-----------------------------------+
| -   [i() (in module               | -   [initialize_cudaq() (in       |
|     cudaq.spin)](api/langua       |     module                        |
| ges/python_api.html#cudaq.spin.i) |     cudaq)](api/languages/python  |
| -   [id                           | _api.html#cudaq.initialize_cudaq) |
|     (cuda                         | -   [InitialState (in module      |
| q.operators.MatrixOperatorElement |     cudaq.dynamics.helpers)](     |
|     property)](api/l              | api/languages/python_api.html#cud |
| anguages/python_api.html#cudaq.op | aq.dynamics.helpers.InitialState) |
| erators.MatrixOperatorElement.id) | -   [InitialStateType (class in   |
| -   [identities() (in module      |     cudaq)](api/languages/python  |
|     c                             | _api.html#cudaq.InitialStateType) |
| udaq.boson)](api/languages/python | -   [instantiate()                |
| _api.html#cudaq.boson.identities) |     (cudaq.operators              |
|     -   [(in module               |     m                             |
|         cudaq                     | ethod)](api/languages/python_api. |
| .fermion)](api/languages/python_a | html#cudaq.operators.instantiate) |
| pi.html#cudaq.fermion.identities) |     -   [(in module               |
|     -   [(in module               |         cudaq.operators.custom)]  |
|         cudaq.operators.custom)   | (api/languages/python_api.html#cu |
| ](api/languages/python_api.html#c | daq.operators.custom.instantiate) |
| udaq.operators.custom.identities) | -   [instructions                 |
|     -   [(in module               |     (cudaq.operators.cust         |
|                                   | om.cudaq.ptsbe.PTSBEExecutionData |
|  cudaq.spin)](api/languages/pytho |     attribute)                    |
| n_api.html#cudaq.spin.identities) | ](api/languages/python_api.html#c |
| -   [identity()                   | udaq.operators.custom.cudaq.ptsbe |
|     (cu                           | .PTSBEExecutionData.instructions) |
| daq.operators.boson.BosonOperator | -   [intermediate_states()        |
|     static                        |     (cudaq.EvolveResult           |
|     method)](api/langu            |     method)](api                  |
| ages/python_api.html#cudaq.operat | /languages/python_api.html#cudaq. |
| ors.boson.BosonOperator.identity) | EvolveResult.intermediate_states) |
|     -   [(cudaq.                  | -   [IntermediateResultSave       |
| operators.fermion.FermionOperator |     (class in                     |
|         static                    |     c                             |
|         method)](api/languages    | udaq)](api/languages/python_api.h |
| /python_api.html#cudaq.operators. | tml#cudaq.IntermediateResultSave) |
| fermion.FermionOperator.identity) | -   [is_compiled()                |
|     -                             |     (cudaq.PyKernelDecorator      |
|  [(cudaq.operators.MatrixOperator |     method)](                     |
|         static                    | api/languages/python_api.html#cud |
|         method)](api/             | aq.PyKernelDecorator.is_compiled) |
| languages/python_api.html#cudaq.o | -   [is_constant()                |
| perators.MatrixOperator.identity) |                                   |
|     -   [(                        |   (cudaq.operators.ScalarOperator |
| cudaq.operators.spin.SpinOperator |     method)](api/lan              |
|         static                    | guages/python_api.html#cudaq.oper |
|         method)](api/lan          | ators.ScalarOperator.is_constant) |
| guages/python_api.html#cudaq.oper | -   [is_emulated() (cudaq.Target  |
| ators.spin.SpinOperator.identity) |                                   |
|     -   [(in module               |   method)](api/languages/python_a |
|                                   | pi.html#cudaq.Target.is_emulated) |
|  cudaq.boson)](api/languages/pyth | -   [is_error                     |
| on_api.html#cudaq.boson.identity) |     (cudaq.operators.             |
|     -   [(in module               | custom.cudaq.ptsbe.KrausSelection |
|         cud                       |     at                            |
| aq.fermion)](api/languages/python | tribute)](api/languages/python_ap |
| _api.html#cudaq.fermion.identity) | i.html#cudaq.operators.custom.cud |
|     -   [(in module               | aq.ptsbe.KrausSelection.is_error) |
|                                   | -   [is_identity()                |
|    cudaq.spin)](api/languages/pyt |     (cudaq.                       |
| hon_api.html#cudaq.spin.identity) | operators.boson.BosonOperatorTerm |
| -   [initial_parameters           |     method)](api/languages/py     |
|     (cudaq.optimizers.Adam        | thon_api.html#cudaq.operators.bos |
|     property)](api/l              | on.BosonOperatorTerm.is_identity) |
| anguages/python_api.html#cudaq.op |     -   [(cudaq.oper              |
| timizers.Adam.initial_parameters) | ators.fermion.FermionOperatorTerm |
|     -   [(cudaq.optimizers.COBYLA |                                   |
|         property)](api/lan        |     method)](api/languages/python |
| guages/python_api.html#cudaq.opti | _api.html#cudaq.operators.fermion |
| mizers.COBYLA.initial_parameters) | .FermionOperatorTerm.is_identity) |
|     -   [                         |     -   [(c                       |
| (cudaq.optimizers.GradientDescent | udaq.operators.MatrixOperatorTerm |
|                                   |         method)](api/languag      |
|       property)](api/languages/py | es/python_api.html#cudaq.operator |
| thon_api.html#cudaq.optimizers.Gr | s.MatrixOperatorTerm.is_identity) |
| adientDescent.initial_parameters) |     -   [(                        |
|     -   [(cudaq.optimizers.LBFGS  | cudaq.operators.spin.SpinOperator |
|         property)](api/la         |         method)](api/langua       |
| nguages/python_api.html#cudaq.opt | ges/python_api.html#cudaq.operato |
| imizers.LBFGS.initial_parameters) | rs.spin.SpinOperator.is_identity) |
|                                   |     -   [(cuda                    |
| -   [(cudaq.optimizers.NelderMead | q.operators.spin.SpinOperatorTerm |
|         property)](api/languag    |         method)](api/languages/   |
| es/python_api.html#cudaq.optimize | python_api.html#cudaq.operators.s |
| rs.NelderMead.initial_parameters) | pin.SpinOperatorTerm.is_identity) |
|     -   [(cudaq.optimizers.SGD    | -   [is_initialized() (in module  |
|         property)](api/           |     c                             |
| languages/python_api.html#cudaq.o | udaq.mpi)](api/languages/python_a |
| ptimizers.SGD.initial_parameters) | pi.html#cudaq.mpi.is_initialized) |
|     -   [(cudaq.optimizers.SPSA   | -   [is_on_gpu() (cudaq.State     |
|         property)](api/l          |     method)](api/languages/pytho  |
| anguages/python_api.html#cudaq.op | n_api.html#cudaq.State.is_on_gpu) |
| timizers.SPSA.initial_parameters) | -   [is_remote() (cudaq.Target    |
| -   [initialize() (in module      |     method)](api/languages/python |
|                                   | _api.html#cudaq.Target.is_remote) |
|    cudaq.mpi)](api/languages/pyth | -   [is_remote_simulator()        |
| on_api.html#cudaq.mpi.initialize) |     (cudaq.Target                 |
|                                   |     method                        |
|                                   | )](api/languages/python_api.html# |
|                                   | cudaq.Target.is_remote_simulator) |
|                                   | -   [items() (cudaq.SampleResult  |
|                                   |                                   |
|                                   |   method)](api/languages/python_a |
|                                   | pi.html#cudaq.SampleResult.items) |
+-----------------------------------+-----------------------------------+

## K {#K}

+-----------------------------------+-----------------------------------+
| -   [Kernel (in module            | -   [kraus_selections             |
|     cudaq)](api/langua            |     (cudaq.operators.c            |
| ges/python_api.html#cudaq.Kernel) | ustom.cudaq.ptsbe.KrausTrajectory |
| -   [kernel() (in module          |     attribute)]                   |
|     cudaq)](api/langua            | (api/languages/python_api.html#cu |
| ges/python_api.html#cudaq.kernel) | daq.operators.custom.cudaq.ptsbe. |
| -   [kraus_operator_index         | KrausTrajectory.kraus_selections) |
|     (cudaq.operators.             | -   [KrausChannel (class in       |
| custom.cudaq.ptsbe.KrausSelection |     cudaq)](api/languages/py      |
|     attribute)](ap                | thon_api.html#cudaq.KrausChannel) |
| i/languages/python_api.html#cudaq | -   [KrausOperator (class in      |
| .operators.custom.cudaq.ptsbe.Kra |     cudaq)](api/languages/pyt     |
| usSelection.kraus_operator_index) | hon_api.html#cudaq.KrausOperator) |
+-----------------------------------+-----------------------------------+

## L {#L}

+-----------------------------------+-----------------------------------+
| -   [launch_args_required()       | -   [lower_quake_to_codegen()     |
|     (cudaq.PyKernelDecorator      |     (cudaq.PyKernelDecorator      |
|     method)](api/langu            |     method)](api/languag          |
| ages/python_api.html#cudaq.PyKern | es/python_api.html#cudaq.PyKernel |
| elDecorator.launch_args_required) | Decorator.lower_quake_to_codegen) |
| -   [LBFGS (class in              |                                   |
|     cudaq.                        |                                   |
| optimizers)](api/languages/python |                                   |
| _api.html#cudaq.optimizers.LBFGS) |                                   |
| -   [left_multiply()              |                                   |
|     (cudaq.SuperOperator static   |                                   |
|     method)                       |                                   |
| ](api/languages/python_api.html#c |                                   |
| udaq.SuperOperator.left_multiply) |                                   |
| -   [left_right_multiply()        |                                   |
|     (cudaq.SuperOperator static   |                                   |
|     method)](api/                 |                                   |
| languages/python_api.html#cudaq.S |                                   |
| uperOperator.left_right_multiply) |                                   |
| -   [LOW_WEIGHT_BIAS              |                                   |
|                                   |                                   |
|    (cudaq.operators.custom.cudaq. |                                   |
| ptsbe.ShotAllocationStrategy.Type |                                   |
|     attribute)](api/langua        |                                   |
| ges/python_api.html#cudaq.operato |                                   |
| rs.custom.cudaq.ptsbe.ShotAllocat |                                   |
| ionStrategy.Type.LOW_WEIGHT_BIAS) |                                   |
| -   [lower_bounds                 |                                   |
|     (cudaq.optimizers.Adam        |                                   |
|     property)]                    |                                   |
| (api/languages/python_api.html#cu |                                   |
| daq.optimizers.Adam.lower_bounds) |                                   |
|     -   [(cudaq.optimizers.COBYLA |                                   |
|         property)](a              |                                   |
| pi/languages/python_api.html#cuda |                                   |
| q.optimizers.COBYLA.lower_bounds) |                                   |
|     -   [                         |                                   |
| (cudaq.optimizers.GradientDescent |                                   |
|         property)](api/langua     |                                   |
| ges/python_api.html#cudaq.optimiz |                                   |
| ers.GradientDescent.lower_bounds) |                                   |
|     -   [(cudaq.optimizers.LBFGS  |                                   |
|         property)](               |                                   |
| api/languages/python_api.html#cud |                                   |
| aq.optimizers.LBFGS.lower_bounds) |                                   |
|                                   |                                   |
| -   [(cudaq.optimizers.NelderMead |                                   |
|         property)](api/l          |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| timizers.NelderMead.lower_bounds) |                                   |
|     -   [(cudaq.optimizers.SGD    |                                   |
|         property)                 |                                   |
| ](api/languages/python_api.html#c |                                   |
| udaq.optimizers.SGD.lower_bounds) |                                   |
|     -   [(cudaq.optimizers.SPSA   |                                   |
|         property)]                |                                   |
| (api/languages/python_api.html#cu |                                   |
| daq.optimizers.SPSA.lower_bounds) |                                   |
+-----------------------------------+-----------------------------------+

## M {#M}

+-----------------------------------+-----------------------------------+
| -   [make_kernel() (in module     | -   [merge_kernel()               |
|     cudaq)](api/languages/p       |     (cudaq.PyKernelDecorator      |
| ython_api.html#cudaq.make_kernel) |     method)](a                    |
| -   [MatrixOperator (class in     | pi/languages/python_api.html#cuda |
|     cudaq.operato                 | q.PyKernelDecorator.merge_kernel) |
| rs)](api/languages/python_api.htm | -   [merge_quake_source()         |
| l#cudaq.operators.MatrixOperator) |     (cudaq.PyKernelDecorator      |
| -   [MatrixOperatorElement (class |     method)](api/lan              |
|     in                            | guages/python_api.html#cudaq.PyKe |
|     cudaq.operators)](ap          | rnelDecorator.merge_quake_source) |
| i/languages/python_api.html#cudaq | -   [min_degree                   |
| .operators.MatrixOperatorElement) |     (cu                           |
| -   [MatrixOperatorTerm (class in | daq.operators.boson.BosonOperator |
|     cudaq.operators)]             |     property)](api/languag        |
| (api/languages/python_api.html#cu | es/python_api.html#cudaq.operator |
| daq.operators.MatrixOperatorTerm) | s.boson.BosonOperator.min_degree) |
| -   [max_degree                   |     -   [(cudaq.                  |
|     (cu                           | operators.boson.BosonOperatorTerm |
| daq.operators.boson.BosonOperator |                                   |
|     property)](api/languag        |        property)](api/languages/p |
| es/python_api.html#cudaq.operator | ython_api.html#cudaq.operators.bo |
| s.boson.BosonOperator.max_degree) | son.BosonOperatorTerm.min_degree) |
|     -   [(cudaq.                  |     -   [(cudaq.                  |
| operators.boson.BosonOperatorTerm | operators.fermion.FermionOperator |
|                                   |                                   |
|        property)](api/languages/p |        property)](api/languages/p |
| ython_api.html#cudaq.operators.bo | ython_api.html#cudaq.operators.fe |
| son.BosonOperatorTerm.max_degree) | rmion.FermionOperator.min_degree) |
|     -   [(cudaq.                  |     -   [(cudaq.oper              |
| operators.fermion.FermionOperator | ators.fermion.FermionOperatorTerm |
|                                   |                                   |
|        property)](api/languages/p |    property)](api/languages/pytho |
| ython_api.html#cudaq.operators.fe | n_api.html#cudaq.operators.fermio |
| rmion.FermionOperator.max_degree) | n.FermionOperatorTerm.min_degree) |
|     -   [(cudaq.oper              |     -                             |
| ators.fermion.FermionOperatorTerm |  [(cudaq.operators.MatrixOperator |
|                                   |         property)](api/la         |
|    property)](api/languages/pytho | nguages/python_api.html#cudaq.ope |
| n_api.html#cudaq.operators.fermio | rators.MatrixOperator.min_degree) |
| n.FermionOperatorTerm.max_degree) |     -   [(c                       |
|     -                             | udaq.operators.MatrixOperatorTerm |
|  [(cudaq.operators.MatrixOperator |         property)](api/langua     |
|         property)](api/la         | ges/python_api.html#cudaq.operato |
| nguages/python_api.html#cudaq.ope | rs.MatrixOperatorTerm.min_degree) |
| rators.MatrixOperator.max_degree) |     -   [(                        |
|     -   [(c                       | cudaq.operators.spin.SpinOperator |
| udaq.operators.MatrixOperatorTerm |         property)](api/langu      |
|         property)](api/langua     | ages/python_api.html#cudaq.operat |
| ges/python_api.html#cudaq.operato | ors.spin.SpinOperator.min_degree) |
| rs.MatrixOperatorTerm.max_degree) |     -   [(cuda                    |
|     -   [(                        | q.operators.spin.SpinOperatorTerm |
| cudaq.operators.spin.SpinOperator |         property)](api/languages  |
|         property)](api/langu      | /python_api.html#cudaq.operators. |
| ages/python_api.html#cudaq.operat | spin.SpinOperatorTerm.min_degree) |
| ors.spin.SpinOperator.max_degree) | -   [minimal_eigenvalue()         |
|     -   [(cuda                    |     (cudaq.ComplexMatrix          |
| q.operators.spin.SpinOperatorTerm |     method)](api                  |
|         property)](api/languages  | /languages/python_api.html#cudaq. |
| /python_api.html#cudaq.operators. | ComplexMatrix.minimal_eigenvalue) |
| spin.SpinOperatorTerm.max_degree) | -   [minus() (in module           |
| -   [max_iterations               |     cudaq.spin)](api/languages/   |
|     (cudaq.optimizers.Adam        | python_api.html#cudaq.spin.minus) |
|     property)](a                  | -   module                        |
| pi/languages/python_api.html#cuda |     -   [cudaq](api/langua        |
| q.optimizers.Adam.max_iterations) | ges/python_api.html#module-cudaq) |
|     -   [(cudaq.optimizers.COBYLA |     -                             |
|         property)](api            |    [cudaq.boson](api/languages/py |
| /languages/python_api.html#cudaq. | thon_api.html#module-cudaq.boson) |
| optimizers.COBYLA.max_iterations) |     -   [                         |
|     -   [                         | cudaq.fermion](api/languages/pyth |
| (cudaq.optimizers.GradientDescent | on_api.html#module-cudaq.fermion) |
|         property)](api/language   |     -   [cudaq.operators.cu       |
| s/python_api.html#cudaq.optimizer | stom](api/languages/python_api.ht |
| s.GradientDescent.max_iterations) | ml#module-cudaq.operators.custom) |
|     -   [(cudaq.optimizers.LBFGS  |                                   |
|         property)](ap             |  -   [cudaq.spin](api/languages/p |
| i/languages/python_api.html#cudaq | ython_api.html#module-cudaq.spin) |
| .optimizers.LBFGS.max_iterations) | -   [momentum() (in module        |
|                                   |                                   |
| -   [(cudaq.optimizers.NelderMead |  cudaq.boson)](api/languages/pyth |
|         property)](api/lan        | on_api.html#cudaq.boson.momentum) |
| guages/python_api.html#cudaq.opti |     -   [(in module               |
| mizers.NelderMead.max_iterations) |         cudaq.operators.custo     |
|     -   [(cudaq.optimizers.SGD    | m)](api/languages/python_api.html |
|         property)](               | #cudaq.operators.custom.momentum) |
| api/languages/python_api.html#cud | -   [most_probable()              |
| aq.optimizers.SGD.max_iterations) |     (cudaq.SampleResult           |
|     -   [(cudaq.optimizers.SPSA   |     method                        |
|         property)](a              | )](api/languages/python_api.html# |
| pi/languages/python_api.html#cuda | cudaq.SampleResult.most_probable) |
| q.optimizers.SPSA.max_iterations) | -   [multiplicity                 |
| -   [mdiag_sparse_matrix (C++     |     (cudaq.operators.c            |
|     type)](api/languages/cpp_api. | ustom.cudaq.ptsbe.KrausTrajectory |
| html#_CPPv419mdiag_sparse_matrix) |     attribu                       |
| -   [Measurement                  | te)](api/languages/python_api.htm |
|     (cudaq.operators.custom       | l#cudaq.operators.custom.cudaq.pt |
| .cudaq.ptsbe.TraceInstructionType | sbe.KrausTrajectory.multiplicity) |
|     attribute)]                   |                                   |
| (api/languages/python_api.html#cu |                                   |
| daq.operators.custom.cudaq.ptsbe. |                                   |
| TraceInstructionType.Measurement) |                                   |
+-----------------------------------+-----------------------------------+

## N {#N}

+-----------------------------------+-----------------------------------+
| -   [name                         | -   [num_columns()                |
|     (cudaq.operators.cu           |     (cudaq.ComplexMatrix          |
| stom.cudaq.ptsbe.TraceInstruction |     metho                         |
|                                   | d)](api/languages/python_api.html |
| attribute)](api/languages/python_ | #cudaq.ComplexMatrix.num_columns) |
| api.html#cudaq.operators.custom.c | -   [num_qpus() (cudaq.Target     |
| udaq.ptsbe.TraceInstruction.name) |     method)](api/languages/pytho  |
|     -   [(cudaq.PyKernel          | n_api.html#cudaq.Target.num_qpus) |
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
| n_api.html#cudaq.spin.Pauli.name) | -   [num_shots                    |
|     -   [(cudaq.Target            |     (cudaq.operators.c            |
|                                   | ustom.cudaq.ptsbe.KrausTrajectory |
|        property)](api/languages/p |     attr                          |
| ython_api.html#cudaq.Target.name) | ibute)](api/languages/python_api. |
| -   [name()                       | html#cudaq.operators.custom.cudaq |
|     (cudaq.operators.custo        | .ptsbe.KrausTrajectory.num_shots) |
| m.cudaq.ptsbe.PTSSamplingStrategy | -   [number() (in module          |
|                                   |                                   |
| method)](api/languages/python_api |    cudaq.boson)](api/languages/py |
| .html#cudaq.operators.custom.cuda | thon_api.html#cudaq.boson.number) |
| q.ptsbe.PTSSamplingStrategy.name) |     -   [(in module               |
| -   [NelderMead (class in         |         c                         |
|     cudaq.optim                   | udaq.fermion)](api/languages/pyth |
| izers)](api/languages/python_api. | on_api.html#cudaq.fermion.number) |
| html#cudaq.optimizers.NelderMead) |     -   [(in module               |
| -   [Noise                        |         cudaq.operators.cus       |
|     (cudaq.operators.custom       | tom)](api/languages/python_api.ht |
| .cudaq.ptsbe.TraceInstructionType | ml#cudaq.operators.custom.number) |
|     attri                         | -   [nvqir::MPSSimulationState    |
| bute)](api/languages/python_api.h |     (C++                          |
| tml#cudaq.operators.custom.cudaq. |     class)]                       |
| ptsbe.TraceInstructionType.Noise) | (api/languages/cpp_api.html#_CPPv |
| -   [NoiseModel (class in         | 4I0EN5nvqir18MPSSimulationStateE) |
|     cudaq)](api/languages/        | -                                 |
| python_api.html#cudaq.NoiseModel) |  [nvqir::TensorNetSimulationState |
| -   [num_available_gpus() (in     |     (C++                          |
|     module                        |     class)](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4I0EN5 |
|    cudaq)](api/languages/python_a | nvqir24TensorNetSimulationStateE) |
| pi.html#cudaq.num_available_gpus) |                                   |
+-----------------------------------+-----------------------------------+

## O {#O}

+-----------------------------------+-----------------------------------+
| -   [observe() (in module         | -   [ops_count                    |
|     cudaq)](api/languag           |     (cudaq.                       |
| es/python_api.html#cudaq.observe) | operators.boson.BosonOperatorTerm |
| -   [observe_async() (in module   |     property)](api/languages/     |
|     cudaq)](api/languages/pyt     | python_api.html#cudaq.operators.b |
| hon_api.html#cudaq.observe_async) | oson.BosonOperatorTerm.ops_count) |
| -   [ObserveResult (class in      |     -   [(cudaq.oper              |
|     cudaq)](api/languages/pyt     | ators.fermion.FermionOperatorTerm |
| hon_api.html#cudaq.ObserveResult) |                                   |
| -   [op_name                      |     property)](api/languages/pyth |
|     (cudaq.operators.             | on_api.html#cudaq.operators.fermi |
| custom.cudaq.ptsbe.KrausSelection | on.FermionOperatorTerm.ops_count) |
|     a                             |     -   [(c                       |
| ttribute)](api/languages/python_a | udaq.operators.MatrixOperatorTerm |
| pi.html#cudaq.operators.custom.cu |         property)](api/langu      |
| daq.ptsbe.KrausSelection.op_name) | ages/python_api.html#cudaq.operat |
| -   [OperatorSum (in module       | ors.MatrixOperatorTerm.ops_count) |
|     cudaq.oper                    |     -   [(cuda                    |
| ators)](api/languages/python_api. | q.operators.spin.SpinOperatorTerm |
| html#cudaq.operators.OperatorSum) |         property)](api/language   |
|                                   | s/python_api.html#cudaq.operators |
|                                   | .spin.SpinOperatorTerm.ops_count) |
|                                   | -   [OptimizationResult (class in |
|                                   |                                   |
|                                   |    cudaq)](api/languages/python_a |
|                                   | pi.html#cudaq.OptimizationResult) |
|                                   | -   [overlap() (cudaq.State       |
|                                   |     method)](api/languages/pyt    |
|                                   | hon_api.html#cudaq.State.overlap) |
+-----------------------------------+-----------------------------------+

## P {#P}

+-----------------------------------+-----------------------------------+
| -   [parameters                   | -   [Pauli1 (class in             |
|     (cu                           |     cudaq)](api/langua            |
| daq.operators.boson.BosonOperator | ges/python_api.html#cudaq.Pauli1) |
|     property)](api/languag        | -   [Pauli2 (class in             |
| es/python_api.html#cudaq.operator |     cudaq)](api/langua            |
| s.boson.BosonOperator.parameters) | ges/python_api.html#cudaq.Pauli2) |
|     -   [(cudaq.                  | -   [PhaseDamping (class in       |
| operators.boson.BosonOperatorTerm |     cudaq)](api/languages/py      |
|                                   | thon_api.html#cudaq.PhaseDamping) |
|        property)](api/languages/p | -   [PhaseFlipChannel (class in   |
| ython_api.html#cudaq.operators.bo |     cudaq)](api/languages/python  |
| son.BosonOperatorTerm.parameters) | _api.html#cudaq.PhaseFlipChannel) |
|     -   [(cudaq.                  | -   [platform (cudaq.Target       |
| operators.fermion.FermionOperator |                                   |
|                                   |    property)](api/languages/pytho |
|        property)](api/languages/p | n_api.html#cudaq.Target.platform) |
| ython_api.html#cudaq.operators.fe | -   [plus() (in module            |
| rmion.FermionOperator.parameters) |     cudaq.spin)](api/languages    |
|     -   [(cudaq.oper              | /python_api.html#cudaq.spin.plus) |
| ators.fermion.FermionOperatorTerm | -   [position() (in module        |
|                                   |                                   |
|    property)](api/languages/pytho |  cudaq.boson)](api/languages/pyth |
| n_api.html#cudaq.operators.fermio | on_api.html#cudaq.boson.position) |
| n.FermionOperatorTerm.parameters) |     -   [(in module               |
|     -                             |         cudaq.operators.custo     |
|  [(cudaq.operators.MatrixOperator | m)](api/languages/python_api.html |
|         property)](api/la         | #cudaq.operators.custom.position) |
| nguages/python_api.html#cudaq.ope | -   [probability                  |
| rators.MatrixOperator.parameters) |     (cudaq.operators.c            |
|     -   [(cuda                    | ustom.cudaq.ptsbe.KrausTrajectory |
| q.operators.MatrixOperatorElement |     attrib                        |
|         property)](api/languages  | ute)](api/languages/python_api.ht |
| /python_api.html#cudaq.operators. | ml#cudaq.operators.custom.cudaq.p |
| MatrixOperatorElement.parameters) | tsbe.KrausTrajectory.probability) |
|     -   [(c                       | -   [probability()                |
| udaq.operators.MatrixOperatorTerm |     (cudaq.SampleResult           |
|         property)](api/langua     |     meth                          |
| ges/python_api.html#cudaq.operato | od)](api/languages/python_api.htm |
| rs.MatrixOperatorTerm.parameters) | l#cudaq.SampleResult.probability) |
|     -                             | -   [ProductOperator (in module   |
|  [(cudaq.operators.ScalarOperator |     cudaq.operator                |
|         property)](api/la         | s)](api/languages/python_api.html |
| nguages/python_api.html#cudaq.ope | #cudaq.operators.ProductOperator) |
| rators.ScalarOperator.parameters) | -   [PROPORTIONAL                 |
|     -   [(                        |                                   |
| cudaq.operators.spin.SpinOperator |    (cudaq.operators.custom.cudaq. |
|         property)](api/langu      | ptsbe.ShotAllocationStrategy.Type |
| ages/python_api.html#cudaq.operat |     attribute)](api/lan           |
| ors.spin.SpinOperator.parameters) | guages/python_api.html#cudaq.oper |
|     -   [(cuda                    | ators.custom.cudaq.ptsbe.ShotAllo |
| q.operators.spin.SpinOperatorTerm | cationStrategy.Type.PROPORTIONAL) |
|         property)](api/languages  | -   [PyKernel (class in           |
| /python_api.html#cudaq.operators. |     cudaq)](api/language          |
| spin.SpinOperatorTerm.parameters) | s/python_api.html#cudaq.PyKernel) |
| -   [ParameterShift (class in     | -   [PyKernelDecorator (class in  |
|     cudaq.gradien                 |     cudaq)](api/languages/python_ |
| ts)](api/languages/python_api.htm | api.html#cudaq.PyKernelDecorator) |
| l#cudaq.gradients.ParameterShift) |                                   |
| -   [params                       |                                   |
|     (cudaq.operators.cu           |                                   |
| stom.cudaq.ptsbe.TraceInstruction |                                   |
|     at                            |                                   |
| tribute)](api/languages/python_ap |                                   |
| i.html#cudaq.operators.custom.cud |                                   |
| aq.ptsbe.TraceInstruction.params) |                                   |
| -   [parity() (in module          |                                   |
|     cudaq.operators.cus           |                                   |
| tom)](api/languages/python_api.ht |                                   |
| ml#cudaq.operators.custom.parity) |                                   |
| -   [Pauli (class in              |                                   |
|     cudaq.spin)](api/languages/   |                                   |
| python_api.html#cudaq.spin.Pauli) |                                   |
+-----------------------------------+-----------------------------------+

## Q {#Q}

+-----------------------------------+-----------------------------------+
| -   [qkeModule                    | -   [qubit_count                  |
|     (cudaq.PyKernelDecorator      |     (                             |
|     property)                     | cudaq.operators.spin.SpinOperator |
| ](api/languages/python_api.html#c |     property)](api/langua         |
| udaq.PyKernelDecorator.qkeModule) | ges/python_api.html#cudaq.operato |
| -   [qreg (in module              | rs.spin.SpinOperator.qubit_count) |
|     cudaq)](api/lang              |     -   [(cuda                    |
| uages/python_api.html#cudaq.qreg) | q.operators.spin.SpinOperatorTerm |
| -   [QuakeValue (class in         |         property)](api/languages/ |
|     cudaq)](api/languages/        | python_api.html#cudaq.operators.s |
| python_api.html#cudaq.QuakeValue) | pin.SpinOperatorTerm.qubit_count) |
| -   [qubit (class in              | -   [qubits                       |
|     cudaq)](api/langu             |     (cudaq.operators.             |
| ages/python_api.html#cudaq.qubit) | custom.cudaq.ptsbe.KrausSelection |
|                                   |                                   |
|                                   | attribute)](api/languages/python_ |
|                                   | api.html#cudaq.operators.custom.c |
|                                   | udaq.ptsbe.KrausSelection.qubits) |
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
+-----------------------------------+-----------------------------------+

## S {#S}

+-----------------------------------+-----------------------------------+
| -   [sample() (in module          | -   [SimulationPrecision (class   |
|     cudaq)](api/langua            |     in                            |
| ges/python_api.html#cudaq.sample) |                                   |
|     -   [(in module               |   cudaq)](api/languages/python_ap |
|                                   | i.html#cudaq.SimulationPrecision) |
|      cudaq.orca)](api/languages/p | -   [simulator (cudaq.Target      |
| ython_api.html#cudaq.orca.sample) |                                   |
| -   [sample_async() (in module    |   property)](api/languages/python |
|     cudaq)](api/languages/py      | _api.html#cudaq.Target.simulator) |
| thon_api.html#cudaq.sample_async) | -   [slice() (cudaq.QuakeValue    |
| -   [SampleResult (class in       |     method)](api/languages/python |
|     cudaq)](api/languages/py      | _api.html#cudaq.QuakeValue.slice) |
| thon_api.html#cudaq.SampleResult) | -   [SpinOperator (class in       |
| -   [ScalarOperator (class in     |     cudaq.operators.spin)         |
|     cudaq.operato                 | ](api/languages/python_api.html#c |
| rs)](api/languages/python_api.htm | udaq.operators.spin.SpinOperator) |
| l#cudaq.operators.ScalarOperator) | -   [SpinOperatorElement (class   |
| -   [Schedule (class in           |     in                            |
|     cudaq)](api/language          |     cudaq.operators.spin)](api/l  |
| s/python_api.html#cudaq.Schedule) | anguages/python_api.html#cudaq.op |
| -   [serialize()                  | erators.spin.SpinOperatorElement) |
|     (                             | -   [SpinOperatorTerm (class in   |
| cudaq.operators.spin.SpinOperator |     cudaq.operators.spin)](ap     |
|     method)](api/lang             | i/languages/python_api.html#cudaq |
| uages/python_api.html#cudaq.opera | .operators.spin.SpinOperatorTerm) |
| tors.spin.SpinOperator.serialize) | -   [SPSA (class in               |
|     -   [(cuda                    |     cudaq                         |
| q.operators.spin.SpinOperatorTerm | .optimizers)](api/languages/pytho |
|         method)](api/language     | n_api.html#cudaq.optimizers.SPSA) |
| s/python_api.html#cudaq.operators | -   [squeeze() (in module         |
| .spin.SpinOperatorTerm.serialize) |     cudaq.operators.cust          |
|     -   [(cudaq.SampleResult      | om)](api/languages/python_api.htm |
|         me                        | l#cudaq.operators.custom.squeeze) |
| thod)](api/languages/python_api.h | -   [State (class in              |
| tml#cudaq.SampleResult.serialize) |     cudaq)](api/langu             |
| -   [set_noise() (in module       | ages/python_api.html#cudaq.State) |
|     cudaq)](api/languages         | -   [step_size                    |
| /python_api.html#cudaq.set_noise) |     (cudaq.optimizers.Adam        |
| -   [set_random_seed() (in module |     propert                       |
|     cudaq)](api/languages/pytho   | y)](api/languages/python_api.html |
| n_api.html#cudaq.set_random_seed) | #cudaq.optimizers.Adam.step_size) |
| -   [set_target() (in module      |     -   [(cudaq.optimizers.SGD    |
|     cudaq)](api/languages/        |         proper                    |
| python_api.html#cudaq.set_target) | ty)](api/languages/python_api.htm |
| -   [SGD (class in                | l#cudaq.optimizers.SGD.step_size) |
|     cuda                          |     -   [(cudaq.optimizers.SPSA   |
| q.optimizers)](api/languages/pyth |         propert                   |
| on_api.html#cudaq.optimizers.SGD) | y)](api/languages/python_api.html |
| -   [signatureWithCallables()     | #cudaq.optimizers.SPSA.step_size) |
|     (cudaq.PyKernelDecorator      | -   [SuperOperator (class in      |
|     method)](api/languag          |     cudaq)](api/languages/pyt     |
| es/python_api.html#cudaq.PyKernel | hon_api.html#cudaq.SuperOperator) |
| Decorator.signatureWithCallables) | -   [supports_compilation()       |
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
| -   [targets                      |    method)](api/languages/python_ |
|     (cudaq.operators.cu           | api.html#cudaq.operators.fermion. |
| stom.cudaq.ptsbe.TraceInstruction | FermionOperator.to_sparse_matrix) |
|     att                           |     -   [(cudaq.oper              |
| ribute)](api/languages/python_api | ators.fermion.FermionOperatorTerm |
| .html#cudaq.operators.custom.cuda |         m                         |
| q.ptsbe.TraceInstruction.targets) | ethod)](api/languages/python_api. |
| -   [Tensor (class in             | html#cudaq.operators.fermion.Ferm |
|     cudaq)](api/langua            | ionOperatorTerm.to_sparse_matrix) |
| ges/python_api.html#cudaq.Tensor) |     -   [(                        |
| -   [term_count                   | cudaq.operators.spin.SpinOperator |
|     (cu                           |         method)](api/languages/p  |
| daq.operators.boson.BosonOperator | ython_api.html#cudaq.operators.sp |
|     property)](api/languag        | in.SpinOperator.to_sparse_matrix) |
| es/python_api.html#cudaq.operator |     -   [(cuda                    |
| s.boson.BosonOperator.term_count) | q.operators.spin.SpinOperatorTerm |
|     -   [(cudaq.                  |                                   |
| operators.fermion.FermionOperator |      method)](api/languages/pytho |
|                                   | n_api.html#cudaq.operators.spin.S |
|        property)](api/languages/p | pinOperatorTerm.to_sparse_matrix) |
| ython_api.html#cudaq.operators.fe | -   [to_string()                  |
| rmion.FermionOperator.term_count) |     (cudaq.ope                    |
|     -                             | rators.boson.BosonOperatorElement |
|  [(cudaq.operators.MatrixOperator |     method)](api/languages/pyt    |
|         property)](api/la         | hon_api.html#cudaq.operators.boso |
| nguages/python_api.html#cudaq.ope | n.BosonOperatorElement.to_string) |
| rators.MatrixOperator.term_count) |     -   [(cudaq.operato           |
|     -   [(                        | rs.fermion.FermionOperatorElement |
| cudaq.operators.spin.SpinOperator |                                   |
|         property)](api/langu      |    method)](api/languages/python_ |
| ages/python_api.html#cudaq.operat | api.html#cudaq.operators.fermion. |
| ors.spin.SpinOperator.term_count) | FermionOperatorElement.to_string) |
|     -   [(cuda                    |     -   [(cuda                    |
| q.operators.spin.SpinOperatorTerm | q.operators.MatrixOperatorElement |
|         property)](api/languages  |         method)](api/language     |
| /python_api.html#cudaq.operators. | s/python_api.html#cudaq.operators |
| spin.SpinOperatorTerm.term_count) | .MatrixOperatorElement.to_string) |
| -   [term_id                      |     -   [(                        |
|     (cudaq.                       | cudaq.operators.spin.SpinOperator |
| operators.boson.BosonOperatorTerm |         method)](api/lang         |
|     property)](api/language       | uages/python_api.html#cudaq.opera |
| s/python_api.html#cudaq.operators | tors.spin.SpinOperator.to_string) |
| .boson.BosonOperatorTerm.term_id) |     -   [(cudaq.o                 |
|     -   [(cudaq.oper              | perators.spin.SpinOperatorElement |
| ators.fermion.FermionOperatorTerm |         method)](api/languages/p  |
|                                   | ython_api.html#cudaq.operators.sp |
|       property)](api/languages/py | in.SpinOperatorElement.to_string) |
| thon_api.html#cudaq.operators.fer |     -   [(cuda                    |
| mion.FermionOperatorTerm.term_id) | q.operators.spin.SpinOperatorTerm |
|     -   [(c                       |         method)](api/language     |
| udaq.operators.MatrixOperatorTerm | s/python_api.html#cudaq.operators |
|         property)](api/lan        | .spin.SpinOperatorTerm.to_string) |
| guages/python_api.html#cudaq.oper | -   [trajectories                 |
| ators.MatrixOperatorTerm.term_id) |     (cudaq.operators.cust         |
|     -   [(cuda                    | om.cudaq.ptsbe.PTSBEExecutionData |
| q.operators.spin.SpinOperatorTerm |     attribute)                    |
|         property)](api/langua     | ](api/languages/python_api.html#c |
| ges/python_api.html#cudaq.operato | udaq.operators.custom.cudaq.ptsbe |
| rs.spin.SpinOperatorTerm.term_id) | .PTSBEExecutionData.trajectories) |
| -   [to_dict() (cudaq.Resources   | -   [trajectory_id                |
|                                   |     (cudaq.operators.c            |
|    method)](api/languages/python_ | ustom.cudaq.ptsbe.KrausTrajectory |
| api.html#cudaq.Resources.to_dict) |     attribut                      |
| -   [to_json()                    | e)](api/languages/python_api.html |
|     (                             | #cudaq.operators.custom.cudaq.pts |
| cudaq.gradients.CentralDifference | be.KrausTrajectory.trajectory_id) |
|     method)](api/la               | -   [translate() (in module       |
| nguages/python_api.html#cudaq.gra |     cudaq)](api/languages         |
| dients.CentralDifference.to_json) | /python_api.html#cudaq.translate) |
|     -   [(                        | -   [trim()                       |
| cudaq.gradients.ForwardDifference |     (cu                           |
|         method)](api/la           | daq.operators.boson.BosonOperator |
| nguages/python_api.html#cudaq.gra |     method)](api/l                |
| dients.ForwardDifference.to_json) | anguages/python_api.html#cudaq.op |
|     -                             | erators.boson.BosonOperator.trim) |
|  [(cudaq.gradients.ParameterShift |     -   [(cudaq.                  |
|         method)](api              | operators.fermion.FermionOperator |
| /languages/python_api.html#cudaq. |         method)](api/langu        |
| gradients.ParameterShift.to_json) | ages/python_api.html#cudaq.operat |
|     -   [(                        | ors.fermion.FermionOperator.trim) |
| cudaq.operators.spin.SpinOperator |     -                             |
|         method)](api/la           |  [(cudaq.operators.MatrixOperator |
| nguages/python_api.html#cudaq.ope |         method)](                 |
| rators.spin.SpinOperator.to_json) | api/languages/python_api.html#cud |
|     -   [(cuda                    | aq.operators.MatrixOperator.trim) |
| q.operators.spin.SpinOperatorTerm |     -   [(                        |
|         method)](api/langua       | cudaq.operators.spin.SpinOperator |
| ges/python_api.html#cudaq.operato |         method)](api              |
| rs.spin.SpinOperatorTerm.to_json) | /languages/python_api.html#cudaq. |
|     -   [(cudaq.optimizers.Adam   | operators.spin.SpinOperator.trim) |
|         met                       | -   [type                         |
| hod)](api/languages/python_api.ht |     (cudaq.operators.cu           |
| ml#cudaq.optimizers.Adam.to_json) | stom.cudaq.ptsbe.TraceInstruction |
|     -   [(cudaq.optimizers.COBYLA |                                   |
|         metho                     | attribute)](api/languages/python_ |
| d)](api/languages/python_api.html | api.html#cudaq.operators.custom.c |
| #cudaq.optimizers.COBYLA.to_json) | udaq.ptsbe.TraceInstruction.type) |
|     -   [                         | -   [type_to_str()                |
| (cudaq.optimizers.GradientDescent |     (cudaq.PyKernelDecorator      |
|         method)](api/l            |     static                        |
| anguages/python_api.html#cudaq.op |     method)](                     |
| timizers.GradientDescent.to_json) | api/languages/python_api.html#cud |
|     -   [(cudaq.optimizers.LBFGS  | aq.PyKernelDecorator.type_to_str) |
|         meth                      |                                   |
| od)](api/languages/python_api.htm |                                   |
| l#cudaq.optimizers.LBFGS.to_json) |                                   |
|                                   |                                   |
| -   [(cudaq.optimizers.NelderMead |                                   |
|         method)](                 |                                   |
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
| -   [UNIFORM                                                          |
|     (cudaq.operators.custom.cudaq.ptsbe.ShotAllocationStrategy.Type   |
|     attribute)](api/languages/python_api.html#cu                      |
| daq.operators.custom.cudaq.ptsbe.ShotAllocationStrategy.Type.UNIFORM) |
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
