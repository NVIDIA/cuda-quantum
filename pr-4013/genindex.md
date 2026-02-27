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
| -   [canonicalize()               | -   [cudaq::produc                |
|     (cu                           | t_op::const_iterator::operator-\> |
| daq.operators.boson.BosonOperator |     (C++                          |
|     method)](api/languages        |     function)](api/lan            |
| /python_api.html#cudaq.operators. | guages/cpp_api.html#_CPPv4N5cudaq |
| boson.BosonOperator.canonicalize) | 10product_op14const_iteratorptEv) |
|     -   [(cudaq.                  | -   [cudaq::produ                 |
| operators.boson.BosonOperatorTerm | ct_op::const_iterator::operator== |
|                                   |     (C++                          |
|        method)](api/languages/pyt |     fun                           |
| hon_api.html#cudaq.operators.boso | ction)](api/languages/cpp_api.htm |
| n.BosonOperatorTerm.canonicalize) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(cudaq.                  | st_iteratoreqERK14const_iterator) |
| operators.fermion.FermionOperator | -   [cudaq::product_op::degrees   |
|                                   |     (C++                          |
|        method)](api/languages/pyt |     function)                     |
| hon_api.html#cudaq.operators.ferm | ](api/languages/cpp_api.html#_CPP |
| ion.FermionOperator.canonicalize) | v4NK5cudaq10product_op7degreesEv) |
|     -   [(cudaq.oper              | -   [cudaq::product_op::dump (C++ |
| ators.fermion.FermionOperatorTerm |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|    method)](api/languages/python_ | CPPv4NK5cudaq10product_op4dumpEv) |
| api.html#cudaq.operators.fermion. | -   [cudaq::product_op::end (C++  |
| FermionOperatorTerm.canonicalize) |     funct                         |
|     -                             | ion)](api/languages/cpp_api.html# |
|  [(cudaq.operators.MatrixOperator | _CPPv4NK5cudaq10product_op3endEv) |
|         method)](api/lang         | -   [c                            |
| uages/python_api.html#cudaq.opera | udaq::product_op::get_coefficient |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     function)](api/lan            |
| udaq.operators.MatrixOperatorTerm | guages/cpp_api.html#_CPPv4NK5cuda |
|         method)](api/language     | q10product_op15get_coefficientEv) |
| s/python_api.html#cudaq.operators | -                                 |
| .MatrixOperatorTerm.canonicalize) |   [cudaq::product_op::get_term_id |
|     -   [(                        |     (C++                          |
| cudaq.operators.spin.SpinOperator |     function)](api                |
|         method)](api/languag      | /languages/cpp_api.html#_CPPv4NK5 |
| es/python_api.html#cudaq.operator | cudaq10product_op11get_term_idEv) |
| s.spin.SpinOperator.canonicalize) | -                                 |
|     -   [(cuda                    |   [cudaq::product_op::is_identity |
| q.operators.spin.SpinOperatorTerm |     (C++                          |
|         method)](api/languages/p  |     function)](api                |
| ython_api.html#cudaq.operators.sp | /languages/cpp_api.html#_CPPv4NK5 |
| in.SpinOperatorTerm.canonicalize) | cudaq10product_op11is_identityEv) |
| -   [canonicalized() (in module   | -   [cudaq::product_op::num_ops   |
|     cuda                          |     (C++                          |
| q.boson)](api/languages/python_ap |     function)                     |
| i.html#cudaq.boson.canonicalized) | ](api/languages/cpp_api.html#_CPP |
|     -   [(in module               | v4NK5cudaq10product_op7num_opsEv) |
|         cudaq.fe                  | -                                 |
| rmion)](api/languages/python_api. |    [cudaq::product_op::operator\* |
| html#cudaq.fermion.canonicalized) |     (C++                          |
|     -   [(in module               |     function)](api/languages/     |
|                                   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|        cudaq.operators.custom)](a | oduct_opmlE10product_opI1TERK15sc |
| pi/languages/python_api.html#cuda | alar_operatorRK10product_opI1TE), |
| q.operators.custom.canonicalized) |     [\[1\]](api/languages/        |
|     -   [(in module               | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|         cu                        | oduct_opmlE10product_opI1TERK15sc |
| daq.spin)](api/languages/python_a | alar_operatorRR10product_opI1TE), |
| pi.html#cudaq.spin.canonicalized) |     [\[2\]](api/languages/        |
| -   [captured_variables()         | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     (cudaq.PyKernelDecorator      | oduct_opmlE10product_opI1TERR15sc |
|     method)](api/lan              | alar_operatorRK10product_opI1TE), |
| guages/python_api.html#cudaq.PyKe |     [\[3\]](api/languages/        |
| rnelDecorator.captured_variables) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [CentralDifference (class in  | oduct_opmlE10product_opI1TERR15sc |
|     cudaq.gradients)              | alar_operatorRR10product_opI1TE), |
| ](api/languages/python_api.html#c |     [\[4\]](api/                  |
| udaq.gradients.CentralDifference) | languages/cpp_api.html#_CPPv4I0EN |
| -   [clear() (cudaq.Resources     | 5cudaq10product_opmlE6sum_opI1TER |
|     method)](api/languages/pytho  | K15scalar_operatorRK6sum_opI1TE), |
| n_api.html#cudaq.Resources.clear) |     [\[5\]](api/                  |
|     -   [(cudaq.SampleResult      | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq10product_opmlE6sum_opI1TER |
|   method)](api/languages/python_a | K15scalar_operatorRR6sum_opI1TE), |
| pi.html#cudaq.SampleResult.clear) |     [\[6\]](api/                  |
| -   [COBYLA (class in             | languages/cpp_api.html#_CPPv4I0EN |
|     cudaq.o                       | 5cudaq10product_opmlE6sum_opI1TER |
| ptimizers)](api/languages/python_ | R15scalar_operatorRK6sum_opI1TE), |
| api.html#cudaq.optimizers.COBYLA) |     [\[7\]](api/                  |
| -   [coefficient                  | languages/cpp_api.html#_CPPv4I0EN |
|     (cudaq.                       | 5cudaq10product_opmlE6sum_opI1TER |
| operators.boson.BosonOperatorTerm | R15scalar_operatorRR6sum_opI1TE), |
|     property)](api/languages/py   |     [\[8\]](api/languages         |
| thon_api.html#cudaq.operators.bos | /cpp_api.html#_CPPv4NK5cudaq10pro |
| on.BosonOperatorTerm.coefficient) | duct_opmlERK6sum_opI9HandlerTyE), |
|     -   [(cudaq.oper              |     [\[9\]](api/languages/cpp_a   |
| ators.fermion.FermionOperatorTerm | pi.html#_CPPv4NKR5cudaq10product_ |
|                                   | opmlERK10product_opI9HandlerTyE), |
|   property)](api/languages/python |     [\[10\]](api/language         |
| _api.html#cudaq.operators.fermion | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| .FermionOperatorTerm.coefficient) | roduct_opmlERK15scalar_operator), |
|     -   [(c                       |     [\[11\]](api/languages/cpp_a  |
| udaq.operators.MatrixOperatorTerm | pi.html#_CPPv4NKR5cudaq10product_ |
|         property)](api/languag    | opmlERR10product_opI9HandlerTyE), |
| es/python_api.html#cudaq.operator |     [\[12\]](api/language         |
| s.MatrixOperatorTerm.coefficient) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     -   [(cuda                    | roduct_opmlERR15scalar_operator), |
| q.operators.spin.SpinOperatorTerm |     [\[13\]](api/languages/cpp_   |
|         property)](api/languages/ | api.html#_CPPv4NO5cudaq10product_ |
| python_api.html#cudaq.operators.s | opmlERK10product_opI9HandlerTyE), |
| pin.SpinOperatorTerm.coefficient) |     [\[14\]](api/languag          |
| -   [col_count                    | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (cudaq.KrausOperator          | roduct_opmlERK15scalar_operator), |
|     prope                         |     [\[15\]](api/languages/cpp_   |
| rty)](api/languages/python_api.ht | api.html#_CPPv4NO5cudaq10product_ |
| ml#cudaq.KrausOperator.col_count) | opmlERR10product_opI9HandlerTyE), |
| -   [compile()                    |     [\[16\]](api/langua           |
|     (cudaq.PyKernelDecorator      | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     metho                         | product_opmlERR15scalar_operator) |
| d)](api/languages/python_api.html | -                                 |
| #cudaq.PyKernelDecorator.compile) |   [cudaq::product_op::operator\*= |
| -   [ComplexMatrix (class in      |     (C++                          |
|     cudaq)](api/languages/pyt     |     function)](api/languages/cpp  |
| hon_api.html#cudaq.ComplexMatrix) | _api.html#_CPPv4N5cudaq10product_ |
| -   [compute()                    | opmLERK10product_opI9HandlerTyE), |
|     (                             |     [\[1\]](api/langua            |
| cudaq.gradients.CentralDifference | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     method)](api/la               | roduct_opmLERK15scalar_operator), |
| nguages/python_api.html#cudaq.gra |     [\[2\]](api/languages/cp      |
| dients.CentralDifference.compute) | p_api.html#_CPPv4N5cudaq10product |
|     -   [(                        | _opmLERR10product_opI9HandlerTyE) |
| cudaq.gradients.ForwardDifference | -   [cudaq::product_op::operator+ |
|         method)](api/la           |     (C++                          |
| nguages/python_api.html#cudaq.gra |     function)](api/langu          |
| dients.ForwardDifference.compute) | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     -                             | q10product_opplE6sum_opI1TERK15sc |
|  [(cudaq.gradients.ParameterShift | alar_operatorRK10product_opI1TE), |
|         method)](api              |     [\[1\]](api/                  |
| /languages/python_api.html#cudaq. | languages/cpp_api.html#_CPPv4I0EN |
| gradients.ParameterShift.compute) | 5cudaq10product_opplE6sum_opI1TER |
| -   [const()                      | K15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[2\]](api/langu             |
|   (cudaq.operators.ScalarOperator | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     class                         | q10product_opplE6sum_opI1TERK15sc |
|     method)](a                    | alar_operatorRR10product_opI1TE), |
| pi/languages/python_api.html#cuda |     [\[3\]](api/                  |
| q.operators.ScalarOperator.const) | languages/cpp_api.html#_CPPv4I0EN |
| -   [copy()                       | 5cudaq10product_opplE6sum_opI1TER |
|     (cu                           | K15scalar_operatorRR6sum_opI1TE), |
| daq.operators.boson.BosonOperator |     [\[4\]](api/langu             |
|     method)](api/l                | ages/cpp_api.html#_CPPv4I0EN5cuda |
| anguages/python_api.html#cudaq.op | q10product_opplE6sum_opI1TERR15sc |
| erators.boson.BosonOperator.copy) | alar_operatorRK10product_opI1TE), |
|     -   [(cudaq.                  |     [\[5\]](api/                  |
| operators.boson.BosonOperatorTerm | languages/cpp_api.html#_CPPv4I0EN |
|         method)](api/langu        | 5cudaq10product_opplE6sum_opI1TER |
| ages/python_api.html#cudaq.operat | R15scalar_operatorRK6sum_opI1TE), |
| ors.boson.BosonOperatorTerm.copy) |     [\[6\]](api/langu             |
|     -   [(cudaq.                  | ages/cpp_api.html#_CPPv4I0EN5cuda |
| operators.fermion.FermionOperator | q10product_opplE6sum_opI1TERR15sc |
|         method)](api/langu        | alar_operatorRR10product_opI1TE), |
| ages/python_api.html#cudaq.operat |     [\[7\]](api/                  |
| ors.fermion.FermionOperator.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(cudaq.oper              | 5cudaq10product_opplE6sum_opI1TER |
| ators.fermion.FermionOperatorTerm | R15scalar_operatorRR6sum_opI1TE), |
|         method)](api/languages    |     [\[8\]](api/languages/cpp_a   |
| /python_api.html#cudaq.operators. | pi.html#_CPPv4NKR5cudaq10product_ |
| fermion.FermionOperatorTerm.copy) | opplERK10product_opI9HandlerTyE), |
|     -                             |     [\[9\]](api/language          |
|  [(cudaq.operators.MatrixOperator | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|         method)](                 | roduct_opplERK15scalar_operator), |
| api/languages/python_api.html#cud |     [\[10\]](api/languages/       |
| aq.operators.MatrixOperator.copy) | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     -   [(c                       | duct_opplERK6sum_opI9HandlerTyE), |
| udaq.operators.MatrixOperatorTerm |     [\[11\]](api/languages/cpp_a  |
|         method)](api/             | pi.html#_CPPv4NKR5cudaq10product_ |
| languages/python_api.html#cudaq.o | opplERR10product_opI9HandlerTyE), |
| perators.MatrixOperatorTerm.copy) |     [\[12\]](api/language         |
|     -   [(                        | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| cudaq.operators.spin.SpinOperator | roduct_opplERR15scalar_operator), |
|         method)](api              |     [\[13\]](api/languages/       |
| /languages/python_api.html#cudaq. | cpp_api.html#_CPPv4NKR5cudaq10pro |
| operators.spin.SpinOperator.copy) | duct_opplERR6sum_opI9HandlerTyE), |
|     -   [(cuda                    |     [\[                           |
| q.operators.spin.SpinOperatorTerm | 14\]](api/languages/cpp_api.html# |
|         method)](api/lan          | _CPPv4NKR5cudaq10product_opplEv), |
| guages/python_api.html#cudaq.oper |     [\[15\]](api/languages/cpp_   |
| ators.spin.SpinOperatorTerm.copy) | api.html#_CPPv4NO5cudaq10product_ |
| -   [count() (cudaq.Resources     | opplERK10product_opI9HandlerTyE), |
|     method)](api/languages/pytho  |     [\[16\]](api/languag          |
| n_api.html#cudaq.Resources.count) | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     -   [(cudaq.SampleResult      | roduct_opplERK15scalar_operator), |
|                                   |     [\[17\]](api/languages        |
|   method)](api/languages/python_a | /cpp_api.html#_CPPv4NO5cudaq10pro |
| pi.html#cudaq.SampleResult.count) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [count_controls()             |     [\[18\]](api/languages/cpp_   |
|     (cudaq.Resources              | api.html#_CPPv4NO5cudaq10product_ |
|     meth                          | opplERR10product_opI9HandlerTyE), |
| od)](api/languages/python_api.htm |     [\[19\]](api/languag          |
| l#cudaq.Resources.count_controls) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [count_instructions()         | roduct_opplERR15scalar_operator), |
|                                   |     [\[20\]](api/languages        |
|   (cudaq.ptsbe.PTSBEExecutionData | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     method)](api/languages/       | duct_opplERR6sum_opI9HandlerTyE), |
| python_api.html#cudaq.ptsbe.PTSBE |     [                             |
| ExecutionData.count_instructions) | \[21\]](api/languages/cpp_api.htm |
| -   [counts()                     | l#_CPPv4NO5cudaq10product_opplEv) |
|     (cudaq.ObserveResult          | -   [cudaq::product_op::operator- |
|                                   |     (C++                          |
| method)](api/languages/python_api |     function)](api/langu          |
| .html#cudaq.ObserveResult.counts) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [create() (in module          | q10product_opmiE6sum_opI1TERK15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|    cudaq.boson)](api/languages/py |     [\[1\]](api/                  |
| thon_api.html#cudaq.boson.create) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(in module               | 5cudaq10product_opmiE6sum_opI1TER |
|         c                         | K15scalar_operatorRK6sum_opI1TE), |
| udaq.fermion)](api/languages/pyth |     [\[2\]](api/langu             |
| on_api.html#cudaq.fermion.create) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [csr_spmatrix (C++            | q10product_opmiE6sum_opI1TERK15sc |
|     type)](api/languages/c        | alar_operatorRR10product_opI1TE), |
| pp_api.html#_CPPv412csr_spmatrix) |     [\[3\]](api/                  |
| -   cudaq                         | languages/cpp_api.html#_CPPv4I0EN |
|     -   [module](api/langua       | 5cudaq10product_opmiE6sum_opI1TER |
| ges/python_api.html#module-cudaq) | K15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq (C++                   |     [\[4\]](api/langu             |
|     type)](api/lan                | ages/cpp_api.html#_CPPv4I0EN5cuda |
| guages/cpp_api.html#_CPPv45cudaq) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cudaq.apply_noise() (in      | alar_operatorRK10product_opI1TE), |
|     module                        |     [\[5\]](api/                  |
|     cudaq)](api/languages/python_ | languages/cpp_api.html#_CPPv4I0EN |
| api.html#cudaq.cudaq.apply_noise) | 5cudaq10product_opmiE6sum_opI1TER |
| -   cudaq.boson                   | R15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/languages/py |     [\[6\]](api/langu             |
| thon_api.html#module-cudaq.boson) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   cudaq.fermion                 | q10product_opmiE6sum_opI1TERR15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|   -   [module](api/languages/pyth |     [\[7\]](api/                  |
| on_api.html#module-cudaq.fermion) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq.operators.custom        | 5cudaq10product_opmiE6sum_opI1TER |
|     -   [mo                       | R15scalar_operatorRR6sum_opI1TE), |
| dule](api/languages/python_api.ht |     [\[8\]](api/languages/cpp_a   |
| ml#module-cudaq.operators.custom) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.spin                    | opmiERK10product_opI9HandlerTyE), |
|     -   [module](api/languages/p  |     [\[9\]](api/language          |
| ython_api.html#module-cudaq.spin) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::amplitude_damping     | roduct_opmiERK15scalar_operator), |
|     (C++                          |     [\[10\]](api/languages/       |
|     cla                           | cpp_api.html#_CPPv4NKR5cudaq10pro |
| ss)](api/languages/cpp_api.html#_ | duct_opmiERK6sum_opI9HandlerTyE), |
| CPPv4N5cudaq17amplitude_dampingE) |     [\[11\]](api/languages/cpp_a  |
| -                                 | pi.html#_CPPv4NKR5cudaq10product_ |
| [cudaq::amplitude_damping_channel | opmiERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[12\]](api/language         |
|     class)](api                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| /languages/cpp_api.html#_CPPv4N5c | roduct_opmiERR15scalar_operator), |
| udaq25amplitude_damping_channelE) |     [\[13\]](api/languages/       |
| -   [cudaq::amplitud              | cpp_api.html#_CPPv4NKR5cudaq10pro |
| e_damping_channel::num_parameters | duct_opmiERR6sum_opI9HandlerTyE), |
|     (C++                          |     [\[                           |
|     member)](api/languages/cpp_a  | 14\]](api/languages/cpp_api.html# |
| pi.html#_CPPv4N5cudaq25amplitude_ | _CPPv4NKR5cudaq10product_opmiEv), |
| damping_channel14num_parametersE) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::ampli                 | api.html#_CPPv4NO5cudaq10product_ |
| tude_damping_channel::num_targets | opmiERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[16\]](api/languag          |
|     member)](api/languages/cp     | es/cpp_api.html#_CPPv4NO5cudaq10p |
| p_api.html#_CPPv4N5cudaq25amplitu | roduct_opmiERK15scalar_operator), |
| de_damping_channel11num_targetsE) |     [\[17\]](api/languages        |
| -   [cudaq::AnalogRemoteRESTQPU   | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERK6sum_opI9HandlerTyE), |
|     class                         |     [\[18\]](api/languages/cpp_   |
| )](api/languages/cpp_api.html#_CP | api.html#_CPPv4NO5cudaq10product_ |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | opmiERR10product_opI9HandlerTyE), |
| -   [cudaq::apply_noise (C++      |     [\[19\]](api/languag          |
|     function)](api/               | es/cpp_api.html#_CPPv4NO5cudaq10p |
| languages/cpp_api.html#_CPPv4I0Dp | roduct_opmiERR15scalar_operator), |
| EN5cudaq11apply_noiseEvDpRR4Args) |     [\[20\]](api/languages        |
| -   [cudaq::async_result (C++     | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     c                             | duct_opmiERR6sum_opI9HandlerTyE), |
| lass)](api/languages/cpp_api.html |     [                             |
| #_CPPv4I0EN5cudaq12async_resultE) | \[21\]](api/languages/cpp_api.htm |
| -   [cudaq::async_result::get     | l#_CPPv4NO5cudaq10product_opmiEv) |
|     (C++                          | -   [cudaq::product_op::operator/ |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function)](api/language       |
| CPPv4N5cudaq12async_result3getEv) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::async_sample_result   | roduct_opdvERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/language          |
|     type                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| )](api/languages/cpp_api.html#_CP | roduct_opdvERR15scalar_operator), |
| Pv4N5cudaq19async_sample_resultE) |     [\[2\]](api/languag           |
| -   [cudaq::BaseRemoteRESTQPU     | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opdvERK15scalar_operator), |
|     cla                           |     [\[3\]](api/langua            |
| ss)](api/languages/cpp_api.html#_ | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | product_opdvERR15scalar_operator) |
| -                                 | -                                 |
|    [cudaq::BaseRemoteSimulatorQPU |    [cudaq::product_op::operator/= |
|     (C++                          |     (C++                          |
|     class)](                      |     function)](api/langu          |
| api/languages/cpp_api.html#_CPPv4 | ages/cpp_api.html#_CPPv4N5cudaq10 |
| N5cudaq22BaseRemoteSimulatorQPUE) | product_opdVERK15scalar_operator) |
| -   [cudaq::bit_flip_channel (C++ | -   [cudaq::product_op::operator= |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     function)](api/la             |
| _CPPv4N5cudaq16bit_flip_channelE) | nguages/cpp_api.html#_CPPv4I0_NSt |
| -   [cudaq:                       | 11enable_if_tIXaantNSt7is_sameI1T |
| :bit_flip_channel::num_parameters | 9HandlerTyE5valueENSt16is_constru |
|     (C++                          | ctibleI9HandlerTy1TE5valueEEbEEEN |
|     member)](api/langua           | 5cudaq10product_opaSER10product_o |
| ges/cpp_api.html#_CPPv4N5cudaq16b | pI9HandlerTyERK10product_opI1TE), |
| it_flip_channel14num_parametersE) |     [\[1\]](api/languages/cpp     |
| -   [cud                          | _api.html#_CPPv4N5cudaq10product_ |
| aq::bit_flip_channel::num_targets | opaSERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     member)](api/lan              | p_api.html#_CPPv4N5cudaq10product |
| guages/cpp_api.html#_CPPv4N5cudaq | _opaSERR10product_opI9HandlerTyE) |
| 16bit_flip_channel11num_targetsE) | -                                 |
| -   [cudaq::boson_handler (C++    |    [cudaq::product_op::operator== |
|                                   |     (C++                          |
|  class)](api/languages/cpp_api.ht |     function)](api/languages/cpp  |
| ml#_CPPv4N5cudaq13boson_handlerE) | _api.html#_CPPv4NK5cudaq10product |
| -   [cudaq::boson_op (C++         | _opeqERK10product_opI9HandlerTyE) |
|     type)](api/languages/cpp_     | -                                 |
| api.html#_CPPv4N5cudaq8boson_opE) |  [cudaq::product_op::operator\[\] |
| -   [cudaq::boson_op_term (C++    |     (C++                          |
|                                   |     function)](ap                 |
|   type)](api/languages/cpp_api.ht | i/languages/cpp_api.html#_CPPv4NK |
| ml#_CPPv4N5cudaq13boson_op_termE) | 5cudaq10product_opixENSt6size_tE) |
| -   [cudaq::CodeGenConfig (C++    | -                                 |
|                                   |    [cudaq::product_op::product_op |
| struct)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     function)](api/languages/c    |
| -   [cudaq::commutation_relations | pp_api.html#_CPPv4I0_NSt11enable_ |
|     (C++                          | if_tIXaaNSt7is_sameI9HandlerTy14m |
|     struct)]                      | atrix_handlerE5valueEaantNSt7is_s |
| (api/languages/cpp_api.html#_CPPv | ameI1T9HandlerTyE5valueENSt16is_c |
| 4N5cudaq21commutation_relationsE) | onstructibleI9HandlerTy1TE5valueE |
| -   [cudaq::complex (C++          | EbEEEN5cudaq10product_op10product |
|     type)](api/languages/cpp      | _opERK10product_opI1TERKN14matrix |
| _api.html#_CPPv4N5cudaq7complexE) | _handler20commutation_behaviorE), |
| -   [cudaq::complex_matrix (C++   |                                   |
|                                   |  [\[1\]](api/languages/cpp_api.ht |
| class)](api/languages/cpp_api.htm | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| l#_CPPv4N5cudaq14complex_matrixE) | tNSt7is_sameI1T9HandlerTyE5valueE |
| -                                 | NSt16is_constructibleI9HandlerTy1 |
|   [cudaq::complex_matrix::adjoint | TE5valueEEbEEEN5cudaq10product_op |
|     (C++                          | 10product_opERK10product_opI1TE), |
|     function)](a                  |                                   |
| pi/languages/cpp_api.html#_CPPv4N |   [\[2\]](api/languages/cpp_api.h |
| 5cudaq14complex_matrix7adjointEv) | tml#_CPPv4N5cudaq10product_op10pr |
| -   [cudaq::                      | oduct_opENSt6size_tENSt6size_tE), |
| complex_matrix::diagonal_elements |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     function)](api/languages      | _op10product_opENSt7complexIdEE), |
| /cpp_api.html#_CPPv4NK5cudaq14com |     [\[4\]](api/l                 |
| plex_matrix17diagonal_elementsEi) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::complex_matrix::dump  | aq10product_op10product_opERK10pr |
|     (C++                          | oduct_opI9HandlerTyENSt6size_tE), |
|     function)](api/language       |     [\[5\]](api/l                 |
| s/cpp_api.html#_CPPv4NK5cudaq14co | anguages/cpp_api.html#_CPPv4N5cud |
| mplex_matrix4dumpERNSt7ostreamE), | aq10product_op10product_opERR10pr |
|     [\[1\]]                       | oduct_opI9HandlerTyENSt6size_tE), |
| (api/languages/cpp_api.html#_CPPv |     [\[6\]](api/languages         |
| 4NK5cudaq14complex_matrix4dumpEv) | /cpp_api.html#_CPPv4N5cudaq10prod |
| -   [c                            | uct_op10product_opERR9HandlerTy), |
| udaq::complex_matrix::eigenvalues |     [\[7\]](ap                    |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](api/lan            | cudaq10product_op10product_opEd), |
| guages/cpp_api.html#_CPPv4NK5cuda |     [\[8\]](a                     |
| q14complex_matrix11eigenvaluesEv) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cu                           | 5cudaq10product_op10product_opEv) |
| daq::complex_matrix::eigenvectors | -   [cuda                         |
|     (C++                          | q::product_op::to_diagonal_matrix |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4NK5cudaq |     function)](api/               |
| 14complex_matrix12eigenvectorsEv) | languages/cpp_api.html#_CPPv4NK5c |
| -   [c                            | udaq10product_op18to_diagonal_mat |
| udaq::complex_matrix::exponential | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     function)](api/la             | apINSt6stringENSt7complexIdEEEEb) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::product_op::to_matrix |
| q14complex_matrix11exponentialEv) |     (C++                          |
| -                                 |     funct                         |
|  [cudaq::complex_matrix::identity | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NK5cudaq10product_op9to_mat |
|     function)](api/languages      | rixENSt13unordered_mapINSt6size_t |
| /cpp_api.html#_CPPv4N5cudaq14comp | ENSt7int64_tEEERKNSt13unordered_m |
| lex_matrix8identityEKNSt6size_tE) | apINSt6stringENSt7complexIdEEEEb) |
| -                                 | -   [cu                           |
| [cudaq::complex_matrix::kronecker | daq::product_op::to_sparse_matrix |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](ap                 |
| uages/cpp_api.html#_CPPv4I00EN5cu | i/languages/cpp_api.html#_CPPv4NK |
| daq14complex_matrix9kroneckerE14c | 5cudaq10product_op16to_sparse_mat |
| omplex_matrix8Iterable8Iterable), | rixENSt13unordered_mapINSt6size_t |
|     [\[1\]](api/l                 | ENSt7int64_tEEERKNSt13unordered_m |
| anguages/cpp_api.html#_CPPv4N5cud | apINSt6stringENSt7complexIdEEEEb) |
| aq14complex_matrix9kroneckerERK14 | -   [cudaq::product_op::to_string |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -   [cudaq::c                     |     function)](                   |
| omplex_matrix::minimal_eigenvalue | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq10product_op9to_stringEv) |
|     function)](api/languages/     | -                                 |
| cpp_api.html#_CPPv4NK5cudaq14comp |  [cudaq::product_op::\~product_op |
| lex_matrix18minimal_eigenvalueEv) |     (C++                          |
| -   [                             |     fu                            |
| cudaq::complex_matrix::operator() | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq10product_opD0Ev) |
|     function)](api/languages/cpp  | -   [cudaq::ptsbe (C++            |
| _api.html#_CPPv4N5cudaq14complex_ |     type)](api/languages/c        |
| matrixclENSt6size_tENSt6size_tE), | pp_api.html#_CPPv4N5cudaq5ptsbeE) |
|     [\[1\]](api/languages/cpp     | -   [cudaq::p                     |
| _api.html#_CPPv4NK5cudaq14complex | tsbe::ConditionalSamplingStrategy |
| _matrixclENSt6size_tENSt6size_tE) |     (C++                          |
| -   [                             |     class)](api/languag           |
| cudaq::complex_matrix::operator\* | es/cpp_api.html#_CPPv4N5cudaq5pts |
|     (C++                          | be27ConditionalSamplingStrategyE) |
|     function)](api/langua         | -   [cudaq::ptsbe::C              |
| ges/cpp_api.html#_CPPv4N5cudaq14c | onditionalSamplingStrategy::clone |
| omplex_matrixmlEN14complex_matrix |     (C++                          |
| 10value_typeERK14complex_matrix), |                                   |
|     [\[1\]                        |    function)](api/languages/cpp_a |
| ](api/languages/cpp_api.html#_CPP | pi.html#_CPPv4NK5cudaq5ptsbe27Con |
| v4N5cudaq14complex_matrixmlERK14c | ditionalSamplingStrategy5cloneEv) |
| omplex_matrixRK14complex_matrix), | -   [cuda                         |
|                                   | q::ptsbe::ConditionalSamplingStra |
|  [\[2\]](api/languages/cpp_api.ht | tegy::ConditionalSamplingStrategy |
| ml#_CPPv4N5cudaq14complex_matrixm |     (C++                          |
| lERK14complex_matrixRKNSt6vectorI |     function)](api/lang           |
| N14complex_matrix10value_typeEEE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -                                 | ptsbe27ConditionalSamplingStrateg |
| [cudaq::complex_matrix::operator+ | y27ConditionalSamplingStrategyE19 |
|     (C++                          | TrajectoryPredicateNSt8uint64_tE) |
|     function                      | -                                 |
| )](api/languages/cpp_api.html#_CP |   [cudaq::ptsbe::ConditionalSampl |
| Pv4N5cudaq14complex_matrixplERK14 | ingStrategy::generateTrajectories |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -                                 |     function)](api                |
| [cudaq::complex_matrix::operator- | /languages/cpp_api.html#_CPPv4NK5 |
|     (C++                          | cudaq5ptsbe27ConditionalSamplingS |
|     function                      | trategy20generateTrajectoriesENSt |
| )](api/languages/cpp_api.html#_CP | 4spanIK10NoisePointEENSt6size_tE) |
| Pv4N5cudaq14complex_matrixmiERK14 | -   [cudaq::ptsbe::               |
| complex_matrixRK14complex_matrix) | ConditionalSamplingStrategy::name |
| -   [cu                           |     (C++                          |
| daq::complex_matrix::operator\[\] |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4NK5cudaq5ptsbe27Co |
|                                   | nditionalSamplingStrategy4nameEv) |
|  function)](api/languages/cpp_api | -   [cudaq:                       |
| .html#_CPPv4N5cudaq14complex_matr | :ptsbe::ConditionalSamplingStrate |
| ixixERKNSt6vectorINSt6size_tEEE), | gy::\~ConditionalSamplingStrategy |
|     [\[1\]](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4NK5cudaq14complex_mat |     function)](api/languages/     |
| rixixERKNSt6vectorINSt6size_tEEE) | cpp_api.html#_CPPv4N5cudaq5ptsbe2 |
| -   [cudaq::complex_matrix::power | 7ConditionalSamplingStrategyD0Ev) |
|     (C++                          | -   [cudaq::                      |
|     function)]                    | ptsbe::ExhaustiveSamplingStrategy |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq14complex_matrix5powerEi) |     class)](api/langua            |
| -                                 | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|  [cudaq::complex_matrix::set_zero | sbe26ExhaustiveSamplingStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     function)](ap                 | ExhaustiveSamplingStrategy::clone |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq14complex_matrix8set_zeroEv) |     function)](api/languages/cpp_ |
| -                                 | api.html#_CPPv4NK5cudaq5ptsbe26Ex |
| [cudaq::complex_matrix::to_string | haustiveSamplingStrategy5cloneEv) |
|     (C++                          | -   [cu                           |
|     function)](api/               | daq::ptsbe::ExhaustiveSamplingStr |
| languages/cpp_api.html#_CPPv4NK5c | ategy::ExhaustiveSamplingStrategy |
| udaq14complex_matrix9to_stringEv) |     (C++                          |
| -   [                             |     function)](api/la             |
| cudaq::complex_matrix::value_type | nguages/cpp_api.html#_CPPv4N5cuda |
|     (C++                          | q5ptsbe26ExhaustiveSamplingStrate |
|     type)](api/                   | gy26ExhaustiveSamplingStrategyEv) |
| languages/cpp_api.html#_CPPv4N5cu | -                                 |
| daq14complex_matrix10value_typeE) |    [cudaq::ptsbe::ExhaustiveSampl |
| -   [cudaq::contrib (C++          | ingStrategy::generateTrajectories |
|     type)](api/languages/cpp      |     (C++                          |
| _api.html#_CPPv4N5cudaq7contribE) |     function)](ap                 |
| -   [cudaq::contrib::draw (C++    | i/languages/cpp_api.html#_CPPv4NK |
|     function)                     | 5cudaq5ptsbe26ExhaustiveSamplingS |
| ](api/languages/cpp_api.html#_CPP | trategy20generateTrajectoriesENSt |
| v4I0DpEN5cudaq7contrib4drawENSt6s | 4spanIK10NoisePointEENSt6size_tE) |
| tringERR13QuantumKernelDpRR4Args) | -   [cudaq::ptsbe:                |
| -                                 | :ExhaustiveSamplingStrategy::name |
| [cudaq::contrib::get_unitary_cmat |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)](api/languages/cp   | _api.html#_CPPv4NK5cudaq5ptsbe26E |
| p_api.html#_CPPv4I0DpEN5cudaq7con | xhaustiveSamplingStrategy4nameEv) |
| trib16get_unitary_cmatE14complex_ | -   [cuda                         |
| matrixRR13QuantumKernelDpRR4Args) | q::ptsbe::ExhaustiveSamplingStrat |
| -   [cudaq::CusvState (C++        | egy::\~ExhaustiveSamplingStrategy |
|                                   |     (C++                          |
|    class)](api/languages/cpp_api. |     function)](api/languages      |
| html#_CPPv4I0EN5cudaq9CusvStateE) | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| -   [cudaq::depolarization1 (C++  | 26ExhaustiveSamplingStrategyD0Ev) |
|     c                             | -   [cudaq::ptsbe::NoisePoint     |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15depolarization1E) |     str                           |
| -   [cudaq::depolarization2 (C++  | uct)](api/languages/cpp_api.html# |
|     c                             | _CPPv4N5cudaq5ptsbe10NoisePointE) |
| lass)](api/languages/cpp_api.html | -   [                             |
| #_CPPv4N5cudaq15depolarization2E) | cudaq::ptsbe::NoisePoint::channel |
| -   [cudaq:                       |     (C++                          |
| :depolarization2::depolarization2 |     member)](ap                   |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](api/languages/cp   | cudaq5ptsbe10NoisePoint7channelE) |
| p_api.html#_CPPv4N5cudaq15depolar | -   [cudaq::pt                    |
| ization215depolarization2EK4real) | sbe::NoisePoint::circuit_location |
| -   [cudaq                        |     (C++                          |
| ::depolarization2::num_parameters |     member)](api/language         |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq5ptsb |
|     member)](api/langu            | e10NoisePoint16circuit_locationE) |
| ages/cpp_api.html#_CPPv4N5cudaq15 | -   [                             |
| depolarization214num_parametersE) | cudaq::ptsbe::NoisePoint::op_name |
| -   [cu                           |     (C++                          |
| daq::depolarization2::num_targets |     member)](ap                   |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     member)](api/la               | cudaq5ptsbe10NoisePoint7op_nameE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -                                 |
| q15depolarization211num_targetsE) | [cudaq::ptsbe::NoisePoint::qubits |
| -                                 |     (C++                          |
|    [cudaq::depolarization_channel |     member)](a                    |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     class)](                      | 5cudaq5ptsbe10NoisePoint6qubitsE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cuda                         |
| N5cudaq22depolarization_channelE) | q::ptsbe::OrderedSamplingStrategy |
| -   [cudaq::depol                 |     (C++                          |
| arization_channel::num_parameters |     class)](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     member)](api/languages/cp     | 5ptsbe23OrderedSamplingStrategyE) |
| p_api.html#_CPPv4N5cudaq22depolar | -   [cudaq::ptsb                  |
| ization_channel14num_parametersE) | e::OrderedSamplingStrategy::clone |
| -   [cudaq::de                    |     (C++                          |
| polarization_channel::num_targets |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq5ptsbe2 |
|     member)](api/languages        | 3OrderedSamplingStrategy5cloneEv) |
| /cpp_api.html#_CPPv4N5cudaq22depo | -   [cudaq::ptsbe::OrderedSampl   |
| larization_channel11num_targetsE) | ingStrategy::generateTrajectories |
| -   [cudaq::details (C++          |     (C++                          |
|     type)](api/languages/cpp      |     function)]                    |
| _api.html#_CPPv4N5cudaq7detailsE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::details::future (C++  | 4NK5cudaq5ptsbe23OrderedSamplingS |
|                                   | trategy20generateTrajectoriesENSt |
|  class)](api/languages/cpp_api.ht | 4spanIK10NoisePointEENSt6size_tE) |
| ml#_CPPv4N5cudaq7details6futureE) | -   [cudaq::pts                   |
| -                                 | be::OrderedSamplingStrategy::name |
|   [cudaq::details::future::future |     (C++                          |
|     (C++                          |     function)](api/languages/     |
|     functio                       | cpp_api.html#_CPPv4NK5cudaq5ptsbe |
| n)](api/languages/cpp_api.html#_C | 23OrderedSamplingStrategy4nameEv) |
| PPv4N5cudaq7details6future6future | -                                 |
| ERNSt6vectorI3JobEERNSt6stringERN |    [cudaq::ptsbe::OrderedSampling |
| St3mapINSt6stringENSt6stringEEE), | Strategy::OrderedSamplingStrategy |
|     [\[1\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     function)](                   |
| details6future6futureERR6future), | api/languages/cpp_api.html#_CPPv4 |
|     [\[2\]]                       | N5cudaq5ptsbe23OrderedSamplingStr |
| (api/languages/cpp_api.html#_CPPv | ategy23OrderedSamplingStrategyEv) |
| 4N5cudaq7details6future6futureEv) | -                                 |
| -   [cu                           |  [cudaq::ptsbe::OrderedSamplingSt |
| daq::details::kernel_builder_base | rategy::\~OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|     class)](api/l                 |     function)](api/langua         |
| anguages/cpp_api.html#_CPPv4N5cud | ges/cpp_api.html#_CPPv4N5cudaq5pt |
| aq7details19kernel_builder_baseE) | sbe23OrderedSamplingStrategyD0Ev) |
| -   [cudaq::details::             | -   [cudaq::pts                   |
| kernel_builder_base::operator\<\< | be::ProbabilisticSamplingStrategy |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     class)](api/languages         |
| ges/cpp_api.html#_CPPv4N5cudaq7de | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
| tails19kernel_builder_baselsERNSt | 29ProbabilisticSamplingStrategyE) |
| 7ostreamERK19kernel_builder_base) | -   [cudaq::ptsbe::Pro            |
| -   [                             | babilisticSamplingStrategy::clone |
| cudaq::details::KernelBuilderType |     (C++                          |
|     (C++                          |                                   |
|     class)](api                   |  function)](api/languages/cpp_api |
| /languages/cpp_api.html#_CPPv4N5c | .html#_CPPv4NK5cudaq5ptsbe29Proba |
| udaq7details17KernelBuilderTypeE) | bilisticSamplingStrategy5cloneEv) |
| -   [cudaq::d                     | -                                 |
| etails::KernelBuilderType::create | [cudaq::ptsbe::ProbabilisticSampl |
|     (C++                          | ingStrategy::generateTrajectories |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/l              |
| v4N5cudaq7details17KernelBuilderT | anguages/cpp_api.html#_CPPv4NK5cu |
| ype6createEPN4mlir11MLIRContextE) | daq5ptsbe29ProbabilisticSamplingS |
| -   [cudaq::details::Ker          | trategy20generateTrajectoriesENSt |
| nelBuilderType::KernelBuilderType | 4spanIK10NoisePointEENSt6size_tE) |
|     (C++                          | -   [cudaq::ptsbe::Pr             |
|     function)](api/lang           | obabilisticSamplingStrategy::name |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     (C++                          |
| details17KernelBuilderType17Kerne |                                   |
| lBuilderTypeERRNSt8functionIFN4ml |   function)](api/languages/cpp_ap |
| ir4TypeEPN4mlir11MLIRContextEEEE) | i.html#_CPPv4NK5cudaq5ptsbe29Prob |
| -   [cudaq::diag_matrix_callback  | abilisticSamplingStrategy4nameEv) |
|     (C++                          | -   [cudaq::p                     |
|     class)                        | tsbe::ProbabilisticSamplingStrate |
| ](api/languages/cpp_api.html#_CPP | gy::ProbabilisticSamplingStrategy |
| v4N5cudaq20diag_matrix_callbackE) |     (C++                          |
| -   [cudaq::dyn (C++              |     function)]                    |
|     member)](api/languages        | (api/languages/cpp_api.html#_CPPv |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | 4N5cudaq5ptsbe29ProbabilisticSamp |
| -   [cudaq::ExecutionContext (C++ | lingStrategy29ProbabilisticSampli |
|     cl                            | ngStrategyENSt8optionalINSt8uint6 |
| ass)](api/languages/cpp_api.html# | 4_tEEENSt8optionalINSt6size_tEEE) |
| _CPPv4N5cudaq16ExecutionContextE) | -   [cudaq::pts                   |
| -   [cudaq                        | be::ProbabilisticSamplingStrategy |
| ::ExecutionContext::amplitudeMaps | ::\~ProbabilisticSamplingStrategy |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api/languages/cp   |
| ages/cpp_api.html#_CPPv4N5cudaq16 | p_api.html#_CPPv4N5cudaq5ptsbe29P |
| ExecutionContext13amplitudeMapsE) | robabilisticSamplingStrategyD0Ev) |
| -   [c                            | -                                 |
| udaq::ExecutionContext::asyncExec | [cudaq::ptsbe::PTSBEExecutionData |
|     (C++                          |     (C++                          |
|     member)](api/                 |     struct)](ap                   |
| languages/cpp_api.html#_CPPv4N5cu | i/languages/cpp_api.html#_CPPv4N5 |
| daq16ExecutionContext9asyncExecE) | cudaq5ptsbe18PTSBEExecutionDataE) |
| -   [cud                          | -   [cudaq::ptsbe::PTSBE          |
| aq::ExecutionContext::asyncResult | ExecutionData::count_instructions |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/l              |
| guages/cpp_api.html#_CPPv4N5cudaq | anguages/cpp_api.html#_CPPv4NK5cu |
| 16ExecutionContext11asyncResultE) | daq5ptsbe18PTSBEExecutionData18co |
| -   [cudaq:                       | unt_instructionsE20TraceInstructi |
| :ExecutionContext::batchIteration | onTypeNSt8optionalINSt6stringEEE) |
|     (C++                          | -   [cudaq::ptsbe::P              |
|     member)](api/langua           | TSBEExecutionData::get_trajectory |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14batchIterationE) |     function                      |
| -   [cudaq::E                     | )](api/languages/cpp_api.html#_CP |
| xecutionContext::canHandleObserve | Pv4NK5cudaq5ptsbe18PTSBEExecution |
|     (C++                          | Data14get_trajectoryENSt6size_tE) |
|     member)](api/language         | -   [cudaq::ptsbe:                |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | :PTSBEExecutionData::instructions |
| cutionContext16canHandleObserveE) |     (C++                          |
| -   [cudaq::E                     |     member)](api/languages/cp     |
| xecutionContext::ExecutionContext | p_api.html#_CPPv4N5cudaq5ptsbe18P |
|     (C++                          | TSBEExecutionData12instructionsE) |
|     func                          | -   [cudaq::ptsbe:                |
| tion)](api/languages/cpp_api.html | :PTSBEExecutionData::trajectories |
| #_CPPv4N5cudaq16ExecutionContext1 |     (C++                          |
| 6ExecutionContextERKNSt6stringE), |     member)](api/languages/cp     |
|     [\[1\]](api/languages/        | p_api.html#_CPPv4N5cudaq5ptsbe18P |
| cpp_api.html#_CPPv4N5cudaq16Execu | TSBEExecutionData12trajectoriesE) |
| tionContext16ExecutionContextERKN | -   [cudaq::ptsbe::PTSBEOptions   |
| St6stringENSt6size_tENSt6size_tE) |     (C++                          |
| -   [cudaq::E                     |     struc                         |
| xecutionContext::expectationValue | t)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
|     member)](api/language         | -   [cudaq::ptsb                  |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | e::PTSBEOptions::max_trajectories |
| cutionContext16expectationValueE) |     (C++                          |
| -   [cudaq::Execu                 |     member)](api/languages/       |
| tionContext::explicitMeasurements | cpp_api.html#_CPPv4N5cudaq5ptsbe1 |
|     (C++                          | 2PTSBEOptions16max_trajectoriesE) |
|     member)](api/languages/cp     | -   [cudaq::ptsbe::PT             |
| p_api.html#_CPPv4N5cudaq16Executi | SBEOptions::return_execution_data |
| onContext20explicitMeasurementsE) |     (C++                          |
| -   [cuda                         |     member)](api/languages/cpp_a  |
| q::ExecutionContext::futureResult | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
|     (C++                          | EOptions21return_execution_dataE) |
|     member)](api/lang             | -   [cudaq::pts                   |
| uages/cpp_api.html#_CPPv4N5cudaq1 | be::PTSBEOptions::shot_allocation |
| 6ExecutionContext12futureResultE) |     (C++                          |
| -   [cudaq::ExecutionContext      |     member)](api/languages        |
| ::hasConditionalsOnMeasureResults | /cpp_api.html#_CPPv4N5cudaq5ptsbe |
|     (C++                          | 12PTSBEOptions15shot_allocationE) |
|     mem                           | -   [cud                          |
| ber)](api/languages/cpp_api.html# | aq::ptsbe::PTSBEOptions::strategy |
| _CPPv4N5cudaq16ExecutionContext31 |     (C++                          |
| hasConditionalsOnMeasureResultsE) |     member)](api/l                |
| -   [cudaq::Executi               | anguages/cpp_api.html#_CPPv4N5cud |
| onContext::invocationResultBuffer | aq5ptsbe12PTSBEOptions8strategyE) |
|     (C++                          | -   [cudaq::ptsbe::PTSBETrace     |
|     member)](api/languages/cpp_   |     (C++                          |
| api.html#_CPPv4N5cudaq16Execution |     t                             |
| Context22invocationResultBufferE) | ype)](api/languages/cpp_api.html# |
| -   [cu                           | _CPPv4N5cudaq5ptsbe10PTSBETraceE) |
| daq::ExecutionContext::kernelName | -   [                             |
|     (C++                          | cudaq::ptsbe::PTSSamplingStrategy |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     class)](api                   |
| q16ExecutionContext10kernelNameE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cud                          | udaq5ptsbe19PTSSamplingStrategyE) |
| aq::ExecutionContext::kernelTrace | -   [cudaq::                      |
|     (C++                          | ptsbe::PTSSamplingStrategy::clone |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/languag        |
| 16ExecutionContext11kernelTraceE) | es/cpp_api.html#_CPPv4NK5cudaq5pt |
| -   [cudaq:                       | sbe19PTSSamplingStrategy5cloneEv) |
| :ExecutionContext::msm_dimensions | -   [cudaq::ptsbe::PTSSampl       |
|     (C++                          | ingStrategy::generateTrajectories |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     functi                        |
| xecutionContext14msm_dimensionsE) | on)](api/languages/cpp_api.html#_ |
| -   [cudaq::                      | CPPv4NK5cudaq5ptsbe19PTSSamplingS |
| ExecutionContext::msm_prob_err_id | trategy20generateTrajectoriesENSt |
|     (C++                          | 4spanIK10NoisePointEENSt6size_tE) |
|     member)](api/languag          | -   [cudaq:                       |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | :ptsbe::PTSSamplingStrategy::name |
| ecutionContext15msm_prob_err_idE) |     (C++                          |
| -   [cudaq::Ex                    |     function)](api/langua         |
| ecutionContext::msm_probabilities | ges/cpp_api.html#_CPPv4NK5cudaq5p |
|     (C++                          | tsbe19PTSSamplingStrategy4nameEv) |
|     member)](api/languages        | -   [cudaq::ptsbe::PTSSampli      |
| /cpp_api.html#_CPPv4N5cudaq16Exec | ngStrategy::\~PTSSamplingStrategy |
| utionContext17msm_probabilitiesE) |     (C++                          |
| -                                 |     function)](api/la             |
|    [cudaq::ExecutionContext::name | nguages/cpp_api.html#_CPPv4N5cuda |
|     (C++                          | q5ptsbe19PTSSamplingStrategyD0Ev) |
|     member)]                      | -   [cudaq::ptsbe::sample (C++    |
| (api/languages/cpp_api.html#_CPPv |                                   |
| 4N5cudaq16ExecutionContext4nameE) |  function)](api/languages/cpp_api |
| -   [cu                           | .html#_CPPv4I0DpEN5cudaq5ptsbe6sa |
| daq::ExecutionContext::noiseModel | mpleE13sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     member)](api/la               |     [\[1\]](api                   |
| nguages/cpp_api.html#_CPPv4N5cuda | /languages/cpp_api.html#_CPPv4I0D |
| q16ExecutionContext10noiseModelE) | pEN5cudaq5ptsbe6sampleE13sample_r |
| -   [cudaq::Exe                   | esultRKN5cudaq11noise_modelENSt6s |
| cutionContext::numberTrajectories | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::ptsbe::sample_async   |
|     member)](api/languages/       |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16Execu |     function)](a                  |
| tionContext18numberTrajectoriesE) | pi/languages/cpp_api.html#_CPPv4I |
| -   [c                            | 0DpEN5cudaq5ptsbe12sample_asyncE1 |
| udaq::ExecutionContext::optResult | 9async_sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     member)](api/                 |     [\[1\]](api/languages/cp      |
| languages/cpp_api.html#_CPPv4N5cu | p_api.html#_CPPv4I0DpEN5cudaq5pts |
| daq16ExecutionContext9optResultE) | be12sample_asyncE19async_sample_r |
| -   [cudaq::Execu                 | esultRKN5cudaq11noise_modelENSt6s |
| tionContext::overlapComputeStates | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::ptsbe::sample_options |
|     member)](api/languages/cp     |     (C++                          |
| p_api.html#_CPPv4N5cudaq16Executi |     struct)                       |
| onContext20overlapComputeStatesE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq                        | v4N5cudaq5ptsbe14sample_optionsE) |
| ::ExecutionContext::overlapResult | -   [cudaq::ptsbe::sample_result  |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     class                         |
| ages/cpp_api.html#_CPPv4N5cudaq16 | )](api/languages/cpp_api.html#_CP |
| ExecutionContext13overlapResultE) | Pv4N5cudaq5ptsbe13sample_resultE) |
| -                                 | -   [cudaq::pts                   |
|   [cudaq::ExecutionContext::qpuId | be::sample_result::execution_data |
|     (C++                          |     (C++                          |
|     member)](                     |     function)](api/languages/c    |
| api/languages/cpp_api.html#_CPPv4 | pp_api.html#_CPPv4NK5cudaq5ptsbe1 |
| N5cudaq16ExecutionContext5qpuIdE) | 3sample_result14execution_dataEv) |
| -   [cudaq                        | -   [cudaq::ptsbe::               |
| ::ExecutionContext::registerNames | sample_result::has_execution_data |
|     (C++                          |     (C++                          |
|     member)](api/langu            |                                   |
| ages/cpp_api.html#_CPPv4N5cudaq16 |    function)](api/languages/cpp_a |
| ExecutionContext13registerNamesE) | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| -   [cu                           | ple_result18has_execution_dataEv) |
| daq::ExecutionContext::reorderIdx | -   [cudaq::pt                    |
|     (C++                          | sbe::sample_result::sample_result |
|     member)](api/la               |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/l              |
| q16ExecutionContext10reorderIdxE) | anguages/cpp_api.html#_CPPv4N5cud |
| -                                 | aq5ptsbe13sample_result13sample_r |
|  [cudaq::ExecutionContext::result | esultERRN5cudaq13sample_resultE), |
|     (C++                          |                                   |
|     member)](a                    |  [\[1\]](api/languages/cpp_api.ht |
| pi/languages/cpp_api.html#_CPPv4N | ml#_CPPv4N5cudaq5ptsbe13sample_re |
| 5cudaq16ExecutionContext6resultE) | sult13sample_resultERRN5cudaq13sa |
| -                                 | mple_resultE18PTSBEExecutionData) |
|   [cudaq::ExecutionContext::shots | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::set_execution_data |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/               |
| N5cudaq16ExecutionContext5shotsE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::                      | daq5ptsbe13sample_result18set_exe |
| ExecutionContext::simulationState | cution_dataE18PTSBEExecutionData) |
|     (C++                          | -   [cud                          |
|     member)](api/languag          | aq::ptsbe::ShotAllocationStrategy |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15simulationStateE) |     struct)](api/la               |
| -                                 | nguages/cpp_api.html#_CPPv4N5cuda |
|    [cudaq::ExecutionContext::spin | q5ptsbe22ShotAllocationStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAllocatio  |
|     member)]                      | nStrategy::ShotAllocationStrategy |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq16ExecutionContext4spinE) |     function)]                    |
| -   [cudaq::                      | (api/languages/cpp_api.html#_CPPv |
| ExecutionContext::totalIterations | 4N5cudaq5ptsbe22ShotAllocationStr |
|     (C++                          | ategy22ShotAllocationStrategyE4Ty |
|     member)](api/languag          | pedNSt8optionalINSt8uint64_tEEE), |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     [\[1\]                        |
| ecutionContext15totalIterationsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::Executio              | v4N5cudaq5ptsbe22ShotAllocationSt |
| nContext::warnedNamedMeasurements | rategy22ShotAllocationStrategyEv) |
|     (C++                          | -   [cudaq::pt                    |
|     member)](api/languages/cpp_a  | sbe::ShotAllocationStrategy::Type |
| pi.html#_CPPv4N5cudaq16ExecutionC |     (C++                          |
| ontext23warnedNamedMeasurementsE) |     enum)](api/languag            |
| -   [cudaq::ExecutionResult (C++  | es/cpp_api.html#_CPPv4N5cudaq5pts |
|     st                            | be22ShotAllocationStrategy4TypeE) |
| ruct)](api/languages/cpp_api.html | -   [cudaq::ptsbe::ShotAllocatio  |
| #_CPPv4N5cudaq15ExecutionResultE) | nStrategy::Type::HIGH_WEIGHT_BIAS |
| -   [cud                          |     (C++                          |
| aq::ExecutionResult::appendResult |     enumerato                     |
|     (C++                          | r)](api/languages/cpp_api.html#_C |
|     functio                       | PPv4N5cudaq5ptsbe22ShotAllocation |
| n)](api/languages/cpp_api.html#_C | Strategy4Type16HIGH_WEIGHT_BIASE) |
| PPv4N5cudaq15ExecutionResult12app | -   [cudaq::ptsbe::ShotAllocati   |
| endResultENSt6stringENSt6size_tE) | onStrategy::Type::LOW_WEIGHT_BIAS |
| -   [cu                           |     (C++                          |
| daq::ExecutionResult::deserialize |     enumerat                      |
|     (C++                          | or)](api/languages/cpp_api.html#_ |
|     function)                     | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| ](api/languages/cpp_api.html#_CPP | nStrategy4Type15LOW_WEIGHT_BIASE) |
| v4N5cudaq15ExecutionResult11deser | -   [cudaq::ptsbe::ShotAlloc      |
| ializeERNSt6vectorINSt6size_tEEE) | ationStrategy::Type::PROPORTIONAL |
| -   [cudaq:                       |     (C++                          |
| :ExecutionResult::ExecutionResult |     enume                         |
|     (C++                          | rator)](api/languages/cpp_api.htm |
|     functio                       | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| n)](api/languages/cpp_api.html#_C | tionStrategy4Type12PROPORTIONALE) |
| PPv4N5cudaq15ExecutionResult15Exe | -   [cudaq::ptsbe::Shot           |
| cutionResultE16CountsDictionary), | AllocationStrategy::Type::UNIFORM |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 15ExecutionResult15ExecutionResul |  enumerator)](api/languages/cpp_a |
| tE16CountsDictionaryNSt6stringE), | pi.html#_CPPv4N5cudaq5ptsbe22Shot |
|     [\[2\                         | AllocationStrategy4Type7UNIFORME) |
| ]](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq15ExecutionResult15Exec |   [cudaq::ptsbe::TraceInstruction |
| utionResultE16CountsDictionaryd), |     (C++                          |
|                                   |     struct)](                     |
|    [\[3\]](api/languages/cpp_api. | api/languages/cpp_api.html#_CPPv4 |
| html#_CPPv4N5cudaq15ExecutionResu | N5cudaq5ptsbe16TraceInstructionE) |
| lt15ExecutionResultENSt6stringE), | -   [cudaq:                       |
|     [\[4\                         | :ptsbe::TraceInstruction::channel |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     member)](api/lang             |
| utionResultERK15ExecutionResult), | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     [\[5\]](api/language          | ptsbe16TraceInstruction7channelE) |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | -   [cudaq::                      |
| cutionResult15ExecutionResultEd), | ptsbe::TraceInstruction::controls |
|     [\[6\]](api/languag           |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |     member)](api/langu            |
| ecutionResult15ExecutionResultEv) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [                             | tsbe16TraceInstruction8controlsE) |
| cudaq::ExecutionResult::operator= | -   [cud                          |
|     (C++                          | aq::ptsbe::TraceInstruction::name |
|     function)](api/languages/     |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq15Execu |     member)](api/l                |
| tionResultaSERK15ExecutionResult) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [c                            | aq5ptsbe16TraceInstruction4nameE) |
| udaq::ExecutionResult::operator== | -   [cudaq                        |
|     (C++                          | ::ptsbe::TraceInstruction::params |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4NK5cudaq15Execu |     member)](api/lan              |
| tionResulteqERK15ExecutionResult) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [cud                          | 5ptsbe16TraceInstruction6paramsE) |
| aq::ExecutionResult::registerName | -   [cudaq:                       |
|     (C++                          | :ptsbe::TraceInstruction::targets |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     member)](api/lang             |
| 15ExecutionResult12registerNameE) | uages/cpp_api.html#_CPPv4N5cudaq5 |
| -   [cudaq                        | ptsbe16TraceInstruction7targetsE) |
| ::ExecutionResult::sequentialData | -   [cudaq::ptsbe::T              |
|     (C++                          | raceInstruction::TraceInstruction |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |                                   |
| ExecutionResult14sequentialDataE) |   function)](api/languages/cpp_ap |
| -   [                             | i.html#_CPPv4N5cudaq5ptsbe16Trace |
| cudaq::ExecutionResult::serialize | Instruction16TraceInstructionE20T |
|     (C++                          | raceInstructionTypeNSt6stringENSt |
|     function)](api/l              | 6vectorINSt6size_tEEENSt6vectorIN |
| anguages/cpp_api.html#_CPPv4NK5cu | St6size_tEEENSt6vectorIdEENSt8opt |
| daq15ExecutionResult9serializeEv) | ionalIN5cudaq13kraus_channelEEE), |
| -   [cudaq::fermion_handler (C++  |     [\[1\]](api/languages/cpp_a   |
|     c                             | pi.html#_CPPv4N5cudaq5ptsbe16Trac |
| lass)](api/languages/cpp_api.html | eInstruction16TraceInstructionEv) |
| #_CPPv4N5cudaq15fermion_handlerE) | -   [cud                          |
| -   [cudaq::fermion_op (C++       | aq::ptsbe::TraceInstruction::type |
|     type)](api/languages/cpp_api  |     (C++                          |
| .html#_CPPv4N5cudaq10fermion_opE) |     member)](api/l                |
| -   [cudaq::fermion_op_term (C++  | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq5ptsbe16TraceInstruction4typeE) |
| type)](api/languages/cpp_api.html | -   [c                            |
| #_CPPv4N5cudaq15fermion_op_termE) | udaq::ptsbe::TraceInstructionType |
| -   [cudaq::FermioniqBaseQPU (C++ |     (C++                          |
|     cl                            |     enum)](api/                   |
| ass)](api/languages/cpp_api.html# | languages/cpp_api.html#_CPPv4N5cu |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | daq5ptsbe20TraceInstructionTypeE) |
| -   [cudaq::get_state (C++        | -   [cudaq::                      |
|                                   | ptsbe::TraceInstructionType::Gate |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4I0DpEN5cudaq9get_st |     enumerator)](api/langu        |
| ateEDaRR13QuantumKernelDpRR4Args) | ages/cpp_api.html#_CPPv4N5cudaq5p |
| -   [cudaq::gradient (C++         | tsbe20TraceInstructionType4GateE) |
|     class)](api/languages/cpp_    | -   [cudaq::ptsbe::               |
| api.html#_CPPv4N5cudaq8gradientE) | TraceInstructionType::Measurement |
| -   [cudaq::gradient::clone (C++  |     (C++                          |
|     fun                           |                                   |
| ction)](api/languages/cpp_api.htm |    enumerator)](api/languages/cpp |
| l#_CPPv4N5cudaq8gradient5cloneEv) | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| -   [cudaq::gradient::compute     | aceInstructionType11MeasurementE) |
|     (C++                          | -   [cudaq::p                     |
|     function)](api/language       | tsbe::TraceInstructionType::Noise |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     (C++                          |
| ient7computeERKNSt6vectorIdEERKNS |     enumerator)](api/langua       |
| t8functionIFdNSt6vectorIdEEEEEd), | ges/cpp_api.html#_CPPv4N5cudaq5pt |
|     [\[1\]](ap                    | sbe20TraceInstructionType5NoiseE) |
| i/languages/cpp_api.html#_CPPv4N5 | -   [                             |
| cudaq8gradient7computeERKNSt6vect | cudaq::ptsbe::TrajectoryPredicate |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradient::gradient    |     type)](api                    |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/lang           | udaq5ptsbe19TrajectoryPredicateE) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -   [cudaq::QPU (C++              |
| daq8gradient8gradientER7KernelT), |     class)](api/languages         |
|                                   | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|    [\[1\]](api/languages/cpp_api. | -   [cudaq::QPU::beginExecution   |
| html#_CPPv4I00EN5cudaq8gradient8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function                      |
|     [\[2\                         | )](api/languages/cpp_api.html#_CP |
| ]](api/languages/cpp_api.html#_CP | Pv4N5cudaq3QPU14beginExecutionEv) |
| Pv4I00EN5cudaq8gradient8gradientE | -   [cuda                         |
| RR13QuantumKernelRR10ArgsMapper), | q::QPU::configureExecutionContext |
|     [\[3                          |     (C++                          |
| \]](api/languages/cpp_api.html#_C |     funct                         |
| PPv4N5cudaq8gradient8gradientERRN | ion)](api/languages/cpp_api.html# |
| St8functionIFvNSt6vectorIdEEEEE), | _CPPv4NK5cudaq3QPU25configureExec |
|     [\[                           | utionContextER16ExecutionContext) |
| 4\]](api/languages/cpp_api.html#_ | -   [cudaq::QPU::endExecution     |
| CPPv4N5cudaq8gradient8gradientEv) |     (C++                          |
| -   [cudaq::gradient::setArgs     |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     fu                            | CPPv4N5cudaq3QPU12endExecutionEv) |
| nction)](api/languages/cpp_api.ht | -   [cudaq::QPU::enqueue (C++     |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     function)](ap                 |
| tArgsEvR13QuantumKernelDpRR4Args) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [cudaq::gradient::setKernel   | cudaq3QPU7enqueueER11QuantumTask) |
|     (C++                          | -   [cud                          |
|     function)](api/languages/c    | aq::QPU::finalizeExecutionContext |
| pp_api.html#_CPPv4I0EN5cudaq8grad |     (C++                          |
| ient9setKernelEvR13QuantumKernel) |     func                          |
| -   [cud                          | tion)](api/languages/cpp_api.html |
| aq::gradients::central_difference | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     class)](api/la                | -   [cudaq::QPU::getConnectivity  |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9gradients18central_differenceE) |     function)                     |
| -   [cudaq::gra                   | ](api/languages/cpp_api.html#_CPP |
| dients::central_difference::clone | v4N5cudaq3QPU15getConnectivityEv) |
|     (C++                          | -                                 |
|     function)](api/languages      | [cudaq::QPU::getExecutionThreadId |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18central_difference5cloneEv) |     function)](api/               |
| -   [cudaq::gradi                 | languages/cpp_api.html#_CPPv4NK5c |
| ents::central_difference::compute | udaq3QPU20getExecutionThreadIdEv) |
|     (C++                          | -   [cudaq::QPU::getNumQubits     |
|     function)](                   |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     functi                        |
| N5cudaq9gradients18central_differ | on)](api/languages/cpp_api.html#_ |
| ence7computeERKNSt6vectorIdEERKNS | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [                             |
|                                   | cudaq::QPU::getRemoteCapabilities |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq9gradients18cent |     function)](api/l              |
| ral_difference7computeERKNSt6vect | anguages/cpp_api.html#_CPPv4NK5cu |
| orIdEERNSt6vectorIdEERK7spin_opd) | daq3QPU21getRemoteCapabilitiesEv) |
| -   [cudaq::gradie                | -   [cudaq::QPU::isEmulated (C++  |
| nts::central_difference::gradient |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|     functio                       | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq::QPU::isSimulator (C++ |
| PPv4I00EN5cudaq9gradients18centra |     funct                         |
| l_difference8gradientER7KernelT), | ion)](api/languages/cpp_api.html# |
|     [\[1\]](api/langua            | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::QPU::launchKernel     |
| q9gradients18central_difference8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)](api/               |
|     [\[2\]](api/languages/cpp_    | languages/cpp_api.html#_CPPv4N5cu |
| api.html#_CPPv4I00EN5cudaq9gradie | daq3QPU12launchKernelERKNSt6strin |
| nts18central_difference8gradientE | gE15KernelThunkTypePvNSt8uint64_t |
| RR13QuantumKernelRR10ArgsMapper), | ENSt8uint64_tERKNSt6vectorIPvEE), |
|     [\[3\]](api/languages/cpp     |                                   |
| _api.html#_CPPv4N5cudaq9gradients |  [\[1\]](api/languages/cpp_api.ht |
| 18central_difference8gradientERRN | ml#_CPPv4N5cudaq3QPU12launchKerne |
| St8functionIFvNSt6vectorIdEEEEE), | lERKNSt6stringERKNSt6vectorIPvEE) |
|     [\[4\]](api/languages/cp      | -   [cudaq::QPU::onRandomSeedSet  |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18central_difference8gradientEv) |     function)](api/lang           |
| -   [cud                          | uages/cpp_api.html#_CPPv4N5cudaq3 |
| aq::gradients::forward_difference | QPU15onRandomSeedSetENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::QPU (C++         |
|     class)](api/la                |     functio                       |
| nguages/cpp_api.html#_CPPv4N5cuda | n)](api/languages/cpp_api.html#_C |
| q9gradients18forward_differenceE) | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| -   [cudaq::gra                   |                                   |
| dients::forward_difference::clone |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     function)](api/languages      |     [\[2\]](api/languages/cpp_    |
| /cpp_api.html#_CPPv4N5cudaq9gradi | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| ents18forward_difference5cloneEv) | -   [cudaq::QPU::setId (C++       |
| -   [cudaq::gradi                 |     function                      |
| ents::forward_difference::compute | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     function)](                   | -   [cudaq::QPU::setShots (C++    |
| api/languages/cpp_api.html#_CPPv4 |     f                             |
| N5cudaq9gradients18forward_differ | unction)](api/languages/cpp_api.h |
| ence7computeERKNSt6vectorIdEERKNS | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::                      |
|                                   | QPU::supportsExplicitMeasurements |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq9gradients18forw |     function)](api/languag        |
| ard_difference7computeERKNSt6vect | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| orIdEERNSt6vectorIdEERK7spin_opd) | 28supportsExplicitMeasurementsEv) |
| -   [cudaq::gradie                | -   [cudaq::QPU::\~QPU (C++       |
| nts::forward_difference::gradient |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
|     functio                       | -   [cudaq::QPUState (C++         |
| n)](api/languages/cpp_api.html#_C |     class)](api/languages/cpp_    |
| PPv4I00EN5cudaq9gradients18forwar | api.html#_CPPv4N5cudaq8QPUStateE) |
| d_difference8gradientER7KernelT), | -   [cudaq::qreg (C++             |
|     [\[1\]](api/langua            |     class)](api/lan               |
| ges/cpp_api.html#_CPPv4I00EN5cuda | guages/cpp_api.html#_CPPv4I_NSt6s |
| q9gradients18forward_difference8g | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::qreg::back (C++       |
|     [\[2\]](api/languages/cpp_    |     function)                     |
| api.html#_CPPv4I00EN5cudaq9gradie | ](api/languages/cpp_api.html#_CPP |
| nts18forward_difference8gradientE | v4N5cudaq4qreg4backENSt6size_tE), |
| RR13QuantumKernelRR10ArgsMapper), |     [\[1\]](api/languages/cpp_ap  |
|     [\[3\]](api/languages/cpp     | i.html#_CPPv4N5cudaq4qreg4backEv) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::qreg::begin (C++      |
| 18forward_difference8gradientERRN |                                   |
| St8functionIFvNSt6vectorIdEEEEE), |  function)](api/languages/cpp_api |
|     [\[4\]](api/languages/cp      | .html#_CPPv4N5cudaq4qreg5beginEv) |
| p_api.html#_CPPv4N5cudaq9gradient | -   [cudaq::qreg::clear (C++      |
| s18forward_difference8gradientEv) |                                   |
| -   [                             |  function)](api/languages/cpp_api |
| cudaq::gradients::parameter_shift | .html#_CPPv4N5cudaq4qreg5clearEv) |
|     (C++                          | -   [cudaq::qreg::front (C++      |
|     class)](api                   |     function)]                    |
| /languages/cpp_api.html#_CPPv4N5c | (api/languages/cpp_api.html#_CPPv |
| udaq9gradients15parameter_shiftE) | 4N5cudaq4qreg5frontENSt6size_tE), |
| -   [cudaq::                      |     [\[1\]](api/languages/cpp_api |
| gradients::parameter_shift::clone | .html#_CPPv4N5cudaq4qreg5frontEv) |
|     (C++                          | -   [cudaq::qreg::operator\[\]    |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     functi                        |
| adients15parameter_shift5cloneEv) | on)](api/languages/cpp_api.html#_ |
| -   [cudaq::gr                    | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| adients::parameter_shift::compute | -   [cudaq::qreg::qreg (C++       |
|     (C++                          |     function)                     |
|     function                      | ](api/languages/cpp_api.html#_CPP |
| )](api/languages/cpp_api.html#_CP | v4N5cudaq4qreg4qregENSt6size_tE), |
| Pv4N5cudaq9gradients15parameter_s |     [\[1\]](api/languages/cpp_ap  |
| hift7computeERKNSt6vectorIdEERKNS | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::qreg::size (C++       |
|     [\[1\]](api/languages/cpp_ap  |                                   |
| i.html#_CPPv4N5cudaq9gradients15p |  function)](api/languages/cpp_api |
| arameter_shift7computeERKNSt6vect | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::qreg::slice (C++      |
| -   [cudaq::gra                   |     function)](api/langu          |
| dients::parameter_shift::gradient | ages/cpp_api.html#_CPPv4N5cudaq4q |
|     (C++                          | reg5sliceENSt6size_tENSt6size_tE) |
|     func                          | -   [cudaq::qreg::value_type (C++ |
| tion)](api/languages/cpp_api.html |                                   |
| #_CPPv4I00EN5cudaq9gradients15par | type)](api/languages/cpp_api.html |
| ameter_shift8gradientER7KernelT), | #_CPPv4N5cudaq4qreg10value_typeE) |
|     [\[1\]](api/lan               | -   [cudaq::qspan (C++            |
| guages/cpp_api.html#_CPPv4I00EN5c |     class)](api/lang              |
| udaq9gradients15parameter_shift8g | uages/cpp_api.html#_CPPv4I_NSt6si |
| radientER7KernelTRR10ArgsMapper), | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|     [\[2\]](api/languages/c       | -   [cudaq::QuakeValue (C++       |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     class)](api/languages/cpp_api |
| dients15parameter_shift8gradientE | .html#_CPPv4N5cudaq10QuakeValueE) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::Q                     |
|     [\[3\]](api/languages/        | uakeValue::canValidateNumElements |
| cpp_api.html#_CPPv4N5cudaq9gradie |     (C++                          |
| nts15parameter_shift8gradientERRN |     function)](api/languages      |
| St8functionIFvNSt6vectorIdEEEEE), | /cpp_api.html#_CPPv4N5cudaq10Quak |
|     [\[4\]](api/languages         | eValue22canValidateNumElementsEv) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -                                 |
| ents15parameter_shift8gradientEv) |  [cudaq::QuakeValue::constantSize |
| -   [cudaq::kernel_builder (C++   |     (C++                          |
|     clas                          |     function)](api                |
| s)](api/languages/cpp_api.html#_C | /languages/cpp_api.html#_CPPv4N5c |
| PPv4IDpEN5cudaq14kernel_builderE) | udaq10QuakeValue12constantSizeEv) |
| -   [c                            | -   [cudaq::QuakeValue::dump (C++ |
| udaq::kernel_builder::constantVal |     function)](api/lan            |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     function)](api/la             | 10QuakeValue4dumpERNSt7ostreamE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\                            |
| q14kernel_builder11constantValEd) | [1\]](api/languages/cpp_api.html# |
| -   [cu                           | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| daq::kernel_builder::getArguments | -   [cudaq                        |
|     (C++                          | ::QuakeValue::getRequiredElements |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/langua         |
| 14kernel_builder12getArgumentsEv) | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| -   [cu                           | uakeValue19getRequiredElementsEv) |
| daq::kernel_builder::getNumParams | -   [cudaq::QuakeValue::getValue  |
|     (C++                          |     (C++                          |
|     function)](api/lan            |     function)]                    |
| guages/cpp_api.html#_CPPv4N5cudaq | (api/languages/cpp_api.html#_CPPv |
| 14kernel_builder12getNumParamsEv) | 4NK5cudaq10QuakeValue8getValueEv) |
| -   [c                            | -   [cudaq::QuakeValue::inverse   |
| udaq::kernel_builder::isArgStdVec |     (C++                          |
|     (C++                          |     function)                     |
|     function)](api/languages/cp   | ](api/languages/cpp_api.html#_CPP |
| p_api.html#_CPPv4N5cudaq14kernel_ | v4NK5cudaq10QuakeValue7inverseEv) |
| builder11isArgStdVecENSt6size_tE) | -   [cudaq::QuakeValue::isStdVec  |
| -   [cuda                         |     (C++                          |
| q::kernel_builder::kernel_builder |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/languages/cpp_ | v4N5cudaq10QuakeValue8isStdVecEv) |
| api.html#_CPPv4N5cudaq14kernel_bu | -                                 |
| ilder14kernel_builderERNSt6vector |    [cudaq::QuakeValue::operator\* |
| IN7details17KernelBuilderTypeEEE) |     (C++                          |
| -   [cudaq::kernel_builder::name  |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     function)                     | udaq10QuakeValuemlE10QuakeValue), |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4N5cudaq14kernel_builder4nameEv) | [\[1\]](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
|    [cudaq::kernel_builder::qalloc | -   [cudaq::QuakeValue::operator+ |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api                |
| s/cpp_api.html#_CPPv4N5cudaq14ker | /languages/cpp_api.html#_CPPv4N5c |
| nel_builder6qallocE10QuakeValue), | udaq10QuakeValueplE10QuakeValue), |
|     [\[1\]](api/language          |     [                             |
| s/cpp_api.html#_CPPv4N5cudaq14ker | \[1\]](api/languages/cpp_api.html |
| nel_builder6qallocEKNSt6size_tE), | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     [\[2                          |                                   |
| \]](api/languages/cpp_api.html#_C | [\[2\]](api/languages/cpp_api.htm |
| PPv4N5cudaq14kernel_builder6qallo | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| cERNSt6vectorINSt7complexIdEEEE), | -   [cudaq::QuakeValue::operator- |
|     [\[3\]](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api                |
| N5cudaq14kernel_builder6qallocEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::kernel_builder::swap  | udaq10QuakeValuemiE10QuakeValue), |
|     (C++                          |     [                             |
|     function)](api/language       | \[1\]](api/languages/cpp_api.html |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| 4kernel_builder4swapEvRK10QuakeVa |     [                             |
| lueRK10QuakeValueRK10QuakeValue), | \[2\]](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| [\[1\]](api/languages/cpp_api.htm |                                   |
| l#_CPPv4I00EN5cudaq14kernel_build | [\[3\]](api/languages/cpp_api.htm |
| er4swapEvRKNSt6vectorI10QuakeValu | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| eEERK10QuakeValueRK10QuakeValue), | -   [cudaq::QuakeValue::operator/ |
|                                   |     (C++                          |
| [\[2\]](api/languages/cpp_api.htm |     function)](api                |
| l#_CPPv4N5cudaq14kernel_builder4s | /languages/cpp_api.html#_CPPv4N5c |
| wapERK10QuakeValueRK10QuakeValue) | udaq10QuakeValuedvE10QuakeValue), |
| -   [cudaq::KernelExecutionTask   |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     type                          | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| )](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq19KernelExecutionTaskE) |  [cudaq::QuakeValue::operator\[\] |
| -   [cudaq::KernelThunkResultType |     (C++                          |
|     (C++                          |     function)](api                |
|     struct)]                      | /languages/cpp_api.html#_CPPv4N5c |
| (api/languages/cpp_api.html#_CPPv | udaq10QuakeValueixEKNSt6size_tE), |
| 4N5cudaq21KernelThunkResultTypeE) |     [\[1\]](api/                  |
| -   [cudaq::KernelThunkType (C++  | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq10QuakeValueixERK10QuakeValue) |
| type)](api/languages/cpp_api.html | -                                 |
| #_CPPv4N5cudaq15KernelThunkTypeE) |    [cudaq::QuakeValue::QuakeValue |
| -   [cudaq::kraus_channel (C++    |     (C++                          |
|                                   |     function)](api/languag        |
|  class)](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| ml#_CPPv4N5cudaq13kraus_channelE) | akeValue10QuakeValueERN4mlir20Imp |
| -   [cudaq::kraus_channel::empty  | licitLocOpBuilderEN4mlir5ValueE), |
|     (C++                          |     [\[1\]                        |
|     function)]                    | ](api/languages/cpp_api.html#_CPP |
| (api/languages/cpp_api.html#_CPPv | v4N5cudaq10QuakeValue10QuakeValue |
| 4NK5cudaq13kraus_channel5emptyEv) | ERN4mlir20ImplicitLocOpBuilderEd) |
| -   [cudaq::kraus_c               | -   [cudaq::QuakeValue::size (C++ |
| hannel::generateUnitaryParameters |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq10QuakeValue4sizeEv) |
|    function)](api/languages/cpp_a | -   [cudaq::QuakeValue::slice     |
| pi.html#_CPPv4N5cudaq13kraus_chan |     (C++                          |
| nel25generateUnitaryParametersEv) |     function)](api/languages/cpp_ |
| -                                 | api.html#_CPPv4N5cudaq10QuakeValu |
|    [cudaq::kraus_channel::get_ops | e5sliceEKNSt6size_tEKNSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_platform (C++ |
|     function)](a                  |     cl                            |
| pi/languages/cpp_api.html#_CPPv4N | ass)](api/languages/cpp_api.html# |
| K5cudaq13kraus_channel7get_opsEv) | _CPPv4N5cudaq16quantum_platformE) |
| -   [cud                          | -   [cudaq:                       |
| aq::kraus_channel::identity_flags | :quantum_platform::beginExecution |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/languag        |
| guages/cpp_api.html#_CPPv4N5cudaq | es/cpp_api.html#_CPPv4N5cudaq16qu |
| 13kraus_channel14identity_flagsE) | antum_platform14beginExecutionEv) |
| -   [cud                          | -   [cudaq::quantum_pl            |
| aq::kraus_channel::is_identity_op | atform::configureExecutionContext |
|     (C++                          |     (C++                          |
|                                   |     function)](api/lang           |
|    function)](api/languages/cpp_a | uages/cpp_api.html#_CPPv4NK5cudaq |
| pi.html#_CPPv4NK5cudaq13kraus_cha | 16quantum_platform25configureExec |
| nnel14is_identity_opENSt6size_tE) | utionContextER16ExecutionContext) |
| -   [cudaq::                      | -   [cuda                         |
| kraus_channel::is_unitary_mixture | q::quantum_platform::connectivity |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/langu          |
| /cpp_api.html#_CPPv4NK5cudaq13kra | ages/cpp_api.html#_CPPv4N5cudaq16 |
| us_channel18is_unitary_mixtureEv) | quantum_platform12connectivityEv) |
| -   [cu                           | -   [cuda                         |
| daq::kraus_channel::kraus_channel | q::quantum_platform::endExecution |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](api/langu          |
| uages/cpp_api.html#_CPPv4IDpEN5cu | ages/cpp_api.html#_CPPv4N5cudaq16 |
| daq13kraus_channel13kraus_channel | quantum_platform12endExecutionEv) |
| EDpRRNSt16initializer_listI1TEE), | -   [cudaq::q                     |
|                                   | uantum_platform::enqueueAsyncTask |
|  [\[1\]](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13kraus_channel13 |     function)](api/languages/     |
| kraus_channelERK13kraus_channel), | cpp_api.html#_CPPv4N5cudaq16quant |
|     [\[2\]                        | um_platform16enqueueAsyncTaskEKNS |
| ](api/languages/cpp_api.html#_CPP | t6size_tER19KernelExecutionTask), |
| v4N5cudaq13kraus_channel13kraus_c |     [\[1\]](api/languag           |
| hannelERKNSt6vectorI8kraus_opEE), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     [\[3\]                        | antum_platform16enqueueAsyncTaskE |
| ](api/languages/cpp_api.html#_CPP | KNSt6size_tERNSt8functionIFvvEEE) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::quantum_p             |
| hannelERRNSt6vectorI8kraus_opEE), | latform::finalizeExecutionContext |
|     [\[4\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/languages/c    |
| 13kraus_channel13kraus_channelEv) | pp_api.html#_CPPv4NK5cudaq16quant |
| -                                 | um_platform24finalizeExecutionCon |
| [cudaq::kraus_channel::noise_type | textERN5cudaq16ExecutionContextE) |
|     (C++                          | -   [cudaq::qua                   |
|     member)](api                  | ntum_platform::get_codegen_config |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq13kraus_channel10noise_typeE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4N5cudaq16quantu |
|   [cudaq::kraus_channel::op_names | m_platform18get_codegen_configEv) |
|     (C++                          | -   [cuda                         |
|     member)](                     | q::quantum_platform::get_exec_ctx |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq13kraus_channel8op_namesE) |     function)](api/langua         |
| -                                 | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|  [cudaq::kraus_channel::operator= | quantum_platform12get_exec_ctxEv) |
|     (C++                          | -   [c                            |
|     function)](api/langua         | udaq::quantum_platform::get_noise |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     (C++                          |
| raus_channelaSERK13kraus_channel) |     function)](api/languages/c    |
| -   [c                            | pp_api.html#_CPPv4N5cudaq16quantu |
| udaq::kraus_channel::operator\[\] | m_platform9get_noiseENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/l              | :quantum_platform::get_num_qubits |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq13kraus_channelixEKNSt6size_tE) |                                   |
| -                                 | function)](api/languages/cpp_api. |
| [cudaq::kraus_channel::parameters | html#_CPPv4NK5cudaq16quantum_plat |
|     (C++                          | form14get_num_qubitsENSt6size_tE) |
|     member)](api                  | -   [cudaq::quantum_              |
| /languages/cpp_api.html#_CPPv4N5c | platform::get_remote_capabilities |
| udaq13kraus_channel10parametersE) |     (C++                          |
| -   [cudaq::krau                  |     function)                     |
| s_channel::populateDefaultOpNames | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq16quantum_platform23get |
|     function)](api/languages/cp   | _remote_capabilitiesENSt6size_tE) |
| p_api.html#_CPPv4N5cudaq13kraus_c | -   [cudaq::qua                   |
| hannel22populateDefaultOpNamesEv) | ntum_platform::get_runtime_target |
| -   [cu                           |     (C++                          |
| daq::kraus_channel::probabilities |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4NK5cudaq16quantu |
|     member)](api/la               | m_platform18get_runtime_targetEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cuda                         |
| q13kraus_channel13probabilitiesE) | q::quantum_platform::getLogStream |
| -                                 |     (C++                          |
|  [cudaq::kraus_channel::push_back |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     function)](api                | quantum_platform12getLogStreamEv) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cud                          |
| udaq13kraus_channel9push_backE8kr | aq::quantum_platform::is_emulated |
| aus_opNSt8optionalINSt6stringEEE) |     (C++                          |
| -   [cudaq::kraus_channel::size   |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|     function)                     | pi.html#_CPPv4NK5cudaq16quantum_p |
| ](api/languages/cpp_api.html#_CPP | latform11is_emulatedENSt6size_tE) |
| v4NK5cudaq13kraus_channel4sizeEv) | -   [c                            |
| -   [                             | udaq::quantum_platform::is_remote |
| cudaq::kraus_channel::unitary_ops |     (C++                          |
|     (C++                          |     function)](api/languages/cp   |
|     member)](api/                 | p_api.html#_CPPv4NK5cudaq16quantu |
| languages/cpp_api.html#_CPPv4N5cu | m_platform9is_remoteENSt6size_tE) |
| daq13kraus_channel11unitary_opsE) | -   [cuda                         |
| -   [cudaq::kraus_op (C++         | q::quantum_platform::is_simulator |
|     struct)](api/languages/cpp_   |     (C++                          |
| api.html#_CPPv4N5cudaq8kraus_opE) |                                   |
| -   [cudaq::kraus_op::adjoint     |   function)](api/languages/cpp_ap |
|     (C++                          | i.html#_CPPv4NK5cudaq16quantum_pl |
|     functi                        | atform12is_simulatorENSt6size_tE) |
| on)](api/languages/cpp_api.html#_ | -   [c                            |
| CPPv4NK5cudaq8kraus_op7adjointEv) | udaq::quantum_platform::launchVQE |
| -   [cudaq::kraus_op::data (C++   |     (C++                          |
|                                   |     function)](                   |
|  member)](api/languages/cpp_api.h | api/languages/cpp_api.html#_CPPv4 |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | N5cudaq16quantum_platform9launchV |
| -   [cudaq::kraus_op::kraus_op    | QEEKNSt6stringEPKvPN5cudaq8gradie |
|     (C++                          | ntERKN5cudaq7spin_opERN5cudaq9opt |
|     func                          | imizerEKiKNSt6size_tENSt6size_tE) |
| tion)](api/languages/cpp_api.html | -   [cudaq:                       |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | :quantum_platform::list_platforms |
| opERRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     function)](api/languag        |
|  [\[1\]](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | antum_platform14list_platformsEv) |
| pENSt6vectorIN5cudaq7complexEEE), | -                                 |
|     [\[2\]](api/l                 |    [cudaq::quantum_platform::name |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq8kraus_op8kraus_opERK8kraus_op) |     function)](a                  |
| -   [cudaq::kraus_op::nCols (C++  | pi/languages/cpp_api.html#_CPPv4N |
|                                   | K5cudaq16quantum_platform4nameEv) |
| member)](api/languages/cpp_api.ht | -   [                             |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | cudaq::quantum_platform::num_qpus |
| -   [cudaq::kraus_op::nRows (C++  |     (C++                          |
|                                   |     function)](api/l              |
| member)](api/languages/cpp_api.ht | anguages/cpp_api.html#_CPPv4NK5cu |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | daq16quantum_platform8num_qpusEv) |
| -   [cudaq::kraus_op::operator=   | -   [cudaq::                      |
|     (C++                          | quantum_platform::onRandomSeedSet |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4N5cudaq8kraus_opaSERK8kraus_op) | function)](api/languages/cpp_api. |
| -   [cudaq::kraus_op::precision   | html#_CPPv4N5cudaq16quantum_platf |
|     (C++                          | orm15onRandomSeedSetENSt6size_tE) |
|     memb                          | -   [cudaq:                       |
| er)](api/languages/cpp_api.html#_ | :quantum_platform::reset_exec_ctx |
| CPPv4N5cudaq8kraus_op9precisionE) |     (C++                          |
| -   [cudaq::KrausSelection (C++   |     function)](api/languag        |
|     s                             | es/cpp_api.html#_CPPv4N5cudaq16qu |
| truct)](api/languages/cpp_api.htm | antum_platform14reset_exec_ctxEv) |
| l#_CPPv4N5cudaq14KrausSelectionE) | -   [cud                          |
| -   [cudaq:                       | aq::quantum_platform::reset_noise |
| :KrausSelection::circuit_location |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     member)](api/langua           | api.html#_CPPv4N5cudaq16quantum_p |
| ges/cpp_api.html#_CPPv4N5cudaq14K | latform11reset_noiseENSt6size_tE) |
| rausSelection16circuit_locationE) | -   [cudaq:                       |
| -                                 | :quantum_platform::resetLogStream |
|  [cudaq::KrausSelection::is_error |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     member)](a                    | es/cpp_api.html#_CPPv4N5cudaq16qu |
| pi/languages/cpp_api.html#_CPPv4N | antum_platform14resetLogStreamEv) |
| 5cudaq14KrausSelection8is_errorE) | -   [cuda                         |
| -   [cudaq::Kra                   | q::quantum_platform::set_exec_ctx |
| usSelection::kraus_operator_index |     (C++                          |
|     (C++                          |     funct                         |
|     member)](api/languages/       | ion)](api/languages/cpp_api.html# |
| cpp_api.html#_CPPv4N5cudaq14Kraus | _CPPv4N5cudaq16quantum_platform12 |
| Selection20kraus_operator_indexE) | set_exec_ctxEP16ExecutionContext) |
| -   [cuda                         | -   [c                            |
| q::KrausSelection::KrausSelection | udaq::quantum_platform::set_noise |
|     (C++                          |     (C++                          |
|     function)](a                  |     function                      |
| pi/languages/cpp_api.html#_CPPv4N | )](api/languages/cpp_api.html#_CP |
| 5cudaq14KrausSelection14KrausSele | Pv4N5cudaq16quantum_platform9set_ |
| ctionENSt6size_tENSt6vectorINSt6s | noiseEPK11noise_modelNSt6size_tE) |
| ize_tEEENSt6stringENSt6size_tEb), | -   [cuda                         |
|     [\[1\]](api/langu             | q::quantum_platform::setLogStream |
| ages/cpp_api.html#_CPPv4N5cudaq14 |     (C++                          |
| KrausSelection14KrausSelectionEv) |                                   |
| -                                 |  function)](api/languages/cpp_api |
|   [cudaq::KrausSelection::op_name | .html#_CPPv4N5cudaq16quantum_plat |
|     (C++                          | form12setLogStreamERNSt7ostreamE) |
|     member)](                     | -   [cudaq::quantum_platfor       |
| api/languages/cpp_api.html#_CPPv4 | m::supports_explicit_measurements |
| N5cudaq14KrausSelection7op_nameE) |     (C++                          |
| -   [                             |     function)](api/l              |
| cudaq::KrausSelection::operator== | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq16quantum_platform30supports_e |
|     function)](api/languages      | xplicit_measurementsENSt6size_tE) |
| /cpp_api.html#_CPPv4NK5cudaq14Kra | -   [cudaq::quantum_pla           |
| usSelectioneqERK14KrausSelection) | tform::supports_task_distribution |
| -                                 |     (C++                          |
|    [cudaq::KrausSelection::qubits |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     member)]                      | ml#_CPPv4NK5cudaq16quantum_platfo |
| (api/languages/cpp_api.html#_CPPv | rm26supports_task_distributionEv) |
| 4N5cudaq14KrausSelection6qubitsE) | -   [cudaq::quantum               |
| -   [cudaq::KrausTrajectory (C++  | _platform::with_execution_context |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |     function)                     |
| #_CPPv4N5cudaq15KrausTrajectoryE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4I0DpEN5cudaq16quantum_platform2 |
|  [cudaq::KrausTrajectory::builder | 2with_execution_contextEDaR16Exec |
|     (C++                          | utionContextRR8CallableDpRR4Args) |
|     function)](ap                 | -   [cudaq::QuantumTask (C++      |
| i/languages/cpp_api.html#_CPPv4N5 |     type)](api/languages/cpp_api. |
| cudaq15KrausTrajectory7builderEv) | html#_CPPv4N5cudaq11QuantumTaskE) |
| -   [cu                           | -   [cudaq::qubit (C++            |
| daq::KrausTrajectory::countErrors |     type)](api/languages/c        |
|     (C++                          | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     function)](api/lang           | -   [cudaq::QubitConnectivity     |
| uages/cpp_api.html#_CPPv4NK5cudaq |     (C++                          |
| 15KrausTrajectory11countErrorsEv) |     ty                            |
| -   [                             | pe)](api/languages/cpp_api.html#_ |
| cudaq::KrausTrajectory::isOrdered | CPPv4N5cudaq17QubitConnectivityE) |
|     (C++                          | -   [cudaq::QubitEdge (C++        |
|     function)](api/l              |     type)](api/languages/cpp_a    |
| anguages/cpp_api.html#_CPPv4NK5cu | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| daq15KrausTrajectory9isOrderedEv) | -   [cudaq::qudit (C++            |
| -   [cudaq::                      |     clas                          |
| KrausTrajectory::kraus_selections | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     member)](api/languag          | -   [cudaq::qudit::qudit (C++     |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |                                   |
| ausTrajectory16kraus_selectionsE) | function)](api/languages/cpp_api. |
| -   [cudaq:                       | html#_CPPv4N5cudaq5qudit5quditEv) |
| :KrausTrajectory::KrausTrajectory | -   [cudaq::qvector (C++          |
|     (C++                          |     class)                        |
|     function                      | ](api/languages/cpp_api.html#_CPP |
| )](api/languages/cpp_api.html#_CP | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| Pv4N5cudaq15KrausTrajectory15Krau | -   [cudaq::qvector::back (C++    |
| sTrajectoryENSt6size_tENSt6vector |     function)](a                  |
| I14KrausSelectionEEdNSt6size_tE), | pi/languages/cpp_api.html#_CPPv4N |
|     [\[1\]](api/languag           | 5cudaq7qvector4backENSt6size_tE), |
| es/cpp_api.html#_CPPv4N5cudaq15Kr |                                   |
| ausTrajectory15KrausTrajectoryEv) |   [\[1\]](api/languages/cpp_api.h |
| -   [cudaq::Kr                    | tml#_CPPv4N5cudaq7qvector4backEv) |
| ausTrajectory::measurement_counts | -   [cudaq::qvector::begin (C++   |
|     (C++                          |     fu                            |
|     member)](api/languages        | nction)](api/languages/cpp_api.ht |
| /cpp_api.html#_CPPv4N5cudaq15Krau | ml#_CPPv4N5cudaq7qvector5beginEv) |
| sTrajectory18measurement_countsE) | -   [cudaq::qvector::clear (C++   |
| -   [cud                          |     fu                            |
| aq::KrausTrajectory::multiplicity | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     member)](api/lan              | -   [cudaq::qvector::end (C++     |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 15KrausTrajectory12multiplicityE) | function)](api/languages/cpp_api. |
| -   [                             | html#_CPPv4N5cudaq7qvector3endEv) |
| cudaq::KrausTrajectory::num_shots | -   [cudaq::qvector::front (C++   |
|     (C++                          |     function)](ap                 |
|     member)](api                  | i/languages/cpp_api.html#_CPPv4N5 |
| /languages/cpp_api.html#_CPPv4N5c | cudaq7qvector5frontENSt6size_tE), |
| udaq15KrausTrajectory9num_shotsE) |                                   |
| -   [c                            |  [\[1\]](api/languages/cpp_api.ht |
| udaq::KrausTrajectory::operator== | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     (C++                          | -   [cudaq::qvector::operator=    |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4NK5cudaq15Kraus |     functio                       |
| TrajectoryeqERK15KrausTrajectory) | n)](api/languages/cpp_api.html#_C |
| -   [cu                           | PPv4N5cudaq7qvectoraSERK7qvector) |
| daq::KrausTrajectory::probability | -   [cudaq::qvector::operator\[\] |
|     (C++                          |     (C++                          |
|     member)](api/la               |     function)                     |
| nguages/cpp_api.html#_CPPv4N5cuda | ](api/languages/cpp_api.html#_CPP |
| q15KrausTrajectory11probabilityE) | v4N5cudaq7qvectorixEKNSt6size_tE) |
| -   [cuda                         | -   [cudaq::qvector::qvector (C++ |
| q::KrausTrajectory::trajectory_id |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     member)](api/lang             | daq7qvector7qvectorENSt6size_tE), |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     [\[1\]](a                     |
| 5KrausTrajectory13trajectory_idE) | pi/languages/cpp_api.html#_CPPv4N |
| -                                 | 5cudaq7qvector7qvectorERK5state), |
|   [cudaq::KrausTrajectory::weight |     [\[2\]](api                   |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](                     | udaq7qvector7qvectorERK7qvector), |
| api/languages/cpp_api.html#_CPPv4 |     [\[3\]](api/languages/cpp     |
| N5cudaq15KrausTrajectory6weightE) | _api.html#_CPPv4N5cudaq7qvector7q |
| -                                 | vectorERKNSt6vectorI7complexEEb), |
|    [cudaq::KrausTrajectoryBuilder |     [\[4\]](ap                    |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     class)](                      | cudaq7qvector7qvectorERR7qvector) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qvector::size (C++    |
| N5cudaq22KrausTrajectoryBuilderE) |     fu                            |
| -   [cud                          | nction)](api/languages/cpp_api.ht |
| aq::KrausTrajectoryBuilder::build | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
|     (C++                          | -   [cudaq::qvector::slice (C++   |
|     function)](api/lang           |     function)](api/language       |
| uages/cpp_api.html#_CPPv4NK5cudaq | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| 22KrausTrajectoryBuilder5buildEv) | tor5sliceENSt6size_tENSt6size_tE) |
| -   [cud                          | -   [cudaq::qvector::value_type   |
| aq::KrausTrajectoryBuilder::setId |     (C++                          |
|     (C++                          |     typ                           |
|     function)](api/languages/cpp  | e)](api/languages/cpp_api.html#_C |
| _api.html#_CPPv4N5cudaq22KrausTra | PPv4N5cudaq7qvector10value_typeE) |
| jectoryBuilder5setIdENSt6size_tE) | -   [cudaq::qview (C++            |
| -   [cudaq::Kraus                 |     clas                          |
| TrajectoryBuilder::setProbability | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     function)](api/languages/cpp  | -   [cudaq::qview::back (C++      |
| _api.html#_CPPv4N5cudaq22KrausTra |     function)                     |
| jectoryBuilder14setProbabilityEd) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::Krau                  | v4N5cudaq5qview4backENSt6size_tE) |
| sTrajectoryBuilder::setSelections | -   [cudaq::qview::begin (C++     |
|     (C++                          |                                   |
|     function)](api/languag        | function)](api/languages/cpp_api. |
| es/cpp_api.html#_CPPv4N5cudaq22Kr | html#_CPPv4N5cudaq5qview5beginEv) |
| ausTrajectoryBuilder13setSelectio | -   [cudaq::qview::end (C++       |
| nsENSt6vectorI14KrausSelectionEE) |                                   |
| -   [cudaq::matrix_callback (C++  |   function)](api/languages/cpp_ap |
|     c                             | i.html#_CPPv4N5cudaq5qview3endEv) |
| lass)](api/languages/cpp_api.html | -   [cudaq::qview::front (C++     |
| #_CPPv4N5cudaq15matrix_callbackE) |     function)](                   |
| -   [cudaq::matrix_handler (C++   | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq5qview5frontENSt6size_tE), |
| class)](api/languages/cpp_api.htm |                                   |
| l#_CPPv4N5cudaq14matrix_handlerE) |    [\[1\]](api/languages/cpp_api. |
| -   [cudaq::mat                   | html#_CPPv4N5cudaq5qview5frontEv) |
| rix_handler::commutation_behavior | -   [cudaq::qview::operator\[\]   |
|     (C++                          |     (C++                          |
|     struct)](api/languages/       |     functio                       |
| cpp_api.html#_CPPv4N5cudaq14matri | n)](api/languages/cpp_api.html#_C |
| x_handler20commutation_behaviorE) | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| -                                 | -   [cudaq::qview::qview (C++     |
|    [cudaq::matrix_handler::define |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function)](a                  | PPv4I0EN5cudaq5qview5qviewERR1R), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[1                          |
| 5cudaq14matrix_handler6defineENSt | \]](api/languages/cpp_api.html#_C |
| 6stringENSt6vectorINSt7int64_tEEE | PPv4N5cudaq5qview5qviewERK5qview) |
| RR15matrix_callbackRKNSt13unorder | -   [cudaq::qview::size (C++      |
| ed_mapINSt6stringENSt6stringEEE), |                                   |
|                                   | function)](api/languages/cpp_api. |
| [\[1\]](api/languages/cpp_api.htm | html#_CPPv4NK5cudaq5qview4sizeEv) |
| l#_CPPv4N5cudaq14matrix_handler6d | -   [cudaq::qview::slice (C++     |
| efineENSt6stringENSt6vectorINSt7i |     function)](api/langua         |
| nt64_tEEERR15matrix_callbackRR20d | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| iag_matrix_callbackRKNSt13unorder | iew5sliceENSt6size_tENSt6size_tE) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cudaq::qview::value_type     |
|     [\[2\]](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     t                             |
| N5cudaq14matrix_handler6defineENS | ype)](api/languages/cpp_api.html# |
| t6stringENSt6vectorINSt7int64_tEE | _CPPv4N5cudaq5qview10value_typeE) |
| ERR15matrix_callbackRRNSt13unorde | -   [cudaq::range (C++            |
| red_mapINSt6stringENSt6stringEEE) |     fun                           |
| -                                 | ction)](api/languages/cpp_api.htm |
|   [cudaq::matrix_handler::degrees | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
|     (C++                          | orI11ElementTypeEE11ElementType), |
|     function)](ap                 |     [\[1\]](api/languages/cpp_    |
| i/languages/cpp_api.html#_CPPv4NK | api.html#_CPPv4I0EN5cudaq5rangeEN |
| 5cudaq14matrix_handler7degreesEv) | St6vectorI11ElementTypeEE11Elemen |
| -                                 | tType11ElementType11ElementType), |
|  [cudaq::matrix_handler::displace |     [                             |
|     (C++                          | \[2\]](api/languages/cpp_api.html |
|     function)](api/language       | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cudaq::real (C++             |
| rix_handler8displaceENSt6size_tE) |     type)](api/languages/         |
| -   [cudaq::matrix                | cpp_api.html#_CPPv4N5cudaq4realE) |
| _handler::get_expected_dimensions | -   [cudaq::registry (C++         |
|     (C++                          |     type)](api/languages/cpp_     |
|                                   | api.html#_CPPv4N5cudaq8registryE) |
|    function)](api/languages/cpp_a | -                                 |
| pi.html#_CPPv4NK5cudaq14matrix_ha |  [cudaq::registry::RegisteredType |
| ndler23get_expected_dimensionsEv) |     (C++                          |
| -   [cudaq::matrix_ha             |     class)](api/                  |
| ndler::get_parameter_descriptions | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq8registry14RegisteredTypeE) |
|                                   | -   [cudaq::RemoteCapabilities    |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4NK5cudaq14matrix_handl |     struc                         |
| er26get_parameter_descriptionsEv) | t)](api/languages/cpp_api.html#_C |
| -   [c                            | PPv4N5cudaq18RemoteCapabilitiesE) |
| udaq::matrix_handler::instantiate | -   [cudaq::Remo                  |
|     (C++                          | teCapabilities::isRemoteSimulator |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     member)](api/languages/c      |
| 5cudaq14matrix_handler11instantia | pp_api.html#_CPPv4N5cudaq18Remote |
| teENSt6stringERKNSt6vectorINSt6si | Capabilities17isRemoteSimulatorE) |
| ze_tEEERK20commutation_behavior), | -   [cudaq::Remot                 |
|     [\[1\]](                      | eCapabilities::RemoteCapabilities |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14matrix_handler11instanti |     function)](api/languages/cpp  |
| ateENSt6stringERRNSt6vectorINSt6s | _api.html#_CPPv4N5cudaq18RemoteCa |
| ize_tEEERK20commutation_behavior) | pabilities18RemoteCapabilitiesEb) |
| -   [cuda                         | -   [cudaq:                       |
| q::matrix_handler::matrix_handler | :RemoteCapabilities::stateOverlap |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     member)](api/langua           |
| es/cpp_api.html#_CPPv4I0_NSt11ena | ges/cpp_api.html#_CPPv4N5cudaq18R |
| ble_if_tINSt12is_base_of_vI16oper | emoteCapabilities12stateOverlapE) |
| ator_handler1TEEbEEEN5cudaq14matr | -                                 |
| ix_handler14matrix_handlerERK1T), |   [cudaq::RemoteCapabilities::vqe |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4I0 |     member)](                     |
| _NSt11enable_if_tINSt12is_base_of | api/languages/cpp_api.html#_CPPv4 |
| _vI16operator_handler1TEEbEEEN5cu | N5cudaq18RemoteCapabilities3vqeE) |
| daq14matrix_handler14matrix_handl | -   [cudaq::RemoteSimulationState |
| erERK1TRK20commutation_behavior), |     (C++                          |
|     [\[2\]](api/languages/cpp_ap  |     class)]                       |
| i.html#_CPPv4N5cudaq14matrix_hand | (api/languages/cpp_api.html#_CPPv |
| ler14matrix_handlerENSt6size_tE), | 4N5cudaq21RemoteSimulationStateE) |
|     [\[3\]](api/                  | -   [cudaq::Resources (C++        |
| languages/cpp_api.html#_CPPv4N5cu |     class)](api/languages/cpp_a   |
| daq14matrix_handler14matrix_handl | pi.html#_CPPv4N5cudaq9ResourcesE) |
| erENSt6stringERKNSt6vectorINSt6si | -   [cudaq::run (C++              |
| ze_tEEERK20commutation_behavior), |     function)]                    |
|     [\[4\]](api/                  | (api/languages/cpp_api.html#_CPPv |
| languages/cpp_api.html#_CPPv4N5cu | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| daq14matrix_handler14matrix_handl | 5invoke_result_tINSt7decay_tI13Qu |
| erENSt6stringERRNSt6vectorINSt6si | antumKernelEEDpNSt7decay_tI4ARGSE |
| ze_tEEERK20commutation_behavior), | EEEEENSt6size_tERN5cudaq11noise_m |
|     [\                            | odelERR13QuantumKernelDpRR4ARGS), |
| [5\]](api/languages/cpp_api.html# |     [\[1\]](api/langu             |
| _CPPv4N5cudaq14matrix_handler14ma | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| trix_handlerERK14matrix_handler), | daq3runENSt6vectorINSt15invoke_re |
|     [                             | sult_tINSt7decay_tI13QuantumKerne |
| \[6\]](api/languages/cpp_api.html | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| #_CPPv4N5cudaq14matrix_handler14m | ize_tERR13QuantumKernelDpRR4ARGS) |
| atrix_handlerERR14matrix_handler) | -   [cudaq::run_async (C++        |
| -                                 |     functio                       |
|  [cudaq::matrix_handler::momentum | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     function)](api/language       | tureINSt6vectorINSt15invoke_resul |
| s/cpp_api.html#_CPPv4N5cudaq14mat | t_tINSt7decay_tI13QuantumKernelEE |
| rix_handler8momentumENSt6size_tE) | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| -                                 | ze_tENSt6size_tERN5cudaq11noise_m |
|    [cudaq::matrix_handler::number | odelERR13QuantumKernelDpRR4ARGS), |
|     (C++                          |     [\[1\]](api/la                |
|     function)](api/langua         | nguages/cpp_api.html#_CPPv4I0DpEN |
| ges/cpp_api.html#_CPPv4N5cudaq14m | 5cudaq9run_asyncENSt6futureINSt6v |
| atrix_handler6numberENSt6size_tE) | ectorINSt15invoke_result_tINSt7de |
| -                                 | cay_tI13QuantumKernelEEDpNSt7deca |
| [cudaq::matrix_handler::operator= | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4ARGS) |
|     fun                           | -   [cudaq::RuntimeTarget (C++    |
| ction)](api/languages/cpp_api.htm |                                   |
| l#_CPPv4I0_NSt11enable_if_tIXaant | struct)](api/languages/cpp_api.ht |
| NSt7is_sameI1T14matrix_handlerE5v | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| alueENSt12is_base_of_vI16operator | -   [cudaq::sample (C++           |
| _handler1TEEEbEEEN5cudaq14matrix_ |     function)](api/languages/c    |
| handleraSER14matrix_handlerRK1T), | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     [\[1\]](api/languages         | mpleE13sample_resultRK14sample_op |
| /cpp_api.html#_CPPv4N5cudaq14matr | tionsRR13QuantumKernelDpRR4Args), |
| ix_handleraSERK14matrix_handler), |     [\[1\                         |
|     [\[2\]](api/language          | ]](api/languages/cpp_api.html#_CP |
| s/cpp_api.html#_CPPv4N5cudaq14mat | Pv4I0DpEN5cudaq6sampleE13sample_r |
| rix_handleraSERR14matrix_handler) | esultRR13QuantumKernelDpRR4Args), |
| -   [                             |     [\                            |
| cudaq::matrix_handler::operator== | [2\]](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|     function)](api/languages      | ize_tERR13QuantumKernelDpRR4Args) |
| /cpp_api.html#_CPPv4NK5cudaq14mat | -   [cudaq::sample_options (C++   |
| rix_handlereqERK14matrix_handler) |     s                             |
| -                                 | truct)](api/languages/cpp_api.htm |
|    [cudaq::matrix_handler::parity | l#_CPPv4N5cudaq14sample_optionsE) |
|     (C++                          | -   [cudaq::sample_result (C++    |
|     function)](api/langua         |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq14m |  class)](api/languages/cpp_api.ht |
| atrix_handler6parityENSt6size_tE) | ml#_CPPv4N5cudaq13sample_resultE) |
| -                                 | -   [cudaq::sample_result::append |
|  [cudaq::matrix_handler::position |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     function)](api/language       | api.html#_CPPv4N5cudaq13sample_re |
| s/cpp_api.html#_CPPv4N5cudaq14mat | sult6appendERK15ExecutionResultb) |
| rix_handler8positionENSt6size_tE) | -   [cudaq::sample_result::begin  |
| -   [cudaq::                      |     (C++                          |
| matrix_handler::remove_definition |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     fu                            | 4N5cudaq13sample_result5beginEv), |
| nction)](api/languages/cpp_api.ht |     [\[1\]]                       |
| ml#_CPPv4N5cudaq14matrix_handler1 | (api/languages/cpp_api.html#_CPPv |
| 7remove_definitionERKNSt6stringE) | 4NK5cudaq13sample_result5beginEv) |
| -                                 | -   [cudaq::sample_result::cbegin |
|   [cudaq::matrix_handler::squeeze |     (C++                          |
|     (C++                          |     function)](                   |
|     function)](api/languag        | api/languages/cpp_api.html#_CPPv4 |
| es/cpp_api.html#_CPPv4N5cudaq14ma | NK5cudaq13sample_result6cbeginEv) |
| trix_handler7squeezeENSt6size_tE) | -   [cudaq::sample_result::cend   |
| -   [cudaq::m                     |     (C++                          |
| atrix_handler::to_diagonal_matrix |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/lang           | v4NK5cudaq13sample_result4cendEv) |
| uages/cpp_api.html#_CPPv4NK5cudaq | -   [cudaq::sample_result::clear  |
| 14matrix_handler18to_diagonal_mat |     (C++                          |
| rixERNSt13unordered_mapINSt6size_ |     function)                     |
| tENSt7int64_tEEERKNSt13unordered_ | ](api/languages/cpp_api.html#_CPP |
| mapINSt6stringENSt7complexIdEEEE) | v4N5cudaq13sample_result5clearEv) |
| -                                 | -   [cudaq::sample_result::count  |
| [cudaq::matrix_handler::to_matrix |     (C++                          |
|     (C++                          |     function)](                   |
|     function)                     | api/languages/cpp_api.html#_CPPv4 |
| ](api/languages/cpp_api.html#_CPP | NK5cudaq13sample_result5countENSt |
| v4NK5cudaq14matrix_handler9to_mat | 11string_viewEKNSt11string_viewE) |
| rixERNSt13unordered_mapINSt6size_ | -   [                             |
| tENSt7int64_tEEERKNSt13unordered_ | cudaq::sample_result::deserialize |
| mapINSt6stringENSt7complexIdEEEE) |     (C++                          |
| -                                 |     functio                       |
| [cudaq::matrix_handler::to_string | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq13sample_result11deser |
|     function)](api/               | ializeERNSt6vectorINSt6size_tEEE) |
| languages/cpp_api.html#_CPPv4NK5c | -   [cudaq::sample_result::dump   |
| udaq14matrix_handler9to_stringEb) |     (C++                          |
| -                                 |     function)](api/languag        |
| [cudaq::matrix_handler::unique_id | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     (C++                          | ample_result4dumpERNSt7ostreamE), |
|     function)](api/               |     [\[1\]                        |
| languages/cpp_api.html#_CPPv4NK5c | ](api/languages/cpp_api.html#_CPP |
| udaq14matrix_handler9unique_idEv) | v4NK5cudaq13sample_result4dumpEv) |
| -   [cudaq:                       | -   [cudaq::sample_result::end    |
| :matrix_handler::\~matrix_handler |     (C++                          |
|     (C++                          |     function                      |
|     functi                        | )](api/languages/cpp_api.html#_CP |
| on)](api/languages/cpp_api.html#_ | Pv4N5cudaq13sample_result3endEv), |
| CPPv4N5cudaq14matrix_handlerD0Ev) |     [\[1\                         |
| -   [cudaq::matrix_op (C++        | ]](api/languages/cpp_api.html#_CP |
|     type)](api/languages/cpp_a    | Pv4NK5cudaq13sample_result3endEv) |
| pi.html#_CPPv4N5cudaq9matrix_opE) | -   [                             |
| -   [cudaq::matrix_op_term (C++   | cudaq::sample_result::expectation |
|                                   |     (C++                          |
|  type)](api/languages/cpp_api.htm |     f                             |
| l#_CPPv4N5cudaq14matrix_op_termE) | unction)](api/languages/cpp_api.h |
| -                                 | tml#_CPPv4NK5cudaq13sample_result |
|    [cudaq::mdiag_operator_handler | 11expectationEKNSt11string_viewE) |
|     (C++                          | -   [c                            |
|     class)](                      | udaq::sample_result::get_marginal |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22mdiag_operator_handlerE) |     function)](api/languages/cpp_ |
| -   [cudaq::mpi (C++              | api.html#_CPPv4NK5cudaq13sample_r |
|     type)](api/languages          | esult12get_marginalERKNSt6vectorI |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | NSt6size_tEEEKNSt11string_viewE), |
| -   [cudaq::mpi::all_gather (C++  |     [\[1\]](api/languages/cpp_    |
|     fu                            | api.html#_CPPv4NK5cudaq13sample_r |
| nction)](api/languages/cpp_api.ht | esult12get_marginalERRKNSt6vector |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | INSt6size_tEEEKNSt11string_viewE) |
| RNSt6vectorIdEERKNSt6vectorIdEE), | -   [cuda                         |
|                                   | q::sample_result::get_total_shots |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi10all_gather |     function)](api/langua         |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| -   [cudaq::mpi::all_reduce (C++  | sample_result15get_total_shotsEv) |
|                                   | -   [cuda                         |
|  function)](api/languages/cpp_api | q::sample_result::has_even_parity |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     (C++                          |
| reduceE1TRK1TRK14BinaryFunction), |     fun                           |
|     [\[1\]](api/langu             | ction)](api/languages/cpp_api.htm |
| ages/cpp_api.html#_CPPv4I00EN5cud | l#_CPPv4N5cudaq13sample_result15h |
| aq3mpi10all_reduceE1TRK1TRK4Func) | as_even_parityENSt11string_viewE) |
| -   [cudaq::mpi::broadcast (C++   | -   [cuda                         |
|     function)](api/               | q::sample_result::has_expectation |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq3mpi9broadcastERNSt6stringEi), |     funct                         |
|     [\[1\]](api/la                | ion)](api/languages/cpp_api.html# |
| nguages/cpp_api.html#_CPPv4N5cuda | _CPPv4NK5cudaq13sample_result15ha |
| q3mpi9broadcastERNSt6vectorIdEEi) | s_expectationEKNSt11string_viewE) |
| -   [cudaq::mpi::finalize (C++    | -   [cu                           |
|     f                             | daq::sample_result::most_probable |
| unction)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     fun                           |
| -   [cudaq::mpi::initialize (C++  | ction)](api/languages/cpp_api.htm |
|     function                      | l#_CPPv4NK5cudaq13sample_result13 |
| )](api/languages/cpp_api.html#_CP | most_probableEKNSt11string_viewE) |
| Pv4N5cudaq3mpi10initializeEiPPc), | -                                 |
|     [                             | [cudaq::sample_result::operator+= |
| \[1\]](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq3mpi10initializeEv) |     function)](api/langua         |
| -   [cudaq::mpi::is_initialized   | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     (C++                          | ample_resultpLERK13sample_result) |
|     function                      | -                                 |
| )](api/languages/cpp_api.html#_CP |  [cudaq::sample_result::operator= |
| Pv4N5cudaq3mpi14is_initializedEv) |     (C++                          |
| -   [cudaq::mpi::num_ranks (C++   |     function)](api/langua         |
|     fu                            | ges/cpp_api.html#_CPPv4N5cudaq13s |
| nction)](api/languages/cpp_api.ht | ample_resultaSERR13sample_result) |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | -                                 |
| -   [cudaq::mpi::rank (C++        | [cudaq::sample_result::operator== |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)](api/languag        |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | es/cpp_api.html#_CPPv4NK5cudaq13s |
| -   [cudaq::noise_model (C++      | ample_resulteqERK13sample_result) |
|                                   | -   [                             |
|    class)](api/languages/cpp_api. | cudaq::sample_result::probability |
| html#_CPPv4N5cudaq11noise_modelE) |     (C++                          |
| -   [cudaq::n                     |     function)](api/lan            |
| oise_model::add_all_qubit_channel | guages/cpp_api.html#_CPPv4NK5cuda |
|     (C++                          | q13sample_result11probabilityENSt |
|     function)](api                | 11string_viewEKNSt11string_viewE) |
| /languages/cpp_api.html#_CPPv4IDp | -   [cud                          |
| EN5cudaq11noise_model21add_all_qu | aq::sample_result::register_names |
| bit_channelEvRK13kraus_channeli), |     (C++                          |
|     [\[1\]](api/langua            |     function)](api/langu          |
| ges/cpp_api.html#_CPPv4N5cudaq11n | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| oise_model21add_all_qubit_channel | 3sample_result14register_namesEv) |
| ERKNSt6stringERK13kraus_channeli) | -                                 |
| -                                 |    [cudaq::sample_result::reorder |
|  [cudaq::noise_model::add_channel |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     funct                         | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ion)](api/languages/cpp_api.html# | ample_result7reorderERKNSt6vector |
| _CPPv4IDpEN5cudaq11noise_model11a | INSt6size_tEEEKNSt11string_viewE) |
| dd_channelEvRK15PredicateFuncTy), | -   [cu                           |
|     [\[1\]](api/languages/cpp_    | daq::sample_result::sample_result |
| api.html#_CPPv4IDpEN5cudaq11noise |     (C++                          |
| _model11add_channelEvRKNSt6vector |     func                          |
| INSt6size_tEEERK13kraus_channel), | tion)](api/languages/cpp_api.html |
|     [\[2\]](ap                    | #_CPPv4N5cudaq13sample_result13sa |
| i/languages/cpp_api.html#_CPPv4N5 | mple_resultERK15ExecutionResult), |
| cudaq11noise_model11add_channelER |     [\[1\]](api/la                |
| KNSt6stringERK15PredicateFuncTy), | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q13sample_result13sample_resultER |
| [\[3\]](api/languages/cpp_api.htm | KNSt6vectorI15ExecutionResultEE), |
| l#_CPPv4N5cudaq11noise_model11add |                                   |
| _channelERKNSt6stringERKNSt6vecto |  [\[2\]](api/languages/cpp_api.ht |
| rINSt6size_tEEERK13kraus_channel) | ml#_CPPv4N5cudaq13sample_result13 |
| -   [cudaq::noise_model::empty    | sample_resultERR13sample_result), |
|     (C++                          |     [                             |
|     function                      | \[3\]](api/languages/cpp_api.html |
| )](api/languages/cpp_api.html#_CP | #_CPPv4N5cudaq13sample_result13sa |
| Pv4NK5cudaq11noise_model5emptyEv) | mple_resultERR15ExecutionResult), |
| -                                 |     [\[4\]](api/lan               |
| [cudaq::noise_model::get_channels | guages/cpp_api.html#_CPPv4N5cudaq |
|     (C++                          | 13sample_result13sample_resultEdR |
|     function)](api/l              | KNSt6vectorI15ExecutionResultEE), |
| anguages/cpp_api.html#_CPPv4I0ENK |     [\[5\]](api/lan               |
| 5cudaq11noise_model12get_channels | guages/cpp_api.html#_CPPv4N5cudaq |
| ENSt6vectorI13kraus_channelEERKNS | 13sample_result13sample_resultEv) |
| t6vectorINSt6size_tEEERKNSt6vecto | -                                 |
| rINSt6size_tEEERKNSt6vectorIdEE), |  [cudaq::sample_result::serialize |
|     [\[1\]](api/languages/cpp_a   |     (C++                          |
| pi.html#_CPPv4NK5cudaq11noise_mod |     function)](api                |
| el12get_channelsERKNSt6stringERKN | /languages/cpp_api.html#_CPPv4NK5 |
| St6vectorINSt6size_tEEERKNSt6vect | cudaq13sample_result9serializeEv) |
| orINSt6size_tEEERKNSt6vectorIdEE) | -   [cudaq::sample_result::size   |
| -                                 |     (C++                          |
|  [cudaq::noise_model::noise_model |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq13sampl |
|     function)](api                | e_result4sizeEKNSt11string_viewE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::sample_result::to_map |
| udaq11noise_model11noise_modelEv) |     (C++                          |
| -   [cu                           |     function)](api/languages/cpp  |
| daq::noise_model::PredicateFuncTy | _api.html#_CPPv4NK5cudaq13sample_ |
|     (C++                          | result6to_mapEKNSt11string_viewE) |
|     type)](api/la                 | -   [cuda                         |
| nguages/cpp_api.html#_CPPv4N5cuda | q::sample_result::\~sample_result |
| q11noise_model15PredicateFuncTyE) |     (C++                          |
| -   [cud                          |     funct                         |
| aq::noise_model::register_channel | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq13sample_resultD0Ev) |
|     function)](api/languages      | -   [cudaq::scalar_callback (C++  |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     c                             |
| noise_model16register_channelEvv) | lass)](api/languages/cpp_api.html |
| -   [cudaq::                      | #_CPPv4N5cudaq15scalar_callbackE) |
| noise_model::requires_constructor | -   [c                            |
|     (C++                          | udaq::scalar_callback::operator() |
|     type)](api/languages/cp       |     (C++                          |
| p_api.html#_CPPv4I0DpEN5cudaq11no |     function)](api/language       |
| ise_model20requires_constructorE) | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| -   [cudaq::noise_model_type (C++ | alar_callbackclERKNSt13unordered_ |
|     e                             | mapINSt6stringENSt7complexIdEEEE) |
| num)](api/languages/cpp_api.html# | -   [                             |
| _CPPv4N5cudaq16noise_model_typeE) | cudaq::scalar_callback::operator= |
| -   [cudaq::no                    |     (C++                          |
| ise_model_type::amplitude_damping |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](api/languages    | _callbackaSERK15scalar_callback), |
| /cpp_api.html#_CPPv4N5cudaq16nois |     [\[1\]](api/languages/        |
| e_model_type17amplitude_dampingE) | cpp_api.html#_CPPv4N5cudaq15scala |
| -   [cudaq::noise_mode            | r_callbackaSERR15scalar_callback) |
| l_type::amplitude_damping_channel | -   [cudaq:                       |
|     (C++                          | :scalar_callback::scalar_callback |
|     e                             |     (C++                          |
| numerator)](api/languages/cpp_api |     function)](api/languag        |
| .html#_CPPv4N5cudaq16noise_model_ | es/cpp_api.html#_CPPv4I0_NSt11ena |
| type25amplitude_damping_channelE) | ble_if_tINSt16is_invocable_r_vINS |
| -   [cudaq::n                     | t7complexIdEE8CallableRKNSt13unor |
| oise_model_type::bit_flip_channel | dered_mapINSt6stringENSt7complexI |
|     (C++                          | dEEEEEEbEEEN5cudaq15scalar_callba |
|     enumerator)](api/language     | ck15scalar_callbackERR8Callable), |
| s/cpp_api.html#_CPPv4N5cudaq16noi |     [\[1\                         |
| se_model_type16bit_flip_channelE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::                      | Pv4N5cudaq15scalar_callback15scal |
| noise_model_type::depolarization1 | ar_callbackERK15scalar_callback), |
|     (C++                          |     [\[2                          |
|     enumerator)](api/languag      | \]](api/languages/cpp_api.html#_C |
| es/cpp_api.html#_CPPv4N5cudaq16no | PPv4N5cudaq15scalar_callback15sca |
| ise_model_type15depolarization1E) | lar_callbackERR15scalar_callback) |
| -   [cudaq::                      | -   [cudaq::scalar_operator (C++  |
| noise_model_type::depolarization2 |     c                             |
|     (C++                          | lass)](api/languages/cpp_api.html |
|     enumerator)](api/languag      | #_CPPv4N5cudaq15scalar_operatorE) |
| es/cpp_api.html#_CPPv4N5cudaq16no | -                                 |
| ise_model_type15depolarization2E) | [cudaq::scalar_operator::evaluate |
| -   [cudaq::noise_m               |     (C++                          |
| odel_type::depolarization_channel |                                   |
|     (C++                          |    function)](api/languages/cpp_a |
|                                   | pi.html#_CPPv4NK5cudaq15scalar_op |
|   enumerator)](api/languages/cpp_ | erator8evaluateERKNSt13unordered_ |
| api.html#_CPPv4N5cudaq16noise_mod | mapINSt6stringENSt7complexIdEEEE) |
| el_type22depolarization_channelE) | -   [cudaq::scalar_ope            |
| -                                 | rator::get_parameter_descriptions |
|  [cudaq::noise_model_type::pauli1 |     (C++                          |
|     (C++                          |     f                             |
|     enumerator)](a                | unction)](api/languages/cpp_api.h |
| pi/languages/cpp_api.html#_CPPv4N | tml#_CPPv4NK5cudaq15scalar_operat |
| 5cudaq16noise_model_type6pauli1E) | or26get_parameter_descriptionsEv) |
| -                                 | -   [cu                           |
|  [cudaq::noise_model_type::pauli2 | daq::scalar_operator::is_constant |
|     (C++                          |     (C++                          |
|     enumerator)](a                |     function)](api/lang           |
| pi/languages/cpp_api.html#_CPPv4N | uages/cpp_api.html#_CPPv4NK5cudaq |
| 5cudaq16noise_model_type6pauli2E) | 15scalar_operator11is_constantEv) |
| -   [cudaq                        | -   [c                            |
| ::noise_model_type::phase_damping | udaq::scalar_operator::operator\* |
|     (C++                          |     (C++                          |
|     enumerator)](api/langu        |     function                      |
| ages/cpp_api.html#_CPPv4N5cudaq16 | )](api/languages/cpp_api.html#_CP |
| noise_model_type13phase_dampingE) | Pv4N5cudaq15scalar_operatormlENSt |
| -   [cudaq::noi                   | 7complexIdEERK15scalar_operator), |
| se_model_type::phase_flip_channel |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](api/languages/   | Pv4N5cudaq15scalar_operatormlENSt |
| cpp_api.html#_CPPv4N5cudaq16noise | 7complexIdEERR15scalar_operator), |
| _model_type18phase_flip_channelE) |     [\[2\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4N5cudaq15scalar_ |
| [cudaq::noise_model_type::unknown | operatormlEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     enumerator)](ap               | p_api.html#_CPPv4N5cudaq15scalar_ |
| i/languages/cpp_api.html#_CPPv4N5 | operatormlEdRR15scalar_operator), |
| cudaq16noise_model_type7unknownE) |     [\[4\]](api/languages         |
| -                                 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| [cudaq::noise_model_type::x_error | alar_operatormlENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     enumerator)](ap               | _api.html#_CPPv4NKR5cudaq15scalar |
| i/languages/cpp_api.html#_CPPv4N5 | _operatormlERK15scalar_operator), |
| cudaq16noise_model_type7x_errorE) |     [\[6\]]                       |
| -                                 | (api/languages/cpp_api.html#_CPPv |
| [cudaq::noise_model_type::y_error | 4NKR5cudaq15scalar_operatormlEd), |
|     (C++                          |     [\[7\]](api/language          |
|     enumerator)](ap               | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| i/languages/cpp_api.html#_CPPv4N5 | alar_operatormlENSt7complexIdEE), |
| cudaq16noise_model_type7y_errorE) |     [\[8\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4NO5cudaq15scalar |
| [cudaq::noise_model_type::z_error | _operatormlERK15scalar_operator), |
|     (C++                          |     [\[9\                         |
|     enumerator)](ap               | ]](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4NO5cudaq15scalar_operatormlEd) |
| cudaq16noise_model_type7z_errorE) | -   [cu                           |
| -   [cudaq::num_available_gpus    | daq::scalar_operator::operator\*= |
|     (C++                          |     (C++                          |
|     function                      |     function)](api/languag        |
| )](api/languages/cpp_api.html#_CP | es/cpp_api.html#_CPPv4N5cudaq15sc |
| Pv4N5cudaq18num_available_gpusEv) | alar_operatormLENSt7complexIdEE), |
| -   [cudaq::observe (C++          |     [\[1\]](api/languages/c       |
|     function)]                    | pp_api.html#_CPPv4N5cudaq15scalar |
| (api/languages/cpp_api.html#_CPPv | _operatormLERK15scalar_operator), |
| 4I00DpEN5cudaq7observeENSt6vector |     [\[2                          |
| I14observe_resultEERR13QuantumKer | \]](api/languages/cpp_api.html#_C |
| nelRK15SpinOpContainerDpRR4Args), | PPv4N5cudaq15scalar_operatormLEd) |
|     [\[1\]](api/languages/cpp_ap  | -   [                             |
| i.html#_CPPv4I0DpEN5cudaq7observe | cudaq::scalar_operator::operator+ |
| E14observe_resultNSt6size_tERR13Q |     (C++                          |
| uantumKernelRK7spin_opDpRR4Args), |     function                      |
|     [\[                           | )](api/languages/cpp_api.html#_CP |
| 2\]](api/languages/cpp_api.html#_ | Pv4N5cudaq15scalar_operatorplENSt |
| CPPv4I0DpEN5cudaq7observeE14obser | 7complexIdEERK15scalar_operator), |
| ve_resultRK15observe_optionsRR13Q |     [\[1\                         |
| uantumKernelRK7spin_opDpRR4Args), | ]](api/languages/cpp_api.html#_CP |
|     [\[3\]](api/lang              | Pv4N5cudaq15scalar_operatorplENSt |
| uages/cpp_api.html#_CPPv4I0DpEN5c | 7complexIdEERR15scalar_operator), |
| udaq7observeE14observe_resultRR13 |     [\[2\]](api/languages/cp      |
| QuantumKernelRK7spin_opDpRR4Args) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::observe_options (C++  | operatorplEdRK15scalar_operator), |
|     st                            |     [\[3\]](api/languages/cp      |
| ruct)](api/languages/cpp_api.html | p_api.html#_CPPv4N5cudaq15scalar_ |
| #_CPPv4N5cudaq15observe_optionsE) | operatorplEdRR15scalar_operator), |
| -   [cudaq::observe_result (C++   |     [\[4\]](api/languages         |
|                                   | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| class)](api/languages/cpp_api.htm | alar_operatorplENSt7complexIdEE), |
| l#_CPPv4N5cudaq14observe_resultE) |     [\[5\]](api/languages/cpp     |
| -                                 | _api.html#_CPPv4NKR5cudaq15scalar |
|    [cudaq::observe_result::counts | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     function)](api/languages/c    | (api/languages/cpp_api.html#_CPPv |
| pp_api.html#_CPPv4N5cudaq14observ | 4NKR5cudaq15scalar_operatorplEd), |
| e_result6countsERK12spin_op_term) |     [\[7\]]                       |
| -   [cudaq::observe_result::dump  | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatorplEv), |
|     function)                     |     [\[8\]](api/language          |
| ](api/languages/cpp_api.html#_CPP | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| v4N5cudaq14observe_result4dumpEv) | alar_operatorplENSt7complexIdEE), |
| -   [c                            |     [\[9\]](api/languages/cp      |
| udaq::observe_result::expectation | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatorplERK15scalar_operator), |
|                                   |     [\[10\]                       |
| function)](api/languages/cpp_api. | ](api/languages/cpp_api.html#_CPP |
| html#_CPPv4N5cudaq14observe_resul | v4NO5cudaq15scalar_operatorplEd), |
| t11expectationERK12spin_op_term), |     [\[11\                        |
|     [\[1\]](api/la                | ]](api/languages/cpp_api.html#_CP |
| nguages/cpp_api.html#_CPPv4N5cuda | Pv4NO5cudaq15scalar_operatorplEv) |
| q14observe_result11expectationEv) | -   [c                            |
| -   [cuda                         | udaq::scalar_operator::operator+= |
| q::observe_result::id_coefficient |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/langu          | es/cpp_api.html#_CPPv4N5cudaq15sc |
| ages/cpp_api.html#_CPPv4N5cudaq14 | alar_operatorpLENSt7complexIdEE), |
| observe_result14id_coefficientEv) |     [\[1\]](api/languages/c       |
| -   [cuda                         | pp_api.html#_CPPv4N5cudaq15scalar |
| q::observe_result::observe_result | _operatorpLERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|                                   | \]](api/languages/cpp_api.html#_C |
|   function)](api/languages/cpp_ap | PPv4N5cudaq15scalar_operatorpLEd) |
| i.html#_CPPv4N5cudaq14observe_res | -   [                             |
| ult14observe_resultEdRK7spin_op), | cudaq::scalar_operator::operator- |
|     [\[1\]](a                     |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function                      |
| 5cudaq14observe_result14observe_r | )](api/languages/cpp_api.html#_CP |
| esultEdRK7spin_op13sample_result) | Pv4N5cudaq15scalar_operatormiENSt |
| -                                 | 7complexIdEERK15scalar_operator), |
|  [cudaq::observe_result::operator |     [\[1\                         |
|     double (C++                   | ]](api/languages/cpp_api.html#_CP |
|     functio                       | Pv4N5cudaq15scalar_operatormiENSt |
| n)](api/languages/cpp_api.html#_C | 7complexIdEERR15scalar_operator), |
| PPv4N5cudaq14observe_resultcvdEv) |     [\[2\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4N5cudaq15scalar_ |
|  [cudaq::observe_result::raw_data | operatormiEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     function)](ap                 | p_api.html#_CPPv4N5cudaq15scalar_ |
| i/languages/cpp_api.html#_CPPv4N5 | operatormiEdRR15scalar_operator), |
| cudaq14observe_result8raw_dataEv) |     [\[4\]](api/languages         |
| -   [cudaq::operator_handler (C++ | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     cl                            | alar_operatormiENSt7complexIdEE), |
| ass)](api/languages/cpp_api.html# |     [\[5\]](api/languages/cpp     |
| _CPPv4N5cudaq16operator_handlerE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::optimizable_function  | _operatormiERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     class)                        | (api/languages/cpp_api.html#_CPPv |
| ](api/languages/cpp_api.html#_CPP | 4NKR5cudaq15scalar_operatormiEd), |
| v4N5cudaq20optimizable_functionE) |     [\[7\]]                       |
| -   [cudaq::optimization_result   | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEv), |
|     type                          |     [\[8\]](api/language          |
| )](api/languages/cpp_api.html#_CP | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| Pv4N5cudaq19optimization_resultE) | alar_operatormiENSt7complexIdEE), |
| -   [cudaq::optimizer (C++        |     [\[9\]](api/languages/cp      |
|     class)](api/languages/cpp_a   | p_api.html#_CPPv4NO5cudaq15scalar |
| pi.html#_CPPv4N5cudaq9optimizerE) | _operatormiERK15scalar_operator), |
| -   [cudaq::optimizer::optimize   |     [\[10\]                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|                                   | v4NO5cudaq15scalar_operatormiEd), |
|  function)](api/languages/cpp_api |     [\[11\                        |
| .html#_CPPv4N5cudaq9optimizer8opt | ]](api/languages/cpp_api.html#_CP |
| imizeEKiRR20optimizable_function) | Pv4NO5cudaq15scalar_operatormiEv) |
| -   [cu                           | -   [c                            |
| daq::optimizer::requiresGradients | udaq::scalar_operator::operator-= |
|     (C++                          |     (C++                          |
|     function)](api/la             |     function)](api/languag        |
| nguages/cpp_api.html#_CPPv4N5cuda | es/cpp_api.html#_CPPv4N5cudaq15sc |
| q9optimizer17requiresGradientsEv) | alar_operatormIENSt7complexIdEE), |
| -   [cudaq::orca (C++             |     [\[1\]](api/languages/c       |
|     type)](api/languages/         | pp_api.html#_CPPv4N5cudaq15scalar |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | _operatormIERK15scalar_operator), |
| -   [cudaq::orca::sample (C++     |     [\[2                          |
|     function)](api/languages/c    | \]](api/languages/cpp_api.html#_C |
| pp_api.html#_CPPv4N5cudaq4orca6sa | PPv4N5cudaq15scalar_operatormIEd) |
| mpleERNSt6vectorINSt6size_tEEERNS | -   [                             |
| t6vectorINSt6size_tEEERNSt6vector | cudaq::scalar_operator::operator/ |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     (C++                          |
|     [\[1\]]                       |     function                      |
| (api/languages/cpp_api.html#_CPPv | )](api/languages/cpp_api.html#_CP |
| 4N5cudaq4orca6sampleERNSt6vectorI | Pv4N5cudaq15scalar_operatordvENSt |
| NSt6size_tEEERNSt6vectorINSt6size | 7complexIdEERK15scalar_operator), |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     [\[1\                         |
| -   [cudaq::orca::sample_async    | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatordvENSt |
|                                   | 7complexIdEERR15scalar_operator), |
| function)](api/languages/cpp_api. |     [\[2\]](api/languages/cp      |
| html#_CPPv4N5cudaq4orca12sample_a | p_api.html#_CPPv4N5cudaq15scalar_ |
| syncERNSt6vectorINSt6size_tEEERNS | operatordvEdRK15scalar_operator), |
| t6vectorINSt6size_tEEERNSt6vector |     [\[3\]](api/languages/cp      |
| IdEERNSt6vectorIdEEiNSt6size_tE), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[1\]](api/la                | operatordvEdRR15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[4\]](api/languages         |
| q4orca12sample_asyncERNSt6vectorI | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| NSt6size_tEEERNSt6vectorINSt6size | alar_operatordvENSt7complexIdEE), |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::OrcaRemoteRESTQPU     | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatordvERK15scalar_operator), |
|     cla                           |     [\[6\]]                       |
| ss)](api/languages/cpp_api.html#_ | (api/languages/cpp_api.html#_CPPv |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | 4NKR5cudaq15scalar_operatordvEd), |
| -   [cudaq::pauli1 (C++           |     [\[7\]](api/language          |
|     class)](api/languages/cp      | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| p_api.html#_CPPv4N5cudaq6pauli1E) | alar_operatordvENSt7complexIdEE), |
| -                                 |     [\[8\]](api/languages/cp      |
|    [cudaq::pauli1::num_parameters | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatordvERK15scalar_operator), |
|     member)]                      |     [\[9\                         |
| (api/languages/cpp_api.html#_CPPv | ]](api/languages/cpp_api.html#_CP |
| 4N5cudaq6pauli114num_parametersE) | Pv4NO5cudaq15scalar_operatordvEd) |
| -   [cudaq::pauli1::num_targets   | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator/= |
|     membe                         |     (C++                          |
| r)](api/languages/cpp_api.html#_C |     function)](api/languag        |
| PPv4N5cudaq6pauli111num_targetsE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::pauli1::pauli1 (C++   | alar_operatordVENSt7complexIdEE), |
|     function)](api/languages/cpp_ |     [\[1\]](api/languages/c       |
| api.html#_CPPv4N5cudaq6pauli16pau | pp_api.html#_CPPv4N5cudaq15scalar |
| li1ERKNSt6vectorIN5cudaq4realEEE) | _operatordVERK15scalar_operator), |
| -   [cudaq::pauli2 (C++           |     [\[2                          |
|     class)](api/languages/cp      | \]](api/languages/cpp_api.html#_C |
| p_api.html#_CPPv4N5cudaq6pauli2E) | PPv4N5cudaq15scalar_operatordVEd) |
| -                                 | -   [                             |
|    [cudaq::pauli2::num_parameters | cudaq::scalar_operator::operator= |
|     (C++                          |     (C++                          |
|     member)]                      |     function)](api/languages/c    |
| (api/languages/cpp_api.html#_CPPv | pp_api.html#_CPPv4N5cudaq15scalar |
| 4N5cudaq6pauli214num_parametersE) | _operatoraSERK15scalar_operator), |
| -   [cudaq::pauli2::num_targets   |     [\[1\]](api/languages/        |
|     (C++                          | cpp_api.html#_CPPv4N5cudaq15scala |
|     membe                         | r_operatoraSERR15scalar_operator) |
| r)](api/languages/cpp_api.html#_C | -   [c                            |
| PPv4N5cudaq6pauli211num_targetsE) | udaq::scalar_operator::operator== |
| -   [cudaq::pauli2::pauli2 (C++   |     (C++                          |
|     function)](api/languages/cpp_ |     function)](api/languages/c    |
| api.html#_CPPv4N5cudaq6pauli26pau | pp_api.html#_CPPv4NK5cudaq15scala |
| li2ERKNSt6vectorIN5cudaq4realEEE) | r_operatoreqERK15scalar_operator) |
| -   [cudaq::phase_damping (C++    | -   [cudaq:                       |
|                                   | :scalar_operator::scalar_operator |
|  class)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13phase_dampingE) |     func                          |
| -   [cud                          | tion)](api/languages/cpp_api.html |
| aq::phase_damping::num_parameters | #_CPPv4N5cudaq15scalar_operator15 |
|     (C++                          | scalar_operatorENSt7complexIdEE), |
|     member)](api/lan              |     [\[1\]](api/langu             |
| guages/cpp_api.html#_CPPv4N5cudaq | ages/cpp_api.html#_CPPv4N5cudaq15 |
| 13phase_damping14num_parametersE) | scalar_operator15scalar_operatorE |
| -   [                             | RK15scalar_callbackRRNSt13unorder |
| cudaq::phase_damping::num_targets | ed_mapINSt6stringENSt6stringEEE), |
|     (C++                          |     [\[2\                         |
|     member)](api/                 | ]](api/languages/cpp_api.html#_CP |
| languages/cpp_api.html#_CPPv4N5cu | Pv4N5cudaq15scalar_operator15scal |
| daq13phase_damping11num_targetsE) | ar_operatorERK15scalar_operator), |
| -   [cudaq::phase_flip_channel    |     [\[3\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     clas                          | scalar_operator15scalar_operatorE |
| s)](api/languages/cpp_api.html#_C | RR15scalar_callbackRRNSt13unorder |
| PPv4N5cudaq18phase_flip_channelE) | ed_mapINSt6stringENSt6stringEEE), |
| -   [cudaq::p                     |     [\[4\                         |
| hase_flip_channel::num_parameters | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operator15scal |
|     member)](api/language         | ar_operatorERR15scalar_operator), |
| s/cpp_api.html#_CPPv4N5cudaq18pha |     [\[5\]](api/language          |
| se_flip_channel14num_parametersE) | s/cpp_api.html#_CPPv4N5cudaq15sca |
| -   [cudaq                        | lar_operator15scalar_operatorEd), |
| ::phase_flip_channel::num_targets |     [\[6\]](api/languag           |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     member)](api/langu            | alar_operator15scalar_operatorEv) |
| ages/cpp_api.html#_CPPv4N5cudaq18 | -   [                             |
| phase_flip_channel11num_targetsE) | cudaq::scalar_operator::to_matrix |
| -   [cudaq::product_op (C++       |     (C++                          |
|                                   |                                   |
|  class)](api/languages/cpp_api.ht |   function)](api/languages/cpp_ap |
| ml#_CPPv4I0EN5cudaq10product_opE) | i.html#_CPPv4NK5cudaq15scalar_ope |
| -   [cudaq::product_op::begin     | rator9to_matrixERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|     functio                       | -   [                             |
| n)](api/languages/cpp_api.html#_C | cudaq::scalar_operator::to_string |
| PPv4NK5cudaq10product_op5beginEv) |     (C++                          |
| -                                 |     function)](api/l              |
|  [cudaq::product_op::canonicalize | anguages/cpp_api.html#_CPPv4NK5cu |
|     (C++                          | daq15scalar_operator9to_stringEv) |
|     func                          | -   [cudaq::s                     |
| tion)](api/languages/cpp_api.html | calar_operator::\~scalar_operator |
| #_CPPv4N5cudaq10product_op12canon |     (C++                          |
| icalizeERKNSt3setINSt6size_tEEE), |     functio                       |
|     [\[1\]](api                   | n)](api/languages/cpp_api.html#_C |
| /languages/cpp_api.html#_CPPv4N5c | PPv4N5cudaq15scalar_operatorD0Ev) |
| udaq10product_op12canonicalizeEv) | -   [cudaq::set_noise (C++        |
| -   [                             |     function)](api/langu          |
| cudaq::product_op::const_iterator | ages/cpp_api.html#_CPPv4N5cudaq9s |
|     (C++                          | et_noiseERKN5cudaq11noise_modelE) |
|     struct)](api/                 | -   [cudaq::set_random_seed (C++  |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/               |
| daq10product_op14const_iteratorE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::product_o             | daq15set_random_seedENSt6size_tE) |
| p::const_iterator::const_iterator | -   [cudaq::simulation_precision  |
|     (C++                          |     (C++                          |
|     fu                            |     enum)                         |
| nction)](api/languages/cpp_api.ht | ](api/languages/cpp_api.html#_CPP |
| ml#_CPPv4N5cudaq10product_op14con | v4N5cudaq20simulation_precisionE) |
| st_iterator14const_iteratorEPK10p | -   [                             |
| roduct_opI9HandlerTyENSt6size_tE) | cudaq::simulation_precision::fp32 |
| -   [cudaq::produ                 |     (C++                          |
| ct_op::const_iterator::operator!= |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     fun                           | udaq20simulation_precision4fp32E) |
| ction)](api/languages/cpp_api.htm | -   [                             |
| l#_CPPv4NK5cudaq10product_op14con | cudaq::simulation_precision::fp64 |
| st_iteratorneERK14const_iterator) |     (C++                          |
| -   [cudaq::produ                 |     enumerator)](api              |
| ct_op::const_iterator::operator\* | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq20simulation_precision4fp64E) |
|     function)](api/lang           | -   [cudaq::SimulationState (C++  |
| uages/cpp_api.html#_CPPv4NK5cudaq |     c                             |
| 10product_op14const_iteratormlEv) | lass)](api/languages/cpp_api.html |
| -   [cudaq::produ                 | #_CPPv4N5cudaq15SimulationStateE) |
| ct_op::const_iterator::operator++ | -   [                             |
|     (C++                          | cudaq::SimulationState::precision |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     enum)](api                    |
| 0product_op14const_iteratorppEi), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[1\]](api/lan               | udaq15SimulationState9precisionE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq:                       |
| 10product_op14const_iteratorppEv) | :SimulationState::precision::fp32 |
| -   [cudaq::produc                |     (C++                          |
| t_op::const_iterator::operator\-- |     enumerator)](api/lang         |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     function)](api/lang           | 5SimulationState9precision4fp32E) |
| uages/cpp_api.html#_CPPv4N5cudaq1 | -   [cudaq:                       |
| 0product_op14const_iteratormmEi), | :SimulationState::precision::fp64 |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     enumerator)](api/lang         |
| 10product_op14const_iteratormmEv) | uages/cpp_api.html#_CPPv4N5cudaq1 |
|                                   | 5SimulationState9precision4fp64E) |
|                                   | -                                 |
|                                   |   [cudaq::SimulationState::Tensor |
|                                   |     (C++                          |
|                                   |     struct)](                     |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq15SimulationState6TensorE) |
|                                   | -   [cudaq::spin_handler (C++     |
|                                   |                                   |
|                                   |   class)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq12spin_handlerE) |
|                                   | -   [cudaq:                       |
|                                   | :spin_handler::to_diagonal_matrix |
|                                   |     (C++                          |
|                                   |     function)](api/la             |
|                                   | nguages/cpp_api.html#_CPPv4NK5cud |
|                                   | aq12spin_handler18to_diagonal_mat |
|                                   | rixERNSt13unordered_mapINSt6size_ |
|                                   | tENSt7int64_tEEERKNSt13unordered_ |
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
| -   [handle_call_arguments()      | -   [has_execution_data()         |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](api/langua           |    (cudaq.ptsbe.PTSBESampleResult |
| ges/python_api.html#cudaq.PyKerne |     method)](api/languages        |
| lDecorator.handle_call_arguments) | /python_api.html#cudaq.ptsbe.PTSB |
|                                   | ESampleResult.has_execution_data) |
|                                   | -   [has_target() (in module      |
|                                   |     cudaq)](api/languages/        |
|                                   | python_api.html#cudaq.has_target) |
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
|         property)](ap             | -   [num_qubits() (cudaq.State    |
| i/languages/python_api.html#cudaq |     method)](api/languages/python |
| .ptsbe.TraceInstructionType.name) | _api.html#cudaq.State.num_qubits) |
|     -   [(cudaq.PyKernel          | -   [num_ranks() (in module       |
|                                   |     cudaq.mpi)](api/languages/pyt |
|     attribute)](api/languages/pyt | hon_api.html#cudaq.mpi.num_ranks) |
| hon_api.html#cudaq.PyKernel.name) | -   [num_rows()                   |
|                                   |     (cudaq.ComplexMatrix          |
|   -   [(cudaq.SimulationPrecision |     me                            |
|         proper                    | thod)](api/languages/python_api.h |
| ty)](api/languages/python_api.htm | tml#cudaq.ComplexMatrix.num_rows) |
| l#cudaq.SimulationPrecision.name) | -   [number() (in module          |
|     -   [(cudaq.spin.Pauli        |                                   |
|                                   |    cudaq.boson)](api/languages/py |
|    property)](api/languages/pytho | thon_api.html#cudaq.boson.number) |
| n_api.html#cudaq.spin.Pauli.name) |     -   [(in module               |
|     -   [(cudaq.Target            |         c                         |
|                                   | udaq.fermion)](api/languages/pyth |
|        property)](api/languages/p | on_api.html#cudaq.fermion.number) |
| ython_api.html#cudaq.Target.name) |     -   [(in module               |
| -   [name()                       |         cudaq.operators.cus       |
|                                   | tom)](api/languages/python_api.ht |
|  (cudaq.ptsbe.PTSSamplingStrategy | ml#cudaq.operators.custom.number) |
|     method)](a                    | -   [nvqir::MPSSimulationState    |
| pi/languages/python_api.html#cuda |     (C++                          |
| q.ptsbe.PTSSamplingStrategy.name) |     class)]                       |
| -   [NelderMead (class in         | (api/languages/cpp_api.html#_CPPv |
|     cudaq.optim                   | 4I0EN5nvqir18MPSSimulationStateE) |
| izers)](api/languages/python_api. | -                                 |
| html#cudaq.optimizers.NelderMead) |  [nvqir::TensorNetSimulationState |
| -   [NoiseModel (class in         |     (C++                          |
|     cudaq)](api/languages/        |     class)](api/l                 |
| python_api.html#cudaq.NoiseModel) | anguages/cpp_api.html#_CPPv4I0EN5 |
| -   [num_available_gpus() (in     | nvqir24TensorNetSimulationStateE) |
|     module                        |                                   |
|                                   |                                   |
|    cudaq)](api/languages/python_a |                                   |
| pi.html#cudaq.num_available_gpus) |                                   |
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
|    property)](api/languages/pytho | -                                 |
| n_api.html#cudaq.operators.fermio |    [ProbabilisticSamplingStrategy |
| n.FermionOperatorTerm.parameters) |     (class in                     |
|     -                             |     cudaq.ptsbe)](api/la          |
|  [(cudaq.operators.MatrixOperator | nguages/python_api.html#cudaq.pts |
|         property)](api/la         | be.ProbabilisticSamplingStrategy) |
| nguages/python_api.html#cudaq.ope | -   [probability()                |
| rators.MatrixOperator.parameters) |     (cudaq.SampleResult           |
|     -   [(cuda                    |     meth                          |
| q.operators.MatrixOperatorElement | od)](api/languages/python_api.htm |
|         property)](api/languages  | l#cudaq.SampleResult.probability) |
| /python_api.html#cudaq.operators. | -   [ProductOperator (in module   |
| MatrixOperatorElement.parameters) |     cudaq.operator                |
|     -   [(c                       | s)](api/languages/python_api.html |
| udaq.operators.MatrixOperatorTerm | #cudaq.operators.ProductOperator) |
|         property)](api/langua     | -   [ptsbe_execution_data         |
| ges/python_api.html#cudaq.operato |                                   |
| rs.MatrixOperatorTerm.parameters) |    (cudaq.ptsbe.PTSBESampleResult |
|     -                             |     property)](api/languages/p    |
|  [(cudaq.operators.ScalarOperator | ython_api.html#cudaq.ptsbe.PTSBES |
|         property)](api/la         | ampleResult.ptsbe_execution_data) |
| nguages/python_api.html#cudaq.ope | -   [PTSBEExecutionData (class in |
| rators.ScalarOperator.parameters) |     cudaq.pts                     |
|     -   [(                        | be)](api/languages/python_api.htm |
| cudaq.operators.spin.SpinOperator | l#cudaq.ptsbe.PTSBEExecutionData) |
|         property)](api/langu      | -   [PTSBESampleResult (class in  |
| ages/python_api.html#cudaq.operat |     cudaq.pt                      |
| ors.spin.SpinOperator.parameters) | sbe)](api/languages/python_api.ht |
|     -   [(cuda                    | ml#cudaq.ptsbe.PTSBESampleResult) |
| q.operators.spin.SpinOperatorTerm | -   [PTSSamplingStrategy (class   |
|         property)](api/languages  |     in                            |
| /python_api.html#cudaq.operators. |     cudaq.ptsb                    |
| spin.SpinOperatorTerm.parameters) | e)](api/languages/python_api.html |
| -   [ParameterShift (class in     | #cudaq.ptsbe.PTSSamplingStrategy) |
|     cudaq.gradien                 | -   [PyKernel (class in           |
| ts)](api/languages/python_api.htm |     cudaq)](api/language          |
| l#cudaq.gradients.ParameterShift) | s/python_api.html#cudaq.PyKernel) |
| -   [parity() (in module          | -   [PyKernelDecorator (class in  |
|     cudaq.operators.cus           |     cudaq)](api/languages/python_ |
| tom)](api/languages/python_api.ht | api.html#cudaq.PyKernelDecorator) |
| ml#cudaq.operators.custom.parity) |                                   |
| -   [Pauli (class in              |                                   |
|     cudaq.spin)](api/languages/   |                                   |
| python_api.html#cudaq.spin.Pauli) |                                   |
| -   [Pauli1 (class in             |                                   |
|     cudaq)](api/langua            |                                   |
| ges/python_api.html#cudaq.Pauli1) |                                   |
| -   [Pauli2 (class in             |                                   |
|     cudaq)](api/langua            |                                   |
| ges/python_api.html#cudaq.Pauli2) |                                   |
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
| ml#cudaq.optimizers.Adam.to_json) | -   [type                         |
|     -   [(cudaq.optimizers.COBYLA |     (c                            |
|         metho                     | udaq.ptsbe.ShotAllocationStrategy |
| d)](api/languages/python_api.html |     property)](api/               |
| #cudaq.optimizers.COBYLA.to_json) | languages/python_api.html#cudaq.p |
|     -   [                         | tsbe.ShotAllocationStrategy.type) |
| (cudaq.optimizers.GradientDescent | -   [type_to_str()                |
|         method)](api/l            |     (cudaq.PyKernelDecorator      |
| anguages/python_api.html#cudaq.op |     static                        |
| timizers.GradientDescent.to_json) |     method)](                     |
|     -   [(cudaq.optimizers.LBFGS  | api/languages/python_api.html#cud |
|         meth                      | aq.PyKernelDecorator.type_to_str) |
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
