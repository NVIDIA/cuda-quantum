::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-3995
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
| -   [canonicalize()               | -   [cudaq::pauli1 (C++           |
|     (cu                           |     class)](api/languages/cp      |
| daq.operators.boson.BosonOperator | p_api.html#_CPPv4N5cudaq6pauli1E) |
|     method)](api/languages        | -                                 |
| /python_api.html#cudaq.operators. |    [cudaq::pauli1::num_parameters |
| boson.BosonOperator.canonicalize) |     (C++                          |
|     -   [(cudaq.                  |     member)]                      |
| operators.boson.BosonOperatorTerm | (api/languages/cpp_api.html#_CPPv |
|                                   | 4N5cudaq6pauli114num_parametersE) |
|        method)](api/languages/pyt | -   [cudaq::pauli1::num_targets   |
| hon_api.html#cudaq.operators.boso |     (C++                          |
| n.BosonOperatorTerm.canonicalize) |     membe                         |
|     -   [(cudaq.                  | r)](api/languages/cpp_api.html#_C |
| operators.fermion.FermionOperator | PPv4N5cudaq6pauli111num_targetsE) |
|                                   | -   [cudaq::pauli1::pauli1 (C++   |
|        method)](api/languages/pyt |     function)](api/languages/cpp_ |
| hon_api.html#cudaq.operators.ferm | api.html#_CPPv4N5cudaq6pauli16pau |
| ion.FermionOperator.canonicalize) | li1ERKNSt6vectorIN5cudaq4realEEE) |
|     -   [(cudaq.oper              | -   [cudaq::pauli2 (C++           |
| ators.fermion.FermionOperatorTerm |     class)](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq6pauli2E) |
|    method)](api/languages/python_ | -                                 |
| api.html#cudaq.operators.fermion. |    [cudaq::pauli2::num_parameters |
| FermionOperatorTerm.canonicalize) |     (C++                          |
|     -                             |     member)]                      |
|  [(cudaq.operators.MatrixOperator | (api/languages/cpp_api.html#_CPPv |
|         method)](api/lang         | 4N5cudaq6pauli214num_parametersE) |
| uages/python_api.html#cudaq.opera | -   [cudaq::pauli2::num_targets   |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     membe                         |
| udaq.operators.MatrixOperatorTerm | r)](api/languages/cpp_api.html#_C |
|         method)](api/language     | PPv4N5cudaq6pauli211num_targetsE) |
| s/python_api.html#cudaq.operators | -   [cudaq::pauli2::pauli2 (C++   |
| .MatrixOperatorTerm.canonicalize) |     function)](api/languages/cpp_ |
|     -   [(                        | api.html#_CPPv4N5cudaq6pauli26pau |
| cudaq.operators.spin.SpinOperator | li2ERKNSt6vectorIN5cudaq4realEEE) |
|         method)](api/languag      | -   [cudaq::phase_damping (C++    |
| es/python_api.html#cudaq.operator |                                   |
| s.spin.SpinOperator.canonicalize) |  class)](api/languages/cpp_api.ht |
|     -   [(cuda                    | ml#_CPPv4N5cudaq13phase_dampingE) |
| q.operators.spin.SpinOperatorTerm | -   [cud                          |
|         method)](api/languages/p  | aq::phase_damping::num_parameters |
| ython_api.html#cudaq.operators.sp |     (C++                          |
| in.SpinOperatorTerm.canonicalize) |     member)](api/lan              |
| -   [canonicalized() (in module   | guages/cpp_api.html#_CPPv4N5cudaq |
|     cuda                          | 13phase_damping14num_parametersE) |
| q.boson)](api/languages/python_ap | -   [                             |
| i.html#cudaq.boson.canonicalized) | cudaq::phase_damping::num_targets |
|     -   [(in module               |     (C++                          |
|         cudaq.fe                  |     member)](api/                 |
| rmion)](api/languages/python_api. | languages/cpp_api.html#_CPPv4N5cu |
| html#cudaq.fermion.canonicalized) | daq13phase_damping11num_targetsE) |
|     -   [(in module               | -   [cudaq::phase_flip_channel    |
|                                   |     (C++                          |
|        cudaq.operators.custom)](a |     clas                          |
| pi/languages/python_api.html#cuda | s)](api/languages/cpp_api.html#_C |
| q.operators.custom.canonicalized) | PPv4N5cudaq18phase_flip_channelE) |
|     -   [(in module               | -   [cudaq::p                     |
|         cu                        | hase_flip_channel::num_parameters |
| daq.spin)](api/languages/python_a |     (C++                          |
| pi.html#cudaq.spin.canonicalized) |     member)](api/language         |
| -   [captured_variables()         | s/cpp_api.html#_CPPv4N5cudaq18pha |
|     (cudaq.PyKernelDecorator      | se_flip_channel14num_parametersE) |
|     method)](api/lan              | -   [cudaq                        |
| guages/python_api.html#cudaq.PyKe | ::phase_flip_channel::num_targets |
| rnelDecorator.captured_variables) |     (C++                          |
| -   [CentralDifference (class in  |     member)](api/langu            |
|     cudaq.gradients)              | ages/cpp_api.html#_CPPv4N5cudaq18 |
| ](api/languages/python_api.html#c | phase_flip_channel11num_targetsE) |
| udaq.gradients.CentralDifference) | -   [cudaq::product_op (C++       |
| -   [clear() (cudaq.Resources     |                                   |
|     method)](api/languages/pytho  |  class)](api/languages/cpp_api.ht |
| n_api.html#cudaq.Resources.clear) | ml#_CPPv4I0EN5cudaq10product_opE) |
|     -   [(cudaq.SampleResult      | -   [cudaq::product_op::begin     |
|                                   |     (C++                          |
|   method)](api/languages/python_a |     functio                       |
| pi.html#cudaq.SampleResult.clear) | n)](api/languages/cpp_api.html#_C |
| -   [COBYLA (class in             | PPv4NK5cudaq10product_op5beginEv) |
|     cudaq.o                       | -                                 |
| ptimizers)](api/languages/python_ |  [cudaq::product_op::canonicalize |
| api.html#cudaq.optimizers.COBYLA) |     (C++                          |
| -   [coefficient                  |     func                          |
|     (cudaq.                       | tion)](api/languages/cpp_api.html |
| operators.boson.BosonOperatorTerm | #_CPPv4N5cudaq10product_op12canon |
|     property)](api/languages/py   | icalizeERKNSt3setINSt6size_tEEE), |
| thon_api.html#cudaq.operators.bos |     [\[1\]](api                   |
| on.BosonOperatorTerm.coefficient) | /languages/cpp_api.html#_CPPv4N5c |
|     -   [(cudaq.oper              | udaq10product_op12canonicalizeEv) |
| ators.fermion.FermionOperatorTerm | -   [                             |
|                                   | cudaq::product_op::const_iterator |
|   property)](api/languages/python |     (C++                          |
| _api.html#cudaq.operators.fermion |     struct)](api/                 |
| .FermionOperatorTerm.coefficient) | languages/cpp_api.html#_CPPv4N5cu |
|     -   [(c                       | daq10product_op14const_iteratorE) |
| udaq.operators.MatrixOperatorTerm | -   [cudaq::product_o             |
|         property)](api/languag    | p::const_iterator::const_iterator |
| es/python_api.html#cudaq.operator |     (C++                          |
| s.MatrixOperatorTerm.coefficient) |     fu                            |
|     -   [(cuda                    | nction)](api/languages/cpp_api.ht |
| q.operators.spin.SpinOperatorTerm | ml#_CPPv4N5cudaq10product_op14con |
|         property)](api/languages/ | st_iterator14const_iteratorEPK10p |
| python_api.html#cudaq.operators.s | roduct_opI9HandlerTyENSt6size_tE) |
| pin.SpinOperatorTerm.coefficient) | -   [cudaq::produ                 |
| -   [col_count                    | ct_op::const_iterator::operator!= |
|     (cudaq.KrausOperator          |     (C++                          |
|     prope                         |     fun                           |
| rty)](api/languages/python_api.ht | ction)](api/languages/cpp_api.htm |
| ml#cudaq.KrausOperator.col_count) | l#_CPPv4NK5cudaq10product_op14con |
| -   [ComplexMatrix (class in      | st_iteratorneERK14const_iterator) |
|     cudaq)](api/languages/pyt     | -   [cudaq::produ                 |
| hon_api.html#cudaq.ComplexMatrix) | ct_op::const_iterator::operator\* |
| -   [compute()                    |     (C++                          |
|     (                             |     function)](api/lang           |
| cudaq.gradients.CentralDifference | uages/cpp_api.html#_CPPv4NK5cudaq |
|     method)](api/la               | 10product_op14const_iteratormlEv) |
| nguages/python_api.html#cudaq.gra | -   [cudaq::produ                 |
| dients.CentralDifference.compute) | ct_op::const_iterator::operator++ |
|     -   [(                        |     (C++                          |
| cudaq.gradients.ForwardDifference |     function)](api/lang           |
|         method)](api/la           | uages/cpp_api.html#_CPPv4N5cudaq1 |
| nguages/python_api.html#cudaq.gra | 0product_op14const_iteratorppEi), |
| dients.ForwardDifference.compute) |     [\[1\]](api/lan               |
|     -                             | guages/cpp_api.html#_CPPv4N5cudaq |
|  [(cudaq.gradients.ParameterShift | 10product_op14const_iteratorppEv) |
|         method)](api              | -   [cudaq::produc                |
| /languages/python_api.html#cudaq. | t_op::const_iterator::operator\-- |
| gradients.ParameterShift.compute) |     (C++                          |
| -   [const()                      |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|   (cudaq.operators.ScalarOperator | 0product_op14const_iteratormmEi), |
|     class                         |     [\[1\]](api/lan               |
|     method)](a                    | guages/cpp_api.html#_CPPv4N5cudaq |
| pi/languages/python_api.html#cuda | 10product_op14const_iteratormmEv) |
| q.operators.ScalarOperator.const) | -   [cudaq::produc                |
| -   [copy()                       | t_op::const_iterator::operator-\> |
|     (cu                           |     (C++                          |
| daq.operators.boson.BosonOperator |     function)](api/lan            |
|     method)](api/l                | guages/cpp_api.html#_CPPv4N5cudaq |
| anguages/python_api.html#cudaq.op | 10product_op14const_iteratorptEv) |
| erators.boson.BosonOperator.copy) | -   [cudaq::produ                 |
|     -   [(cudaq.                  | ct_op::const_iterator::operator== |
| operators.boson.BosonOperatorTerm |     (C++                          |
|         method)](api/langu        |     fun                           |
| ages/python_api.html#cudaq.operat | ction)](api/languages/cpp_api.htm |
| ors.boson.BosonOperatorTerm.copy) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(cudaq.                  | st_iteratoreqERK14const_iterator) |
| operators.fermion.FermionOperator | -   [cudaq::product_op::degrees   |
|         method)](api/langu        |     (C++                          |
| ages/python_api.html#cudaq.operat |     function)                     |
| ors.fermion.FermionOperator.copy) | ](api/languages/cpp_api.html#_CPP |
|     -   [(cudaq.oper              | v4NK5cudaq10product_op7degreesEv) |
| ators.fermion.FermionOperatorTerm | -   [cudaq::product_op::dump (C++ |
|         method)](api/languages    |     functi                        |
| /python_api.html#cudaq.operators. | on)](api/languages/cpp_api.html#_ |
| fermion.FermionOperatorTerm.copy) | CPPv4NK5cudaq10product_op4dumpEv) |
|     -                             | -   [cudaq::product_op::end (C++  |
|  [(cudaq.operators.MatrixOperator |     funct                         |
|         method)](                 | ion)](api/languages/cpp_api.html# |
| api/languages/python_api.html#cud | _CPPv4NK5cudaq10product_op3endEv) |
| aq.operators.MatrixOperator.copy) | -   [c                            |
|     -   [(c                       | udaq::product_op::get_coefficient |
| udaq.operators.MatrixOperatorTerm |     (C++                          |
|         method)](api/             |     function)](api/lan            |
| languages/python_api.html#cudaq.o | guages/cpp_api.html#_CPPv4NK5cuda |
| perators.MatrixOperatorTerm.copy) | q10product_op15get_coefficientEv) |
|     -   [(                        | -                                 |
| cudaq.operators.spin.SpinOperator |   [cudaq::product_op::get_term_id |
|         method)](api              |     (C++                          |
| /languages/python_api.html#cudaq. |     function)](api                |
| operators.spin.SpinOperator.copy) | /languages/cpp_api.html#_CPPv4NK5 |
|     -   [(cuda                    | cudaq10product_op11get_term_idEv) |
| q.operators.spin.SpinOperatorTerm | -                                 |
|         method)](api/lan          |   [cudaq::product_op::is_identity |
| guages/python_api.html#cudaq.oper |     (C++                          |
| ators.spin.SpinOperatorTerm.copy) |     function)](api                |
| -   [count() (cudaq.Resources     | /languages/cpp_api.html#_CPPv4NK5 |
|     method)](api/languages/pytho  | cudaq10product_op11is_identityEv) |
| n_api.html#cudaq.Resources.count) | -   [cudaq::product_op::num_ops   |
|     -   [(cudaq.SampleResult      |     (C++                          |
|                                   |     function)                     |
|   method)](api/languages/python_a | ](api/languages/cpp_api.html#_CPP |
| pi.html#cudaq.SampleResult.count) | v4NK5cudaq10product_op7num_opsEv) |
| -   [count_controls()             | -                                 |
|     (cudaq.Resources              |    [cudaq::product_op::operator\* |
|     meth                          |     (C++                          |
| od)](api/languages/python_api.htm |     function)](api/languages/     |
| l#cudaq.Resources.count_controls) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [counts()                     | oduct_opmlE10product_opI1TERK15sc |
|     (cudaq.ObserveResult          | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]](api/languages/        |
| method)](api/languages/python_api | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| .html#cudaq.ObserveResult.counts) | oduct_opmlE10product_opI1TERK15sc |
| -   [create() (in module          | alar_operatorRR10product_opI1TE), |
|                                   |     [\[2\]](api/languages/        |
|    cudaq.boson)](api/languages/py | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| thon_api.html#cudaq.boson.create) | oduct_opmlE10product_opI1TERR15sc |
|     -   [(in module               | alar_operatorRK10product_opI1TE), |
|         c                         |     [\[3\]](api/languages/        |
| udaq.fermion)](api/languages/pyth | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| on_api.html#cudaq.fermion.create) | oduct_opmlE10product_opI1TERR15sc |
| -   [csr_spmatrix (C++            | alar_operatorRR10product_opI1TE), |
|     type)](api/languages/c        |     [\[4\]](api/                  |
| pp_api.html#_CPPv412csr_spmatrix) | languages/cpp_api.html#_CPPv4I0EN |
| -   cudaq                         | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [module](api/langua       | K15scalar_operatorRK6sum_opI1TE), |
| ges/python_api.html#module-cudaq) |     [\[5\]](api/                  |
| -   [cudaq (C++                   | languages/cpp_api.html#_CPPv4I0EN |
|     type)](api/lan                | 5cudaq10product_opmlE6sum_opI1TER |
| guages/cpp_api.html#_CPPv45cudaq) | K15scalar_operatorRR6sum_opI1TE), |
| -   [cudaq.apply_noise() (in      |     [\[6\]](api/                  |
|     module                        | languages/cpp_api.html#_CPPv4I0EN |
|     cudaq)](api/languages/python_ | 5cudaq10product_opmlE6sum_opI1TER |
| api.html#cudaq.cudaq.apply_noise) | R15scalar_operatorRK6sum_opI1TE), |
| -   cudaq.boson                   |     [\[7\]](api/                  |
|     -   [module](api/languages/py | languages/cpp_api.html#_CPPv4I0EN |
| thon_api.html#module-cudaq.boson) | 5cudaq10product_opmlE6sum_opI1TER |
| -   cudaq.fermion                 | R15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[8\]](api/languages         |
|   -   [module](api/languages/pyth | /cpp_api.html#_CPPv4NK5cudaq10pro |
| on_api.html#module-cudaq.fermion) | duct_opmlERK6sum_opI9HandlerTyE), |
| -   cudaq.operators.custom        |     [\[9\]](api/languages/cpp_a   |
|     -   [mo                       | pi.html#_CPPv4NKR5cudaq10product_ |
| dule](api/languages/python_api.ht | opmlERK10product_opI9HandlerTyE), |
| ml#module-cudaq.operators.custom) |     [\[10\]](api/language         |
| -   cudaq.spin                    | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     -   [module](api/languages/p  | roduct_opmlERK15scalar_operator), |
| ython_api.html#module-cudaq.spin) |     [\[11\]](api/languages/cpp_a  |
| -   [cudaq::amplitude_damping     | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opmlERR10product_opI9HandlerTyE), |
|     cla                           |     [\[12\]](api/language         |
| ss)](api/languages/cpp_api.html#_ | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| CPPv4N5cudaq17amplitude_dampingE) | roduct_opmlERR15scalar_operator), |
| -                                 |     [\[13\]](api/languages/cpp_   |
| [cudaq::amplitude_damping_channel | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmlERK10product_opI9HandlerTyE), |
|     class)](api                   |     [\[14\]](api/languag          |
| /languages/cpp_api.html#_CPPv4N5c | es/cpp_api.html#_CPPv4NO5cudaq10p |
| udaq25amplitude_damping_channelE) | roduct_opmlERK15scalar_operator), |
| -   [cudaq::amplitud              |     [\[15\]](api/languages/cpp_   |
| e_damping_channel::num_parameters | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmlERR10product_opI9HandlerTyE), |
|     member)](api/languages/cpp_a  |     [\[16\]](api/langua           |
| pi.html#_CPPv4N5cudaq25amplitude_ | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| damping_channel14num_parametersE) | product_opmlERR15scalar_operator) |
| -   [cudaq::ampli                 | -                                 |
| tude_damping_channel::num_targets |   [cudaq::product_op::operator\*= |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     function)](api/languages/cpp  |
| p_api.html#_CPPv4N5cudaq25amplitu | _api.html#_CPPv4N5cudaq10product_ |
| de_damping_channel11num_targetsE) | opmLERK10product_opI9HandlerTyE), |
| -   [cudaq::AnalogRemoteRESTQPU   |     [\[1\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     class                         | roduct_opmLERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[2\]](api/languages/cp      |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq::apply_noise (C++      | _opmLERR10product_opI9HandlerTyE) |
|     function)](api/               | -   [cudaq::product_op::operator+ |
| languages/cpp_api.html#_CPPv4I0Dp |     (C++                          |
| EN5cudaq11apply_noiseEvDpRR4Args) |     function)](api/langu          |
| -   [cudaq::async_result (C++     | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     c                             | q10product_opplE6sum_opI1TERK15sc |
| lass)](api/languages/cpp_api.html | alar_operatorRK10product_opI1TE), |
| #_CPPv4I0EN5cudaq12async_resultE) |     [\[1\]](api/                  |
| -   [cudaq::async_result::get     | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     functi                        | K15scalar_operatorRK6sum_opI1TE), |
| on)](api/languages/cpp_api.html#_ |     [\[2\]](api/langu             |
| CPPv4N5cudaq12async_result3getEv) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::async_sample_result   | q10product_opplE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     type                          |     [\[3\]](api/                  |
| )](api/languages/cpp_api.html#_CP | languages/cpp_api.html#_CPPv4I0EN |
| Pv4N5cudaq19async_sample_resultE) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq::BaseRemoteRESTQPU     | K15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[4\]](api/langu             |
|     cla                           | ages/cpp_api.html#_CPPv4I0EN5cuda |
| ss)](api/languages/cpp_api.html#_ | q10product_opplE6sum_opI1TERR15sc |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | alar_operatorRK10product_opI1TE), |
| -                                 |     [\[5\]](api/                  |
|    [cudaq::BaseRemoteSimulatorQPU | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opplE6sum_opI1TER |
|     class)](                      | R15scalar_operatorRK6sum_opI1TE), |
| api/languages/cpp_api.html#_CPPv4 |     [\[6\]](api/langu             |
| N5cudaq22BaseRemoteSimulatorQPUE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::bit_flip_channel (C++ | q10product_opplE6sum_opI1TERR15sc |
|     cl                            | alar_operatorRR10product_opI1TE), |
| ass)](api/languages/cpp_api.html# |     [\[7\]](api/                  |
| _CPPv4N5cudaq16bit_flip_channelE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq:                       | 5cudaq10product_opplE6sum_opI1TER |
| :bit_flip_channel::num_parameters | R15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[8\]](api/languages/cpp_a   |
|     member)](api/langua           | pi.html#_CPPv4NKR5cudaq10product_ |
| ges/cpp_api.html#_CPPv4N5cudaq16b | opplERK10product_opI9HandlerTyE), |
| it_flip_channel14num_parametersE) |     [\[9\]](api/language          |
| -   [cud                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| aq::bit_flip_channel::num_targets | roduct_opplERK15scalar_operator), |
|     (C++                          |     [\[10\]](api/languages/       |
|     member)](api/lan              | cpp_api.html#_CPPv4NKR5cudaq10pro |
| guages/cpp_api.html#_CPPv4N5cudaq | duct_opplERK6sum_opI9HandlerTyE), |
| 16bit_flip_channel11num_targetsE) |     [\[11\]](api/languages/cpp_a  |
| -   [cudaq::boson_handler (C++    | pi.html#_CPPv4NKR5cudaq10product_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|  class)](api/languages/cpp_api.ht |     [\[12\]](api/language         |
| ml#_CPPv4N5cudaq13boson_handlerE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::boson_op (C++         | roduct_opplERR15scalar_operator), |
|     type)](api/languages/cpp_     |     [\[13\]](api/languages/       |
| api.html#_CPPv4N5cudaq8boson_opE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::boson_op_term (C++    | duct_opplERR6sum_opI9HandlerTyE), |
|                                   |     [\[                           |
|   type)](api/languages/cpp_api.ht | 14\]](api/languages/cpp_api.html# |
| ml#_CPPv4N5cudaq13boson_op_termE) | _CPPv4NKR5cudaq10product_opplEv), |
| -   [cudaq::CodeGenConfig (C++    |     [\[15\]](api/languages/cpp_   |
|                                   | api.html#_CPPv4NO5cudaq10product_ |
| struct)](api/languages/cpp_api.ht | opplERK10product_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     [\[16\]](api/languag          |
| -   [cudaq::commutation_relations | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opplERK15scalar_operator), |
|     struct)]                      |     [\[17\]](api/languages        |
| (api/languages/cpp_api.html#_CPPv | /cpp_api.html#_CPPv4NO5cudaq10pro |
| 4N5cudaq21commutation_relationsE) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cudaq::complex (C++          |     [\[18\]](api/languages/cpp_   |
|     type)](api/languages/cpp      | api.html#_CPPv4NO5cudaq10product_ |
| _api.html#_CPPv4N5cudaq7complexE) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq::complex_matrix (C++   |     [\[19\]](api/languag          |
|                                   | es/cpp_api.html#_CPPv4NO5cudaq10p |
| class)](api/languages/cpp_api.htm | roduct_opplERR15scalar_operator), |
| l#_CPPv4N5cudaq14complex_matrixE) |     [\[20\]](api/languages        |
| -                                 | /cpp_api.html#_CPPv4NO5cudaq10pro |
|   [cudaq::complex_matrix::adjoint | duct_opplERR6sum_opI9HandlerTyE), |
|     (C++                          |     [                             |
|     function)](a                  | \[21\]](api/languages/cpp_api.htm |
| pi/languages/cpp_api.html#_CPPv4N | l#_CPPv4NO5cudaq10product_opplEv) |
| 5cudaq14complex_matrix7adjointEv) | -   [cudaq::product_op::operator- |
| -   [cudaq::                      |     (C++                          |
| complex_matrix::diagonal_elements |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     function)](api/languages      | q10product_opmiE6sum_opI1TERK15sc |
| /cpp_api.html#_CPPv4NK5cudaq14com | alar_operatorRK10product_opI1TE), |
| plex_matrix17diagonal_elementsEi) |     [\[1\]](api/                  |
| -   [cudaq::complex_matrix::dump  | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/language       | K15scalar_operatorRK6sum_opI1TE), |
| s/cpp_api.html#_CPPv4NK5cudaq14co |     [\[2\]](api/langu             |
| mplex_matrix4dumpERNSt7ostreamE), | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     [\[1\]]                       | q10product_opmiE6sum_opI1TERK15sc |
| (api/languages/cpp_api.html#_CPPv | alar_operatorRR10product_opI1TE), |
| 4NK5cudaq14complex_matrix4dumpEv) |     [\[3\]](api/                  |
| -   [c                            | languages/cpp_api.html#_CPPv4I0EN |
| udaq::complex_matrix::eigenvalues | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRR6sum_opI1TE), |
|     function)](api/lan            |     [\[4\]](api/langu             |
| guages/cpp_api.html#_CPPv4NK5cuda | ages/cpp_api.html#_CPPv4I0EN5cuda |
| q14complex_matrix11eigenvaluesEv) | q10product_opmiE6sum_opI1TERR15sc |
| -   [cu                           | alar_operatorRK10product_opI1TE), |
| daq::complex_matrix::eigenvectors |     [\[5\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function)](api/lang           | 5cudaq10product_opmiE6sum_opI1TER |
| uages/cpp_api.html#_CPPv4NK5cudaq | R15scalar_operatorRK6sum_opI1TE), |
| 14complex_matrix12eigenvectorsEv) |     [\[6\]](api/langu             |
| -   [c                            | ages/cpp_api.html#_CPPv4I0EN5cuda |
| udaq::complex_matrix::exponential | q10product_opmiE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     function)](api/la             |     [\[7\]](api/                  |
| nguages/cpp_api.html#_CPPv4N5cuda | languages/cpp_api.html#_CPPv4I0EN |
| q14complex_matrix11exponentialEv) | 5cudaq10product_opmiE6sum_opI1TER |
| -                                 | R15scalar_operatorRR6sum_opI1TE), |
|  [cudaq::complex_matrix::identity |     [\[8\]](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4NKR5cudaq10product_ |
|     function)](api/languages      | opmiERK10product_opI9HandlerTyE), |
| /cpp_api.html#_CPPv4N5cudaq14comp |     [\[9\]](api/language          |
| lex_matrix8identityEKNSt6size_tE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -                                 | roduct_opmiERK15scalar_operator), |
| [cudaq::complex_matrix::kronecker |     [\[10\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     function)](api/lang           | duct_opmiERK6sum_opI9HandlerTyE), |
| uages/cpp_api.html#_CPPv4I00EN5cu |     [\[11\]](api/languages/cpp_a  |
| daq14complex_matrix9kroneckerE14c | pi.html#_CPPv4NKR5cudaq10product_ |
| omplex_matrix8Iterable8Iterable), | opmiERR10product_opI9HandlerTyE), |
|     [\[1\]](api/l                 |     [\[12\]](api/language         |
| anguages/cpp_api.html#_CPPv4N5cud | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| aq14complex_matrix9kroneckerERK14 | roduct_opmiERR15scalar_operator), |
| complex_matrixRK14complex_matrix) |     [\[13\]](api/languages/       |
| -   [cudaq::c                     | cpp_api.html#_CPPv4NKR5cudaq10pro |
| omplex_matrix::minimal_eigenvalue | duct_opmiERR6sum_opI9HandlerTyE), |
|     (C++                          |     [\[                           |
|     function)](api/languages/     | 14\]](api/languages/cpp_api.html# |
| cpp_api.html#_CPPv4NK5cudaq14comp | _CPPv4NKR5cudaq10product_opmiEv), |
| lex_matrix18minimal_eigenvalueEv) |     [\[15\]](api/languages/cpp_   |
| -   [                             | api.html#_CPPv4NO5cudaq10product_ |
| cudaq::complex_matrix::operator() | opmiERK10product_opI9HandlerTyE), |
|     (C++                          |     [\[16\]](api/languag          |
|     function)](api/languages/cpp  | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _api.html#_CPPv4N5cudaq14complex_ | roduct_opmiERK15scalar_operator), |
| matrixclENSt6size_tENSt6size_tE), |     [\[17\]](api/languages        |
|     [\[1\]](api/languages/cpp     | /cpp_api.html#_CPPv4NO5cudaq10pro |
| _api.html#_CPPv4NK5cudaq14complex | duct_opmiERK6sum_opI9HandlerTyE), |
| _matrixclENSt6size_tENSt6size_tE) |     [\[18\]](api/languages/cpp_   |
| -   [                             | api.html#_CPPv4NO5cudaq10product_ |
| cudaq::complex_matrix::operator\* | opmiERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[19\]](api/languag          |
|     function)](api/langua         | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ges/cpp_api.html#_CPPv4N5cudaq14c | roduct_opmiERR15scalar_operator), |
| omplex_matrixmlEN14complex_matrix |     [\[20\]](api/languages        |
| 10value_typeERK14complex_matrix), | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     [\[1\]                        | duct_opmiERR6sum_opI9HandlerTyE), |
| ](api/languages/cpp_api.html#_CPP |     [                             |
| v4N5cudaq14complex_matrixmlERK14c | \[21\]](api/languages/cpp_api.htm |
| omplex_matrixRK14complex_matrix), | l#_CPPv4NO5cudaq10product_opmiEv) |
|                                   | -   [cudaq::product_op::operator/ |
|  [\[2\]](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14complex_matrixm |     function)](api/language       |
| lERK14complex_matrixRKNSt6vectorI | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| N14complex_matrix10value_typeEEE) | roduct_opdvERK15scalar_operator), |
| -                                 |     [\[1\]](api/language          |
| [cudaq::complex_matrix::operator+ | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opdvERR15scalar_operator), |
|     function                      |     [\[2\]](api/languag           |
| )](api/languages/cpp_api.html#_CP | es/cpp_api.html#_CPPv4NO5cudaq10p |
| Pv4N5cudaq14complex_matrixplERK14 | roduct_opdvERK15scalar_operator), |
| complex_matrixRK14complex_matrix) |     [\[3\]](api/langua            |
| -                                 | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| [cudaq::complex_matrix::operator- | product_opdvERR15scalar_operator) |
|     (C++                          | -                                 |
|     function                      |    [cudaq::product_op::operator/= |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq14complex_matrixmiERK14 |     function)](api/langu          |
| complex_matrixRK14complex_matrix) | ages/cpp_api.html#_CPPv4N5cudaq10 |
| -   [cu                           | product_opdVERK15scalar_operator) |
| daq::complex_matrix::operator\[\] | -   [cudaq::product_op::operator= |
|     (C++                          |     (C++                          |
|                                   |     function)](api/la             |
|  function)](api/languages/cpp_api | nguages/cpp_api.html#_CPPv4I0_NSt |
| .html#_CPPv4N5cudaq14complex_matr | 11enable_if_tIXaantNSt7is_sameI1T |
| ixixERKNSt6vectorINSt6size_tEEE), | 9HandlerTyE5valueENSt16is_constru |
|     [\[1\]](api/languages/cpp_api | ctibleI9HandlerTy1TE5valueEEbEEEN |
| .html#_CPPv4NK5cudaq14complex_mat | 5cudaq10product_opaSER10product_o |
| rixixERKNSt6vectorINSt6size_tEEE) | pI9HandlerTyERK10product_opI1TE), |
| -   [cudaq::complex_matrix::power |     [\[1\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4N5cudaq10product_ |
|     function)]                    | opaSERK10product_opI9HandlerTyE), |
| (api/languages/cpp_api.html#_CPPv |     [\[2\]](api/languages/cp      |
| 4N5cudaq14complex_matrix5powerEi) | p_api.html#_CPPv4N5cudaq10product |
| -                                 | _opaSERR10product_opI9HandlerTyE) |
|  [cudaq::complex_matrix::set_zero | -                                 |
|     (C++                          |    [cudaq::product_op::operator== |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languages/cpp  |
| cudaq14complex_matrix8set_zeroEv) | _api.html#_CPPv4NK5cudaq10product |
| -                                 | _opeqERK10product_opI9HandlerTyE) |
| [cudaq::complex_matrix::to_string | -                                 |
|     (C++                          |  [cudaq::product_op::operator\[\] |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     function)](ap                 |
| udaq14complex_matrix9to_stringEv) | i/languages/cpp_api.html#_CPPv4NK |
| -   [                             | 5cudaq10product_opixENSt6size_tE) |
| cudaq::complex_matrix::value_type | -                                 |
|     (C++                          |    [cudaq::product_op::product_op |
|     type)](api/                   |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/languages/c    |
| daq14complex_matrix10value_typeE) | pp_api.html#_CPPv4I0_NSt11enable_ |
| -   [cudaq::contrib (C++          | if_tIXaaNSt7is_sameI9HandlerTy14m |
|     type)](api/languages/cpp      | atrix_handlerE5valueEaantNSt7is_s |
| _api.html#_CPPv4N5cudaq7contribE) | ameI1T9HandlerTyE5valueENSt16is_c |
| -   [cudaq::contrib::draw (C++    | onstructibleI9HandlerTy1TE5valueE |
|     function)                     | EbEEEN5cudaq10product_op10product |
| ](api/languages/cpp_api.html#_CPP | _opERK10product_opI1TERKN14matrix |
| v4I0DpEN5cudaq7contrib4drawENSt6s | _handler20commutation_behaviorE), |
| tringERR13QuantumKernelDpRR4Args) |                                   |
| -                                 |  [\[1\]](api/languages/cpp_api.ht |
| [cudaq::contrib::get_unitary_cmat | ml#_CPPv4I0_NSt11enable_if_tIXaan |
|     (C++                          | tNSt7is_sameI1T9HandlerTyE5valueE |
|     function)](api/languages/cp   | NSt16is_constructibleI9HandlerTy1 |
| p_api.html#_CPPv4I0DpEN5cudaq7con | TE5valueEEbEEEN5cudaq10product_op |
| trib16get_unitary_cmatE14complex_ | 10product_opERK10product_opI1TE), |
| matrixRR13QuantumKernelDpRR4Args) |                                   |
| -   [cudaq::CusvState (C++        |   [\[2\]](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq10product_op10pr |
|    class)](api/languages/cpp_api. | oduct_opENSt6size_tENSt6size_tE), |
| html#_CPPv4I0EN5cudaq9CusvStateE) |     [\[3\]](api/languages/cp      |
| -   [cudaq::depolarization1 (C++  | p_api.html#_CPPv4N5cudaq10product |
|     c                             | _op10product_opENSt7complexIdEE), |
| lass)](api/languages/cpp_api.html |     [\[4\]](api/l                 |
| #_CPPv4N5cudaq15depolarization1E) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq::depolarization2 (C++  | aq10product_op10product_opERK10pr |
|     c                             | oduct_opI9HandlerTyENSt6size_tE), |
| lass)](api/languages/cpp_api.html |     [\[5\]](api/l                 |
| #_CPPv4N5cudaq15depolarization2E) | anguages/cpp_api.html#_CPPv4N5cud |
| -   [cudaq:                       | aq10product_op10product_opERR10pr |
| :depolarization2::depolarization2 | oduct_opI9HandlerTyENSt6size_tE), |
|     (C++                          |     [\[6\]](api/languages         |
|     function)](api/languages/cp   | /cpp_api.html#_CPPv4N5cudaq10prod |
| p_api.html#_CPPv4N5cudaq15depolar | uct_op10product_opERR9HandlerTy), |
| ization215depolarization2EK4real) |     [\[7\]](ap                    |
| -   [cudaq                        | i/languages/cpp_api.html#_CPPv4N5 |
| ::depolarization2::num_parameters | cudaq10product_op10product_opEd), |
|     (C++                          |     [\[8\]](a                     |
|     member)](api/langu            | pi/languages/cpp_api.html#_CPPv4N |
| ages/cpp_api.html#_CPPv4N5cudaq15 | 5cudaq10product_op10product_opEv) |
| depolarization214num_parametersE) | -   [cuda                         |
| -   [cu                           | q::product_op::to_diagonal_matrix |
| daq::depolarization2::num_targets |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)](api/la               | languages/cpp_api.html#_CPPv4NK5c |
| nguages/cpp_api.html#_CPPv4N5cuda | udaq10product_op18to_diagonal_mat |
| q15depolarization211num_targetsE) | rixENSt13unordered_mapINSt6size_t |
| -                                 | ENSt7int64_tEEERKNSt13unordered_m |
|    [cudaq::depolarization_channel | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -   [cudaq::product_op::to_matrix |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     funct                         |
| N5cudaq22depolarization_channelE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::depol                 | _CPPv4NK5cudaq10product_op9to_mat |
| arization_channel::num_parameters | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     member)](api/languages/cp     | apINSt6stringENSt7complexIdEEEEb) |
| p_api.html#_CPPv4N5cudaq22depolar | -   [cu                           |
| ization_channel14num_parametersE) | daq::product_op::to_sparse_matrix |
| -   [cudaq::de                    |     (C++                          |
| polarization_channel::num_targets |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     member)](api/languages        | 5cudaq10product_op16to_sparse_mat |
| /cpp_api.html#_CPPv4N5cudaq22depo | rixENSt13unordered_mapINSt6size_t |
| larization_channel11num_targetsE) | ENSt7int64_tEEERKNSt13unordered_m |
| -   [cudaq::details (C++          | apINSt6stringENSt7complexIdEEEEb) |
|     type)](api/languages/cpp      | -   [cudaq::product_op::to_string |
| _api.html#_CPPv4N5cudaq7detailsE) |     (C++                          |
| -   [cudaq::details::future (C++  |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|  class)](api/languages/cpp_api.ht | NK5cudaq10product_op9to_stringEv) |
| ml#_CPPv4N5cudaq7details6futureE) | -                                 |
| -                                 |  [cudaq::product_op::\~product_op |
|   [cudaq::details::future::future |     (C++                          |
|     (C++                          |     fu                            |
|     functio                       | nction)](api/languages/cpp_api.ht |
| n)](api/languages/cpp_api.html#_C | ml#_CPPv4N5cudaq10product_opD0Ev) |
| PPv4N5cudaq7details6future6future | -   [cudaq::QPU (C++              |
| ERNSt6vectorI3JobEERNSt6stringERN |     class)](api/languages         |
| St3mapINSt6stringENSt6stringEEE), | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     [\[1\]](api/lang              | -   [cudaq::QPU::beginExecution   |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     (C++                          |
| details6future6futureERR6future), |     function                      |
|     [\[2\]]                       | )](api/languages/cpp_api.html#_CP |
| (api/languages/cpp_api.html#_CPPv | Pv4N5cudaq3QPU14beginExecutionEv) |
| 4N5cudaq7details6future6futureEv) | -   [cuda                         |
| -   [cu                           | q::QPU::configureExecutionContext |
| daq::details::kernel_builder_base |     (C++                          |
|     (C++                          |     funct                         |
|     class)](api/l                 | ion)](api/languages/cpp_api.html# |
| anguages/cpp_api.html#_CPPv4N5cud | _CPPv4NK5cudaq3QPU25configureExec |
| aq7details19kernel_builder_baseE) | utionContextER16ExecutionContext) |
| -   [cudaq::details::             | -   [cudaq::QPU::endExecution     |
| kernel_builder_base::operator\<\< |     (C++                          |
|     (C++                          |     functi                        |
|     function)](api/langua         | on)](api/languages/cpp_api.html#_ |
| ges/cpp_api.html#_CPPv4N5cudaq7de | CPPv4N5cudaq3QPU12endExecutionEv) |
| tails19kernel_builder_baselsERNSt | -   [cudaq::QPU::enqueue (C++     |
| 7ostreamERK19kernel_builder_base) |     function)](ap                 |
| -   [                             | i/languages/cpp_api.html#_CPPv4N5 |
| cudaq::details::KernelBuilderType | cudaq3QPU7enqueueER11QuantumTask) |
|     (C++                          | -   [cud                          |
|     class)](api                   | aq::QPU::finalizeExecutionContext |
| /languages/cpp_api.html#_CPPv4N5c |     (C++                          |
| udaq7details17KernelBuilderTypeE) |     func                          |
| -   [cudaq::d                     | tion)](api/languages/cpp_api.html |
| etails::KernelBuilderType::create | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     function)                     | -   [cudaq::QPU::getConnectivity  |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq7details17KernelBuilderT |     function)                     |
| ype6createEPN4mlir11MLIRContextE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::details::Ker          | v4N5cudaq3QPU15getConnectivityEv) |
| nelBuilderType::KernelBuilderType | -                                 |
|     (C++                          | [cudaq::QPU::getExecutionThreadId |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     function)](api/               |
| details17KernelBuilderType17Kerne | languages/cpp_api.html#_CPPv4NK5c |
| lBuilderTypeERRNSt8functionIFN4ml | udaq3QPU20getExecutionThreadIdEv) |
| ir4TypeEPN4mlir11MLIRContextEEEE) | -   [cudaq::QPU::getNumQubits     |
| -   [cudaq::diag_matrix_callback  |     (C++                          |
|     (C++                          |     functi                        |
|     class)                        | on)](api/languages/cpp_api.html#_ |
| ](api/languages/cpp_api.html#_CPP | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| v4N5cudaq20diag_matrix_callbackE) | -   [                             |
| -   [cudaq::dyn (C++              | cudaq::QPU::getRemoteCapabilities |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq3dynE) |     function)](api/l              |
| -   [cudaq::ExecutionContext (C++ | anguages/cpp_api.html#_CPPv4NK5cu |
|     cl                            | daq3QPU21getRemoteCapabilitiesEv) |
| ass)](api/languages/cpp_api.html# | -   [cudaq::QPU::isEmulated (C++  |
| _CPPv4N5cudaq16ExecutionContextE) |     func                          |
| -   [cudaq                        | tion)](api/languages/cpp_api.html |
| ::ExecutionContext::amplitudeMaps | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     (C++                          | -   [cudaq::QPU::isSimulator (C++ |
|     member)](api/langu            |     funct                         |
| ages/cpp_api.html#_CPPv4N5cudaq16 | ion)](api/languages/cpp_api.html# |
| ExecutionContext13amplitudeMapsE) | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| -   [c                            | -   [cudaq::QPU::launchKernel     |
| udaq::ExecutionContext::asyncExec |     (C++                          |
|     (C++                          |     function)](api/               |
|     member)](api/                 | languages/cpp_api.html#_CPPv4N5cu |
| languages/cpp_api.html#_CPPv4N5cu | daq3QPU12launchKernelERKNSt6strin |
| daq16ExecutionContext9asyncExecE) | gE15KernelThunkTypePvNSt8uint64_t |
| -   [cud                          | ENSt8uint64_tERKNSt6vectorIPvEE), |
| aq::ExecutionContext::asyncResult |                                   |
|     (C++                          |  [\[1\]](api/languages/cpp_api.ht |
|     member)](api/lan              | ml#_CPPv4N5cudaq3QPU12launchKerne |
| guages/cpp_api.html#_CPPv4N5cudaq | lERKNSt6stringERKNSt6vectorIPvEE) |
| 16ExecutionContext11asyncResultE) | -   [cudaq::QPU::onRandomSeedSet  |
| -   [cudaq:                       |     (C++                          |
| :ExecutionContext::batchIteration |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq3 |
|     member)](api/langua           | QPU15onRandomSeedSetENSt6size_tE) |
| ges/cpp_api.html#_CPPv4N5cudaq16E | -   [cudaq::QPU::QPU (C++         |
| xecutionContext14batchIterationE) |     functio                       |
| -   [cudaq::E                     | n)](api/languages/cpp_api.html#_C |
| xecutionContext::canHandleObserve | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
|     (C++                          |                                   |
|     member)](api/language         |  [\[1\]](api/languages/cpp_api.ht |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| cutionContext16canHandleObserveE) |     [\[2\]](api/languages/cpp_    |
| -   [cudaq::E                     | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| xecutionContext::ExecutionContext | -   [cudaq::QPU::setId (C++       |
|     (C++                          |     function                      |
|     func                          | )](api/languages/cpp_api.html#_CP |
| tion)](api/languages/cpp_api.html | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| #_CPPv4N5cudaq16ExecutionContext1 | -   [cudaq::QPU::setShots (C++    |
| 6ExecutionContextERKNSt6stringE), |     f                             |
|     [\[1\]](api/languages/        | unction)](api/languages/cpp_api.h |
| cpp_api.html#_CPPv4N5cudaq16Execu | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| tionContext16ExecutionContextERKN | -   [cudaq::                      |
| St6stringENSt6size_tENSt6size_tE) | QPU::supportsExplicitMeasurements |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::expectationValue |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq3QPU |
|     member)](api/language         | 28supportsExplicitMeasurementsEv) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::QPU::\~QPU (C++       |
| cutionContext16expectationValueE) |     function)](api/languages/cp   |
| -   [cudaq::Execu                 | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| tionContext::explicitMeasurements | -   [cudaq::QPUState (C++         |
|     (C++                          |     class)](api/languages/cpp_    |
|     member)](api/languages/cp     | api.html#_CPPv4N5cudaq8QPUStateE) |
| p_api.html#_CPPv4N5cudaq16Executi | -   [cudaq::qreg (C++             |
| onContext20explicitMeasurementsE) |     class)](api/lan               |
| -   [cuda                         | guages/cpp_api.html#_CPPv4I_NSt6s |
| q::ExecutionContext::futureResult | ize_tE_NSt6size_tEEN5cudaq4qregE) |
|     (C++                          | -   [cudaq::qreg::back (C++       |
|     member)](api/lang             |     function)                     |
| uages/cpp_api.html#_CPPv4N5cudaq1 | ](api/languages/cpp_api.html#_CPP |
| 6ExecutionContext12futureResultE) | v4N5cudaq4qreg4backENSt6size_tE), |
| -   [cudaq::ExecutionContext      |     [\[1\]](api/languages/cpp_ap  |
| ::hasConditionalsOnMeasureResults | i.html#_CPPv4N5cudaq4qreg4backEv) |
|     (C++                          | -   [cudaq::qreg::begin (C++      |
|     mem                           |                                   |
| ber)](api/languages/cpp_api.html# |  function)](api/languages/cpp_api |
| _CPPv4N5cudaq16ExecutionContext31 | .html#_CPPv4N5cudaq4qreg5beginEv) |
| hasConditionalsOnMeasureResultsE) | -   [cudaq::qreg::clear (C++      |
| -   [cudaq::Executi               |                                   |
| onContext::invocationResultBuffer |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5clearEv) |
|     member)](api/languages/cpp_   | -   [cudaq::qreg::front (C++      |
| api.html#_CPPv4N5cudaq16Execution |     function)]                    |
| Context22invocationResultBufferE) | (api/languages/cpp_api.html#_CPPv |
| -   [cu                           | 4N5cudaq4qreg5frontENSt6size_tE), |
| daq::ExecutionContext::kernelName |     [\[1\]](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5frontEv) |
|     member)](api/la               | -   [cudaq::qreg::operator\[\]    |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10kernelNameE) |     functi                        |
| -   [cud                          | on)](api/languages/cpp_api.html#_ |
| aq::ExecutionContext::kernelTrace | CPPv4N5cudaq4qregixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qreg::qreg (C++       |
|     member)](api/lan              |     function)                     |
| guages/cpp_api.html#_CPPv4N5cudaq | ](api/languages/cpp_api.html#_CPP |
| 16ExecutionContext11kernelTraceE) | v4N5cudaq4qreg4qregENSt6size_tE), |
| -   [cudaq:                       |     [\[1\]](api/languages/cpp_ap  |
| :ExecutionContext::msm_dimensions | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|     (C++                          | -   [cudaq::qreg::size (C++       |
|     member)](api/langua           |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq16E |  function)](api/languages/cpp_api |
| xecutionContext14msm_dimensionsE) | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| -   [cudaq::                      | -   [cudaq::qreg::slice (C++      |
| ExecutionContext::msm_prob_err_id |     function)](api/langu          |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq4q |
|     member)](api/languag          | reg5sliceENSt6size_tENSt6size_tE) |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | -   [cudaq::qreg::value_type (C++ |
| ecutionContext15msm_prob_err_idE) |                                   |
| -   [cudaq::Ex                    | type)](api/languages/cpp_api.html |
| ecutionContext::msm_probabilities | #_CPPv4N5cudaq4qreg10value_typeE) |
|     (C++                          | -   [cudaq::qspan (C++            |
|     member)](api/languages        |     class)](api/lang              |
| /cpp_api.html#_CPPv4N5cudaq16Exec | uages/cpp_api.html#_CPPv4I_NSt6si |
| utionContext17msm_probabilitiesE) | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| -                                 | -   [cudaq::QuakeValue (C++       |
|    [cudaq::ExecutionContext::name |     class)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq10QuakeValueE) |
|     member)]                      | -   [cudaq::Q                     |
| (api/languages/cpp_api.html#_CPPv | uakeValue::canValidateNumElements |
| 4N5cudaq16ExecutionContext4nameE) |     (C++                          |
| -   [cu                           |     function)](api/languages      |
| daq::ExecutionContext::noiseModel | /cpp_api.html#_CPPv4N5cudaq10Quak |
|     (C++                          | eValue22canValidateNumElementsEv) |
|     member)](api/la               | -                                 |
| nguages/cpp_api.html#_CPPv4N5cuda |  [cudaq::QuakeValue::constantSize |
| q16ExecutionContext10noiseModelE) |     (C++                          |
| -   [cudaq::Exe                   |     function)](api                |
| cutionContext::numberTrajectories | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValue12constantSizeEv) |
|     member)](api/languages/       | -   [cudaq::QuakeValue::dump (C++ |
| cpp_api.html#_CPPv4N5cudaq16Execu |     function)](api/lan            |
| tionContext18numberTrajectoriesE) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [c                            | 10QuakeValue4dumpERNSt7ostreamE), |
| udaq::ExecutionContext::optResult |     [\                            |
|     (C++                          | [1\]](api/languages/cpp_api.html# |
|     member)](api/                 | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq                        |
| daq16ExecutionContext9optResultE) | ::QuakeValue::getRequiredElements |
| -   [cudaq::Execu                 |     (C++                          |
| tionContext::overlapComputeStates |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq10Q |
|     member)](api/languages/cp     | uakeValue19getRequiredElementsEv) |
| p_api.html#_CPPv4N5cudaq16Executi | -   [cudaq::QuakeValue::getValue  |
| onContext20overlapComputeStatesE) |     (C++                          |
| -   [cudaq                        |     function)]                    |
| ::ExecutionContext::overlapResult | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NK5cudaq10QuakeValue8getValueEv) |
|     member)](api/langu            | -   [cudaq::QuakeValue::inverse   |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13overlapResultE) |     function)                     |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|   [cudaq::ExecutionContext::qpuId | v4NK5cudaq10QuakeValue7inverseEv) |
|     (C++                          | -   [cudaq::QuakeValue::isStdVec  |
|     member)](                     |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)                     |
| N5cudaq16ExecutionContext5qpuIdE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq                        | v4N5cudaq10QuakeValue8isStdVecEv) |
| ::ExecutionContext::registerNames | -                                 |
|     (C++                          |    [cudaq::QuakeValue::operator\* |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     function)](api                |
| ExecutionContext13registerNamesE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cu                           | udaq10QuakeValuemlE10QuakeValue), |
| daq::ExecutionContext::reorderIdx |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     member)](api/la               | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QuakeValue::operator+ |
| q16ExecutionContext10reorderIdxE) |     (C++                          |
| -                                 |     function)](api                |
|  [cudaq::ExecutionContext::result | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValueplE10QuakeValue), |
|     member)](a                    |     [                             |
| pi/languages/cpp_api.html#_CPPv4N | \[1\]](api/languages/cpp_api.html |
| 5cudaq16ExecutionContext6resultE) | #_CPPv4N5cudaq10QuakeValueplEKd), |
| -                                 |                                   |
|   [cudaq::ExecutionContext::shots | [\[2\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|     member)](                     | -   [cudaq::QuakeValue::operator- |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5shotsE) |     function)](api                |
| -   [cudaq::                      | /languages/cpp_api.html#_CPPv4N5c |
| ExecutionContext::simulationState | udaq10QuakeValuemiE10QuakeValue), |
|     (C++                          |     [                             |
|     member)](api/languag          | \[1\]](api/languages/cpp_api.html |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| ecutionContext15simulationStateE) |     [                             |
| -                                 | \[2\]](api/languages/cpp_api.html |
|    [cudaq::ExecutionContext::spin | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     (C++                          |                                   |
|     member)]                      | [\[3\]](api/languages/cpp_api.htm |
| (api/languages/cpp_api.html#_CPPv | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| 4N5cudaq16ExecutionContext4spinE) | -   [cudaq::QuakeValue::operator/ |
| -   [cudaq::                      |     (C++                          |
| ExecutionContext::totalIterations |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/languag          | udaq10QuakeValuedvE10QuakeValue), |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |                                   |
| ecutionContext15totalIterationsE) | [\[1\]](api/languages/cpp_api.htm |
| -   [cudaq::Executio              | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| nContext::warnedNamedMeasurements | -                                 |
|     (C++                          |  [cudaq::QuakeValue::operator\[\] |
|     member)](api/languages/cpp_a  |     (C++                          |
| pi.html#_CPPv4N5cudaq16ExecutionC |     function)](api                |
| ontext23warnedNamedMeasurementsE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::ExecutionResult (C++  | udaq10QuakeValueixEKNSt6size_tE), |
|     st                            |     [\[1\]](api/                  |
| ruct)](api/languages/cpp_api.html | languages/cpp_api.html#_CPPv4N5cu |
| #_CPPv4N5cudaq15ExecutionResultE) | daq10QuakeValueixERK10QuakeValue) |
| -   [cud                          | -                                 |
| aq::ExecutionResult::appendResult |    [cudaq::QuakeValue::QuakeValue |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api/languag        |
| n)](api/languages/cpp_api.html#_C | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| PPv4N5cudaq15ExecutionResult12app | akeValue10QuakeValueERN4mlir20Imp |
| endResultENSt6stringENSt6size_tE) | licitLocOpBuilderEN4mlir5ValueE), |
| -   [cu                           |     [\[1\]                        |
| daq::ExecutionResult::deserialize | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq10QuakeValue10QuakeValue |
|     function)                     | ERN4mlir20ImplicitLocOpBuilderEd) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::QuakeValue::size (C++ |
| v4N5cudaq15ExecutionResult11deser |     funct                         |
| ializeERNSt6vectorINSt6size_tEEE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq:                       | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| :ExecutionResult::ExecutionResult | -   [cudaq::QuakeValue::slice     |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api/languages/cpp_ |
| n)](api/languages/cpp_api.html#_C | api.html#_CPPv4N5cudaq10QuakeValu |
| PPv4N5cudaq15ExecutionResult15Exe | e5sliceEKNSt6size_tEKNSt6size_tE) |
| cutionResultE16CountsDictionary), | -   [cudaq::quantum_platform (C++ |
|     [\[1\]](api/lan               |     cl                            |
| guages/cpp_api.html#_CPPv4N5cudaq | ass)](api/languages/cpp_api.html# |
| 15ExecutionResult15ExecutionResul | _CPPv4N5cudaq16quantum_platformE) |
| tE16CountsDictionaryNSt6stringE), | -   [cudaq:                       |
|     [\[2\                         | :quantum_platform::beginExecution |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     function)](api/languag        |
| utionResultE16CountsDictionaryd), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|                                   | antum_platform14beginExecutionEv) |
|    [\[3\]](api/languages/cpp_api. | -   [cudaq::quantum_pl            |
| html#_CPPv4N5cudaq15ExecutionResu | atform::configureExecutionContext |
| lt15ExecutionResultENSt6stringE), |     (C++                          |
|     [\[4\                         |     function)](api/lang           |
| ]](api/languages/cpp_api.html#_CP | uages/cpp_api.html#_CPPv4NK5cudaq |
| Pv4N5cudaq15ExecutionResult15Exec | 16quantum_platform25configureExec |
| utionResultERK15ExecutionResult), | utionContextER16ExecutionContext) |
|     [\[5\]](api/language          | -   [cuda                         |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | q::quantum_platform::connectivity |
| cutionResult15ExecutionResultEd), |     (C++                          |
|     [\[6\]](api/languag           |     function)](api/langu          |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | ages/cpp_api.html#_CPPv4N5cudaq16 |
| ecutionResult15ExecutionResultEv) | quantum_platform12connectivityEv) |
| -   [                             | -   [cuda                         |
| cudaq::ExecutionResult::operator= | q::quantum_platform::endExecution |
|     (C++                          |     (C++                          |
|     function)](api/languages/     |     function)](api/langu          |
| cpp_api.html#_CPPv4N5cudaq15Execu | ages/cpp_api.html#_CPPv4N5cudaq16 |
| tionResultaSERK15ExecutionResult) | quantum_platform12endExecutionEv) |
| -   [c                            | -   [cudaq::q                     |
| udaq::ExecutionResult::operator== | uantum_platform::enqueueAsyncTask |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     function)](api/languages/     |
| pp_api.html#_CPPv4NK5cudaq15Execu | cpp_api.html#_CPPv4N5cudaq16quant |
| tionResulteqERK15ExecutionResult) | um_platform16enqueueAsyncTaskEKNS |
| -   [cud                          | t6size_tER19KernelExecutionTask), |
| aq::ExecutionResult::registerName |     [\[1\]](api/languag           |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     member)](api/lan              | antum_platform16enqueueAsyncTaskE |
| guages/cpp_api.html#_CPPv4N5cudaq | KNSt6size_tERNSt8functionIFvvEEE) |
| 15ExecutionResult12registerNameE) | -   [cudaq::quantum_p             |
| -   [cudaq                        | latform::finalizeExecutionContext |
| ::ExecutionResult::sequentialData |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     member)](api/langu            | pp_api.html#_CPPv4NK5cudaq16quant |
| ages/cpp_api.html#_CPPv4N5cudaq15 | um_platform24finalizeExecutionCon |
| ExecutionResult14sequentialDataE) | textERN5cudaq16ExecutionContextE) |
| -   [                             | -   [cudaq::qua                   |
| cudaq::ExecutionResult::serialize | ntum_platform::get_codegen_config |
|     (C++                          |     (C++                          |
|     function)](api/l              |     function)](api/languages/c    |
| anguages/cpp_api.html#_CPPv4NK5cu | pp_api.html#_CPPv4N5cudaq16quantu |
| daq15ExecutionResult9serializeEv) | m_platform18get_codegen_configEv) |
| -   [cudaq::fermion_handler (C++  | -   [cuda                         |
|     c                             | q::quantum_platform::get_exec_ctx |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15fermion_handlerE) |     function)](api/langua         |
| -   [cudaq::fermion_op (C++       | ges/cpp_api.html#_CPPv4NK5cudaq16 |
|     type)](api/languages/cpp_api  | quantum_platform12get_exec_ctxEv) |
| .html#_CPPv4N5cudaq10fermion_opE) | -   [c                            |
| -   [cudaq::fermion_op_term (C++  | udaq::quantum_platform::get_noise |
|                                   |     (C++                          |
| type)](api/languages/cpp_api.html |     function)](api/languages/c    |
| #_CPPv4N5cudaq15fermion_op_termE) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [cudaq::FermioniqBaseQPU (C++ | m_platform9get_noiseENSt6size_tE) |
|     cl                            | -   [cudaq:                       |
| ass)](api/languages/cpp_api.html# | :quantum_platform::get_num_qubits |
| _CPPv4N5cudaq16FermioniqBaseQPUE) |     (C++                          |
| -   [cudaq::get_state (C++        |                                   |
|                                   | function)](api/languages/cpp_api. |
|    function)](api/languages/cpp_a | html#_CPPv4NK5cudaq16quantum_plat |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | form14get_num_qubitsENSt6size_tE) |
| ateEDaRR13QuantumKernelDpRR4Args) | -   [cudaq::quantum_              |
| -   [cudaq::gradient (C++         | platform::get_remote_capabilities |
|     class)](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4N5cudaq8gradientE) |     function)                     |
| -   [cudaq::gradient::clone (C++  | ](api/languages/cpp_api.html#_CPP |
|     fun                           | v4NK5cudaq16quantum_platform23get |
| ction)](api/languages/cpp_api.htm | _remote_capabilitiesENSt6size_tE) |
| l#_CPPv4N5cudaq8gradient5cloneEv) | -   [cudaq::qua                   |
| -   [cudaq::gradient::compute     | ntum_platform::get_runtime_target |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api/languages/cp   |
| s/cpp_api.html#_CPPv4N5cudaq8grad | p_api.html#_CPPv4NK5cudaq16quantu |
| ient7computeERKNSt6vectorIdEERKNS | m_platform18get_runtime_targetEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cuda                         |
|     [\[1\]](ap                    | q::quantum_platform::getLogStream |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq8gradient7computeERKNSt6vect |     function)](api/langu          |
| orIdEERNSt6vectorIdEERK7spin_opd) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [cudaq::gradient::gradient    | quantum_platform12getLogStreamEv) |
|     (C++                          | -   [cud                          |
|     function)](api/lang           | aq::quantum_platform::is_emulated |
| uages/cpp_api.html#_CPPv4I00EN5cu |     (C++                          |
| daq8gradient8gradientER7KernelT), |                                   |
|                                   |    function)](api/languages/cpp_a |
|    [\[1\]](api/languages/cpp_api. | pi.html#_CPPv4NK5cudaq16quantum_p |
| html#_CPPv4I00EN5cudaq8gradient8g | latform11is_emulatedENSt6size_tE) |
| radientER7KernelTRR10ArgsMapper), | -   [c                            |
|     [\[2\                         | udaq::quantum_platform::is_remote |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4I00EN5cudaq8gradient8gradientE |     function)](api/languages/cp   |
| RR13QuantumKernelRR10ArgsMapper), | p_api.html#_CPPv4NK5cudaq16quantu |
|     [\[3                          | m_platform9is_remoteENSt6size_tE) |
| \]](api/languages/cpp_api.html#_C | -   [cuda                         |
| PPv4N5cudaq8gradient8gradientERRN | q::quantum_platform::is_simulator |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[                           |                                   |
| 4\]](api/languages/cpp_api.html#_ |   function)](api/languages/cpp_ap |
| CPPv4N5cudaq8gradient8gradientEv) | i.html#_CPPv4NK5cudaq16quantum_pl |
| -   [cudaq::gradient::setArgs     | atform12is_simulatorENSt6size_tE) |
|     (C++                          | -   [c                            |
|     fu                            | udaq::quantum_platform::launchVQE |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     function)](                   |
| tArgsEvR13QuantumKernelDpRR4Args) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::gradient::setKernel   | N5cudaq16quantum_platform9launchV |
|     (C++                          | QEEKNSt6stringEPKvPN5cudaq8gradie |
|     function)](api/languages/c    | ntERKN5cudaq7spin_opERN5cudaq9opt |
| pp_api.html#_CPPv4I0EN5cudaq8grad | imizerEKiKNSt6size_tENSt6size_tE) |
| ient9setKernelEvR13QuantumKernel) | -   [cudaq:                       |
| -   [cud                          | :quantum_platform::list_platforms |
| aq::gradients::central_difference |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     class)](api/la                | es/cpp_api.html#_CPPv4N5cudaq16qu |
| nguages/cpp_api.html#_CPPv4N5cuda | antum_platform14list_platformsEv) |
| q9gradients18central_differenceE) | -                                 |
| -   [cudaq::gra                   |    [cudaq::quantum_platform::name |
| dients::central_difference::clone |     (C++                          |
|     (C++                          |     function)](a                  |
|     function)](api/languages      | pi/languages/cpp_api.html#_CPPv4N |
| /cpp_api.html#_CPPv4N5cudaq9gradi | K5cudaq16quantum_platform4nameEv) |
| ents18central_difference5cloneEv) | -   [                             |
| -   [cudaq::gradi                 | cudaq::quantum_platform::num_qpus |
| ents::central_difference::compute |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)](                   | anguages/cpp_api.html#_CPPv4NK5cu |
| api/languages/cpp_api.html#_CPPv4 | daq16quantum_platform8num_qpusEv) |
| N5cudaq9gradients18central_differ | -   [cudaq::                      |
| ence7computeERKNSt6vectorIdEERKNS | quantum_platform::onRandomSeedSet |
| t8functionIFdNSt6vectorIdEEEEEd), |     (C++                          |
|                                   |                                   |
|   [\[1\]](api/languages/cpp_api.h | function)](api/languages/cpp_api. |
| tml#_CPPv4N5cudaq9gradients18cent | html#_CPPv4N5cudaq16quantum_platf |
| ral_difference7computeERKNSt6vect | orm15onRandomSeedSetENSt6size_tE) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq:                       |
| -   [cudaq::gradie                | :quantum_platform::reset_exec_ctx |
| nts::central_difference::gradient |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     functio                       | es/cpp_api.html#_CPPv4N5cudaq16qu |
| n)](api/languages/cpp_api.html#_C | antum_platform14reset_exec_ctxEv) |
| PPv4I00EN5cudaq9gradients18centra | -   [cud                          |
| l_difference8gradientER7KernelT), | aq::quantum_platform::reset_noise |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     function)](api/languages/cpp_ |
| q9gradients18central_difference8g | api.html#_CPPv4N5cudaq16quantum_p |
| radientER7KernelTRR10ArgsMapper), | latform11reset_noiseENSt6size_tE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq:                       |
| api.html#_CPPv4I00EN5cudaq9gradie | :quantum_platform::resetLogStream |
| nts18central_difference8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |     function)](api/languag        |
|     [\[3\]](api/languages/cpp     | es/cpp_api.html#_CPPv4N5cudaq16qu |
| _api.html#_CPPv4N5cudaq9gradients | antum_platform14resetLogStreamEv) |
| 18central_difference8gradientERRN | -   [cuda                         |
| St8functionIFvNSt6vectorIdEEEEE), | q::quantum_platform::set_exec_ctx |
|     [\[4\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq9gradient |     funct                         |
| s18central_difference8gradientEv) | ion)](api/languages/cpp_api.html# |
| -   [cud                          | _CPPv4N5cudaq16quantum_platform12 |
| aq::gradients::forward_difference | set_exec_ctxEP16ExecutionContext) |
|     (C++                          | -   [c                            |
|     class)](api/la                | udaq::quantum_platform::set_noise |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9gradients18forward_differenceE) |     function                      |
| -   [cudaq::gra                   | )](api/languages/cpp_api.html#_CP |
| dients::forward_difference::clone | Pv4N5cudaq16quantum_platform9set_ |
|     (C++                          | noiseEPK11noise_modelNSt6size_tE) |
|     function)](api/languages      | -   [cuda                         |
| /cpp_api.html#_CPPv4N5cudaq9gradi | q::quantum_platform::setLogStream |
| ents18forward_difference5cloneEv) |     (C++                          |
| -   [cudaq::gradi                 |                                   |
| ents::forward_difference::compute |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq16quantum_plat |
|     function)](                   | form12setLogStreamERNSt7ostreamE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::quantum_platfor       |
| N5cudaq9gradients18forward_differ | m::supports_explicit_measurements |
| ence7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)](api/l              |
|                                   | anguages/cpp_api.html#_CPPv4NK5cu |
|   [\[1\]](api/languages/cpp_api.h | daq16quantum_platform30supports_e |
| tml#_CPPv4N5cudaq9gradients18forw | xplicit_measurementsENSt6size_tE) |
| ard_difference7computeERKNSt6vect | -   [cudaq::quantum_pla           |
| orIdEERNSt6vectorIdEERK7spin_opd) | tform::supports_task_distribution |
| -   [cudaq::gradie                |     (C++                          |
| nts::forward_difference::gradient |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     functio                       | ml#_CPPv4NK5cudaq16quantum_platfo |
| n)](api/languages/cpp_api.html#_C | rm26supports_task_distributionEv) |
| PPv4I00EN5cudaq9gradients18forwar | -   [cudaq::quantum               |
| d_difference8gradientER7KernelT), | _platform::with_execution_context |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     function)                     |
| q9gradients18forward_difference8g | ](api/languages/cpp_api.html#_CPP |
| radientER7KernelTRR10ArgsMapper), | v4I0DpEN5cudaq16quantum_platform2 |
|     [\[2\]](api/languages/cpp_    | 2with_execution_contextEDaR16Exec |
| api.html#_CPPv4I00EN5cudaq9gradie | utionContextRR8CallableDpRR4Args) |
| nts18forward_difference8gradientE | -   [cudaq::QuantumTask (C++      |
| RR13QuantumKernelRR10ArgsMapper), |     type)](api/languages/cpp_api. |
|     [\[3\]](api/languages/cpp     | html#_CPPv4N5cudaq11QuantumTaskE) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::qubit (C++            |
| 18forward_difference8gradientERRN |     type)](api/languages/c        |
| St8functionIFvNSt6vectorIdEEEEE), | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     [\[4\]](api/languages/cp      | -   [cudaq::QubitConnectivity     |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18forward_difference8gradientEv) |     ty                            |
| -   [                             | pe)](api/languages/cpp_api.html#_ |
| cudaq::gradients::parameter_shift | CPPv4N5cudaq17QubitConnectivityE) |
|     (C++                          | -   [cudaq::QubitEdge (C++        |
|     class)](api                   |     type)](api/languages/cpp_a    |
| /languages/cpp_api.html#_CPPv4N5c | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| udaq9gradients15parameter_shiftE) | -   [cudaq::qudit (C++            |
| -   [cudaq::                      |     clas                          |
| gradients::parameter_shift::clone | s)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     function)](api/langua         | -   [cudaq::qudit::qudit (C++     |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |                                   |
| adients15parameter_shift5cloneEv) | function)](api/languages/cpp_api. |
| -   [cudaq::gr                    | html#_CPPv4N5cudaq5qudit5quditEv) |
| adients::parameter_shift::compute | -   [cudaq::qvector (C++          |
|     (C++                          |     class)                        |
|     function                      | ](api/languages/cpp_api.html#_CPP |
| )](api/languages/cpp_api.html#_CP | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| Pv4N5cudaq9gradients15parameter_s | -   [cudaq::qvector::back (C++    |
| hift7computeERKNSt6vectorIdEERKNS |     function)](a                  |
| t8functionIFdNSt6vectorIdEEEEEd), | pi/languages/cpp_api.html#_CPPv4N |
|     [\[1\]](api/languages/cpp_ap  | 5cudaq7qvector4backENSt6size_tE), |
| i.html#_CPPv4N5cudaq9gradients15p |                                   |
| arameter_shift7computeERKNSt6vect |   [\[1\]](api/languages/cpp_api.h |
| orIdEERNSt6vectorIdEERK7spin_opd) | tml#_CPPv4N5cudaq7qvector4backEv) |
| -   [cudaq::gra                   | -   [cudaq::qvector::begin (C++   |
| dients::parameter_shift::gradient |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     func                          | ml#_CPPv4N5cudaq7qvector5beginEv) |
| tion)](api/languages/cpp_api.html | -   [cudaq::qvector::clear (C++   |
| #_CPPv4I00EN5cudaq9gradients15par |     fu                            |
| ameter_shift8gradientER7KernelT), | nction)](api/languages/cpp_api.ht |
|     [\[1\]](api/lan               | ml#_CPPv4N5cudaq7qvector5clearEv) |
| guages/cpp_api.html#_CPPv4I00EN5c | -   [cudaq::qvector::end (C++     |
| udaq9gradients15parameter_shift8g |                                   |
| radientER7KernelTRR10ArgsMapper), | function)](api/languages/cpp_api. |
|     [\[2\]](api/languages/c       | html#_CPPv4N5cudaq7qvector3endEv) |
| pp_api.html#_CPPv4I00EN5cudaq9gra | -   [cudaq::qvector::front (C++   |
| dients15parameter_shift8gradientE |     function)](ap                 |
| RR13QuantumKernelRR10ArgsMapper), | i/languages/cpp_api.html#_CPPv4N5 |
|     [\[3\]](api/languages/        | cudaq7qvector5frontENSt6size_tE), |
| cpp_api.html#_CPPv4N5cudaq9gradie |                                   |
| nts15parameter_shift8gradientERRN |  [\[1\]](api/languages/cpp_api.ht |
| St8functionIFvNSt6vectorIdEEEEE), | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     [\[4\]](api/languages         | -   [cudaq::qvector::operator=    |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents15parameter_shift8gradientEv) |     functio                       |
| -   [cudaq::kernel_builder (C++   | n)](api/languages/cpp_api.html#_C |
|     clas                          | PPv4N5cudaq7qvectoraSERK7qvector) |
| s)](api/languages/cpp_api.html#_C | -   [cudaq::qvector::operator\[\] |
| PPv4IDpEN5cudaq14kernel_builderE) |     (C++                          |
| -   [c                            |     function)                     |
| udaq::kernel_builder::constantVal | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq7qvectorixEKNSt6size_tE) |
|     function)](api/la             | -   [cudaq::qvector::qvector (C++ |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/               |
| q14kernel_builder11constantValEd) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cu                           | daq7qvector7qvectorENSt6size_tE), |
| daq::kernel_builder::getArguments |     [\[1\]](a                     |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function)](api/lan            | 5cudaq7qvector7qvectorERK5state), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[2\]](api                   |
| 14kernel_builder12getArgumentsEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cu                           | udaq7qvector7qvectorERK7qvector), |
| daq::kernel_builder::getNumParams |     [\[3\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4N5cudaq7qvector7q |
|     function)](api/lan            | vectorERKNSt6vectorI7complexEEb), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[4\]](ap                    |
| 14kernel_builder12getNumParamsEv) | i/languages/cpp_api.html#_CPPv4N5 |
| -   [c                            | cudaq7qvector7qvectorERR7qvector) |
| udaq::kernel_builder::isArgStdVec | -   [cudaq::qvector::size (C++    |
|     (C++                          |     fu                            |
|     function)](api/languages/cp   | nction)](api/languages/cpp_api.ht |
| p_api.html#_CPPv4N5cudaq14kernel_ | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| builder11isArgStdVecENSt6size_tE) | -   [cudaq::qvector::slice (C++   |
| -   [cuda                         |     function)](api/language       |
| q::kernel_builder::kernel_builder | s/cpp_api.html#_CPPv4N5cudaq7qvec |
|     (C++                          | tor5sliceENSt6size_tENSt6size_tE) |
|     function)](api/languages/cpp_ | -   [cudaq::qvector::value_type   |
| api.html#_CPPv4N5cudaq14kernel_bu |     (C++                          |
| ilder14kernel_builderERNSt6vector |     typ                           |
| IN7details17KernelBuilderTypeEEE) | e)](api/languages/cpp_api.html#_C |
| -   [cudaq::kernel_builder::name  | PPv4N5cudaq7qvector10value_typeE) |
|     (C++                          | -   [cudaq::qview (C++            |
|     function)                     |     clas                          |
| ](api/languages/cpp_api.html#_CPP | s)](api/languages/cpp_api.html#_C |
| v4N5cudaq14kernel_builder4nameEv) | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| -                                 | -   [cudaq::qview::back (C++      |
|    [cudaq::kernel_builder::qalloc |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/language       | v4N5cudaq5qview4backENSt6size_tE) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cudaq::qview::begin (C++     |
| nel_builder6qallocE10QuakeValue), |                                   |
|     [\[1\]](api/language          | function)](api/languages/cpp_api. |
| s/cpp_api.html#_CPPv4N5cudaq14ker | html#_CPPv4N5cudaq5qview5beginEv) |
| nel_builder6qallocEKNSt6size_tE), | -   [cudaq::qview::end (C++       |
|     [\[2                          |                                   |
| \]](api/languages/cpp_api.html#_C |   function)](api/languages/cpp_ap |
| PPv4N5cudaq14kernel_builder6qallo | i.html#_CPPv4N5cudaq5qview3endEv) |
| cERNSt6vectorINSt7complexIdEEEE), | -   [cudaq::qview::front (C++     |
|     [\[3\]](                      |     function)](                   |
| api/languages/cpp_api.html#_CPPv4 | api/languages/cpp_api.html#_CPPv4 |
| N5cudaq14kernel_builder6qallocEv) | N5cudaq5qview5frontENSt6size_tE), |
| -   [cudaq::kernel_builder::swap  |                                   |
|     (C++                          |    [\[1\]](api/languages/cpp_api. |
|     function)](api/language       | html#_CPPv4N5cudaq5qview5frontEv) |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | -   [cudaq::qview::operator\[\]   |
| 4kernel_builder4swapEvRK10QuakeVa |     (C++                          |
| lueRK10QuakeValueRK10QuakeValue), |     functio                       |
|                                   | n)](api/languages/cpp_api.html#_C |
| [\[1\]](api/languages/cpp_api.htm | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| l#_CPPv4I00EN5cudaq14kernel_build | -   [cudaq::qview::qview (C++     |
| er4swapEvRKNSt6vectorI10QuakeValu |     functio                       |
| eEERK10QuakeValueRK10QuakeValue), | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4I0EN5cudaq5qview5qviewERR1R), |
| [\[2\]](api/languages/cpp_api.htm |     [\[1                          |
| l#_CPPv4N5cudaq14kernel_builder4s | \]](api/languages/cpp_api.html#_C |
| wapERK10QuakeValueRK10QuakeValue) | PPv4N5cudaq5qview5qviewERK5qview) |
| -   [cudaq::KernelExecutionTask   | -   [cudaq::qview::size (C++      |
|     (C++                          |                                   |
|     type                          | function)](api/languages/cpp_api. |
| )](api/languages/cpp_api.html#_CP | html#_CPPv4NK5cudaq5qview4sizeEv) |
| Pv4N5cudaq19KernelExecutionTaskE) | -   [cudaq::qview::slice (C++     |
| -   [cudaq::KernelThunkResultType |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|     struct)]                      | iew5sliceENSt6size_tENSt6size_tE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::qview::value_type     |
| 4N5cudaq21KernelThunkResultTypeE) |     (C++                          |
| -   [cudaq::KernelThunkType (C++  |     t                             |
|                                   | ype)](api/languages/cpp_api.html# |
| type)](api/languages/cpp_api.html | _CPPv4N5cudaq5qview10value_typeE) |
| #_CPPv4N5cudaq15KernelThunkTypeE) | -   [cudaq::range (C++            |
| -   [cudaq::kraus_channel (C++    |     fun                           |
|                                   | ction)](api/languages/cpp_api.htm |
|  class)](api/languages/cpp_api.ht | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| ml#_CPPv4N5cudaq13kraus_channelE) | orI11ElementTypeEE11ElementType), |
| -   [cudaq::kraus_channel::empty  |     [\[1\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4I0EN5cudaq5rangeEN |
|     function)]                    | St6vectorI11ElementTypeEE11Elemen |
| (api/languages/cpp_api.html#_CPPv | tType11ElementType11ElementType), |
| 4NK5cudaq13kraus_channel5emptyEv) |     [                             |
| -   [cudaq::kraus_c               | \[2\]](api/languages/cpp_api.html |
| hannel::generateUnitaryParameters | #_CPPv4N5cudaq5rangeENSt6size_tE) |
|     (C++                          | -   [cudaq::real (C++             |
|                                   |     type)](api/languages/         |
|    function)](api/languages/cpp_a | cpp_api.html#_CPPv4N5cudaq4realE) |
| pi.html#_CPPv4N5cudaq13kraus_chan | -   [cudaq::registry (C++         |
| nel25generateUnitaryParametersEv) |     type)](api/languages/cpp_     |
| -                                 | api.html#_CPPv4N5cudaq8registryE) |
|    [cudaq::kraus_channel::get_ops | -                                 |
|     (C++                          |  [cudaq::registry::RegisteredType |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     class)](api/                  |
| K5cudaq13kraus_channel7get_opsEv) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cud                          | 5cudaq8registry14RegisteredTypeE) |
| aq::kraus_channel::identity_flags | -   [cudaq::RemoteCapabilities    |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     struc                         |
| guages/cpp_api.html#_CPPv4N5cudaq | t)](api/languages/cpp_api.html#_C |
| 13kraus_channel14identity_flagsE) | PPv4N5cudaq18RemoteCapabilitiesE) |
| -   [cud                          | -   [cudaq::Remo                  |
| aq::kraus_channel::is_identity_op | teCapabilities::isRemoteSimulator |
|     (C++                          |     (C++                          |
|                                   |     member)](api/languages/c      |
|    function)](api/languages/cpp_a | pp_api.html#_CPPv4N5cudaq18Remote |
| pi.html#_CPPv4NK5cudaq13kraus_cha | Capabilities17isRemoteSimulatorE) |
| nnel14is_identity_opENSt6size_tE) | -   [cudaq::Remot                 |
| -   [cudaq::                      | eCapabilities::RemoteCapabilities |
| kraus_channel::is_unitary_mixture |     (C++                          |
|     (C++                          |     function)](api/languages/cpp  |
|     function)](api/languages      | _api.html#_CPPv4N5cudaq18RemoteCa |
| /cpp_api.html#_CPPv4NK5cudaq13kra | pabilities18RemoteCapabilitiesEb) |
| us_channel18is_unitary_mixtureEv) | -   [cudaq:                       |
| -   [cu                           | :RemoteCapabilities::stateOverlap |
| daq::kraus_channel::kraus_channel |     (C++                          |
|     (C++                          |     member)](api/langua           |
|     function)](api/lang           | ges/cpp_api.html#_CPPv4N5cudaq18R |
| uages/cpp_api.html#_CPPv4IDpEN5cu | emoteCapabilities12stateOverlapE) |
| daq13kraus_channel13kraus_channel | -                                 |
| EDpRRNSt16initializer_listI1TEE), |   [cudaq::RemoteCapabilities::vqe |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     member)](                     |
| ml#_CPPv4N5cudaq13kraus_channel13 | api/languages/cpp_api.html#_CPPv4 |
| kraus_channelERK13kraus_channel), | N5cudaq18RemoteCapabilities3vqeE) |
|     [\[2\]                        | -   [cudaq::RemoteSimulationState |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq13kraus_channel13kraus_c |     class)]                       |
| hannelERKNSt6vectorI8kraus_opEE), | (api/languages/cpp_api.html#_CPPv |
|     [\[3\]                        | 4N5cudaq21RemoteSimulationStateE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::Resources (C++        |
| v4N5cudaq13kraus_channel13kraus_c |     class)](api/languages/cpp_a   |
| hannelERRNSt6vectorI8kraus_opEE), | pi.html#_CPPv4N5cudaq9ResourcesE) |
|     [\[4\]](api/lan               | -   [cudaq::run (C++              |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)]                    |
| 13kraus_channel13kraus_channelEv) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| [cudaq::kraus_channel::noise_type | 5invoke_result_tINSt7decay_tI13Qu |
|     (C++                          | antumKernelEEDpNSt7decay_tI4ARGSE |
|     member)](api                  | EEEEENSt6size_tERN5cudaq11noise_m |
| /languages/cpp_api.html#_CPPv4N5c | odelERR13QuantumKernelDpRR4ARGS), |
| udaq13kraus_channel10noise_typeE) |     [\[1\]](api/langu             |
| -                                 | ages/cpp_api.html#_CPPv4I0DpEN5cu |
|   [cudaq::kraus_channel::op_names | daq3runENSt6vectorINSt15invoke_re |
|     (C++                          | sult_tINSt7decay_tI13QuantumKerne |
|     member)](                     | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| api/languages/cpp_api.html#_CPPv4 | ize_tERR13QuantumKernelDpRR4ARGS) |
| N5cudaq13kraus_channel8op_namesE) | -   [cudaq::run_async (C++        |
| -                                 |     functio                       |
|  [cudaq::kraus_channel::operator= | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     function)](api/langua         | tureINSt6vectorINSt15invoke_resul |
| ges/cpp_api.html#_CPPv4N5cudaq13k | t_tINSt7decay_tI13QuantumKernelEE |
| raus_channelaSERK13kraus_channel) | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| -   [c                            | ze_tENSt6size_tERN5cudaq11noise_m |
| udaq::kraus_channel::operator\[\] | odelERR13QuantumKernelDpRR4ARGS), |
|     (C++                          |     [\[1\]](api/la                |
|     function)](api/l              | nguages/cpp_api.html#_CPPv4I0DpEN |
| anguages/cpp_api.html#_CPPv4N5cud | 5cudaq9run_asyncENSt6futureINSt6v |
| aq13kraus_channelixEKNSt6size_tE) | ectorINSt15invoke_result_tINSt7de |
| -                                 | cay_tI13QuantumKernelEEDpNSt7deca |
| [cudaq::kraus_channel::parameters | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4ARGS) |
|     member)](api                  | -   [cudaq::RuntimeTarget (C++    |
| /languages/cpp_api.html#_CPPv4N5c |                                   |
| udaq13kraus_channel10parametersE) | struct)](api/languages/cpp_api.ht |
| -   [cudaq::krau                  | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| s_channel::populateDefaultOpNames | -   [cudaq::sample (C++           |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/languages/cp   | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| p_api.html#_CPPv4N5cudaq13kraus_c | mpleE13sample_resultRK14sample_op |
| hannel22populateDefaultOpNamesEv) | tionsRR13QuantumKernelDpRR4Args), |
| -   [cu                           |     [\[1\                         |
| daq::kraus_channel::probabilities | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     member)](api/la               | esultRR13QuantumKernelDpRR4Args), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\                            |
| q13kraus_channel13probabilitiesE) | [2\]](api/languages/cpp_api.html# |
| -                                 | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|  [cudaq::kraus_channel::push_back | ize_tERR13QuantumKernelDpRR4Args) |
|     (C++                          | -   [cudaq::sample_options (C++   |
|     function)](api                |     s                             |
| /languages/cpp_api.html#_CPPv4N5c | truct)](api/languages/cpp_api.htm |
| udaq13kraus_channel9push_backE8kr | l#_CPPv4N5cudaq14sample_optionsE) |
| aus_opNSt8optionalINSt6stringEEE) | -   [cudaq::sample_result (C++    |
| -   [cudaq::kraus_channel::size   |                                   |
|     (C++                          |  class)](api/languages/cpp_api.ht |
|     function)                     | ml#_CPPv4N5cudaq13sample_resultE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::sample_result::append |
| v4NK5cudaq13kraus_channel4sizeEv) |     (C++                          |
| -   [                             |     function)](api/languages/cpp_ |
| cudaq::kraus_channel::unitary_ops | api.html#_CPPv4N5cudaq13sample_re |
|     (C++                          | sult6appendERK15ExecutionResultb) |
|     member)](api/                 | -   [cudaq::sample_result::begin  |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq13kraus_channel11unitary_opsE) |     function)]                    |
| -   [cudaq::kraus_op (C++         | (api/languages/cpp_api.html#_CPPv |
|     struct)](api/languages/cpp_   | 4N5cudaq13sample_result5beginEv), |
| api.html#_CPPv4N5cudaq8kraus_opE) |     [\[1\]]                       |
| -   [cudaq::kraus_op::adjoint     | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NK5cudaq13sample_result5beginEv) |
|     functi                        | -   [cudaq::sample_result::cbegin |
| on)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4NK5cudaq8kraus_op7adjointEv) |     function)](                   |
| -   [cudaq::kraus_op::data (C++   | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq13sample_result6cbeginEv) |
|  member)](api/languages/cpp_api.h | -   [cudaq::sample_result::cend   |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     (C++                          |
| -   [cudaq::kraus_op::kraus_op    |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     func                          | v4NK5cudaq13sample_result4cendEv) |
| tion)](api/languages/cpp_api.html | -   [cudaq::sample_result::clear  |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |     (C++                          |
| opERRNSt16initializer_listI1TEE), |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
|  [\[1\]](api/languages/cpp_api.ht | v4N5cudaq13sample_result5clearEv) |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | -   [cudaq::sample_result::count  |
| pENSt6vectorIN5cudaq7complexEEE), |     (C++                          |
|     [\[2\]](api/l                 |     function)](                   |
| anguages/cpp_api.html#_CPPv4N5cud | api/languages/cpp_api.html#_CPPv4 |
| aq8kraus_op8kraus_opERK8kraus_op) | NK5cudaq13sample_result5countENSt |
| -   [cudaq::kraus_op::nCols (C++  | 11string_viewEKNSt11string_viewE) |
|                                   | -   [                             |
| member)](api/languages/cpp_api.ht | cudaq::sample_result::deserialize |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) |     (C++                          |
| -   [cudaq::kraus_op::nRows (C++  |     functio                       |
|                                   | n)](api/languages/cpp_api.html#_C |
| member)](api/languages/cpp_api.ht | PPv4N5cudaq13sample_result11deser |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | ializeERNSt6vectorINSt6size_tEEE) |
| -   [cudaq::kraus_op::operator=   | -   [cudaq::sample_result::dump   |
|     (C++                          |     (C++                          |
|     function)                     |     function)](api/languag        |
| ](api/languages/cpp_api.html#_CPP | es/cpp_api.html#_CPPv4NK5cudaq13s |
| v4N5cudaq8kraus_opaSERK8kraus_op) | ample_result4dumpERNSt7ostreamE), |
| -   [cudaq::kraus_op::precision   |     [\[1\]                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     memb                          | v4NK5cudaq13sample_result4dumpEv) |
| er)](api/languages/cpp_api.html#_ | -   [cudaq::sample_result::end    |
| CPPv4N5cudaq8kraus_op9precisionE) |     (C++                          |
| -   [cudaq::matrix_callback (C++  |     function                      |
|     c                             | )](api/languages/cpp_api.html#_CP |
| lass)](api/languages/cpp_api.html | Pv4N5cudaq13sample_result3endEv), |
| #_CPPv4N5cudaq15matrix_callbackE) |     [\[1\                         |
| -   [cudaq::matrix_handler (C++   | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq13sample_result3endEv) |
| class)](api/languages/cpp_api.htm | -   [                             |
| l#_CPPv4N5cudaq14matrix_handlerE) | cudaq::sample_result::expectation |
| -   [cudaq::mat                   |     (C++                          |
| rix_handler::commutation_behavior |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     struct)](api/languages/       | tml#_CPPv4NK5cudaq13sample_result |
| cpp_api.html#_CPPv4N5cudaq14matri | 11expectationEKNSt11string_viewE) |
| x_handler20commutation_behaviorE) | -   [c                            |
| -                                 | udaq::sample_result::get_marginal |
|    [cudaq::matrix_handler::define |     (C++                          |
|     (C++                          |     function)](api/languages/cpp_ |
|     function)](a                  | api.html#_CPPv4NK5cudaq13sample_r |
| pi/languages/cpp_api.html#_CPPv4N | esult12get_marginalERKNSt6vectorI |
| 5cudaq14matrix_handler6defineENSt | NSt6size_tEEEKNSt11string_viewE), |
| 6stringENSt6vectorINSt7int64_tEEE |     [\[1\]](api/languages/cpp_    |
| RR15matrix_callbackRKNSt13unorder | api.html#_CPPv4NK5cudaq13sample_r |
| ed_mapINSt6stringENSt6stringEEE), | esult12get_marginalERRKNSt6vector |
|                                   | INSt6size_tEEEKNSt11string_viewE) |
| [\[1\]](api/languages/cpp_api.htm | -   [cuda                         |
| l#_CPPv4N5cudaq14matrix_handler6d | q::sample_result::get_total_shots |
| efineENSt6stringENSt6vectorINSt7i |     (C++                          |
| nt64_tEEERR15matrix_callbackRR20d |     function)](api/langua         |
| iag_matrix_callbackRKNSt13unorder | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| ed_mapINSt6stringENSt6stringEEE), | sample_result15get_total_shotsEv) |
|     [\[2\]](                      | -   [cuda                         |
| api/languages/cpp_api.html#_CPPv4 | q::sample_result::has_even_parity |
| N5cudaq14matrix_handler6defineENS |     (C++                          |
| t6stringENSt6vectorINSt7int64_tEE |     fun                           |
| ERR15matrix_callbackRRNSt13unorde | ction)](api/languages/cpp_api.htm |
| red_mapINSt6stringENSt6stringEEE) | l#_CPPv4N5cudaq13sample_result15h |
| -                                 | as_even_parityENSt11string_viewE) |
|   [cudaq::matrix_handler::degrees | -   [cuda                         |
|     (C++                          | q::sample_result::has_expectation |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4NK |     funct                         |
| 5cudaq14matrix_handler7degreesEv) | ion)](api/languages/cpp_api.html# |
| -                                 | _CPPv4NK5cudaq13sample_result15ha |
|  [cudaq::matrix_handler::displace | s_expectationEKNSt11string_viewE) |
|     (C++                          | -   [cu                           |
|     function)](api/language       | daq::sample_result::most_probable |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handler8displaceENSt6size_tE) |     fun                           |
| -   [cudaq::matrix                | ction)](api/languages/cpp_api.htm |
| _handler::get_expected_dimensions | l#_CPPv4NK5cudaq13sample_result13 |
|     (C++                          | most_probableEKNSt11string_viewE) |
|                                   | -                                 |
|    function)](api/languages/cpp_a | [cudaq::sample_result::operator+= |
| pi.html#_CPPv4NK5cudaq14matrix_ha |     (C++                          |
| ndler23get_expected_dimensionsEv) |     function)](api/langua         |
| -   [cudaq::matrix_ha             | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ndler::get_parameter_descriptions | ample_resultpLERK13sample_result) |
|     (C++                          | -                                 |
|                                   |  [cudaq::sample_result::operator= |
| function)](api/languages/cpp_api. |     (C++                          |
| html#_CPPv4NK5cudaq14matrix_handl |     function)](api/langua         |
| er26get_parameter_descriptionsEv) | ges/cpp_api.html#_CPPv4N5cudaq13s |
| -   [c                            | ample_resultaSERR13sample_result) |
| udaq::matrix_handler::instantiate | -                                 |
|     (C++                          | [cudaq::sample_result::operator== |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function)](api/languag        |
| 5cudaq14matrix_handler11instantia | es/cpp_api.html#_CPPv4NK5cudaq13s |
| teENSt6stringERKNSt6vectorINSt6si | ample_resulteqERK13sample_result) |
| ze_tEEERK20commutation_behavior), | -   [                             |
|     [\[1\]](                      | cudaq::sample_result::probability |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14matrix_handler11instanti |     function)](api/lan            |
| ateENSt6stringERRNSt6vectorINSt6s | guages/cpp_api.html#_CPPv4NK5cuda |
| ize_tEEERK20commutation_behavior) | q13sample_result11probabilityENSt |
| -   [cuda                         | 11string_viewEKNSt11string_viewE) |
| q::matrix_handler::matrix_handler | -   [cud                          |
|     (C++                          | aq::sample_result::register_names |
|     function)](api/languag        |     (C++                          |
| es/cpp_api.html#_CPPv4I0_NSt11ena |     function)](api/langu          |
| ble_if_tINSt12is_base_of_vI16oper | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| ator_handler1TEEbEEEN5cudaq14matr | 3sample_result14register_namesEv) |
| ix_handler14matrix_handlerERK1T), | -                                 |
|     [\[1\]](ap                    |    [cudaq::sample_result::reorder |
| i/languages/cpp_api.html#_CPPv4I0 |     (C++                          |
| _NSt11enable_if_tINSt12is_base_of |     function)](api/langua         |
| _vI16operator_handler1TEEbEEEN5cu | ges/cpp_api.html#_CPPv4N5cudaq13s |
| daq14matrix_handler14matrix_handl | ample_result7reorderERKNSt6vector |
| erERK1TRK20commutation_behavior), | INSt6size_tEEEKNSt11string_viewE) |
|     [\[2\]](api/languages/cpp_ap  | -   [cu                           |
| i.html#_CPPv4N5cudaq14matrix_hand | daq::sample_result::sample_result |
| ler14matrix_handlerENSt6size_tE), |     (C++                          |
|     [\[3\]](api/                  |     func                          |
| languages/cpp_api.html#_CPPv4N5cu | tion)](api/languages/cpp_api.html |
| daq14matrix_handler14matrix_handl | #_CPPv4N5cudaq13sample_result13sa |
| erENSt6stringERKNSt6vectorINSt6si | mple_resultERK15ExecutionResult), |
| ze_tEEERK20commutation_behavior), |     [\[1\]](api/la                |
|     [\[4\]](api/                  | nguages/cpp_api.html#_CPPv4N5cuda |
| languages/cpp_api.html#_CPPv4N5cu | q13sample_result13sample_resultER |
| daq14matrix_handler14matrix_handl | KNSt6vectorI15ExecutionResultEE), |
| erENSt6stringERRNSt6vectorINSt6si |                                   |
| ze_tEEERK20commutation_behavior), |  [\[2\]](api/languages/cpp_api.ht |
|     [\                            | ml#_CPPv4N5cudaq13sample_result13 |
| [5\]](api/languages/cpp_api.html# | sample_resultERR13sample_result), |
| _CPPv4N5cudaq14matrix_handler14ma |     [                             |
| trix_handlerERK14matrix_handler), | \[3\]](api/languages/cpp_api.html |
|     [                             | #_CPPv4N5cudaq13sample_result13sa |
| \[6\]](api/languages/cpp_api.html | mple_resultERR15ExecutionResult), |
| #_CPPv4N5cudaq14matrix_handler14m |     [\[4\]](api/lan               |
| atrix_handlerERR14matrix_handler) | guages/cpp_api.html#_CPPv4N5cudaq |
| -                                 | 13sample_result13sample_resultEdR |
|  [cudaq::matrix_handler::momentum | KNSt6vectorI15ExecutionResultEE), |
|     (C++                          |     [\[5\]](api/lan               |
|     function)](api/language       | guages/cpp_api.html#_CPPv4N5cudaq |
| s/cpp_api.html#_CPPv4N5cudaq14mat | 13sample_result13sample_resultEv) |
| rix_handler8momentumENSt6size_tE) | -                                 |
| -                                 |  [cudaq::sample_result::serialize |
|    [cudaq::matrix_handler::number |     (C++                          |
|     (C++                          |     function)](api                |
|     function)](api/langua         | /languages/cpp_api.html#_CPPv4NK5 |
| ges/cpp_api.html#_CPPv4N5cudaq14m | cudaq13sample_result9serializeEv) |
| atrix_handler6numberENSt6size_tE) | -   [cudaq::sample_result::size   |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::operator= |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4NK5cudaq13sampl |
|     fun                           | e_result4sizeEKNSt11string_viewE) |
| ction)](api/languages/cpp_api.htm | -   [cudaq::sample_result::to_map |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     (C++                          |
| NSt7is_sameI1T14matrix_handlerE5v |     function)](api/languages/cpp  |
| alueENSt12is_base_of_vI16operator | _api.html#_CPPv4NK5cudaq13sample_ |
| _handler1TEEEbEEEN5cudaq14matrix_ | result6to_mapEKNSt11string_viewE) |
| handleraSER14matrix_handlerRK1T), | -   [cuda                         |
|     [\[1\]](api/languages         | q::sample_result::\~sample_result |
| /cpp_api.html#_CPPv4N5cudaq14matr |     (C++                          |
| ix_handleraSERK14matrix_handler), |     funct                         |
|     [\[2\]](api/language          | ion)](api/languages/cpp_api.html# |
| s/cpp_api.html#_CPPv4N5cudaq14mat | _CPPv4N5cudaq13sample_resultD0Ev) |
| rix_handleraSERR14matrix_handler) | -   [cudaq::scalar_callback (C++  |
| -   [                             |     c                             |
| cudaq::matrix_handler::operator== | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_callbackE) |
|     function)](api/languages      | -   [c                            |
| /cpp_api.html#_CPPv4NK5cudaq14mat | udaq::scalar_callback::operator() |
| rix_handlereqERK14matrix_handler) |     (C++                          |
| -                                 |     function)](api/language       |
|    [cudaq::matrix_handler::parity | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|     (C++                          | alar_callbackclERKNSt13unordered_ |
|     function)](api/langua         | mapINSt6stringENSt7complexIdEEEE) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -   [                             |
| atrix_handler6parityENSt6size_tE) | cudaq::scalar_callback::operator= |
| -                                 |     (C++                          |
|  [cudaq::matrix_handler::position |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function)](api/language       | _callbackaSERK15scalar_callback), |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     [\[1\]](api/languages/        |
| rix_handler8positionENSt6size_tE) | cpp_api.html#_CPPv4N5cudaq15scala |
| -   [cudaq::                      | r_callbackaSERR15scalar_callback) |
| matrix_handler::remove_definition | -   [cudaq:                       |
|     (C++                          | :scalar_callback::scalar_callback |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api/languag        |
| ml#_CPPv4N5cudaq14matrix_handler1 | es/cpp_api.html#_CPPv4I0_NSt11ena |
| 7remove_definitionERKNSt6stringE) | ble_if_tINSt16is_invocable_r_vINS |
| -                                 | t7complexIdEE8CallableRKNSt13unor |
|   [cudaq::matrix_handler::squeeze | dered_mapINSt6stringENSt7complexI |
|     (C++                          | dEEEEEEbEEEN5cudaq15scalar_callba |
|     function)](api/languag        | ck15scalar_callbackERR8Callable), |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     [\[1\                         |
| trix_handler7squeezeENSt6size_tE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::m                     | Pv4N5cudaq15scalar_callback15scal |
| atrix_handler::to_diagonal_matrix | ar_callbackERK15scalar_callback), |
|     (C++                          |     [\[2                          |
|     function)](api/lang           | \]](api/languages/cpp_api.html#_C |
| uages/cpp_api.html#_CPPv4NK5cudaq | PPv4N5cudaq15scalar_callback15sca |
| 14matrix_handler18to_diagonal_mat | lar_callbackERR15scalar_callback) |
| rixERNSt13unordered_mapINSt6size_ | -   [cudaq::scalar_operator (C++  |
| tENSt7int64_tEEERKNSt13unordered_ |     c                             |
| mapINSt6stringENSt7complexIdEEEE) | lass)](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq15scalar_operatorE) |
| [cudaq::matrix_handler::to_matrix | -                                 |
|     (C++                          | [cudaq::scalar_operator::evaluate |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4NK5cudaq14matrix_handler9to_mat |    function)](api/languages/cpp_a |
| rixERNSt13unordered_mapINSt6size_ | pi.html#_CPPv4NK5cudaq15scalar_op |
| tENSt7int64_tEEERKNSt13unordered_ | erator8evaluateERKNSt13unordered_ |
| mapINSt6stringENSt7complexIdEEEE) | mapINSt6stringENSt7complexIdEEEE) |
| -                                 | -   [cudaq::scalar_ope            |
| [cudaq::matrix_handler::to_string | rator::get_parameter_descriptions |
|     (C++                          |     (C++                          |
|     function)](api/               |     f                             |
| languages/cpp_api.html#_CPPv4NK5c | unction)](api/languages/cpp_api.h |
| udaq14matrix_handler9to_stringEb) | tml#_CPPv4NK5cudaq15scalar_operat |
| -                                 | or26get_parameter_descriptionsEv) |
| [cudaq::matrix_handler::unique_id | -   [cu                           |
|     (C++                          | daq::scalar_operator::is_constant |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     function)](api/lang           |
| udaq14matrix_handler9unique_idEv) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [cudaq:                       | 15scalar_operator11is_constantEv) |
| :matrix_handler::\~matrix_handler | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator\* |
|     functi                        |     (C++                          |
| on)](api/languages/cpp_api.html#_ |     function                      |
| CPPv4N5cudaq14matrix_handlerD0Ev) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::matrix_op (C++        | Pv4N5cudaq15scalar_operatormlENSt |
|     type)](api/languages/cpp_a    | 7complexIdEERK15scalar_operator), |
| pi.html#_CPPv4N5cudaq9matrix_opE) |     [\[1\                         |
| -   [cudaq::matrix_op_term (C++   | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq15scalar_operatormlENSt |
|  type)](api/languages/cpp_api.htm | 7complexIdEERR15scalar_operator), |
| l#_CPPv4N5cudaq14matrix_op_termE) |     [\[2\]](api/languages/cp      |
| -                                 | p_api.html#_CPPv4N5cudaq15scalar_ |
|    [cudaq::mdiag_operator_handler | operatormlEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     class)](                      | p_api.html#_CPPv4N5cudaq15scalar_ |
| api/languages/cpp_api.html#_CPPv4 | operatormlEdRR15scalar_operator), |
| N5cudaq22mdiag_operator_handlerE) |     [\[4\]](api/languages         |
| -   [cudaq::mpi (C++              | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     type)](api/languages          | alar_operatormlENSt7complexIdEE), |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::mpi::all_gather (C++  | _api.html#_CPPv4NKR5cudaq15scalar |
|     fu                            | _operatormlERK15scalar_operator), |
| nction)](api/languages/cpp_api.ht |     [\[6\]]                       |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | (api/languages/cpp_api.html#_CPPv |
| RNSt6vectorIdEERKNSt6vectorIdEE), | 4NKR5cudaq15scalar_operatormlEd), |
|                                   |     [\[7\]](api/language          |
|   [\[1\]](api/languages/cpp_api.h | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| tml#_CPPv4N5cudaq3mpi10all_gather | alar_operatormlENSt7complexIdEE), |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     [\[8\]](api/languages/cp      |
| -   [cudaq::mpi::all_reduce (C++  | p_api.html#_CPPv4NO5cudaq15scalar |
|                                   | _operatormlERK15scalar_operator), |
|  function)](api/languages/cpp_api |     [\[9\                         |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | ]](api/languages/cpp_api.html#_CP |
| reduceE1TRK1TRK14BinaryFunction), | Pv4NO5cudaq15scalar_operatormlEd) |
|     [\[1\]](api/langu             | -   [cu                           |
| ages/cpp_api.html#_CPPv4I00EN5cud | daq::scalar_operator::operator\*= |
| aq3mpi10all_reduceE1TRK1TRK4Func) |     (C++                          |
| -   [cudaq::mpi::broadcast (C++   |     function)](api/languag        |
|     function)](api/               | es/cpp_api.html#_CPPv4N5cudaq15sc |
| languages/cpp_api.html#_CPPv4N5cu | alar_operatormLENSt7complexIdEE), |
| daq3mpi9broadcastERNSt6stringEi), |     [\[1\]](api/languages/c       |
|     [\[1\]](api/la                | pp_api.html#_CPPv4N5cudaq15scalar |
| nguages/cpp_api.html#_CPPv4N5cuda | _operatormLERK15scalar_operator), |
| q3mpi9broadcastERNSt6vectorIdEEi) |     [\[2                          |
| -   [cudaq::mpi::finalize (C++    | \]](api/languages/cpp_api.html#_C |
|     f                             | PPv4N5cudaq15scalar_operatormLEd) |
| unction)](api/languages/cpp_api.h | -   [                             |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | cudaq::scalar_operator::operator+ |
| -   [cudaq::mpi::initialize (C++  |     (C++                          |
|     function                      |     function                      |
| )](api/languages/cpp_api.html#_CP | )](api/languages/cpp_api.html#_CP |
| Pv4N5cudaq3mpi10initializeEiPPc), | Pv4N5cudaq15scalar_operatorplENSt |
|     [                             | 7complexIdEERK15scalar_operator), |
| \[1\]](api/languages/cpp_api.html |     [\[1\                         |
| #_CPPv4N5cudaq3mpi10initializeEv) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::mpi::is_initialized   | Pv4N5cudaq15scalar_operatorplENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     function                      |     [\[2\]](api/languages/cp      |
| )](api/languages/cpp_api.html#_CP | p_api.html#_CPPv4N5cudaq15scalar_ |
| Pv4N5cudaq3mpi14is_initializedEv) | operatorplEdRK15scalar_operator), |
| -   [cudaq::mpi::num_ranks (C++   |     [\[3\]](api/languages/cp      |
|     fu                            | p_api.html#_CPPv4N5cudaq15scalar_ |
| nction)](api/languages/cpp_api.ht | operatorplEdRR15scalar_operator), |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     [\[4\]](api/languages         |
| -   [cudaq::mpi::rank (C++        | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|                                   | alar_operatorplENSt7complexIdEE), |
|    function)](api/languages/cpp_a |     [\[5\]](api/languages/cpp     |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::noise_model (C++      | _operatorplERK15scalar_operator), |
|                                   |     [\[6\]]                       |
|    class)](api/languages/cpp_api. | (api/languages/cpp_api.html#_CPPv |
| html#_CPPv4N5cudaq11noise_modelE) | 4NKR5cudaq15scalar_operatorplEd), |
| -   [cudaq::n                     |     [\[7\]]                       |
| oise_model::add_all_qubit_channel | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatorplEv), |
|     function)](api                |     [\[8\]](api/language          |
| /languages/cpp_api.html#_CPPv4IDp | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| EN5cudaq11noise_model21add_all_qu | alar_operatorplENSt7complexIdEE), |
| bit_channelEvRK13kraus_channeli), |     [\[9\]](api/languages/cp      |
|     [\[1\]](api/langua            | p_api.html#_CPPv4NO5cudaq15scalar |
| ges/cpp_api.html#_CPPv4N5cudaq11n | _operatorplERK15scalar_operator), |
| oise_model21add_all_qubit_channel |     [\[10\]                       |
| ERKNSt6stringERK13kraus_channeli) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4NO5cudaq15scalar_operatorplEd), |
|  [cudaq::noise_model::add_channel |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     funct                         | Pv4NO5cudaq15scalar_operatorplEv) |
| ion)](api/languages/cpp_api.html# | -   [c                            |
| _CPPv4IDpEN5cudaq11noise_model11a | udaq::scalar_operator::operator+= |
| dd_channelEvRK15PredicateFuncTy), |     (C++                          |
|     [\[1\]](api/languages/cpp_    |     function)](api/languag        |
| api.html#_CPPv4IDpEN5cudaq11noise | es/cpp_api.html#_CPPv4N5cudaq15sc |
| _model11add_channelEvRKNSt6vector | alar_operatorpLENSt7complexIdEE), |
| INSt6size_tEEERK13kraus_channel), |     [\[1\]](api/languages/c       |
|     [\[2\]](ap                    | pp_api.html#_CPPv4N5cudaq15scalar |
| i/languages/cpp_api.html#_CPPv4N5 | _operatorpLERK15scalar_operator), |
| cudaq11noise_model11add_channelER |     [\[2                          |
| KNSt6stringERK15PredicateFuncTy), | \]](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq15scalar_operatorpLEd) |
| [\[3\]](api/languages/cpp_api.htm | -   [                             |
| l#_CPPv4N5cudaq11noise_model11add | cudaq::scalar_operator::operator- |
| _channelERKNSt6stringERKNSt6vecto |     (C++                          |
| rINSt6size_tEEERK13kraus_channel) |     function                      |
| -   [cudaq::noise_model::empty    | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatormiENSt |
|     function                      | 7complexIdEERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[1\                         |
| Pv4NK5cudaq11noise_model5emptyEv) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatormiENSt |
| [cudaq::noise_model::get_channels | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     function)](api/l              | p_api.html#_CPPv4N5cudaq15scalar_ |
| anguages/cpp_api.html#_CPPv4I0ENK | operatormiEdRK15scalar_operator), |
| 5cudaq11noise_model12get_channels |     [\[3\]](api/languages/cp      |
| ENSt6vectorI13kraus_channelEERKNS | p_api.html#_CPPv4N5cudaq15scalar_ |
| t6vectorINSt6size_tEEERKNSt6vecto | operatormiEdRR15scalar_operator), |
| rINSt6size_tEEERKNSt6vectorIdEE), |     [\[4\]](api/languages         |
|     [\[1\]](api/languages/cpp_a   | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| pi.html#_CPPv4NK5cudaq11noise_mod | alar_operatormiENSt7complexIdEE), |
| el12get_channelsERKNSt6stringERKN |     [\[5\]](api/languages/cpp     |
| St6vectorINSt6size_tEEERKNSt6vect | _api.html#_CPPv4NKR5cudaq15scalar |
| orINSt6size_tEEERKNSt6vectorIdEE) | _operatormiERK15scalar_operator), |
| -                                 |     [\[6\]]                       |
|  [cudaq::noise_model::noise_model | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEd), |
|     function)](api                |     [\[7\]]                       |
| /languages/cpp_api.html#_CPPv4N5c | (api/languages/cpp_api.html#_CPPv |
| udaq11noise_model11noise_modelEv) | 4NKR5cudaq15scalar_operatormiEv), |
| -   [cu                           |     [\[8\]](api/language          |
| daq::noise_model::PredicateFuncTy | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     type)](api/la                 |     [\[9\]](api/languages/cp      |
| nguages/cpp_api.html#_CPPv4N5cuda | p_api.html#_CPPv4NO5cudaq15scalar |
| q11noise_model15PredicateFuncTyE) | _operatormiERK15scalar_operator), |
| -   [cud                          |     [\[10\]                       |
| aq::noise_model::register_channel | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NO5cudaq15scalar_operatormiEd), |
|     function)](api/languages      |     [\[11\                        |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | ]](api/languages/cpp_api.html#_CP |
| noise_model16register_channelEvv) | Pv4NO5cudaq15scalar_operatormiEv) |
| -   [cudaq::                      | -   [c                            |
| noise_model::requires_constructor | udaq::scalar_operator::operator-= |
|     (C++                          |     (C++                          |
|     type)](api/languages/cp       |     function)](api/languag        |
| p_api.html#_CPPv4I0DpEN5cudaq11no | es/cpp_api.html#_CPPv4N5cudaq15sc |
| ise_model20requires_constructorE) | alar_operatormIENSt7complexIdEE), |
| -   [cudaq::noise_model_type (C++ |     [\[1\]](api/languages/c       |
|     e                             | pp_api.html#_CPPv4N5cudaq15scalar |
| num)](api/languages/cpp_api.html# | _operatormIERK15scalar_operator), |
| _CPPv4N5cudaq16noise_model_typeE) |     [\[2                          |
| -   [cudaq::no                    | \]](api/languages/cpp_api.html#_C |
| ise_model_type::amplitude_damping | PPv4N5cudaq15scalar_operatormIEd) |
|     (C++                          | -   [                             |
|     enumerator)](api/languages    | cudaq::scalar_operator::operator/ |
| /cpp_api.html#_CPPv4N5cudaq16nois |     (C++                          |
| e_model_type17amplitude_dampingE) |     function                      |
| -   [cudaq::noise_mode            | )](api/languages/cpp_api.html#_CP |
| l_type::amplitude_damping_channel | Pv4N5cudaq15scalar_operatordvENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     e                             |     [\[1\                         |
| numerator)](api/languages/cpp_api | ]](api/languages/cpp_api.html#_CP |
| .html#_CPPv4N5cudaq16noise_model_ | Pv4N5cudaq15scalar_operatordvENSt |
| type25amplitude_damping_channelE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::n                     |     [\[2\]](api/languages/cp      |
| oise_model_type::bit_flip_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRK15scalar_operator), |
|     enumerator)](api/language     |     [\[3\]](api/languages/cp      |
| s/cpp_api.html#_CPPv4N5cudaq16noi | p_api.html#_CPPv4N5cudaq15scalar_ |
| se_model_type16bit_flip_channelE) | operatordvEdRR15scalar_operator), |
| -   [cudaq::                      |     [\[4\]](api/languages         |
| noise_model_type::depolarization1 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatordvENSt7complexIdEE), |
|     enumerator)](api/languag      |     [\[5\]](api/languages/cpp     |
| es/cpp_api.html#_CPPv4N5cudaq16no | _api.html#_CPPv4NKR5cudaq15scalar |
| ise_model_type15depolarization1E) | _operatordvERK15scalar_operator), |
| -   [cudaq::                      |     [\[6\]]                       |
| noise_model_type::depolarization2 | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatordvEd), |
|     enumerator)](api/languag      |     [\[7\]](api/language          |
| es/cpp_api.html#_CPPv4N5cudaq16no | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| ise_model_type15depolarization2E) | alar_operatordvENSt7complexIdEE), |
| -   [cudaq::noise_m               |     [\[8\]](api/languages/cp      |
| odel_type::depolarization_channel | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatordvERK15scalar_operator), |
|                                   |     [\[9\                         |
|   enumerator)](api/languages/cpp_ | ]](api/languages/cpp_api.html#_CP |
| api.html#_CPPv4N5cudaq16noise_mod | Pv4NO5cudaq15scalar_operatordvEd) |
| el_type22depolarization_channelE) | -   [c                            |
| -                                 | udaq::scalar_operator::operator/= |
|  [cudaq::noise_model_type::pauli1 |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     enumerator)](a                | es/cpp_api.html#_CPPv4N5cudaq15sc |
| pi/languages/cpp_api.html#_CPPv4N | alar_operatordVENSt7complexIdEE), |
| 5cudaq16noise_model_type6pauli1E) |     [\[1\]](api/languages/c       |
| -                                 | pp_api.html#_CPPv4N5cudaq15scalar |
|  [cudaq::noise_model_type::pauli2 | _operatordVERK15scalar_operator), |
|     (C++                          |     [\[2                          |
|     enumerator)](a                | \]](api/languages/cpp_api.html#_C |
| pi/languages/cpp_api.html#_CPPv4N | PPv4N5cudaq15scalar_operatordVEd) |
| 5cudaq16noise_model_type6pauli2E) | -   [                             |
| -   [cudaq                        | cudaq::scalar_operator::operator= |
| ::noise_model_type::phase_damping |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     enumerator)](api/langu        | pp_api.html#_CPPv4N5cudaq15scalar |
| ages/cpp_api.html#_CPPv4N5cudaq16 | _operatoraSERK15scalar_operator), |
| noise_model_type13phase_dampingE) |     [\[1\]](api/languages/        |
| -   [cudaq::noi                   | cpp_api.html#_CPPv4N5cudaq15scala |
| se_model_type::phase_flip_channel | r_operatoraSERR15scalar_operator) |
|     (C++                          | -   [c                            |
|     enumerator)](api/languages/   | udaq::scalar_operator::operator== |
| cpp_api.html#_CPPv4N5cudaq16noise |     (C++                          |
| _model_type18phase_flip_channelE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4NK5cudaq15scala |
| [cudaq::noise_model_type::unknown | r_operatoreqERK15scalar_operator) |
|     (C++                          | -   [cudaq:                       |
|     enumerator)](ap               | :scalar_operator::scalar_operator |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq16noise_model_type7unknownE) |     func                          |
| -                                 | tion)](api/languages/cpp_api.html |
| [cudaq::noise_model_type::x_error | #_CPPv4N5cudaq15scalar_operator15 |
|     (C++                          | scalar_operatorENSt7complexIdEE), |
|     enumerator)](ap               |     [\[1\]](api/langu             |
| i/languages/cpp_api.html#_CPPv4N5 | ages/cpp_api.html#_CPPv4N5cudaq15 |
| cudaq16noise_model_type7x_errorE) | scalar_operator15scalar_operatorE |
| -                                 | RK15scalar_callbackRRNSt13unorder |
| [cudaq::noise_model_type::y_error | ed_mapINSt6stringENSt6stringEEE), |
|     (C++                          |     [\[2\                         |
|     enumerator)](ap               | ]](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4N5cudaq15scalar_operator15scal |
| cudaq16noise_model_type7y_errorE) | ar_operatorERK15scalar_operator), |
| -                                 |     [\[3\]](api/langu             |
| [cudaq::noise_model_type::z_error | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     (C++                          | scalar_operator15scalar_operatorE |
|     enumerator)](ap               | RR15scalar_callbackRRNSt13unorder |
| i/languages/cpp_api.html#_CPPv4N5 | ed_mapINSt6stringENSt6stringEEE), |
| cudaq16noise_model_type7z_errorE) |     [\[4\                         |
| -   [cudaq::num_available_gpus    | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operator15scal |
|     function                      | ar_operatorERR15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[5\]](api/language          |
| Pv4N5cudaq18num_available_gpusEv) | s/cpp_api.html#_CPPv4N5cudaq15sca |
| -   [cudaq::observe (C++          | lar_operator15scalar_operatorEd), |
|     function)]                    |     [\[6\]](api/languag           |
| (api/languages/cpp_api.html#_CPPv | es/cpp_api.html#_CPPv4N5cudaq15sc |
| 4I00DpEN5cudaq7observeENSt6vector | alar_operator15scalar_operatorEv) |
| I14observe_resultEERR13QuantumKer | -   [                             |
| nelRK15SpinOpContainerDpRR4Args), | cudaq::scalar_operator::to_matrix |
|     [\[1\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4I0DpEN5cudaq7observe |                                   |
| E14observe_resultNSt6size_tERR13Q |   function)](api/languages/cpp_ap |
| uantumKernelRK7spin_opDpRR4Args), | i.html#_CPPv4NK5cudaq15scalar_ope |
|     [\[                           | rator9to_matrixERKNSt13unordered_ |
| 2\]](api/languages/cpp_api.html#_ | mapINSt6stringENSt7complexIdEEEE) |
| CPPv4I0DpEN5cudaq7observeE14obser | -   [                             |
| ve_resultRK15observe_optionsRR13Q | cudaq::scalar_operator::to_string |
| uantumKernelRK7spin_opDpRR4Args), |     (C++                          |
|     [\[3\]](api/lang              |     function)](api/l              |
| uages/cpp_api.html#_CPPv4I0DpEN5c | anguages/cpp_api.html#_CPPv4NK5cu |
| udaq7observeE14observe_resultRR13 | daq15scalar_operator9to_stringEv) |
| QuantumKernelRK7spin_opDpRR4Args) | -   [cudaq::s                     |
| -   [cudaq::observe_options (C++  | calar_operator::\~scalar_operator |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |     functio                       |
| #_CPPv4N5cudaq15observe_optionsE) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::observe_result (C++   | PPv4N5cudaq15scalar_operatorD0Ev) |
|                                   | -   [cudaq::set_noise (C++        |
| class)](api/languages/cpp_api.htm |     function)](api/langu          |
| l#_CPPv4N5cudaq14observe_resultE) | ages/cpp_api.html#_CPPv4N5cudaq9s |
| -                                 | et_noiseERKN5cudaq11noise_modelE) |
|    [cudaq::observe_result::counts | -   [cudaq::set_random_seed (C++  |
|     (C++                          |     function)](api/               |
|     function)](api/languages/c    | languages/cpp_api.html#_CPPv4N5cu |
| pp_api.html#_CPPv4N5cudaq14observ | daq15set_random_seedENSt6size_tE) |
| e_result6countsERK12spin_op_term) | -   [cudaq::simulation_precision  |
| -   [cudaq::observe_result::dump  |     (C++                          |
|     (C++                          |     enum)                         |
|     function)                     | ](api/languages/cpp_api.html#_CPP |
| ](api/languages/cpp_api.html#_CPP | v4N5cudaq20simulation_precisionE) |
| v4N5cudaq14observe_result4dumpEv) | -   [                             |
| -   [c                            | cudaq::simulation_precision::fp32 |
| udaq::observe_result::expectation |     (C++                          |
|     (C++                          |     enumerator)](api              |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
| function)](api/languages/cpp_api. | udaq20simulation_precision4fp32E) |
| html#_CPPv4N5cudaq14observe_resul | -   [                             |
| t11expectationERK12spin_op_term), | cudaq::simulation_precision::fp64 |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     enumerator)](api              |
| q14observe_result11expectationEv) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cuda                         | udaq20simulation_precision4fp64E) |
| q::observe_result::id_coefficient | -   [cudaq::SimulationState (C++  |
|     (C++                          |     c                             |
|     function)](api/langu          | lass)](api/languages/cpp_api.html |
| ages/cpp_api.html#_CPPv4N5cudaq14 | #_CPPv4N5cudaq15SimulationStateE) |
| observe_result14id_coefficientEv) | -   [                             |
| -   [cuda                         | cudaq::SimulationState::precision |
| q::observe_result::observe_result |     (C++                          |
|     (C++                          |     enum)](api                    |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
|   function)](api/languages/cpp_ap | udaq15SimulationState9precisionE) |
| i.html#_CPPv4N5cudaq14observe_res | -   [cudaq:                       |
| ult14observe_resultEdRK7spin_op), | :SimulationState::precision::fp32 |
|     [\[1\]](a                     |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     enumerator)](api/lang         |
| 5cudaq14observe_result14observe_r | uages/cpp_api.html#_CPPv4N5cudaq1 |
| esultEdRK7spin_op13sample_result) | 5SimulationState9precision4fp32E) |
| -                                 | -   [cudaq:                       |
|  [cudaq::observe_result::operator | :SimulationState::precision::fp64 |
|     double (C++                   |     (C++                          |
|     functio                       |     enumerator)](api/lang         |
| n)](api/languages/cpp_api.html#_C | uages/cpp_api.html#_CPPv4N5cudaq1 |
| PPv4N5cudaq14observe_resultcvdEv) | 5SimulationState9precision4fp64E) |
| -                                 | -                                 |
|  [cudaq::observe_result::raw_data |   [cudaq::SimulationState::Tensor |
|     (C++                          |     (C++                          |
|     function)](ap                 |     struct)](                     |
| i/languages/cpp_api.html#_CPPv4N5 | api/languages/cpp_api.html#_CPPv4 |
| cudaq14observe_result8raw_dataEv) | N5cudaq15SimulationState6TensorE) |
| -   [cudaq::operator_handler (C++ | -   [cudaq::spin_handler (C++     |
|     cl                            |                                   |
| ass)](api/languages/cpp_api.html# |   class)](api/languages/cpp_api.h |
| _CPPv4N5cudaq16operator_handlerE) | tml#_CPPv4N5cudaq12spin_handlerE) |
| -   [cudaq::optimizable_function  | -   [cudaq:                       |
|     (C++                          | :spin_handler::to_diagonal_matrix |
|     class)                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/la             |
| v4N5cudaq20optimizable_functionE) | nguages/cpp_api.html#_CPPv4NK5cud |
| -   [cudaq::optimization_result   | aq12spin_handler18to_diagonal_mat |
|     (C++                          | rixERNSt13unordered_mapINSt6size_ |
|     type                          | tENSt7int64_tEEERKNSt13unordered_ |
| )](api/languages/cpp_api.html#_CP | mapINSt6stringENSt7complexIdEEEE) |
| Pv4N5cudaq19optimization_resultE) | -                                 |
| -   [cudaq::optimizer (C++        |   [cudaq::spin_handler::to_matrix |
|     class)](api/languages/cpp_a   |     (C++                          |
| pi.html#_CPPv4N5cudaq9optimizerE) |     function                      |
| -   [cudaq::optimizer::optimize   | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq12spin_handler9to_matri |
|                                   | xERKNSt6stringENSt7complexIdEEb), |
|  function)](api/languages/cpp_api |     [\[1                          |
| .html#_CPPv4N5cudaq9optimizer8opt | \]](api/languages/cpp_api.html#_C |
| imizeEKiRR20optimizable_function) | PPv4NK5cudaq12spin_handler9to_mat |
| -   [cu                           | rixERNSt13unordered_mapINSt6size_ |
| daq::optimizer::requiresGradients | tENSt7int64_tEEERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|     function)](api/la             | -   [cuda                         |
| nguages/cpp_api.html#_CPPv4N5cuda | q::spin_handler::to_sparse_matrix |
| q9optimizer17requiresGradientsEv) |     (C++                          |
| -   [cudaq::orca (C++             |     function)](api/               |
|     type)](api/languages/         | languages/cpp_api.html#_CPPv4N5cu |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | daq12spin_handler16to_sparse_matr |
| -   [cudaq::orca::sample (C++     | ixERKNSt6stringENSt7complexIdEEb) |
|     function)](api/languages/c    | -                                 |
| pp_api.html#_CPPv4N5cudaq4orca6sa |   [cudaq::spin_handler::to_string |
| mpleERNSt6vectorINSt6size_tEEERNS |     (C++                          |
| t6vectorINSt6size_tEEERNSt6vector |     function)](ap                 |
| IdEERNSt6vectorIdEEiNSt6size_tE), | i/languages/cpp_api.html#_CPPv4NK |
|     [\[1\]]                       | 5cudaq12spin_handler9to_stringEb) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq4orca6sampleERNSt6vectorI |   [cudaq::spin_handler::unique_id |
| NSt6size_tEEERNSt6vectorINSt6size |     (C++                          |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     function)](ap                 |
| -   [cudaq::orca::sample_async    | i/languages/cpp_api.html#_CPPv4NK |
|     (C++                          | 5cudaq12spin_handler9unique_idEv) |
|                                   | -   [cudaq::spin_op (C++          |
| function)](api/languages/cpp_api. |     type)](api/languages/cpp      |
| html#_CPPv4N5cudaq4orca12sample_a | _api.html#_CPPv4N5cudaq7spin_opE) |
| syncERNSt6vectorINSt6size_tEEERNS | -   [cudaq::spin_op_term (C++     |
| t6vectorINSt6size_tEEERNSt6vector |                                   |
| IdEERNSt6vectorIdEEiNSt6size_tE), |    type)](api/languages/cpp_api.h |
|     [\[1\]](api/la                | tml#_CPPv4N5cudaq12spin_op_termE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::state (C++            |
| q4orca12sample_asyncERNSt6vectorI |     class)](api/languages/c       |
| NSt6size_tEEERNSt6vectorINSt6size | pp_api.html#_CPPv4N5cudaq5stateE) |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | -   [cudaq::state::amplitude (C++ |
| -   [cudaq::OrcaRemoteRESTQPU     |     function)](api/lang           |
|     (C++                          | uages/cpp_api.html#_CPPv4N5cudaq5 |
|     cla                           | state9amplitudeERKNSt6vectorIiEE) |
| ss)](api/languages/cpp_api.html#_ | -   [cudaq::state::amplitudes     |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) |     (C++                          |
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
| stom)](api/languages/python_api.h | -   [expectation()                |
| tml#cudaq.operators.custom.empty) |     (cudaq.ObserveResult          |
|     -   [(in module               |     metho                         |
|                                   | d)](api/languages/python_api.html |
|       cudaq.spin)](api/languages/ | #cudaq.ObserveResult.expectation) |
| python_api.html#cudaq.spin.empty) |     -   [(cudaq.SampleResult      |
| -   [empty_op()                   |         meth                      |
|     (                             | od)](api/languages/python_api.htm |
| cudaq.operators.spin.SpinOperator | l#cudaq.SampleResult.expectation) |
|     static                        | -   [expectation_values()         |
|     method)](api/lan              |     (cudaq.EvolveResult           |
| guages/python_api.html#cudaq.oper |     method)](ap                   |
| ators.spin.SpinOperator.empty_op) | i/languages/python_api.html#cudaq |
| -   [enable_return_to_log()       | .EvolveResult.expectation_values) |
|     (cudaq.PyKernelDecorator      | -   [expectation_z()              |
|     method)](api/langu            |     (cudaq.SampleResult           |
| ages/python_api.html#cudaq.PyKern |     method                        |
| elDecorator.enable_return_to_log) | )](api/languages/python_api.html# |
| -   [epsilon                      | cudaq.SampleResult.expectation_z) |
|     (cudaq.optimizers.Adam        | -   [expected_dimensions          |
|     prope                         |     (cuda                         |
| rty)](api/languages/python_api.ht | q.operators.MatrixOperatorElement |
| ml#cudaq.optimizers.Adam.epsilon) |                                   |
| -   [estimate_resources() (in     | property)](api/languages/python_a |
|     module                        | pi.html#cudaq.operators.MatrixOpe |
|                                   | ratorElement.expected_dimensions) |
|    cudaq)](api/languages/python_a |                                   |
| pi.html#cudaq.estimate_resources) |                                   |
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
|     (cudaq.SampleResult           | -   [getTensor() (cudaq.State     |
|     method)](api                  |     method)](api/languages/pytho  |
| /languages/python_api.html#cudaq. | n_api.html#cudaq.State.getTensor) |
| SampleResult.get_marginal_counts) | -   [getTensors() (cudaq.State    |
| -   [get_ops()                    |     method)](api/languages/python |
|     (cudaq.KrausChannel           | _api.html#cudaq.State.getTensors) |
|                                   | -   [gradient (class in           |
| method)](api/languages/python_api |     cudaq.g                       |
| .html#cudaq.KrausChannel.get_ops) | radients)](api/languages/python_a |
| -   [get_pauli_word()             | pi.html#cudaq.gradients.gradient) |
|     (cuda                         | -   [GradientDescent (class in    |
| q.operators.spin.SpinOperatorTerm |     cudaq.optimizers              |
|     method)](api/languages/pyt    | )](api/languages/python_api.html# |
| hon_api.html#cudaq.operators.spin | cudaq.optimizers.GradientDescent) |
| .SpinOperatorTerm.get_pauli_word) |                                   |
| -   [get_precision()              |                                   |
|     (cudaq.Target                 |                                   |
|                                   |                                   |
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
| -   [handle_call_arguments()      | -   [has_target() (in module      |
|     (cudaq.PyKernelDecorator      |     cudaq)](api/languages/        |
|     method)](api/langua           | python_api.html#cudaq.has_target) |
| ges/python_api.html#cudaq.PyKerne |                                   |
| lDecorator.handle_call_arguments) |                                   |
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
|         static                    | -   [is_constant()                |
|         method)](api/languages    |                                   |
| /python_api.html#cudaq.operators. |   (cudaq.operators.ScalarOperator |
| fermion.FermionOperator.identity) |     method)](api/lan              |
|     -                             | guages/python_api.html#cudaq.oper |
|  [(cudaq.operators.MatrixOperator | ators.ScalarOperator.is_constant) |
|         static                    | -   [is_emulated() (cudaq.Target  |
|         method)](api/             |                                   |
| languages/python_api.html#cudaq.o |   method)](api/languages/python_a |
| perators.MatrixOperator.identity) | pi.html#cudaq.Target.is_emulated) |
|     -   [(                        | -   [is_identity()                |
| cudaq.operators.spin.SpinOperator |     (cudaq.                       |
|         static                    | operators.boson.BosonOperatorTerm |
|         method)](api/lan          |     method)](api/languages/py     |
| guages/python_api.html#cudaq.oper | thon_api.html#cudaq.operators.bos |
| ators.spin.SpinOperator.identity) | on.BosonOperatorTerm.is_identity) |
|     -   [(in module               |     -   [(cudaq.oper              |
|                                   | ators.fermion.FermionOperatorTerm |
|  cudaq.boson)](api/languages/pyth |                                   |
| on_api.html#cudaq.boson.identity) |     method)](api/languages/python |
|     -   [(in module               | _api.html#cudaq.operators.fermion |
|         cud                       | .FermionOperatorTerm.is_identity) |
| aq.fermion)](api/languages/python |     -   [(c                       |
| _api.html#cudaq.fermion.identity) | udaq.operators.MatrixOperatorTerm |
|     -   [(in module               |         method)](api/languag      |
|                                   | es/python_api.html#cudaq.operator |
|    cudaq.spin)](api/languages/pyt | s.MatrixOperatorTerm.is_identity) |
| hon_api.html#cudaq.spin.identity) |     -   [(                        |
| -   [initial_parameters           | cudaq.operators.spin.SpinOperator |
|     (cudaq.optimizers.Adam        |         method)](api/langua       |
|     property)](api/l              | ges/python_api.html#cudaq.operato |
| anguages/python_api.html#cudaq.op | rs.spin.SpinOperator.is_identity) |
| timizers.Adam.initial_parameters) |     -   [(cuda                    |
|     -   [(cudaq.optimizers.COBYLA | q.operators.spin.SpinOperatorTerm |
|         property)](api/lan        |         method)](api/languages/   |
| guages/python_api.html#cudaq.opti | python_api.html#cudaq.operators.s |
| mizers.COBYLA.initial_parameters) | pin.SpinOperatorTerm.is_identity) |
|     -   [                         | -   [is_initialized() (in module  |
| (cudaq.optimizers.GradientDescent |     c                             |
|                                   | udaq.mpi)](api/languages/python_a |
|       property)](api/languages/py | pi.html#cudaq.mpi.is_initialized) |
| thon_api.html#cudaq.optimizers.Gr | -   [is_on_gpu() (cudaq.State     |
| adientDescent.initial_parameters) |     method)](api/languages/pytho  |
|     -   [(cudaq.optimizers.LBFGS  | n_api.html#cudaq.State.is_on_gpu) |
|         property)](api/la         | -   [is_remote() (cudaq.Target    |
| nguages/python_api.html#cudaq.opt |     method)](api/languages/python |
| imizers.LBFGS.initial_parameters) | _api.html#cudaq.Target.is_remote) |
|                                   | -   [is_remote_simulator()        |
| -   [(cudaq.optimizers.NelderMead |     (cudaq.Target                 |
|         property)](api/languag    |     method                        |
| es/python_api.html#cudaq.optimize | )](api/languages/python_api.html# |
| rs.NelderMead.initial_parameters) | cudaq.Target.is_remote_simulator) |
|     -   [(cudaq.optimizers.SGD    | -   [items() (cudaq.SampleResult  |
|         property)](api/           |                                   |
| languages/python_api.html#cudaq.o |   method)](api/languages/python_a |
| ptimizers.SGD.initial_parameters) | pi.html#cudaq.SampleResult.items) |
|     -   [(cudaq.optimizers.SPSA   |                                   |
|         property)](api/l          |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| timizers.SPSA.initial_parameters) |                                   |
+-----------------------------------+-----------------------------------+

## K {#K}

+-----------------------------------+-----------------------------------+
| -   [Kernel (in module            | -   [KrausChannel (class in       |
|     cudaq)](api/langua            |     cudaq)](api/languages/py      |
| ges/python_api.html#cudaq.Kernel) | thon_api.html#cudaq.KrausChannel) |
| -   [kernel() (in module          | -   [KrausOperator (class in      |
|     cudaq)](api/langua            |     cudaq)](api/languages/pyt     |
| ges/python_api.html#cudaq.kernel) | hon_api.html#cudaq.KrausOperator) |
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
| q.optimizers.SPSA.max_iterations) |                                   |
| -   [mdiag_sparse_matrix (C++     |                                   |
|     type)](api/languages/cpp_api. |                                   |
| html#_CPPv419mdiag_sparse_matrix) |                                   |
+-----------------------------------+-----------------------------------+

## N {#N}

+-----------------------------------+-----------------------------------+
| -   [name (cudaq.PyKernel         | -   [num_qpus() (cudaq.Target     |
|     attribute)](api/languages/pyt |     method)](api/languages/pytho  |
| hon_api.html#cudaq.PyKernel.name) | n_api.html#cudaq.Target.num_qpus) |
|                                   | -   [num_qubits() (cudaq.State    |
|   -   [(cudaq.SimulationPrecision |     method)](api/languages/python |
|         proper                    | _api.html#cudaq.State.num_qubits) |
| ty)](api/languages/python_api.htm | -   [num_ranks() (in module       |
| l#cudaq.SimulationPrecision.name) |     cudaq.mpi)](api/languages/pyt |
|     -   [(cudaq.spin.Pauli        | hon_api.html#cudaq.mpi.num_ranks) |
|                                   | -   [num_rows()                   |
|    property)](api/languages/pytho |     (cudaq.ComplexMatrix          |
| n_api.html#cudaq.spin.Pauli.name) |     me                            |
|     -   [(cudaq.Target            | thod)](api/languages/python_api.h |
|                                   | tml#cudaq.ComplexMatrix.num_rows) |
|        property)](api/languages/p | -   [number() (in module          |
| ython_api.html#cudaq.Target.name) |                                   |
| -   [NelderMead (class in         |    cudaq.boson)](api/languages/py |
|     cudaq.optim                   | thon_api.html#cudaq.boson.number) |
| izers)](api/languages/python_api. |     -   [(in module               |
| html#cudaq.optimizers.NelderMead) |         c                         |
| -   [NoiseModel (class in         | udaq.fermion)](api/languages/pyth |
|     cudaq)](api/languages/        | on_api.html#cudaq.fermion.number) |
| python_api.html#cudaq.NoiseModel) |     -   [(in module               |
| -   [num_available_gpus() (in     |         cudaq.operators.cus       |
|     module                        | tom)](api/languages/python_api.ht |
|                                   | ml#cudaq.operators.custom.number) |
|    cudaq)](api/languages/python_a | -   [nvqir::MPSSimulationState    |
| pi.html#cudaq.num_available_gpus) |     (C++                          |
| -   [num_columns()                |     class)]                       |
|     (cudaq.ComplexMatrix          | (api/languages/cpp_api.html#_CPPv |
|     metho                         | 4I0EN5nvqir18MPSSimulationStateE) |
| d)](api/languages/python_api.html | -                                 |
| #cudaq.ComplexMatrix.num_columns) |  [nvqir::TensorNetSimulationState |
|                                   |     (C++                          |
|                                   |     class)](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4I0EN5 |
|                                   | nvqir24TensorNetSimulationStateE) |
+-----------------------------------+-----------------------------------+

## O {#O}

+-----------------------------------+-----------------------------------+
| -   [observe() (in module         | -   [OptimizationResult (class in |
|     cudaq)](api/languag           |                                   |
| es/python_api.html#cudaq.observe) |    cudaq)](api/languages/python_a |
| -   [observe_async() (in module   | pi.html#cudaq.OptimizationResult) |
|     cudaq)](api/languages/pyt     | -   [overlap() (cudaq.State       |
| hon_api.html#cudaq.observe_async) |     method)](api/languages/pyt    |
| -   [ObserveResult (class in      | hon_api.html#cudaq.State.overlap) |
|     cudaq)](api/languages/pyt     |                                   |
| hon_api.html#cudaq.ObserveResult) |                                   |
| -   [OperatorSum (in module       |                                   |
|     cudaq.oper                    |                                   |
| ators)](api/languages/python_api. |                                   |
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
| nguages/python_api.html#cudaq.ope | -   [pre_compile()                |
| rators.MatrixOperator.parameters) |     (cudaq.PyKernelDecorator      |
|     -   [(cuda                    |     method)](                     |
| q.operators.MatrixOperatorElement | api/languages/python_api.html#cud |
|         property)](api/languages  | aq.PyKernelDecorator.pre_compile) |
| /python_api.html#cudaq.operators. | -   [probability()                |
| MatrixOperatorElement.parameters) |     (cudaq.SampleResult           |
|     -   [(c                       |     meth                          |
| udaq.operators.MatrixOperatorTerm | od)](api/languages/python_api.htm |
|         property)](api/langua     | l#cudaq.SampleResult.probability) |
| ges/python_api.html#cudaq.operato | -   [ProductOperator (in module   |
| rs.MatrixOperatorTerm.parameters) |     cudaq.operator                |
|     -                             | s)](api/languages/python_api.html |
|  [(cudaq.operators.ScalarOperator | #cudaq.operators.ProductOperator) |
|         property)](api/la         | -   [PyKernel (class in           |
| nguages/python_api.html#cudaq.ope |     cudaq)](api/language          |
| rators.ScalarOperator.parameters) | s/python_api.html#cudaq.PyKernel) |
|     -   [(                        | -   [PyKernelDecorator (class in  |
| cudaq.operators.spin.SpinOperator |     cudaq)](api/languages/python_ |
|         property)](api/langu      | api.html#cudaq.PyKernelDecorator) |
| ages/python_api.html#cudaq.operat |                                   |
| ors.spin.SpinOperator.parameters) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         property)](api/languages  |                                   |
| /python_api.html#cudaq.operators. |                                   |
| spin.SpinOperatorTerm.parameters) |                                   |
| -   [ParameterShift (class in     |                                   |
|     cudaq.gradien                 |                                   |
| ts)](api/languages/python_api.htm |                                   |
| l#cudaq.gradients.ParameterShift) |                                   |
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
| -   [qreg (in module              | -   [qubit_count                  |
|     cudaq)](api/lang              |     (                             |
| uages/python_api.html#cudaq.qreg) | cudaq.operators.spin.SpinOperator |
| -   [QuakeValue (class in         |     property)](api/langua         |
|     cudaq)](api/languages/        | ges/python_api.html#cudaq.operato |
| python_api.html#cudaq.QuakeValue) | rs.spin.SpinOperator.qubit_count) |
| -   [qubit (class in              |     -   [(cuda                    |
|     cudaq)](api/langu             | q.operators.spin.SpinOperatorTerm |
| ages/python_api.html#cudaq.qubit) |         property)](api/languages/ |
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
| -   [sample() (in module          | -   [signatureWithCallables()     |
|     cudaq)](api/langua            |     (cudaq.PyKernelDecorator      |
| ges/python_api.html#cudaq.sample) |     method)](api/languag          |
|     -   [(in module               | es/python_api.html#cudaq.PyKernel |
|                                   | Decorator.signatureWithCallables) |
|      cudaq.orca)](api/languages/p | -   [SimulationPrecision (class   |
| ython_api.html#cudaq.orca.sample) |     in                            |
| -   [sample_async() (in module    |                                   |
|     cudaq)](api/languages/py      |   cudaq)](api/languages/python_ap |
| thon_api.html#cudaq.sample_async) | i.html#cudaq.SimulationPrecision) |
| -   [SampleResult (class in       | -   [simulator (cudaq.Target      |
|     cudaq)](api/languages/py      |                                   |
| thon_api.html#cudaq.SampleResult) |   property)](api/languages/python |
| -   [ScalarOperator (class in     | _api.html#cudaq.Target.simulator) |
|     cudaq.operato                 | -   [slice() (cudaq.QuakeValue    |
| rs)](api/languages/python_api.htm |     method)](api/languages/python |
| l#cudaq.operators.ScalarOperator) | _api.html#cudaq.QuakeValue.slice) |
| -   [Schedule (class in           | -   [SpinOperator (class in       |
|     cudaq)](api/language          |     cudaq.operators.spin)         |
| s/python_api.html#cudaq.Schedule) | ](api/languages/python_api.html#c |
| -   [serialize()                  | udaq.operators.spin.SpinOperator) |
|     (                             | -   [SpinOperatorElement (class   |
| cudaq.operators.spin.SpinOperator |     in                            |
|     method)](api/lang             |     cudaq.operators.spin)](api/l  |
| uages/python_api.html#cudaq.opera | anguages/python_api.html#cudaq.op |
| tors.spin.SpinOperator.serialize) | erators.spin.SpinOperatorElement) |
|     -   [(cuda                    | -   [SpinOperatorTerm (class in   |
| q.operators.spin.SpinOperatorTerm |     cudaq.operators.spin)](ap     |
|         method)](api/language     | i/languages/python_api.html#cudaq |
| s/python_api.html#cudaq.operators | .operators.spin.SpinOperatorTerm) |
| .spin.SpinOperatorTerm.serialize) | -   [SPSA (class in               |
|     -   [(cudaq.SampleResult      |     cudaq                         |
|         me                        | .optimizers)](api/languages/pytho |
| thod)](api/languages/python_api.h | n_api.html#cudaq.optimizers.SPSA) |
| tml#cudaq.SampleResult.serialize) | -   [squeeze() (in module         |
| -   [set_noise() (in module       |     cudaq.operators.cust          |
|     cudaq)](api/languages         | om)](api/languages/python_api.htm |
| /python_api.html#cudaq.set_noise) | l#cudaq.operators.custom.squeeze) |
| -   [set_random_seed() (in module | -   [State (class in              |
|     cudaq)](api/languages/pytho   |     cudaq)](api/langu             |
| n_api.html#cudaq.set_random_seed) | ages/python_api.html#cudaq.State) |
| -   [set_target() (in module      | -   [step_size                    |
|     cudaq)](api/languages/        |     (cudaq.optimizers.Adam        |
| python_api.html#cudaq.set_target) |     propert                       |
| -   [SGD (class in                | y)](api/languages/python_api.html |
|     cuda                          | #cudaq.optimizers.Adam.step_size) |
| q.optimizers)](api/languages/pyth |     -   [(cudaq.optimizers.SGD    |
| on_api.html#cudaq.optimizers.SGD) |         proper                    |
|                                   | ty)](api/languages/python_api.htm |
|                                   | l#cudaq.optimizers.SGD.step_size) |
|                                   |     -   [(cudaq.optimizers.SPSA   |
|                                   |         propert                   |
|                                   | y)](api/languages/python_api.html |
|                                   | #cudaq.optimizers.SPSA.step_size) |
|                                   | -   [SuperOperator (class in      |
|                                   |     cudaq)](api/languages/pyt     |
|                                   | hon_api.html#cudaq.SuperOperator) |
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
| -   [to_dict() (cudaq.Resources   | -   [translate() (in module       |
|                                   |     cudaq)](api/languages         |
|    method)](api/languages/python_ | /python_api.html#cudaq.translate) |
| api.html#cudaq.Resources.to_dict) | -   [trim()                       |
| -   [to_json()                    |     (cu                           |
|     (                             | daq.operators.boson.BosonOperator |
| cudaq.gradients.CentralDifference |     method)](api/l                |
|     method)](api/la               | anguages/python_api.html#cudaq.op |
| nguages/python_api.html#cudaq.gra | erators.boson.BosonOperator.trim) |
| dients.CentralDifference.to_json) |     -   [(cudaq.                  |
|     -   [(                        | operators.fermion.FermionOperator |
| cudaq.gradients.ForwardDifference |         method)](api/langu        |
|         method)](api/la           | ages/python_api.html#cudaq.operat |
| nguages/python_api.html#cudaq.gra | ors.fermion.FermionOperator.trim) |
| dients.ForwardDifference.to_json) |     -                             |
|     -                             |  [(cudaq.operators.MatrixOperator |
|  [(cudaq.gradients.ParameterShift |         method)](                 |
|         method)](api              | api/languages/python_api.html#cud |
| /languages/python_api.html#cudaq. | aq.operators.MatrixOperator.trim) |
| gradients.ParameterShift.to_json) |     -   [(                        |
|     -   [(                        | cudaq.operators.spin.SpinOperator |
| cudaq.operators.spin.SpinOperator |         method)](api              |
|         method)](api/la           | /languages/python_api.html#cudaq. |
| nguages/python_api.html#cudaq.ope | operators.spin.SpinOperator.trim) |
| rators.spin.SpinOperator.to_json) | -   [type_to_str()                |
|     -   [(cuda                    |     (cudaq.PyKernelDecorator      |
| q.operators.spin.SpinOperatorTerm |     static                        |
|         method)](api/langua       |     method)](                     |
| ges/python_api.html#cudaq.operato | api/languages/python_api.html#cud |
| rs.spin.SpinOperatorTerm.to_json) | aq.PyKernelDecorator.type_to_str) |
|     -   [(cudaq.optimizers.Adam   |                                   |
|         met                       |                                   |
| hod)](api/languages/python_api.ht |                                   |
| ml#cudaq.optimizers.Adam.to_json) |                                   |
|     -   [(cudaq.optimizers.COBYLA |                                   |
|         metho                     |                                   |
| d)](api/languages/python_api.html |                                   |
| #cudaq.optimizers.COBYLA.to_json) |                                   |
|     -   [                         |                                   |
| (cudaq.optimizers.GradientDescent |                                   |
|         method)](api/l            |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| timizers.GradientDescent.to_json) |                                   |
|     -   [(cudaq.optimizers.LBFGS  |                                   |
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
