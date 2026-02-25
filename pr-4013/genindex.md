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
    -   [PTSBE API](api/ptsbe_api.html){.reference .internal}
        -   [Python API --- [`cudaq.ptsbe`{.docutils .literal
            .notranslate}]{.pre}](api/ptsbe_api.html#python-api-cudaq-ptsbe){.reference
            .internal}
            -   [Sampling
                Functions](api/ptsbe_api.html#sampling-functions){.reference
                .internal}
            -   [Result Type](api/ptsbe_api.html#result-type){.reference
                .internal}
            -   [Trajectory Sampling
                Strategies](api/ptsbe_api.html#trajectory-sampling-strategies){.reference
                .internal}
            -   [Shot Allocation
                Strategy](api/ptsbe_api.html#shot-allocation-strategy){.reference
                .internal}
            -   [Execution
                Data](api/ptsbe_api.html#execution-data){.reference
                .internal}
        -   [C++ API --- [`cudaq::ptsbe`{.docutils .literal
            .notranslate}]{.pre}](api/ptsbe_api.html#c-api-cudaq-ptsbe){.reference
            .internal}
            -   [Sampling Functions](api/ptsbe_api.html#id1){.reference
                .internal}
            -   [Options](api/ptsbe_api.html#options){.reference
                .internal}
            -   [Result Type](api/ptsbe_api.html#id2){.reference
                .internal}
            -   [Trajectory Sampling
                Strategies](api/ptsbe_api.html#id3){.reference
                .internal}
            -   [Shot Allocation
                Strategy](api/ptsbe_api.html#id4){.reference .internal}
            -   [Execution Data](api/ptsbe_api.html#id5){.reference
                .internal}
            -   [Trajectory and Selection
                Types](api/ptsbe_api.html#trajectory-and-selection-types){.reference
                .internal}
-   [User Guide](using/user_guide.html){.reference .internal}
    -   [Pre-Trajectory Sampling with Batch Execution
        (PTSBE)](using/ptsbe.html){.reference .internal}
        -   [Conceptual
            Overview](using/ptsbe.html#conceptual-overview){.reference
            .internal}
        -   [When to Use
            PTSBE](using/ptsbe.html#when-to-use-ptsbe){.reference
            .internal}
        -   [Quick Start](using/ptsbe.html#quick-start){.reference
            .internal}
        -   [Usage Tutorial](using/ptsbe.html#usage-tutorial){.reference
            .internal}
            -   [Controlling the Number of
                Trajectories](using/ptsbe.html#controlling-the-number-of-trajectories){.reference
                .internal}
            -   [Choosing a Trajectory Sampling
                Strategy](using/ptsbe.html#choosing-a-trajectory-sampling-strategy){.reference
                .internal}
            -   [Shot Allocation
                Strategies](using/ptsbe.html#shot-allocation-strategies){.reference
                .internal}
            -   [Inspecting Execution
                Data](using/ptsbe.html#inspecting-execution-data){.reference
                .internal}
            -   [Asynchronous
                Execution](using/ptsbe.html#asynchronous-execution){.reference
                .internal}
        -   [Trajectory vs Shot
            Trade-offs](using/ptsbe.html#trajectory-vs-shot-trade-offs){.reference
            .internal}
        -   [Backend
            Requirements](using/ptsbe.html#backend-requirements){.reference
            .internal}
        -   [Related
            Approaches](using/ptsbe.html#related-approaches){.reference
            .internal}
        -   [References](using/ptsbe.html#references){.reference
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
| -   [BaseIntegrator (class in     | -   [BosonOperator (class in      |
|                                   |     cudaq.operators.boson)](      |
| cudaq.dynamics.integrator)](api/l | api/languages/python_api.html#cud |
| anguages/python_api.html#cudaq.dy | aq.operators.boson.BosonOperator) |
| namics.integrator.BaseIntegrator) | -   [BosonOperatorElement (class  |
| -   [batch_size                   |     in                            |
|     (cudaq.optimizers.Adam        |                                   |
|     property                      |   cudaq.operators.boson)](api/lan |
| )](api/languages/python_api.html# | guages/python_api.html#cudaq.oper |
| cudaq.optimizers.Adam.batch_size) | ators.boson.BosonOperatorElement) |
|     -   [(cudaq.optimizers.SGD    | -   [BosonOperatorTerm (class in  |
|         propert                   |     cudaq.operators.boson)](api/  |
| y)](api/languages/python_api.html | languages/python_api.html#cudaq.o |
| #cudaq.optimizers.SGD.batch_size) | perators.boson.BosonOperatorTerm) |
| -   [beta1 (cudaq.optimizers.Adam | -   [broadcast() (in module       |
|     pro                           |     cudaq.mpi)](api/languages/pyt |
| perty)](api/languages/python_api. | hon_api.html#cudaq.mpi.broadcast) |
| html#cudaq.optimizers.Adam.beta1) | -   built-in function             |
| -   [beta2 (cudaq.optimizers.Adam |                                   |
|     pro                           |  -   [cudaq.ptsbe.sample()](api/p |
| perty)](api/languages/python_api. | tsbe_api.html#cudaq.ptsbe.sample) |
| html#cudaq.optimizers.Adam.beta2) |     -   [cudaq.                   |
| -   [beta_reduction()             | ptsbe.sample_async()](api/ptsbe_a |
|     (cudaq.PyKernelDecorator      | pi.html#cudaq.ptsbe.sample_async) |
|     method)](api                  |                                   |
| /languages/python_api.html#cudaq. |                                   |
| PyKernelDecorator.beta_reduction) |                                   |
| -   [BitFlipChannel (class in     |                                   |
|     cudaq)](api/languages/pyth    |                                   |
| on_api.html#cudaq.BitFlipChannel) |                                   |
+-----------------------------------+-----------------------------------+

## C {#C}

+-----------------------------------+-----------------------------------+
| -   [canonicalize()               | -   [cudaq::pauli2::pauli2 (C++   |
|     (cu                           |     function)](api/languages/cpp_ |
| daq.operators.boson.BosonOperator | api.html#_CPPv4N5cudaq6pauli26pau |
|     method)](api/languages        | li2ERKNSt6vectorIN5cudaq4realEEE) |
| /python_api.html#cudaq.operators. | -   [cudaq::phase_damping (C++    |
| boson.BosonOperator.canonicalize) |                                   |
|     -   [(cudaq.                  |  class)](api/languages/cpp_api.ht |
| operators.boson.BosonOperatorTerm | ml#_CPPv4N5cudaq13phase_dampingE) |
|                                   | -   [cud                          |
|        method)](api/languages/pyt | aq::phase_damping::num_parameters |
| hon_api.html#cudaq.operators.boso |     (C++                          |
| n.BosonOperatorTerm.canonicalize) |     member)](api/lan              |
|     -   [(cudaq.                  | guages/cpp_api.html#_CPPv4N5cudaq |
| operators.fermion.FermionOperator | 13phase_damping14num_parametersE) |
|                                   | -   [                             |
|        method)](api/languages/pyt | cudaq::phase_damping::num_targets |
| hon_api.html#cudaq.operators.ferm |     (C++                          |
| ion.FermionOperator.canonicalize) |     member)](api/                 |
|     -   [(cudaq.oper              | languages/cpp_api.html#_CPPv4N5cu |
| ators.fermion.FermionOperatorTerm | daq13phase_damping11num_targetsE) |
|                                   | -   [cudaq::phase_flip_channel    |
|    method)](api/languages/python_ |     (C++                          |
| api.html#cudaq.operators.fermion. |     clas                          |
| FermionOperatorTerm.canonicalize) | s)](api/languages/cpp_api.html#_C |
|     -                             | PPv4N5cudaq18phase_flip_channelE) |
|  [(cudaq.operators.MatrixOperator | -   [cudaq::p                     |
|         method)](api/lang         | hase_flip_channel::num_parameters |
| uages/python_api.html#cudaq.opera |     (C++                          |
| tors.MatrixOperator.canonicalize) |     member)](api/language         |
|     -   [(c                       | s/cpp_api.html#_CPPv4N5cudaq18pha |
| udaq.operators.MatrixOperatorTerm | se_flip_channel14num_parametersE) |
|         method)](api/language     | -   [cudaq                        |
| s/python_api.html#cudaq.operators | ::phase_flip_channel::num_targets |
| .MatrixOperatorTerm.canonicalize) |     (C++                          |
|     -   [(                        |     member)](api/langu            |
| cudaq.operators.spin.SpinOperator | ages/cpp_api.html#_CPPv4N5cudaq18 |
|         method)](api/languag      | phase_flip_channel11num_targetsE) |
| es/python_api.html#cudaq.operator | -   [cudaq::product_op (C++       |
| s.spin.SpinOperator.canonicalize) |                                   |
|     -   [(cuda                    |  class)](api/languages/cpp_api.ht |
| q.operators.spin.SpinOperatorTerm | ml#_CPPv4I0EN5cudaq10product_opE) |
|         method)](api/languages/p  | -   [cudaq::product_op::begin     |
| ython_api.html#cudaq.operators.sp |     (C++                          |
| in.SpinOperatorTerm.canonicalize) |     functio                       |
| -   [canonicalized() (in module   | n)](api/languages/cpp_api.html#_C |
|     cuda                          | PPv4NK5cudaq10product_op5beginEv) |
| q.boson)](api/languages/python_ap | -                                 |
| i.html#cudaq.boson.canonicalized) |  [cudaq::product_op::canonicalize |
|     -   [(in module               |     (C++                          |
|         cudaq.fe                  |     func                          |
| rmion)](api/languages/python_api. | tion)](api/languages/cpp_api.html |
| html#cudaq.fermion.canonicalized) | #_CPPv4N5cudaq10product_op12canon |
|     -   [(in module               | icalizeERKNSt3setINSt6size_tEEE), |
|                                   |     [\[1\]](api                   |
|        cudaq.operators.custom)](a | /languages/cpp_api.html#_CPPv4N5c |
| pi/languages/python_api.html#cuda | udaq10product_op12canonicalizeEv) |
| q.operators.custom.canonicalized) | -   [                             |
|     -   [(in module               | cudaq::product_op::const_iterator |
|         cu                        |     (C++                          |
| daq.spin)](api/languages/python_a |     struct)](api/                 |
| pi.html#cudaq.spin.canonicalized) | languages/cpp_api.html#_CPPv4N5cu |
| -   [captured_variables()         | daq10product_op14const_iteratorE) |
|     (cudaq.PyKernelDecorator      | -   [cudaq::product_o             |
|     method)](api/lan              | p::const_iterator::const_iterator |
| guages/python_api.html#cudaq.PyKe |     (C++                          |
| rnelDecorator.captured_variables) |     fu                            |
| -   [CentralDifference (class in  | nction)](api/languages/cpp_api.ht |
|     cudaq.gradients)              | ml#_CPPv4N5cudaq10product_op14con |
| ](api/languages/python_api.html#c | st_iterator14const_iteratorEPK10p |
| udaq.gradients.CentralDifference) | roduct_opI9HandlerTyENSt6size_tE) |
| -   [channel                      | -   [cudaq::produ                 |
|     (cudaq.ptsbe.TraceInstruction | ct_op::const_iterator::operator!= |
|     at                            |     (C++                          |
| tribute)](api/ptsbe_api.html#cuda |     fun                           |
| q.ptsbe.TraceInstruction.channel) | ction)](api/languages/cpp_api.htm |
| -   [circuit_location             | l#_CPPv4NK5cudaq10product_op14con |
|     (cudaq.ptsbe.KrausSelection   | st_iteratorneERK14const_iterator) |
|     attribute                     | -   [cudaq::produ                 |
| )](api/ptsbe_api.html#cudaq.ptsbe | ct_op::const_iterator::operator\* |
| .KrausSelection.circuit_location) |     (C++                          |
| -   [clear() (cudaq.Resources     |     function)](api/lang           |
|     method)](api/languages/pytho  | uages/cpp_api.html#_CPPv4NK5cudaq |
| n_api.html#cudaq.Resources.clear) | 10product_op14const_iteratormlEv) |
|     -   [(cudaq.SampleResult      | -   [cudaq::produ                 |
|                                   | ct_op::const_iterator::operator++ |
|   method)](api/languages/python_a |     (C++                          |
| pi.html#cudaq.SampleResult.clear) |     function)](api/lang           |
| -   [COBYLA (class in             | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     cudaq.o                       | 0product_op14const_iteratorppEi), |
| ptimizers)](api/languages/python_ |     [\[1\]](api/lan               |
| api.html#cudaq.optimizers.COBYLA) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [coefficient                  | 10product_op14const_iteratorppEv) |
|     (cudaq.                       | -   [cudaq::produc                |
| operators.boson.BosonOperatorTerm | t_op::const_iterator::operator\-- |
|     property)](api/languages/py   |     (C++                          |
| thon_api.html#cudaq.operators.bos |     function)](api/lang           |
| on.BosonOperatorTerm.coefficient) | uages/cpp_api.html#_CPPv4N5cudaq1 |
|     -   [(cudaq.oper              | 0product_op14const_iteratormmEi), |
| ators.fermion.FermionOperatorTerm |     [\[1\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|   property)](api/languages/python | 10product_op14const_iteratormmEv) |
| _api.html#cudaq.operators.fermion | -   [cudaq::produc                |
| .FermionOperatorTerm.coefficient) | t_op::const_iterator::operator-\> |
|     -   [(c                       |     (C++                          |
| udaq.operators.MatrixOperatorTerm |     function)](api/lan            |
|         property)](api/languag    | guages/cpp_api.html#_CPPv4N5cudaq |
| es/python_api.html#cudaq.operator | 10product_op14const_iteratorptEv) |
| s.MatrixOperatorTerm.coefficient) | -   [cudaq::produ                 |
|     -   [(cuda                    | ct_op::const_iterator::operator== |
| q.operators.spin.SpinOperatorTerm |     (C++                          |
|         property)](api/languages/ |     fun                           |
| python_api.html#cudaq.operators.s | ction)](api/languages/cpp_api.htm |
| pin.SpinOperatorTerm.coefficient) | l#_CPPv4NK5cudaq10product_op14con |
| -   [col_count                    | st_iteratoreqERK14const_iterator) |
|     (cudaq.KrausOperator          | -   [cudaq::product_op::degrees   |
|     prope                         |     (C++                          |
| rty)](api/languages/python_api.ht |     function)                     |
| ml#cudaq.KrausOperator.col_count) | ](api/languages/cpp_api.html#_CPP |
| -   [ComplexMatrix (class in      | v4NK5cudaq10product_op7degreesEv) |
|     cudaq)](api/languages/pyt     | -   [cudaq::product_op::dump (C++ |
| hon_api.html#cudaq.ComplexMatrix) |     functi                        |
| -   [compute()                    | on)](api/languages/cpp_api.html#_ |
|     (                             | CPPv4NK5cudaq10product_op4dumpEv) |
| cudaq.gradients.CentralDifference | -   [cudaq::product_op::end (C++  |
|     method)](api/la               |     funct                         |
| nguages/python_api.html#cudaq.gra | ion)](api/languages/cpp_api.html# |
| dients.CentralDifference.compute) | _CPPv4NK5cudaq10product_op3endEv) |
|     -   [(                        | -   [c                            |
| cudaq.gradients.ForwardDifference | udaq::product_op::get_coefficient |
|         method)](api/la           |     (C++                          |
| nguages/python_api.html#cudaq.gra |     function)](api/lan            |
| dients.ForwardDifference.compute) | guages/cpp_api.html#_CPPv4NK5cuda |
|     -                             | q10product_op15get_coefficientEv) |
|  [(cudaq.gradients.ParameterShift | -                                 |
|         method)](api              |   [cudaq::product_op::get_term_id |
| /languages/python_api.html#cudaq. |     (C++                          |
| gradients.ParameterShift.compute) |     function)](api                |
| -   [const()                      | /languages/cpp_api.html#_CPPv4NK5 |
|                                   | cudaq10product_op11get_term_idEv) |
|   (cudaq.operators.ScalarOperator | -                                 |
|     class                         |   [cudaq::product_op::is_identity |
|     method)](a                    |     (C++                          |
| pi/languages/python_api.html#cuda |     function)](api                |
| q.operators.ScalarOperator.const) | /languages/cpp_api.html#_CPPv4NK5 |
| -   [controls                     | cudaq10product_op11is_identityEv) |
|     (cudaq.ptsbe.TraceInstruction | -   [cudaq::product_op::num_ops   |
|     att                           |     (C++                          |
| ribute)](api/ptsbe_api.html#cudaq |     function)                     |
| .ptsbe.TraceInstruction.controls) | ](api/languages/cpp_api.html#_CPP |
| -   [copy()                       | v4NK5cudaq10product_op7num_opsEv) |
|     (cu                           | -                                 |
| daq.operators.boson.BosonOperator |    [cudaq::product_op::operator\* |
|     method)](api/l                |     (C++                          |
| anguages/python_api.html#cudaq.op |     function)](api/languages/     |
| erators.boson.BosonOperator.copy) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(cudaq.                  | oduct_opmlE10product_opI1TERK15sc |
| operators.boson.BosonOperatorTerm | alar_operatorRK10product_opI1TE), |
|         method)](api/langu        |     [\[1\]](api/languages/        |
| ages/python_api.html#cudaq.operat | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| ors.boson.BosonOperatorTerm.copy) | oduct_opmlE10product_opI1TERK15sc |
|     -   [(cudaq.                  | alar_operatorRR10product_opI1TE), |
| operators.fermion.FermionOperator |     [\[2\]](api/languages/        |
|         method)](api/langu        | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| ages/python_api.html#cudaq.operat | oduct_opmlE10product_opI1TERR15sc |
| ors.fermion.FermionOperator.copy) | alar_operatorRK10product_opI1TE), |
|     -   [(cudaq.oper              |     [\[3\]](api/languages/        |
| ators.fermion.FermionOperatorTerm | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|         method)](api/languages    | oduct_opmlE10product_opI1TERR15sc |
| /python_api.html#cudaq.operators. | alar_operatorRR10product_opI1TE), |
| fermion.FermionOperatorTerm.copy) |     [\[4\]](api/                  |
|     -                             | languages/cpp_api.html#_CPPv4I0EN |
|  [(cudaq.operators.MatrixOperator | 5cudaq10product_opmlE6sum_opI1TER |
|         method)](                 | K15scalar_operatorRK6sum_opI1TE), |
| api/languages/python_api.html#cud |     [\[5\]](api/                  |
| aq.operators.MatrixOperator.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(c                       | 5cudaq10product_opmlE6sum_opI1TER |
| udaq.operators.MatrixOperatorTerm | K15scalar_operatorRR6sum_opI1TE), |
|         method)](api/             |     [\[6\]](api/                  |
| languages/python_api.html#cudaq.o | languages/cpp_api.html#_CPPv4I0EN |
| perators.MatrixOperatorTerm.copy) | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [(                        | R15scalar_operatorRK6sum_opI1TE), |
| cudaq.operators.spin.SpinOperator |     [\[7\]](api/                  |
|         method)](api              | languages/cpp_api.html#_CPPv4I0EN |
| /languages/python_api.html#cudaq. | 5cudaq10product_opmlE6sum_opI1TER |
| operators.spin.SpinOperator.copy) | R15scalar_operatorRR6sum_opI1TE), |
|     -   [(cuda                    |     [\[8\]](api/languages         |
| q.operators.spin.SpinOperatorTerm | /cpp_api.html#_CPPv4NK5cudaq10pro |
|         method)](api/lan          | duct_opmlERK6sum_opI9HandlerTyE), |
| guages/python_api.html#cudaq.oper |     [\[9\]](api/languages/cpp_a   |
| ators.spin.SpinOperatorTerm.copy) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [count() (cudaq.Resources     | opmlERK10product_opI9HandlerTyE), |
|     method)](api/languages/pytho  |     [\[10\]](api/language         |
| n_api.html#cudaq.Resources.count) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     -   [(cudaq.SampleResult      | roduct_opmlERK15scalar_operator), |
|                                   |     [\[11\]](api/languages/cpp_a  |
|   method)](api/languages/python_a | pi.html#_CPPv4NKR5cudaq10product_ |
| pi.html#cudaq.SampleResult.count) | opmlERR10product_opI9HandlerTyE), |
| -   [count_controls()             |     [\[12\]](api/language         |
|     (cudaq.Resources              | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     meth                          | roduct_opmlERR15scalar_operator), |
| od)](api/languages/python_api.htm |     [\[13\]](api/languages/cpp_   |
| l#cudaq.Resources.count_controls) | api.html#_CPPv4NO5cudaq10product_ |
| -   [count_errors()               | opmlERK10product_opI9HandlerTyE), |
|     (cudaq.ptsbe.KrausTrajectory  |     [\[14\]](api/languag          |
|     met                           | es/cpp_api.html#_CPPv4NO5cudaq10p |
| hod)](api/ptsbe_api.html#cudaq.pt | roduct_opmlERK15scalar_operator), |
| sbe.KrausTrajectory.count_errors) |     [\[15\]](api/languages/cpp_   |
| -   [count_instructions()         | api.html#_CPPv4NO5cudaq10product_ |
|                                   | opmlERR10product_opI9HandlerTyE), |
|   (cudaq.ptsbe.PTSBEExecutionData |     [\[16\]](api/langua           |
|     method)](api                  | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| /ptsbe_api.html#cudaq.ptsbe.PTSBE | product_opmlERR15scalar_operator) |
| ExecutionData.count_instructions) | -                                 |
| -   [counts()                     |   [cudaq::product_op::operator\*= |
|     (cudaq.ObserveResult          |     (C++                          |
|                                   |     function)](api/languages/cpp  |
| method)](api/languages/python_api | _api.html#_CPPv4N5cudaq10product_ |
| .html#cudaq.ObserveResult.counts) | opmLERK10product_opI9HandlerTyE), |
| -   [create() (in module          |     [\[1\]](api/langua            |
|                                   | ges/cpp_api.html#_CPPv4N5cudaq10p |
|    cudaq.boson)](api/languages/py | roduct_opmLERK15scalar_operator), |
| thon_api.html#cudaq.boson.create) |     [\[2\]](api/languages/cp      |
|     -   [(in module               | p_api.html#_CPPv4N5cudaq10product |
|         c                         | _opmLERR10product_opI9HandlerTyE) |
| udaq.fermion)](api/languages/pyth | -   [cudaq::product_op::operator+ |
| on_api.html#cudaq.fermion.create) |     (C++                          |
| -   [csr_spmatrix (C++            |     function)](api/langu          |
|     type)](api/languages/c        | ages/cpp_api.html#_CPPv4I0EN5cuda |
| pp_api.html#_CPPv412csr_spmatrix) | q10product_opplE6sum_opI1TERK15sc |
| -   cudaq                         | alar_operatorRK10product_opI1TE), |
|     -   [module](api/langua       |     [\[1\]](api/                  |
| ges/python_api.html#module-cudaq) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq (C++                   | 5cudaq10product_opplE6sum_opI1TER |
|     type)](api/lan                | K15scalar_operatorRK6sum_opI1TE), |
| guages/cpp_api.html#_CPPv45cudaq) |     [\[2\]](api/langu             |
| -   [cudaq.apply_noise() (in      | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     module                        | q10product_opplE6sum_opI1TERK15sc |
|     cudaq)](api/languages/python_ | alar_operatorRR10product_opI1TE), |
| api.html#cudaq.cudaq.apply_noise) |     [\[3\]](api/                  |
| -   cudaq.boson                   | languages/cpp_api.html#_CPPv4I0EN |
|     -   [module](api/languages/py | 5cudaq10product_opplE6sum_opI1TER |
| thon_api.html#module-cudaq.boson) | K15scalar_operatorRR6sum_opI1TE), |
| -   cudaq.fermion                 |     [\[4\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|   -   [module](api/languages/pyth | q10product_opplE6sum_opI1TERR15sc |
| on_api.html#module-cudaq.fermion) | alar_operatorRK10product_opI1TE), |
| -   cudaq.operators.custom        |     [\[5\]](api/                  |
|     -   [mo                       | languages/cpp_api.html#_CPPv4I0EN |
| dule](api/languages/python_api.ht | 5cudaq10product_opplE6sum_opI1TER |
| ml#module-cudaq.operators.custom) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq.                       |     [\[6\]](api/langu             |
| ptsbe.ConditionalSamplingStrategy | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (built-in                     | q10product_opplE6sum_opI1TERR15sc |
|     c                             | alar_operatorRR10product_opI1TE), |
| lass)](api/ptsbe_api.html#cudaq.p |     [\[7\]](api/                  |
| tsbe.ConditionalSamplingStrategy) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq                        | 5cudaq10product_opplE6sum_opI1TER |
| .ptsbe.ExhaustiveSamplingStrategy | R15scalar_operatorRR6sum_opI1TE), |
|     (built-in                     |     [\[8\]](api/languages/cpp_a   |
|                                   | pi.html#_CPPv4NKR5cudaq10product_ |
| class)](api/ptsbe_api.html#cudaq. | opplERK10product_opI9HandlerTyE), |
| ptsbe.ExhaustiveSamplingStrategy) |     [\[9\]](api/language          |
| -   [cudaq.ptsbe.KrausSelection   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (built-in                     | roduct_opplERK15scalar_operator), |
|     class)](api/ptsbe_api         |     [\[10\]](api/languages/       |
| .html#cudaq.ptsbe.KrausSelection) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq.ptsbe.KrausTrajectory  | duct_opplERK6sum_opI9HandlerTyE), |
|     (built-in                     |     [\[11\]](api/languages/cpp_a  |
|     class)](api/ptsbe_api.        | pi.html#_CPPv4NKR5cudaq10product_ |
| html#cudaq.ptsbe.KrausTrajectory) | opplERR10product_opI9HandlerTyE), |
| -   [cu                           |     [\[12\]](api/language         |
| daq.ptsbe.OrderedSamplingStrategy | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (built-in                     | roduct_opplERR15scalar_operator), |
|                                   |     [\[13\]](api/languages/       |
|    class)](api/ptsbe_api.html#cud | cpp_api.html#_CPPv4NKR5cudaq10pro |
| aq.ptsbe.OrderedSamplingStrategy) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [cudaq.pt                     |     [\[                           |
| sbe.ProbabilisticSamplingStrategy | 14\]](api/languages/cpp_api.html# |
|     (built-in                     | _CPPv4NKR5cudaq10product_opplEv), |
|     cla                           |     [\[15\]](api/languages/cpp_   |
| ss)](api/ptsbe_api.html#cudaq.pts | api.html#_CPPv4NO5cudaq10product_ |
| be.ProbabilisticSamplingStrategy) | opplERK10product_opI9HandlerTyE), |
| -                                 |     [\[16\]](api/languag          |
|   [cudaq.ptsbe.PTSBEExecutionData | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (built-in                     | roduct_opplERK15scalar_operator), |
|     class)](api/ptsbe_api.htm     |     [\[17\]](api/languages        |
| l#cudaq.ptsbe.PTSBEExecutionData) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -                                 | duct_opplERK6sum_opI9HandlerTyE), |
|    [cudaq.ptsbe.PTSBESampleResult |     [\[18\]](api/languages/cpp_   |
|     (built-in                     | api.html#_CPPv4NO5cudaq10product_ |
|     class)](api/ptsbe_api.ht      | opplERR10product_opI9HandlerTyE), |
| ml#cudaq.ptsbe.PTSBESampleResult) |     [\[19\]](api/languag          |
| -                                 | es/cpp_api.html#_CPPv4NO5cudaq10p |
|  [cudaq.ptsbe.PTSSamplingStrategy | roduct_opplERR15scalar_operator), |
|     (built-in                     |     [\[20\]](api/languages        |
|     class)](api/ptsbe_api.html    | /cpp_api.html#_CPPv4NO5cudaq10pro |
| #cudaq.ptsbe.PTSSamplingStrategy) | duct_opplERR6sum_opI9HandlerTyE), |
| -   cudaq.ptsbe.sample()          |     [                             |
|     -   [built-in                 | \[21\]](api/languages/cpp_api.htm |
|         function](api/p           | l#_CPPv4NO5cudaq10product_opplEv) |
| tsbe_api.html#cudaq.ptsbe.sample) | -   [cudaq::product_op::operator- |
| -   cudaq.ptsbe.sample_async()    |     (C++                          |
|     -   [built-in                 |     function)](api/langu          |
|         function](api/ptsbe_a     | ages/cpp_api.html#_CPPv4I0EN5cuda |
| pi.html#cudaq.ptsbe.sample_async) | q10product_opmiE6sum_opI1TERK15sc |
| -   [c                            | alar_operatorRK10product_opI1TE), |
| udaq.ptsbe.ShotAllocationStrategy |     [\[1\]](api/                  |
|     (built-in                     | languages/cpp_api.html#_CPPv4I0EN |
|     class)](api/ptsbe_api.html#cu | 5cudaq10product_opmiE6sum_opI1TER |
| daq.ptsbe.ShotAllocationStrategy) | K15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq.                       |     [\[2\]](api/langu             |
| ptsbe.ShotAllocationStrategy.Type | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (built-in                     | q10product_opmiE6sum_opI1TERK15sc |
|     c                             | alar_operatorRR10product_opI1TE), |
| lass)](api/ptsbe_api.html#cudaq.p |     [\[3\]](api/                  |
| tsbe.ShotAllocationStrategy.Type) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq.ptsbe.TraceInstruction | 5cudaq10product_opmiE6sum_opI1TER |
|     (built-in                     | K15scalar_operatorRR6sum_opI1TE), |
|     class)](api/ptsbe_api.h       |     [\[4\]](api/langu             |
| tml#cudaq.ptsbe.TraceInstruction) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -                                 | q10product_opmiE6sum_opI1TERR15sc |
| [cudaq.ptsbe.TraceInstructionType | alar_operatorRK10product_opI1TE), |
|     (built-in                     |     [\[5\]](api/                  |
|     class)](api/ptsbe_api.html#   | languages/cpp_api.html#_CPPv4I0EN |
| cudaq.ptsbe.TraceInstructionType) | 5cudaq10product_opmiE6sum_opI1TER |
| -   cudaq.spin                    | R15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/languages/p  |     [\[6\]](api/langu             |
| ython_api.html#module-cudaq.spin) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::amplitude_damping     | q10product_opmiE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     cla                           |     [\[7\]](api/                  |
| ss)](api/languages/cpp_api.html#_ | languages/cpp_api.html#_CPPv4I0EN |
| CPPv4N5cudaq17amplitude_dampingE) | 5cudaq10product_opmiE6sum_opI1TER |
| -                                 | R15scalar_operatorRR6sum_opI1TE), |
| [cudaq::amplitude_damping_channel |     [\[8\]](api/languages/cpp_a   |
|     (C++                          | pi.html#_CPPv4NKR5cudaq10product_ |
|     class)](api                   | opmiERK10product_opI9HandlerTyE), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[9\]](api/language          |
| udaq25amplitude_damping_channelE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   [cudaq::amplitud              | roduct_opmiERK15scalar_operator), |
| e_damping_channel::num_parameters |     [\[10\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     member)](api/languages/cpp_a  | duct_opmiERK6sum_opI9HandlerTyE), |
| pi.html#_CPPv4N5cudaq25amplitude_ |     [\[11\]](api/languages/cpp_a  |
| damping_channel14num_parametersE) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [cudaq::ampli                 | opmiERR10product_opI9HandlerTyE), |
| tude_damping_channel::num_targets |     [\[12\]](api/language         |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     member)](api/languages/cp     | roduct_opmiERR15scalar_operator), |
| p_api.html#_CPPv4N5cudaq25amplitu |     [\[13\]](api/languages/       |
| de_damping_channel11num_targetsE) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq::AnalogRemoteRESTQPU   | duct_opmiERR6sum_opI9HandlerTyE), |
|     (C++                          |     [\[                           |
|     class                         | 14\]](api/languages/cpp_api.html# |
| )](api/languages/cpp_api.html#_CP | _CPPv4NKR5cudaq10product_opmiEv), |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::apply_noise (C++      | api.html#_CPPv4NO5cudaq10product_ |
|     function)](api/               | opmiERK10product_opI9HandlerTyE), |
| languages/cpp_api.html#_CPPv4I0Dp |     [\[16\]](api/languag          |
| EN5cudaq11apply_noiseEvDpRR4Args) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::async_result (C++     | roduct_opmiERK15scalar_operator), |
|     c                             |     [\[17\]](api/languages        |
| lass)](api/languages/cpp_api.html | /cpp_api.html#_CPPv4NO5cudaq10pro |
| #_CPPv4I0EN5cudaq12async_resultE) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [cudaq::async_result::get     |     [\[18\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     functi                        | opmiERR10product_opI9HandlerTyE), |
| on)](api/languages/cpp_api.html#_ |     [\[19\]](api/languag          |
| CPPv4N5cudaq12async_result3getEv) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::async_sample_result   | roduct_opmiERR15scalar_operator), |
|     (C++                          |     [\[20\]](api/languages        |
|     type                          | /cpp_api.html#_CPPv4NO5cudaq10pro |
| )](api/languages/cpp_api.html#_CP | duct_opmiERR6sum_opI9HandlerTyE), |
| Pv4N5cudaq19async_sample_resultE) |     [                             |
| -   [cudaq::BaseRemoteRESTQPU     | \[21\]](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NO5cudaq10product_opmiEv) |
|     cla                           | -   [cudaq::product_op::operator/ |
| ss)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     function)](api/language       |
| -                                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|    [cudaq::BaseRemoteSimulatorQPU | roduct_opdvERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/language          |
|     class)](                      | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| api/languages/cpp_api.html#_CPPv4 | roduct_opdvERR15scalar_operator), |
| N5cudaq22BaseRemoteSimulatorQPUE) |     [\[2\]](api/languag           |
| -   [cudaq::bit_flip_channel (C++ | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     cl                            | roduct_opdvERK15scalar_operator), |
| ass)](api/languages/cpp_api.html# |     [\[3\]](api/langua            |
| _CPPv4N5cudaq16bit_flip_channelE) | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| -   [cudaq:                       | product_opdvERR15scalar_operator) |
| :bit_flip_channel::num_parameters | -                                 |
|     (C++                          |    [cudaq::product_op::operator/= |
|     member)](api/langua           |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq16b |     function)](api/langu          |
| it_flip_channel14num_parametersE) | ages/cpp_api.html#_CPPv4N5cudaq10 |
| -   [cud                          | product_opdVERK15scalar_operator) |
| aq::bit_flip_channel::num_targets | -   [cudaq::product_op::operator= |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/la             |
| guages/cpp_api.html#_CPPv4N5cudaq | nguages/cpp_api.html#_CPPv4I0_NSt |
| 16bit_flip_channel11num_targetsE) | 11enable_if_tIXaantNSt7is_sameI1T |
| -   [cudaq::boson_handler (C++    | 9HandlerTyE5valueENSt16is_constru |
|                                   | ctibleI9HandlerTy1TE5valueEEbEEEN |
|  class)](api/languages/cpp_api.ht | 5cudaq10product_opaSER10product_o |
| ml#_CPPv4N5cudaq13boson_handlerE) | pI9HandlerTyERK10product_opI1TE), |
| -   [cudaq::boson_op (C++         |     [\[1\]](api/languages/cpp     |
|     type)](api/languages/cpp_     | _api.html#_CPPv4N5cudaq10product_ |
| api.html#_CPPv4N5cudaq8boson_opE) | opaSERK10product_opI9HandlerTyE), |
| -   [cudaq::boson_op_term (C++    |     [\[2\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4N5cudaq10product |
|   type)](api/languages/cpp_api.ht | _opaSERR10product_opI9HandlerTyE) |
| ml#_CPPv4N5cudaq13boson_op_termE) | -                                 |
| -   [cudaq::CodeGenConfig (C++    |    [cudaq::product_op::operator== |
|                                   |     (C++                          |
| struct)](api/languages/cpp_api.ht |     function)](api/languages/cpp  |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | _api.html#_CPPv4NK5cudaq10product |
| -   [cudaq::commutation_relations | _opeqERK10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|     struct)]                      |  [cudaq::product_op::operator\[\] |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq21commutation_relationsE) |     function)](ap                 |
| -   [cudaq::complex (C++          | i/languages/cpp_api.html#_CPPv4NK |
|     type)](api/languages/cpp      | 5cudaq10product_opixENSt6size_tE) |
| _api.html#_CPPv4N5cudaq7complexE) | -                                 |
| -   [cudaq::complex_matrix (C++   |    [cudaq::product_op::product_op |
|                                   |     (C++                          |
| class)](api/languages/cpp_api.htm |     function)](api/languages/c    |
| l#_CPPv4N5cudaq14complex_matrixE) | pp_api.html#_CPPv4I0_NSt11enable_ |
| -                                 | if_tIXaaNSt7is_sameI9HandlerTy14m |
|   [cudaq::complex_matrix::adjoint | atrix_handlerE5valueEaantNSt7is_s |
|     (C++                          | ameI1T9HandlerTyE5valueENSt16is_c |
|     function)](a                  | onstructibleI9HandlerTy1TE5valueE |
| pi/languages/cpp_api.html#_CPPv4N | EbEEEN5cudaq10product_op10product |
| 5cudaq14complex_matrix7adjointEv) | _opERK10product_opI1TERKN14matrix |
| -   [cudaq::                      | _handler20commutation_behaviorE), |
| complex_matrix::diagonal_elements |                                   |
|     (C++                          |  [\[1\]](api/languages/cpp_api.ht |
|     function)](api/languages      | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| /cpp_api.html#_CPPv4NK5cudaq14com | tNSt7is_sameI1T9HandlerTyE5valueE |
| plex_matrix17diagonal_elementsEi) | NSt16is_constructibleI9HandlerTy1 |
| -   [cudaq::complex_matrix::dump  | TE5valueEEbEEEN5cudaq10product_op |
|     (C++                          | 10product_opERK10product_opI1TE), |
|     function)](api/language       |                                   |
| s/cpp_api.html#_CPPv4NK5cudaq14co |   [\[2\]](api/languages/cpp_api.h |
| mplex_matrix4dumpERNSt7ostreamE), | tml#_CPPv4N5cudaq10product_op10pr |
|     [\[1\]]                       | oduct_opENSt6size_tENSt6size_tE), |
| (api/languages/cpp_api.html#_CPPv |     [\[3\]](api/languages/cp      |
| 4NK5cudaq14complex_matrix4dumpEv) | p_api.html#_CPPv4N5cudaq10product |
| -   [c                            | _op10product_opENSt7complexIdEE), |
| udaq::complex_matrix::eigenvalues |     [\[4\]](api/l                 |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     function)](api/lan            | aq10product_op10product_opERK10pr |
| guages/cpp_api.html#_CPPv4NK5cuda | oduct_opI9HandlerTyENSt6size_tE), |
| q14complex_matrix11eigenvaluesEv) |     [\[5\]](api/l                 |
| -   [cu                           | anguages/cpp_api.html#_CPPv4N5cud |
| daq::complex_matrix::eigenvectors | aq10product_op10product_opERR10pr |
|     (C++                          | oduct_opI9HandlerTyENSt6size_tE), |
|     function)](api/lang           |     [\[6\]](api/languages         |
| uages/cpp_api.html#_CPPv4NK5cudaq | /cpp_api.html#_CPPv4N5cudaq10prod |
| 14complex_matrix12eigenvectorsEv) | uct_op10product_opERR9HandlerTy), |
| -   [c                            |     [\[7\]](ap                    |
| udaq::complex_matrix::exponential | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq10product_op10product_opEd), |
|     function)](api/la             |     [\[8\]](a                     |
| nguages/cpp_api.html#_CPPv4N5cuda | pi/languages/cpp_api.html#_CPPv4N |
| q14complex_matrix11exponentialEv) | 5cudaq10product_op10product_opEv) |
| -                                 | -   [cuda                         |
|  [cudaq::complex_matrix::identity | q::product_op::to_diagonal_matrix |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/               |
| /cpp_api.html#_CPPv4N5cudaq14comp | languages/cpp_api.html#_CPPv4NK5c |
| lex_matrix8identityEKNSt6size_tE) | udaq10product_op18to_diagonal_mat |
| -                                 | rixENSt13unordered_mapINSt6size_t |
| [cudaq::complex_matrix::kronecker | ENSt7int64_tEEERKNSt13unordered_m |
|     (C++                          | apINSt6stringENSt7complexIdEEEEb) |
|     function)](api/lang           | -   [cudaq::product_op::to_matrix |
| uages/cpp_api.html#_CPPv4I00EN5cu |     (C++                          |
| daq14complex_matrix9kroneckerE14c |     funct                         |
| omplex_matrix8Iterable8Iterable), | ion)](api/languages/cpp_api.html# |
|     [\[1\]](api/l                 | _CPPv4NK5cudaq10product_op9to_mat |
| anguages/cpp_api.html#_CPPv4N5cud | rixENSt13unordered_mapINSt6size_t |
| aq14complex_matrix9kroneckerERK14 | ENSt7int64_tEEERKNSt13unordered_m |
| complex_matrixRK14complex_matrix) | apINSt6stringENSt7complexIdEEEEb) |
| -   [cudaq::c                     | -   [cu                           |
| omplex_matrix::minimal_eigenvalue | daq::product_op::to_sparse_matrix |
|     (C++                          |     (C++                          |
|     function)](api/languages/     |     function)](ap                 |
| cpp_api.html#_CPPv4NK5cudaq14comp | i/languages/cpp_api.html#_CPPv4NK |
| lex_matrix18minimal_eigenvalueEv) | 5cudaq10product_op16to_sparse_mat |
| -   [                             | rixENSt13unordered_mapINSt6size_t |
| cudaq::complex_matrix::operator() | ENSt7int64_tEEERKNSt13unordered_m |
|     (C++                          | apINSt6stringENSt7complexIdEEEEb) |
|     function)](api/languages/cpp  | -   [cudaq::product_op::to_string |
| _api.html#_CPPv4N5cudaq14complex_ |     (C++                          |
| matrixclENSt6size_tENSt6size_tE), |     function)](                   |
|     [\[1\]](api/languages/cpp     | api/languages/cpp_api.html#_CPPv4 |
| _api.html#_CPPv4NK5cudaq14complex | NK5cudaq10product_op9to_stringEv) |
| _matrixclENSt6size_tENSt6size_tE) | -                                 |
| -   [                             |  [cudaq::product_op::\~product_op |
| cudaq::complex_matrix::operator\* |     (C++                          |
|     (C++                          |     fu                            |
|     function)](api/langua         | nction)](api/languages/cpp_api.ht |
| ges/cpp_api.html#_CPPv4N5cudaq14c | ml#_CPPv4N5cudaq10product_opD0Ev) |
| omplex_matrixmlEN14complex_matrix | -   [cudaq::p                     |
| 10value_typeERK14complex_matrix), | tsbe::ConditionalSamplingStrategy |
|     [\[1\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     class)](api                   |
| v4N5cudaq14complex_matrixmlERK14c | /ptsbe_api.html#_CPPv4N5cudaq5pts |
| omplex_matrixRK14complex_matrix), | be27ConditionalSamplingStrategyE) |
|                                   | -   [cuda                         |
|  [\[2\]](api/languages/cpp_api.ht | q::ptsbe::ConditionalSamplingStra |
| ml#_CPPv4N5cudaq14complex_matrixm | tegy::ConditionalSamplingStrategy |
| lERK14complex_matrixRKNSt6vectorI |     (C++                          |
| N14complex_matrix10value_typeEEE) |     function)](                   |
| -                                 | api/ptsbe_api.html#_CPPv4N5cudaq5 |
| [cudaq::complex_matrix::operator+ | ptsbe27ConditionalSamplingStrateg |
|     (C++                          | y27ConditionalSamplingStrategyE19 |
|     function                      | TrajectoryPredicateNSt8uint64_tE) |
| )](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq14complex_matrixplERK14 |    [cudaq::ptsbe::ConditionalSamp |
| complex_matrixRK14complex_matrix) | lingStrategy::TrajectoryPredicate |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::operator- |                                   |
|     (C++                          |   type)](api/ptsbe_api.html#_CPPv |
|     function                      | 4N5cudaq5ptsbe27ConditionalSampli |
| )](api/languages/cpp_api.html#_CP | ngStrategy19TrajectoryPredicateE) |
| Pv4N5cudaq14complex_matrixmiERK14 | -   [cudaq::                      |
| complex_matrixRK14complex_matrix) | ptsbe::ExhaustiveSamplingStrategy |
| -   [cu                           |     (C++                          |
| daq::complex_matrix::operator\[\] |     class)](ap                    |
|     (C++                          | i/ptsbe_api.html#_CPPv4N5cudaq5pt |
|                                   | sbe26ExhaustiveSamplingStrategyE) |
|  function)](api/languages/cpp_api | -   [cuda                         |
| .html#_CPPv4N5cudaq14complex_matr | q::ptsbe::OrderedSamplingStrategy |
| ixixERKNSt6vectorINSt6size_tEEE), |     (C++                          |
|     [\[1\]](api/languages/cpp_api |     class)]                       |
| .html#_CPPv4NK5cudaq14complex_mat | (api/ptsbe_api.html#_CPPv4N5cudaq |
| rixixERKNSt6vectorINSt6size_tEEE) | 5ptsbe23OrderedSamplingStrategyE) |
| -   [cudaq::complex_matrix::power | -   [cudaq::pts                   |
|     (C++                          | be::ProbabilisticSamplingStrategy |
|     function)]                    |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     class)](api/p                 |
| 4N5cudaq14complex_matrix5powerEi) | tsbe_api.html#_CPPv4N5cudaq5ptsbe |
| -                                 | 29ProbabilisticSamplingStrategyE) |
|  [cudaq::complex_matrix::set_zero | -   [cudaq::p                     |
|     (C++                          | tsbe::ProbabilisticSamplingStrate |
|     function)](ap                 | gy::ProbabilisticSamplingStrategy |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq14complex_matrix8set_zeroEv) |     function)](api/ptsbe_api.ht   |
| -                                 | ml#_CPPv4N5cudaq5ptsbe29Probabili |
| [cudaq::complex_matrix::to_string | sticSamplingStrategy29Probabilist |
|     (C++                          | icSamplingStrategyENSt8uint64_tE) |
|     function)](api/               | -                                 |
| languages/cpp_api.html#_CPPv4NK5c | [cudaq::ptsbe::PTSBEExecutionData |
| udaq14complex_matrix9to_stringEv) |     (C++                          |
| -   [                             |     str                           |
| cudaq::complex_matrix::value_type | uct)](api/ptsbe_api.html#_CPPv4N5 |
|     (C++                          | cudaq5ptsbe18PTSBEExecutionDataE) |
|     type)](api/                   | -   [cudaq::ptsbe::PTSBE          |
| languages/cpp_api.html#_CPPv4N5cu | ExecutionData::count_instructions |
| daq14complex_matrix10value_typeE) |     (C++                          |
| -   [cudaq::contrib (C++          |     function                      |
|     type)](api/languages/cpp      | )](api/ptsbe_api.html#_CPPv4NK5cu |
| _api.html#_CPPv4N5cudaq7contribE) | daq5ptsbe18PTSBEExecutionData18co |
| -   [cudaq::contrib::draw (C++    | unt_instructionsE20TraceInstructi |
|     function)                     | onTypeNSt8optionalINSt6stringEEE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::ptsbe::P              |
| v4I0DpEN5cudaq7contrib4drawENSt6s | TSBEExecutionData::get_trajectory |
| tringERR13QuantumKernelDpRR4Args) |     (C++                          |
| -                                 |                                   |
| [cudaq::contrib::get_unitary_cmat | function)](api/ptsbe_api.html#_CP |
|     (C++                          | Pv4NK5cudaq5ptsbe18PTSBEExecution |
|     function)](api/languages/cp   | Data14get_trajectoryENSt6size_tE) |
| p_api.html#_CPPv4I0DpEN5cudaq7con | -   [cudaq::ptsbe:                |
| trib16get_unitary_cmatE14complex_ | :PTSBEExecutionData::instructions |
| matrixRR13QuantumKernelDpRR4Args) |     (C++                          |
| -   [cudaq::CusvState (C++        |     member)](api/ptsb             |
|                                   | e_api.html#_CPPv4N5cudaq5ptsbe18P |
|    class)](api/languages/cpp_api. | TSBEExecutionData12instructionsE) |
| html#_CPPv4I0EN5cudaq9CusvStateE) | -   [cudaq::ptsbe:                |
| -   [cudaq::depolarization1 (C++  | :PTSBEExecutionData::trajectories |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     member)](api/ptsb             |
| #_CPPv4N5cudaq15depolarization1E) | e_api.html#_CPPv4N5cudaq5ptsbe18P |
| -   [cudaq::depolarization2 (C++  | TSBEExecutionData12trajectoriesE) |
|     c                             | -   [cudaq::ptsbe::PTSBEOptions   |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15depolarization2E) |                                   |
| -   [cudaq:                       |    struct)](api/ptsbe_api.html#_C |
| :depolarization2::depolarization2 | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
|     (C++                          | -   [cudaq::ptsb                  |
|     function)](api/languages/cp   | e::PTSBEOptions::max_trajectories |
| p_api.html#_CPPv4N5cudaq15depolar |     (C++                          |
| ization215depolarization2EK4real) |     member)](api/pt               |
| -   [cudaq                        | sbe_api.html#_CPPv4N5cudaq5ptsbe1 |
| ::depolarization2::num_parameters | 2PTSBEOptions16max_trajectoriesE) |
|     (C++                          | -   [cudaq::ptsbe::PT             |
|     member)](api/langu            | SBEOptions::return_execution_data |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     (C++                          |
| depolarization214num_parametersE) |     member)](api/ptsbe_a          |
| -   [cu                           | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
| daq::depolarization2::num_targets | EOptions21return_execution_dataE) |
|     (C++                          | -   [cudaq::pts                   |
|     member)](api/la               | be::PTSBEOptions::shot_allocation |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q15depolarization211num_targetsE) |     member)](api/p                |
| -                                 | tsbe_api.html#_CPPv4N5cudaq5ptsbe |
|    [cudaq::depolarization_channel | 12PTSBEOptions15shot_allocationE) |
|     (C++                          | -   [cud                          |
|     class)](                      | aq::ptsbe::PTSBEOptions::strategy |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq22depolarization_channelE) |     member                        |
| -   [cudaq::depol                 | )](api/ptsbe_api.html#_CPPv4N5cud |
| arization_channel::num_parameters | aq5ptsbe12PTSBEOptions8strategyE) |
|     (C++                          | -   [                             |
|     member)](api/languages/cp     | cudaq::ptsbe::PTSSamplingStrategy |
| p_api.html#_CPPv4N5cudaq22depolar |     (C++                          |
| ization_channel14num_parametersE) |     cla                           |
| -   [cudaq::de                    | ss)](api/ptsbe_api.html#_CPPv4N5c |
| polarization_channel::num_targets | udaq5ptsbe19PTSSamplingStrategyE) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/languages        | ptsbe::PTSSamplingStrategy::clone |
| /cpp_api.html#_CPPv4N5cudaq22depo |     (C++                          |
| larization_channel11num_targetsE) |     function)](api                |
| -   [cudaq::details (C++          | /ptsbe_api.html#_CPPv4NK5cudaq5pt |
|     type)](api/languages/cpp      | sbe19PTSSamplingStrategy5cloneEv) |
| _api.html#_CPPv4N5cudaq7detailsE) | -   [cudaq::ptsbe::PTSSampl       |
| -   [cudaq::details::future (C++  | ingStrategy::generateTrajectories |
|                                   |     (C++                          |
|  class)](api/languages/cpp_api.ht |     function)](                   |
| ml#_CPPv4N5cudaq7details6futureE) | api/ptsbe_api.html#_CPPv4NK5cudaq |
| -                                 | 5ptsbe19PTSSamplingStrategy20gene |
|   [cudaq::details::future::future | rateTrajectoriesENSt4spanIKN5cuda |
|     (C++                          | q15KrausTrajectoryEEENSt6size_tE) |
|     functio                       | -   [cudaq:                       |
| n)](api/languages/cpp_api.html#_C | :ptsbe::PTSSamplingStrategy::name |
| PPv4N5cudaq7details6future6future |     (C++                          |
| ERNSt6vectorI3JobEERNSt6stringERN |     function)](ap                 |
| St3mapINSt6stringENSt6stringEEE), | i/ptsbe_api.html#_CPPv4NK5cudaq5p |
|     [\[1\]](api/lang              | tsbe19PTSSamplingStrategy4nameEv) |
| uages/cpp_api.html#_CPPv4N5cudaq7 | -   [cudaq::ptsbe::sample (C++    |
| details6future6futureERR6future), |     function)](api/ptsbe_ap       |
|     [\[2\]]                       | i.html#_CPPv4I0DpEN5cudaq5ptsbe6s |
| (api/languages/cpp_api.html#_CPPv | ampleE13sample_resultRK14sample_o |
| 4N5cudaq7details6future6futureEv) | ptionsRR13QuantumKernelDpRR4Args) |
| -   [cu                           | -   [cudaq::ptsbe::sample_async   |
| daq::details::kernel_builder_base |     (C++                          |
|     (C++                          |     function)](api/ptsbe_a        |
|     class)](api/l                 | pi.html#_CPPv4I0DpEN5cudaq5ptsbe1 |
| anguages/cpp_api.html#_CPPv4N5cud | 2sample_asyncEN5cudaq19async_samp |
| aq7details19kernel_builder_baseE) | le_resultERK14sample_optionsRR13Q |
| -   [cudaq::details::             | uantumKernelDpRR4ArgsNSt6size_tE) |
| kernel_builder_base::operator\<\< | -   [cudaq::ptsbe::sample_options |
|     (C++                          |     (C++                          |
|     function)](api/langua         |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq7de |  struct)](api/ptsbe_api.html#_CPP |
| tails19kernel_builder_baselsERNSt | v4N5cudaq5ptsbe14sample_optionsE) |
| 7ostreamERK19kernel_builder_base) | -   [cu                           |
| -   [                             | daq::ptsbe::sample_options::noise |
| cudaq::details::KernelBuilderType |     (C++                          |
|     (C++                          |     membe                         |
|     class)](api                   | r)](api/ptsbe_api.html#_CPPv4N5cu |
| /languages/cpp_api.html#_CPPv4N5c | daq5ptsbe14sample_options5noiseE) |
| udaq7details17KernelBuilderTypeE) | -   [cu                           |
| -   [cudaq::d                     | daq::ptsbe::sample_options::ptsbe |
| etails::KernelBuilderType::create |     (C++                          |
|     (C++                          |     membe                         |
|     function)                     | r)](api/ptsbe_api.html#_CPPv4N5cu |
| ](api/languages/cpp_api.html#_CPP | daq5ptsbe14sample_options5ptsbeE) |
| v4N5cudaq7details17KernelBuilderT | -   [cu                           |
| ype6createEPN4mlir11MLIRContextE) | daq::ptsbe::sample_options::shots |
| -   [cudaq::details::Ker          |     (C++                          |
| nelBuilderType::KernelBuilderType |     membe                         |
|     (C++                          | r)](api/ptsbe_api.html#_CPPv4N5cu |
|     function)](api/lang           | daq5ptsbe14sample_options5shotsE) |
| uages/cpp_api.html#_CPPv4N5cudaq7 | -   [cudaq::ptsbe::sample_result  |
| details17KernelBuilderType17Kerne |     (C++                          |
| lBuilderTypeERRNSt8functionIFN4ml |                                   |
| ir4TypeEPN4mlir11MLIRContextEEEE) |    class)](api/ptsbe_api.html#_CP |
| -   [cudaq::diag_matrix_callback  | Pv4N5cudaq5ptsbe13sample_resultE) |
|     (C++                          | -   [cudaq::pts                   |
|     class)                        | be::sample_result::execution_data |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq20diag_matrix_callbackE) |     function)](api/pts            |
| -   [cudaq::dyn (C++              | be_api.html#_CPPv4NK5cudaq5ptsbe1 |
|     member)](api/languages        | 3sample_result14execution_dataEv) |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | -   [cudaq::ptsbe::               |
| -   [cudaq::ExecutionContext (C++ | sample_result::has_execution_data |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     function)](api/ptsbe_a        |
| _CPPv4N5cudaq16ExecutionContextE) | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| -   [cudaq                        | ple_result18has_execution_dataEv) |
| ::ExecutionContext::amplitudeMaps | -   [cudaq::ptsbe::               |
|     (C++                          | sample_result::set_execution_data |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     functio                       |
| ExecutionContext13amplitudeMapsE) | n)](api/ptsbe_api.html#_CPPv4N5cu |
| -   [c                            | daq5ptsbe13sample_result18set_exe |
| udaq::ExecutionContext::asyncExec | cution_dataE18PTSBEExecutionData) |
|     (C++                          | -   [cud                          |
|     member)](api/                 | aq::ptsbe::ShotAllocationStrategy |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq16ExecutionContext9asyncExecE) |     struct)                       |
| -   [cud                          | ](api/ptsbe_api.html#_CPPv4N5cuda |
| aq::ExecutionContext::asyncResult | q5ptsbe22ShotAllocationStrategyE) |
|     (C++                          | -   [cudaq::ptsbe::Shot           |
|     member)](api/lan              | AllocationStrategy::bias_strength |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11asyncResultE) |     member)](api/ptsbe_api        |
| -   [cudaq:                       | .html#_CPPv4N5cudaq5ptsbe22ShotAl |
| :ExecutionContext::batchIteration | locationStrategy13bias_strengthE) |
|     (C++                          | -   [cudaq::pt                    |
|     member)](api/langua           | sbe::ShotAllocationStrategy::seed |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14batchIterationE) |     member)](api                  |
| -   [cudaq::E                     | /ptsbe_api.html#_CPPv4N5cudaq5pts |
| xecutionContext::canHandleObserve | be22ShotAllocationStrategy4seedE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAllocatio  |
|     member)](api/language         | nStrategy::ShotAllocationStrategy |
| s/cpp_api.html#_CPPv4N5cudaq16Exe |     (C++                          |
| cutionContext16canHandleObserveE) |     function)](api/ptsb           |
| -   [cudaq::E                     | e_api.html#_CPPv4N5cudaq5ptsbe22S |
| xecutionContext::ExecutionContext | hotAllocationStrategy22ShotAlloca |
|     (C++                          | tionStrategyE4TypedNSt8uint64_tE) |
|     func                          | -   [cudaq::pt                    |
| tion)](api/languages/cpp_api.html | sbe::ShotAllocationStrategy::Type |
| #_CPPv4N5cudaq16ExecutionContext1 |     (C++                          |
| 6ExecutionContextERKNSt6stringE), |     enum)](api                    |
|     [\[1\]](api/languages/        | /ptsbe_api.html#_CPPv4N5cudaq5pts |
| cpp_api.html#_CPPv4N5cudaq16Execu | be22ShotAllocationStrategy4TypeE) |
| tionContext16ExecutionContextERKN | -   [cudaq::pt                    |
| St6stringENSt6size_tENSt6size_tE) | sbe::ShotAllocationStrategy::type |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::expectationValue |     member)](api                  |
|     (C++                          | /ptsbe_api.html#_CPPv4N5cudaq5pts |
|     member)](api/language         | be22ShotAllocationStrategy4typeE) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::ptsbe::ShotAllocatio  |
| cutionContext16expectationValueE) | nStrategy::Type::HIGH_WEIGHT_BIAS |
| -   [cudaq::Execu                 |     (C++                          |
| tionContext::explicitMeasurements |     e                             |
|     (C++                          | numerator)](api/ptsbe_api.html#_C |
|     member)](api/languages/cp     | PPv4N5cudaq5ptsbe22ShotAllocation |
| p_api.html#_CPPv4N5cudaq16Executi | Strategy4Type16HIGH_WEIGHT_BIASE) |
| onContext20explicitMeasurementsE) | -   [cudaq::ptsbe::ShotAllocati   |
| -   [cuda                         | onStrategy::Type::LOW_WEIGHT_BIAS |
| q::ExecutionContext::futureResult |     (C++                          |
|     (C++                          |                                   |
|     member)](api/lang             | enumerator)](api/ptsbe_api.html#_ |
| uages/cpp_api.html#_CPPv4N5cudaq1 | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| 6ExecutionContext12futureResultE) | nStrategy4Type15LOW_WEIGHT_BIASE) |
| -   [cudaq::ExecutionContext      | -   [cudaq::ptsbe::ShotAlloc      |
| ::hasConditionalsOnMeasureResults | ationStrategy::Type::PROPORTIONAL |
|     (C++                          |     (C++                          |
|     mem                           |                                   |
| ber)](api/languages/cpp_api.html# |    enumerator)](api/ptsbe_api.htm |
| _CPPv4N5cudaq16ExecutionContext31 | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
| hasConditionalsOnMeasureResultsE) | tionStrategy4Type12PROPORTIONALE) |
| -   [cudaq::Executi               | -   [cudaq::ptsbe::Shot           |
| onContext::invocationResultBuffer | AllocationStrategy::Type::UNIFORM |
|     (C++                          |     (C++                          |
|     member)](api/languages/cpp_   |     enumerator)](api/ptsbe_a      |
| api.html#_CPPv4N5cudaq16Execution | pi.html#_CPPv4N5cudaq5ptsbe22Shot |
| Context22invocationResultBufferE) | AllocationStrategy4Type7UNIFORME) |
| -   [cu                           | -                                 |
| daq::ExecutionContext::kernelName |   [cudaq::ptsbe::TraceInstruction |
|     (C++                          |     (C++                          |
|     member)](api/la               |     s                             |
| nguages/cpp_api.html#_CPPv4N5cuda | truct)](api/ptsbe_api.html#_CPPv4 |
| q16ExecutionContext10kernelNameE) | N5cudaq5ptsbe16TraceInstructionE) |
| -   [cud                          | -   [cudaq:                       |
| aq::ExecutionContext::kernelTrace | :ptsbe::TraceInstruction::channel |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     member)](                     |
| guages/cpp_api.html#_CPPv4N5cudaq | api/ptsbe_api.html#_CPPv4N5cudaq5 |
| 16ExecutionContext11kernelTraceE) | ptsbe16TraceInstruction7channelE) |
| -   [cudaq:                       | -   [cudaq::                      |
| :ExecutionContext::msm_dimensions | ptsbe::TraceInstruction::controls |
|     (C++                          |     (C++                          |
|     member)](api/langua           |     member)](a                    |
| ges/cpp_api.html#_CPPv4N5cudaq16E | pi/ptsbe_api.html#_CPPv4N5cudaq5p |
| xecutionContext14msm_dimensionsE) | tsbe16TraceInstruction8controlsE) |
| -   [cudaq::                      | -   [cud                          |
| ExecutionContext::msm_prob_err_id | aq::ptsbe::TraceInstruction::name |
|     (C++                          |     (C++                          |
|     member)](api/languag          |     member                        |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | )](api/ptsbe_api.html#_CPPv4N5cud |
| ecutionContext15msm_prob_err_idE) | aq5ptsbe16TraceInstruction4nameE) |
| -   [cudaq::Ex                    | -   [cudaq                        |
| ecutionContext::msm_probabilities | ::ptsbe::TraceInstruction::params |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     member)]                      |
| /cpp_api.html#_CPPv4N5cudaq16Exec | (api/ptsbe_api.html#_CPPv4N5cudaq |
| utionContext17msm_probabilitiesE) | 5ptsbe16TraceInstruction6paramsE) |
| -                                 | -   [cudaq:                       |
|    [cudaq::ExecutionContext::name | :ptsbe::TraceInstruction::targets |
|     (C++                          |     (C++                          |
|     member)]                      |     member)](                     |
| (api/languages/cpp_api.html#_CPPv | api/ptsbe_api.html#_CPPv4N5cudaq5 |
| 4N5cudaq16ExecutionContext4nameE) | ptsbe16TraceInstruction7targetsE) |
| -   [cu                           | -   [cud                          |
| daq::ExecutionContext::noiseModel | aq::ptsbe::TraceInstruction::type |
|     (C++                          |     (C++                          |
|     member)](api/la               |     member                        |
| nguages/cpp_api.html#_CPPv4N5cuda | )](api/ptsbe_api.html#_CPPv4N5cud |
| q16ExecutionContext10noiseModelE) | aq5ptsbe16TraceInstruction4typeE) |
| -   [cudaq::Exe                   | -   [c                            |
| cutionContext::numberTrajectories | udaq::ptsbe::TraceInstructionType |
|     (C++                          |     (C++                          |
|     member)](api/languages/       |     enu                           |
| cpp_api.html#_CPPv4N5cudaq16Execu | m)](api/ptsbe_api.html#_CPPv4N5cu |
| tionContext18numberTrajectoriesE) | daq5ptsbe20TraceInstructionTypeE) |
| -   [c                            | -   [cudaq::                      |
| udaq::ExecutionContext::optResult | ptsbe::TraceInstructionType::Gate |
|     (C++                          |     (C++                          |
|     member)](api/                 |     enumerator)](a                |
| languages/cpp_api.html#_CPPv4N5cu | pi/ptsbe_api.html#_CPPv4N5cudaq5p |
| daq16ExecutionContext9optResultE) | tsbe20TraceInstructionType4GateE) |
| -   [cudaq::Execu                 | -   [cudaq::ptsbe::               |
| tionContext::overlapComputeStates | TraceInstructionType::Measurement |
|     (C++                          |     (C++                          |
|     member)](api/languages/cp     |     enumerator)](api/ptsbe        |
| p_api.html#_CPPv4N5cudaq16Executi | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| onContext20overlapComputeStatesE) | aceInstructionType11MeasurementE) |
| -   [cudaq                        | -   [cudaq::p                     |
| ::ExecutionContext::overlapResult | tsbe::TraceInstructionType::Noise |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     enumerator)](ap               |
| ages/cpp_api.html#_CPPv4N5cudaq16 | i/ptsbe_api.html#_CPPv4N5cudaq5pt |
| ExecutionContext13overlapResultE) | sbe20TraceInstructionType5NoiseE) |
| -                                 | -   [cudaq::QPU (C++              |
|   [cudaq::ExecutionContext::qpuId |     class)](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     member)](                     | -   [cudaq::QPU::beginExecution   |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5qpuIdE) |     function                      |
| -   [cudaq                        | )](api/languages/cpp_api.html#_CP |
| ::ExecutionContext::registerNames | Pv4N5cudaq3QPU14beginExecutionEv) |
|     (C++                          | -   [cuda                         |
|     member)](api/langu            | q::QPU::configureExecutionContext |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13registerNamesE) |     funct                         |
| -   [cu                           | ion)](api/languages/cpp_api.html# |
| daq::ExecutionContext::reorderIdx | _CPPv4NK5cudaq3QPU25configureExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     member)](api/la               | -   [cudaq::QPU::endExecution     |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10reorderIdxE) |     functi                        |
| -                                 | on)](api/languages/cpp_api.html#_ |
|  [cudaq::ExecutionContext::result | CPPv4N5cudaq3QPU12endExecutionEv) |
|     (C++                          | -   [cudaq::QPU::enqueue (C++     |
|     member)](a                    |     function)](ap                 |
| pi/languages/cpp_api.html#_CPPv4N | i/languages/cpp_api.html#_CPPv4N5 |
| 5cudaq16ExecutionContext6resultE) | cudaq3QPU7enqueueER11QuantumTask) |
| -                                 | -   [cud                          |
|   [cudaq::ExecutionContext::shots | aq::QPU::finalizeExecutionContext |
|     (C++                          |     (C++                          |
|     member)](                     |     func                          |
| api/languages/cpp_api.html#_CPPv4 | tion)](api/languages/cpp_api.html |
| N5cudaq16ExecutionContext5shotsE) | #_CPPv4NK5cudaq3QPU24finalizeExec |
| -   [cudaq::                      | utionContextER16ExecutionContext) |
| ExecutionContext::simulationState | -   [cudaq::QPU::getConnectivity  |
|     (C++                          |     (C++                          |
|     member)](api/languag          |     function)                     |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | ](api/languages/cpp_api.html#_CPP |
| ecutionContext15simulationStateE) | v4N5cudaq3QPU15getConnectivityEv) |
| -                                 | -                                 |
|    [cudaq::ExecutionContext::spin | [cudaq::QPU::getExecutionThreadId |
|     (C++                          |     (C++                          |
|     member)]                      |     function)](api/               |
| (api/languages/cpp_api.html#_CPPv | languages/cpp_api.html#_CPPv4NK5c |
| 4N5cudaq16ExecutionContext4spinE) | udaq3QPU20getExecutionThreadIdEv) |
| -   [cudaq::                      | -   [cudaq::QPU::getNumQubits     |
| ExecutionContext::totalIterations |     (C++                          |
|     (C++                          |     functi                        |
|     member)](api/languag          | on)](api/languages/cpp_api.html#_ |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| ecutionContext15totalIterationsE) | -   [                             |
| -   [cudaq::Executio              | cudaq::QPU::getRemoteCapabilities |
| nContext::warnedNamedMeasurements |     (C++                          |
|     (C++                          |     function)](api/l              |
|     member)](api/languages/cpp_a  | anguages/cpp_api.html#_CPPv4NK5cu |
| pi.html#_CPPv4N5cudaq16ExecutionC | daq3QPU21getRemoteCapabilitiesEv) |
| ontext23warnedNamedMeasurementsE) | -   [cudaq::QPU::isEmulated (C++  |
| -   [cudaq::ExecutionResult (C++  |     func                          |
|     st                            | tion)](api/languages/cpp_api.html |
| ruct)](api/languages/cpp_api.html | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| #_CPPv4N5cudaq15ExecutionResultE) | -   [cudaq::QPU::isSimulator (C++ |
| -   [cud                          |     funct                         |
| aq::ExecutionResult::appendResult | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|     functio                       | -   [cudaq::QPU::launchKernel     |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq15ExecutionResult12app |     function)](api/               |
| endResultENSt6stringENSt6size_tE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cu                           | daq3QPU12launchKernelERKNSt6strin |
| daq::ExecutionResult::deserialize | gE15KernelThunkTypePvNSt8uint64_t |
|     (C++                          | ENSt8uint64_tERKNSt6vectorIPvEE), |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP |  [\[1\]](api/languages/cpp_api.ht |
| v4N5cudaq15ExecutionResult11deser | ml#_CPPv4N5cudaq3QPU12launchKerne |
| ializeERNSt6vectorINSt6size_tEEE) | lERKNSt6stringERKNSt6vectorIPvEE) |
| -   [cudaq:                       | -   [cudaq::QPU::onRandomSeedSet  |
| :ExecutionResult::ExecutionResult |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     functio                       | uages/cpp_api.html#_CPPv4N5cudaq3 |
| n)](api/languages/cpp_api.html#_C | QPU15onRandomSeedSetENSt6size_tE) |
| PPv4N5cudaq15ExecutionResult15Exe | -   [cudaq::QPU::QPU (C++         |
| cutionResultE16CountsDictionary), |     functio                       |
|     [\[1\]](api/lan               | n)](api/languages/cpp_api.html#_C |
| guages/cpp_api.html#_CPPv4N5cudaq | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| 15ExecutionResult15ExecutionResul |                                   |
| tE16CountsDictionaryNSt6stringE), |  [\[1\]](api/languages/cpp_api.ht |
|     [\[2\                         | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| ]](api/languages/cpp_api.html#_CP |     [\[2\]](api/languages/cpp_    |
| Pv4N5cudaq15ExecutionResult15Exec | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| utionResultE16CountsDictionaryd), | -   [cudaq::QPU::setId (C++       |
|                                   |     function                      |
|    [\[3\]](api/languages/cpp_api. | )](api/languages/cpp_api.html#_CP |
| html#_CPPv4N5cudaq15ExecutionResu | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| lt15ExecutionResultENSt6stringE), | -   [cudaq::QPU::setShots (C++    |
|     [\[4\                         |     f                             |
| ]](api/languages/cpp_api.html#_CP | unction)](api/languages/cpp_api.h |
| Pv4N5cudaq15ExecutionResult15Exec | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| utionResultERK15ExecutionResult), | -   [cudaq::                      |
|     [\[5\]](api/language          | QPU::supportsExplicitMeasurements |
| s/cpp_api.html#_CPPv4N5cudaq15Exe |     (C++                          |
| cutionResult15ExecutionResultEd), |     function)](api/languag        |
|     [\[6\]](api/languag           | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | 28supportsExplicitMeasurementsEv) |
| ecutionResult15ExecutionResultEv) | -   [cudaq::QPU::\~QPU (C++       |
| -   [                             |     function)](api/languages/cp   |
| cudaq::ExecutionResult::operator= | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
|     (C++                          | -   [cudaq::QPUState (C++         |
|     function)](api/languages/     |     class)](api/languages/cpp_    |
| cpp_api.html#_CPPv4N5cudaq15Execu | api.html#_CPPv4N5cudaq8QPUStateE) |
| tionResultaSERK15ExecutionResult) | -   [cudaq::qreg (C++             |
| -   [c                            |     class)](api/lan               |
| udaq::ExecutionResult::operator== | guages/cpp_api.html#_CPPv4I_NSt6s |
|     (C++                          | ize_tE_NSt6size_tEEN5cudaq4qregE) |
|     function)](api/languages/c    | -   [cudaq::qreg::back (C++       |
| pp_api.html#_CPPv4NK5cudaq15Execu |     function)                     |
| tionResulteqERK15ExecutionResult) | ](api/languages/cpp_api.html#_CPP |
| -   [cud                          | v4N5cudaq4qreg4backENSt6size_tE), |
| aq::ExecutionResult::registerName |     [\[1\]](api/languages/cpp_ap  |
|     (C++                          | i.html#_CPPv4N5cudaq4qreg4backEv) |
|     member)](api/lan              | -   [cudaq::qreg::begin (C++      |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 15ExecutionResult12registerNameE) |  function)](api/languages/cpp_api |
| -   [cudaq                        | .html#_CPPv4N5cudaq4qreg5beginEv) |
| ::ExecutionResult::sequentialData | -   [cudaq::qreg::clear (C++      |
|     (C++                          |                                   |
|     member)](api/langu            |  function)](api/languages/cpp_api |
| ages/cpp_api.html#_CPPv4N5cudaq15 | .html#_CPPv4N5cudaq4qreg5clearEv) |
| ExecutionResult14sequentialDataE) | -   [cudaq::qreg::front (C++      |
| -   [                             |     function)]                    |
| cudaq::ExecutionResult::serialize | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4N5cudaq4qreg5frontENSt6size_tE), |
|     function)](api/l              |     [\[1\]](api/languages/cpp_api |
| anguages/cpp_api.html#_CPPv4NK5cu | .html#_CPPv4N5cudaq4qreg5frontEv) |
| daq15ExecutionResult9serializeEv) | -   [cudaq::qreg::operator\[\]    |
| -   [cudaq::fermion_handler (C++  |     (C++                          |
|     c                             |     functi                        |
| lass)](api/languages/cpp_api.html | on)](api/languages/cpp_api.html#_ |
| #_CPPv4N5cudaq15fermion_handlerE) | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| -   [cudaq::fermion_op (C++       | -   [cudaq::qreg::qreg (C++       |
|     type)](api/languages/cpp_api  |     function)                     |
| .html#_CPPv4N5cudaq10fermion_opE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::fermion_op_term (C++  | v4N5cudaq4qreg4qregENSt6size_tE), |
|                                   |     [\[1\]](api/languages/cpp_ap  |
| type)](api/languages/cpp_api.html | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| #_CPPv4N5cudaq15fermion_op_termE) | -   [cudaq::qreg::size (C++       |
| -   [cudaq::FermioniqBaseQPU (C++ |                                   |
|     cl                            |  function)](api/languages/cpp_api |
| ass)](api/languages/cpp_api.html# | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | -   [cudaq::qreg::slice (C++      |
| -   [cudaq::get_state (C++        |     function)](api/langu          |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq4q |
|    function)](api/languages/cpp_a | reg5sliceENSt6size_tENSt6size_tE) |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | -   [cudaq::qreg::value_type (C++ |
| ateEDaRR13QuantumKernelDpRR4Args) |                                   |
| -   [cudaq::gradient (C++         | type)](api/languages/cpp_api.html |
|     class)](api/languages/cpp_    | #_CPPv4N5cudaq4qreg10value_typeE) |
| api.html#_CPPv4N5cudaq8gradientE) | -   [cudaq::qspan (C++            |
| -   [cudaq::gradient::clone (C++  |     class)](api/lang              |
|     fun                           | uages/cpp_api.html#_CPPv4I_NSt6si |
| ction)](api/languages/cpp_api.htm | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| l#_CPPv4N5cudaq8gradient5cloneEv) | -   [cudaq::QuakeValue (C++       |
| -   [cudaq::gradient::compute     |     class)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq10QuakeValueE) |
|     function)](api/language       | -   [cudaq::Q                     |
| s/cpp_api.html#_CPPv4N5cudaq8grad | uakeValue::canValidateNumElements |
| ient7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)](api/languages      |
|     [\[1\]](ap                    | /cpp_api.html#_CPPv4N5cudaq10Quak |
| i/languages/cpp_api.html#_CPPv4N5 | eValue22canValidateNumElementsEv) |
| cudaq8gradient7computeERKNSt6vect | -                                 |
| orIdEERNSt6vectorIdEERK7spin_opd) |  [cudaq::QuakeValue::constantSize |
| -   [cudaq::gradient::gradient    |     (C++                          |
|     (C++                          |     function)](api                |
|     function)](api/lang           | /languages/cpp_api.html#_CPPv4N5c |
| uages/cpp_api.html#_CPPv4I00EN5cu | udaq10QuakeValue12constantSizeEv) |
| daq8gradient8gradientER7KernelT), | -   [cudaq::QuakeValue::dump (C++ |
|                                   |     function)](api/lan            |
|    [\[1\]](api/languages/cpp_api. | guages/cpp_api.html#_CPPv4N5cudaq |
| html#_CPPv4I00EN5cudaq8gradient8g | 10QuakeValue4dumpERNSt7ostreamE), |
| radientER7KernelTRR10ArgsMapper), |     [\                            |
|     [\[2\                         | [1\]](api/languages/cpp_api.html# |
| ]](api/languages/cpp_api.html#_CP | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| Pv4I00EN5cudaq8gradient8gradientE | -   [cudaq                        |
| RR13QuantumKernelRR10ArgsMapper), | ::QuakeValue::getRequiredElements |
|     [\[3                          |     (C++                          |
| \]](api/languages/cpp_api.html#_C |     function)](api/langua         |
| PPv4N5cudaq8gradient8gradientERRN | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| St8functionIFvNSt6vectorIdEEEEE), | uakeValue19getRequiredElementsEv) |
|     [\[                           | -   [cudaq::QuakeValue::getValue  |
| 4\]](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq8gradient8gradientEv) |     function)]                    |
| -   [cudaq::gradient::setArgs     | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NK5cudaq10QuakeValue8getValueEv) |
|     fu                            | -   [cudaq::QuakeValue::inverse   |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     function)                     |
| tArgsEvR13QuantumKernelDpRR4Args) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::gradient::setKernel   | v4NK5cudaq10QuakeValue7inverseEv) |
|     (C++                          | -   [cudaq::QuakeValue::isStdVec  |
|     function)](api/languages/c    |     (C++                          |
| pp_api.html#_CPPv4I0EN5cudaq8grad |     function)                     |
| ient9setKernelEvR13QuantumKernel) | ](api/languages/cpp_api.html#_CPP |
| -   [cud                          | v4N5cudaq10QuakeValue8isStdVecEv) |
| aq::gradients::central_difference | -                                 |
|     (C++                          |    [cudaq::QuakeValue::operator\* |
|     class)](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api                |
| q9gradients18central_differenceE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gra                   | udaq10QuakeValuemlE10QuakeValue), |
| dients::central_difference::clone |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     function)](api/languages      | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| /cpp_api.html#_CPPv4N5cudaq9gradi | -   [cudaq::QuakeValue::operator+ |
| ents18central_difference5cloneEv) |     (C++                          |
| -   [cudaq::gradi                 |     function)](api                |
| ents::central_difference::compute | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValueplE10QuakeValue), |
|     function)](                   |     [                             |
| api/languages/cpp_api.html#_CPPv4 | \[1\]](api/languages/cpp_api.html |
| N5cudaq9gradients18central_differ | #_CPPv4N5cudaq10QuakeValueplEKd), |
| ence7computeERKNSt6vectorIdEERKNS |                                   |
| t8functionIFdNSt6vectorIdEEEEEd), | [\[2\]](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq10QuakeValueplEKi) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::QuakeValue::operator- |
| tml#_CPPv4N5cudaq9gradients18cent |     (C++                          |
| ral_difference7computeERKNSt6vect |     function)](api                |
| orIdEERNSt6vectorIdEERK7spin_opd) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::gradie                | udaq10QuakeValuemiE10QuakeValue), |
| nts::central_difference::gradient |     [                             |
|     (C++                          | \[1\]](api/languages/cpp_api.html |
|     functio                       | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| n)](api/languages/cpp_api.html#_C |     [                             |
| PPv4I00EN5cudaq9gradients18centra | \[2\]](api/languages/cpp_api.html |
| l_difference8gradientER7KernelT), | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     [\[1\]](api/langua            |                                   |
| ges/cpp_api.html#_CPPv4I00EN5cuda | [\[3\]](api/languages/cpp_api.htm |
| q9gradients18central_difference8g | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::QuakeValue::operator/ |
|     [\[2\]](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4I00EN5cudaq9gradie |     function)](api                |
| nts18central_difference8gradientE | /languages/cpp_api.html#_CPPv4N5c |
| RR13QuantumKernelRR10ArgsMapper), | udaq10QuakeValuedvE10QuakeValue), |
|     [\[3\]](api/languages/cpp     |                                   |
| _api.html#_CPPv4N5cudaq9gradients | [\[1\]](api/languages/cpp_api.htm |
| 18central_difference8gradientERRN | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| St8functionIFvNSt6vectorIdEEEEE), | -                                 |
|     [\[4\]](api/languages/cp      |  [cudaq::QuakeValue::operator\[\] |
| p_api.html#_CPPv4N5cudaq9gradient |     (C++                          |
| s18central_difference8gradientEv) |     function)](api                |
| -   [cud                          | /languages/cpp_api.html#_CPPv4N5c |
| aq::gradients::forward_difference | udaq10QuakeValueixEKNSt6size_tE), |
|     (C++                          |     [\[1\]](api/                  |
|     class)](api/la                | languages/cpp_api.html#_CPPv4N5cu |
| nguages/cpp_api.html#_CPPv4N5cuda | daq10QuakeValueixERK10QuakeValue) |
| q9gradients18forward_differenceE) | -                                 |
| -   [cudaq::gra                   |    [cudaq::QuakeValue::QuakeValue |
| dients::forward_difference::clone |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/languages      | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| /cpp_api.html#_CPPv4N5cudaq9gradi | akeValue10QuakeValueERN4mlir20Imp |
| ents18forward_difference5cloneEv) | licitLocOpBuilderEN4mlir5ValueE), |
| -   [cudaq::gradi                 |     [\[1\]                        |
| ents::forward_difference::compute | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq10QuakeValue10QuakeValue |
|     function)](                   | ERN4mlir20ImplicitLocOpBuilderEd) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::QuakeValue::size (C++ |
| N5cudaq9gradients18forward_differ |     funct                         |
| ence7computeERKNSt6vectorIdEERKNS | ion)](api/languages/cpp_api.html# |
| t8functionIFdNSt6vectorIdEEEEEd), | _CPPv4N5cudaq10QuakeValue4sizeEv) |
|                                   | -   [cudaq::QuakeValue::slice     |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq9gradients18forw |     function)](api/languages/cpp_ |
| ard_difference7computeERKNSt6vect | api.html#_CPPv4N5cudaq10QuakeValu |
| orIdEERNSt6vectorIdEERK7spin_opd) | e5sliceEKNSt6size_tEKNSt6size_tE) |
| -   [cudaq::gradie                | -   [cudaq::quantum_platform (C++ |
| nts::forward_difference::gradient |     cl                            |
|     (C++                          | ass)](api/languages/cpp_api.html# |
|     functio                       | _CPPv4N5cudaq16quantum_platformE) |
| n)](api/languages/cpp_api.html#_C | -   [cudaq:                       |
| PPv4I00EN5cudaq9gradients18forwar | :quantum_platform::beginExecution |
| d_difference8gradientER7KernelT), |     (C++                          |
|     [\[1\]](api/langua            |     function)](api/languag        |
| ges/cpp_api.html#_CPPv4I00EN5cuda | es/cpp_api.html#_CPPv4N5cudaq16qu |
| q9gradients18forward_difference8g | antum_platform14beginExecutionEv) |
| radientER7KernelTRR10ArgsMapper), | -   [cudaq::quantum_pl            |
|     [\[2\]](api/languages/cpp_    | atform::configureExecutionContext |
| api.html#_CPPv4I00EN5cudaq9gradie |     (C++                          |
| nts18forward_difference8gradientE |     function)](api/lang           |
| RR13QuantumKernelRR10ArgsMapper), | uages/cpp_api.html#_CPPv4NK5cudaq |
|     [\[3\]](api/languages/cpp     | 16quantum_platform25configureExec |
| _api.html#_CPPv4N5cudaq9gradients | utionContextER16ExecutionContext) |
| 18forward_difference8gradientERRN | -   [cuda                         |
| St8functionIFvNSt6vectorIdEEEEE), | q::quantum_platform::connectivity |
|     [\[4\]](api/languages/cp      |     (C++                          |
| p_api.html#_CPPv4N5cudaq9gradient |     function)](api/langu          |
| s18forward_difference8gradientEv) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [                             | quantum_platform12connectivityEv) |
| cudaq::gradients::parameter_shift | -   [cuda                         |
|     (C++                          | q::quantum_platform::endExecution |
|     class)](api                   |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     function)](api/langu          |
| udaq9gradients15parameter_shiftE) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [cudaq::                      | quantum_platform12endExecutionEv) |
| gradients::parameter_shift::clone | -   [cudaq::q                     |
|     (C++                          | uantum_platform::enqueueAsyncTask |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     function)](api/languages/     |
| adients15parameter_shift5cloneEv) | cpp_api.html#_CPPv4N5cudaq16quant |
| -   [cudaq::gr                    | um_platform16enqueueAsyncTaskEKNS |
| adients::parameter_shift::compute | t6size_tER19KernelExecutionTask), |
|     (C++                          |     [\[1\]](api/languag           |
|     function                      | es/cpp_api.html#_CPPv4N5cudaq16qu |
| )](api/languages/cpp_api.html#_CP | antum_platform16enqueueAsyncTaskE |
| Pv4N5cudaq9gradients15parameter_s | KNSt6size_tERNSt8functionIFvvEEE) |
| hift7computeERKNSt6vectorIdEERKNS | -   [cudaq::quantum_p             |
| t8functionIFdNSt6vectorIdEEEEEd), | latform::finalizeExecutionContext |
|     [\[1\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4N5cudaq9gradients15p |     function)](api/languages/c    |
| arameter_shift7computeERKNSt6vect | pp_api.html#_CPPv4NK5cudaq16quant |
| orIdEERNSt6vectorIdEERK7spin_opd) | um_platform24finalizeExecutionCon |
| -   [cudaq::gra                   | textERN5cudaq16ExecutionContextE) |
| dients::parameter_shift::gradient | -   [cudaq::qua                   |
|     (C++                          | ntum_platform::get_codegen_config |
|     func                          |     (C++                          |
| tion)](api/languages/cpp_api.html |     function)](api/languages/c    |
| #_CPPv4I00EN5cudaq9gradients15par | pp_api.html#_CPPv4N5cudaq16quantu |
| ameter_shift8gradientER7KernelT), | m_platform18get_codegen_configEv) |
|     [\[1\]](api/lan               | -   [cuda                         |
| guages/cpp_api.html#_CPPv4I00EN5c | q::quantum_platform::get_exec_ctx |
| udaq9gradients15parameter_shift8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |     function)](api/langua         |
|     [\[2\]](api/languages/c       | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| pp_api.html#_CPPv4I00EN5cudaq9gra | quantum_platform12get_exec_ctxEv) |
| dients15parameter_shift8gradientE | -   [c                            |
| RR13QuantumKernelRR10ArgsMapper), | udaq::quantum_platform::get_noise |
|     [\[3\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq9gradie |     function)](api/languages/c    |
| nts15parameter_shift8gradientERRN | pp_api.html#_CPPv4N5cudaq16quantu |
| St8functionIFvNSt6vectorIdEEEEE), | m_platform9get_noiseENSt6size_tE) |
|     [\[4\]](api/languages         | -   [cudaq:                       |
| /cpp_api.html#_CPPv4N5cudaq9gradi | :quantum_platform::get_num_qubits |
| ents15parameter_shift8gradientEv) |     (C++                          |
| -   [cudaq::kernel_builder (C++   |                                   |
|     clas                          | function)](api/languages/cpp_api. |
| s)](api/languages/cpp_api.html#_C | html#_CPPv4NK5cudaq16quantum_plat |
| PPv4IDpEN5cudaq14kernel_builderE) | form14get_num_qubitsENSt6size_tE) |
| -   [c                            | -   [cudaq::quantum_              |
| udaq::kernel_builder::constantVal | platform::get_remote_capabilities |
|     (C++                          |     (C++                          |
|     function)](api/la             |     function)                     |
| nguages/cpp_api.html#_CPPv4N5cuda | ](api/languages/cpp_api.html#_CPP |
| q14kernel_builder11constantValEd) | v4NK5cudaq16quantum_platform23get |
| -   [cu                           | _remote_capabilitiesENSt6size_tE) |
| daq::kernel_builder::getArguments | -   [cudaq::qua                   |
|     (C++                          | ntum_platform::get_runtime_target |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/languages/cp   |
| 14kernel_builder12getArgumentsEv) | p_api.html#_CPPv4NK5cudaq16quantu |
| -   [cu                           | m_platform18get_runtime_targetEv) |
| daq::kernel_builder::getNumParams | -   [cuda                         |
|     (C++                          | q::quantum_platform::getLogStream |
|     function)](api/lan            |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/langu          |
| 14kernel_builder12getNumParamsEv) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [c                            | quantum_platform12getLogStreamEv) |
| udaq::kernel_builder::isArgStdVec | -   [cud                          |
|     (C++                          | aq::quantum_platform::is_emulated |
|     function)](api/languages/cp   |     (C++                          |
| p_api.html#_CPPv4N5cudaq14kernel_ |                                   |
| builder11isArgStdVecENSt6size_tE) |    function)](api/languages/cpp_a |
| -   [cuda                         | pi.html#_CPPv4NK5cudaq16quantum_p |
| q::kernel_builder::kernel_builder | latform11is_emulatedENSt6size_tE) |
|     (C++                          | -   [c                            |
|     function)](api/languages/cpp_ | udaq::quantum_platform::is_remote |
| api.html#_CPPv4N5cudaq14kernel_bu |     (C++                          |
| ilder14kernel_builderERNSt6vector |     function)](api/languages/cp   |
| IN7details17KernelBuilderTypeEEE) | p_api.html#_CPPv4NK5cudaq16quantu |
| -   [cudaq::kernel_builder::name  | m_platform9is_remoteENSt6size_tE) |
|     (C++                          | -   [cuda                         |
|     function)                     | q::quantum_platform::is_simulator |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq14kernel_builder4nameEv) |                                   |
| -                                 |   function)](api/languages/cpp_ap |
|    [cudaq::kernel_builder::qalloc | i.html#_CPPv4NK5cudaq16quantum_pl |
|     (C++                          | atform12is_simulatorENSt6size_tE) |
|     function)](api/language       | -   [c                            |
| s/cpp_api.html#_CPPv4N5cudaq14ker | udaq::quantum_platform::launchVQE |
| nel_builder6qallocE10QuakeValue), |     (C++                          |
|     [\[1\]](api/language          |     function)](                   |
| s/cpp_api.html#_CPPv4N5cudaq14ker | api/languages/cpp_api.html#_CPPv4 |
| nel_builder6qallocEKNSt6size_tE), | N5cudaq16quantum_platform9launchV |
|     [\[2                          | QEEKNSt6stringEPKvPN5cudaq8gradie |
| \]](api/languages/cpp_api.html#_C | ntERKN5cudaq7spin_opERN5cudaq9opt |
| PPv4N5cudaq14kernel_builder6qallo | imizerEKiKNSt6size_tENSt6size_tE) |
| cERNSt6vectorINSt7complexIdEEEE), | -   [cudaq:                       |
|     [\[3\]](                      | :quantum_platform::list_platforms |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq14kernel_builder6qallocEv) |     function)](api/languag        |
| -   [cudaq::kernel_builder::swap  | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14list_platformsEv) |
|     function)](api/language       | -                                 |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 |    [cudaq::quantum_platform::name |
| 4kernel_builder4swapEvRK10QuakeVa |     (C++                          |
| lueRK10QuakeValueRK10QuakeValue), |     function)](a                  |
|                                   | pi/languages/cpp_api.html#_CPPv4N |
| [\[1\]](api/languages/cpp_api.htm | K5cudaq16quantum_platform4nameEv) |
| l#_CPPv4I00EN5cudaq14kernel_build | -   [                             |
| er4swapEvRKNSt6vectorI10QuakeValu | cudaq::quantum_platform::num_qpus |
| eEERK10QuakeValueRK10QuakeValue), |     (C++                          |
|                                   |     function)](api/l              |
| [\[2\]](api/languages/cpp_api.htm | anguages/cpp_api.html#_CPPv4NK5cu |
| l#_CPPv4N5cudaq14kernel_builder4s | daq16quantum_platform8num_qpusEv) |
| wapERK10QuakeValueRK10QuakeValue) | -   [cudaq::                      |
| -   [cudaq::KernelExecutionTask   | quantum_platform::onRandomSeedSet |
|     (C++                          |     (C++                          |
|     type                          |                                   |
| )](api/languages/cpp_api.html#_CP | function)](api/languages/cpp_api. |
| Pv4N5cudaq19KernelExecutionTaskE) | html#_CPPv4N5cudaq16quantum_platf |
| -   [cudaq::KernelThunkResultType | orm15onRandomSeedSetENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     struct)]                      | :quantum_platform::reset_exec_ctx |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq21KernelThunkResultTypeE) |     function)](api/languag        |
| -   [cudaq::KernelThunkType (C++  | es/cpp_api.html#_CPPv4N5cudaq16qu |
|                                   | antum_platform14reset_exec_ctxEv) |
| type)](api/languages/cpp_api.html | -   [cud                          |
| #_CPPv4N5cudaq15KernelThunkTypeE) | aq::quantum_platform::reset_noise |
| -   [cudaq::kraus_channel (C++    |     (C++                          |
|                                   |     function)](api/languages/cpp_ |
|  class)](api/languages/cpp_api.ht | api.html#_CPPv4N5cudaq16quantum_p |
| ml#_CPPv4N5cudaq13kraus_channelE) | latform11reset_noiseENSt6size_tE) |
| -   [cudaq::kraus_channel::empty  | -   [cudaq:                       |
|     (C++                          | :quantum_platform::resetLogStream |
|     function)]                    |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/languag        |
| 4NK5cudaq13kraus_channel5emptyEv) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [cudaq::kraus_c               | antum_platform14resetLogStreamEv) |
| hannel::generateUnitaryParameters | -   [cuda                         |
|     (C++                          | q::quantum_platform::set_exec_ctx |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     funct                         |
| pi.html#_CPPv4N5cudaq13kraus_chan | ion)](api/languages/cpp_api.html# |
| nel25generateUnitaryParametersEv) | _CPPv4N5cudaq16quantum_platform12 |
| -                                 | set_exec_ctxEP16ExecutionContext) |
|    [cudaq::kraus_channel::get_ops | -   [c                            |
|     (C++                          | udaq::quantum_platform::set_noise |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     function                      |
| K5cudaq13kraus_channel7get_opsEv) | )](api/languages/cpp_api.html#_CP |
| -   [cud                          | Pv4N5cudaq16quantum_platform9set_ |
| aq::kraus_channel::identity_flags | noiseEPK11noise_modelNSt6size_tE) |
|     (C++                          | -   [cuda                         |
|     member)](api/lan              | q::quantum_platform::setLogStream |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 13kraus_channel14identity_flagsE) |                                   |
| -   [cud                          |  function)](api/languages/cpp_api |
| aq::kraus_channel::is_identity_op | .html#_CPPv4N5cudaq16quantum_plat |
|     (C++                          | form12setLogStreamERNSt7ostreamE) |
|                                   | -   [cudaq::quantum_platfor       |
|    function)](api/languages/cpp_a | m::supports_explicit_measurements |
| pi.html#_CPPv4NK5cudaq13kraus_cha |     (C++                          |
| nnel14is_identity_opENSt6size_tE) |     function)](api/l              |
| -   [cudaq::                      | anguages/cpp_api.html#_CPPv4NK5cu |
| kraus_channel::is_unitary_mixture | daq16quantum_platform30supports_e |
|     (C++                          | xplicit_measurementsENSt6size_tE) |
|     function)](api/languages      | -   [cudaq::quantum_pla           |
| /cpp_api.html#_CPPv4NK5cudaq13kra | tform::supports_task_distribution |
| us_channel18is_unitary_mixtureEv) |     (C++                          |
| -   [cu                           |     fu                            |
| daq::kraus_channel::kraus_channel | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4NK5cudaq16quantum_platfo |
|     function)](api/lang           | rm26supports_task_distributionEv) |
| uages/cpp_api.html#_CPPv4IDpEN5cu | -   [cudaq::quantum               |
| daq13kraus_channel13kraus_channel | _platform::with_execution_context |
| EDpRRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     function)                     |
|  [\[1\]](api/languages/cpp_api.ht | ](api/languages/cpp_api.html#_CPP |
| ml#_CPPv4N5cudaq13kraus_channel13 | v4I0DpEN5cudaq16quantum_platform2 |
| kraus_channelERK13kraus_channel), | 2with_execution_contextEDaR16Exec |
|     [\[2\]                        | utionContextRR8CallableDpRR4Args) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::QuantumTask (C++      |
| v4N5cudaq13kraus_channel13kraus_c |     type)](api/languages/cpp_api. |
| hannelERKNSt6vectorI8kraus_opEE), | html#_CPPv4N5cudaq11QuantumTaskE) |
|     [\[3\]                        | -   [cudaq::qubit (C++            |
| ](api/languages/cpp_api.html#_CPP |     type)](api/languages/c        |
| v4N5cudaq13kraus_channel13kraus_c | pp_api.html#_CPPv4N5cudaq5qubitE) |
| hannelERRNSt6vectorI8kraus_opEE), | -   [cudaq::QubitConnectivity     |
|     [\[4\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     ty                            |
| 13kraus_channel13kraus_channelEv) | pe)](api/languages/cpp_api.html#_ |
| -                                 | CPPv4N5cudaq17QubitConnectivityE) |
| [cudaq::kraus_channel::noise_type | -   [cudaq::QubitEdge (C++        |
|     (C++                          |     type)](api/languages/cpp_a    |
|     member)](api                  | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::qudit (C++            |
| udaq13kraus_channel10noise_typeE) |     clas                          |
| -                                 | s)](api/languages/cpp_api.html#_C |
|   [cudaq::kraus_channel::op_names | PPv4I_NSt6size_tEEN5cudaq5quditE) |
|     (C++                          | -   [cudaq::qudit::qudit (C++     |
|     member)](                     |                                   |
| api/languages/cpp_api.html#_CPPv4 | function)](api/languages/cpp_api. |
| N5cudaq13kraus_channel8op_namesE) | html#_CPPv4N5cudaq5qudit5quditEv) |
| -                                 | -   [cudaq::qvector (C++          |
|  [cudaq::kraus_channel::operator= |     class)                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/langua         | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| ges/cpp_api.html#_CPPv4N5cudaq13k | -   [cudaq::qvector::back (C++    |
| raus_channelaSERK13kraus_channel) |     function)](a                  |
| -   [c                            | pi/languages/cpp_api.html#_CPPv4N |
| udaq::kraus_channel::operator\[\] | 5cudaq7qvector4backENSt6size_tE), |
|     (C++                          |                                   |
|     function)](api/l              |   [\[1\]](api/languages/cpp_api.h |
| anguages/cpp_api.html#_CPPv4N5cud | tml#_CPPv4N5cudaq7qvector4backEv) |
| aq13kraus_channelixEKNSt6size_tE) | -   [cudaq::qvector::begin (C++   |
| -                                 |     fu                            |
| [cudaq::kraus_channel::parameters | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5beginEv) |
|     member)](api                  | -   [cudaq::qvector::clear (C++   |
| /languages/cpp_api.html#_CPPv4N5c |     fu                            |
| udaq13kraus_channel10parametersE) | nction)](api/languages/cpp_api.ht |
| -   [cudaq::krau                  | ml#_CPPv4N5cudaq7qvector5clearEv) |
| s_channel::populateDefaultOpNames | -   [cudaq::qvector::end (C++     |
|     (C++                          |                                   |
|     function)](api/languages/cp   | function)](api/languages/cpp_api. |
| p_api.html#_CPPv4N5cudaq13kraus_c | html#_CPPv4N5cudaq7qvector3endEv) |
| hannel22populateDefaultOpNamesEv) | -   [cudaq::qvector::front (C++   |
| -   [cu                           |     function)](ap                 |
| daq::kraus_channel::probabilities | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq7qvector5frontENSt6size_tE), |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda |  [\[1\]](api/languages/cpp_api.ht |
| q13kraus_channel13probabilitiesE) | ml#_CPPv4N5cudaq7qvector5frontEv) |
| -                                 | -   [cudaq::qvector::operator=    |
|  [cudaq::kraus_channel::push_back |     (C++                          |
|     (C++                          |     functio                       |
|     function)](api                | n)](api/languages/cpp_api.html#_C |
| /languages/cpp_api.html#_CPPv4N5c | PPv4N5cudaq7qvectoraSERK7qvector) |
| udaq13kraus_channel9push_backE8kr | -   [cudaq::qvector::operator\[\] |
| aus_opNSt8optionalINSt6stringEEE) |     (C++                          |
| -   [cudaq::kraus_channel::size   |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)                     | v4N5cudaq7qvectorixEKNSt6size_tE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::qvector::qvector (C++ |
| v4NK5cudaq13kraus_channel4sizeEv) |     function)](api/               |
| -   [                             | languages/cpp_api.html#_CPPv4N5cu |
| cudaq::kraus_channel::unitary_ops | daq7qvector7qvectorENSt6size_tE), |
|     (C++                          |     [\[1\]](a                     |
|     member)](api/                 | pi/languages/cpp_api.html#_CPPv4N |
| languages/cpp_api.html#_CPPv4N5cu | 5cudaq7qvector7qvectorERK5state), |
| daq13kraus_channel11unitary_opsE) |     [\[2\]](api                   |
| -   [cudaq::kraus_op (C++         | /languages/cpp_api.html#_CPPv4N5c |
|     struct)](api/languages/cpp_   | udaq7qvector7qvectorERK7qvector), |
| api.html#_CPPv4N5cudaq8kraus_opE) |     [\[3\]](api/languages/cpp     |
| -   [cudaq::kraus_op::adjoint     | _api.html#_CPPv4N5cudaq7qvector7q |
|     (C++                          | vectorERKNSt6vectorI7complexEEb), |
|     functi                        |     [\[4\]](ap                    |
| on)](api/languages/cpp_api.html#_ | i/languages/cpp_api.html#_CPPv4N5 |
| CPPv4NK5cudaq8kraus_op7adjointEv) | cudaq7qvector7qvectorERR7qvector) |
| -   [cudaq::kraus_op::data (C++   | -   [cudaq::qvector::size (C++    |
|                                   |     fu                            |
|  member)](api/languages/cpp_api.h | nction)](api/languages/cpp_api.ht |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| -   [cudaq::kraus_op::kraus_op    | -   [cudaq::qvector::slice (C++   |
|     (C++                          |     function)](api/language       |
|     func                          | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| tion)](api/languages/cpp_api.html | tor5sliceENSt6size_tENSt6size_tE) |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | -   [cudaq::qvector::value_type   |
| opERRNSt16initializer_listI1TEE), |     (C++                          |
|                                   |     typ                           |
|  [\[1\]](api/languages/cpp_api.ht | e)](api/languages/cpp_api.html#_C |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | PPv4N5cudaq7qvector10value_typeE) |
| pENSt6vectorIN5cudaq7complexEEE), | -   [cudaq::qview (C++            |
|     [\[2\]](api/l                 |     clas                          |
| anguages/cpp_api.html#_CPPv4N5cud | s)](api/languages/cpp_api.html#_C |
| aq8kraus_op8kraus_opERK8kraus_op) | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| -   [cudaq::kraus_op::nCols (C++  | -   [cudaq::qview::back (C++      |
|                                   |     function)                     |
| member)](api/languages/cpp_api.ht | ](api/languages/cpp_api.html#_CPP |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | v4N5cudaq5qview4backENSt6size_tE) |
| -   [cudaq::kraus_op::nRows (C++  | -   [cudaq::qview::begin (C++     |
|                                   |                                   |
| member)](api/languages/cpp_api.ht | function)](api/languages/cpp_api. |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | html#_CPPv4N5cudaq5qview5beginEv) |
| -   [cudaq::kraus_op::operator=   | -   [cudaq::qview::end (C++       |
|     (C++                          |                                   |
|     function)                     |   function)](api/languages/cpp_ap |
| ](api/languages/cpp_api.html#_CPP | i.html#_CPPv4N5cudaq5qview3endEv) |
| v4N5cudaq8kraus_opaSERK8kraus_op) | -   [cudaq::qview::front (C++     |
| -   [cudaq::kraus_op::precision   |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     memb                          | N5cudaq5qview5frontENSt6size_tE), |
| er)](api/languages/cpp_api.html#_ |                                   |
| CPPv4N5cudaq8kraus_op9precisionE) |    [\[1\]](api/languages/cpp_api. |
| -   [cudaq::KrausOperatorType     | html#_CPPv4N5cudaq5qview5frontEv) |
|     (C++                          | -   [cudaq::qview::operator\[\]   |
|     enum)](api/ptsbe_api.html#_   |     (C++                          |
| CPPv4N5cudaq17KrausOperatorTypeE) |     functio                       |
| -   [c                            | n)](api/languages/cpp_api.html#_C |
| udaq::KrausOperatorType::IDENTITY | PPv4N5cudaq5qviewixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qview::qview (C++     |
|     enumerato                     |     functio                       |
| r)](api/ptsbe_api.html#_CPPv4N5cu | n)](api/languages/cpp_api.html#_C |
| daq17KrausOperatorType8IDENTITYE) | PPv4I0EN5cudaq5qview5qviewERR1R), |
| -   [cudaq::KrausSelection (C++   |     [\[1                          |
|     struct)](api/ptsbe_api.htm    | \]](api/languages/cpp_api.html#_C |
| l#_CPPv4N5cudaq14KrausSelectionE) | PPv4N5cudaq5qview5qviewERK5qview) |
| -   [cudaq:                       | -   [cudaq::qview::size (C++      |
| :KrausSelection::circuit_location |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     member)](ap                   | html#_CPPv4NK5cudaq5qview4sizeEv) |
| i/ptsbe_api.html#_CPPv4N5cudaq14K | -   [cudaq::qview::slice (C++     |
| rausSelection16circuit_locationE) |     function)](api/langua         |
| -   [cudaq::Kra                   | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| usSelection::kraus_operator_index | iew5sliceENSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq::qview::value_type     |
|     member)](api/pt               |     (C++                          |
| sbe_api.html#_CPPv4N5cudaq14Kraus |     t                             |
| Selection20kraus_operator_indexE) | ype)](api/languages/cpp_api.html# |
| -                                 | _CPPv4N5cudaq5qview10value_typeE) |
|   [cudaq::KrausSelection::op_name | -   [cudaq::range (C++            |
|     (C++                          |     fun                           |
|     m                             | ction)](api/languages/cpp_api.htm |
| ember)](api/ptsbe_api.html#_CPPv4 | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| N5cudaq14KrausSelection7op_nameE) | orI11ElementTypeEE11ElementType), |
| -                                 |     [\[1\]](api/languages/cpp_    |
|    [cudaq::KrausSelection::qubits | api.html#_CPPv4I0EN5cudaq5rangeEN |
|     (C++                          | St6vectorI11ElementTypeEE11Elemen |
|                                   | tType11ElementType11ElementType), |
| member)](api/ptsbe_api.html#_CPPv |     [                             |
| 4N5cudaq14KrausSelection6qubitsE) | \[2\]](api/languages/cpp_api.html |
| -   [cudaq::KrausTrajectory (C++  | #_CPPv4N5cudaq5rangeENSt6size_tE) |
|     struct)](api/ptsbe_api.html   | -   [cudaq::real (C++             |
| #_CPPv4N5cudaq15KrausTrajectoryE) |     type)](api/languages/         |
| -   [cu                           | cpp_api.html#_CPPv4N5cudaq4realE) |
| daq::KrausTrajectory::countErrors | -   [cudaq::registry (C++         |
|     (C++                          |     type)](api/languages/cpp_     |
|     function)](                   | api.html#_CPPv4N5cudaq8registryE) |
| api/ptsbe_api.html#_CPPv4NK5cudaq | -                                 |
| 15KrausTrajectory11countErrorsEv) |  [cudaq::registry::RegisteredType |
| -   [                             |     (C++                          |
| cudaq::KrausTrajectory::isOrdered |     class)](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     function                      | 5cudaq8registry14RegisteredTypeE) |
| )](api/ptsbe_api.html#_CPPv4NK5cu | -   [cudaq::RemoteCapabilities    |
| daq15KrausTrajectory9isOrderedEv) |     (C++                          |
| -   [cudaq::                      |     struc                         |
| KrausTrajectory::kraus_selections | t)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq18RemoteCapabilitiesE) |
|     member)](api                  | -   [cudaq::Remo                  |
| /ptsbe_api.html#_CPPv4N5cudaq15Kr | teCapabilities::isRemoteSimulator |
| ausTrajectory16kraus_selectionsE) |     (C++                          |
| -   [cudaq::Kr                    |     member)](api/languages/c      |
| ausTrajectory::measurement_counts | pp_api.html#_CPPv4N5cudaq18Remote |
|     (C++                          | Capabilities17isRemoteSimulatorE) |
|     member)](api/p                | -   [cudaq::Remot                 |
| tsbe_api.html#_CPPv4N5cudaq15Krau | eCapabilities::RemoteCapabilities |
| sTrajectory18measurement_countsE) |     (C++                          |
| -   [cud                          |     function)](api/languages/cpp  |
| aq::KrausTrajectory::multiplicity | _api.html#_CPPv4N5cudaq18RemoteCa |
|     (C++                          | pabilities18RemoteCapabilitiesEb) |
|     member)]                      | -   [cudaq:                       |
| (api/ptsbe_api.html#_CPPv4N5cudaq | :RemoteCapabilities::stateOverlap |
| 15KrausTrajectory12multiplicityE) |     (C++                          |
| -   [                             |     member)](api/langua           |
| cudaq::KrausTrajectory::num_shots | ges/cpp_api.html#_CPPv4N5cudaq18R |
|     (C++                          | emoteCapabilities12stateOverlapE) |
|     memb                          | -                                 |
| er)](api/ptsbe_api.html#_CPPv4N5c |   [cudaq::RemoteCapabilities::vqe |
| udaq15KrausTrajectory9num_shotsE) |     (C++                          |
| -   [cu                           |     member)](                     |
| daq::KrausTrajectory::probability | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq18RemoteCapabilities3vqeE) |
|     member)                       | -   [cudaq::RemoteSimulationState |
| ](api/ptsbe_api.html#_CPPv4N5cuda |     (C++                          |
| q15KrausTrajectory11probabilityE) |     class)]                       |
| -   [cuda                         | (api/languages/cpp_api.html#_CPPv |
| q::KrausTrajectory::trajectory_id | 4N5cudaq21RemoteSimulationStateE) |
|     (C++                          | -   [cudaq::Resources (C++        |
|     member)](                     |     class)](api/languages/cpp_a   |
| api/ptsbe_api.html#_CPPv4N5cudaq1 | pi.html#_CPPv4N5cudaq9ResourcesE) |
| 5KrausTrajectory13trajectory_idE) | -   [cudaq::run (C++              |
| -   [cudaq::matrix_callback (C++  |     function)]                    |
|     c                             | (api/languages/cpp_api.html#_CPPv |
| lass)](api/languages/cpp_api.html | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| #_CPPv4N5cudaq15matrix_callbackE) | 5invoke_result_tINSt7decay_tI13Qu |
| -   [cudaq::matrix_handler (C++   | antumKernelEEDpNSt7decay_tI4ARGSE |
|                                   | EEEEENSt6size_tERN5cudaq11noise_m |
| class)](api/languages/cpp_api.htm | odelERR13QuantumKernelDpRR4ARGS), |
| l#_CPPv4N5cudaq14matrix_handlerE) |     [\[1\]](api/langu             |
| -   [cudaq::mat                   | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| rix_handler::commutation_behavior | daq3runENSt6vectorINSt15invoke_re |
|     (C++                          | sult_tINSt7decay_tI13QuantumKerne |
|     struct)](api/languages/       | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| cpp_api.html#_CPPv4N5cudaq14matri | ize_tERR13QuantumKernelDpRR4ARGS) |
| x_handler20commutation_behaviorE) | -   [cudaq::run_async (C++        |
| -                                 |     functio                       |
|    [cudaq::matrix_handler::define | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
|     function)](a                  | tureINSt6vectorINSt15invoke_resul |
| pi/languages/cpp_api.html#_CPPv4N | t_tINSt7decay_tI13QuantumKernelEE |
| 5cudaq14matrix_handler6defineENSt | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| 6stringENSt6vectorINSt7int64_tEEE | ze_tENSt6size_tERN5cudaq11noise_m |
| RR15matrix_callbackRKNSt13unorder | odelERR13QuantumKernelDpRR4ARGS), |
| ed_mapINSt6stringENSt6stringEEE), |     [\[1\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4I0DpEN |
| [\[1\]](api/languages/cpp_api.htm | 5cudaq9run_asyncENSt6futureINSt6v |
| l#_CPPv4N5cudaq14matrix_handler6d | ectorINSt15invoke_result_tINSt7de |
| efineENSt6stringENSt6vectorINSt7i | cay_tI13QuantumKernelEEDpNSt7deca |
| nt64_tEEERR15matrix_callbackRR20d | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| iag_matrix_callbackRKNSt13unorder | ize_tERR13QuantumKernelDpRR4ARGS) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cudaq::RuntimeTarget (C++    |
|     [\[2\]](                      |                                   |
| api/languages/cpp_api.html#_CPPv4 | struct)](api/languages/cpp_api.ht |
| N5cudaq14matrix_handler6defineENS | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| t6stringENSt6vectorINSt7int64_tEE | -   [cudaq::sample (C++           |
| ERR15matrix_callbackRRNSt13unorde |     function)](api/languages/c    |
| red_mapINSt6stringENSt6stringEEE) | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| -                                 | mpleE13sample_resultRK14sample_op |
|   [cudaq::matrix_handler::degrees | tionsRR13QuantumKernelDpRR4Args), |
|     (C++                          |     [\[1\                         |
|     function)](ap                 | ]](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4NK | Pv4I0DpEN5cudaq6sampleE13sample_r |
| 5cudaq14matrix_handler7degreesEv) | esultRR13QuantumKernelDpRR4Args), |
| -                                 |     [\                            |
|  [cudaq::matrix_handler::displace | [2\]](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|     function)](api/language       | ize_tERR13QuantumKernelDpRR4Args) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cudaq::sample_options (C++   |
| rix_handler8displaceENSt6size_tE) |     s                             |
| -   [cudaq::matrix                | truct)](api/languages/cpp_api.htm |
| _handler::get_expected_dimensions | l#_CPPv4N5cudaq14sample_optionsE) |
|     (C++                          | -   [cudaq::sample_result (C++    |
|                                   |                                   |
|    function)](api/languages/cpp_a |  class)](api/languages/cpp_api.ht |
| pi.html#_CPPv4NK5cudaq14matrix_ha | ml#_CPPv4N5cudaq13sample_resultE) |
| ndler23get_expected_dimensionsEv) | -   [cudaq::sample_result::append |
| -   [cudaq::matrix_ha             |     (C++                          |
| ndler::get_parameter_descriptions |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4N5cudaq13sample_re |
|                                   | sult6appendERK15ExecutionResultb) |
| function)](api/languages/cpp_api. | -   [cudaq::sample_result::begin  |
| html#_CPPv4NK5cudaq14matrix_handl |     (C++                          |
| er26get_parameter_descriptionsEv) |     function)]                    |
| -   [c                            | (api/languages/cpp_api.html#_CPPv |
| udaq::matrix_handler::instantiate | 4N5cudaq13sample_result5beginEv), |
|     (C++                          |     [\[1\]]                       |
|     function)](a                  | (api/languages/cpp_api.html#_CPPv |
| pi/languages/cpp_api.html#_CPPv4N | 4NK5cudaq13sample_result5beginEv) |
| 5cudaq14matrix_handler11instantia | -   [cudaq::sample_result::cbegin |
| teENSt6stringERKNSt6vectorINSt6si |     (C++                          |
| ze_tEEERK20commutation_behavior), |     function)](                   |
|     [\[1\]](                      | api/languages/cpp_api.html#_CPPv4 |
| api/languages/cpp_api.html#_CPPv4 | NK5cudaq13sample_result6cbeginEv) |
| N5cudaq14matrix_handler11instanti | -   [cudaq::sample_result::cend   |
| ateENSt6stringERRNSt6vectorINSt6s |     (C++                          |
| ize_tEEERK20commutation_behavior) |     function)                     |
| -   [cuda                         | ](api/languages/cpp_api.html#_CPP |
| q::matrix_handler::matrix_handler | v4NK5cudaq13sample_result4cendEv) |
|     (C++                          | -   [cudaq::sample_result::clear  |
|     function)](api/languag        |     (C++                          |
| es/cpp_api.html#_CPPv4I0_NSt11ena |     function)                     |
| ble_if_tINSt12is_base_of_vI16oper | ](api/languages/cpp_api.html#_CPP |
| ator_handler1TEEbEEEN5cudaq14matr | v4N5cudaq13sample_result5clearEv) |
| ix_handler14matrix_handlerERK1T), | -   [cudaq::sample_result::count  |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4I0 |     function)](                   |
| _NSt11enable_if_tINSt12is_base_of | api/languages/cpp_api.html#_CPPv4 |
| _vI16operator_handler1TEEbEEEN5cu | NK5cudaq13sample_result5countENSt |
| daq14matrix_handler14matrix_handl | 11string_viewEKNSt11string_viewE) |
| erERK1TRK20commutation_behavior), | -   [                             |
|     [\[2\]](api/languages/cpp_ap  | cudaq::sample_result::deserialize |
| i.html#_CPPv4N5cudaq14matrix_hand |     (C++                          |
| ler14matrix_handlerENSt6size_tE), |     functio                       |
|     [\[3\]](api/                  | n)](api/languages/cpp_api.html#_C |
| languages/cpp_api.html#_CPPv4N5cu | PPv4N5cudaq13sample_result11deser |
| daq14matrix_handler14matrix_handl | ializeERNSt6vectorINSt6size_tEEE) |
| erENSt6stringERKNSt6vectorINSt6si | -   [cudaq::sample_result::dump   |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[4\]](api/                  |     function)](api/languag        |
| languages/cpp_api.html#_CPPv4N5cu | es/cpp_api.html#_CPPv4NK5cudaq13s |
| daq14matrix_handler14matrix_handl | ample_result4dumpERNSt7ostreamE), |
| erENSt6stringERRNSt6vectorINSt6si |     [\[1\]                        |
| ze_tEEERK20commutation_behavior), | ](api/languages/cpp_api.html#_CPP |
|     [\                            | v4NK5cudaq13sample_result4dumpEv) |
| [5\]](api/languages/cpp_api.html# | -   [cudaq::sample_result::end    |
| _CPPv4N5cudaq14matrix_handler14ma |     (C++                          |
| trix_handlerERK14matrix_handler), |     function                      |
|     [                             | )](api/languages/cpp_api.html#_CP |
| \[6\]](api/languages/cpp_api.html | Pv4N5cudaq13sample_result3endEv), |
| #_CPPv4N5cudaq14matrix_handler14m |     [\[1\                         |
| atrix_handlerERR14matrix_handler) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4NK5cudaq13sample_result3endEv) |
|  [cudaq::matrix_handler::momentum | -   [                             |
|     (C++                          | cudaq::sample_result::expectation |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     f                             |
| rix_handler8momentumENSt6size_tE) | unction)](api/languages/cpp_api.h |
| -                                 | tml#_CPPv4NK5cudaq13sample_result |
|    [cudaq::matrix_handler::number | 11expectationEKNSt11string_viewE) |
|     (C++                          | -   [c                            |
|     function)](api/langua         | udaq::sample_result::get_marginal |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     (C++                          |
| atrix_handler6numberENSt6size_tE) |     function)](api/languages/cpp_ |
| -                                 | api.html#_CPPv4NK5cudaq13sample_r |
| [cudaq::matrix_handler::operator= | esult12get_marginalERKNSt6vectorI |
|     (C++                          | NSt6size_tEEEKNSt11string_viewE), |
|     fun                           |     [\[1\]](api/languages/cpp_    |
| ction)](api/languages/cpp_api.htm | api.html#_CPPv4NK5cudaq13sample_r |
| l#_CPPv4I0_NSt11enable_if_tIXaant | esult12get_marginalERRKNSt6vector |
| NSt7is_sameI1T14matrix_handlerE5v | INSt6size_tEEEKNSt11string_viewE) |
| alueENSt12is_base_of_vI16operator | -   [cuda                         |
| _handler1TEEEbEEEN5cudaq14matrix_ | q::sample_result::get_total_shots |
| handleraSER14matrix_handlerRK1T), |     (C++                          |
|     [\[1\]](api/languages         |     function)](api/langua         |
| /cpp_api.html#_CPPv4N5cudaq14matr | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| ix_handleraSERK14matrix_handler), | sample_result15get_total_shotsEv) |
|     [\[2\]](api/language          | -   [cuda                         |
| s/cpp_api.html#_CPPv4N5cudaq14mat | q::sample_result::has_even_parity |
| rix_handleraSERR14matrix_handler) |     (C++                          |
| -   [                             |     fun                           |
| cudaq::matrix_handler::operator== | ction)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq13sample_result15h |
|     function)](api/languages      | as_even_parityENSt11string_viewE) |
| /cpp_api.html#_CPPv4NK5cudaq14mat | -   [cuda                         |
| rix_handlereqERK14matrix_handler) | q::sample_result::has_expectation |
| -                                 |     (C++                          |
|    [cudaq::matrix_handler::parity |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/langua         | _CPPv4NK5cudaq13sample_result15ha |
| ges/cpp_api.html#_CPPv4N5cudaq14m | s_expectationEKNSt11string_viewE) |
| atrix_handler6parityENSt6size_tE) | -   [cu                           |
| -                                 | daq::sample_result::most_probable |
|  [cudaq::matrix_handler::position |     (C++                          |
|     (C++                          |     fun                           |
|     function)](api/language       | ction)](api/languages/cpp_api.htm |
| s/cpp_api.html#_CPPv4N5cudaq14mat | l#_CPPv4NK5cudaq13sample_result13 |
| rix_handler8positionENSt6size_tE) | most_probableEKNSt11string_viewE) |
| -   [cudaq::                      | -                                 |
| matrix_handler::remove_definition | [cudaq::sample_result::operator+= |
|     (C++                          |     (C++                          |
|     fu                            |     function)](api/langua         |
| nction)](api/languages/cpp_api.ht | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ml#_CPPv4N5cudaq14matrix_handler1 | ample_resultpLERK13sample_result) |
| 7remove_definitionERKNSt6stringE) | -                                 |
| -                                 |  [cudaq::sample_result::operator= |
|   [cudaq::matrix_handler::squeeze |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     function)](api/languag        | ges/cpp_api.html#_CPPv4N5cudaq13s |
| es/cpp_api.html#_CPPv4N5cudaq14ma | ample_resultaSERR13sample_result) |
| trix_handler7squeezeENSt6size_tE) | -                                 |
| -   [cudaq::m                     | [cudaq::sample_result::operator== |
| atrix_handler::to_diagonal_matrix |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)](api/lang           | es/cpp_api.html#_CPPv4NK5cudaq13s |
| uages/cpp_api.html#_CPPv4NK5cudaq | ample_resulteqERK13sample_result) |
| 14matrix_handler18to_diagonal_mat | -   [                             |
| rixERNSt13unordered_mapINSt6size_ | cudaq::sample_result::probability |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |     function)](api/lan            |
| -                                 | guages/cpp_api.html#_CPPv4NK5cuda |
| [cudaq::matrix_handler::to_matrix | q13sample_result11probabilityENSt |
|     (C++                          | 11string_viewEKNSt11string_viewE) |
|     function)                     | -   [cud                          |
| ](api/languages/cpp_api.html#_CPP | aq::sample_result::register_names |
| v4NK5cudaq14matrix_handler9to_mat |     (C++                          |
| rixERNSt13unordered_mapINSt6size_ |     function)](api/langu          |
| tENSt7int64_tEEERKNSt13unordered_ | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| mapINSt6stringENSt7complexIdEEEE) | 3sample_result14register_namesEv) |
| -                                 | -                                 |
| [cudaq::matrix_handler::to_string |    [cudaq::sample_result::reorder |
|     (C++                          |     (C++                          |
|     function)](api/               |     function)](api/langua         |
| languages/cpp_api.html#_CPPv4NK5c | ges/cpp_api.html#_CPPv4N5cudaq13s |
| udaq14matrix_handler9to_stringEb) | ample_result7reorderERKNSt6vector |
| -                                 | INSt6size_tEEEKNSt11string_viewE) |
| [cudaq::matrix_handler::unique_id | -   [cu                           |
|     (C++                          | daq::sample_result::sample_result |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |     func                          |
| udaq14matrix_handler9unique_idEv) | tion)](api/languages/cpp_api.html |
| -   [cudaq:                       | #_CPPv4N5cudaq13sample_result13sa |
| :matrix_handler::\~matrix_handler | mple_resultERK15ExecutionResult), |
|     (C++                          |     [\[1\]](api/la                |
|     functi                        | nguages/cpp_api.html#_CPPv4N5cuda |
| on)](api/languages/cpp_api.html#_ | q13sample_result13sample_resultER |
| CPPv4N5cudaq14matrix_handlerD0Ev) | KNSt6vectorI15ExecutionResultEE), |
| -   [cudaq::matrix_op (C++        |                                   |
|     type)](api/languages/cpp_a    |  [\[2\]](api/languages/cpp_api.ht |
| pi.html#_CPPv4N5cudaq9matrix_opE) | ml#_CPPv4N5cudaq13sample_result13 |
| -   [cudaq::matrix_op_term (C++   | sample_resultERR13sample_result), |
|                                   |     [                             |
|  type)](api/languages/cpp_api.htm | \[3\]](api/languages/cpp_api.html |
| l#_CPPv4N5cudaq14matrix_op_termE) | #_CPPv4N5cudaq13sample_result13sa |
| -                                 | mple_resultERR15ExecutionResult), |
|    [cudaq::mdiag_operator_handler |     [\[4\]](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     class)](                      | 13sample_result13sample_resultEdR |
| api/languages/cpp_api.html#_CPPv4 | KNSt6vectorI15ExecutionResultEE), |
| N5cudaq22mdiag_operator_handlerE) |     [\[5\]](api/lan               |
| -   [cudaq::mpi (C++              | guages/cpp_api.html#_CPPv4N5cudaq |
|     type)](api/languages          | 13sample_result13sample_resultEv) |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | -                                 |
| -   [cudaq::mpi::all_gather (C++  |  [cudaq::sample_result::serialize |
|     fu                            |     (C++                          |
| nction)](api/languages/cpp_api.ht |     function)](api                |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | /languages/cpp_api.html#_CPPv4NK5 |
| RNSt6vectorIdEERKNSt6vectorIdEE), | cudaq13sample_result9serializeEv) |
|                                   | -   [cudaq::sample_result::size   |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi10all_gather |     function)](api/languages/c    |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | pp_api.html#_CPPv4NK5cudaq13sampl |
| -   [cudaq::mpi::all_reduce (C++  | e_result4sizeEKNSt11string_viewE) |
|                                   | -   [cudaq::sample_result::to_map |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     function)](api/languages/cpp  |
| reduceE1TRK1TRK14BinaryFunction), | _api.html#_CPPv4NK5cudaq13sample_ |
|     [\[1\]](api/langu             | result6to_mapEKNSt11string_viewE) |
| ages/cpp_api.html#_CPPv4I00EN5cud | -   [cuda                         |
| aq3mpi10all_reduceE1TRK1TRK4Func) | q::sample_result::\~sample_result |
| -   [cudaq::mpi::broadcast (C++   |     (C++                          |
|     function)](api/               |     funct                         |
| languages/cpp_api.html#_CPPv4N5cu | ion)](api/languages/cpp_api.html# |
| daq3mpi9broadcastERNSt6stringEi), | _CPPv4N5cudaq13sample_resultD0Ev) |
|     [\[1\]](api/la                | -   [cudaq::scalar_callback (C++  |
| nguages/cpp_api.html#_CPPv4N5cuda |     c                             |
| q3mpi9broadcastERNSt6vectorIdEEi) | lass)](api/languages/cpp_api.html |
| -   [cudaq::mpi::finalize (C++    | #_CPPv4N5cudaq15scalar_callbackE) |
|     f                             | -   [c                            |
| unction)](api/languages/cpp_api.h | udaq::scalar_callback::operator() |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     (C++                          |
| -   [cudaq::mpi::initialize (C++  |     function)](api/language       |
|     function                      | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| )](api/languages/cpp_api.html#_CP | alar_callbackclERKNSt13unordered_ |
| Pv4N5cudaq3mpi10initializeEiPPc), | mapINSt6stringENSt7complexIdEEEE) |
|     [                             | -   [                             |
| \[1\]](api/languages/cpp_api.html | cudaq::scalar_callback::operator= |
| #_CPPv4N5cudaq3mpi10initializeEv) |     (C++                          |
| -   [cudaq::mpi::is_initialized   |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function                      | _callbackaSERK15scalar_callback), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]](api/languages/        |
| Pv4N5cudaq3mpi14is_initializedEv) | cpp_api.html#_CPPv4N5cudaq15scala |
| -   [cudaq::mpi::num_ranks (C++   | r_callbackaSERR15scalar_callback) |
|     fu                            | -   [cudaq:                       |
| nction)](api/languages/cpp_api.ht | :scalar_callback::scalar_callback |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     (C++                          |
| -   [cudaq::mpi::rank (C++        |     function)](api/languag        |
|                                   | es/cpp_api.html#_CPPv4I0_NSt11ena |
|    function)](api/languages/cpp_a | ble_if_tINSt16is_invocable_r_vINS |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | t7complexIdEE8CallableRKNSt13unor |
| -   [cudaq::noise_model (C++      | dered_mapINSt6stringENSt7complexI |
|                                   | dEEEEEEbEEEN5cudaq15scalar_callba |
|    class)](api/languages/cpp_api. | ck15scalar_callbackERR8Callable), |
| html#_CPPv4N5cudaq11noise_modelE) |     [\[1\                         |
| -   [cudaq::n                     | ]](api/languages/cpp_api.html#_CP |
| oise_model::add_all_qubit_channel | Pv4N5cudaq15scalar_callback15scal |
|     (C++                          | ar_callbackERK15scalar_callback), |
|     function)](api                |     [\[2                          |
| /languages/cpp_api.html#_CPPv4IDp | \]](api/languages/cpp_api.html#_C |
| EN5cudaq11noise_model21add_all_qu | PPv4N5cudaq15scalar_callback15sca |
| bit_channelEvRK13kraus_channeli), | lar_callbackERR15scalar_callback) |
|     [\[1\]](api/langua            | -   [cudaq::scalar_operator (C++  |
| ges/cpp_api.html#_CPPv4N5cudaq11n |     c                             |
| oise_model21add_all_qubit_channel | lass)](api/languages/cpp_api.html |
| ERKNSt6stringERK13kraus_channeli) | #_CPPv4N5cudaq15scalar_operatorE) |
| -                                 | -                                 |
|  [cudaq::noise_model::add_channel | [cudaq::scalar_operator::evaluate |
|     (C++                          |     (C++                          |
|     funct                         |                                   |
| ion)](api/languages/cpp_api.html# |    function)](api/languages/cpp_a |
| _CPPv4IDpEN5cudaq11noise_model11a | pi.html#_CPPv4NK5cudaq15scalar_op |
| dd_channelEvRK15PredicateFuncTy), | erator8evaluateERKNSt13unordered_ |
|     [\[1\]](api/languages/cpp_    | mapINSt6stringENSt7complexIdEEEE) |
| api.html#_CPPv4IDpEN5cudaq11noise | -   [cudaq::scalar_ope            |
| _model11add_channelEvRKNSt6vector | rator::get_parameter_descriptions |
| INSt6size_tEEERK13kraus_channel), |     (C++                          |
|     [\[2\]](ap                    |     f                             |
| i/languages/cpp_api.html#_CPPv4N5 | unction)](api/languages/cpp_api.h |
| cudaq11noise_model11add_channelER | tml#_CPPv4NK5cudaq15scalar_operat |
| KNSt6stringERK15PredicateFuncTy), | or26get_parameter_descriptionsEv) |
|                                   | -   [cu                           |
| [\[3\]](api/languages/cpp_api.htm | daq::scalar_operator::is_constant |
| l#_CPPv4N5cudaq11noise_model11add |     (C++                          |
| _channelERKNSt6stringERKNSt6vecto |     function)](api/lang           |
| rINSt6size_tEEERK13kraus_channel) | uages/cpp_api.html#_CPPv4NK5cudaq |
| -   [cudaq::noise_model::empty    | 15scalar_operator11is_constantEv) |
|     (C++                          | -   [c                            |
|     function                      | udaq::scalar_operator::operator\* |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4NK5cudaq11noise_model5emptyEv) |     function                      |
| -                                 | )](api/languages/cpp_api.html#_CP |
| [cudaq::noise_model::get_channels | Pv4N5cudaq15scalar_operatormlENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     function)](api/l              |     [\[1\                         |
| anguages/cpp_api.html#_CPPv4I0ENK | ]](api/languages/cpp_api.html#_CP |
| 5cudaq11noise_model12get_channels | Pv4N5cudaq15scalar_operatormlENSt |
| ENSt6vectorI13kraus_channelEERKNS | 7complexIdEERR15scalar_operator), |
| t6vectorINSt6size_tEEERKNSt6vecto |     [\[2\]](api/languages/cp      |
| rINSt6size_tEEERKNSt6vectorIdEE), | p_api.html#_CPPv4N5cudaq15scalar_ |
|     [\[1\]](api/languages/cpp_a   | operatormlEdRK15scalar_operator), |
| pi.html#_CPPv4NK5cudaq11noise_mod |     [\[3\]](api/languages/cp      |
| el12get_channelsERKNSt6stringERKN | p_api.html#_CPPv4N5cudaq15scalar_ |
| St6vectorINSt6size_tEEERKNSt6vect | operatormlEdRR15scalar_operator), |
| orINSt6size_tEEERKNSt6vectorIdEE) |     [\[4\]](api/languages         |
| -                                 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|  [cudaq::noise_model::noise_model | alar_operatormlENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     function)](api                | _api.html#_CPPv4NKR5cudaq15scalar |
| /languages/cpp_api.html#_CPPv4N5c | _operatormlERK15scalar_operator), |
| udaq11noise_model11noise_modelEv) |     [\[6\]]                       |
| -   [cu                           | (api/languages/cpp_api.html#_CPPv |
| daq::noise_model::PredicateFuncTy | 4NKR5cudaq15scalar_operatormlEd), |
|     (C++                          |     [\[7\]](api/language          |
|     type)](api/la                 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| nguages/cpp_api.html#_CPPv4N5cuda | alar_operatormlENSt7complexIdEE), |
| q11noise_model15PredicateFuncTyE) |     [\[8\]](api/languages/cp      |
| -   [cud                          | p_api.html#_CPPv4NO5cudaq15scalar |
| aq::noise_model::register_channel | _operatormlERK15scalar_operator), |
|     (C++                          |     [\[9\                         |
|     function)](api/languages      | ]](api/languages/cpp_api.html#_CP |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | Pv4NO5cudaq15scalar_operatormlEd) |
| noise_model16register_channelEvv) | -   [cu                           |
| -   [cudaq::                      | daq::scalar_operator::operator\*= |
| noise_model::requires_constructor |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     type)](api/languages/cp       | es/cpp_api.html#_CPPv4N5cudaq15sc |
| p_api.html#_CPPv4I0DpEN5cudaq11no | alar_operatormLENSt7complexIdEE), |
| ise_model20requires_constructorE) |     [\[1\]](api/languages/c       |
| -   [cudaq::noise_model_type (C++ | pp_api.html#_CPPv4N5cudaq15scalar |
|     e                             | _operatormLERK15scalar_operator), |
| num)](api/languages/cpp_api.html# |     [\[2                          |
| _CPPv4N5cudaq16noise_model_typeE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::no                    | PPv4N5cudaq15scalar_operatormLEd) |
| ise_model_type::amplitude_damping | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator+ |
|     enumerator)](api/languages    |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq16nois |     function                      |
| e_model_type17amplitude_dampingE) | )](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise_mode            | Pv4N5cudaq15scalar_operatorplENSt |
| l_type::amplitude_damping_channel | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     e                             | ]](api/languages/cpp_api.html#_CP |
| numerator)](api/languages/cpp_api | Pv4N5cudaq15scalar_operatorplENSt |
| .html#_CPPv4N5cudaq16noise_model_ | 7complexIdEERR15scalar_operator), |
| type25amplitude_damping_channelE) |     [\[2\]](api/languages/cp      |
| -   [cudaq::n                     | p_api.html#_CPPv4N5cudaq15scalar_ |
| oise_model_type::bit_flip_channel | operatorplEdRK15scalar_operator), |
|     (C++                          |     [\[3\]](api/languages/cp      |
|     enumerator)](api/language     | p_api.html#_CPPv4N5cudaq15scalar_ |
| s/cpp_api.html#_CPPv4N5cudaq16noi | operatorplEdRR15scalar_operator), |
| se_model_type16bit_flip_channelE) |     [\[4\]](api/languages         |
| -   [cudaq::                      | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| noise_model_type::depolarization1 | alar_operatorplENSt7complexIdEE), |
|     (C++                          |     [\[5\]](api/languages/cpp     |
|     enumerator)](api/languag      | _api.html#_CPPv4NKR5cudaq15scalar |
| es/cpp_api.html#_CPPv4N5cudaq16no | _operatorplERK15scalar_operator), |
| ise_model_type15depolarization1E) |     [\[6\]]                       |
| -   [cudaq::                      | (api/languages/cpp_api.html#_CPPv |
| noise_model_type::depolarization2 | 4NKR5cudaq15scalar_operatorplEd), |
|     (C++                          |     [\[7\]]                       |
|     enumerator)](api/languag      | (api/languages/cpp_api.html#_CPPv |
| es/cpp_api.html#_CPPv4N5cudaq16no | 4NKR5cudaq15scalar_operatorplEv), |
| ise_model_type15depolarization2E) |     [\[8\]](api/language          |
| -   [cudaq::noise_m               | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| odel_type::depolarization_channel | alar_operatorplENSt7complexIdEE), |
|     (C++                          |     [\[9\]](api/languages/cp      |
|                                   | p_api.html#_CPPv4NO5cudaq15scalar |
|   enumerator)](api/languages/cpp_ | _operatorplERK15scalar_operator), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[10\]                       |
| el_type22depolarization_channelE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4NO5cudaq15scalar_operatorplEd), |
|  [cudaq::noise_model_type::pauli1 |     [\[11\                        |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     enumerator)](a                | Pv4NO5cudaq15scalar_operatorplEv) |
| pi/languages/cpp_api.html#_CPPv4N | -   [c                            |
| 5cudaq16noise_model_type6pauli1E) | udaq::scalar_operator::operator+= |
| -                                 |     (C++                          |
|  [cudaq::noise_model_type::pauli2 |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     enumerator)](a                | alar_operatorpLENSt7complexIdEE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[1\]](api/languages/c       |
| 5cudaq16noise_model_type6pauli2E) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq                        | _operatorpLERK15scalar_operator), |
| ::noise_model_type::phase_damping |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     enumerator)](api/langu        | PPv4N5cudaq15scalar_operatorpLEd) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [                             |
| noise_model_type13phase_dampingE) | cudaq::scalar_operator::operator- |
| -   [cudaq::noi                   |     (C++                          |
| se_model_type::phase_flip_channel |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     enumerator)](api/languages/   | Pv4N5cudaq15scalar_operatormiENSt |
| cpp_api.html#_CPPv4N5cudaq16noise | 7complexIdEERK15scalar_operator), |
| _model_type18phase_flip_channelE) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
| [cudaq::noise_model_type::unknown | Pv4N5cudaq15scalar_operatormiENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     enumerator)](ap               |     [\[2\]](api/languages/cp      |
| i/languages/cpp_api.html#_CPPv4N5 | p_api.html#_CPPv4N5cudaq15scalar_ |
| cudaq16noise_model_type7unknownE) | operatormiEdRK15scalar_operator), |
| -                                 |     [\[3\]](api/languages/cp      |
| [cudaq::noise_model_type::x_error | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRR15scalar_operator), |
|     enumerator)](ap               |     [\[4\]](api/languages         |
| i/languages/cpp_api.html#_CPPv4N5 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| cudaq16noise_model_type7x_errorE) | alar_operatormiENSt7complexIdEE), |
| -                                 |     [\[5\]](api/languages/cpp     |
| [cudaq::noise_model_type::y_error | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatormiERK15scalar_operator), |
|     enumerator)](ap               |     [\[6\]]                       |
| i/languages/cpp_api.html#_CPPv4N5 | (api/languages/cpp_api.html#_CPPv |
| cudaq16noise_model_type7y_errorE) | 4NKR5cudaq15scalar_operatormiEd), |
| -                                 |     [\[7\]]                       |
| [cudaq::noise_model_type::z_error | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEv), |
|     enumerator)](ap               |     [\[8\]](api/language          |
| i/languages/cpp_api.html#_CPPv4N5 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| cudaq16noise_model_type7z_errorE) | alar_operatormiENSt7complexIdEE), |
| -   [cudaq::num_available_gpus    |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     function                      | _operatormiERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[10\]                       |
| Pv4N5cudaq18num_available_gpusEv) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::observe (C++          | v4NO5cudaq15scalar_operatormiEd), |
|     function)]                    |     [\[11\                        |
| (api/languages/cpp_api.html#_CPPv | ]](api/languages/cpp_api.html#_CP |
| 4I00DpEN5cudaq7observeENSt6vector | Pv4NO5cudaq15scalar_operatormiEv) |
| I14observe_resultEERR13QuantumKer | -   [c                            |
| nelRK15SpinOpContainerDpRR4Args), | udaq::scalar_operator::operator-= |
|     [\[1\]](api/languages/cpp_ap  |     (C++                          |
| i.html#_CPPv4I0DpEN5cudaq7observe |     function)](api/languag        |
| E14observe_resultNSt6size_tERR13Q | es/cpp_api.html#_CPPv4N5cudaq15sc |
| uantumKernelRK7spin_opDpRR4Args), | alar_operatormIENSt7complexIdEE), |
|     [\[                           |     [\[1\]](api/languages/c       |
| 2\]](api/languages/cpp_api.html#_ | pp_api.html#_CPPv4N5cudaq15scalar |
| CPPv4I0DpEN5cudaq7observeE14obser | _operatormIERK15scalar_operator), |
| ve_resultRK15observe_optionsRR13Q |     [\[2                          |
| uantumKernelRK7spin_opDpRR4Args), | \]](api/languages/cpp_api.html#_C |
|     [\[3\]](api/lang              | PPv4N5cudaq15scalar_operatormIEd) |
| uages/cpp_api.html#_CPPv4I0DpEN5c | -   [                             |
| udaq7observeE14observe_resultRR13 | cudaq::scalar_operator::operator/ |
| QuantumKernelRK7spin_opDpRR4Args) |     (C++                          |
| -   [cudaq::observe_options (C++  |     function                      |
|     st                            | )](api/languages/cpp_api.html#_CP |
| ruct)](api/languages/cpp_api.html | Pv4N5cudaq15scalar_operatordvENSt |
| #_CPPv4N5cudaq15observe_optionsE) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::observe_result (C++   |     [\[1\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
| class)](api/languages/cpp_api.htm | Pv4N5cudaq15scalar_operatordvENSt |
| l#_CPPv4N5cudaq14observe_resultE) | 7complexIdEERR15scalar_operator), |
| -                                 |     [\[2\]](api/languages/cp      |
|    [cudaq::observe_result::counts | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRK15scalar_operator), |
|     function)](api/languages/c    |     [\[3\]](api/languages/cp      |
| pp_api.html#_CPPv4N5cudaq14observ | p_api.html#_CPPv4N5cudaq15scalar_ |
| e_result6countsERK12spin_op_term) | operatordvEdRR15scalar_operator), |
| -   [cudaq::observe_result::dump  |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     function)                     | alar_operatordvENSt7complexIdEE), |
| ](api/languages/cpp_api.html#_CPP |     [\[5\]](api/languages/cpp     |
| v4N5cudaq14observe_result4dumpEv) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [c                            | _operatordvERK15scalar_operator), |
| udaq::observe_result::expectation |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NKR5cudaq15scalar_operatordvEd), |
| function)](api/languages/cpp_api. |     [\[7\]](api/language          |
| html#_CPPv4N5cudaq14observe_resul | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| t11expectationERK12spin_op_term), | alar_operatordvENSt7complexIdEE), |
|     [\[1\]](api/la                |     [\[8\]](api/languages/cp      |
| nguages/cpp_api.html#_CPPv4N5cuda | p_api.html#_CPPv4NO5cudaq15scalar |
| q14observe_result11expectationEv) | _operatordvERK15scalar_operator), |
| -   [cuda                         |     [\[9\                         |
| q::observe_result::id_coefficient | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4NO5cudaq15scalar_operatordvEd) |
|     function)](api/langu          | -   [c                            |
| ages/cpp_api.html#_CPPv4N5cudaq14 | udaq::scalar_operator::operator/= |
| observe_result14id_coefficientEv) |     (C++                          |
| -   [cuda                         |     function)](api/languag        |
| q::observe_result::observe_result | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     (C++                          | alar_operatordVENSt7complexIdEE), |
|                                   |     [\[1\]](api/languages/c       |
|   function)](api/languages/cpp_ap | pp_api.html#_CPPv4N5cudaq15scalar |
| i.html#_CPPv4N5cudaq14observe_res | _operatordVERK15scalar_operator), |
| ult14observe_resultEdRK7spin_op), |     [\[2                          |
|     [\[1\]](a                     | \]](api/languages/cpp_api.html#_C |
| pi/languages/cpp_api.html#_CPPv4N | PPv4N5cudaq15scalar_operatordVEd) |
| 5cudaq14observe_result14observe_r | -   [                             |
| esultEdRK7spin_op13sample_result) | cudaq::scalar_operator::operator= |
| -                                 |     (C++                          |
|  [cudaq::observe_result::operator |     function)](api/languages/c    |
|     double (C++                   | pp_api.html#_CPPv4N5cudaq15scalar |
|     functio                       | _operatoraSERK15scalar_operator), |
| n)](api/languages/cpp_api.html#_C |     [\[1\]](api/languages/        |
| PPv4N5cudaq14observe_resultcvdEv) | cpp_api.html#_CPPv4N5cudaq15scala |
| -                                 | r_operatoraSERR15scalar_operator) |
|  [cudaq::observe_result::raw_data | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator== |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languages/c    |
| cudaq14observe_result8raw_dataEv) | pp_api.html#_CPPv4NK5cudaq15scala |
| -   [cudaq::operator_handler (C++ | r_operatoreqERK15scalar_operator) |
|     cl                            | -   [cudaq:                       |
| ass)](api/languages/cpp_api.html# | :scalar_operator::scalar_operator |
| _CPPv4N5cudaq16operator_handlerE) |     (C++                          |
| -   [cudaq::optimizable_function  |     func                          |
|     (C++                          | tion)](api/languages/cpp_api.html |
|     class)                        | #_CPPv4N5cudaq15scalar_operator15 |
| ](api/languages/cpp_api.html#_CPP | scalar_operatorENSt7complexIdEE), |
| v4N5cudaq20optimizable_functionE) |     [\[1\]](api/langu             |
| -   [cudaq::optimization_result   | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     (C++                          | scalar_operator15scalar_operatorE |
|     type                          | RK15scalar_callbackRRNSt13unorder |
| )](api/languages/cpp_api.html#_CP | ed_mapINSt6stringENSt6stringEEE), |
| Pv4N5cudaq19optimization_resultE) |     [\[2\                         |
| -   [cudaq::optimizer (C++        | ]](api/languages/cpp_api.html#_CP |
|     class)](api/languages/cpp_a   | Pv4N5cudaq15scalar_operator15scal |
| pi.html#_CPPv4N5cudaq9optimizerE) | ar_operatorERK15scalar_operator), |
| -   [cudaq::optimizer::optimize   |     [\[3\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|                                   | scalar_operator15scalar_operatorE |
|  function)](api/languages/cpp_api | RR15scalar_callbackRRNSt13unorder |
| .html#_CPPv4N5cudaq9optimizer8opt | ed_mapINSt6stringENSt6stringEEE), |
| imizeEKiRR20optimizable_function) |     [\[4\                         |
| -   [cu                           | ]](api/languages/cpp_api.html#_CP |
| daq::optimizer::requiresGradients | Pv4N5cudaq15scalar_operator15scal |
|     (C++                          | ar_operatorERR15scalar_operator), |
|     function)](api/la             |     [\[5\]](api/language          |
| nguages/cpp_api.html#_CPPv4N5cuda | s/cpp_api.html#_CPPv4N5cudaq15sca |
| q9optimizer17requiresGradientsEv) | lar_operator15scalar_operatorEd), |
| -   [cudaq::orca (C++             |     [\[6\]](api/languag           |
|     type)](api/languages/         | es/cpp_api.html#_CPPv4N5cudaq15sc |
| cpp_api.html#_CPPv4N5cudaq4orcaE) | alar_operator15scalar_operatorEv) |
| -   [cudaq::orca::sample (C++     | -   [                             |
|     function)](api/languages/c    | cudaq::scalar_operator::to_matrix |
| pp_api.html#_CPPv4N5cudaq4orca6sa |     (C++                          |
| mpleERNSt6vectorINSt6size_tEEERNS |                                   |
| t6vectorINSt6size_tEEERNSt6vector |   function)](api/languages/cpp_ap |
| IdEERNSt6vectorIdEEiNSt6size_tE), | i.html#_CPPv4NK5cudaq15scalar_ope |
|     [\[1\]]                       | rator9to_matrixERKNSt13unordered_ |
| (api/languages/cpp_api.html#_CPPv | mapINSt6stringENSt7complexIdEEEE) |
| 4N5cudaq4orca6sampleERNSt6vectorI | -   [                             |
| NSt6size_tEEERNSt6vectorINSt6size | cudaq::scalar_operator::to_string |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     (C++                          |
| -   [cudaq::orca::sample_async    |     function)](api/l              |
|     (C++                          | anguages/cpp_api.html#_CPPv4NK5cu |
|                                   | daq15scalar_operator9to_stringEv) |
| function)](api/languages/cpp_api. | -   [cudaq::s                     |
| html#_CPPv4N5cudaq4orca12sample_a | calar_operator::\~scalar_operator |
| syncERNSt6vectorINSt6size_tEEERNS |     (C++                          |
| t6vectorINSt6size_tEEERNSt6vector |     functio                       |
| IdEERNSt6vectorIdEEiNSt6size_tE), | n)](api/languages/cpp_api.html#_C |
|     [\[1\]](api/la                | PPv4N5cudaq15scalar_operatorD0Ev) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::set_noise (C++        |
| q4orca12sample_asyncERNSt6vectorI |     function)](api/langu          |
| NSt6size_tEEERNSt6vectorINSt6size | ages/cpp_api.html#_CPPv4N5cudaq9s |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | et_noiseERKN5cudaq11noise_modelE) |
| -   [cudaq::OrcaRemoteRESTQPU     | -   [cudaq::set_random_seed (C++  |
|     (C++                          |     function)](api/               |
|     cla                           | languages/cpp_api.html#_CPPv4N5cu |
| ss)](api/languages/cpp_api.html#_ | daq15set_random_seedENSt6size_tE) |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | -   [cudaq::simulation_precision  |
| -   [cudaq::pauli1 (C++           |     (C++                          |
|     class)](api/languages/cp      |     enum)                         |
| p_api.html#_CPPv4N5cudaq6pauli1E) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4N5cudaq20simulation_precisionE) |
|    [cudaq::pauli1::num_parameters | -   [                             |
|     (C++                          | cudaq::simulation_precision::fp32 |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     enumerator)](api              |
| 4N5cudaq6pauli114num_parametersE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::pauli1::num_targets   | udaq20simulation_precision4fp32E) |
|     (C++                          | -   [                             |
|     membe                         | cudaq::simulation_precision::fp64 |
| r)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq6pauli111num_targetsE) |     enumerator)](api              |
| -   [cudaq::pauli1::pauli1 (C++   | /languages/cpp_api.html#_CPPv4N5c |
|     function)](api/languages/cpp_ | udaq20simulation_precision4fp64E) |
| api.html#_CPPv4N5cudaq6pauli16pau | -   [cudaq::SimulationState (C++  |
| li1ERKNSt6vectorIN5cudaq4realEEE) |     c                             |
| -   [cudaq::pauli2 (C++           | lass)](api/languages/cpp_api.html |
|     class)](api/languages/cp      | #_CPPv4N5cudaq15SimulationStateE) |
| p_api.html#_CPPv4N5cudaq6pauli2E) | -   [                             |
| -                                 | cudaq::SimulationState::precision |
|    [cudaq::pauli2::num_parameters |     (C++                          |
|     (C++                          |     enum)](api                    |
|     member)]                      | /languages/cpp_api.html#_CPPv4N5c |
| (api/languages/cpp_api.html#_CPPv | udaq15SimulationState9precisionE) |
| 4N5cudaq6pauli214num_parametersE) | -   [cudaq:                       |
| -   [cudaq::pauli2::num_targets   | :SimulationState::precision::fp32 |
|     (C++                          |     (C++                          |
|     membe                         |     enumerator)](api/lang         |
| r)](api/languages/cpp_api.html#_C | uages/cpp_api.html#_CPPv4N5cudaq1 |
| PPv4N5cudaq6pauli211num_targetsE) | 5SimulationState9precision4fp32E) |
|                                   | -   [cudaq:                       |
|                                   | :SimulationState::precision::fp64 |
|                                   |     (C++                          |
|                                   |     enumerator)](api/lang         |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq1 |
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
| stom)](api/languages/python_api.h | -   [execution_data()             |
| tml#cudaq.operators.custom.empty) |                                   |
|     -   [(in module               |    (cudaq.ptsbe.PTSBESampleResult |
|                                   |     method)                       |
|       cudaq.spin)](api/languages/ | ](api/ptsbe_api.html#cudaq.ptsbe. |
| python_api.html#cudaq.spin.empty) | PTSBESampleResult.execution_data) |
| -   [empty_op()                   | -   [expectation()                |
|     (                             |     (cudaq.ObserveResult          |
| cudaq.operators.spin.SpinOperator |     metho                         |
|     static                        | d)](api/languages/python_api.html |
|     method)](api/lan              | #cudaq.ObserveResult.expectation) |
| guages/python_api.html#cudaq.oper |     -   [(cudaq.SampleResult      |
| ators.spin.SpinOperator.empty_op) |         meth                      |
| -   [enable_return_to_log()       | od)](api/languages/python_api.htm |
|     (cudaq.PyKernelDecorator      | l#cudaq.SampleResult.expectation) |
|     method)](api/langu            | -   [expectation_values()         |
| ages/python_api.html#cudaq.PyKern |     (cudaq.EvolveResult           |
| elDecorator.enable_return_to_log) |     method)](ap                   |
| -   [epsilon                      | i/languages/python_api.html#cudaq |
|     (cudaq.optimizers.Adam        | .EvolveResult.expectation_values) |
|     prope                         | -   [expectation_z()              |
| rty)](api/languages/python_api.ht |     (cudaq.SampleResult           |
| ml#cudaq.optimizers.Adam.epsilon) |     method                        |
| -   [estimate_resources() (in     | )](api/languages/python_api.html# |
|     module                        | cudaq.SampleResult.expectation_z) |
|                                   | -   [expected_dimensions          |
|    cudaq)](api/languages/python_a |     (cuda                         |
| pi.html#cudaq.estimate_resources) | q.operators.MatrixOperatorElement |
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
|                                   | s.spin.SpinOperator.get_raw_data) |
| (cudaq.ptsbe.TraceInstructionType |     -   [(cuda                    |
|     att                           | q.operators.spin.SpinOperatorTerm |
| ribute)](api/ptsbe_api.html#cudaq |         method)](api/languages/p  |
| .ptsbe.TraceInstructionType.Gate) | ython_api.html#cudaq.operators.sp |
| -   [generate_trajectories()      | in.SpinOperatorTerm.get_raw_data) |
|                                   | -   [get_register_counts()        |
|  (cudaq.ptsbe.PTSSamplingStrategy |     (cudaq.SampleResult           |
|     method)](api/pts              |     method)](api                  |
| be_api.html#cudaq.ptsbe.PTSSampli | /languages/python_api.html#cudaq. |
| ngStrategy.generate_trajectories) | SampleResult.get_register_counts) |
| -   [get()                        | -   [get_sequential_data()        |
|     (cudaq.AsyncEvolveResult      |     (cudaq.SampleResult           |
|     m                             |     method)](api                  |
| ethod)](api/languages/python_api. | /languages/python_api.html#cudaq. |
| html#cudaq.AsyncEvolveResult.get) | SampleResult.get_sequential_data) |
|                                   | -   [get_spin()                   |
|    -   [(cudaq.AsyncObserveResult |     (cudaq.ObserveResult          |
|         me                        |     me                            |
| thod)](api/languages/python_api.h | thod)](api/languages/python_api.h |
| tml#cudaq.AsyncObserveResult.get) | tml#cudaq.ObserveResult.get_spin) |
|     -   [(cudaq.AsyncStateResult  | -   [get_state() (in module       |
|                                   |     cudaq)](api/languages         |
| method)](api/languages/python_api | /python_api.html#cudaq.get_state) |
| .html#cudaq.AsyncStateResult.get) | -   [get_state_async() (in module |
| -   [get_binary_symplectic_form() |     cudaq)](api/languages/pytho   |
|     (cuda                         | n_api.html#cudaq.get_state_async) |
| q.operators.spin.SpinOperatorTerm | -   [get_state_refval()           |
|     metho                         |     (cudaq.State                  |
| d)](api/languages/python_api.html |     me                            |
| #cudaq.operators.spin.SpinOperato | thod)](api/languages/python_api.h |
| rTerm.get_binary_symplectic_form) | tml#cudaq.State.get_state_refval) |
| -   [get_channels()               | -   [get_target() (in module      |
|     (cudaq.NoiseModel             |     cudaq)](api/languages/        |
|     met                           | python_api.html#cudaq.get_target) |
| hod)](api/languages/python_api.ht | -   [get_targets() (in module     |
| ml#cudaq.NoiseModel.get_channels) |     cudaq)](api/languages/p       |
| -   [get_coefficient()            | ython_api.html#cudaq.get_targets) |
|     (                             | -   [get_term_count()             |
| cudaq.operators.spin.SpinOperator |     (                             |
|     method)](api/languages/       | cudaq.operators.spin.SpinOperator |
| python_api.html#cudaq.operators.s |     method)](api/languages        |
| pin.SpinOperator.get_coefficient) | /python_api.html#cudaq.operators. |
|     -   [(cuda                    | spin.SpinOperator.get_term_count) |
| q.operators.spin.SpinOperatorTerm | -   [get_total_shots()            |
|                                   |     (cudaq.SampleResult           |
|       method)](api/languages/pyth |     method)]                      |
| on_api.html#cudaq.operators.spin. | (api/languages/python_api.html#cu |
| SpinOperatorTerm.get_coefficient) | daq.SampleResult.get_total_shots) |
| -   [get_marginal_counts()        | -   [get_trajectory()             |
|     (cudaq.SampleResult           |                                   |
|     method)](api                  |   (cudaq.ptsbe.PTSBEExecutionData |
| /languages/python_api.html#cudaq. |     method)]                      |
| SampleResult.get_marginal_counts) | (api/ptsbe_api.html#cudaq.ptsbe.P |
| -   [get_ops()                    | TSBEExecutionData.get_trajectory) |
|     (cudaq.KrausChannel           | -   [getTensor() (cudaq.State     |
|                                   |     method)](api/languages/pytho  |
| method)](api/languages/python_api | n_api.html#cudaq.State.getTensor) |
| .html#cudaq.KrausChannel.get_ops) | -   [getTensors() (cudaq.State    |
| -   [get_pauli_word()             |     method)](api/languages/python |
|     (cuda                         | _api.html#cudaq.State.getTensors) |
| q.operators.spin.SpinOperatorTerm | -   [gradient (class in           |
|     method)](api/languages/pyt    |     cudaq.g                       |
| hon_api.html#cudaq.operators.spin | radients)](api/languages/python_a |
| .SpinOperatorTerm.get_pauli_word) | pi.html#cudaq.gradients.gradient) |
| -   [get_precision()              | -   [GradientDescent (class in    |
|     (cudaq.Target                 |     cudaq.optimizers              |
|                                   | )](api/languages/python_api.html# |
| method)](api/languages/python_api | cudaq.optimizers.GradientDescent) |
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
| lDecorator.handle_call_arguments) |     (cudaq.                       |
| -   [has_execution_data()         | ptsbe.ShotAllocationStrategy.Type |
|                                   |     attribute)](api/ptsbe_        |
|    (cudaq.ptsbe.PTSBESampleResult | api.html#cudaq.ptsbe.ShotAllocati |
|     method)](ap                   | onStrategy.Type.HIGH_WEIGHT_BIAS) |
| i/ptsbe_api.html#cudaq.ptsbe.PTSB |                                   |
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
| n_api.html#cudaq.spin.identities) | -   [instructions                 |
| -   [identity()                   |                                   |
|     (cu                           |   (cudaq.ptsbe.PTSBEExecutionData |
| daq.operators.boson.BosonOperator |     attribute                     |
|     static                        | )](api/ptsbe_api.html#cudaq.ptsbe |
|     method)](api/langu            | .PTSBEExecutionData.instructions) |
| ages/python_api.html#cudaq.operat | -   [intermediate_states()        |
| ors.boson.BosonOperator.identity) |     (cudaq.EvolveResult           |
|     -   [(cudaq.                  |     method)](api                  |
| operators.fermion.FermionOperator | /languages/python_api.html#cudaq. |
|         static                    | EvolveResult.intermediate_states) |
|         method)](api/languages    | -   [IntermediateResultSave       |
| /python_api.html#cudaq.operators. |     (class in                     |
| fermion.FermionOperator.identity) |     c                             |
|     -                             | udaq)](api/languages/python_api.h |
|  [(cudaq.operators.MatrixOperator | tml#cudaq.IntermediateResultSave) |
|         static                    | -   [is_constant()                |
|         method)](api/             |                                   |
| languages/python_api.html#cudaq.o |   (cudaq.operators.ScalarOperator |
| perators.MatrixOperator.identity) |     method)](api/lan              |
|     -   [(                        | guages/python_api.html#cudaq.oper |
| cudaq.operators.spin.SpinOperator | ators.ScalarOperator.is_constant) |
|         static                    | -   [is_emulated() (cudaq.Target  |
|         method)](api/lan          |                                   |
| guages/python_api.html#cudaq.oper |   method)](api/languages/python_a |
| ators.spin.SpinOperator.identity) | pi.html#cudaq.Target.is_emulated) |
|     -   [(in module               | -   [is_identity()                |
|                                   |     (cudaq.                       |
|  cudaq.boson)](api/languages/pyth | operators.boson.BosonOperatorTerm |
| on_api.html#cudaq.boson.identity) |     method)](api/languages/py     |
|     -   [(in module               | thon_api.html#cudaq.operators.bos |
|         cud                       | on.BosonOperatorTerm.is_identity) |
| aq.fermion)](api/languages/python |     -   [(cudaq.oper              |
| _api.html#cudaq.fermion.identity) | ators.fermion.FermionOperatorTerm |
|     -   [(in module               |                                   |
|                                   |     method)](api/languages/python |
|    cudaq.spin)](api/languages/pyt | _api.html#cudaq.operators.fermion |
| hon_api.html#cudaq.spin.identity) | .FermionOperatorTerm.is_identity) |
| -   [initial_parameters           |     -   [(c                       |
|     (cudaq.optimizers.Adam        | udaq.operators.MatrixOperatorTerm |
|     property)](api/l              |         method)](api/languag      |
| anguages/python_api.html#cudaq.op | es/python_api.html#cudaq.operator |
| timizers.Adam.initial_parameters) | s.MatrixOperatorTerm.is_identity) |
|     -   [(cudaq.optimizers.COBYLA |     -   [(                        |
|         property)](api/lan        | cudaq.operators.spin.SpinOperator |
| guages/python_api.html#cudaq.opti |         method)](api/langua       |
| mizers.COBYLA.initial_parameters) | ges/python_api.html#cudaq.operato |
|     -   [                         | rs.spin.SpinOperator.is_identity) |
| (cudaq.optimizers.GradientDescent |     -   [(cuda                    |
|                                   | q.operators.spin.SpinOperatorTerm |
|       property)](api/languages/py |         method)](api/languages/   |
| thon_api.html#cudaq.optimizers.Gr | python_api.html#cudaq.operators.s |
| adientDescent.initial_parameters) | pin.SpinOperatorTerm.is_identity) |
|     -   [(cudaq.optimizers.LBFGS  | -   [is_initialized() (in module  |
|         property)](api/la         |     c                             |
| nguages/python_api.html#cudaq.opt | udaq.mpi)](api/languages/python_a |
| imizers.LBFGS.initial_parameters) | pi.html#cudaq.mpi.is_initialized) |
|                                   | -   [is_on_gpu() (cudaq.State     |
| -   [(cudaq.optimizers.NelderMead |     method)](api/languages/pytho  |
|         property)](api/languag    | n_api.html#cudaq.State.is_on_gpu) |
| es/python_api.html#cudaq.optimize | -   [is_remote() (cudaq.Target    |
| rs.NelderMead.initial_parameters) |     method)](api/languages/python |
|     -   [(cudaq.optimizers.SGD    | _api.html#cudaq.Target.is_remote) |
|         property)](api/           | -   [is_remote_simulator()        |
| languages/python_api.html#cudaq.o |     (cudaq.Target                 |
| ptimizers.SGD.initial_parameters) |     method                        |
|     -   [(cudaq.optimizers.SPSA   | )](api/languages/python_api.html# |
|         property)](api/l          | cudaq.Target.is_remote_simulator) |
| anguages/python_api.html#cudaq.op | -   [items() (cudaq.SampleResult  |
| timizers.SPSA.initial_parameters) |                                   |
|                                   |   method)](api/languages/python_a |
|                                   | pi.html#cudaq.SampleResult.items) |
+-----------------------------------+-----------------------------------+

## K {#K}

+-----------------------------------+-----------------------------------+
| -   [Kernel (in module            | -   [kraus_selections             |
|     cudaq)](api/langua            |     (cudaq.ptsbe.KrausTrajectory  |
| ges/python_api.html#cudaq.Kernel) |     attribute)                    |
| -   [kernel() (in module          | ](api/ptsbe_api.html#cudaq.ptsbe. |
|     cudaq)](api/langua            | KrausTrajectory.kraus_selections) |
| ges/python_api.html#cudaq.kernel) | -   [KrausChannel (class in       |
| -   [kraus_operator_index         |     cudaq)](api/languages/py      |
|     (cudaq.ptsbe.KrausSelection   | thon_api.html#cudaq.KrausChannel) |
|     attribute)](a                 | -   [KrausOperator (class in      |
| pi/ptsbe_api.html#cudaq.ptsbe.Kra |     cudaq)](api/languages/pyt     |
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
|     (cudaq.                       |                                   |
| ptsbe.ShotAllocationStrategy.Type |                                   |
|     attribute)](api/ptsbe         |                                   |
| _api.html#cudaq.ptsbe.ShotAllocat |                                   |
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
| -   [mdiag_sparse_matrix (C++     |     (cudaq.ptsbe.KrausTrajectory  |
|     type)](api/languages/cpp_api. |     attrib                        |
| html#_CPPv419mdiag_sparse_matrix) | ute)](api/ptsbe_api.html#cudaq.pt |
| -   [Measurement                  | sbe.KrausTrajectory.multiplicity) |
|                                   |                                   |
| (cudaq.ptsbe.TraceInstructionType |                                   |
|     attribute)                    |                                   |
| ](api/ptsbe_api.html#cudaq.ptsbe. |                                   |
| TraceInstructionType.Measurement) |                                   |
+-----------------------------------+-----------------------------------+

## N {#N}

+-----------------------------------+-----------------------------------+
| -   [name                         | -   [num_columns()                |
|     (cudaq.ptsbe.TraceInstruction |     (cudaq.ComplexMatrix          |
|                                   |     metho                         |
|  attribute)](api/ptsbe_api.html#c | d)](api/languages/python_api.html |
| udaq.ptsbe.TraceInstruction.name) | #cudaq.ComplexMatrix.num_columns) |
|     -   [(cudaq.PyKernel          | -   [num_qpus() (cudaq.Target     |
|                                   |     method)](api/languages/pytho  |
|     attribute)](api/languages/pyt | n_api.html#cudaq.Target.num_qpus) |
| hon_api.html#cudaq.PyKernel.name) | -   [num_qubits() (cudaq.State    |
|                                   |     method)](api/languages/python |
|   -   [(cudaq.SimulationPrecision | _api.html#cudaq.State.num_qubits) |
|         proper                    | -   [num_ranks() (in module       |
| ty)](api/languages/python_api.htm |     cudaq.mpi)](api/languages/pyt |
| l#cudaq.SimulationPrecision.name) | hon_api.html#cudaq.mpi.num_ranks) |
|     -   [(cudaq.spin.Pauli        | -   [num_rows()                   |
|                                   |     (cudaq.ComplexMatrix          |
|    property)](api/languages/pytho |     me                            |
| n_api.html#cudaq.spin.Pauli.name) | thod)](api/languages/python_api.h |
|     -   [(cudaq.Target            | tml#cudaq.ComplexMatrix.num_rows) |
|                                   | -   [num_shots                    |
|        property)](api/languages/p |     (cudaq.ptsbe.KrausTrajectory  |
| ython_api.html#cudaq.Target.name) |     att                           |
| -   [name()                       | ribute)](api/ptsbe_api.html#cudaq |
|                                   | .ptsbe.KrausTrajectory.num_shots) |
|  (cudaq.ptsbe.PTSSamplingStrategy | -   [number() (in module          |
|                                   |                                   |
|  method)](api/ptsbe_api.html#cuda |    cudaq.boson)](api/languages/py |
| q.ptsbe.PTSSamplingStrategy.name) | thon_api.html#cudaq.boson.number) |
| -   [NelderMead (class in         |     -   [(in module               |
|     cudaq.optim                   |         c                         |
| izers)](api/languages/python_api. | udaq.fermion)](api/languages/pyth |
| html#cudaq.optimizers.NelderMead) | on_api.html#cudaq.fermion.number) |
| -   [Noise                        |     -   [(in module               |
|                                   |         cudaq.operators.cus       |
| (cudaq.ptsbe.TraceInstructionType | tom)](api/languages/python_api.ht |
|     attr                          | ml#cudaq.operators.custom.number) |
| ibute)](api/ptsbe_api.html#cudaq. | -   [nvqir::MPSSimulationState    |
| ptsbe.TraceInstructionType.Noise) |     (C++                          |
| -   [NoiseModel (class in         |     class)]                       |
|     cudaq)](api/languages/        | (api/languages/cpp_api.html#_CPPv |
| python_api.html#cudaq.NoiseModel) | 4I0EN5nvqir18MPSSimulationStateE) |
| -   [num_available_gpus() (in     | -                                 |
|     module                        |  [nvqir::TensorNetSimulationState |
|                                   |     (C++                          |
|    cudaq)](api/languages/python_a |     class)](api/l                 |
| pi.html#cudaq.num_available_gpus) | anguages/cpp_api.html#_CPPv4I0EN5 |
|                                   | nvqir24TensorNetSimulationStateE) |
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
|     (cudaq.ptsbe.KrausSelection   | on_api.html#cudaq.operators.fermi |
|                                   | on.FermionOperatorTerm.ops_count) |
| attribute)](api/ptsbe_api.html#cu |     -   [(c                       |
| daq.ptsbe.KrausSelection.op_name) | udaq.operators.MatrixOperatorTerm |
| -   [OperatorSum (in module       |         property)](api/langu      |
|     cudaq.oper                    | ages/python_api.html#cudaq.operat |
| ators)](api/languages/python_api. | ors.MatrixOperatorTerm.ops_count) |
| html#cudaq.operators.OperatorSum) |     -   [(cuda                    |
|                                   | q.operators.spin.SpinOperatorTerm |
|                                   |         property)](api/language   |
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
| nguages/python_api.html#cudaq.ope | -   [pre_compile()                |
| rators.MatrixOperator.parameters) |     (cudaq.PyKernelDecorator      |
|     -   [(cuda                    |     method)](                     |
| q.operators.MatrixOperatorElement | api/languages/python_api.html#cud |
|         property)](api/languages  | aq.PyKernelDecorator.pre_compile) |
| /python_api.html#cudaq.operators. | -   [probability                  |
| MatrixOperatorElement.parameters) |     (cudaq.ptsbe.KrausTrajectory  |
|     -   [(c                       |     attri                         |
| udaq.operators.MatrixOperatorTerm | bute)](api/ptsbe_api.html#cudaq.p |
|         property)](api/langua     | tsbe.KrausTrajectory.probability) |
| ges/python_api.html#cudaq.operato | -   [probability()                |
| rs.MatrixOperatorTerm.parameters) |     (cudaq.SampleResult           |
|     -                             |     meth                          |
|  [(cudaq.operators.ScalarOperator | od)](api/languages/python_api.htm |
|         property)](api/la         | l#cudaq.SampleResult.probability) |
| nguages/python_api.html#cudaq.ope | -   [ProductOperator (in module   |
| rators.ScalarOperator.parameters) |     cudaq.operator                |
|     -   [(                        | s)](api/languages/python_api.html |
| cudaq.operators.spin.SpinOperator | #cudaq.operators.ProductOperator) |
|         property)](api/langu      | -   [PROPORTIONAL                 |
| ages/python_api.html#cudaq.operat |     (cudaq.                       |
| ors.spin.SpinOperator.parameters) | ptsbe.ShotAllocationStrategy.Type |
|     -   [(cuda                    |     attribute)](api/pt            |
| q.operators.spin.SpinOperatorTerm | sbe_api.html#cudaq.ptsbe.ShotAllo |
|         property)](api/languages  | cationStrategy.Type.PROPORTIONAL) |
| /python_api.html#cudaq.operators. | -   [PyKernel (class in           |
| spin.SpinOperatorTerm.parameters) |     cudaq)](api/language          |
| -   [ParameterShift (class in     | s/python_api.html#cudaq.PyKernel) |
|     cudaq.gradien                 | -   [PyKernelDecorator (class in  |
| ts)](api/languages/python_api.htm |     cudaq)](api/languages/python_ |
| l#cudaq.gradients.ParameterShift) | api.html#cudaq.PyKernelDecorator) |
| -   [params                       |                                   |
|     (cudaq.ptsbe.TraceInstruction |                                   |
|     a                             |                                   |
| ttribute)](api/ptsbe_api.html#cud |                                   |
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
|                                   | -   [qubits                       |
|                                   |     (cudaq.ptsbe.KrausSelection   |
|                                   |                                   |
|                                   |  attribute)](api/ptsbe_api.html#c |
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
| -   [targets                      |    method)](api/languages/python_ |
|     (cudaq.ptsbe.TraceInstruction | api.html#cudaq.operators.fermion. |
|     at                            | FermionOperator.to_sparse_matrix) |
| tribute)](api/ptsbe_api.html#cuda |     -   [(cudaq.oper              |
| q.ptsbe.TraceInstruction.targets) | ators.fermion.FermionOperatorTerm |
| -   [Tensor (class in             |         m                         |
|     cudaq)](api/langua            | ethod)](api/languages/python_api. |
| ges/python_api.html#cudaq.Tensor) | html#cudaq.operators.fermion.Ferm |
| -   [term_count                   | ionOperatorTerm.to_sparse_matrix) |
|     (cu                           |     -   [(                        |
| daq.operators.boson.BosonOperator | cudaq.operators.spin.SpinOperator |
|     property)](api/languag        |         method)](api/languages/p  |
| es/python_api.html#cudaq.operator | ython_api.html#cudaq.operators.sp |
| s.boson.BosonOperator.term_count) | in.SpinOperator.to_sparse_matrix) |
|     -   [(cudaq.                  |     -   [(cuda                    |
| operators.fermion.FermionOperator | q.operators.spin.SpinOperatorTerm |
|                                   |                                   |
|        property)](api/languages/p |      method)](api/languages/pytho |
| ython_api.html#cudaq.operators.fe | n_api.html#cudaq.operators.spin.S |
| rmion.FermionOperator.term_count) | pinOperatorTerm.to_sparse_matrix) |
|     -                             | -   [to_string()                  |
|  [(cudaq.operators.MatrixOperator |     (cudaq.ope                    |
|         property)](api/la         | rators.boson.BosonOperatorElement |
| nguages/python_api.html#cudaq.ope |     method)](api/languages/pyt    |
| rators.MatrixOperator.term_count) | hon_api.html#cudaq.operators.boso |
|     -   [(                        | n.BosonOperatorElement.to_string) |
| cudaq.operators.spin.SpinOperator |     -   [(cudaq.operato           |
|         property)](api/langu      | rs.fermion.FermionOperatorElement |
| ages/python_api.html#cudaq.operat |                                   |
| ors.spin.SpinOperator.term_count) |    method)](api/languages/python_ |
|     -   [(cuda                    | api.html#cudaq.operators.fermion. |
| q.operators.spin.SpinOperatorTerm | FermionOperatorElement.to_string) |
|         property)](api/languages  |     -   [(cuda                    |
| /python_api.html#cudaq.operators. | q.operators.MatrixOperatorElement |
| spin.SpinOperatorTerm.term_count) |         method)](api/language     |
| -   [term_id                      | s/python_api.html#cudaq.operators |
|     (cudaq.                       | .MatrixOperatorElement.to_string) |
| operators.boson.BosonOperatorTerm |     -   [(                        |
|     property)](api/language       | cudaq.operators.spin.SpinOperator |
| s/python_api.html#cudaq.operators |         method)](api/lang         |
| .boson.BosonOperatorTerm.term_id) | uages/python_api.html#cudaq.opera |
|     -   [(cudaq.oper              | tors.spin.SpinOperator.to_string) |
| ators.fermion.FermionOperatorTerm |     -   [(cudaq.o                 |
|                                   | perators.spin.SpinOperatorElement |
|       property)](api/languages/py |         method)](api/languages/p  |
| thon_api.html#cudaq.operators.fer | ython_api.html#cudaq.operators.sp |
| mion.FermionOperatorTerm.term_id) | in.SpinOperatorElement.to_string) |
|     -   [(c                       |     -   [(cuda                    |
| udaq.operators.MatrixOperatorTerm | q.operators.spin.SpinOperatorTerm |
|         property)](api/lan        |         method)](api/language     |
| guages/python_api.html#cudaq.oper | s/python_api.html#cudaq.operators |
| ators.MatrixOperatorTerm.term_id) | .spin.SpinOperatorTerm.to_string) |
|     -   [(cuda                    | -   [trajectories                 |
| q.operators.spin.SpinOperatorTerm |                                   |
|         property)](api/langua     |   (cudaq.ptsbe.PTSBEExecutionData |
| ges/python_api.html#cudaq.operato |     attribute                     |
| rs.spin.SpinOperatorTerm.term_id) | )](api/ptsbe_api.html#cudaq.ptsbe |
| -   [to_dict() (cudaq.Resources   | .PTSBEExecutionData.trajectories) |
|                                   | -   [trajectory_id                |
|    method)](api/languages/python_ |     (cudaq.ptsbe.KrausTrajectory  |
| api.html#cudaq.Resources.to_dict) |     attribu                       |
| -   [to_json()                    | te)](api/ptsbe_api.html#cudaq.pts |
|     (                             | be.KrausTrajectory.trajectory_id) |
| cudaq.gradients.CentralDifference | -   [translate() (in module       |
|     method)](api/la               |     cudaq)](api/languages         |
| nguages/python_api.html#cudaq.gra | /python_api.html#cudaq.translate) |
| dients.CentralDifference.to_json) | -   [trim()                       |
|     -   [(                        |     (cu                           |
| cudaq.gradients.ForwardDifference | daq.operators.boson.BosonOperator |
|         method)](api/la           |     method)](api/l                |
| nguages/python_api.html#cudaq.gra | anguages/python_api.html#cudaq.op |
| dients.ForwardDifference.to_json) | erators.boson.BosonOperator.trim) |
|     -                             |     -   [(cudaq.                  |
|  [(cudaq.gradients.ParameterShift | operators.fermion.FermionOperator |
|         method)](api              |         method)](api/langu        |
| /languages/python_api.html#cudaq. | ages/python_api.html#cudaq.operat |
| gradients.ParameterShift.to_json) | ors.fermion.FermionOperator.trim) |
|     -   [(                        |     -                             |
| cudaq.operators.spin.SpinOperator |  [(cudaq.operators.MatrixOperator |
|         method)](api/la           |         method)](                 |
| nguages/python_api.html#cudaq.ope | api/languages/python_api.html#cud |
| rators.spin.SpinOperator.to_json) | aq.operators.MatrixOperator.trim) |
|     -   [(cuda                    |     -   [(                        |
| q.operators.spin.SpinOperatorTerm | cudaq.operators.spin.SpinOperator |
|         method)](api/langua       |         method)](api              |
| ges/python_api.html#cudaq.operato | /languages/python_api.html#cudaq. |
| rs.spin.SpinOperatorTerm.to_json) | operators.spin.SpinOperator.trim) |
|     -   [(cudaq.optimizers.Adam   | -   [type                         |
|         met                       |     (cudaq.ptsbe.TraceInstruction |
| hod)](api/languages/python_api.ht |                                   |
| ml#cudaq.optimizers.Adam.to_json) |  attribute)](api/ptsbe_api.html#c |
|     -   [(cudaq.optimizers.COBYLA | udaq.ptsbe.TraceInstruction.type) |
|         metho                     | -   [type_to_str()                |
| d)](api/languages/python_api.html |     (cudaq.PyKernelDecorator      |
| #cudaq.optimizers.COBYLA.to_json) |     static                        |
|     -   [                         |     method)](                     |
| (cudaq.optimizers.GradientDescent | api/languages/python_api.html#cud |
|         method)](api/l            | aq.PyKernelDecorator.type_to_str) |
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
| -   [UNIFORM (cudaq.ptsbe.ShotAllocationStrategy.Type                 |
|     attribute)                                                        |
| ](api/ptsbe_api.html#cudaq.ptsbe.ShotAllocationStrategy.Type.UNIFORM) |
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
