::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-4061
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
| -   [compile()                    | st_iteratorneERK14const_iterator) |
|     (cudaq.PyKernelDecorator      | -   [cudaq::produ                 |
|     metho                         | ct_op::const_iterator::operator\* |
| d)](api/languages/python_api.html |     (C++                          |
| #cudaq.PyKernelDecorator.compile) |     function)](api/lang           |
| -   [ComplexMatrix (class in      | uages/cpp_api.html#_CPPv4NK5cudaq |
|     cudaq)](api/languages/pyt     | 10product_op14const_iteratormlEv) |
| hon_api.html#cudaq.ComplexMatrix) | -   [cudaq::produ                 |
| -   [compute()                    | ct_op::const_iterator::operator++ |
|     (                             |     (C++                          |
| cudaq.gradients.CentralDifference |     function)](api/lang           |
|     method)](api/la               | uages/cpp_api.html#_CPPv4N5cudaq1 |
| nguages/python_api.html#cudaq.gra | 0product_op14const_iteratorppEi), |
| dients.CentralDifference.compute) |     [\[1\]](api/lan               |
|     -   [(                        | guages/cpp_api.html#_CPPv4N5cudaq |
| cudaq.gradients.ForwardDifference | 10product_op14const_iteratorppEv) |
|         method)](api/la           | -   [cudaq::produc                |
| nguages/python_api.html#cudaq.gra | t_op::const_iterator::operator\-- |
| dients.ForwardDifference.compute) |     (C++                          |
|     -                             |     function)](api/lang           |
|  [(cudaq.gradients.ParameterShift | uages/cpp_api.html#_CPPv4N5cudaq1 |
|         method)](api              | 0product_op14const_iteratormmEi), |
| /languages/python_api.html#cudaq. |     [\[1\]](api/lan               |
| gradients.ParameterShift.compute) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [const()                      | 10product_op14const_iteratormmEv) |
|                                   | -   [cudaq::produc                |
|   (cudaq.operators.ScalarOperator | t_op::const_iterator::operator-\> |
|     class                         |     (C++                          |
|     method)](a                    |     function)](api/lan            |
| pi/languages/python_api.html#cuda | guages/cpp_api.html#_CPPv4N5cudaq |
| q.operators.ScalarOperator.const) | 10product_op14const_iteratorptEv) |
| -   [copy()                       | -   [cudaq::produ                 |
|     (cu                           | ct_op::const_iterator::operator== |
| daq.operators.boson.BosonOperator |     (C++                          |
|     method)](api/l                |     fun                           |
| anguages/python_api.html#cudaq.op | ction)](api/languages/cpp_api.htm |
| erators.boson.BosonOperator.copy) | l#_CPPv4NK5cudaq10product_op14con |
|     -   [(cudaq.                  | st_iteratoreqERK14const_iterator) |
| operators.boson.BosonOperatorTerm | -   [cudaq::product_op::degrees   |
|         method)](api/langu        |     (C++                          |
| ages/python_api.html#cudaq.operat |     function)                     |
| ors.boson.BosonOperatorTerm.copy) | ](api/languages/cpp_api.html#_CPP |
|     -   [(cudaq.                  | v4NK5cudaq10product_op7degreesEv) |
| operators.fermion.FermionOperator | -   [cudaq::product_op::dump (C++ |
|         method)](api/langu        |     functi                        |
| ages/python_api.html#cudaq.operat | on)](api/languages/cpp_api.html#_ |
| ors.fermion.FermionOperator.copy) | CPPv4NK5cudaq10product_op4dumpEv) |
|     -   [(cudaq.oper              | -   [cudaq::product_op::end (C++  |
| ators.fermion.FermionOperatorTerm |     funct                         |
|         method)](api/languages    | ion)](api/languages/cpp_api.html# |
| /python_api.html#cudaq.operators. | _CPPv4NK5cudaq10product_op3endEv) |
| fermion.FermionOperatorTerm.copy) | -   [c                            |
|     -                             | udaq::product_op::get_coefficient |
|  [(cudaq.operators.MatrixOperator |     (C++                          |
|         method)](                 |     function)](api/lan            |
| api/languages/python_api.html#cud | guages/cpp_api.html#_CPPv4NK5cuda |
| aq.operators.MatrixOperator.copy) | q10product_op15get_coefficientEv) |
|     -   [(c                       | -                                 |
| udaq.operators.MatrixOperatorTerm |   [cudaq::product_op::get_term_id |
|         method)](api/             |     (C++                          |
| languages/python_api.html#cudaq.o |     function)](api                |
| perators.MatrixOperatorTerm.copy) | /languages/cpp_api.html#_CPPv4NK5 |
|     -   [(                        | cudaq10product_op11get_term_idEv) |
| cudaq.operators.spin.SpinOperator | -                                 |
|         method)](api              |   [cudaq::product_op::is_identity |
| /languages/python_api.html#cudaq. |     (C++                          |
| operators.spin.SpinOperator.copy) |     function)](api                |
|     -   [(cuda                    | /languages/cpp_api.html#_CPPv4NK5 |
| q.operators.spin.SpinOperatorTerm | cudaq10product_op11is_identityEv) |
|         method)](api/lan          | -   [cudaq::product_op::num_ops   |
| guages/python_api.html#cudaq.oper |     (C++                          |
| ators.spin.SpinOperatorTerm.copy) |     function)                     |
| -   [count() (cudaq.Resources     | ](api/languages/cpp_api.html#_CPP |
|     method)](api/languages/pytho  | v4NK5cudaq10product_op7num_opsEv) |
| n_api.html#cudaq.Resources.count) | -                                 |
|     -   [(cudaq.SampleResult      |    [cudaq::product_op::operator\* |
|                                   |     (C++                          |
|   method)](api/languages/python_a |     function)](api/languages/     |
| pi.html#cudaq.SampleResult.count) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [count_controls()             | oduct_opmlE10product_opI1TERK15sc |
|     (cudaq.Resources              | alar_operatorRK10product_opI1TE), |
|     meth                          |     [\[1\]](api/languages/        |
| od)](api/languages/python_api.htm | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| l#cudaq.Resources.count_controls) | oduct_opmlE10product_opI1TERK15sc |
| -   [counts()                     | alar_operatorRR10product_opI1TE), |
|     (cudaq.ObserveResult          |     [\[2\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| method)](api/languages/python_api | oduct_opmlE10product_opI1TERR15sc |
| .html#cudaq.ObserveResult.counts) | alar_operatorRK10product_opI1TE), |
| -   [create() (in module          |     [\[3\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|    cudaq.boson)](api/languages/py | oduct_opmlE10product_opI1TERR15sc |
| thon_api.html#cudaq.boson.create) | alar_operatorRR10product_opI1TE), |
|     -   [(in module               |     [\[4\]](api/                  |
|         c                         | languages/cpp_api.html#_CPPv4I0EN |
| udaq.fermion)](api/languages/pyth | 5cudaq10product_opmlE6sum_opI1TER |
| on_api.html#cudaq.fermion.create) | K15scalar_operatorRK6sum_opI1TE), |
| -   [csr_spmatrix (C++            |     [\[5\]](api/                  |
|     type)](api/languages/c        | languages/cpp_api.html#_CPPv4I0EN |
| pp_api.html#_CPPv412csr_spmatrix) | 5cudaq10product_opmlE6sum_opI1TER |
| -   cudaq                         | K15scalar_operatorRR6sum_opI1TE), |
|     -   [module](api/langua       |     [\[6\]](api/                  |
| ges/python_api.html#module-cudaq) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq (C++                   | 5cudaq10product_opmlE6sum_opI1TER |
|     type)](api/lan                | R15scalar_operatorRK6sum_opI1TE), |
| guages/cpp_api.html#_CPPv45cudaq) |     [\[7\]](api/                  |
| -   [cudaq.apply_noise() (in      | languages/cpp_api.html#_CPPv4I0EN |
|     module                        | 5cudaq10product_opmlE6sum_opI1TER |
|     cudaq)](api/languages/python_ | R15scalar_operatorRR6sum_opI1TE), |
| api.html#cudaq.cudaq.apply_noise) |     [\[8\]](api/languages         |
| -   cudaq.boson                   | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     -   [module](api/languages/py | duct_opmlERK6sum_opI9HandlerTyE), |
| thon_api.html#module-cudaq.boson) |     [\[9\]](api/languages/cpp_a   |
| -   cudaq.fermion                 | pi.html#_CPPv4NKR5cudaq10product_ |
|                                   | opmlERK10product_opI9HandlerTyE), |
|   -   [module](api/languages/pyth |     [\[10\]](api/language         |
| on_api.html#module-cudaq.fermion) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   cudaq.operators.custom        | roduct_opmlERK15scalar_operator), |
|     -   [mo                       |     [\[11\]](api/languages/cpp_a  |
| dule](api/languages/python_api.ht | pi.html#_CPPv4NKR5cudaq10product_ |
| ml#module-cudaq.operators.custom) | opmlERR10product_opI9HandlerTyE), |
| -   cudaq.spin                    |     [\[12\]](api/language         |
|     -   [module](api/languages/p  | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ython_api.html#module-cudaq.spin) | roduct_opmlERR15scalar_operator), |
| -   [cudaq::amplitude_damping     |     [\[13\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     cla                           | opmlERK10product_opI9HandlerTyE), |
| ss)](api/languages/cpp_api.html#_ |     [\[14\]](api/languag          |
| CPPv4N5cudaq17amplitude_dampingE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -                                 | roduct_opmlERK15scalar_operator), |
| [cudaq::amplitude_damping_channel |     [\[15\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     class)](api                   | opmlERR10product_opI9HandlerTyE), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[16\]](api/langua           |
| udaq25amplitude_damping_channelE) | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| -   [cudaq::amplitud              | product_opmlERR15scalar_operator) |
| e_damping_channel::num_parameters | -                                 |
|     (C++                          |   [cudaq::product_op::operator\*= |
|     member)](api/languages/cpp_a  |     (C++                          |
| pi.html#_CPPv4N5cudaq25amplitude_ |     function)](api/languages/cpp  |
| damping_channel14num_parametersE) | _api.html#_CPPv4N5cudaq10product_ |
| -   [cudaq::ampli                 | opmLERK10product_opI9HandlerTyE), |
| tude_damping_channel::num_targets |     [\[1\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq10p |
|     member)](api/languages/cp     | roduct_opmLERK15scalar_operator), |
| p_api.html#_CPPv4N5cudaq25amplitu |     [\[2\]](api/languages/cp      |
| de_damping_channel11num_targetsE) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq::AnalogRemoteRESTQPU   | _opmLERR10product_opI9HandlerTyE) |
|     (C++                          | -   [cudaq::product_op::operator+ |
|     class                         |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     function)](api/langu          |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::apply_noise (C++      | q10product_opplE6sum_opI1TERK15sc |
|     function)](api/               | alar_operatorRK10product_opI1TE), |
| languages/cpp_api.html#_CPPv4I0Dp |     [\[1\]](api/                  |
| EN5cudaq11apply_noiseEvDpRR4Args) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::async_result (C++     | 5cudaq10product_opplE6sum_opI1TER |
|     c                             | K15scalar_operatorRK6sum_opI1TE), |
| lass)](api/languages/cpp_api.html |     [\[2\]](api/langu             |
| #_CPPv4I0EN5cudaq12async_resultE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::async_result::get     | q10product_opplE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     functi                        |     [\[3\]](api/                  |
| on)](api/languages/cpp_api.html#_ | languages/cpp_api.html#_CPPv4I0EN |
| CPPv4N5cudaq12async_result3getEv) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq::async_sample_result   | K15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[4\]](api/langu             |
|     type                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
| )](api/languages/cpp_api.html#_CP | q10product_opplE6sum_opI1TERR15sc |
| Pv4N5cudaq19async_sample_resultE) | alar_operatorRK10product_opI1TE), |
| -   [cudaq::BaseRemoteRESTQPU     |     [\[5\]](api/                  |
|     (C++                          | languages/cpp_api.html#_CPPv4I0EN |
|     cla                           | 5cudaq10product_opplE6sum_opI1TER |
| ss)](api/languages/cpp_api.html#_ | R15scalar_operatorRK6sum_opI1TE), |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     [\[6\]](api/langu             |
| -                                 | ages/cpp_api.html#_CPPv4I0EN5cuda |
|    [cudaq::BaseRemoteSimulatorQPU | q10product_opplE6sum_opI1TERR15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     class)](                      |     [\[7\]](api/                  |
| api/languages/cpp_api.html#_CPPv4 | languages/cpp_api.html#_CPPv4I0EN |
| N5cudaq22BaseRemoteSimulatorQPUE) | 5cudaq10product_opplE6sum_opI1TER |
| -   [cudaq::bit_flip_channel (C++ | R15scalar_operatorRR6sum_opI1TE), |
|     cl                            |     [\[8\]](api/languages/cpp_a   |
| ass)](api/languages/cpp_api.html# | pi.html#_CPPv4NKR5cudaq10product_ |
| _CPPv4N5cudaq16bit_flip_channelE) | opplERK10product_opI9HandlerTyE), |
| -   [cudaq:                       |     [\[9\]](api/language          |
| :bit_flip_channel::num_parameters | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opplERK15scalar_operator), |
|     member)](api/langua           |     [\[10\]](api/languages/       |
| ges/cpp_api.html#_CPPv4N5cudaq16b | cpp_api.html#_CPPv4NKR5cudaq10pro |
| it_flip_channel14num_parametersE) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cud                          |     [\[11\]](api/languages/cpp_a  |
| aq::bit_flip_channel::num_targets | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opplERR10product_opI9HandlerTyE), |
|     member)](api/lan              |     [\[12\]](api/language         |
| guages/cpp_api.html#_CPPv4N5cudaq | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| 16bit_flip_channel11num_targetsE) | roduct_opplERR15scalar_operator), |
| -   [cudaq::boson_handler (C++    |     [\[13\]](api/languages/       |
|                                   | cpp_api.html#_CPPv4NKR5cudaq10pro |
|  class)](api/languages/cpp_api.ht | duct_opplERR6sum_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13boson_handlerE) |     [\[                           |
| -   [cudaq::boson_op (C++         | 14\]](api/languages/cpp_api.html# |
|     type)](api/languages/cpp_     | _CPPv4NKR5cudaq10product_opplEv), |
| api.html#_CPPv4N5cudaq8boson_opE) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq::boson_op_term (C++    | api.html#_CPPv4NO5cudaq10product_ |
|                                   | opplERK10product_opI9HandlerTyE), |
|   type)](api/languages/cpp_api.ht |     [\[16\]](api/languag          |
| ml#_CPPv4N5cudaq13boson_op_termE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::CodeGenConfig (C++    | roduct_opplERK15scalar_operator), |
|                                   |     [\[17\]](api/languages        |
| struct)](api/languages/cpp_api.ht | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ml#_CPPv4N5cudaq13CodeGenConfigE) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cudaq::commutation_relations |     [\[18\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     struct)]                      | opplERR10product_opI9HandlerTyE), |
| (api/languages/cpp_api.html#_CPPv |     [\[19\]](api/languag          |
| 4N5cudaq21commutation_relationsE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::complex (C++          | roduct_opplERR15scalar_operator), |
|     type)](api/languages/cpp      |     [\[20\]](api/languages        |
| _api.html#_CPPv4N5cudaq7complexE) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -   [cudaq::complex_matrix (C++   | duct_opplERR6sum_opI9HandlerTyE), |
|                                   |     [                             |
| class)](api/languages/cpp_api.htm | \[21\]](api/languages/cpp_api.htm |
| l#_CPPv4N5cudaq14complex_matrixE) | l#_CPPv4NO5cudaq10product_opplEv) |
| -                                 | -   [cudaq::product_op::operator- |
|   [cudaq::complex_matrix::adjoint |     (C++                          |
|     (C++                          |     function)](api/langu          |
|     function)](a                  | ages/cpp_api.html#_CPPv4I0EN5cuda |
| pi/languages/cpp_api.html#_CPPv4N | q10product_opmiE6sum_opI1TERK15sc |
| 5cudaq14complex_matrix7adjointEv) | alar_operatorRK10product_opI1TE), |
| -   [cudaq::                      |     [\[1\]](api/                  |
| complex_matrix::diagonal_elements | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/languages      | K15scalar_operatorRK6sum_opI1TE), |
| /cpp_api.html#_CPPv4NK5cudaq14com |     [\[2\]](api/langu             |
| plex_matrix17diagonal_elementsEi) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::complex_matrix::dump  | q10product_opmiE6sum_opI1TERK15sc |
|     (C++                          | alar_operatorRR10product_opI1TE), |
|     function)](api/language       |     [\[3\]](api/                  |
| s/cpp_api.html#_CPPv4NK5cudaq14co | languages/cpp_api.html#_CPPv4I0EN |
| mplex_matrix4dumpERNSt7ostreamE), | 5cudaq10product_opmiE6sum_opI1TER |
|     [\[1\]]                       | K15scalar_operatorRR6sum_opI1TE), |
| (api/languages/cpp_api.html#_CPPv |     [\[4\]](api/langu             |
| 4NK5cudaq14complex_matrix4dumpEv) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [c                            | q10product_opmiE6sum_opI1TERR15sc |
| udaq::complex_matrix::eigenvalues | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[5\]](api/                  |
|     function)](api/lan            | languages/cpp_api.html#_CPPv4I0EN |
| guages/cpp_api.html#_CPPv4NK5cuda | 5cudaq10product_opmiE6sum_opI1TER |
| q14complex_matrix11eigenvaluesEv) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cu                           |     [\[6\]](api/langu             |
| daq::complex_matrix::eigenvectors | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERR15sc |
|     function)](api/lang           | alar_operatorRR10product_opI1TE), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[7\]](api/                  |
| 14complex_matrix12eigenvectorsEv) | languages/cpp_api.html#_CPPv4I0EN |
| -   [c                            | 5cudaq10product_opmiE6sum_opI1TER |
| udaq::complex_matrix::exponential | R15scalar_operatorRR6sum_opI1TE), |
|     (C++                          |     [\[8\]](api/languages/cpp_a   |
|     function)](api/la             | pi.html#_CPPv4NKR5cudaq10product_ |
| nguages/cpp_api.html#_CPPv4N5cuda | opmiERK10product_opI9HandlerTyE), |
| q14complex_matrix11exponentialEv) |     [\[9\]](api/language          |
| -                                 | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|  [cudaq::complex_matrix::identity | roduct_opmiERK15scalar_operator), |
|     (C++                          |     [\[10\]](api/languages/       |
|     function)](api/languages      | cpp_api.html#_CPPv4NKR5cudaq10pro |
| /cpp_api.html#_CPPv4N5cudaq14comp | duct_opmiERK6sum_opI9HandlerTyE), |
| lex_matrix8identityEKNSt6size_tE) |     [\[11\]](api/languages/cpp_a  |
| -                                 | pi.html#_CPPv4NKR5cudaq10product_ |
| [cudaq::complex_matrix::kronecker | opmiERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[12\]](api/language         |
|     function)](api/lang           | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| uages/cpp_api.html#_CPPv4I00EN5cu | roduct_opmiERR15scalar_operator), |
| daq14complex_matrix9kroneckerE14c |     [\[13\]](api/languages/       |
| omplex_matrix8Iterable8Iterable), | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     [\[1\]](api/l                 | duct_opmiERR6sum_opI9HandlerTyE), |
| anguages/cpp_api.html#_CPPv4N5cud |     [\[                           |
| aq14complex_matrix9kroneckerERK14 | 14\]](api/languages/cpp_api.html# |
| complex_matrixRK14complex_matrix) | _CPPv4NKR5cudaq10product_opmiEv), |
| -   [cudaq::c                     |     [\[15\]](api/languages/cpp_   |
| omplex_matrix::minimal_eigenvalue | api.html#_CPPv4NO5cudaq10product_ |
|     (C++                          | opmiERK10product_opI9HandlerTyE), |
|     function)](api/languages/     |     [\[16\]](api/languag          |
| cpp_api.html#_CPPv4NK5cudaq14comp | es/cpp_api.html#_CPPv4NO5cudaq10p |
| lex_matrix18minimal_eigenvalueEv) | roduct_opmiERK15scalar_operator), |
| -   [                             |     [\[17\]](api/languages        |
| cudaq::complex_matrix::operator() | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERK6sum_opI9HandlerTyE), |
|     function)](api/languages/cpp  |     [\[18\]](api/languages/cpp_   |
| _api.html#_CPPv4N5cudaq14complex_ | api.html#_CPPv4NO5cudaq10product_ |
| matrixclENSt6size_tENSt6size_tE), | opmiERR10product_opI9HandlerTyE), |
|     [\[1\]](api/languages/cpp     |     [\[19\]](api/languag          |
| _api.html#_CPPv4NK5cudaq14complex | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _matrixclENSt6size_tENSt6size_tE) | roduct_opmiERR15scalar_operator), |
| -   [                             |     [\[20\]](api/languages        |
| cudaq::complex_matrix::operator\* | /cpp_api.html#_CPPv4NO5cudaq10pro |
|     (C++                          | duct_opmiERR6sum_opI9HandlerTyE), |
|     function)](api/langua         |     [                             |
| ges/cpp_api.html#_CPPv4N5cudaq14c | \[21\]](api/languages/cpp_api.htm |
| omplex_matrixmlEN14complex_matrix | l#_CPPv4NO5cudaq10product_opmiEv) |
| 10value_typeERK14complex_matrix), | -   [cudaq::product_op::operator/ |
|     [\[1\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/language       |
| v4N5cudaq14complex_matrixmlERK14c | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| omplex_matrixRK14complex_matrix), | roduct_opdvERK15scalar_operator), |
|                                   |     [\[1\]](api/language          |
|  [\[2\]](api/languages/cpp_api.ht | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ml#_CPPv4N5cudaq14complex_matrixm | roduct_opdvERR15scalar_operator), |
| lERK14complex_matrixRKNSt6vectorI |     [\[2\]](api/languag           |
| N14complex_matrix10value_typeEEE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -                                 | roduct_opdvERK15scalar_operator), |
| [cudaq::complex_matrix::operator+ |     [\[3\]](api/langua            |
|     (C++                          | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     function                      | product_opdvERR15scalar_operator) |
| )](api/languages/cpp_api.html#_CP | -                                 |
| Pv4N5cudaq14complex_matrixplERK14 |    [cudaq::product_op::operator/= |
| complex_matrixRK14complex_matrix) |     (C++                          |
| -                                 |     function)](api/langu          |
| [cudaq::complex_matrix::operator- | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     (C++                          | product_opdVERK15scalar_operator) |
|     function                      | -   [cudaq::product_op::operator= |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq14complex_matrixmiERK14 |     function)](api/la             |
| complex_matrixRK14complex_matrix) | nguages/cpp_api.html#_CPPv4I0_NSt |
| -   [cu                           | 11enable_if_tIXaantNSt7is_sameI1T |
| daq::complex_matrix::operator\[\] | 9HandlerTyE5valueENSt16is_constru |
|     (C++                          | ctibleI9HandlerTy1TE5valueEEbEEEN |
|                                   | 5cudaq10product_opaSER10product_o |
|  function)](api/languages/cpp_api | pI9HandlerTyERK10product_opI1TE), |
| .html#_CPPv4N5cudaq14complex_matr |     [\[1\]](api/languages/cpp     |
| ixixERKNSt6vectorINSt6size_tEEE), | _api.html#_CPPv4N5cudaq10product_ |
|     [\[1\]](api/languages/cpp_api | opaSERK10product_opI9HandlerTyE), |
| .html#_CPPv4NK5cudaq14complex_mat |     [\[2\]](api/languages/cp      |
| rixixERKNSt6vectorINSt6size_tEEE) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq::complex_matrix::power | _opaSERR10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|     function)]                    |    [cudaq::product_op::operator== |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq14complex_matrix5powerEi) |     function)](api/languages/cpp  |
| -                                 | _api.html#_CPPv4NK5cudaq10product |
|  [cudaq::complex_matrix::set_zero | _opeqERK10product_opI9HandlerTyE) |
|     (C++                          | -                                 |
|     function)](ap                 |  [cudaq::product_op::operator\[\] |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq14complex_matrix8set_zeroEv) |     function)](ap                 |
| -                                 | i/languages/cpp_api.html#_CPPv4NK |
| [cudaq::complex_matrix::to_string | 5cudaq10product_opixENSt6size_tE) |
|     (C++                          | -                                 |
|     function)](api/               |    [cudaq::product_op::product_op |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14complex_matrix9to_stringEv) |     function)](api/languages/c    |
| -   [                             | pp_api.html#_CPPv4I0_NSt11enable_ |
| cudaq::complex_matrix::value_type | if_tIXaaNSt7is_sameI9HandlerTy14m |
|     (C++                          | atrix_handlerE5valueEaantNSt7is_s |
|     type)](api/                   | ameI1T9HandlerTyE5valueENSt16is_c |
| languages/cpp_api.html#_CPPv4N5cu | onstructibleI9HandlerTy1TE5valueE |
| daq14complex_matrix10value_typeE) | EbEEEN5cudaq10product_op10product |
| -   [cudaq::contrib (C++          | _opERK10product_opI1TERKN14matrix |
|     type)](api/languages/cpp      | _handler20commutation_behaviorE), |
| _api.html#_CPPv4N5cudaq7contribE) |                                   |
| -   [cudaq::contrib::draw (C++    |  [\[1\]](api/languages/cpp_api.ht |
|     function)                     | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| ](api/languages/cpp_api.html#_CPP | tNSt7is_sameI1T9HandlerTyE5valueE |
| v4I0DpEN5cudaq7contrib4drawENSt6s | NSt16is_constructibleI9HandlerTy1 |
| tringERR13QuantumKernelDpRR4Args) | TE5valueEEbEEEN5cudaq10product_op |
| -                                 | 10product_opERK10product_opI1TE), |
| [cudaq::contrib::get_unitary_cmat |                                   |
|     (C++                          |   [\[2\]](api/languages/cpp_api.h |
|     function)](api/languages/cp   | tml#_CPPv4N5cudaq10product_op10pr |
| p_api.html#_CPPv4I0DpEN5cudaq7con | oduct_opENSt6size_tENSt6size_tE), |
| trib16get_unitary_cmatE14complex_ |     [\[3\]](api/languages/cp      |
| matrixRR13QuantumKernelDpRR4Args) | p_api.html#_CPPv4N5cudaq10product |
| -   [cudaq::CusvState (C++        | _op10product_opENSt7complexIdEE), |
|                                   |     [\[4\]](api/l                 |
|    class)](api/languages/cpp_api. | anguages/cpp_api.html#_CPPv4N5cud |
| html#_CPPv4I0EN5cudaq9CusvStateE) | aq10product_op10product_opERK10pr |
| -   [cudaq::depolarization1 (C++  | oduct_opI9HandlerTyENSt6size_tE), |
|     c                             |     [\[5\]](api/l                 |
| lass)](api/languages/cpp_api.html | anguages/cpp_api.html#_CPPv4N5cud |
| #_CPPv4N5cudaq15depolarization1E) | aq10product_op10product_opERR10pr |
| -   [cudaq::depolarization2 (C++  | oduct_opI9HandlerTyENSt6size_tE), |
|     c                             |     [\[6\]](api/languages         |
| lass)](api/languages/cpp_api.html | /cpp_api.html#_CPPv4N5cudaq10prod |
| #_CPPv4N5cudaq15depolarization2E) | uct_op10product_opERR9HandlerTy), |
| -   [cudaq:                       |     [\[7\]](ap                    |
| :depolarization2::depolarization2 | i/languages/cpp_api.html#_CPPv4N5 |
|     (C++                          | cudaq10product_op10product_opEd), |
|     function)](api/languages/cp   |     [\[8\]](a                     |
| p_api.html#_CPPv4N5cudaq15depolar | pi/languages/cpp_api.html#_CPPv4N |
| ization215depolarization2EK4real) | 5cudaq10product_op10product_opEv) |
| -   [cudaq                        | -   [cuda                         |
| ::depolarization2::num_parameters | q::product_op::to_diagonal_matrix |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api/               |
| ages/cpp_api.html#_CPPv4N5cudaq15 | languages/cpp_api.html#_CPPv4NK5c |
| depolarization214num_parametersE) | udaq10product_op18to_diagonal_mat |
| -   [cu                           | rixENSt13unordered_mapINSt6size_t |
| daq::depolarization2::num_targets | ENSt7int64_tEEERKNSt13unordered_m |
|     (C++                          | apINSt6stringENSt7complexIdEEEEb) |
|     member)](api/la               | -   [cudaq::product_op::to_matrix |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q15depolarization211num_targetsE) |     funct                         |
| -                                 | ion)](api/languages/cpp_api.html# |
|    [cudaq::depolarization_channel | _CPPv4NK5cudaq10product_op9to_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     class)](                      | ENSt7int64_tEEERKNSt13unordered_m |
| api/languages/cpp_api.html#_CPPv4 | apINSt6stringENSt7complexIdEEEEb) |
| N5cudaq22depolarization_channelE) | -   [cu                           |
| -   [cudaq::depol                 | daq::product_op::to_sparse_matrix |
| arization_channel::num_parameters |     (C++                          |
|     (C++                          |     function)](ap                 |
|     member)](api/languages/cp     | i/languages/cpp_api.html#_CPPv4NK |
| p_api.html#_CPPv4N5cudaq22depolar | 5cudaq10product_op16to_sparse_mat |
| ization_channel14num_parametersE) | rixENSt13unordered_mapINSt6size_t |
| -   [cudaq::de                    | ENSt7int64_tEEERKNSt13unordered_m |
| polarization_channel::num_targets | apINSt6stringENSt7complexIdEEEEb) |
|     (C++                          | -   [cudaq::product_op::to_string |
|     member)](api/languages        |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq22depo |     function)](                   |
| larization_channel11num_targetsE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::details (C++          | NK5cudaq10product_op9to_stringEv) |
|     type)](api/languages/cpp      | -                                 |
| _api.html#_CPPv4N5cudaq7detailsE) |  [cudaq::product_op::\~product_op |
| -   [cudaq::details::future (C++  |     (C++                          |
|                                   |     fu                            |
|  class)](api/languages/cpp_api.ht | nction)](api/languages/cpp_api.ht |
| ml#_CPPv4N5cudaq7details6futureE) | ml#_CPPv4N5cudaq10product_opD0Ev) |
| -                                 | -   [cudaq::QPU (C++              |
|   [cudaq::details::future::future |     class)](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
|     functio                       | -   [cudaq::QPU::beginExecution   |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq7details6future6future |     function                      |
| ERNSt6vectorI3JobEERNSt6stringERN | )](api/languages/cpp_api.html#_CP |
| St3mapINSt6stringENSt6stringEEE), | Pv4N5cudaq3QPU14beginExecutionEv) |
|     [\[1\]](api/lang              | -   [cuda                         |
| uages/cpp_api.html#_CPPv4N5cudaq7 | q::QPU::configureExecutionContext |
| details6future6futureERR6future), |     (C++                          |
|     [\[2\]]                       |     funct                         |
| (api/languages/cpp_api.html#_CPPv | ion)](api/languages/cpp_api.html# |
| 4N5cudaq7details6future6futureEv) | _CPPv4NK5cudaq3QPU25configureExec |
| -   [cu                           | utionContextER16ExecutionContext) |
| daq::details::kernel_builder_base | -   [cudaq::QPU::endExecution     |
|     (C++                          |     (C++                          |
|     class)](api/l                 |     functi                        |
| anguages/cpp_api.html#_CPPv4N5cud | on)](api/languages/cpp_api.html#_ |
| aq7details19kernel_builder_baseE) | CPPv4N5cudaq3QPU12endExecutionEv) |
| -   [cudaq::details::             | -   [cudaq::QPU::enqueue (C++     |
| kernel_builder_base::operator\<\< |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4N5 |
|     function)](api/langua         | cudaq3QPU7enqueueER11QuantumTask) |
| ges/cpp_api.html#_CPPv4N5cudaq7de | -   [cud                          |
| tails19kernel_builder_baselsERNSt | aq::QPU::finalizeExecutionContext |
| 7ostreamERK19kernel_builder_base) |     (C++                          |
| -   [                             |     func                          |
| cudaq::details::KernelBuilderType | tion)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     class)](api                   | utionContextER16ExecutionContext) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::QPU::getConnectivity  |
| udaq7details17KernelBuilderTypeE) |     (C++                          |
| -   [cudaq::d                     |     function)                     |
| etails::KernelBuilderType::create | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq3QPU15getConnectivityEv) |
|     function)                     | -                                 |
| ](api/languages/cpp_api.html#_CPP | [cudaq::QPU::getExecutionThreadId |
| v4N5cudaq7details17KernelBuilderT |     (C++                          |
| ype6createEPN4mlir11MLIRContextE) |     function)](api/               |
| -   [cudaq::details::Ker          | languages/cpp_api.html#_CPPv4NK5c |
| nelBuilderType::KernelBuilderType | udaq3QPU20getExecutionThreadIdEv) |
|     (C++                          | -   [cudaq::QPU::getNumQubits     |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4N5cudaq7 |     functi                        |
| details17KernelBuilderType17Kerne | on)](api/languages/cpp_api.html#_ |
| lBuilderTypeERRNSt8functionIFN4ml | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| ir4TypeEPN4mlir11MLIRContextEEEE) | -   [                             |
| -   [cudaq::diag_matrix_callback  | cudaq::QPU::getRemoteCapabilities |
|     (C++                          |     (C++                          |
|     class)                        |     function)](api/l              |
| ](api/languages/cpp_api.html#_CPP | anguages/cpp_api.html#_CPPv4NK5cu |
| v4N5cudaq20diag_matrix_callbackE) | daq3QPU21getRemoteCapabilitiesEv) |
| -   [cudaq::dyn (C++              | -   [cudaq::QPU::isEmulated (C++  |
|     member)](api/languages        |     func                          |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::ExecutionContext (C++ | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
|     cl                            | -   [cudaq::QPU::isSimulator (C++ |
| ass)](api/languages/cpp_api.html# |     funct                         |
| _CPPv4N5cudaq16ExecutionContextE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq                        | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| ::ExecutionContext::amplitudeMaps | -   [cudaq::QPU::launchKernel     |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api/               |
| ages/cpp_api.html#_CPPv4N5cudaq16 | languages/cpp_api.html#_CPPv4N5cu |
| ExecutionContext13amplitudeMapsE) | daq3QPU12launchKernelERKNSt6strin |
| -   [c                            | gE15KernelThunkTypePvNSt8uint64_t |
| udaq::ExecutionContext::asyncExec | ENSt8uint64_tERKNSt6vectorIPvEE), |
|     (C++                          |                                   |
|     member)](api/                 |  [\[1\]](api/languages/cpp_api.ht |
| languages/cpp_api.html#_CPPv4N5cu | ml#_CPPv4N5cudaq3QPU12launchKerne |
| daq16ExecutionContext9asyncExecE) | lERKNSt6stringERKNSt6vectorIPvEE) |
| -   [cud                          | -   [cudaq::QPU::onRandomSeedSet  |
| aq::ExecutionContext::asyncResult |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     member)](api/lan              | uages/cpp_api.html#_CPPv4N5cudaq3 |
| guages/cpp_api.html#_CPPv4N5cudaq | QPU15onRandomSeedSetENSt6size_tE) |
| 16ExecutionContext11asyncResultE) | -   [cudaq::QPU::QPU (C++         |
| -   [cudaq:                       |     functio                       |
| :ExecutionContext::batchIteration | n)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
|     member)](api/langua           |                                   |
| ges/cpp_api.html#_CPPv4N5cudaq16E |  [\[1\]](api/languages/cpp_api.ht |
| xecutionContext14batchIterationE) | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| -   [cudaq::E                     |     [\[2\]](api/languages/cpp_    |
| xecutionContext::canHandleObserve | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
|     (C++                          | -   [cudaq::QPU::setId (C++       |
|     member)](api/language         |     function                      |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | )](api/languages/cpp_api.html#_CP |
| cutionContext16canHandleObserveE) | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| -   [cudaq::E                     | -   [cudaq::QPU::setShots (C++    |
| xecutionContext::ExecutionContext |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     func                          | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| tion)](api/languages/cpp_api.html | -   [cudaq::                      |
| #_CPPv4N5cudaq16ExecutionContext1 | QPU::supportsExplicitMeasurements |
| 6ExecutionContextERKNSt6stringE), |     (C++                          |
|     [\[1\]](api/languages/        |     function)](api/languag        |
| cpp_api.html#_CPPv4N5cudaq16Execu | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| tionContext16ExecutionContextERKN | 28supportsExplicitMeasurementsEv) |
| St6stringENSt6size_tENSt6size_tE) | -   [cudaq::QPU::\~QPU (C++       |
| -   [cudaq::E                     |     function)](api/languages/cp   |
| xecutionContext::expectationValue | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
|     (C++                          | -   [cudaq::QPUState (C++         |
|     member)](api/language         |     class)](api/languages/cpp_    |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | api.html#_CPPv4N5cudaq8QPUStateE) |
| cutionContext16expectationValueE) | -   [cudaq::qreg (C++             |
| -   [cudaq::Execu                 |     class)](api/lan               |
| tionContext::explicitMeasurements | guages/cpp_api.html#_CPPv4I_NSt6s |
|     (C++                          | ize_tE_NSt6size_tEEN5cudaq4qregE) |
|     member)](api/languages/cp     | -   [cudaq::qreg::back (C++       |
| p_api.html#_CPPv4N5cudaq16Executi |     function)                     |
| onContext20explicitMeasurementsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cuda                         | v4N5cudaq4qreg4backENSt6size_tE), |
| q::ExecutionContext::futureResult |     [\[1\]](api/languages/cpp_ap  |
|     (C++                          | i.html#_CPPv4N5cudaq4qreg4backEv) |
|     member)](api/lang             | -   [cudaq::qreg::begin (C++      |
| uages/cpp_api.html#_CPPv4N5cudaq1 |                                   |
| 6ExecutionContext12futureResultE) |  function)](api/languages/cpp_api |
| -   [cudaq::ExecutionContext      | .html#_CPPv4N5cudaq4qreg5beginEv) |
| ::hasConditionalsOnMeasureResults | -   [cudaq::qreg::clear (C++      |
|     (C++                          |                                   |
|     mem                           |  function)](api/languages/cpp_api |
| ber)](api/languages/cpp_api.html# | .html#_CPPv4N5cudaq4qreg5clearEv) |
| _CPPv4N5cudaq16ExecutionContext31 | -   [cudaq::qreg::front (C++      |
| hasConditionalsOnMeasureResultsE) |     function)]                    |
| -   [cudaq::Executi               | (api/languages/cpp_api.html#_CPPv |
| onContext::invocationResultBuffer | 4N5cudaq4qreg5frontENSt6size_tE), |
|     (C++                          |     [\[1\]](api/languages/cpp_api |
|     member)](api/languages/cpp_   | .html#_CPPv4N5cudaq4qreg5frontEv) |
| api.html#_CPPv4N5cudaq16Execution | -   [cudaq::qreg::operator\[\]    |
| Context22invocationResultBufferE) |     (C++                          |
| -   [cu                           |     functi                        |
| daq::ExecutionContext::kernelName | on)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq4qregixEKNSt6size_tE) |
|     member)](api/la               | -   [cudaq::qreg::qreg (C++       |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)                     |
| q16ExecutionContext10kernelNameE) | ](api/languages/cpp_api.html#_CPP |
| -   [cud                          | v4N5cudaq4qreg4qregENSt6size_tE), |
| aq::ExecutionContext::kernelTrace |     [\[1\]](api/languages/cpp_ap  |
|     (C++                          | i.html#_CPPv4N5cudaq4qreg4qregEv) |
|     member)](api/lan              | -   [cudaq::qreg::size (C++       |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 16ExecutionContext11kernelTraceE) |  function)](api/languages/cpp_api |
| -   [cudaq:                       | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| :ExecutionContext::msm_dimensions | -   [cudaq::qreg::slice (C++      |
|     (C++                          |     function)](api/langu          |
|     member)](api/langua           | ages/cpp_api.html#_CPPv4N5cudaq4q |
| ges/cpp_api.html#_CPPv4N5cudaq16E | reg5sliceENSt6size_tENSt6size_tE) |
| xecutionContext14msm_dimensionsE) | -   [cudaq::qreg::value_type (C++ |
| -   [cudaq::                      |                                   |
| ExecutionContext::msm_prob_err_id | type)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq4qreg10value_typeE) |
|     member)](api/languag          | -   [cudaq::qspan (C++            |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     class)](api/lang              |
| ecutionContext15msm_prob_err_idE) | uages/cpp_api.html#_CPPv4I_NSt6si |
| -   [cudaq::Ex                    | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| ecutionContext::msm_probabilities | -   [cudaq::QuakeValue (C++       |
|     (C++                          |     class)](api/languages/cpp_api |
|     member)](api/languages        | .html#_CPPv4N5cudaq10QuakeValueE) |
| /cpp_api.html#_CPPv4N5cudaq16Exec | -   [cudaq::Q                     |
| utionContext17msm_probabilitiesE) | uakeValue::canValidateNumElements |
| -                                 |     (C++                          |
|    [cudaq::ExecutionContext::name |     function)](api/languages      |
|     (C++                          | /cpp_api.html#_CPPv4N5cudaq10Quak |
|     member)]                      | eValue22canValidateNumElementsEv) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq16ExecutionContext4nameE) |  [cudaq::QuakeValue::constantSize |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::noiseModel |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/la               | udaq10QuakeValue12constantSizeEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QuakeValue::dump (C++ |
| q16ExecutionContext10noiseModelE) |     function)](api/lan            |
| -   [cudaq::Exe                   | guages/cpp_api.html#_CPPv4N5cudaq |
| cutionContext::numberTrajectories | 10QuakeValue4dumpERNSt7ostreamE), |
|     (C++                          |     [\                            |
|     member)](api/languages/       | [1\]](api/languages/cpp_api.html# |
| cpp_api.html#_CPPv4N5cudaq16Execu | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| tionContext18numberTrajectoriesE) | -   [cudaq                        |
| -   [c                            | ::QuakeValue::getRequiredElements |
| udaq::ExecutionContext::optResult |     (C++                          |
|     (C++                          |     function)](api/langua         |
|     member)](api/                 | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| languages/cpp_api.html#_CPPv4N5cu | uakeValue19getRequiredElementsEv) |
| daq16ExecutionContext9optResultE) | -   [cudaq::QuakeValue::getValue  |
| -   [cudaq::Execu                 |     (C++                          |
| tionContext::overlapComputeStates |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     member)](api/languages/cp     | 4NK5cudaq10QuakeValue8getValueEv) |
| p_api.html#_CPPv4N5cudaq16Executi | -   [cudaq::QuakeValue::inverse   |
| onContext20overlapComputeStatesE) |     (C++                          |
| -   [cudaq                        |     function)                     |
| ::ExecutionContext::overlapResult | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4NK5cudaq10QuakeValue7inverseEv) |
|     member)](api/langu            | -   [cudaq::QuakeValue::isStdVec  |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| ExecutionContext13overlapResultE) |     function)                     |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|   [cudaq::ExecutionContext::qpuId | v4N5cudaq10QuakeValue8isStdVecEv) |
|     (C++                          | -                                 |
|     member)](                     |    [cudaq::QuakeValue::operator\* |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5qpuIdE) |     function)](api                |
| -   [cudaq                        | /languages/cpp_api.html#_CPPv4N5c |
| ::ExecutionContext::registerNames | udaq10QuakeValuemlE10QuakeValue), |
|     (C++                          |                                   |
|     member)](api/langu            | [\[1\]](api/languages/cpp_api.htm |
| ages/cpp_api.html#_CPPv4N5cudaq16 | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| ExecutionContext13registerNamesE) | -   [cudaq::QuakeValue::operator+ |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::reorderIdx |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/la               | udaq10QuakeValueplE10QuakeValue), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [                             |
| q16ExecutionContext10reorderIdxE) | \[1\]](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq10QuakeValueplEKd), |
|  [cudaq::ExecutionContext::result |                                   |
|     (C++                          | [\[2\]](api/languages/cpp_api.htm |
|     member)](a                    | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cudaq::QuakeValue::operator- |
| 5cudaq16ExecutionContext6resultE) |     (C++                          |
| -                                 |     function)](api                |
|   [cudaq::ExecutionContext::shots | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuemiE10QuakeValue), |
|     member)](                     |     [                             |
| api/languages/cpp_api.html#_CPPv4 | \[1\]](api/languages/cpp_api.html |
| N5cudaq16ExecutionContext5shotsE) | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| -   [cudaq::                      |     [                             |
| ExecutionContext::simulationState | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     member)](api/languag          |                                   |
| es/cpp_api.html#_CPPv4N5cudaq16Ex | [\[3\]](api/languages/cpp_api.htm |
| ecutionContext15simulationStateE) | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| -                                 | -   [cudaq::QuakeValue::operator/ |
|    [cudaq::ExecutionContext::spin |     (C++                          |
|     (C++                          |     function)](api                |
|     member)]                      | /languages/cpp_api.html#_CPPv4N5c |
| (api/languages/cpp_api.html#_CPPv | udaq10QuakeValuedvE10QuakeValue), |
| 4N5cudaq16ExecutionContext4spinE) |                                   |
| -   [cudaq::                      | [\[1\]](api/languages/cpp_api.htm |
| ExecutionContext::totalIterations | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
|     (C++                          | -                                 |
|     member)](api/languag          |  [cudaq::QuakeValue::operator\[\] |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15totalIterationsE) |     function)](api                |
| -   [cudaq::Executio              | /languages/cpp_api.html#_CPPv4N5c |
| nContext::warnedNamedMeasurements | udaq10QuakeValueixEKNSt6size_tE), |
|     (C++                          |     [\[1\]](api/                  |
|     member)](api/languages/cpp_a  | languages/cpp_api.html#_CPPv4N5cu |
| pi.html#_CPPv4N5cudaq16ExecutionC | daq10QuakeValueixERK10QuakeValue) |
| ontext23warnedNamedMeasurementsE) | -                                 |
| -   [cudaq::ExecutionResult (C++  |    [cudaq::QuakeValue::QuakeValue |
|     st                            |     (C++                          |
| ruct)](api/languages/cpp_api.html |     function)](api/languag        |
| #_CPPv4N5cudaq15ExecutionResultE) | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| -   [cud                          | akeValue10QuakeValueERN4mlir20Imp |
| aq::ExecutionResult::appendResult | licitLocOpBuilderEN4mlir5ValueE), |
|     (C++                          |     [\[1\]                        |
|     functio                       | ](api/languages/cpp_api.html#_CPP |
| n)](api/languages/cpp_api.html#_C | v4N5cudaq10QuakeValue10QuakeValue |
| PPv4N5cudaq15ExecutionResult12app | ERN4mlir20ImplicitLocOpBuilderEd) |
| endResultENSt6stringENSt6size_tE) | -   [cudaq::QuakeValue::size (C++ |
| -   [cu                           |     funct                         |
| daq::ExecutionResult::deserialize | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq10QuakeValue4sizeEv) |
|     function)                     | -   [cudaq::QuakeValue::slice     |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq15ExecutionResult11deser |     function)](api/languages/cpp_ |
| ializeERNSt6vectorINSt6size_tEEE) | api.html#_CPPv4N5cudaq10QuakeValu |
| -   [cudaq:                       | e5sliceEKNSt6size_tEKNSt6size_tE) |
| :ExecutionResult::ExecutionResult | -   [cudaq::quantum_platform (C++ |
|     (C++                          |     cl                            |
|     functio                       | ass)](api/languages/cpp_api.html# |
| n)](api/languages/cpp_api.html#_C | _CPPv4N5cudaq16quantum_platformE) |
| PPv4N5cudaq15ExecutionResult15Exe | -   [cudaq:                       |
| cutionResultE16CountsDictionary), | :quantum_platform::beginExecution |
|     [\[1\]](api/lan               |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/languag        |
| 15ExecutionResult15ExecutionResul | es/cpp_api.html#_CPPv4N5cudaq16qu |
| tE16CountsDictionaryNSt6stringE), | antum_platform14beginExecutionEv) |
|     [\[2\                         | -   [cudaq::quantum_pl            |
| ]](api/languages/cpp_api.html#_CP | atform::configureExecutionContext |
| Pv4N5cudaq15ExecutionResult15Exec |     (C++                          |
| utionResultE16CountsDictionaryd), |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|    [\[3\]](api/languages/cpp_api. | 16quantum_platform25configureExec |
| html#_CPPv4N5cudaq15ExecutionResu | utionContextER16ExecutionContext) |
| lt15ExecutionResultENSt6stringE), | -   [cuda                         |
|     [\[4\                         | q::quantum_platform::connectivity |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     function)](api/langu          |
| utionResultERK15ExecutionResult), | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     [\[5\]](api/language          | quantum_platform12connectivityEv) |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | -   [cuda                         |
| cutionResult15ExecutionResultEd), | q::quantum_platform::endExecution |
|     [\[6\]](api/languag           |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq15Ex |     function)](api/langu          |
| ecutionResult15ExecutionResultEv) | ages/cpp_api.html#_CPPv4N5cudaq16 |
| -   [                             | quantum_platform12endExecutionEv) |
| cudaq::ExecutionResult::operator= | -   [cudaq::q                     |
|     (C++                          | uantum_platform::enqueueAsyncTask |
|     function)](api/languages/     |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq15Execu |     function)](api/languages/     |
| tionResultaSERK15ExecutionResult) | cpp_api.html#_CPPv4N5cudaq16quant |
| -   [c                            | um_platform16enqueueAsyncTaskEKNS |
| udaq::ExecutionResult::operator== | t6size_tER19KernelExecutionTask), |
|     (C++                          |     [\[1\]](api/languag           |
|     function)](api/languages/c    | es/cpp_api.html#_CPPv4N5cudaq16qu |
| pp_api.html#_CPPv4NK5cudaq15Execu | antum_platform16enqueueAsyncTaskE |
| tionResulteqERK15ExecutionResult) | KNSt6size_tERNSt8functionIFvvEEE) |
| -   [cud                          | -   [cudaq::quantum_p             |
| aq::ExecutionResult::registerName | latform::finalizeExecutionContext |
|     (C++                          |     (C++                          |
|     member)](api/lan              |     function)](api/languages/c    |
| guages/cpp_api.html#_CPPv4N5cudaq | pp_api.html#_CPPv4NK5cudaq16quant |
| 15ExecutionResult12registerNameE) | um_platform24finalizeExecutionCon |
| -   [cudaq                        | textERN5cudaq16ExecutionContextE) |
| ::ExecutionResult::sequentialData | -   [cudaq::qua                   |
|     (C++                          | ntum_platform::get_codegen_config |
|     member)](api/langu            |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq15 |     function)](api/languages/c    |
| ExecutionResult14sequentialDataE) | pp_api.html#_CPPv4N5cudaq16quantu |
| -   [                             | m_platform18get_codegen_configEv) |
| cudaq::ExecutionResult::serialize | -   [cuda                         |
|     (C++                          | q::quantum_platform::get_exec_ctx |
|     function)](api/l              |     (C++                          |
| anguages/cpp_api.html#_CPPv4NK5cu |     function)](api/langua         |
| daq15ExecutionResult9serializeEv) | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| -   [cudaq::fermion_handler (C++  | quantum_platform12get_exec_ctxEv) |
|     c                             | -   [c                            |
| lass)](api/languages/cpp_api.html | udaq::quantum_platform::get_noise |
| #_CPPv4N5cudaq15fermion_handlerE) |     (C++                          |
| -   [cudaq::fermion_op (C++       |     function)](api/languages/c    |
|     type)](api/languages/cpp_api  | pp_api.html#_CPPv4N5cudaq16quantu |
| .html#_CPPv4N5cudaq10fermion_opE) | m_platform9get_noiseENSt6size_tE) |
| -   [cudaq::fermion_op_term (C++  | -   [cudaq:                       |
|                                   | :quantum_platform::get_num_qubits |
| type)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15fermion_op_termE) |                                   |
| -   [cudaq::FermioniqBaseQPU (C++ | function)](api/languages/cpp_api. |
|     cl                            | html#_CPPv4NK5cudaq16quantum_plat |
| ass)](api/languages/cpp_api.html# | form14get_num_qubitsENSt6size_tE) |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | -   [cudaq::quantum_              |
| -   [cudaq::get_state (C++        | platform::get_remote_capabilities |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)                     |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | ](api/languages/cpp_api.html#_CPP |
| ateEDaRR13QuantumKernelDpRR4Args) | v4NK5cudaq16quantum_platform23get |
| -   [cudaq::gradient (C++         | _remote_capabilitiesENSt6size_tE) |
|     class)](api/languages/cpp_    | -   [cudaq::qua                   |
| api.html#_CPPv4N5cudaq8gradientE) | ntum_platform::get_runtime_target |
| -   [cudaq::gradient::clone (C++  |     (C++                          |
|     fun                           |     function)](api/languages/cp   |
| ction)](api/languages/cpp_api.htm | p_api.html#_CPPv4NK5cudaq16quantu |
| l#_CPPv4N5cudaq8gradient5cloneEv) | m_platform18get_runtime_targetEv) |
| -   [cudaq::gradient::compute     | -   [cuda                         |
|     (C++                          | q::quantum_platform::getLogStream |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq8grad |     function)](api/langu          |
| ient7computeERKNSt6vectorIdEERKNS | ages/cpp_api.html#_CPPv4N5cudaq16 |
| t8functionIFdNSt6vectorIdEEEEEd), | quantum_platform12getLogStreamEv) |
|     [\[1\]](ap                    | -   [cud                          |
| i/languages/cpp_api.html#_CPPv4N5 | aq::quantum_platform::is_emulated |
| cudaq8gradient7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |                                   |
| -   [cudaq::gradient::gradient    |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq16quantum_p |
|     function)](api/lang           | latform11is_emulatedENSt6size_tE) |
| uages/cpp_api.html#_CPPv4I00EN5cu | -   [c                            |
| daq8gradient8gradientER7KernelT), | udaq::quantum_platform::is_remote |
|                                   |     (C++                          |
|    [\[1\]](api/languages/cpp_api. |     function)](api/languages/cp   |
| html#_CPPv4I00EN5cudaq8gradient8g | p_api.html#_CPPv4NK5cudaq16quantu |
| radientER7KernelTRR10ArgsMapper), | m_platform9is_remoteENSt6size_tE) |
|     [\[2\                         | -   [cuda                         |
| ]](api/languages/cpp_api.html#_CP | q::quantum_platform::is_simulator |
| Pv4I00EN5cudaq8gradient8gradientE |     (C++                          |
| RR13QuantumKernelRR10ArgsMapper), |                                   |
|     [\[3                          |   function)](api/languages/cpp_ap |
| \]](api/languages/cpp_api.html#_C | i.html#_CPPv4NK5cudaq16quantum_pl |
| PPv4N5cudaq8gradient8gradientERRN | atform12is_simulatorENSt6size_tE) |
| St8functionIFvNSt6vectorIdEEEEE), | -   [c                            |
|     [\[                           | udaq::quantum_platform::launchVQE |
| 4\]](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4N5cudaq8gradient8gradientEv) |     function)](                   |
| -   [cudaq::gradient::setArgs     | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq16quantum_platform9launchV |
|     fu                            | QEEKNSt6stringEPKvPN5cudaq8gradie |
| nction)](api/languages/cpp_api.ht | ntERKN5cudaq7spin_opERN5cudaq9opt |
| ml#_CPPv4I0DpEN5cudaq8gradient7se | imizerEKiKNSt6size_tENSt6size_tE) |
| tArgsEvR13QuantumKernelDpRR4Args) | -   [cudaq:                       |
| -   [cudaq::gradient::setKernel   | :quantum_platform::list_platforms |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     function)](api/languag        |
| pp_api.html#_CPPv4I0EN5cudaq8grad | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ient9setKernelEvR13QuantumKernel) | antum_platform14list_platformsEv) |
| -   [cud                          | -                                 |
| aq::gradients::central_difference |    [cudaq::quantum_platform::name |
|     (C++                          |     (C++                          |
|     class)](api/la                |     function)](a                  |
| nguages/cpp_api.html#_CPPv4N5cuda | pi/languages/cpp_api.html#_CPPv4N |
| q9gradients18central_differenceE) | K5cudaq16quantum_platform4nameEv) |
| -   [cudaq::gra                   | -   [                             |
| dients::central_difference::clone | cudaq::quantum_platform::num_qpus |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/l              |
| /cpp_api.html#_CPPv4N5cudaq9gradi | anguages/cpp_api.html#_CPPv4NK5cu |
| ents18central_difference5cloneEv) | daq16quantum_platform8num_qpusEv) |
| -   [cudaq::gradi                 | -   [cudaq::                      |
| ents::central_difference::compute | quantum_platform::onRandomSeedSet |
|     (C++                          |     (C++                          |
|     function)](                   |                                   |
| api/languages/cpp_api.html#_CPPv4 | function)](api/languages/cpp_api. |
| N5cudaq9gradients18central_differ | html#_CPPv4N5cudaq16quantum_platf |
| ence7computeERKNSt6vectorIdEERKNS | orm15onRandomSeedSetENSt6size_tE) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq:                       |
|                                   | :quantum_platform::reset_exec_ctx |
|   [\[1\]](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq9gradients18cent |     function)](api/languag        |
| ral_difference7computeERKNSt6vect | es/cpp_api.html#_CPPv4N5cudaq16qu |
| orIdEERNSt6vectorIdEERK7spin_opd) | antum_platform14reset_exec_ctxEv) |
| -   [cudaq::gradie                | -   [cud                          |
| nts::central_difference::gradient | aq::quantum_platform::reset_noise |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api/languages/cpp_ |
| n)](api/languages/cpp_api.html#_C | api.html#_CPPv4N5cudaq16quantum_p |
| PPv4I00EN5cudaq9gradients18centra | latform11reset_noiseENSt6size_tE) |
| l_difference8gradientER7KernelT), | -   [cudaq:                       |
|     [\[1\]](api/langua            | :quantum_platform::resetLogStream |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     (C++                          |
| q9gradients18central_difference8g |     function)](api/languag        |
| radientER7KernelTRR10ArgsMapper), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     [\[2\]](api/languages/cpp_    | antum_platform14resetLogStreamEv) |
| api.html#_CPPv4I00EN5cudaq9gradie | -   [cuda                         |
| nts18central_difference8gradientE | q::quantum_platform::set_exec_ctx |
| RR13QuantumKernelRR10ArgsMapper), |     (C++                          |
|     [\[3\]](api/languages/cpp     |     funct                         |
| _api.html#_CPPv4N5cudaq9gradients | ion)](api/languages/cpp_api.html# |
| 18central_difference8gradientERRN | _CPPv4N5cudaq16quantum_platform12 |
| St8functionIFvNSt6vectorIdEEEEE), | set_exec_ctxEP16ExecutionContext) |
|     [\[4\]](api/languages/cp      | -   [c                            |
| p_api.html#_CPPv4N5cudaq9gradient | udaq::quantum_platform::set_noise |
| s18central_difference8gradientEv) |     (C++                          |
| -   [cud                          |     function                      |
| aq::gradients::forward_difference | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq16quantum_platform9set_ |
|     class)](api/la                | noiseEPK11noise_modelNSt6size_tE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cuda                         |
| q9gradients18forward_differenceE) | q::quantum_platform::setLogStream |
| -   [cudaq::gra                   |     (C++                          |
| dients::forward_difference::clone |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     function)](api/languages      | .html#_CPPv4N5cudaq16quantum_plat |
| /cpp_api.html#_CPPv4N5cudaq9gradi | form12setLogStreamERNSt7ostreamE) |
| ents18forward_difference5cloneEv) | -   [cudaq::quantum_platfor       |
| -   [cudaq::gradi                 | m::supports_explicit_measurements |
| ents::forward_difference::compute |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)](                   | anguages/cpp_api.html#_CPPv4NK5cu |
| api/languages/cpp_api.html#_CPPv4 | daq16quantum_platform30supports_e |
| N5cudaq9gradients18forward_differ | xplicit_measurementsENSt6size_tE) |
| ence7computeERKNSt6vectorIdEERKNS | -   [cudaq::quantum_pla           |
| t8functionIFdNSt6vectorIdEEEEEd), | tform::supports_task_distribution |
|                                   |     (C++                          |
|   [\[1\]](api/languages/cpp_api.h |     fu                            |
| tml#_CPPv4N5cudaq9gradients18forw | nction)](api/languages/cpp_api.ht |
| ard_difference7computeERKNSt6vect | ml#_CPPv4NK5cudaq16quantum_platfo |
| orIdEERNSt6vectorIdEERK7spin_opd) | rm26supports_task_distributionEv) |
| -   [cudaq::gradie                | -   [cudaq::quantum               |
| nts::forward_difference::gradient | _platform::with_execution_context |
|     (C++                          |     (C++                          |
|     functio                       |     function)                     |
| n)](api/languages/cpp_api.html#_C | ](api/languages/cpp_api.html#_CPP |
| PPv4I00EN5cudaq9gradients18forwar | v4I0DpEN5cudaq16quantum_platform2 |
| d_difference8gradientER7KernelT), | 2with_execution_contextEDaR16Exec |
|     [\[1\]](api/langua            | utionContextRR8CallableDpRR4Args) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::QuantumTask (C++      |
| q9gradients18forward_difference8g |     type)](api/languages/cpp_api. |
| radientER7KernelTRR10ArgsMapper), | html#_CPPv4N5cudaq11QuantumTaskE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::qubit (C++            |
| api.html#_CPPv4I00EN5cudaq9gradie |     type)](api/languages/c        |
| nts18forward_difference8gradientE | pp_api.html#_CPPv4N5cudaq5qubitE) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::QubitConnectivity     |
|     [\[3\]](api/languages/cpp     |     (C++                          |
| _api.html#_CPPv4N5cudaq9gradients |     ty                            |
| 18forward_difference8gradientERRN | pe)](api/languages/cpp_api.html#_ |
| St8functionIFvNSt6vectorIdEEEEE), | CPPv4N5cudaq17QubitConnectivityE) |
|     [\[4\]](api/languages/cp      | -   [cudaq::QubitEdge (C++        |
| p_api.html#_CPPv4N5cudaq9gradient |     type)](api/languages/cpp_a    |
| s18forward_difference8gradientEv) | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
| -   [                             | -   [cudaq::qudit (C++            |
| cudaq::gradients::parameter_shift |     clas                          |
|     (C++                          | s)](api/languages/cpp_api.html#_C |
|     class)](api                   | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| /languages/cpp_api.html#_CPPv4N5c | -   [cudaq::qudit::qudit (C++     |
| udaq9gradients15parameter_shiftE) |                                   |
| -   [cudaq::                      | function)](api/languages/cpp_api. |
| gradients::parameter_shift::clone | html#_CPPv4N5cudaq5qudit5quditEv) |
|     (C++                          | -   [cudaq::qvector (C++          |
|     function)](api/langua         |     class)                        |
| ges/cpp_api.html#_CPPv4N5cudaq9gr | ](api/languages/cpp_api.html#_CPP |
| adients15parameter_shift5cloneEv) | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| -   [cudaq::gr                    | -   [cudaq::qvector::back (C++    |
| adients::parameter_shift::compute |     function)](a                  |
|     (C++                          | pi/languages/cpp_api.html#_CPPv4N |
|     function                      | 5cudaq7qvector4backENSt6size_tE), |
| )](api/languages/cpp_api.html#_CP |                                   |
| Pv4N5cudaq9gradients15parameter_s |   [\[1\]](api/languages/cpp_api.h |
| hift7computeERKNSt6vectorIdEERKNS | tml#_CPPv4N5cudaq7qvector4backEv) |
| t8functionIFdNSt6vectorIdEEEEEd), | -   [cudaq::qvector::begin (C++   |
|     [\[1\]](api/languages/cpp_ap  |     fu                            |
| i.html#_CPPv4N5cudaq9gradients15p | nction)](api/languages/cpp_api.ht |
| arameter_shift7computeERKNSt6vect | ml#_CPPv4N5cudaq7qvector5beginEv) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [cudaq::qvector::clear (C++   |
| -   [cudaq::gra                   |     fu                            |
| dients::parameter_shift::gradient | nction)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5clearEv) |
|     func                          | -   [cudaq::qvector::end (C++     |
| tion)](api/languages/cpp_api.html |                                   |
| #_CPPv4I00EN5cudaq9gradients15par | function)](api/languages/cpp_api. |
| ameter_shift8gradientER7KernelT), | html#_CPPv4N5cudaq7qvector3endEv) |
|     [\[1\]](api/lan               | -   [cudaq::qvector::front (C++   |
| guages/cpp_api.html#_CPPv4I00EN5c |     function)](ap                 |
| udaq9gradients15parameter_shift8g | i/languages/cpp_api.html#_CPPv4N5 |
| radientER7KernelTRR10ArgsMapper), | cudaq7qvector5frontENSt6size_tE), |
|     [\[2\]](api/languages/c       |                                   |
| pp_api.html#_CPPv4I00EN5cudaq9gra |  [\[1\]](api/languages/cpp_api.ht |
| dients15parameter_shift8gradientE | ml#_CPPv4N5cudaq7qvector5frontEv) |
| RR13QuantumKernelRR10ArgsMapper), | -   [cudaq::qvector::operator=    |
|     [\[3\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq9gradie |     functio                       |
| nts15parameter_shift8gradientERRN | n)](api/languages/cpp_api.html#_C |
| St8functionIFvNSt6vectorIdEEEEE), | PPv4N5cudaq7qvectoraSERK7qvector) |
|     [\[4\]](api/languages         | -   [cudaq::qvector::operator\[\] |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents15parameter_shift8gradientEv) |     function)                     |
| -   [cudaq::kernel_builder (C++   | ](api/languages/cpp_api.html#_CPP |
|     clas                          | v4N5cudaq7qvectorixEKNSt6size_tE) |
| s)](api/languages/cpp_api.html#_C | -   [cudaq::qvector::qvector (C++ |
| PPv4IDpEN5cudaq14kernel_builderE) |     function)](api/               |
| -   [c                            | languages/cpp_api.html#_CPPv4N5cu |
| udaq::kernel_builder::constantVal | daq7qvector7qvectorENSt6size_tE), |
|     (C++                          |     [\[1\]](a                     |
|     function)](api/la             | pi/languages/cpp_api.html#_CPPv4N |
| nguages/cpp_api.html#_CPPv4N5cuda | 5cudaq7qvector7qvectorERK5state), |
| q14kernel_builder11constantValEd) |     [\[2\]](api                   |
| -   [cu                           | /languages/cpp_api.html#_CPPv4N5c |
| daq::kernel_builder::getArguments | udaq7qvector7qvectorERK7qvector), |
|     (C++                          |     [\[3\]](api/languages/cpp     |
|     function)](api/lan            | _api.html#_CPPv4N5cudaq7qvector7q |
| guages/cpp_api.html#_CPPv4N5cudaq | vectorERKNSt6vectorI7complexEEb), |
| 14kernel_builder12getArgumentsEv) |     [\[4\]](ap                    |
| -   [cu                           | i/languages/cpp_api.html#_CPPv4N5 |
| daq::kernel_builder::getNumParams | cudaq7qvector7qvectorERR7qvector) |
|     (C++                          | -   [cudaq::qvector::size (C++    |
|     function)](api/lan            |     fu                            |
| guages/cpp_api.html#_CPPv4N5cudaq | nction)](api/languages/cpp_api.ht |
| 14kernel_builder12getNumParamsEv) | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| -   [c                            | -   [cudaq::qvector::slice (C++   |
| udaq::kernel_builder::isArgStdVec |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq7qvec |
|     function)](api/languages/cp   | tor5sliceENSt6size_tENSt6size_tE) |
| p_api.html#_CPPv4N5cudaq14kernel_ | -   [cudaq::qvector::value_type   |
| builder11isArgStdVecENSt6size_tE) |     (C++                          |
| -   [cuda                         |     typ                           |
| q::kernel_builder::kernel_builder | e)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq7qvector10value_typeE) |
|     function)](api/languages/cpp_ | -   [cudaq::qview (C++            |
| api.html#_CPPv4N5cudaq14kernel_bu |     clas                          |
| ilder14kernel_builderERNSt6vector | s)](api/languages/cpp_api.html#_C |
| IN7details17KernelBuilderTypeEEE) | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| -   [cudaq::kernel_builder::name  | -   [cudaq::qview::back (C++      |
|     (C++                          |     function)                     |
|     function)                     | ](api/languages/cpp_api.html#_CPP |
| ](api/languages/cpp_api.html#_CPP | v4N5cudaq5qview4backENSt6size_tE) |
| v4N5cudaq14kernel_builder4nameEv) | -   [cudaq::qview::begin (C++     |
| -                                 |                                   |
|    [cudaq::kernel_builder::qalloc | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq5qview5beginEv) |
|     function)](api/language       | -   [cudaq::qview::end (C++       |
| s/cpp_api.html#_CPPv4N5cudaq14ker |                                   |
| nel_builder6qallocE10QuakeValue), |   function)](api/languages/cpp_ap |
|     [\[1\]](api/language          | i.html#_CPPv4N5cudaq5qview3endEv) |
| s/cpp_api.html#_CPPv4N5cudaq14ker | -   [cudaq::qview::front (C++     |
| nel_builder6qallocEKNSt6size_tE), |     function)](                   |
|     [\[2                          | api/languages/cpp_api.html#_CPPv4 |
| \]](api/languages/cpp_api.html#_C | N5cudaq5qview5frontENSt6size_tE), |
| PPv4N5cudaq14kernel_builder6qallo |                                   |
| cERNSt6vectorINSt7complexIdEEEE), |    [\[1\]](api/languages/cpp_api. |
|     [\[3\]](                      | html#_CPPv4N5cudaq5qview5frontEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::qview::operator\[\]   |
| N5cudaq14kernel_builder6qallocEv) |     (C++                          |
| -   [cudaq::kernel_builder::swap  |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     function)](api/language       | PPv4N5cudaq5qviewixEKNSt6size_tE) |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | -   [cudaq::qview::qview (C++     |
| 4kernel_builder4swapEvRK10QuakeVa |     functio                       |
| lueRK10QuakeValueRK10QuakeValue), | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4I0EN5cudaq5qview5qviewERR1R), |
| [\[1\]](api/languages/cpp_api.htm |     [\[1                          |
| l#_CPPv4I00EN5cudaq14kernel_build | \]](api/languages/cpp_api.html#_C |
| er4swapEvRKNSt6vectorI10QuakeValu | PPv4N5cudaq5qview5qviewERK5qview) |
| eEERK10QuakeValueRK10QuakeValue), | -   [cudaq::qview::size (C++      |
|                                   |                                   |
| [\[2\]](api/languages/cpp_api.htm | function)](api/languages/cpp_api. |
| l#_CPPv4N5cudaq14kernel_builder4s | html#_CPPv4NK5cudaq5qview4sizeEv) |
| wapERK10QuakeValueRK10QuakeValue) | -   [cudaq::qview::slice (C++     |
| -   [cudaq::KernelExecutionTask   |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|     type                          | iew5sliceENSt6size_tENSt6size_tE) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qview::value_type     |
| Pv4N5cudaq19KernelExecutionTaskE) |     (C++                          |
| -   [cudaq::KernelThunkResultType |     t                             |
|     (C++                          | ype)](api/languages/cpp_api.html# |
|     struct)]                      | _CPPv4N5cudaq5qview10value_typeE) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::range (C++            |
| 4N5cudaq21KernelThunkResultTypeE) |     fun                           |
| -   [cudaq::KernelThunkType (C++  | ction)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| type)](api/languages/cpp_api.html | orI11ElementTypeEE11ElementType), |
| #_CPPv4N5cudaq15KernelThunkTypeE) |     [\[1\]](api/languages/cpp_    |
| -   [cudaq::kraus_channel (C++    | api.html#_CPPv4I0EN5cudaq5rangeEN |
|                                   | St6vectorI11ElementTypeEE11Elemen |
|  class)](api/languages/cpp_api.ht | tType11ElementType11ElementType), |
| ml#_CPPv4N5cudaq13kraus_channelE) |     [                             |
| -   [cudaq::kraus_channel::empty  | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq5rangeENSt6size_tE) |
|     function)]                    | -   [cudaq::real (C++             |
| (api/languages/cpp_api.html#_CPPv |     type)](api/languages/         |
| 4NK5cudaq13kraus_channel5emptyEv) | cpp_api.html#_CPPv4N5cudaq4realE) |
| -   [cudaq::kraus_c               | -   [cudaq::registry (C++         |
| hannel::generateUnitaryParameters |     type)](api/languages/cpp_     |
|     (C++                          | api.html#_CPPv4N5cudaq8registryE) |
|                                   | -                                 |
|    function)](api/languages/cpp_a |  [cudaq::registry::RegisteredType |
| pi.html#_CPPv4N5cudaq13kraus_chan |     (C++                          |
| nel25generateUnitaryParametersEv) |     class)](api/                  |
| -                                 | languages/cpp_api.html#_CPPv4I0EN |
|    [cudaq::kraus_channel::get_ops | 5cudaq8registry14RegisteredTypeE) |
|     (C++                          | -   [cudaq::RemoteCapabilities    |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     struc                         |
| K5cudaq13kraus_channel7get_opsEv) | t)](api/languages/cpp_api.html#_C |
| -   [cud                          | PPv4N5cudaq18RemoteCapabilitiesE) |
| aq::kraus_channel::identity_flags | -   [cudaq::Remo                  |
|     (C++                          | teCapabilities::isRemoteSimulator |
|     member)](api/lan              |     (C++                          |
| guages/cpp_api.html#_CPPv4N5cudaq |     member)](api/languages/c      |
| 13kraus_channel14identity_flagsE) | pp_api.html#_CPPv4N5cudaq18Remote |
| -   [cud                          | Capabilities17isRemoteSimulatorE) |
| aq::kraus_channel::is_identity_op | -   [cudaq::Remot                 |
|     (C++                          | eCapabilities::RemoteCapabilities |
|                                   |     (C++                          |
|    function)](api/languages/cpp_a |     function)](api/languages/cpp  |
| pi.html#_CPPv4NK5cudaq13kraus_cha | _api.html#_CPPv4N5cudaq18RemoteCa |
| nnel14is_identity_opENSt6size_tE) | pabilities18RemoteCapabilitiesEb) |
| -   [cudaq::                      | -   [cudaq:                       |
| kraus_channel::is_unitary_mixture | :RemoteCapabilities::stateOverlap |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     member)](api/langua           |
| /cpp_api.html#_CPPv4NK5cudaq13kra | ges/cpp_api.html#_CPPv4N5cudaq18R |
| us_channel18is_unitary_mixtureEv) | emoteCapabilities12stateOverlapE) |
| -   [cu                           | -                                 |
| daq::kraus_channel::kraus_channel |   [cudaq::RemoteCapabilities::vqe |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     member)](                     |
| uages/cpp_api.html#_CPPv4IDpEN5cu | api/languages/cpp_api.html#_CPPv4 |
| daq13kraus_channel13kraus_channel | N5cudaq18RemoteCapabilities3vqeE) |
| EDpRRNSt16initializer_listI1TEE), | -   [cudaq::RemoteSimulationState |
|                                   |     (C++                          |
|  [\[1\]](api/languages/cpp_api.ht |     class)]                       |
| ml#_CPPv4N5cudaq13kraus_channel13 | (api/languages/cpp_api.html#_CPPv |
| kraus_channelERK13kraus_channel), | 4N5cudaq21RemoteSimulationStateE) |
|     [\[2\]                        | -   [cudaq::Resources (C++        |
| ](api/languages/cpp_api.html#_CPP |     class)](api/languages/cpp_a   |
| v4N5cudaq13kraus_channel13kraus_c | pi.html#_CPPv4N5cudaq9ResourcesE) |
| hannelERKNSt6vectorI8kraus_opEE), | -   [cudaq::run (C++              |
|     [\[3\]                        |     function)]                    |
| ](api/languages/cpp_api.html#_CPP | (api/languages/cpp_api.html#_CPPv |
| v4N5cudaq13kraus_channel13kraus_c | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| hannelERRNSt6vectorI8kraus_opEE), | 5invoke_result_tINSt7decay_tI13Qu |
|     [\[4\]](api/lan               | antumKernelEEDpNSt7decay_tI4ARGSE |
| guages/cpp_api.html#_CPPv4N5cudaq | EEEEENSt6size_tERN5cudaq11noise_m |
| 13kraus_channel13kraus_channelEv) | odelERR13QuantumKernelDpRR4ARGS), |
| -                                 |     [\[1\]](api/langu             |
| [cudaq::kraus_channel::noise_type | ages/cpp_api.html#_CPPv4I0DpEN5cu |
|     (C++                          | daq3runENSt6vectorINSt15invoke_re |
|     member)](api                  | sult_tINSt7decay_tI13QuantumKerne |
| /languages/cpp_api.html#_CPPv4N5c | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| udaq13kraus_channel10noise_typeE) | ize_tERR13QuantumKernelDpRR4ARGS) |
| -                                 | -   [cudaq::run_async (C++        |
|   [cudaq::kraus_channel::op_names |     functio                       |
|     (C++                          | n)](api/languages/cpp_api.html#_C |
|     member)](                     | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| api/languages/cpp_api.html#_CPPv4 | tureINSt6vectorINSt15invoke_resul |
| N5cudaq13kraus_channel8op_namesE) | t_tINSt7decay_tI13QuantumKernelEE |
| -                                 | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
|  [cudaq::kraus_channel::operator= | ze_tENSt6size_tERN5cudaq11noise_m |
|     (C++                          | odelERR13QuantumKernelDpRR4ARGS), |
|     function)](api/langua         |     [\[1\]](api/la                |
| ges/cpp_api.html#_CPPv4N5cudaq13k | nguages/cpp_api.html#_CPPv4I0DpEN |
| raus_channelaSERK13kraus_channel) | 5cudaq9run_asyncENSt6futureINSt6v |
| -   [c                            | ectorINSt15invoke_result_tINSt7de |
| udaq::kraus_channel::operator\[\] | cay_tI13QuantumKernelEEDpNSt7deca |
|     (C++                          | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|     function)](api/l              | ize_tERR13QuantumKernelDpRR4ARGS) |
| anguages/cpp_api.html#_CPPv4N5cud | -   [cudaq::RuntimeTarget (C++    |
| aq13kraus_channelixEKNSt6size_tE) |                                   |
| -                                 | struct)](api/languages/cpp_api.ht |
| [cudaq::kraus_channel::parameters | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|     (C++                          | -   [cudaq::sample (C++           |
|     member)](api                  |     function)](api/languages/c    |
| /languages/cpp_api.html#_CPPv4N5c | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| udaq13kraus_channel10parametersE) | mpleE13sample_resultRK14sample_op |
| -   [cudaq::krau                  | tionsRR13QuantumKernelDpRR4Args), |
| s_channel::populateDefaultOpNames |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/languages/cp   | Pv4I0DpEN5cudaq6sampleE13sample_r |
| p_api.html#_CPPv4N5cudaq13kraus_c | esultRR13QuantumKernelDpRR4Args), |
| hannel22populateDefaultOpNamesEv) |     [\                            |
| -   [cu                           | [2\]](api/languages/cpp_api.html# |
| daq::kraus_channel::probabilities | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
|     (C++                          | ize_tERR13QuantumKernelDpRR4Args) |
|     member)](api/la               | -   [cudaq::sample_options (C++   |
| nguages/cpp_api.html#_CPPv4N5cuda |     s                             |
| q13kraus_channel13probabilitiesE) | truct)](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4N5cudaq14sample_optionsE) |
|  [cudaq::kraus_channel::push_back | -   [cudaq::sample_result (C++    |
|     (C++                          |                                   |
|     function)](api                |  class)](api/languages/cpp_api.ht |
| /languages/cpp_api.html#_CPPv4N5c | ml#_CPPv4N5cudaq13sample_resultE) |
| udaq13kraus_channel9push_backE8kr | -   [cudaq::sample_result::append |
| aus_opNSt8optionalINSt6stringEEE) |     (C++                          |
| -   [cudaq::kraus_channel::size   |     function)](api/languages/cpp_ |
|     (C++                          | api.html#_CPPv4N5cudaq13sample_re |
|     function)                     | sult6appendERK15ExecutionResultb) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::sample_result::begin  |
| v4NK5cudaq13kraus_channel4sizeEv) |     (C++                          |
| -   [                             |     function)]                    |
| cudaq::kraus_channel::unitary_ops | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4N5cudaq13sample_result5beginEv), |
|     member)](api/                 |     [\[1\]]                       |
| languages/cpp_api.html#_CPPv4N5cu | (api/languages/cpp_api.html#_CPPv |
| daq13kraus_channel11unitary_opsE) | 4NK5cudaq13sample_result5beginEv) |
| -   [cudaq::kraus_op (C++         | -   [cudaq::sample_result::cbegin |
|     struct)](api/languages/cpp_   |     (C++                          |
| api.html#_CPPv4N5cudaq8kraus_opE) |     function)](                   |
| -   [cudaq::kraus_op::adjoint     | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq13sample_result6cbeginEv) |
|     functi                        | -   [cudaq::sample_result::cend   |
| on)](api/languages/cpp_api.html#_ |     (C++                          |
| CPPv4NK5cudaq8kraus_op7adjointEv) |     function)                     |
| -   [cudaq::kraus_op::data (C++   | ](api/languages/cpp_api.html#_CPP |
|                                   | v4NK5cudaq13sample_result4cendEv) |
|  member)](api/languages/cpp_api.h | -   [cudaq::sample_result::clear  |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     (C++                          |
| -   [cudaq::kraus_op::kraus_op    |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     func                          | v4N5cudaq13sample_result5clearEv) |
| tion)](api/languages/cpp_api.html | -   [cudaq::sample_result::count  |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ |     (C++                          |
| opERRNSt16initializer_listI1TEE), |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|  [\[1\]](api/languages/cpp_api.ht | NK5cudaq13sample_result5countENSt |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | 11string_viewEKNSt11string_viewE) |
| pENSt6vectorIN5cudaq7complexEEE), | -   [                             |
|     [\[2\]](api/l                 | cudaq::sample_result::deserialize |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq8kraus_op8kraus_opERK8kraus_op) |     functio                       |
| -   [cudaq::kraus_op::nCols (C++  | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq13sample_result11deser |
| member)](api/languages/cpp_api.ht | ializeERNSt6vectorINSt6size_tEEE) |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | -   [cudaq::sample_result::dump   |
| -   [cudaq::kraus_op::nRows (C++  |     (C++                          |
|                                   |     function)](api/languag        |
| member)](api/languages/cpp_api.ht | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | ample_result4dumpERNSt7ostreamE), |
| -   [cudaq::kraus_op::operator=   |     [\[1\]                        |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)                     | v4NK5cudaq13sample_result4dumpEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::sample_result::end    |
| v4N5cudaq8kraus_opaSERK8kraus_op) |     (C++                          |
| -   [cudaq::kraus_op::precision   |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     memb                          | Pv4N5cudaq13sample_result3endEv), |
| er)](api/languages/cpp_api.html#_ |     [\[1\                         |
| CPPv4N5cudaq8kraus_op9precisionE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::matrix_callback (C++  | Pv4NK5cudaq13sample_result3endEv) |
|     c                             | -   [                             |
| lass)](api/languages/cpp_api.html | cudaq::sample_result::expectation |
| #_CPPv4N5cudaq15matrix_callbackE) |     (C++                          |
| -   [cudaq::matrix_handler (C++   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
| class)](api/languages/cpp_api.htm | tml#_CPPv4NK5cudaq13sample_result |
| l#_CPPv4N5cudaq14matrix_handlerE) | 11expectationEKNSt11string_viewE) |
| -   [cudaq::mat                   | -   [c                            |
| rix_handler::commutation_behavior | udaq::sample_result::get_marginal |
|     (C++                          |     (C++                          |
|     struct)](api/languages/       |     function)](api/languages/cpp_ |
| cpp_api.html#_CPPv4N5cudaq14matri | api.html#_CPPv4NK5cudaq13sample_r |
| x_handler20commutation_behaviorE) | esult12get_marginalERKNSt6vectorI |
| -                                 | NSt6size_tEEEKNSt11string_viewE), |
|    [cudaq::matrix_handler::define |     [\[1\]](api/languages/cpp_    |
|     (C++                          | api.html#_CPPv4NK5cudaq13sample_r |
|     function)](a                  | esult12get_marginalERRKNSt6vector |
| pi/languages/cpp_api.html#_CPPv4N | INSt6size_tEEEKNSt11string_viewE) |
| 5cudaq14matrix_handler6defineENSt | -   [cuda                         |
| 6stringENSt6vectorINSt7int64_tEEE | q::sample_result::get_total_shots |
| RR15matrix_callbackRKNSt13unorder |     (C++                          |
| ed_mapINSt6stringENSt6stringEEE), |     function)](api/langua         |
|                                   | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| [\[1\]](api/languages/cpp_api.htm | sample_result15get_total_shotsEv) |
| l#_CPPv4N5cudaq14matrix_handler6d | -   [cuda                         |
| efineENSt6stringENSt6vectorINSt7i | q::sample_result::has_even_parity |
| nt64_tEEERR15matrix_callbackRR20d |     (C++                          |
| iag_matrix_callbackRKNSt13unorder |     fun                           |
| ed_mapINSt6stringENSt6stringEEE), | ction)](api/languages/cpp_api.htm |
|     [\[2\]](                      | l#_CPPv4N5cudaq13sample_result15h |
| api/languages/cpp_api.html#_CPPv4 | as_even_parityENSt11string_viewE) |
| N5cudaq14matrix_handler6defineENS | -   [cuda                         |
| t6stringENSt6vectorINSt7int64_tEE | q::sample_result::has_expectation |
| ERR15matrix_callbackRRNSt13unorde |     (C++                          |
| red_mapINSt6stringENSt6stringEEE) |     funct                         |
| -                                 | ion)](api/languages/cpp_api.html# |
|   [cudaq::matrix_handler::degrees | _CPPv4NK5cudaq13sample_result15ha |
|     (C++                          | s_expectationEKNSt11string_viewE) |
|     function)](ap                 | -   [cu                           |
| i/languages/cpp_api.html#_CPPv4NK | daq::sample_result::most_probable |
| 5cudaq14matrix_handler7degreesEv) |     (C++                          |
| -                                 |     fun                           |
|  [cudaq::matrix_handler::displace | ction)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NK5cudaq13sample_result13 |
|     function)](api/language       | most_probableEKNSt11string_viewE) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -                                 |
| rix_handler8displaceENSt6size_tE) | [cudaq::sample_result::operator+= |
| -   [cudaq::matrix                |     (C++                          |
| _handler::get_expected_dimensions |     function)](api/langua         |
|     (C++                          | ges/cpp_api.html#_CPPv4N5cudaq13s |
|                                   | ample_resultpLERK13sample_result) |
|    function)](api/languages/cpp_a | -                                 |
| pi.html#_CPPv4NK5cudaq14matrix_ha |  [cudaq::sample_result::operator= |
| ndler23get_expected_dimensionsEv) |     (C++                          |
| -   [cudaq::matrix_ha             |     function)](api/langua         |
| ndler::get_parameter_descriptions | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     (C++                          | ample_resultaSERR13sample_result) |
|                                   | -                                 |
| function)](api/languages/cpp_api. | [cudaq::sample_result::operator== |
| html#_CPPv4NK5cudaq14matrix_handl |     (C++                          |
| er26get_parameter_descriptionsEv) |     function)](api/languag        |
| -   [c                            | es/cpp_api.html#_CPPv4NK5cudaq13s |
| udaq::matrix_handler::instantiate | ample_resulteqERK13sample_result) |
|     (C++                          | -   [                             |
|     function)](a                  | cudaq::sample_result::probability |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14matrix_handler11instantia |     function)](api/lan            |
| teENSt6stringERKNSt6vectorINSt6si | guages/cpp_api.html#_CPPv4NK5cuda |
| ze_tEEERK20commutation_behavior), | q13sample_result11probabilityENSt |
|     [\[1\]](                      | 11string_viewEKNSt11string_viewE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cud                          |
| N5cudaq14matrix_handler11instanti | aq::sample_result::register_names |
| ateENSt6stringERRNSt6vectorINSt6s |     (C++                          |
| ize_tEEERK20commutation_behavior) |     function)](api/langu          |
| -   [cuda                         | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| q::matrix_handler::matrix_handler | 3sample_result14register_namesEv) |
|     (C++                          | -                                 |
|     function)](api/languag        |    [cudaq::sample_result::reorder |
| es/cpp_api.html#_CPPv4I0_NSt11ena |     (C++                          |
| ble_if_tINSt12is_base_of_vI16oper |     function)](api/langua         |
| ator_handler1TEEbEEEN5cudaq14matr | ges/cpp_api.html#_CPPv4N5cudaq13s |
| ix_handler14matrix_handlerERK1T), | ample_result7reorderERKNSt6vector |
|     [\[1\]](ap                    | INSt6size_tEEEKNSt11string_viewE) |
| i/languages/cpp_api.html#_CPPv4I0 | -   [cu                           |
| _NSt11enable_if_tINSt12is_base_of | daq::sample_result::sample_result |
| _vI16operator_handler1TEEbEEEN5cu |     (C++                          |
| daq14matrix_handler14matrix_handl |     func                          |
| erERK1TRK20commutation_behavior), | tion)](api/languages/cpp_api.html |
|     [\[2\]](api/languages/cpp_ap  | #_CPPv4N5cudaq13sample_result13sa |
| i.html#_CPPv4N5cudaq14matrix_hand | mple_resultERK15ExecutionResult), |
| ler14matrix_handlerENSt6size_tE), |     [\[1\]](api/la                |
|     [\[3\]](api/                  | nguages/cpp_api.html#_CPPv4N5cuda |
| languages/cpp_api.html#_CPPv4N5cu | q13sample_result13sample_resultER |
| daq14matrix_handler14matrix_handl | KNSt6vectorI15ExecutionResultEE), |
| erENSt6stringERKNSt6vectorINSt6si |                                   |
| ze_tEEERK20commutation_behavior), |  [\[2\]](api/languages/cpp_api.ht |
|     [\[4\]](api/                  | ml#_CPPv4N5cudaq13sample_result13 |
| languages/cpp_api.html#_CPPv4N5cu | sample_resultERR13sample_result), |
| daq14matrix_handler14matrix_handl |     [                             |
| erENSt6stringERRNSt6vectorINSt6si | \[3\]](api/languages/cpp_api.html |
| ze_tEEERK20commutation_behavior), | #_CPPv4N5cudaq13sample_result13sa |
|     [\                            | mple_resultERR15ExecutionResult), |
| [5\]](api/languages/cpp_api.html# |     [\[4\]](api/lan               |
| _CPPv4N5cudaq14matrix_handler14ma | guages/cpp_api.html#_CPPv4N5cudaq |
| trix_handlerERK14matrix_handler), | 13sample_result13sample_resultEdR |
|     [                             | KNSt6vectorI15ExecutionResultEE), |
| \[6\]](api/languages/cpp_api.html |     [\[5\]](api/lan               |
| #_CPPv4N5cudaq14matrix_handler14m | guages/cpp_api.html#_CPPv4N5cudaq |
| atrix_handlerERR14matrix_handler) | 13sample_result13sample_resultEv) |
| -                                 | -                                 |
|  [cudaq::matrix_handler::momentum |  [cudaq::sample_result::serialize |
|     (C++                          |     (C++                          |
|     function)](api/language       |     function)](api                |
| s/cpp_api.html#_CPPv4N5cudaq14mat | /languages/cpp_api.html#_CPPv4NK5 |
| rix_handler8momentumENSt6size_tE) | cudaq13sample_result9serializeEv) |
| -                                 | -   [cudaq::sample_result::size   |
|    [cudaq::matrix_handler::number |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/langua         | pp_api.html#_CPPv4NK5cudaq13sampl |
| ges/cpp_api.html#_CPPv4N5cudaq14m | e_result4sizeEKNSt11string_viewE) |
| atrix_handler6numberENSt6size_tE) | -   [cudaq::sample_result::to_map |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::operator= |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq13sample_ |
|     fun                           | result6to_mapEKNSt11string_viewE) |
| ction)](api/languages/cpp_api.htm | -   [cuda                         |
| l#_CPPv4I0_NSt11enable_if_tIXaant | q::sample_result::\~sample_result |
| NSt7is_sameI1T14matrix_handlerE5v |     (C++                          |
| alueENSt12is_base_of_vI16operator |     funct                         |
| _handler1TEEEbEEEN5cudaq14matrix_ | ion)](api/languages/cpp_api.html# |
| handleraSER14matrix_handlerRK1T), | _CPPv4N5cudaq13sample_resultD0Ev) |
|     [\[1\]](api/languages         | -   [cudaq::scalar_callback (C++  |
| /cpp_api.html#_CPPv4N5cudaq14matr |     c                             |
| ix_handleraSERK14matrix_handler), | lass)](api/languages/cpp_api.html |
|     [\[2\]](api/language          | #_CPPv4N5cudaq15scalar_callbackE) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [c                            |
| rix_handleraSERR14matrix_handler) | udaq::scalar_callback::operator() |
| -   [                             |     (C++                          |
| cudaq::matrix_handler::operator== |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4NK5cudaq15sc |
|     function)](api/languages      | alar_callbackclERKNSt13unordered_ |
| /cpp_api.html#_CPPv4NK5cudaq14mat | mapINSt6stringENSt7complexIdEEEE) |
| rix_handlereqERK14matrix_handler) | -   [                             |
| -                                 | cudaq::scalar_callback::operator= |
|    [cudaq::matrix_handler::parity |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     function)](api/langua         | pp_api.html#_CPPv4N5cudaq15scalar |
| ges/cpp_api.html#_CPPv4N5cudaq14m | _callbackaSERK15scalar_callback), |
| atrix_handler6parityENSt6size_tE) |     [\[1\]](api/languages/        |
| -                                 | cpp_api.html#_CPPv4N5cudaq15scala |
|  [cudaq::matrix_handler::position | r_callbackaSERR15scalar_callback) |
|     (C++                          | -   [cudaq:                       |
|     function)](api/language       | :scalar_callback::scalar_callback |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     (C++                          |
| rix_handler8positionENSt6size_tE) |     function)](api/languag        |
| -   [cudaq::                      | es/cpp_api.html#_CPPv4I0_NSt11ena |
| matrix_handler::remove_definition | ble_if_tINSt16is_invocable_r_vINS |
|     (C++                          | t7complexIdEE8CallableRKNSt13unor |
|     fu                            | dered_mapINSt6stringENSt7complexI |
| nction)](api/languages/cpp_api.ht | dEEEEEEbEEEN5cudaq15scalar_callba |
| ml#_CPPv4N5cudaq14matrix_handler1 | ck15scalar_callbackERR8Callable), |
| 7remove_definitionERKNSt6stringE) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|   [cudaq::matrix_handler::squeeze | Pv4N5cudaq15scalar_callback15scal |
|     (C++                          | ar_callbackERK15scalar_callback), |
|     function)](api/languag        |     [\[2                          |
| es/cpp_api.html#_CPPv4N5cudaq14ma | \]](api/languages/cpp_api.html#_C |
| trix_handler7squeezeENSt6size_tE) | PPv4N5cudaq15scalar_callback15sca |
| -   [cudaq::m                     | lar_callbackERR15scalar_callback) |
| atrix_handler::to_diagonal_matrix | -   [cudaq::scalar_operator (C++  |
|     (C++                          |     c                             |
|     function)](api/lang           | lass)](api/languages/cpp_api.html |
| uages/cpp_api.html#_CPPv4NK5cudaq | #_CPPv4N5cudaq15scalar_operatorE) |
| 14matrix_handler18to_diagonal_mat | -                                 |
| rixERNSt13unordered_mapINSt6size_ | [cudaq::scalar_operator::evaluate |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |                                   |
| -                                 |    function)](api/languages/cpp_a |
| [cudaq::matrix_handler::to_matrix | pi.html#_CPPv4NK5cudaq15scalar_op |
|     (C++                          | erator8evaluateERKNSt13unordered_ |
|     function)                     | mapINSt6stringENSt7complexIdEEEE) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::scalar_ope            |
| v4NK5cudaq14matrix_handler9to_mat | rator::get_parameter_descriptions |
| rixERNSt13unordered_mapINSt6size_ |     (C++                          |
| tENSt7int64_tEEERKNSt13unordered_ |     f                             |
| mapINSt6stringENSt7complexIdEEEE) | unction)](api/languages/cpp_api.h |
| -                                 | tml#_CPPv4NK5cudaq15scalar_operat |
| [cudaq::matrix_handler::to_string | or26get_parameter_descriptionsEv) |
|     (C++                          | -   [cu                           |
|     function)](api/               | daq::scalar_operator::is_constant |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14matrix_handler9to_stringEb) |     function)](api/lang           |
| -                                 | uages/cpp_api.html#_CPPv4NK5cudaq |
| [cudaq::matrix_handler::unique_id | 15scalar_operator11is_constantEv) |
|     (C++                          | -   [c                            |
|     function)](api/               | udaq::scalar_operator::operator\* |
| languages/cpp_api.html#_CPPv4NK5c |     (C++                          |
| udaq14matrix_handler9unique_idEv) |     function                      |
| -   [cudaq:                       | )](api/languages/cpp_api.html#_CP |
| :matrix_handler::\~matrix_handler | Pv4N5cudaq15scalar_operatormlENSt |
|     (C++                          | 7complexIdEERK15scalar_operator), |
|     functi                        |     [\[1\                         |
| on)](api/languages/cpp_api.html#_ | ]](api/languages/cpp_api.html#_CP |
| CPPv4N5cudaq14matrix_handlerD0Ev) | Pv4N5cudaq15scalar_operatormlENSt |
| -   [cudaq::matrix_op (C++        | 7complexIdEERR15scalar_operator), |
|     type)](api/languages/cpp_a    |     [\[2\]](api/languages/cp      |
| pi.html#_CPPv4N5cudaq9matrix_opE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::matrix_op_term (C++   | operatormlEdRK15scalar_operator), |
|                                   |     [\[3\]](api/languages/cp      |
|  type)](api/languages/cpp_api.htm | p_api.html#_CPPv4N5cudaq15scalar_ |
| l#_CPPv4N5cudaq14matrix_op_termE) | operatormlEdRR15scalar_operator), |
| -                                 |     [\[4\]](api/languages         |
|    [cudaq::mdiag_operator_handler | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormlENSt7complexIdEE), |
|     class)](                      |     [\[5\]](api/languages/cpp     |
| api/languages/cpp_api.html#_CPPv4 | _api.html#_CPPv4NKR5cudaq15scalar |
| N5cudaq22mdiag_operator_handlerE) | _operatormlERK15scalar_operator), |
| -   [cudaq::mpi (C++              |     [\[6\]]                       |
|     type)](api/languages          | (api/languages/cpp_api.html#_CPPv |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | 4NKR5cudaq15scalar_operatormlEd), |
| -   [cudaq::mpi::all_gather (C++  |     [\[7\]](api/language          |
|     fu                            | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| nction)](api/languages/cpp_api.ht | alar_operatormlENSt7complexIdEE), |
| ml#_CPPv4N5cudaq3mpi10all_gatherE |     [\[8\]](api/languages/cp      |
| RNSt6vectorIdEERKNSt6vectorIdEE), | p_api.html#_CPPv4NO5cudaq15scalar |
|                                   | _operatormlERK15scalar_operator), |
|   [\[1\]](api/languages/cpp_api.h |     [\[9\                         |
| tml#_CPPv4N5cudaq3mpi10all_gather | ]](api/languages/cpp_api.html#_CP |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | Pv4NO5cudaq15scalar_operatormlEd) |
| -   [cudaq::mpi::all_reduce (C++  | -   [cu                           |
|                                   | daq::scalar_operator::operator\*= |
|  function)](api/languages/cpp_api |     (C++                          |
| .html#_CPPv4I00EN5cudaq3mpi10all_ |     function)](api/languag        |
| reduceE1TRK1TRK14BinaryFunction), | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     [\[1\]](api/langu             | alar_operatormLENSt7complexIdEE), |
| ages/cpp_api.html#_CPPv4I00EN5cud |     [\[1\]](api/languages/c       |
| aq3mpi10all_reduceE1TRK1TRK4Func) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::mpi::broadcast (C++   | _operatormLERK15scalar_operator), |
|     function)](api/               |     [\[2                          |
| languages/cpp_api.html#_CPPv4N5cu | \]](api/languages/cpp_api.html#_C |
| daq3mpi9broadcastERNSt6stringEi), | PPv4N5cudaq15scalar_operatormLEd) |
|     [\[1\]](api/la                | -   [                             |
| nguages/cpp_api.html#_CPPv4N5cuda | cudaq::scalar_operator::operator+ |
| q3mpi9broadcastERNSt6vectorIdEEi) |     (C++                          |
| -   [cudaq::mpi::finalize (C++    |     function                      |
|     f                             | )](api/languages/cpp_api.html#_CP |
| unction)](api/languages/cpp_api.h | Pv4N5cudaq15scalar_operatorplENSt |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::mpi::initialize (C++  |     [\[1\                         |
|     function                      | ]](api/languages/cpp_api.html#_CP |
| )](api/languages/cpp_api.html#_CP | Pv4N5cudaq15scalar_operatorplENSt |
| Pv4N5cudaq3mpi10initializeEiPPc), | 7complexIdEERR15scalar_operator), |
|     [                             |     [\[2\]](api/languages/cp      |
| \[1\]](api/languages/cpp_api.html | p_api.html#_CPPv4N5cudaq15scalar_ |
| #_CPPv4N5cudaq3mpi10initializeEv) | operatorplEdRK15scalar_operator), |
| -   [cudaq::mpi::is_initialized   |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|     function                      | operatorplEdRR15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[4\]](api/languages         |
| Pv4N5cudaq3mpi14is_initializedEv) | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| -   [cudaq::mpi::num_ranks (C++   | alar_operatorplENSt7complexIdEE), |
|     fu                            |     [\[5\]](api/languages/cpp     |
| nction)](api/languages/cpp_api.ht | _api.html#_CPPv4NKR5cudaq15scalar |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) | _operatorplERK15scalar_operator), |
| -   [cudaq::mpi::rank (C++        |     [\[6\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|    function)](api/languages/cpp_a | 4NKR5cudaq15scalar_operatorplEd), |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     [\[7\]]                       |
| -   [cudaq::noise_model (C++      | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NKR5cudaq15scalar_operatorplEv), |
|    class)](api/languages/cpp_api. |     [\[8\]](api/language          |
| html#_CPPv4N5cudaq11noise_modelE) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::n                     | alar_operatorplENSt7complexIdEE), |
| oise_model::add_all_qubit_channel |     [\[9\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     function)](api                | _operatorplERK15scalar_operator), |
| /languages/cpp_api.html#_CPPv4IDp |     [\[10\]                       |
| EN5cudaq11noise_model21add_all_qu | ](api/languages/cpp_api.html#_CPP |
| bit_channelEvRK13kraus_channeli), | v4NO5cudaq15scalar_operatorplEd), |
|     [\[1\]](api/langua            |     [\[11\                        |
| ges/cpp_api.html#_CPPv4N5cudaq11n | ]](api/languages/cpp_api.html#_CP |
| oise_model21add_all_qubit_channel | Pv4NO5cudaq15scalar_operatorplEv) |
| ERKNSt6stringERK13kraus_channeli) | -   [c                            |
| -                                 | udaq::scalar_operator::operator+= |
|  [cudaq::noise_model::add_channel |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     funct                         | es/cpp_api.html#_CPPv4N5cudaq15sc |
| ion)](api/languages/cpp_api.html# | alar_operatorpLENSt7complexIdEE), |
| _CPPv4IDpEN5cudaq11noise_model11a |     [\[1\]](api/languages/c       |
| dd_channelEvRK15PredicateFuncTy), | pp_api.html#_CPPv4N5cudaq15scalar |
|     [\[1\]](api/languages/cpp_    | _operatorpLERK15scalar_operator), |
| api.html#_CPPv4IDpEN5cudaq11noise |     [\[2                          |
| _model11add_channelEvRKNSt6vector | \]](api/languages/cpp_api.html#_C |
| INSt6size_tEEERK13kraus_channel), | PPv4N5cudaq15scalar_operatorpLEd) |
|     [\[2\]](ap                    | -   [                             |
| i/languages/cpp_api.html#_CPPv4N5 | cudaq::scalar_operator::operator- |
| cudaq11noise_model11add_channelER |     (C++                          |
| KNSt6stringERK15PredicateFuncTy), |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
| [\[3\]](api/languages/cpp_api.htm | Pv4N5cudaq15scalar_operatormiENSt |
| l#_CPPv4N5cudaq11noise_model11add | 7complexIdEERK15scalar_operator), |
| _channelERKNSt6stringERKNSt6vecto |     [\[1\                         |
| rINSt6size_tEEERK13kraus_channel) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise_model::empty    | Pv4N5cudaq15scalar_operatormiENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     function                      |     [\[2\]](api/languages/cp      |
| )](api/languages/cpp_api.html#_CP | p_api.html#_CPPv4N5cudaq15scalar_ |
| Pv4NK5cudaq11noise_model5emptyEv) | operatormiEdRK15scalar_operator), |
| -                                 |     [\[3\]](api/languages/cp      |
| [cudaq::noise_model::get_channels | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRR15scalar_operator), |
|     function)](api/l              |     [\[4\]](api/languages         |
| anguages/cpp_api.html#_CPPv4I0ENK | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| 5cudaq11noise_model12get_channels | alar_operatormiENSt7complexIdEE), |
| ENSt6vectorI13kraus_channelEERKNS |     [\[5\]](api/languages/cpp     |
| t6vectorINSt6size_tEEERKNSt6vecto | _api.html#_CPPv4NKR5cudaq15scalar |
| rINSt6size_tEEERKNSt6vectorIdEE), | _operatormiERK15scalar_operator), |
|     [\[1\]](api/languages/cpp_a   |     [\[6\]]                       |
| pi.html#_CPPv4NK5cudaq11noise_mod | (api/languages/cpp_api.html#_CPPv |
| el12get_channelsERKNSt6stringERKN | 4NKR5cudaq15scalar_operatormiEd), |
| St6vectorINSt6size_tEEERKNSt6vect |     [\[7\]]                       |
| orINSt6size_tEEERKNSt6vectorIdEE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatormiEv), |
|  [cudaq::noise_model::noise_model |     [\[8\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     function)](api                | alar_operatormiENSt7complexIdEE), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[9\]](api/languages/cp      |
| udaq11noise_model11noise_modelEv) | p_api.html#_CPPv4NO5cudaq15scalar |
| -   [cu                           | _operatormiERK15scalar_operator), |
| daq::noise_model::PredicateFuncTy |     [\[10\]                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     type)](api/la                 | v4NO5cudaq15scalar_operatormiEd), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[11\                        |
| q11noise_model15PredicateFuncTyE) | ]](api/languages/cpp_api.html#_CP |
| -   [cud                          | Pv4NO5cudaq15scalar_operatormiEv) |
| aq::noise_model::register_channel | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator-= |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4I00EN5cudaq11 |     function)](api/languag        |
| noise_model16register_channelEvv) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::                      | alar_operatormIENSt7complexIdEE), |
| noise_model::requires_constructor |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     type)](api/languages/cp       | _operatormIERK15scalar_operator), |
| p_api.html#_CPPv4I0DpEN5cudaq11no |     [\[2                          |
| ise_model20requires_constructorE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::noise_model_type (C++ | PPv4N5cudaq15scalar_operatormIEd) |
|     e                             | -   [                             |
| num)](api/languages/cpp_api.html# | cudaq::scalar_operator::operator/ |
| _CPPv4N5cudaq16noise_model_typeE) |     (C++                          |
| -   [cudaq::no                    |     function                      |
| ise_model_type::amplitude_damping | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatordvENSt |
|     enumerator)](api/languages    | 7complexIdEERK15scalar_operator), |
| /cpp_api.html#_CPPv4N5cudaq16nois |     [\[1\                         |
| e_model_type17amplitude_dampingE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise_mode            | Pv4N5cudaq15scalar_operatordvENSt |
| l_type::amplitude_damping_channel | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     e                             | p_api.html#_CPPv4N5cudaq15scalar_ |
| numerator)](api/languages/cpp_api | operatordvEdRK15scalar_operator), |
| .html#_CPPv4N5cudaq16noise_model_ |     [\[3\]](api/languages/cp      |
| type25amplitude_damping_channelE) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [cudaq::n                     | operatordvEdRR15scalar_operator), |
| oise_model_type::bit_flip_channel |     [\[4\]](api/languages         |
|     (C++                          | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     enumerator)](api/language     | alar_operatordvENSt7complexIdEE), |
| s/cpp_api.html#_CPPv4N5cudaq16noi |     [\[5\]](api/languages/cpp     |
| se_model_type16bit_flip_channelE) | _api.html#_CPPv4NKR5cudaq15scalar |
| -   [cudaq::                      | _operatordvERK15scalar_operator), |
| noise_model_type::depolarization1 |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     enumerator)](api/languag      | 4NKR5cudaq15scalar_operatordvEd), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[7\]](api/language          |
| ise_model_type15depolarization1E) | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| -   [cudaq::                      | alar_operatordvENSt7complexIdEE), |
| noise_model_type::depolarization2 |     [\[8\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4NO5cudaq15scalar |
|     enumerator)](api/languag      | _operatordvERK15scalar_operator), |
| es/cpp_api.html#_CPPv4N5cudaq16no |     [\[9\                         |
| ise_model_type15depolarization2E) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::noise_m               | Pv4NO5cudaq15scalar_operatordvEd) |
| odel_type::depolarization_channel | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator/= |
|                                   |     (C++                          |
|   enumerator)](api/languages/cpp_ |     function)](api/languag        |
| api.html#_CPPv4N5cudaq16noise_mod | es/cpp_api.html#_CPPv4N5cudaq15sc |
| el_type22depolarization_channelE) | alar_operatordVENSt7complexIdEE), |
| -                                 |     [\[1\]](api/languages/c       |
|  [cudaq::noise_model_type::pauli1 | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatordVERK15scalar_operator), |
|     enumerator)](a                |     [\[2                          |
| pi/languages/cpp_api.html#_CPPv4N | \]](api/languages/cpp_api.html#_C |
| 5cudaq16noise_model_type6pauli1E) | PPv4N5cudaq15scalar_operatordVEd) |
| -                                 | -   [                             |
|  [cudaq::noise_model_type::pauli2 | cudaq::scalar_operator::operator= |
|     (C++                          |     (C++                          |
|     enumerator)](a                |     function)](api/languages/c    |
| pi/languages/cpp_api.html#_CPPv4N | pp_api.html#_CPPv4N5cudaq15scalar |
| 5cudaq16noise_model_type6pauli2E) | _operatoraSERK15scalar_operator), |
| -   [cudaq                        |     [\[1\]](api/languages/        |
| ::noise_model_type::phase_damping | cpp_api.html#_CPPv4N5cudaq15scala |
|     (C++                          | r_operatoraSERR15scalar_operator) |
|     enumerator)](api/langu        | -   [c                            |
| ages/cpp_api.html#_CPPv4N5cudaq16 | udaq::scalar_operator::operator== |
| noise_model_type13phase_dampingE) |     (C++                          |
| -   [cudaq::noi                   |     function)](api/languages/c    |
| se_model_type::phase_flip_channel | pp_api.html#_CPPv4NK5cudaq15scala |
|     (C++                          | r_operatoreqERK15scalar_operator) |
|     enumerator)](api/languages/   | -   [cudaq:                       |
| cpp_api.html#_CPPv4N5cudaq16noise | :scalar_operator::scalar_operator |
| _model_type18phase_flip_channelE) |     (C++                          |
| -                                 |     func                          |
| [cudaq::noise_model_type::unknown | tion)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15scalar_operator15 |
|     enumerator)](ap               | scalar_operatorENSt7complexIdEE), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[1\]](api/langu             |
| cudaq16noise_model_type7unknownE) | ages/cpp_api.html#_CPPv4N5cudaq15 |
| -                                 | scalar_operator15scalar_operatorE |
| [cudaq::noise_model_type::x_error | RK15scalar_callbackRRNSt13unorder |
|     (C++                          | ed_mapINSt6stringENSt6stringEEE), |
|     enumerator)](ap               |     [\[2\                         |
| i/languages/cpp_api.html#_CPPv4N5 | ]](api/languages/cpp_api.html#_CP |
| cudaq16noise_model_type7x_errorE) | Pv4N5cudaq15scalar_operator15scal |
| -                                 | ar_operatorERK15scalar_operator), |
| [cudaq::noise_model_type::y_error |     [\[3\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     enumerator)](ap               | scalar_operator15scalar_operatorE |
| i/languages/cpp_api.html#_CPPv4N5 | RR15scalar_callbackRRNSt13unorder |
| cudaq16noise_model_type7y_errorE) | ed_mapINSt6stringENSt6stringEEE), |
| -                                 |     [\[4\                         |
| [cudaq::noise_model_type::z_error | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operator15scal |
|     enumerator)](ap               | ar_operatorERR15scalar_operator), |
| i/languages/cpp_api.html#_CPPv4N5 |     [\[5\]](api/language          |
| cudaq16noise_model_type7z_errorE) | s/cpp_api.html#_CPPv4N5cudaq15sca |
| -   [cudaq::num_available_gpus    | lar_operator15scalar_operatorEd), |
|     (C++                          |     [\[6\]](api/languag           |
|     function                      | es/cpp_api.html#_CPPv4N5cudaq15sc |
| )](api/languages/cpp_api.html#_CP | alar_operator15scalar_operatorEv) |
| Pv4N5cudaq18num_available_gpusEv) | -   [                             |
| -   [cudaq::observe (C++          | cudaq::scalar_operator::to_matrix |
|     function)]                    |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |                                   |
| 4I00DpEN5cudaq7observeENSt6vector |   function)](api/languages/cpp_ap |
| I14observe_resultEERR13QuantumKer | i.html#_CPPv4NK5cudaq15scalar_ope |
| nelRK15SpinOpContainerDpRR4Args), | rator9to_matrixERKNSt13unordered_ |
|     [\[1\]](api/languages/cpp_ap  | mapINSt6stringENSt7complexIdEEEE) |
| i.html#_CPPv4I0DpEN5cudaq7observe | -   [                             |
| E14observe_resultNSt6size_tERR13Q | cudaq::scalar_operator::to_string |
| uantumKernelRK7spin_opDpRR4Args), |     (C++                          |
|     [\[                           |     function)](api/l              |
| 2\]](api/languages/cpp_api.html#_ | anguages/cpp_api.html#_CPPv4NK5cu |
| CPPv4I0DpEN5cudaq7observeE14obser | daq15scalar_operator9to_stringEv) |
| ve_resultRK15observe_optionsRR13Q | -   [cudaq::s                     |
| uantumKernelRK7spin_opDpRR4Args), | calar_operator::\~scalar_operator |
|     [\[3\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     functio                       |
| udaq7observeE14observe_resultRR13 | n)](api/languages/cpp_api.html#_C |
| QuantumKernelRK7spin_opDpRR4Args) | PPv4N5cudaq15scalar_operatorD0Ev) |
| -   [cudaq::observe_options (C++  | -   [cudaq::set_noise (C++        |
|     st                            |     function)](api/langu          |
| ruct)](api/languages/cpp_api.html | ages/cpp_api.html#_CPPv4N5cudaq9s |
| #_CPPv4N5cudaq15observe_optionsE) | et_noiseERKN5cudaq11noise_modelE) |
| -   [cudaq::observe_result (C++   | -   [cudaq::set_random_seed (C++  |
|                                   |     function)](api/               |
| class)](api/languages/cpp_api.htm | languages/cpp_api.html#_CPPv4N5cu |
| l#_CPPv4N5cudaq14observe_resultE) | daq15set_random_seedENSt6size_tE) |
| -                                 | -   [cudaq::simulation_precision  |
|    [cudaq::observe_result::counts |     (C++                          |
|     (C++                          |     enum)                         |
|     function)](api/languages/c    | ](api/languages/cpp_api.html#_CPP |
| pp_api.html#_CPPv4N5cudaq14observ | v4N5cudaq20simulation_precisionE) |
| e_result6countsERK12spin_op_term) | -   [                             |
| -   [cudaq::observe_result::dump  | cudaq::simulation_precision::fp32 |
|     (C++                          |     (C++                          |
|     function)                     |     enumerator)](api              |
| ](api/languages/cpp_api.html#_CPP | /languages/cpp_api.html#_CPPv4N5c |
| v4N5cudaq14observe_result4dumpEv) | udaq20simulation_precision4fp32E) |
| -   [c                            | -   [                             |
| udaq::observe_result::expectation | cudaq::simulation_precision::fp64 |
|     (C++                          |     (C++                          |
|                                   |     enumerator)](api              |
| function)](api/languages/cpp_api. | /languages/cpp_api.html#_CPPv4N5c |
| html#_CPPv4N5cudaq14observe_resul | udaq20simulation_precision4fp64E) |
| t11expectationERK12spin_op_term), | -   [cudaq::SimulationState (C++  |
|     [\[1\]](api/la                |     c                             |
| nguages/cpp_api.html#_CPPv4N5cuda | lass)](api/languages/cpp_api.html |
| q14observe_result11expectationEv) | #_CPPv4N5cudaq15SimulationStateE) |
| -   [cuda                         | -   [                             |
| q::observe_result::id_coefficient | cudaq::SimulationState::precision |
|     (C++                          |     (C++                          |
|     function)](api/langu          |     enum)](api                    |
| ages/cpp_api.html#_CPPv4N5cudaq14 | /languages/cpp_api.html#_CPPv4N5c |
| observe_result14id_coefficientEv) | udaq15SimulationState9precisionE) |
| -   [cuda                         | -   [cudaq:                       |
| q::observe_result::observe_result | :SimulationState::precision::fp32 |
|     (C++                          |     (C++                          |
|                                   |     enumerator)](api/lang         |
|   function)](api/languages/cpp_ap | uages/cpp_api.html#_CPPv4N5cudaq1 |
| i.html#_CPPv4N5cudaq14observe_res | 5SimulationState9precision4fp32E) |
| ult14observe_resultEdRK7spin_op), | -   [cudaq:                       |
|     [\[1\]](a                     | :SimulationState::precision::fp64 |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14observe_result14observe_r |     enumerator)](api/lang         |
| esultEdRK7spin_op13sample_result) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -                                 | 5SimulationState9precision4fp64E) |
|  [cudaq::observe_result::operator | -                                 |
|     double (C++                   |   [cudaq::SimulationState::Tensor |
|     functio                       |     (C++                          |
| n)](api/languages/cpp_api.html#_C |     struct)](                     |
| PPv4N5cudaq14observe_resultcvdEv) | api/languages/cpp_api.html#_CPPv4 |
| -                                 | N5cudaq15SimulationState6TensorE) |
|  [cudaq::observe_result::raw_data | -   [cudaq::spin_handler (C++     |
|     (C++                          |                                   |
|     function)](ap                 |   class)](api/languages/cpp_api.h |
| i/languages/cpp_api.html#_CPPv4N5 | tml#_CPPv4N5cudaq12spin_handlerE) |
| cudaq14observe_result8raw_dataEv) | -   [cudaq:                       |
| -   [cudaq::operator_handler (C++ | :spin_handler::to_diagonal_matrix |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     function)](api/la             |
| _CPPv4N5cudaq16operator_handlerE) | nguages/cpp_api.html#_CPPv4NK5cud |
| -   [cudaq::optimizable_function  | aq12spin_handler18to_diagonal_mat |
|     (C++                          | rixERNSt13unordered_mapINSt6size_ |
|     class)                        | tENSt7int64_tEEERKNSt13unordered_ |
| ](api/languages/cpp_api.html#_CPP | mapINSt6stringENSt7complexIdEEEE) |
| v4N5cudaq20optimizable_functionE) | -                                 |
| -   [cudaq::optimization_result   |   [cudaq::spin_handler::to_matrix |
|     (C++                          |     (C++                          |
|     type                          |     function                      |
| )](api/languages/cpp_api.html#_CP | )](api/languages/cpp_api.html#_CP |
| Pv4N5cudaq19optimization_resultE) | Pv4N5cudaq12spin_handler9to_matri |
| -   [cudaq::optimizer (C++        | xERKNSt6stringENSt7complexIdEEb), |
|     class)](api/languages/cpp_a   |     [\[1                          |
| pi.html#_CPPv4N5cudaq9optimizerE) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::optimizer::optimize   | PPv4NK5cudaq12spin_handler9to_mat |
|     (C++                          | rixERNSt13unordered_mapINSt6size_ |
|                                   | tENSt7int64_tEEERKNSt13unordered_ |
|  function)](api/languages/cpp_api | mapINSt6stringENSt7complexIdEEEE) |
| .html#_CPPv4N5cudaq9optimizer8opt | -   [cuda                         |
| imizeEKiRR20optimizable_function) | q::spin_handler::to_sparse_matrix |
| -   [cu                           |     (C++                          |
| daq::optimizer::requiresGradients |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     function)](api/la             | daq12spin_handler16to_sparse_matr |
| nguages/cpp_api.html#_CPPv4N5cuda | ixERKNSt6stringENSt7complexIdEEb) |
| q9optimizer17requiresGradientsEv) | -                                 |
| -   [cudaq::orca (C++             |   [cudaq::spin_handler::to_string |
|     type)](api/languages/         |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     function)](ap                 |
| -   [cudaq::orca::sample (C++     | i/languages/cpp_api.html#_CPPv4NK |
|     function)](api/languages/c    | 5cudaq12spin_handler9to_stringEb) |
| pp_api.html#_CPPv4N5cudaq4orca6sa | -                                 |
| mpleERNSt6vectorINSt6size_tEEERNS |   [cudaq::spin_handler::unique_id |
| t6vectorINSt6size_tEEERNSt6vector |     (C++                          |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     function)](ap                 |
|     [\[1\]]                       | i/languages/cpp_api.html#_CPPv4NK |
| (api/languages/cpp_api.html#_CPPv | 5cudaq12spin_handler9unique_idEv) |
| 4N5cudaq4orca6sampleERNSt6vectorI | -   [cudaq::spin_op (C++          |
| NSt6size_tEEERNSt6vectorINSt6size |     type)](api/languages/cpp      |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | _api.html#_CPPv4N5cudaq7spin_opE) |
| -   [cudaq::orca::sample_async    | -   [cudaq::spin_op_term (C++     |
|     (C++                          |                                   |
|                                   |    type)](api/languages/cpp_api.h |
| function)](api/languages/cpp_api. | tml#_CPPv4N5cudaq12spin_op_termE) |
| html#_CPPv4N5cudaq4orca12sample_a | -   [cudaq::state (C++            |
| syncERNSt6vectorINSt6size_tEEERNS |     class)](api/languages/c       |
| t6vectorINSt6size_tEEERNSt6vector | pp_api.html#_CPPv4N5cudaq5stateE) |
| IdEERNSt6vectorIdEEiNSt6size_tE), | -   [cudaq::state::amplitude (C++ |
|     [\[1\]](api/la                |     function)](api/lang           |
| nguages/cpp_api.html#_CPPv4N5cuda | uages/cpp_api.html#_CPPv4N5cudaq5 |
| q4orca12sample_asyncERNSt6vectorI | state9amplitudeERKNSt6vectorIiEE) |
| NSt6size_tEEERNSt6vectorINSt6size | -   [cudaq::state::amplitudes     |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |     (C++                          |
| -   [cudaq::OrcaRemoteRESTQPU     |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     cla                           | tml#_CPPv4N5cudaq5state10amplitud |
| ss)](api/languages/cpp_api.html#_ | esERKNSt6vectorINSt6vectorIiEEEE) |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | -   [cudaq::state::dump (C++      |
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

+-----------------------------------------------------------------------+
| -   [has_target() (in module                                          |
|     cudaq)](api/languages/python_api.html#cudaq.has_target)           |
+-----------------------------------------------------------------------+

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
| -   [Kernel (in module            | -   [KrausChannel (class in       |
|     cudaq)](api/langua            |     cudaq)](api/languages/py      |
| ges/python_api.html#cudaq.Kernel) | thon_api.html#cudaq.KrausChannel) |
| -   [kernel() (in module          | -   [KrausOperator (class in      |
|     cudaq)](api/langua            |     cudaq)](api/languages/pyt     |
| ges/python_api.html#cudaq.kernel) | hon_api.html#cudaq.KrausOperator) |
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
| nguages/python_api.html#cudaq.ope | -   [prepare_call()               |
| rators.MatrixOperator.parameters) |     (cudaq.PyKernelDecorator      |
|     -   [(cuda                    |     method)](a                    |
| q.operators.MatrixOperatorElement | pi/languages/python_api.html#cuda |
|         property)](api/languages  | q.PyKernelDecorator.prepare_call) |
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
|         property)](api/langu      | -   [PyKernel (class in           |
| ages/python_api.html#cudaq.operat |     cudaq)](api/language          |
| ors.spin.SpinOperator.parameters) | s/python_api.html#cudaq.PyKernel) |
|     -   [(cuda                    | -   [PyKernelDecorator (class in  |
| q.operators.spin.SpinOperatorTerm |     cudaq)](api/languages/python_ |
|         property)](api/languages  | api.html#cudaq.PyKernelDecorator) |
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
