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
    -   [PTSBE Accuracy
        Validation](examples/python/ptsbe_accuracy_validation.html){.reference
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
            -   [Submission from
                C++](using/backends/cloud/braket.html#submission-from-c){.reference
                .internal}
            -   [Submission from
                Python](using/backends/cloud/braket.html#submission-from-python){.reference
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
    -   [PTSBE](using/ptsbe.html){.reference .internal}
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
|     (cudaq.ptsbe.TraceInstruction | l#_CPPv4NK5cudaq10product_op14con |
|     at                            | st_iteratorneERK14const_iterator) |
| tribute)](api/ptsbe_api.html#cuda | -   [cudaq::produ                 |
| q.ptsbe.TraceInstruction.channel) | ct_op::const_iterator::operator\* |
| -   [circuit_location             |     (C++                          |
|     (cudaq.ptsbe.KrausSelection   |     function)](api/lang           |
|     attribute                     | uages/cpp_api.html#_CPPv4NK5cudaq |
| )](api/ptsbe_api.html#cudaq.ptsbe | 10product_op14const_iteratormlEv) |
| .KrausSelection.circuit_location) | -   [cudaq::produ                 |
| -   [clear() (cudaq.Resources     | ct_op::const_iterator::operator++ |
|     method)](api/languages/pytho  |     (C++                          |
| n_api.html#cudaq.Resources.clear) |     function)](api/lang           |
|     -   [(cudaq.SampleResult      | uages/cpp_api.html#_CPPv4N5cudaq1 |
|                                   | 0product_op14const_iteratorppEi), |
|   method)](api/languages/python_a |     [\[1\]](api/lan               |
| pi.html#cudaq.SampleResult.clear) | guages/cpp_api.html#_CPPv4N5cudaq |
| -   [COBYLA (class in             | 10product_op14const_iteratorppEv) |
|     cudaq.o                       | -   [cudaq::produc                |
| ptimizers)](api/languages/python_ | t_op::const_iterator::operator\-- |
| api.html#cudaq.optimizers.COBYLA) |     (C++                          |
| -   [coefficient                  |     function)](api/lang           |
|     (cudaq.                       | uages/cpp_api.html#_CPPv4N5cudaq1 |
| operators.boson.BosonOperatorTerm | 0product_op14const_iteratormmEi), |
|     property)](api/languages/py   |     [\[1\]](api/lan               |
| thon_api.html#cudaq.operators.bos | guages/cpp_api.html#_CPPv4N5cudaq |
| on.BosonOperatorTerm.coefficient) | 10product_op14const_iteratormmEv) |
|     -   [(cudaq.oper              | -   [cudaq::produc                |
| ators.fermion.FermionOperatorTerm | t_op::const_iterator::operator-\> |
|                                   |     (C++                          |
|   property)](api/languages/python |     function)](api/lan            |
| _api.html#cudaq.operators.fermion | guages/cpp_api.html#_CPPv4N5cudaq |
| .FermionOperatorTerm.coefficient) | 10product_op14const_iteratorptEv) |
|     -   [(c                       | -   [cudaq::produ                 |
| udaq.operators.MatrixOperatorTerm | ct_op::const_iterator::operator== |
|         property)](api/languag    |     (C++                          |
| es/python_api.html#cudaq.operator |     fun                           |
| s.MatrixOperatorTerm.coefficient) | ction)](api/languages/cpp_api.htm |
|     -   [(cuda                    | l#_CPPv4NK5cudaq10product_op14con |
| q.operators.spin.SpinOperatorTerm | st_iteratoreqERK14const_iterator) |
|         property)](api/languages/ | -   [cudaq::product_op::degrees   |
| python_api.html#cudaq.operators.s |     (C++                          |
| pin.SpinOperatorTerm.coefficient) |     function)                     |
| -   [col_count                    | ](api/languages/cpp_api.html#_CPP |
|     (cudaq.KrausOperator          | v4NK5cudaq10product_op7degreesEv) |
|     prope                         | -   [cudaq::product_op::dump (C++ |
| rty)](api/languages/python_api.ht |     functi                        |
| ml#cudaq.KrausOperator.col_count) | on)](api/languages/cpp_api.html#_ |
| -   [ComplexMatrix (class in      | CPPv4NK5cudaq10product_op4dumpEv) |
|     cudaq)](api/languages/pyt     | -   [cudaq::product_op::end (C++  |
| hon_api.html#cudaq.ComplexMatrix) |     funct                         |
| -   [compute()                    | ion)](api/languages/cpp_api.html# |
|     (                             | _CPPv4NK5cudaq10product_op3endEv) |
| cudaq.gradients.CentralDifference | -   [c                            |
|     method)](api/la               | udaq::product_op::get_coefficient |
| nguages/python_api.html#cudaq.gra |     (C++                          |
| dients.CentralDifference.compute) |     function)](api/lan            |
|     -   [(                        | guages/cpp_api.html#_CPPv4NK5cuda |
| cudaq.gradients.ForwardDifference | q10product_op15get_coefficientEv) |
|         method)](api/la           | -                                 |
| nguages/python_api.html#cudaq.gra |   [cudaq::product_op::get_term_id |
| dients.ForwardDifference.compute) |     (C++                          |
|     -                             |     function)](api                |
|  [(cudaq.gradients.ParameterShift | /languages/cpp_api.html#_CPPv4NK5 |
|         method)](api              | cudaq10product_op11get_term_idEv) |
| /languages/python_api.html#cudaq. | -                                 |
| gradients.ParameterShift.compute) |   [cudaq::product_op::is_identity |
| -   [const()                      |     (C++                          |
|                                   |     function)](api                |
|   (cudaq.operators.ScalarOperator | /languages/cpp_api.html#_CPPv4NK5 |
|     class                         | cudaq10product_op11is_identityEv) |
|     method)](a                    | -   [cudaq::product_op::num_ops   |
| pi/languages/python_api.html#cuda |     (C++                          |
| q.operators.ScalarOperator.const) |     function)                     |
| -   [controls                     | ](api/languages/cpp_api.html#_CPP |
|     (cudaq.ptsbe.TraceInstruction | v4NK5cudaq10product_op7num_opsEv) |
|     att                           | -                                 |
| ribute)](api/ptsbe_api.html#cudaq |    [cudaq::product_op::operator\* |
| .ptsbe.TraceInstruction.controls) |     (C++                          |
| -   [copy()                       |     function)](api/languages/     |
|     (cu                           | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| daq.operators.boson.BosonOperator | oduct_opmlE10product_opI1TERK15sc |
|     method)](api/l                | alar_operatorRK10product_opI1TE), |
| anguages/python_api.html#cudaq.op |     [\[1\]](api/languages/        |
| erators.boson.BosonOperator.copy) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(cudaq.                  | oduct_opmlE10product_opI1TERK15sc |
| operators.boson.BosonOperatorTerm | alar_operatorRR10product_opI1TE), |
|         method)](api/langu        |     [\[2\]](api/languages/        |
| ages/python_api.html#cudaq.operat | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| ors.boson.BosonOperatorTerm.copy) | oduct_opmlE10product_opI1TERR15sc |
|     -   [(cudaq.                  | alar_operatorRK10product_opI1TE), |
| operators.fermion.FermionOperator |     [\[3\]](api/languages/        |
|         method)](api/langu        | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| ages/python_api.html#cudaq.operat | oduct_opmlE10product_opI1TERR15sc |
| ors.fermion.FermionOperator.copy) | alar_operatorRR10product_opI1TE), |
|     -   [(cudaq.oper              |     [\[4\]](api/                  |
| ators.fermion.FermionOperatorTerm | languages/cpp_api.html#_CPPv4I0EN |
|         method)](api/languages    | 5cudaq10product_opmlE6sum_opI1TER |
| /python_api.html#cudaq.operators. | K15scalar_operatorRK6sum_opI1TE), |
| fermion.FermionOperatorTerm.copy) |     [\[5\]](api/                  |
|     -                             | languages/cpp_api.html#_CPPv4I0EN |
|  [(cudaq.operators.MatrixOperator | 5cudaq10product_opmlE6sum_opI1TER |
|         method)](                 | K15scalar_operatorRR6sum_opI1TE), |
| api/languages/python_api.html#cud |     [\[6\]](api/                  |
| aq.operators.MatrixOperator.copy) | languages/cpp_api.html#_CPPv4I0EN |
|     -   [(c                       | 5cudaq10product_opmlE6sum_opI1TER |
| udaq.operators.MatrixOperatorTerm | R15scalar_operatorRK6sum_opI1TE), |
|         method)](api/             |     [\[7\]](api/                  |
| languages/python_api.html#cudaq.o | languages/cpp_api.html#_CPPv4I0EN |
| perators.MatrixOperatorTerm.copy) | 5cudaq10product_opmlE6sum_opI1TER |
|     -   [(                        | R15scalar_operatorRR6sum_opI1TE), |
| cudaq.operators.spin.SpinOperator |     [\[8\]](api/languages         |
|         method)](api              | /cpp_api.html#_CPPv4NK5cudaq10pro |
| /languages/python_api.html#cudaq. | duct_opmlERK6sum_opI9HandlerTyE), |
| operators.spin.SpinOperator.copy) |     [\[9\]](api/languages/cpp_a   |
|     -   [(cuda                    | pi.html#_CPPv4NKR5cudaq10product_ |
| q.operators.spin.SpinOperatorTerm | opmlERK10product_opI9HandlerTyE), |
|         method)](api/lan          |     [\[10\]](api/language         |
| guages/python_api.html#cudaq.oper | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ators.spin.SpinOperatorTerm.copy) | roduct_opmlERK15scalar_operator), |
| -   [count() (cudaq.Resources     |     [\[11\]](api/languages/cpp_a  |
|     method)](api/languages/pytho  | pi.html#_CPPv4NKR5cudaq10product_ |
| n_api.html#cudaq.Resources.count) | opmlERR10product_opI9HandlerTyE), |
|     -   [(cudaq.SampleResult      |     [\[12\]](api/language         |
|                                   | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|   method)](api/languages/python_a | roduct_opmlERR15scalar_operator), |
| pi.html#cudaq.SampleResult.count) |     [\[13\]](api/languages/cpp_   |
| -   [count_controls()             | api.html#_CPPv4NO5cudaq10product_ |
|     (cudaq.Resources              | opmlERK10product_opI9HandlerTyE), |
|     meth                          |     [\[14\]](api/languag          |
| od)](api/languages/python_api.htm | es/cpp_api.html#_CPPv4NO5cudaq10p |
| l#cudaq.Resources.count_controls) | roduct_opmlERK15scalar_operator), |
| -   [count_errors()               |     [\[15\]](api/languages/cpp_   |
|     (cudaq.ptsbe.KrausTrajectory  | api.html#_CPPv4NO5cudaq10product_ |
|     met                           | opmlERR10product_opI9HandlerTyE), |
| hod)](api/ptsbe_api.html#cudaq.pt |     [\[16\]](api/langua           |
| sbe.KrausTrajectory.count_errors) | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| -   [count_instructions()         | product_opmlERR15scalar_operator) |
|                                   | -                                 |
|   (cudaq.ptsbe.PTSBEExecutionData |   [cudaq::product_op::operator\*= |
|     method)](api                  |     (C++                          |
| /ptsbe_api.html#cudaq.ptsbe.PTSBE |     function)](api/languages/cpp  |
| ExecutionData.count_instructions) | _api.html#_CPPv4N5cudaq10product_ |
| -   [counts()                     | opmLERK10product_opI9HandlerTyE), |
|     (cudaq.ObserveResult          |     [\[1\]](api/langua            |
|                                   | ges/cpp_api.html#_CPPv4N5cudaq10p |
| method)](api/languages/python_api | roduct_opmLERK15scalar_operator), |
| .html#cudaq.ObserveResult.counts) |     [\[2\]](api/languages/cp      |
| -   [create() (in module          | p_api.html#_CPPv4N5cudaq10product |
|                                   | _opmLERR10product_opI9HandlerTyE) |
|    cudaq.boson)](api/languages/py | -   [cudaq::product_op::operator+ |
| thon_api.html#cudaq.boson.create) |     (C++                          |
|     -   [(in module               |     function)](api/langu          |
|         c                         | ages/cpp_api.html#_CPPv4I0EN5cuda |
| udaq.fermion)](api/languages/pyth | q10product_opplE6sum_opI1TERK15sc |
| on_api.html#cudaq.fermion.create) | alar_operatorRK10product_opI1TE), |
| -   [csr_spmatrix (C++            |     [\[1\]](api/                  |
|     type)](api/languages/c        | languages/cpp_api.html#_CPPv4I0EN |
| pp_api.html#_CPPv412csr_spmatrix) | 5cudaq10product_opplE6sum_opI1TER |
| -   cudaq                         | K15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/langua       |     [\[2\]](api/langu             |
| ges/python_api.html#module-cudaq) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq (C++                   | q10product_opplE6sum_opI1TERK15sc |
|     type)](api/lan                | alar_operatorRR10product_opI1TE), |
| guages/cpp_api.html#_CPPv45cudaq) |     [\[3\]](api/                  |
| -   [cudaq.apply_noise() (in      | languages/cpp_api.html#_CPPv4I0EN |
|     module                        | 5cudaq10product_opplE6sum_opI1TER |
|     cudaq)](api/languages/python_ | K15scalar_operatorRR6sum_opI1TE), |
| api.html#cudaq.cudaq.apply_noise) |     [\[4\]](api/langu             |
| -   cudaq.boson                   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     -   [module](api/languages/py | q10product_opplE6sum_opI1TERR15sc |
| thon_api.html#module-cudaq.boson) | alar_operatorRK10product_opI1TE), |
| -   cudaq.fermion                 |     [\[5\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|   -   [module](api/languages/pyth | 5cudaq10product_opplE6sum_opI1TER |
| on_api.html#module-cudaq.fermion) | R15scalar_operatorRK6sum_opI1TE), |
| -   cudaq.operators.custom        |     [\[6\]](api/langu             |
|     -   [mo                       | ages/cpp_api.html#_CPPv4I0EN5cuda |
| dule](api/languages/python_api.ht | q10product_opplE6sum_opI1TERR15sc |
| ml#module-cudaq.operators.custom) | alar_operatorRR10product_opI1TE), |
| -   [cudaq.                       |     [\[7\]](api/                  |
| ptsbe.ConditionalSamplingStrategy | languages/cpp_api.html#_CPPv4I0EN |
|     (built-in                     | 5cudaq10product_opplE6sum_opI1TER |
|     c                             | R15scalar_operatorRR6sum_opI1TE), |
| lass)](api/ptsbe_api.html#cudaq.p |     [\[8\]](api/languages/cpp_a   |
| tsbe.ConditionalSamplingStrategy) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [cudaq                        | opplERK10product_opI9HandlerTyE), |
| .ptsbe.ExhaustiveSamplingStrategy |     [\[9\]](api/language          |
|     (built-in                     | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|                                   | roduct_opplERK15scalar_operator), |
| class)](api/ptsbe_api.html#cudaq. |     [\[10\]](api/languages/       |
| ptsbe.ExhaustiveSamplingStrategy) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -   [cudaq.ptsbe.KrausSelection   | duct_opplERK6sum_opI9HandlerTyE), |
|     (built-in                     |     [\[11\]](api/languages/cpp_a  |
|     class)](api/ptsbe_api         | pi.html#_CPPv4NKR5cudaq10product_ |
| .html#cudaq.ptsbe.KrausSelection) | opplERR10product_opI9HandlerTyE), |
| -   [cudaq.ptsbe.KrausTrajectory  |     [\[12\]](api/language         |
|     (built-in                     | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     class)](api/ptsbe_api.        | roduct_opplERR15scalar_operator), |
| html#cudaq.ptsbe.KrausTrajectory) |     [\[13\]](api/languages/       |
| -   [cu                           | cpp_api.html#_CPPv4NKR5cudaq10pro |
| daq.ptsbe.OrderedSamplingStrategy | duct_opplERR6sum_opI9HandlerTyE), |
|     (built-in                     |     [\[                           |
|                                   | 14\]](api/languages/cpp_api.html# |
|    class)](api/ptsbe_api.html#cud | _CPPv4NKR5cudaq10product_opplEv), |
| aq.ptsbe.OrderedSamplingStrategy) |     [\[15\]](api/languages/cpp_   |
| -   [cudaq.pt                     | api.html#_CPPv4NO5cudaq10product_ |
| sbe.ProbabilisticSamplingStrategy | opplERK10product_opI9HandlerTyE), |
|     (built-in                     |     [\[16\]](api/languag          |
|     cla                           | es/cpp_api.html#_CPPv4NO5cudaq10p |
| ss)](api/ptsbe_api.html#cudaq.pts | roduct_opplERK15scalar_operator), |
| be.ProbabilisticSamplingStrategy) |     [\[17\]](api/languages        |
| -                                 | /cpp_api.html#_CPPv4NO5cudaq10pro |
|   [cudaq.ptsbe.PTSBEExecutionData | duct_opplERK6sum_opI9HandlerTyE), |
|     (built-in                     |     [\[18\]](api/languages/cpp_   |
|     class)](api/ptsbe_api.htm     | api.html#_CPPv4NO5cudaq10product_ |
| l#cudaq.ptsbe.PTSBEExecutionData) | opplERR10product_opI9HandlerTyE), |
| -                                 |     [\[19\]](api/languag          |
|    [cudaq.ptsbe.PTSBESampleResult | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (built-in                     | roduct_opplERR15scalar_operator), |
|     class)](api/ptsbe_api.ht      |     [\[20\]](api/languages        |
| ml#cudaq.ptsbe.PTSBESampleResult) | /cpp_api.html#_CPPv4NO5cudaq10pro |
| -                                 | duct_opplERR6sum_opI9HandlerTyE), |
|  [cudaq.ptsbe.PTSSamplingStrategy |     [                             |
|     (built-in                     | \[21\]](api/languages/cpp_api.htm |
|     class)](api/ptsbe_api.html    | l#_CPPv4NO5cudaq10product_opplEv) |
| #cudaq.ptsbe.PTSSamplingStrategy) | -   [cudaq::product_op::operator- |
| -   cudaq.ptsbe.sample()          |     (C++                          |
|     -   [built-in                 |     function)](api/langu          |
|         function](api/p           | ages/cpp_api.html#_CPPv4I0EN5cuda |
| tsbe_api.html#cudaq.ptsbe.sample) | q10product_opmiE6sum_opI1TERK15sc |
| -   cudaq.ptsbe.sample_async()    | alar_operatorRK10product_opI1TE), |
|     -   [built-in                 |     [\[1\]](api/                  |
|         function](api/ptsbe_a     | languages/cpp_api.html#_CPPv4I0EN |
| pi.html#cudaq.ptsbe.sample_async) | 5cudaq10product_opmiE6sum_opI1TER |
| -   [c                            | K15scalar_operatorRK6sum_opI1TE), |
| udaq.ptsbe.ShotAllocationStrategy |     [\[2\]](api/langu             |
|     (built-in                     | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     class)](api/ptsbe_api.html#cu | q10product_opmiE6sum_opI1TERK15sc |
| daq.ptsbe.ShotAllocationStrategy) | alar_operatorRR10product_opI1TE), |
| -   [cudaq.                       |     [\[3\]](api/                  |
| ptsbe.ShotAllocationStrategy.Type | languages/cpp_api.html#_CPPv4I0EN |
|     (built-in                     | 5cudaq10product_opmiE6sum_opI1TER |
|     c                             | K15scalar_operatorRR6sum_opI1TE), |
| lass)](api/ptsbe_api.html#cudaq.p |     [\[4\]](api/langu             |
| tsbe.ShotAllocationStrategy.Type) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq.ptsbe.TraceInstruction | q10product_opmiE6sum_opI1TERR15sc |
|     (built-in                     | alar_operatorRK10product_opI1TE), |
|     class)](api/ptsbe_api.h       |     [\[5\]](api/                  |
| tml#cudaq.ptsbe.TraceInstruction) | languages/cpp_api.html#_CPPv4I0EN |
| -                                 | 5cudaq10product_opmiE6sum_opI1TER |
| [cudaq.ptsbe.TraceInstructionType | R15scalar_operatorRK6sum_opI1TE), |
|     (built-in                     |     [\[6\]](api/langu             |
|     class)](api/ptsbe_api.html#   | ages/cpp_api.html#_CPPv4I0EN5cuda |
| cudaq.ptsbe.TraceInstructionType) | q10product_opmiE6sum_opI1TERR15sc |
| -   cudaq.spin                    | alar_operatorRR10product_opI1TE), |
|     -   [module](api/languages/p  |     [\[7\]](api/                  |
| ython_api.html#module-cudaq.spin) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::amplitude_damping     | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | R15scalar_operatorRR6sum_opI1TE), |
|     cla                           |     [\[8\]](api/languages/cpp_a   |
| ss)](api/languages/cpp_api.html#_ | pi.html#_CPPv4NKR5cudaq10product_ |
| CPPv4N5cudaq17amplitude_dampingE) | opmiERK10product_opI9HandlerTyE), |
| -                                 |     [\[9\]](api/language          |
| [cudaq::amplitude_damping_channel | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     (C++                          | roduct_opmiERK15scalar_operator), |
|     class)](api                   |     [\[10\]](api/languages/       |
| /languages/cpp_api.html#_CPPv4N5c | cpp_api.html#_CPPv4NKR5cudaq10pro |
| udaq25amplitude_damping_channelE) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [cudaq::amplitud              |     [\[11\]](api/languages/cpp_a  |
| e_damping_channel::num_parameters | pi.html#_CPPv4NKR5cudaq10product_ |
|     (C++                          | opmiERR10product_opI9HandlerTyE), |
|     member)](api/languages/cpp_a  |     [\[12\]](api/language         |
| pi.html#_CPPv4N5cudaq25amplitude_ | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| damping_channel14num_parametersE) | roduct_opmiERR15scalar_operator), |
| -   [cudaq::ampli                 |     [\[13\]](api/languages/       |
| tude_damping_channel::num_targets | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     (C++                          | duct_opmiERR6sum_opI9HandlerTyE), |
|     member)](api/languages/cp     |     [\[                           |
| p_api.html#_CPPv4N5cudaq25amplitu | 14\]](api/languages/cpp_api.html# |
| de_damping_channel11num_targetsE) | _CPPv4NKR5cudaq10product_opmiEv), |
| -   [cudaq::AnalogRemoteRESTQPU   |     [\[15\]](api/languages/cpp_   |
|     (C++                          | api.html#_CPPv4NO5cudaq10product_ |
|     class                         | opmiERK10product_opI9HandlerTyE), |
| )](api/languages/cpp_api.html#_CP |     [\[16\]](api/languag          |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::apply_noise (C++      | roduct_opmiERK15scalar_operator), |
|     function)](api/               |     [\[17\]](api/languages        |
| languages/cpp_api.html#_CPPv4I0Dp | /cpp_api.html#_CPPv4NO5cudaq10pro |
| EN5cudaq11apply_noiseEvDpRR4Args) | duct_opmiERK6sum_opI9HandlerTyE), |
| -   [cudaq::async_result (C++     |     [\[18\]](api/languages/cpp_   |
|     c                             | api.html#_CPPv4NO5cudaq10product_ |
| lass)](api/languages/cpp_api.html | opmiERR10product_opI9HandlerTyE), |
| #_CPPv4I0EN5cudaq12async_resultE) |     [\[19\]](api/languag          |
| -   [cudaq::async_result::get     | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opmiERR15scalar_operator), |
|     functi                        |     [\[20\]](api/languages        |
| on)](api/languages/cpp_api.html#_ | /cpp_api.html#_CPPv4NO5cudaq10pro |
| CPPv4N5cudaq12async_result3getEv) | duct_opmiERR6sum_opI9HandlerTyE), |
| -   [cudaq::async_sample_result   |     [                             |
|     (C++                          | \[21\]](api/languages/cpp_api.htm |
|     type                          | l#_CPPv4NO5cudaq10product_opmiEv) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::product_op::operator/ |
| Pv4N5cudaq19async_sample_resultE) |     (C++                          |
| -   [cudaq::BaseRemoteRESTQPU     |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     cla                           | roduct_opdvERK15scalar_operator), |
| ss)](api/languages/cpp_api.html#_ |     [\[1\]](api/language          |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -                                 | roduct_opdvERR15scalar_operator), |
|    [cudaq::BaseRemoteSimulatorQPU |     [\[2\]](api/languag           |
|     (C++                          | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     class)](                      | roduct_opdvERK15scalar_operator), |
| api/languages/cpp_api.html#_CPPv4 |     [\[3\]](api/langua            |
| N5cudaq22BaseRemoteSimulatorQPUE) | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| -   [cudaq::bit_flip_channel (C++ | product_opdvERR15scalar_operator) |
|     cl                            | -                                 |
| ass)](api/languages/cpp_api.html# |    [cudaq::product_op::operator/= |
| _CPPv4N5cudaq16bit_flip_channelE) |     (C++                          |
| -   [cudaq:                       |     function)](api/langu          |
| :bit_flip_channel::num_parameters | ages/cpp_api.html#_CPPv4N5cudaq10 |
|     (C++                          | product_opdVERK15scalar_operator) |
|     member)](api/langua           | -   [cudaq::product_op::operator= |
| ges/cpp_api.html#_CPPv4N5cudaq16b |     (C++                          |
| it_flip_channel14num_parametersE) |     function)](api/la             |
| -   [cud                          | nguages/cpp_api.html#_CPPv4I0_NSt |
| aq::bit_flip_channel::num_targets | 11enable_if_tIXaantNSt7is_sameI1T |
|     (C++                          | 9HandlerTyE5valueENSt16is_constru |
|     member)](api/lan              | ctibleI9HandlerTy1TE5valueEEbEEEN |
| guages/cpp_api.html#_CPPv4N5cudaq | 5cudaq10product_opaSER10product_o |
| 16bit_flip_channel11num_targetsE) | pI9HandlerTyERK10product_opI1TE), |
| -   [cudaq::boson_handler (C++    |     [\[1\]](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq10product_ |
|  class)](api/languages/cpp_api.ht | opaSERK10product_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13boson_handlerE) |     [\[2\]](api/languages/cp      |
| -   [cudaq::boson_op (C++         | p_api.html#_CPPv4N5cudaq10product |
|     type)](api/languages/cpp_     | _opaSERR10product_opI9HandlerTyE) |
| api.html#_CPPv4N5cudaq8boson_opE) | -                                 |
| -   [cudaq::boson_op_term (C++    |    [cudaq::product_op::operator== |
|                                   |     (C++                          |
|   type)](api/languages/cpp_api.ht |     function)](api/languages/cpp  |
| ml#_CPPv4N5cudaq13boson_op_termE) | _api.html#_CPPv4NK5cudaq10product |
| -   [cudaq::CodeGenConfig (C++    | _opeqERK10product_opI9HandlerTyE) |
|                                   | -                                 |
| struct)](api/languages/cpp_api.ht |  [cudaq::product_op::operator\[\] |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     (C++                          |
| -   [cudaq::commutation_relations |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     struct)]                      | 5cudaq10product_opixENSt6size_tE) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq21commutation_relationsE) |    [cudaq::product_op::product_op |
| -   [cudaq::complex (C++          |     (C++                          |
|     type)](api/languages/cpp      |     function)](api/languages/c    |
| _api.html#_CPPv4N5cudaq7complexE) | pp_api.html#_CPPv4I0_NSt11enable_ |
| -   [cudaq::complex_matrix (C++   | if_tIXaaNSt7is_sameI9HandlerTy14m |
|                                   | atrix_handlerE5valueEaantNSt7is_s |
| class)](api/languages/cpp_api.htm | ameI1T9HandlerTyE5valueENSt16is_c |
| l#_CPPv4N5cudaq14complex_matrixE) | onstructibleI9HandlerTy1TE5valueE |
| -                                 | EbEEEN5cudaq10product_op10product |
|   [cudaq::complex_matrix::adjoint | _opERK10product_opI1TERKN14matrix |
|     (C++                          | _handler20commutation_behaviorE), |
|     function)](a                  |                                   |
| pi/languages/cpp_api.html#_CPPv4N |  [\[1\]](api/languages/cpp_api.ht |
| 5cudaq14complex_matrix7adjointEv) | ml#_CPPv4I0_NSt11enable_if_tIXaan |
| -   [cudaq::                      | tNSt7is_sameI1T9HandlerTyE5valueE |
| complex_matrix::diagonal_elements | NSt16is_constructibleI9HandlerTy1 |
|     (C++                          | TE5valueEEbEEEN5cudaq10product_op |
|     function)](api/languages      | 10product_opERK10product_opI1TE), |
| /cpp_api.html#_CPPv4NK5cudaq14com |                                   |
| plex_matrix17diagonal_elementsEi) |   [\[2\]](api/languages/cpp_api.h |
| -   [cudaq::complex_matrix::dump  | tml#_CPPv4N5cudaq10product_op10pr |
|     (C++                          | oduct_opENSt6size_tENSt6size_tE), |
|     function)](api/language       |     [\[3\]](api/languages/cp      |
| s/cpp_api.html#_CPPv4NK5cudaq14co | p_api.html#_CPPv4N5cudaq10product |
| mplex_matrix4dumpERNSt7ostreamE), | _op10product_opENSt7complexIdEE), |
|     [\[1\]]                       |     [\[4\]](api/l                 |
| (api/languages/cpp_api.html#_CPPv | anguages/cpp_api.html#_CPPv4N5cud |
| 4NK5cudaq14complex_matrix4dumpEv) | aq10product_op10product_opERK10pr |
| -   [c                            | oduct_opI9HandlerTyENSt6size_tE), |
| udaq::complex_matrix::eigenvalues |     [\[5\]](api/l                 |
|     (C++                          | anguages/cpp_api.html#_CPPv4N5cud |
|     function)](api/lan            | aq10product_op10product_opERR10pr |
| guages/cpp_api.html#_CPPv4NK5cuda | oduct_opI9HandlerTyENSt6size_tE), |
| q14complex_matrix11eigenvaluesEv) |     [\[6\]](api/languages         |
| -   [cu                           | /cpp_api.html#_CPPv4N5cudaq10prod |
| daq::complex_matrix::eigenvectors | uct_op10product_opERR9HandlerTy), |
|     (C++                          |     [\[7\]](ap                    |
|     function)](api/lang           | i/languages/cpp_api.html#_CPPv4N5 |
| uages/cpp_api.html#_CPPv4NK5cudaq | cudaq10product_op10product_opEd), |
| 14complex_matrix12eigenvectorsEv) |     [\[8\]](a                     |
| -   [c                            | pi/languages/cpp_api.html#_CPPv4N |
| udaq::complex_matrix::exponential | 5cudaq10product_op10product_opEv) |
|     (C++                          | -   [cuda                         |
|     function)](api/la             | q::product_op::to_diagonal_matrix |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q14complex_matrix11exponentialEv) |     function)](api/               |
| -                                 | languages/cpp_api.html#_CPPv4NK5c |
|  [cudaq::complex_matrix::identity | udaq10product_op18to_diagonal_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     function)](api/languages      | ENSt7int64_tEEERKNSt13unordered_m |
| /cpp_api.html#_CPPv4N5cudaq14comp | apINSt6stringENSt7complexIdEEEEb) |
| lex_matrix8identityEKNSt6size_tE) | -   [cudaq::product_op::to_matrix |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::kronecker |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](api/lang           | _CPPv4NK5cudaq10product_op9to_mat |
| uages/cpp_api.html#_CPPv4I00EN5cu | rixENSt13unordered_mapINSt6size_t |
| daq14complex_matrix9kroneckerE14c | ENSt7int64_tEEERKNSt13unordered_m |
| omplex_matrix8Iterable8Iterable), | apINSt6stringENSt7complexIdEEEEb) |
|     [\[1\]](api/l                 | -   [cu                           |
| anguages/cpp_api.html#_CPPv4N5cud | daq::product_op::to_sparse_matrix |
| aq14complex_matrix9kroneckerERK14 |     (C++                          |
| complex_matrixRK14complex_matrix) |     function)](ap                 |
| -   [cudaq::c                     | i/languages/cpp_api.html#_CPPv4NK |
| omplex_matrix::minimal_eigenvalue | 5cudaq10product_op16to_sparse_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     function)](api/languages/     | ENSt7int64_tEEERKNSt13unordered_m |
| cpp_api.html#_CPPv4NK5cudaq14comp | apINSt6stringENSt7complexIdEEEEb) |
| lex_matrix18minimal_eigenvalueEv) | -   [cudaq::product_op::to_string |
| -   [                             |     (C++                          |
| cudaq::complex_matrix::operator() |     function)](                   |
|     (C++                          | api/languages/cpp_api.html#_CPPv4 |
|     function)](api/languages/cpp  | NK5cudaq10product_op9to_stringEv) |
| _api.html#_CPPv4N5cudaq14complex_ | -                                 |
| matrixclENSt6size_tENSt6size_tE), |  [cudaq::product_op::\~product_op |
|     [\[1\]](api/languages/cpp     |     (C++                          |
| _api.html#_CPPv4NK5cudaq14complex |     fu                            |
| _matrixclENSt6size_tENSt6size_tE) | nction)](api/languages/cpp_api.ht |
| -   [                             | ml#_CPPv4N5cudaq10product_opD0Ev) |
| cudaq::complex_matrix::operator\* | -   [cudaq::p                     |
|     (C++                          | tsbe::ConditionalSamplingStrategy |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14c |     class)](api                   |
| omplex_matrixmlEN14complex_matrix | /ptsbe_api.html#_CPPv4N5cudaq5pts |
| 10value_typeERK14complex_matrix), | be27ConditionalSamplingStrategyE) |
|     [\[1\]                        | -   [cuda                         |
| ](api/languages/cpp_api.html#_CPP | q::ptsbe::ConditionalSamplingStra |
| v4N5cudaq14complex_matrixmlERK14c | tegy::ConditionalSamplingStrategy |
| omplex_matrixRK14complex_matrix), |     (C++                          |
|                                   |     function)](                   |
|  [\[2\]](api/languages/cpp_api.ht | api/ptsbe_api.html#_CPPv4N5cudaq5 |
| ml#_CPPv4N5cudaq14complex_matrixm | ptsbe27ConditionalSamplingStrateg |
| lERK14complex_matrixRKNSt6vectorI | y27ConditionalSamplingStrategyE19 |
| N14complex_matrix10value_typeEEE) | TrajectoryPredicateNSt8uint64_tE) |
| -                                 | -                                 |
| [cudaq::complex_matrix::operator+ |    [cudaq::ptsbe::ConditionalSamp |
|     (C++                          | lingStrategy::TrajectoryPredicate |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |                                   |
| Pv4N5cudaq14complex_matrixplERK14 |   type)](api/ptsbe_api.html#_CPPv |
| complex_matrixRK14complex_matrix) | 4N5cudaq5ptsbe27ConditionalSampli |
| -                                 | ngStrategy19TrajectoryPredicateE) |
| [cudaq::complex_matrix::operator- | -   [cudaq::                      |
|     (C++                          | ptsbe::ExhaustiveSamplingStrategy |
|     function                      |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     class)](ap                    |
| Pv4N5cudaq14complex_matrixmiERK14 | i/ptsbe_api.html#_CPPv4N5cudaq5pt |
| complex_matrixRK14complex_matrix) | sbe26ExhaustiveSamplingStrategyE) |
| -   [cu                           | -   [cuda                         |
| daq::complex_matrix::operator\[\] | q::ptsbe::OrderedSamplingStrategy |
|     (C++                          |     (C++                          |
|                                   |     class)]                       |
|  function)](api/languages/cpp_api | (api/ptsbe_api.html#_CPPv4N5cudaq |
| .html#_CPPv4N5cudaq14complex_matr | 5ptsbe23OrderedSamplingStrategyE) |
| ixixERKNSt6vectorINSt6size_tEEE), | -   [cudaq::pts                   |
|     [\[1\]](api/languages/cpp_api | be::ProbabilisticSamplingStrategy |
| .html#_CPPv4NK5cudaq14complex_mat |     (C++                          |
| rixixERKNSt6vectorINSt6size_tEEE) |     class)](api/p                 |
| -   [cudaq::complex_matrix::power | tsbe_api.html#_CPPv4N5cudaq5ptsbe |
|     (C++                          | 29ProbabilisticSamplingStrategyE) |
|     function)]                    | -   [cudaq::p                     |
| (api/languages/cpp_api.html#_CPPv | tsbe::ProbabilisticSamplingStrate |
| 4N5cudaq14complex_matrix5powerEi) | gy::ProbabilisticSamplingStrategy |
| -                                 |     (C++                          |
|  [cudaq::complex_matrix::set_zero |     function)](api/ptsbe_api.ht   |
|     (C++                          | ml#_CPPv4N5cudaq5ptsbe29Probabili |
|     function)](ap                 | sticSamplingStrategy29Probabilist |
| i/languages/cpp_api.html#_CPPv4N5 | icSamplingStrategyENSt8uint64_tE) |
| cudaq14complex_matrix8set_zeroEv) | -                                 |
| -                                 | [cudaq::ptsbe::PTSBEExecutionData |
| [cudaq::complex_matrix::to_string |     (C++                          |
|     (C++                          |     str                           |
|     function)](api/               | uct)](api/ptsbe_api.html#_CPPv4N5 |
| languages/cpp_api.html#_CPPv4NK5c | cudaq5ptsbe18PTSBEExecutionDataE) |
| udaq14complex_matrix9to_stringEv) | -   [cudaq::ptsbe::PTSBE          |
| -   [                             | ExecutionData::count_instructions |
| cudaq::complex_matrix::value_type |     (C++                          |
|     (C++                          |     function                      |
|     type)](api/                   | )](api/ptsbe_api.html#_CPPv4NK5cu |
| languages/cpp_api.html#_CPPv4N5cu | daq5ptsbe18PTSBEExecutionData18co |
| daq14complex_matrix10value_typeE) | unt_instructionsE20TraceInstructi |
| -   [cudaq::contrib (C++          | onTypeNSt8optionalINSt6stringEEE) |
|     type)](api/languages/cpp      | -   [cudaq::ptsbe::P              |
| _api.html#_CPPv4N5cudaq7contribE) | TSBEExecutionData::get_trajectory |
| -   [cudaq::contrib::draw (C++    |     (C++                          |
|     function)                     |                                   |
| ](api/languages/cpp_api.html#_CPP | function)](api/ptsbe_api.html#_CP |
| v4I0DpEN5cudaq7contrib4drawENSt6s | Pv4NK5cudaq5ptsbe18PTSBEExecution |
| tringERR13QuantumKernelDpRR4Args) | Data14get_trajectoryENSt6size_tE) |
| -                                 | -   [cudaq::ptsbe:                |
| [cudaq::contrib::get_unitary_cmat | :PTSBEExecutionData::instructions |
|     (C++                          |     (C++                          |
|     function)](api/languages/cp   |     member)](api/ptsb             |
| p_api.html#_CPPv4I0DpEN5cudaq7con | e_api.html#_CPPv4N5cudaq5ptsbe18P |
| trib16get_unitary_cmatE14complex_ | TSBEExecutionData12instructionsE) |
| matrixRR13QuantumKernelDpRR4Args) | -   [cudaq::ptsbe:                |
| -   [cudaq::CusvState (C++        | :PTSBEExecutionData::trajectories |
|                                   |     (C++                          |
|    class)](api/languages/cpp_api. |     member)](api/ptsb             |
| html#_CPPv4I0EN5cudaq9CusvStateE) | e_api.html#_CPPv4N5cudaq5ptsbe18P |
| -   [cudaq::depolarization1 (C++  | TSBEExecutionData12trajectoriesE) |
|     c                             | -   [cudaq::ptsbe::PTSBEOptions   |
| lass)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15depolarization1E) |                                   |
| -   [cudaq::depolarization2 (C++  |    struct)](api/ptsbe_api.html#_C |
|     c                             | PPv4N5cudaq5ptsbe12PTSBEOptionsE) |
| lass)](api/languages/cpp_api.html | -   [cudaq::ptsb                  |
| #_CPPv4N5cudaq15depolarization2E) | e::PTSBEOptions::max_trajectories |
| -   [cudaq:                       |     (C++                          |
| :depolarization2::depolarization2 |     member)](api/pt               |
|     (C++                          | sbe_api.html#_CPPv4N5cudaq5ptsbe1 |
|     function)](api/languages/cp   | 2PTSBEOptions16max_trajectoriesE) |
| p_api.html#_CPPv4N5cudaq15depolar | -   [cudaq::ptsbe::PT             |
| ization215depolarization2EK4real) | SBEOptions::return_execution_data |
| -   [cudaq                        |     (C++                          |
| ::depolarization2::num_parameters |     member)](api/ptsbe_a          |
|     (C++                          | pi.html#_CPPv4N5cudaq5ptsbe12PTSB |
|     member)](api/langu            | EOptions21return_execution_dataE) |
| ages/cpp_api.html#_CPPv4N5cudaq15 | -   [cudaq::pts                   |
| depolarization214num_parametersE) | be::PTSBEOptions::shot_allocation |
| -   [cu                           |     (C++                          |
| daq::depolarization2::num_targets |     member)](api/p                |
|     (C++                          | tsbe_api.html#_CPPv4N5cudaq5ptsbe |
|     member)](api/la               | 12PTSBEOptions15shot_allocationE) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cud                          |
| q15depolarization211num_targetsE) | aq::ptsbe::PTSBEOptions::strategy |
| -                                 |     (C++                          |
|    [cudaq::depolarization_channel |     member                        |
|     (C++                          | )](api/ptsbe_api.html#_CPPv4N5cud |
|     class)](                      | aq5ptsbe12PTSBEOptions8strategyE) |
| api/languages/cpp_api.html#_CPPv4 | -   [                             |
| N5cudaq22depolarization_channelE) | cudaq::ptsbe::PTSSamplingStrategy |
| -   [cudaq::depol                 |     (C++                          |
| arization_channel::num_parameters |     cla                           |
|     (C++                          | ss)](api/ptsbe_api.html#_CPPv4N5c |
|     member)](api/languages/cp     | udaq5ptsbe19PTSSamplingStrategyE) |
| p_api.html#_CPPv4N5cudaq22depolar | -   [cudaq::                      |
| ization_channel14num_parametersE) | ptsbe::PTSSamplingStrategy::clone |
| -   [cudaq::de                    |     (C++                          |
| polarization_channel::num_targets |     function)](api                |
|     (C++                          | /ptsbe_api.html#_CPPv4NK5cudaq5pt |
|     member)](api/languages        | sbe19PTSSamplingStrategy5cloneEv) |
| /cpp_api.html#_CPPv4N5cudaq22depo | -   [cudaq::ptsbe::PTSSampl       |
| larization_channel11num_targetsE) | ingStrategy::generateTrajectories |
| -   [cudaq::details (C++          |     (C++                          |
|     type)](api/languages/cpp      |     function)](                   |
| _api.html#_CPPv4N5cudaq7detailsE) | api/ptsbe_api.html#_CPPv4NK5cudaq |
| -   [cudaq::details::future (C++  | 5ptsbe19PTSSamplingStrategy20gene |
|                                   | rateTrajectoriesENSt4spanIKN5cuda |
|  class)](api/languages/cpp_api.ht | q15KrausTrajectoryEEENSt6size_tE) |
| ml#_CPPv4N5cudaq7details6futureE) | -   [cudaq:                       |
| -                                 | :ptsbe::PTSSamplingStrategy::name |
|   [cudaq::details::future::future |     (C++                          |
|     (C++                          |     function)](ap                 |
|     functio                       | i/ptsbe_api.html#_CPPv4NK5cudaq5p |
| n)](api/languages/cpp_api.html#_C | tsbe19PTSSamplingStrategy4nameEv) |
| PPv4N5cudaq7details6future6future | -   [cudaq::ptsbe::sample (C++    |
| ERNSt6vectorI3JobEERNSt6stringERN |     function)](api/ptsbe_ap       |
| St3mapINSt6stringENSt6stringEEE), | i.html#_CPPv4I0DpEN5cudaq5ptsbe6s |
|     [\[1\]](api/lang              | ampleE13sample_resultRK14sample_o |
| uages/cpp_api.html#_CPPv4N5cudaq7 | ptionsRR13QuantumKernelDpRR4Args) |
| details6future6futureERR6future), | -   [cudaq::ptsbe::sample_async   |
|     [\[2\]]                       |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     function)](api/ptsbe_a        |
| 4N5cudaq7details6future6futureEv) | pi.html#_CPPv4I0DpEN5cudaq5ptsbe1 |
| -   [cu                           | 2sample_asyncEN5cudaq19async_samp |
| daq::details::kernel_builder_base | le_resultERK14sample_optionsRR13Q |
|     (C++                          | uantumKernelDpRR4ArgsNSt6size_tE) |
|     class)](api/l                 | -   [cudaq::ptsbe::sample_options |
| anguages/cpp_api.html#_CPPv4N5cud |     (C++                          |
| aq7details19kernel_builder_baseE) |                                   |
| -   [cudaq::details::             |  struct)](api/ptsbe_api.html#_CPP |
| kernel_builder_base::operator\<\< | v4N5cudaq5ptsbe14sample_optionsE) |
|     (C++                          | -   [cu                           |
|     function)](api/langua         | daq::ptsbe::sample_options::noise |
| ges/cpp_api.html#_CPPv4N5cudaq7de |     (C++                          |
| tails19kernel_builder_baselsERNSt |     membe                         |
| 7ostreamERK19kernel_builder_base) | r)](api/ptsbe_api.html#_CPPv4N5cu |
| -   [                             | daq5ptsbe14sample_options5noiseE) |
| cudaq::details::KernelBuilderType | -   [cu                           |
|     (C++                          | daq::ptsbe::sample_options::ptsbe |
|     class)](api                   |     (C++                          |
| /languages/cpp_api.html#_CPPv4N5c |     membe                         |
| udaq7details17KernelBuilderTypeE) | r)](api/ptsbe_api.html#_CPPv4N5cu |
| -   [cudaq::d                     | daq5ptsbe14sample_options5ptsbeE) |
| etails::KernelBuilderType::create | -   [cu                           |
|     (C++                          | daq::ptsbe::sample_options::shots |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     membe                         |
| v4N5cudaq7details17KernelBuilderT | r)](api/ptsbe_api.html#_CPPv4N5cu |
| ype6createEPN4mlir11MLIRContextE) | daq5ptsbe14sample_options5shotsE) |
| -   [cudaq::details::Ker          | -   [cudaq::ptsbe::sample_result  |
| nelBuilderType::KernelBuilderType |     (C++                          |
|     (C++                          |                                   |
|     function)](api/lang           |    class)](api/ptsbe_api.html#_CP |
| uages/cpp_api.html#_CPPv4N5cudaq7 | Pv4N5cudaq5ptsbe13sample_resultE) |
| details17KernelBuilderType17Kerne | -   [cudaq::pts                   |
| lBuilderTypeERRNSt8functionIFN4ml | be::sample_result::execution_data |
| ir4TypeEPN4mlir11MLIRContextEEEE) |     (C++                          |
| -   [cudaq::diag_matrix_callback  |     function)](api/pts            |
|     (C++                          | be_api.html#_CPPv4NK5cudaq5ptsbe1 |
|     class)                        | 3sample_result14execution_dataEv) |
| ](api/languages/cpp_api.html#_CPP | -   [cudaq::ptsbe::               |
| v4N5cudaq20diag_matrix_callbackE) | sample_result::has_execution_data |
| -   [cudaq::dyn (C++              |     (C++                          |
|     member)](api/languages        |     function)](api/ptsbe_a        |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | pi.html#_CPPv4NK5cudaq5ptsbe13sam |
| -   [cudaq::ExecutionContext (C++ | ple_result18has_execution_dataEv) |
|     cl                            | -   [cudaq::ptsbe::               |
| ass)](api/languages/cpp_api.html# | sample_result::set_execution_data |
| _CPPv4N5cudaq16ExecutionContextE) |     (C++                          |
| -   [cudaq                        |     functio                       |
| ::ExecutionContext::amplitudeMaps | n)](api/ptsbe_api.html#_CPPv4N5cu |
|     (C++                          | daq5ptsbe13sample_result18set_exe |
|     member)](api/langu            | cution_dataE18PTSBEExecutionData) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -   [cud                          |
| ExecutionContext13amplitudeMapsE) | aq::ptsbe::ShotAllocationStrategy |
| -   [c                            |     (C++                          |
| udaq::ExecutionContext::asyncExec |     struct)                       |
|     (C++                          | ](api/ptsbe_api.html#_CPPv4N5cuda |
|     member)](api/                 | q5ptsbe22ShotAllocationStrategyE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::ptsbe::Shot           |
| daq16ExecutionContext9asyncExecE) | AllocationStrategy::bias_strength |
| -   [cud                          |     (C++                          |
| aq::ExecutionContext::asyncResult |     member)](api/ptsbe_api        |
|     (C++                          | .html#_CPPv4N5cudaq5ptsbe22ShotAl |
|     member)](api/lan              | locationStrategy13bias_strengthE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::pt                    |
| 16ExecutionContext11asyncResultE) | sbe::ShotAllocationStrategy::seed |
| -   [cudaq:                       |     (C++                          |
| :ExecutionContext::batchIteration |     member)](api                  |
|     (C++                          | /ptsbe_api.html#_CPPv4N5cudaq5pts |
|     member)](api/langua           | be22ShotAllocationStrategy4seedE) |
| ges/cpp_api.html#_CPPv4N5cudaq16E | -   [cudaq::ptsbe::ShotAllocatio  |
| xecutionContext14batchIterationE) | nStrategy::ShotAllocationStrategy |
| -   [cudaq::E                     |     (C++                          |
| xecutionContext::canHandleObserve |     function)](api/ptsb           |
|     (C++                          | e_api.html#_CPPv4N5cudaq5ptsbe22S |
|     member)](api/language         | hotAllocationStrategy22ShotAlloca |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | tionStrategyE4TypedNSt8uint64_tE) |
| cutionContext16canHandleObserveE) | -   [cudaq::pt                    |
| -   [cudaq::E                     | sbe::ShotAllocationStrategy::Type |
| xecutionContext::ExecutionContext |     (C++                          |
|     (C++                          |     enum)](api                    |
|     func                          | /ptsbe_api.html#_CPPv4N5cudaq5pts |
| tion)](api/languages/cpp_api.html | be22ShotAllocationStrategy4TypeE) |
| #_CPPv4N5cudaq16ExecutionContext1 | -   [cudaq::pt                    |
| 6ExecutionContextERKNSt6stringE), | sbe::ShotAllocationStrategy::type |
|     [\[1\]](api/languages/        |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16Execu |     member)](api                  |
| tionContext16ExecutionContextERKN | /ptsbe_api.html#_CPPv4N5cudaq5pts |
| St6stringENSt6size_tENSt6size_tE) | be22ShotAllocationStrategy4typeE) |
| -   [cudaq::E                     | -   [cudaq::ptsbe::ShotAllocatio  |
| xecutionContext::expectationValue | nStrategy::Type::HIGH_WEIGHT_BIAS |
|     (C++                          |     (C++                          |
|     member)](api/language         |     e                             |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | numerator)](api/ptsbe_api.html#_C |
| cutionContext16expectationValueE) | PPv4N5cudaq5ptsbe22ShotAllocation |
| -   [cudaq::Execu                 | Strategy4Type16HIGH_WEIGHT_BIASE) |
| tionContext::explicitMeasurements | -   [cudaq::ptsbe::ShotAllocati   |
|     (C++                          | onStrategy::Type::LOW_WEIGHT_BIAS |
|     member)](api/languages/cp     |     (C++                          |
| p_api.html#_CPPv4N5cudaq16Executi |                                   |
| onContext20explicitMeasurementsE) | enumerator)](api/ptsbe_api.html#_ |
| -   [cuda                         | CPPv4N5cudaq5ptsbe22ShotAllocatio |
| q::ExecutionContext::futureResult | nStrategy4Type15LOW_WEIGHT_BIASE) |
|     (C++                          | -   [cudaq::ptsbe::ShotAlloc      |
|     member)](api/lang             | ationStrategy::Type::PROPORTIONAL |
| uages/cpp_api.html#_CPPv4N5cudaq1 |     (C++                          |
| 6ExecutionContext12futureResultE) |                                   |
| -   [cudaq::ExecutionContext      |    enumerator)](api/ptsbe_api.htm |
| ::hasConditionalsOnMeasureResults | l#_CPPv4N5cudaq5ptsbe22ShotAlloca |
|     (C++                          | tionStrategy4Type12PROPORTIONALE) |
|     mem                           | -   [cudaq::ptsbe::Shot           |
| ber)](api/languages/cpp_api.html# | AllocationStrategy::Type::UNIFORM |
| _CPPv4N5cudaq16ExecutionContext31 |     (C++                          |
| hasConditionalsOnMeasureResultsE) |     enumerator)](api/ptsbe_a      |
| -   [cudaq::Executi               | pi.html#_CPPv4N5cudaq5ptsbe22Shot |
| onContext::invocationResultBuffer | AllocationStrategy4Type7UNIFORME) |
|     (C++                          | -                                 |
|     member)](api/languages/cpp_   |   [cudaq::ptsbe::TraceInstruction |
| api.html#_CPPv4N5cudaq16Execution |     (C++                          |
| Context22invocationResultBufferE) |     s                             |
| -   [cu                           | truct)](api/ptsbe_api.html#_CPPv4 |
| daq::ExecutionContext::kernelName | N5cudaq5ptsbe16TraceInstructionE) |
|     (C++                          | -   [cudaq:                       |
|     member)](api/la               | :ptsbe::TraceInstruction::channel |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10kernelNameE) |     member)](                     |
| -   [cud                          | api/ptsbe_api.html#_CPPv4N5cudaq5 |
| aq::ExecutionContext::kernelTrace | ptsbe16TraceInstruction7channelE) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/lan              | ptsbe::TraceInstruction::controls |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 16ExecutionContext11kernelTraceE) |     member)](a                    |
| -   [cudaq:                       | pi/ptsbe_api.html#_CPPv4N5cudaq5p |
| :ExecutionContext::msm_dimensions | tsbe16TraceInstruction8controlsE) |
|     (C++                          | -   [cud                          |
|     member)](api/langua           | aq::ptsbe::TraceInstruction::name |
| ges/cpp_api.html#_CPPv4N5cudaq16E |     (C++                          |
| xecutionContext14msm_dimensionsE) |     member                        |
| -   [cudaq::                      | )](api/ptsbe_api.html#_CPPv4N5cud |
| ExecutionContext::msm_prob_err_id | aq5ptsbe16TraceInstruction4nameE) |
|     (C++                          | -   [cudaq                        |
|     member)](api/languag          | ::ptsbe::TraceInstruction::params |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15msm_prob_err_idE) |     member)]                      |
| -   [cudaq::Ex                    | (api/ptsbe_api.html#_CPPv4N5cudaq |
| ecutionContext::msm_probabilities | 5ptsbe16TraceInstruction6paramsE) |
|     (C++                          | -   [cudaq:                       |
|     member)](api/languages        | :ptsbe::TraceInstruction::targets |
| /cpp_api.html#_CPPv4N5cudaq16Exec |     (C++                          |
| utionContext17msm_probabilitiesE) |     member)](                     |
| -                                 | api/ptsbe_api.html#_CPPv4N5cudaq5 |
|    [cudaq::ExecutionContext::name | ptsbe16TraceInstruction7targetsE) |
|     (C++                          | -   [cud                          |
|     member)]                      | aq::ptsbe::TraceInstruction::type |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq16ExecutionContext4nameE) |     member                        |
| -   [cu                           | )](api/ptsbe_api.html#_CPPv4N5cud |
| daq::ExecutionContext::noiseModel | aq5ptsbe16TraceInstruction4typeE) |
|     (C++                          | -   [c                            |
|     member)](api/la               | udaq::ptsbe::TraceInstructionType |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q16ExecutionContext10noiseModelE) |     enu                           |
| -   [cudaq::Exe                   | m)](api/ptsbe_api.html#_CPPv4N5cu |
| cutionContext::numberTrajectories | daq5ptsbe20TraceInstructionTypeE) |
|     (C++                          | -   [cudaq::                      |
|     member)](api/languages/       | ptsbe::TraceInstructionType::Gate |
| cpp_api.html#_CPPv4N5cudaq16Execu |     (C++                          |
| tionContext18numberTrajectoriesE) |     enumerator)](a                |
| -   [c                            | pi/ptsbe_api.html#_CPPv4N5cudaq5p |
| udaq::ExecutionContext::optResult | tsbe20TraceInstructionType4GateE) |
|     (C++                          | -   [cudaq::ptsbe::               |
|     member)](api/                 | TraceInstructionType::Measurement |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq16ExecutionContext9optResultE) |     enumerator)](api/ptsbe        |
| -   [cudaq::Execu                 | _api.html#_CPPv4N5cudaq5ptsbe20Tr |
| tionContext::overlapComputeStates | aceInstructionType11MeasurementE) |
|     (C++                          | -   [cudaq::p                     |
|     member)](api/languages/cp     | tsbe::TraceInstructionType::Noise |
| p_api.html#_CPPv4N5cudaq16Executi |     (C++                          |
| onContext20overlapComputeStatesE) |     enumerator)](ap               |
| -   [cudaq                        | i/ptsbe_api.html#_CPPv4N5cudaq5pt |
| ::ExecutionContext::overlapResult | sbe20TraceInstructionType5NoiseE) |
|     (C++                          | -   [cudaq::QPU (C++              |
|     member)](api/langu            |     class)](api/languages         |
| ages/cpp_api.html#_CPPv4N5cudaq16 | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| ExecutionContext13overlapResultE) | -   [cudaq::QPU::beginExecution   |
| -                                 |     (C++                          |
|   [cudaq::ExecutionContext::qpuId |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     member)](                     | Pv4N5cudaq3QPU14beginExecutionEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cuda                         |
| N5cudaq16ExecutionContext5qpuIdE) | q::QPU::configureExecutionContext |
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::registerNames |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     member)](api/langu            | _CPPv4NK5cudaq3QPU25configureExec |
| ages/cpp_api.html#_CPPv4N5cudaq16 | utionContextER16ExecutionContext) |
| ExecutionContext13registerNamesE) | -   [cudaq::QPU::endExecution     |
| -   [cu                           |     (C++                          |
| daq::ExecutionContext::reorderIdx |     functi                        |
|     (C++                          | on)](api/languages/cpp_api.html#_ |
|     member)](api/la               | CPPv4N5cudaq3QPU12endExecutionEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::QPU::enqueue (C++     |
| q16ExecutionContext10reorderIdxE) |     function)](ap                 |
| -                                 | i/languages/cpp_api.html#_CPPv4N5 |
|  [cudaq::ExecutionContext::result | cudaq3QPU7enqueueER11QuantumTask) |
|     (C++                          | -   [cud                          |
|     member)](a                    | aq::QPU::finalizeExecutionContext |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq16ExecutionContext6resultE) |     func                          |
| -                                 | tion)](api/languages/cpp_api.html |
|   [cudaq::ExecutionContext::shots | #_CPPv4NK5cudaq3QPU24finalizeExec |
|     (C++                          | utionContextER16ExecutionContext) |
|     member)](                     | -   [cudaq::QPU::getConnectivity  |
| api/languages/cpp_api.html#_CPPv4 |     (C++                          |
| N5cudaq16ExecutionContext5shotsE) |     function)                     |
| -   [cudaq::                      | ](api/languages/cpp_api.html#_CPP |
| ExecutionContext::simulationState | v4N5cudaq3QPU15getConnectivityEv) |
|     (C++                          | -                                 |
|     member)](api/languag          | [cudaq::QPU::getExecutionThreadId |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     (C++                          |
| ecutionContext15simulationStateE) |     function)](api/               |
| -                                 | languages/cpp_api.html#_CPPv4NK5c |
|    [cudaq::ExecutionContext::spin | udaq3QPU20getExecutionThreadIdEv) |
|     (C++                          | -   [cudaq::QPU::getNumQubits     |
|     member)]                      |     (C++                          |
| (api/languages/cpp_api.html#_CPPv |     functi                        |
| 4N5cudaq16ExecutionContext4spinE) | on)](api/languages/cpp_api.html#_ |
| -   [cudaq::                      | CPPv4N5cudaq3QPU12getNumQubitsEv) |
| ExecutionContext::totalIterations | -   [                             |
|     (C++                          | cudaq::QPU::getRemoteCapabilities |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)](api/l              |
| ecutionContext15totalIterationsE) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cudaq::Executio              | daq3QPU21getRemoteCapabilitiesEv) |
| nContext::warnedNamedMeasurements | -   [cudaq::QPU::isEmulated (C++  |
|     (C++                          |     func                          |
|     member)](api/languages/cpp_a  | tion)](api/languages/cpp_api.html |
| pi.html#_CPPv4N5cudaq16ExecutionC | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| ontext23warnedNamedMeasurementsE) | -   [cudaq::QPU::isSimulator (C++ |
| -   [cudaq::ExecutionResult (C++  |     funct                         |
|     st                            | ion)](api/languages/cpp_api.html# |
| ruct)](api/languages/cpp_api.html | _CPPv4N5cudaq3QPU11isSimulatorEv) |
| #_CPPv4N5cudaq15ExecutionResultE) | -   [cudaq::QPU::launchKernel     |
| -   [cud                          |     (C++                          |
| aq::ExecutionResult::appendResult |     function)](api/               |
|     (C++                          | languages/cpp_api.html#_CPPv4N5cu |
|     functio                       | daq3QPU12launchKernelERKNSt6strin |
| n)](api/languages/cpp_api.html#_C | gE15KernelThunkTypePvNSt8uint64_t |
| PPv4N5cudaq15ExecutionResult12app | ENSt8uint64_tERKNSt6vectorIPvEE), |
| endResultENSt6stringENSt6size_tE) |                                   |
| -   [cu                           |  [\[1\]](api/languages/cpp_api.ht |
| daq::ExecutionResult::deserialize | ml#_CPPv4N5cudaq3QPU12launchKerne |
|     (C++                          | lERKNSt6stringERKNSt6vectorIPvEE) |
|     function)                     | -   [cudaq::QPU::onRandomSeedSet  |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq15ExecutionResult11deser |     function)](api/lang           |
| ializeERNSt6vectorINSt6size_tEEE) | uages/cpp_api.html#_CPPv4N5cudaq3 |
| -   [cudaq:                       | QPU15onRandomSeedSetENSt6size_tE) |
| :ExecutionResult::ExecutionResult | -   [cudaq::QPU::QPU (C++         |
|     (C++                          |     functio                       |
|     functio                       | n)](api/languages/cpp_api.html#_C |
| n)](api/languages/cpp_api.html#_C | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| PPv4N5cudaq15ExecutionResult15Exe |                                   |
| cutionResultE16CountsDictionary), |  [\[1\]](api/languages/cpp_api.ht |
|     [\[1\]](api/lan               | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
| guages/cpp_api.html#_CPPv4N5cudaq |     [\[2\]](api/languages/cpp_    |
| 15ExecutionResult15ExecutionResul | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| tE16CountsDictionaryNSt6stringE), | -   [cudaq::QPU::setId (C++       |
|     [\[2\                         |     function                      |
| ]](api/languages/cpp_api.html#_CP | )](api/languages/cpp_api.html#_CP |
| Pv4N5cudaq15ExecutionResult15Exec | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
| utionResultE16CountsDictionaryd), | -   [cudaq::QPU::setShots (C++    |
|                                   |     f                             |
|    [\[3\]](api/languages/cpp_api. | unction)](api/languages/cpp_api.h |
| html#_CPPv4N5cudaq15ExecutionResu | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| lt15ExecutionResultENSt6stringE), | -   [cudaq::                      |
|     [\[4\                         | QPU::supportsExplicitMeasurements |
| ]](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq15ExecutionResult15Exec |     function)](api/languag        |
| utionResultERK15ExecutionResult), | es/cpp_api.html#_CPPv4N5cudaq3QPU |
|     [\[5\]](api/language          | 28supportsExplicitMeasurementsEv) |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | -   [cudaq::QPU::\~QPU (C++       |
| cutionResult15ExecutionResultEd), |     function)](api/languages/cp   |
|     [\[6\]](api/languag           | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | -   [cudaq::QPUState (C++         |
| ecutionResult15ExecutionResultEv) |     class)](api/languages/cpp_    |
| -   [                             | api.html#_CPPv4N5cudaq8QPUStateE) |
| cudaq::ExecutionResult::operator= | -   [cudaq::qreg (C++             |
|     (C++                          |     class)](api/lan               |
|     function)](api/languages/     | guages/cpp_api.html#_CPPv4I_NSt6s |
| cpp_api.html#_CPPv4N5cudaq15Execu | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| tionResultaSERK15ExecutionResult) | -   [cudaq::qreg::back (C++       |
| -   [c                            |     function)                     |
| udaq::ExecutionResult::operator== | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4backENSt6size_tE), |
|     function)](api/languages/c    |     [\[1\]](api/languages/cpp_ap  |
| pp_api.html#_CPPv4NK5cudaq15Execu | i.html#_CPPv4N5cudaq4qreg4backEv) |
| tionResulteqERK15ExecutionResult) | -   [cudaq::qreg::begin (C++      |
| -   [cud                          |                                   |
| aq::ExecutionResult::registerName |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5beginEv) |
|     member)](api/lan              | -   [cudaq::qreg::clear (C++      |
| guages/cpp_api.html#_CPPv4N5cudaq |                                   |
| 15ExecutionResult12registerNameE) |  function)](api/languages/cpp_api |
| -   [cudaq                        | .html#_CPPv4N5cudaq4qreg5clearEv) |
| ::ExecutionResult::sequentialData | -   [cudaq::qreg::front (C++      |
|     (C++                          |     function)]                    |
|     member)](api/langu            | (api/languages/cpp_api.html#_CPPv |
| ages/cpp_api.html#_CPPv4N5cudaq15 | 4N5cudaq4qreg5frontENSt6size_tE), |
| ExecutionResult14sequentialDataE) |     [\[1\]](api/languages/cpp_api |
| -   [                             | .html#_CPPv4N5cudaq4qreg5frontEv) |
| cudaq::ExecutionResult::serialize | -   [cudaq::qreg::operator\[\]    |
|     (C++                          |     (C++                          |
|     function)](api/l              |     functi                        |
| anguages/cpp_api.html#_CPPv4NK5cu | on)](api/languages/cpp_api.html#_ |
| daq15ExecutionResult9serializeEv) | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| -   [cudaq::fermion_handler (C++  | -   [cudaq::qreg::qreg (C++       |
|     c                             |     function)                     |
| lass)](api/languages/cpp_api.html | ](api/languages/cpp_api.html#_CPP |
| #_CPPv4N5cudaq15fermion_handlerE) | v4N5cudaq4qreg4qregENSt6size_tE), |
| -   [cudaq::fermion_op (C++       |     [\[1\]](api/languages/cpp_ap  |
|     type)](api/languages/cpp_api  | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| .html#_CPPv4N5cudaq10fermion_opE) | -   [cudaq::qreg::size (C++       |
| -   [cudaq::fermion_op_term (C++  |                                   |
|                                   |  function)](api/languages/cpp_api |
| type)](api/languages/cpp_api.html | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
| #_CPPv4N5cudaq15fermion_op_termE) | -   [cudaq::qreg::slice (C++      |
| -   [cudaq::FermioniqBaseQPU (C++ |     function)](api/langu          |
|     cl                            | ages/cpp_api.html#_CPPv4N5cudaq4q |
| ass)](api/languages/cpp_api.html# | reg5sliceENSt6size_tENSt6size_tE) |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | -   [cudaq::qreg::value_type (C++ |
| -   [cudaq::get_state (C++        |                                   |
|                                   | type)](api/languages/cpp_api.html |
|    function)](api/languages/cpp_a | #_CPPv4N5cudaq4qreg10value_typeE) |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | -   [cudaq::qspan (C++            |
| ateEDaRR13QuantumKernelDpRR4Args) |     class)](api/lang              |
| -   [cudaq::gradient (C++         | uages/cpp_api.html#_CPPv4I_NSt6si |
|     class)](api/languages/cpp_    | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
| api.html#_CPPv4N5cudaq8gradientE) | -   [cudaq::QuakeValue (C++       |
| -   [cudaq::gradient::clone (C++  |     class)](api/languages/cpp_api |
|     fun                           | .html#_CPPv4N5cudaq10QuakeValueE) |
| ction)](api/languages/cpp_api.htm | -   [cudaq::Q                     |
| l#_CPPv4N5cudaq8gradient5cloneEv) | uakeValue::canValidateNumElements |
| -   [cudaq::gradient::compute     |     (C++                          |
|     (C++                          |     function)](api/languages      |
|     function)](api/language       | /cpp_api.html#_CPPv4N5cudaq10Quak |
| s/cpp_api.html#_CPPv4N5cudaq8grad | eValue22canValidateNumElementsEv) |
| ient7computeERKNSt6vectorIdEERKNS | -                                 |
| t8functionIFdNSt6vectorIdEEEEEd), |  [cudaq::QuakeValue::constantSize |
|     [\[1\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api                |
| cudaq8gradient7computeERKNSt6vect | /languages/cpp_api.html#_CPPv4N5c |
| orIdEERNSt6vectorIdEERK7spin_opd) | udaq10QuakeValue12constantSizeEv) |
| -   [cudaq::gradient::gradient    | -   [cudaq::QuakeValue::dump (C++ |
|     (C++                          |     function)](api/lan            |
|     function)](api/lang           | guages/cpp_api.html#_CPPv4N5cudaq |
| uages/cpp_api.html#_CPPv4I00EN5cu | 10QuakeValue4dumpERNSt7ostreamE), |
| daq8gradient8gradientER7KernelT), |     [\                            |
|                                   | [1\]](api/languages/cpp_api.html# |
|    [\[1\]](api/languages/cpp_api. | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| html#_CPPv4I00EN5cudaq8gradient8g | -   [cudaq                        |
| radientER7KernelTRR10ArgsMapper), | ::QuakeValue::getRequiredElements |
|     [\[2\                         |     (C++                          |
| ]](api/languages/cpp_api.html#_CP |     function)](api/langua         |
| Pv4I00EN5cudaq8gradient8gradientE | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| RR13QuantumKernelRR10ArgsMapper), | uakeValue19getRequiredElementsEv) |
|     [\[3                          | -   [cudaq::QuakeValue::getValue  |
| \]](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq8gradient8gradientERRN |     function)]                    |
| St8functionIFvNSt6vectorIdEEEEE), | (api/languages/cpp_api.html#_CPPv |
|     [\[                           | 4NK5cudaq10QuakeValue8getValueEv) |
| 4\]](api/languages/cpp_api.html#_ | -   [cudaq::QuakeValue::inverse   |
| CPPv4N5cudaq8gradient8gradientEv) |     (C++                          |
| -   [cudaq::gradient::setArgs     |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     fu                            | v4NK5cudaq10QuakeValue7inverseEv) |
| nction)](api/languages/cpp_api.ht | -   [cudaq::QuakeValue::isStdVec  |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     (C++                          |
| tArgsEvR13QuantumKernelDpRR4Args) |     function)                     |
| -   [cudaq::gradient::setKernel   | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq10QuakeValue8isStdVecEv) |
|     function)](api/languages/c    | -                                 |
| pp_api.html#_CPPv4I0EN5cudaq8grad |    [cudaq::QuakeValue::operator\* |
| ient9setKernelEvR13QuantumKernel) |     (C++                          |
| -   [cud                          |     function)](api                |
| aq::gradients::central_difference | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq10QuakeValuemlE10QuakeValue), |
|     class)](api/la                |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda | [\[1\]](api/languages/cpp_api.htm |
| q9gradients18central_differenceE) | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| -   [cudaq::gra                   | -   [cudaq::QuakeValue::operator+ |
| dients::central_difference::clone |     (C++                          |
|     (C++                          |     function)](api                |
|     function)](api/languages      | /languages/cpp_api.html#_CPPv4N5c |
| /cpp_api.html#_CPPv4N5cudaq9gradi | udaq10QuakeValueplE10QuakeValue), |
| ents18central_difference5cloneEv) |     [                             |
| -   [cudaq::gradi                 | \[1\]](api/languages/cpp_api.html |
| ents::central_difference::compute | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     (C++                          |                                   |
|     function)](                   | [\[2\]](api/languages/cpp_api.htm |
| api/languages/cpp_api.html#_CPPv4 | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| N5cudaq9gradients18central_differ | -   [cudaq::QuakeValue::operator- |
| ence7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |     function)](api                |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
|   [\[1\]](api/languages/cpp_api.h | udaq10QuakeValuemiE10QuakeValue), |
| tml#_CPPv4N5cudaq9gradients18cent |     [                             |
| ral_difference7computeERKNSt6vect | \[1\]](api/languages/cpp_api.html |
| orIdEERNSt6vectorIdEERK7spin_opd) | #_CPPv4N5cudaq10QuakeValuemiEKd), |
| -   [cudaq::gradie                |     [                             |
| nts::central_difference::gradient | \[2\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValuemiEKi), |
|     functio                       |                                   |
| n)](api/languages/cpp_api.html#_C | [\[3\]](api/languages/cpp_api.htm |
| PPv4I00EN5cudaq9gradients18centra | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
| l_difference8gradientER7KernelT), | -   [cudaq::QuakeValue::operator/ |
|     [\[1\]](api/langua            |     (C++                          |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     function)](api                |
| q9gradients18central_difference8g | /languages/cpp_api.html#_CPPv4N5c |
| radientER7KernelTRR10ArgsMapper), | udaq10QuakeValuedvE10QuakeValue), |
|     [\[2\]](api/languages/cpp_    |                                   |
| api.html#_CPPv4I00EN5cudaq9gradie | [\[1\]](api/languages/cpp_api.htm |
| nts18central_difference8gradientE | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| RR13QuantumKernelRR10ArgsMapper), | -                                 |
|     [\[3\]](api/languages/cpp     |  [cudaq::QuakeValue::operator\[\] |
| _api.html#_CPPv4N5cudaq9gradients |     (C++                          |
| 18central_difference8gradientERRN |     function)](api                |
| St8functionIFvNSt6vectorIdEEEEE), | /languages/cpp_api.html#_CPPv4N5c |
|     [\[4\]](api/languages/cp      | udaq10QuakeValueixEKNSt6size_tE), |
| p_api.html#_CPPv4N5cudaq9gradient |     [\[1\]](api/                  |
| s18central_difference8gradientEv) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cud                          | daq10QuakeValueixERK10QuakeValue) |
| aq::gradients::forward_difference | -                                 |
|     (C++                          |    [cudaq::QuakeValue::QuakeValue |
|     class)](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/languag        |
| q9gradients18forward_differenceE) | es/cpp_api.html#_CPPv4N5cudaq10Qu |
| -   [cudaq::gra                   | akeValue10QuakeValueERN4mlir20Imp |
| dients::forward_difference::clone | licitLocOpBuilderEN4mlir5ValueE), |
|     (C++                          |     [\[1\]                        |
|     function)](api/languages      | ](api/languages/cpp_api.html#_CPP |
| /cpp_api.html#_CPPv4N5cudaq9gradi | v4N5cudaq10QuakeValue10QuakeValue |
| ents18forward_difference5cloneEv) | ERN4mlir20ImplicitLocOpBuilderEd) |
| -   [cudaq::gradi                 | -   [cudaq::QuakeValue::size (C++ |
| ents::forward_difference::compute |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     function)](                   | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::QuakeValue::slice     |
| N5cudaq9gradients18forward_differ |     (C++                          |
| ence7computeERKNSt6vectorIdEERKNS |     function)](api/languages/cpp_ |
| t8functionIFdNSt6vectorIdEEEEEd), | api.html#_CPPv4N5cudaq10QuakeValu |
|                                   | e5sliceEKNSt6size_tEKNSt6size_tE) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq::quantum_platform (C++ |
| tml#_CPPv4N5cudaq9gradients18forw |     cl                            |
| ard_difference7computeERKNSt6vect | ass)](api/languages/cpp_api.html# |
| orIdEERNSt6vectorIdEERK7spin_opd) | _CPPv4N5cudaq16quantum_platformE) |
| -   [cudaq::gradie                | -   [cudaq:                       |
| nts::forward_difference::gradient | :quantum_platform::beginExecution |
|     (C++                          |     (C++                          |
|     functio                       |     function)](api/languag        |
| n)](api/languages/cpp_api.html#_C | es/cpp_api.html#_CPPv4N5cudaq16qu |
| PPv4I00EN5cudaq9gradients18forwar | antum_platform14beginExecutionEv) |
| d_difference8gradientER7KernelT), | -   [cudaq::quantum_pl            |
|     [\[1\]](api/langua            | atform::configureExecutionContext |
| ges/cpp_api.html#_CPPv4I00EN5cuda |     (C++                          |
| q9gradients18forward_difference8g |     function)](api/lang           |
| radientER7KernelTRR10ArgsMapper), | uages/cpp_api.html#_CPPv4NK5cudaq |
|     [\[2\]](api/languages/cpp_    | 16quantum_platform25configureExec |
| api.html#_CPPv4I00EN5cudaq9gradie | utionContextER16ExecutionContext) |
| nts18forward_difference8gradientE | -   [cuda                         |
| RR13QuantumKernelRR10ArgsMapper), | q::quantum_platform::connectivity |
|     [\[3\]](api/languages/cpp     |     (C++                          |
| _api.html#_CPPv4N5cudaq9gradients |     function)](api/langu          |
| 18forward_difference8gradientERRN | ages/cpp_api.html#_CPPv4N5cudaq16 |
| St8functionIFvNSt6vectorIdEEEEE), | quantum_platform12connectivityEv) |
|     [\[4\]](api/languages/cp      | -   [cuda                         |
| p_api.html#_CPPv4N5cudaq9gradient | q::quantum_platform::endExecution |
| s18forward_difference8gradientEv) |     (C++                          |
| -   [                             |     function)](api/langu          |
| cudaq::gradients::parameter_shift | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     (C++                          | quantum_platform12endExecutionEv) |
|     class)](api                   | -   [cudaq::q                     |
| /languages/cpp_api.html#_CPPv4N5c | uantum_platform::enqueueAsyncTask |
| udaq9gradients15parameter_shiftE) |     (C++                          |
| -   [cudaq::                      |     function)](api/languages/     |
| gradients::parameter_shift::clone | cpp_api.html#_CPPv4N5cudaq16quant |
|     (C++                          | um_platform16enqueueAsyncTaskEKNS |
|     function)](api/langua         | t6size_tER19KernelExecutionTask), |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     [\[1\]](api/languag           |
| adients15parameter_shift5cloneEv) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [cudaq::gr                    | antum_platform16enqueueAsyncTaskE |
| adients::parameter_shift::compute | KNSt6size_tERNSt8functionIFvvEEE) |
|     (C++                          | -   [cudaq::quantum_p             |
|     function                      | latform::finalizeExecutionContext |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq9gradients15parameter_s |     function)](api/languages/c    |
| hift7computeERKNSt6vectorIdEERKNS | pp_api.html#_CPPv4NK5cudaq16quant |
| t8functionIFdNSt6vectorIdEEEEEd), | um_platform24finalizeExecutionCon |
|     [\[1\]](api/languages/cpp_ap  | textERN5cudaq16ExecutionContextE) |
| i.html#_CPPv4N5cudaq9gradients15p | -   [cudaq::qua                   |
| arameter_shift7computeERKNSt6vect | ntum_platform::get_codegen_config |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gra                   |     function)](api/languages/c    |
| dients::parameter_shift::gradient | pp_api.html#_CPPv4N5cudaq16quantu |
|     (C++                          | m_platform18get_codegen_configEv) |
|     func                          | -   [cuda                         |
| tion)](api/languages/cpp_api.html | q::quantum_platform::get_exec_ctx |
| #_CPPv4I00EN5cudaq9gradients15par |     (C++                          |
| ameter_shift8gradientER7KernelT), |     function)](api/langua         |
|     [\[1\]](api/lan               | ges/cpp_api.html#_CPPv4NK5cudaq16 |
| guages/cpp_api.html#_CPPv4I00EN5c | quantum_platform12get_exec_ctxEv) |
| udaq9gradients15parameter_shift8g | -   [c                            |
| radientER7KernelTRR10ArgsMapper), | udaq::quantum_platform::get_noise |
|     [\[2\]](api/languages/c       |     (C++                          |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     function)](api/languages/c    |
| dients15parameter_shift8gradientE | pp_api.html#_CPPv4N5cudaq16quantu |
| RR13QuantumKernelRR10ArgsMapper), | m_platform9get_noiseENSt6size_tE) |
|     [\[3\]](api/languages/        | -   [cudaq:                       |
| cpp_api.html#_CPPv4N5cudaq9gradie | :quantum_platform::get_num_qubits |
| nts15parameter_shift8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |                                   |
|     [\[4\]](api/languages         | function)](api/languages/cpp_api. |
| /cpp_api.html#_CPPv4N5cudaq9gradi | html#_CPPv4NK5cudaq16quantum_plat |
| ents15parameter_shift8gradientEv) | form14get_num_qubitsENSt6size_tE) |
| -   [cudaq::kernel_builder (C++   | -   [cudaq::quantum_              |
|     clas                          | platform::get_remote_capabilities |
| s)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4IDpEN5cudaq14kernel_builderE) |     function)                     |
| -   [c                            | ](api/languages/cpp_api.html#_CPP |
| udaq::kernel_builder::constantVal | v4NK5cudaq16quantum_platform23get |
|     (C++                          | _remote_capabilitiesENSt6size_tE) |
|     function)](api/la             | -   [cudaq::qua                   |
| nguages/cpp_api.html#_CPPv4N5cuda | ntum_platform::get_runtime_target |
| q14kernel_builder11constantValEd) |     (C++                          |
| -   [cu                           |     function)](api/languages/cp   |
| daq::kernel_builder::getArguments | p_api.html#_CPPv4NK5cudaq16quantu |
|     (C++                          | m_platform18get_runtime_targetEv) |
|     function)](api/lan            | -   [cuda                         |
| guages/cpp_api.html#_CPPv4N5cudaq | q::quantum_platform::getLogStream |
| 14kernel_builder12getArgumentsEv) |     (C++                          |
| -   [cu                           |     function)](api/langu          |
| daq::kernel_builder::getNumParams | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     (C++                          | quantum_platform12getLogStreamEv) |
|     function)](api/lan            | -   [cud                          |
| guages/cpp_api.html#_CPPv4N5cudaq | aq::quantum_platform::is_emulated |
| 14kernel_builder12getNumParamsEv) |     (C++                          |
| -   [c                            |                                   |
| udaq::kernel_builder::isArgStdVec |    function)](api/languages/cpp_a |
|     (C++                          | pi.html#_CPPv4NK5cudaq16quantum_p |
|     function)](api/languages/cp   | latform11is_emulatedENSt6size_tE) |
| p_api.html#_CPPv4N5cudaq14kernel_ | -   [c                            |
| builder11isArgStdVecENSt6size_tE) | udaq::quantum_platform::is_remote |
| -   [cuda                         |     (C++                          |
| q::kernel_builder::kernel_builder |     function)](api/languages/cp   |
|     (C++                          | p_api.html#_CPPv4NK5cudaq16quantu |
|     function)](api/languages/cpp_ | m_platform9is_remoteENSt6size_tE) |
| api.html#_CPPv4N5cudaq14kernel_bu | -   [cuda                         |
| ilder14kernel_builderERNSt6vector | q::quantum_platform::is_simulator |
| IN7details17KernelBuilderTypeEEE) |     (C++                          |
| -   [cudaq::kernel_builder::name  |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     function)                     | i.html#_CPPv4NK5cudaq16quantum_pl |
| ](api/languages/cpp_api.html#_CPP | atform12is_simulatorENSt6size_tE) |
| v4N5cudaq14kernel_builder4nameEv) | -   [c                            |
| -                                 | udaq::quantum_platform::launchVQE |
|    [cudaq::kernel_builder::qalloc |     (C++                          |
|     (C++                          |     function)](                   |
|     function)](api/language       | api/languages/cpp_api.html#_CPPv4 |
| s/cpp_api.html#_CPPv4N5cudaq14ker | N5cudaq16quantum_platform9launchV |
| nel_builder6qallocE10QuakeValue), | QEEKNSt6stringEPKvPN5cudaq8gradie |
|     [\[1\]](api/language          | ntERKN5cudaq7spin_opERN5cudaq9opt |
| s/cpp_api.html#_CPPv4N5cudaq14ker | imizerEKiKNSt6size_tENSt6size_tE) |
| nel_builder6qallocEKNSt6size_tE), | -   [cudaq:                       |
|     [\[2                          | :quantum_platform::list_platforms |
| \]](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq14kernel_builder6qallo |     function)](api/languag        |
| cERNSt6vectorINSt7complexIdEEEE), | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     [\[3\]](                      | antum_platform14list_platformsEv) |
| api/languages/cpp_api.html#_CPPv4 | -                                 |
| N5cudaq14kernel_builder6qallocEv) |    [cudaq::quantum_platform::name |
| -   [cudaq::kernel_builder::swap  |     (C++                          |
|     (C++                          |     function)](a                  |
|     function)](api/language       | pi/languages/cpp_api.html#_CPPv4N |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | K5cudaq16quantum_platform4nameEv) |
| 4kernel_builder4swapEvRK10QuakeVa | -   [                             |
| lueRK10QuakeValueRK10QuakeValue), | cudaq::quantum_platform::num_qpus |
|                                   |     (C++                          |
| [\[1\]](api/languages/cpp_api.htm |     function)](api/l              |
| l#_CPPv4I00EN5cudaq14kernel_build | anguages/cpp_api.html#_CPPv4NK5cu |
| er4swapEvRKNSt6vectorI10QuakeValu | daq16quantum_platform8num_qpusEv) |
| eEERK10QuakeValueRK10QuakeValue), | -   [cudaq::                      |
|                                   | quantum_platform::onRandomSeedSet |
| [\[2\]](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14kernel_builder4s |                                   |
| wapERK10QuakeValueRK10QuakeValue) | function)](api/languages/cpp_api. |
| -   [cudaq::KernelExecutionTask   | html#_CPPv4N5cudaq16quantum_platf |
|     (C++                          | orm15onRandomSeedSetENSt6size_tE) |
|     type                          | -   [cudaq:                       |
| )](api/languages/cpp_api.html#_CP | :quantum_platform::reset_exec_ctx |
| Pv4N5cudaq19KernelExecutionTaskE) |     (C++                          |
| -   [cudaq::KernelThunkResultType |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     struct)]                      | antum_platform14reset_exec_ctxEv) |
| (api/languages/cpp_api.html#_CPPv | -   [cud                          |
| 4N5cudaq21KernelThunkResultTypeE) | aq::quantum_platform::reset_noise |
| -   [cudaq::KernelThunkType (C++  |     (C++                          |
|                                   |     function)](api/languages/cpp_ |
| type)](api/languages/cpp_api.html | api.html#_CPPv4N5cudaq16quantum_p |
| #_CPPv4N5cudaq15KernelThunkTypeE) | latform11reset_noiseENSt6size_tE) |
| -   [cudaq::kraus_channel (C++    | -   [cudaq:                       |
|                                   | :quantum_platform::resetLogStream |
|  class)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13kraus_channelE) |     function)](api/languag        |
| -   [cudaq::kraus_channel::empty  | es/cpp_api.html#_CPPv4N5cudaq16qu |
|     (C++                          | antum_platform14resetLogStreamEv) |
|     function)]                    | -   [cuda                         |
| (api/languages/cpp_api.html#_CPPv | q::quantum_platform::set_exec_ctx |
| 4NK5cudaq13kraus_channel5emptyEv) |     (C++                          |
| -   [cudaq::kraus_c               |     funct                         |
| hannel::generateUnitaryParameters | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq16quantum_platform12 |
|                                   | set_exec_ctxEP16ExecutionContext) |
|    function)](api/languages/cpp_a | -   [c                            |
| pi.html#_CPPv4N5cudaq13kraus_chan | udaq::quantum_platform::set_noise |
| nel25generateUnitaryParametersEv) |     (C++                          |
| -                                 |     function                      |
|    [cudaq::kraus_channel::get_ops | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq16quantum_platform9set_ |
|     function)](a                  | noiseEPK11noise_modelNSt6size_tE) |
| pi/languages/cpp_api.html#_CPPv4N | -   [cuda                         |
| K5cudaq13kraus_channel7get_opsEv) | q::quantum_platform::setLogStream |
| -   [cudaq::                      |     (C++                          |
| kraus_channel::is_unitary_mixture |                                   |
|     (C++                          |  function)](api/languages/cpp_api |
|     function)](api/languages      | .html#_CPPv4N5cudaq16quantum_plat |
| /cpp_api.html#_CPPv4NK5cudaq13kra | form12setLogStreamERNSt7ostreamE) |
| us_channel18is_unitary_mixtureEv) | -   [cudaq::quantum_platfor       |
| -   [cu                           | m::supports_explicit_measurements |
| daq::kraus_channel::kraus_channel |     (C++                          |
|     (C++                          |     function)](api/l              |
|     function)](api/lang           | anguages/cpp_api.html#_CPPv4NK5cu |
| uages/cpp_api.html#_CPPv4IDpEN5cu | daq16quantum_platform30supports_e |
| daq13kraus_channel13kraus_channel | xplicit_measurementsENSt6size_tE) |
| EDpRRNSt16initializer_listI1TEE), | -   [cudaq::quantum_pla           |
|                                   | tform::supports_task_distribution |
|  [\[1\]](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq13kraus_channel13 |     fu                            |
| kraus_channelERK13kraus_channel), | nction)](api/languages/cpp_api.ht |
|     [\[2\]                        | ml#_CPPv4NK5cudaq16quantum_platfo |
| ](api/languages/cpp_api.html#_CPP | rm26supports_task_distributionEv) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::quantum               |
| hannelERKNSt6vectorI8kraus_opEE), | _platform::with_execution_context |
|     [\[3\]                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)                     |
| v4N5cudaq13kraus_channel13kraus_c | ](api/languages/cpp_api.html#_CPP |
| hannelERRNSt6vectorI8kraus_opEE), | v4I0DpEN5cudaq16quantum_platform2 |
|     [\[4\]](api/lan               | 2with_execution_contextEDaR16Exec |
| guages/cpp_api.html#_CPPv4N5cudaq | utionContextRR8CallableDpRR4Args) |
| 13kraus_channel13kraus_channelEv) | -   [cudaq::QuantumTask (C++      |
| -                                 |     type)](api/languages/cpp_api. |
| [cudaq::kraus_channel::noise_type | html#_CPPv4N5cudaq11QuantumTaskE) |
|     (C++                          | -   [cudaq::qubit (C++            |
|     member)](api                  |     type)](api/languages/c        |
| /languages/cpp_api.html#_CPPv4N5c | pp_api.html#_CPPv4N5cudaq5qubitE) |
| udaq13kraus_channel10noise_typeE) | -   [cudaq::QubitConnectivity     |
| -                                 |     (C++                          |
|   [cudaq::kraus_channel::op_names |     ty                            |
|     (C++                          | pe)](api/languages/cpp_api.html#_ |
|     member)](                     | CPPv4N5cudaq17QubitConnectivityE) |
| api/languages/cpp_api.html#_CPPv4 | -   [cudaq::QubitEdge (C++        |
| N5cudaq13kraus_channel8op_namesE) |     type)](api/languages/cpp_a    |
| -                                 | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|  [cudaq::kraus_channel::operator= | -   [cudaq::qudit (C++            |
|     (C++                          |     clas                          |
|     function)](api/langua         | s)](api/languages/cpp_api.html#_C |
| ges/cpp_api.html#_CPPv4N5cudaq13k | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| raus_channelaSERK13kraus_channel) | -   [cudaq::qudit::qudit (C++     |
| -   [c                            |                                   |
| udaq::kraus_channel::operator\[\] | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq5qudit5quditEv) |
|     function)](api/l              | -   [cudaq::qvector (C++          |
| anguages/cpp_api.html#_CPPv4N5cud |     class)                        |
| aq13kraus_channelixEKNSt6size_tE) | ](api/languages/cpp_api.html#_CPP |
| -                                 | v4I_NSt6size_tEEN5cudaq7qvectorE) |
| [cudaq::kraus_channel::parameters | -   [cudaq::qvector::back (C++    |
|     (C++                          |     function)](a                  |
|     member)](api                  | pi/languages/cpp_api.html#_CPPv4N |
| /languages/cpp_api.html#_CPPv4N5c | 5cudaq7qvector4backENSt6size_tE), |
| udaq13kraus_channel10parametersE) |                                   |
| -   [cudaq::krau                  |   [\[1\]](api/languages/cpp_api.h |
| s_channel::populateDefaultOpNames | tml#_CPPv4N5cudaq7qvector4backEv) |
|     (C++                          | -   [cudaq::qvector::begin (C++   |
|     function)](api/languages/cp   |     fu                            |
| p_api.html#_CPPv4N5cudaq13kraus_c | nction)](api/languages/cpp_api.ht |
| hannel22populateDefaultOpNamesEv) | ml#_CPPv4N5cudaq7qvector5beginEv) |
| -   [cu                           | -   [cudaq::qvector::clear (C++   |
| daq::kraus_channel::probabilities |     fu                            |
|     (C++                          | nction)](api/languages/cpp_api.ht |
|     member)](api/la               | ml#_CPPv4N5cudaq7qvector5clearEv) |
| nguages/cpp_api.html#_CPPv4N5cuda | -   [cudaq::qvector::end (C++     |
| q13kraus_channel13probabilitiesE) |                                   |
| -                                 | function)](api/languages/cpp_api. |
|  [cudaq::kraus_channel::push_back | html#_CPPv4N5cudaq7qvector3endEv) |
|     (C++                          | -   [cudaq::qvector::front (C++   |
|     function)](api                |     function)](ap                 |
| /languages/cpp_api.html#_CPPv4N5c | i/languages/cpp_api.html#_CPPv4N5 |
| udaq13kraus_channel9push_backE8kr | cudaq7qvector5frontENSt6size_tE), |
| aus_opNSt8optionalINSt6stringEEE) |                                   |
| -   [cudaq::kraus_channel::size   |  [\[1\]](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     function)                     | -   [cudaq::qvector::operator=    |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4NK5cudaq13kraus_channel4sizeEv) |     functio                       |
| -   [                             | n)](api/languages/cpp_api.html#_C |
| cudaq::kraus_channel::unitary_ops | PPv4N5cudaq7qvectoraSERK7qvector) |
|     (C++                          | -   [cudaq::qvector::operator\[\] |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)                     |
| daq13kraus_channel11unitary_opsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kraus_op (C++         | v4N5cudaq7qvectorixEKNSt6size_tE) |
|     struct)](api/languages/cpp_   | -   [cudaq::qvector::qvector (C++ |
| api.html#_CPPv4N5cudaq8kraus_opE) |     function)](api/               |
| -   [cudaq::kraus_op::adjoint     | languages/cpp_api.html#_CPPv4N5cu |
|     (C++                          | daq7qvector7qvectorENSt6size_tE), |
|     functi                        |     [\[1\]](a                     |
| on)](api/languages/cpp_api.html#_ | pi/languages/cpp_api.html#_CPPv4N |
| CPPv4NK5cudaq8kraus_op7adjointEv) | 5cudaq7qvector7qvectorERK5state), |
| -   [cudaq::kraus_op::data (C++   |     [\[2\]](api                   |
|                                   | /languages/cpp_api.html#_CPPv4N5c |
|  member)](api/languages/cpp_api.h | udaq7qvector7qvectorERK7qvector), |
| tml#_CPPv4N5cudaq8kraus_op4dataE) |     [\[3\]](api/languages/cpp     |
| -   [cudaq::kraus_op::kraus_op    | _api.html#_CPPv4N5cudaq7qvector7q |
|     (C++                          | vectorERKNSt6vectorI7complexEEb), |
|     func                          |     [\[4\]](ap                    |
| tion)](api/languages/cpp_api.html | i/languages/cpp_api.html#_CPPv4N5 |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | cudaq7qvector7qvectorERR7qvector) |
| opERRNSt16initializer_listI1TEE), | -   [cudaq::qvector::size (C++    |
|                                   |     fu                            |
|  [\[1\]](api/languages/cpp_api.ht | nction)](api/languages/cpp_api.ht |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| pENSt6vectorIN5cudaq7complexEEE), | -   [cudaq::qvector::slice (C++   |
|     [\[2\]](api/l                 |     function)](api/language       |
| anguages/cpp_api.html#_CPPv4N5cud | s/cpp_api.html#_CPPv4N5cudaq7qvec |
| aq8kraus_op8kraus_opERK8kraus_op) | tor5sliceENSt6size_tENSt6size_tE) |
| -   [cudaq::kraus_op::nCols (C++  | -   [cudaq::qvector::value_type   |
|                                   |     (C++                          |
| member)](api/languages/cpp_api.ht |     typ                           |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | e)](api/languages/cpp_api.html#_C |
| -   [cudaq::kraus_op::nRows (C++  | PPv4N5cudaq7qvector10value_typeE) |
|                                   | -   [cudaq::qview (C++            |
| member)](api/languages/cpp_api.ht |     clas                          |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) | s)](api/languages/cpp_api.html#_C |
| -   [cudaq::kraus_op::operator=   | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
|     (C++                          | -   [cudaq::qview::back (C++      |
|     function)                     |     function)                     |
| ](api/languages/cpp_api.html#_CPP | ](api/languages/cpp_api.html#_CPP |
| v4N5cudaq8kraus_opaSERK8kraus_op) | v4N5cudaq5qview4backENSt6size_tE) |
| -   [cudaq::kraus_op::precision   | -   [cudaq::qview::begin (C++     |
|     (C++                          |                                   |
|     memb                          | function)](api/languages/cpp_api. |
| er)](api/languages/cpp_api.html#_ | html#_CPPv4N5cudaq5qview5beginEv) |
| CPPv4N5cudaq8kraus_op9precisionE) | -   [cudaq::qview::end (C++       |
| -   [cudaq::KrausOperatorType     |                                   |
|     (C++                          |   function)](api/languages/cpp_ap |
|     enum)](api/ptsbe_api.html#_   | i.html#_CPPv4N5cudaq5qview3endEv) |
| CPPv4N5cudaq17KrausOperatorTypeE) | -   [cudaq::qview::front (C++     |
| -   [c                            |     function)](                   |
| udaq::KrausOperatorType::IDENTITY | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | N5cudaq5qview5frontENSt6size_tE), |
|     enumerato                     |                                   |
| r)](api/ptsbe_api.html#_CPPv4N5cu |    [\[1\]](api/languages/cpp_api. |
| daq17KrausOperatorType8IDENTITYE) | html#_CPPv4N5cudaq5qview5frontEv) |
| -   [cudaq::KrausSelection (C++   | -   [cudaq::qview::operator\[\]   |
|     struct)](api/ptsbe_api.htm    |     (C++                          |
| l#_CPPv4N5cudaq14KrausSelectionE) |     functio                       |
| -   [cudaq:                       | n)](api/languages/cpp_api.html#_C |
| :KrausSelection::circuit_location | PPv4N5cudaq5qviewixEKNSt6size_tE) |
|     (C++                          | -   [cudaq::qview::qview (C++     |
|     member)](ap                   |     functio                       |
| i/ptsbe_api.html#_CPPv4N5cudaq14K | n)](api/languages/cpp_api.html#_C |
| rausSelection16circuit_locationE) | PPv4I0EN5cudaq5qview5qviewERR1R), |
| -   [cudaq::Kra                   |     [\[1                          |
| usSelection::kraus_operator_index | \]](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq5qview5qviewERK5qview) |
|     member)](api/pt               | -   [cudaq::qview::size (C++      |
| sbe_api.html#_CPPv4N5cudaq14Kraus |                                   |
| Selection20kraus_operator_indexE) | function)](api/languages/cpp_api. |
| -                                 | html#_CPPv4NK5cudaq5qview4sizeEv) |
|   [cudaq::KrausSelection::op_name | -   [cudaq::qview::slice (C++     |
|     (C++                          |     function)](api/langua         |
|     m                             | ges/cpp_api.html#_CPPv4N5cudaq5qv |
| ember)](api/ptsbe_api.html#_CPPv4 | iew5sliceENSt6size_tENSt6size_tE) |
| N5cudaq14KrausSelection7op_nameE) | -   [cudaq::qview::value_type     |
| -                                 |     (C++                          |
|    [cudaq::KrausSelection::qubits |     t                             |
|     (C++                          | ype)](api/languages/cpp_api.html# |
|                                   | _CPPv4N5cudaq5qview10value_typeE) |
| member)](api/ptsbe_api.html#_CPPv | -   [cudaq::range (C++            |
| 4N5cudaq14KrausSelection6qubitsE) |     fun                           |
| -   [cudaq::KrausTrajectory (C++  | ction)](api/languages/cpp_api.htm |
|     struct)](api/ptsbe_api.html   | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
| #_CPPv4N5cudaq15KrausTrajectoryE) | orI11ElementTypeEE11ElementType), |
| -   [cu                           |     [\[1\]](api/languages/cpp_    |
| daq::KrausTrajectory::countErrors | api.html#_CPPv4I0EN5cudaq5rangeEN |
|     (C++                          | St6vectorI11ElementTypeEE11Elemen |
|     function)](                   | tType11ElementType11ElementType), |
| api/ptsbe_api.html#_CPPv4NK5cudaq |     [                             |
| 15KrausTrajectory11countErrorsEv) | \[2\]](api/languages/cpp_api.html |
| -   [                             | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| cudaq::KrausTrajectory::isOrdered | -   [cudaq::real (C++             |
|     (C++                          |     type)](api/languages/         |
|     function                      | cpp_api.html#_CPPv4N5cudaq4realE) |
| )](api/ptsbe_api.html#_CPPv4NK5cu | -   [cudaq::registry (C++         |
| daq15KrausTrajectory9isOrderedEv) |     type)](api/languages/cpp_     |
| -   [cudaq::                      | api.html#_CPPv4N5cudaq8registryE) |
| KrausTrajectory::kraus_selections | -                                 |
|     (C++                          |  [cudaq::registry::RegisteredType |
|     member)](api                  |     (C++                          |
| /ptsbe_api.html#_CPPv4N5cudaq15Kr |     class)](api/                  |
| ausTrajectory16kraus_selectionsE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::Kr                    | 5cudaq8registry14RegisteredTypeE) |
| ausTrajectory::measurement_counts | -   [cudaq::RemoteCapabilities    |
|     (C++                          |     (C++                          |
|     member)](api/p                |     struc                         |
| tsbe_api.html#_CPPv4N5cudaq15Krau | t)](api/languages/cpp_api.html#_C |
| sTrajectory18measurement_countsE) | PPv4N5cudaq18RemoteCapabilitiesE) |
| -   [cud                          | -   [cudaq::Remo                  |
| aq::KrausTrajectory::multiplicity | teCapabilities::isRemoteSimulator |
|     (C++                          |     (C++                          |
|     member)]                      |     member)](api/languages/c      |
| (api/ptsbe_api.html#_CPPv4N5cudaq | pp_api.html#_CPPv4N5cudaq18Remote |
| 15KrausTrajectory12multiplicityE) | Capabilities17isRemoteSimulatorE) |
| -   [                             | -   [cudaq::Remot                 |
| cudaq::KrausTrajectory::num_shots | eCapabilities::RemoteCapabilities |
|     (C++                          |     (C++                          |
|     memb                          |     function)](api/languages/cpp  |
| er)](api/ptsbe_api.html#_CPPv4N5c | _api.html#_CPPv4N5cudaq18RemoteCa |
| udaq15KrausTrajectory9num_shotsE) | pabilities18RemoteCapabilitiesEb) |
| -   [cu                           | -   [cudaq:                       |
| daq::KrausTrajectory::probability | :RemoteCapabilities::stateOverlap |
|     (C++                          |     (C++                          |
|     member)                       |     member)](api/langua           |
| ](api/ptsbe_api.html#_CPPv4N5cuda | ges/cpp_api.html#_CPPv4N5cudaq18R |
| q15KrausTrajectory11probabilityE) | emoteCapabilities12stateOverlapE) |
| -   [cuda                         | -                                 |
| q::KrausTrajectory::trajectory_id |   [cudaq::RemoteCapabilities::vqe |
|     (C++                          |     (C++                          |
|     member)](                     |     member)](                     |
| api/ptsbe_api.html#_CPPv4N5cudaq1 | api/languages/cpp_api.html#_CPPv4 |
| 5KrausTrajectory13trajectory_idE) | N5cudaq18RemoteCapabilities3vqeE) |
| -   [cudaq::matrix_callback (C++  | -   [cudaq::RemoteSimulationState |
|     c                             |     (C++                          |
| lass)](api/languages/cpp_api.html |     class)]                       |
| #_CPPv4N5cudaq15matrix_callbackE) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::matrix_handler (C++   | 4N5cudaq21RemoteSimulationStateE) |
|                                   | -   [cudaq::Resources (C++        |
| class)](api/languages/cpp_api.htm |     class)](api/languages/cpp_a   |
| l#_CPPv4N5cudaq14matrix_handlerE) | pi.html#_CPPv4N5cudaq9ResourcesE) |
| -   [cudaq::mat                   | -   [cudaq::run (C++              |
| rix_handler::commutation_behavior |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     struct)](api/languages/       | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
| cpp_api.html#_CPPv4N5cudaq14matri | 5invoke_result_tINSt7decay_tI13Qu |
| x_handler20commutation_behaviorE) | antumKernelEEDpNSt7decay_tI4ARGSE |
| -                                 | EEEEENSt6size_tERN5cudaq11noise_m |
|    [cudaq::matrix_handler::define | odelERR13QuantumKernelDpRR4ARGS), |
|     (C++                          |     [\[1\]](api/langu             |
|     function)](a                  | ages/cpp_api.html#_CPPv4I0DpEN5cu |
| pi/languages/cpp_api.html#_CPPv4N | daq3runENSt6vectorINSt15invoke_re |
| 5cudaq14matrix_handler6defineENSt | sult_tINSt7decay_tI13QuantumKerne |
| 6stringENSt6vectorINSt7int64_tEEE | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| RR15matrix_callbackRKNSt13unorder | ize_tERR13QuantumKernelDpRR4ARGS) |
| ed_mapINSt6stringENSt6stringEEE), | -   [cudaq::run_async (C++        |
|                                   |     functio                       |
| [\[1\]](api/languages/cpp_api.htm | n)](api/languages/cpp_api.html#_C |
| l#_CPPv4N5cudaq14matrix_handler6d | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| efineENSt6stringENSt6vectorINSt7i | tureINSt6vectorINSt15invoke_resul |
| nt64_tEEERR15matrix_callbackRR20d | t_tINSt7decay_tI13QuantumKernelEE |
| iag_matrix_callbackRKNSt13unorder | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
| ed_mapINSt6stringENSt6stringEEE), | ze_tENSt6size_tERN5cudaq11noise_m |
|     [\[2\]](                      | odelERR13QuantumKernelDpRR4ARGS), |
| api/languages/cpp_api.html#_CPPv4 |     [\[1\]](api/la                |
| N5cudaq14matrix_handler6defineENS | nguages/cpp_api.html#_CPPv4I0DpEN |
| t6stringENSt6vectorINSt7int64_tEE | 5cudaq9run_asyncENSt6futureINSt6v |
| ERR15matrix_callbackRRNSt13unorde | ectorINSt15invoke_result_tINSt7de |
| red_mapINSt6stringENSt6stringEEE) | cay_tI13QuantumKernelEEDpNSt7deca |
| -                                 | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
|   [cudaq::matrix_handler::degrees | ize_tERR13QuantumKernelDpRR4ARGS) |
|     (C++                          | -   [cudaq::RuntimeTarget (C++    |
|     function)](ap                 |                                   |
| i/languages/cpp_api.html#_CPPv4NK | struct)](api/languages/cpp_api.ht |
| 5cudaq14matrix_handler7degreesEv) | ml#_CPPv4N5cudaq13RuntimeTargetE) |
| -                                 | -   [cudaq::sample (C++           |
|  [cudaq::matrix_handler::displace |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
|     function)](api/language       | mpleE13sample_resultRK14sample_op |
| s/cpp_api.html#_CPPv4N5cudaq14mat | tionsRR13QuantumKernelDpRR4Args), |
| rix_handler8displaceENSt6size_tE) |     [\[1\                         |
| -   [cudaq::matrix                | ]](api/languages/cpp_api.html#_CP |
| _handler::get_expected_dimensions | Pv4I0DpEN5cudaq6sampleE13sample_r |
|     (C++                          | esultRR13QuantumKernelDpRR4Args), |
|                                   |     [\                            |
|    function)](api/languages/cpp_a | [2\]](api/languages/cpp_api.html# |
| pi.html#_CPPv4NK5cudaq14matrix_ha | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| ndler23get_expected_dimensionsEv) | ize_tERR13QuantumKernelDpRR4Args) |
| -   [cudaq::matrix_ha             | -   [cudaq::sample_options (C++   |
| ndler::get_parameter_descriptions |     s                             |
|     (C++                          | truct)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq14sample_optionsE) |
| function)](api/languages/cpp_api. | -   [cudaq::sample_result (C++    |
| html#_CPPv4NK5cudaq14matrix_handl |                                   |
| er26get_parameter_descriptionsEv) |  class)](api/languages/cpp_api.ht |
| -   [c                            | ml#_CPPv4N5cudaq13sample_resultE) |
| udaq::matrix_handler::instantiate | -   [cudaq::sample_result::append |
|     (C++                          |     (C++                          |
|     function)](a                  |     function)](api/languages/cpp_ |
| pi/languages/cpp_api.html#_CPPv4N | api.html#_CPPv4N5cudaq13sample_re |
| 5cudaq14matrix_handler11instantia | sult6appendERK15ExecutionResultb) |
| teENSt6stringERKNSt6vectorINSt6si | -   [cudaq::sample_result::begin  |
| ze_tEEERK20commutation_behavior), |     (C++                          |
|     [\[1\]](                      |     function)]                    |
| api/languages/cpp_api.html#_CPPv4 | (api/languages/cpp_api.html#_CPPv |
| N5cudaq14matrix_handler11instanti | 4N5cudaq13sample_result5beginEv), |
| ateENSt6stringERRNSt6vectorINSt6s |     [\[1\]]                       |
| ize_tEEERK20commutation_behavior) | (api/languages/cpp_api.html#_CPPv |
| -   [cuda                         | 4NK5cudaq13sample_result5beginEv) |
| q::matrix_handler::matrix_handler | -   [cudaq::sample_result::cbegin |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     function)](                   |
| es/cpp_api.html#_CPPv4I0_NSt11ena | api/languages/cpp_api.html#_CPPv4 |
| ble_if_tINSt12is_base_of_vI16oper | NK5cudaq13sample_result6cbeginEv) |
| ator_handler1TEEbEEEN5cudaq14matr | -   [cudaq::sample_result::cend   |
| ix_handler14matrix_handlerERK1T), |     (C++                          |
|     [\[1\]](ap                    |     function)                     |
| i/languages/cpp_api.html#_CPPv4I0 | ](api/languages/cpp_api.html#_CPP |
| _NSt11enable_if_tINSt12is_base_of | v4NK5cudaq13sample_result4cendEv) |
| _vI16operator_handler1TEEbEEEN5cu | -   [cudaq::sample_result::clear  |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erERK1TRK20commutation_behavior), |     function)                     |
|     [\[2\]](api/languages/cpp_ap  | ](api/languages/cpp_api.html#_CPP |
| i.html#_CPPv4N5cudaq14matrix_hand | v4N5cudaq13sample_result5clearEv) |
| ler14matrix_handlerENSt6size_tE), | -   [cudaq::sample_result::count  |
|     [\[3\]](api/                  |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](                   |
| daq14matrix_handler14matrix_handl | api/languages/cpp_api.html#_CPPv4 |
| erENSt6stringERKNSt6vectorINSt6si | NK5cudaq13sample_result5countENSt |
| ze_tEEERK20commutation_behavior), | 11string_viewEKNSt11string_viewE) |
|     [\[4\]](api/                  | -   [                             |
| languages/cpp_api.html#_CPPv4N5cu | cudaq::sample_result::deserialize |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erENSt6stringERRNSt6vectorINSt6si |     functio                       |
| ze_tEEERK20commutation_behavior), | n)](api/languages/cpp_api.html#_C |
|     [\                            | PPv4N5cudaq13sample_result11deser |
| [5\]](api/languages/cpp_api.html# | ializeERNSt6vectorINSt6size_tEEE) |
| _CPPv4N5cudaq14matrix_handler14ma | -   [cudaq::sample_result::dump   |
| trix_handlerERK14matrix_handler), |     (C++                          |
|     [                             |     function)](api/languag        |
| \[6\]](api/languages/cpp_api.html | es/cpp_api.html#_CPPv4NK5cudaq13s |
| #_CPPv4N5cudaq14matrix_handler14m | ample_result4dumpERNSt7ostreamE), |
| atrix_handlerERR14matrix_handler) |     [\[1\]                        |
| -                                 | ](api/languages/cpp_api.html#_CPP |
|  [cudaq::matrix_handler::momentum | v4NK5cudaq13sample_result4dumpEv) |
|     (C++                          | -   [cudaq::sample_result::end    |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     function                      |
| rix_handler8momentumENSt6size_tE) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq13sample_result3endEv), |
|    [cudaq::matrix_handler::number |     [\[1\                         |
|     (C++                          | ]](api/languages/cpp_api.html#_CP |
|     function)](api/langua         | Pv4NK5cudaq13sample_result3endEv) |
| ges/cpp_api.html#_CPPv4N5cudaq14m | -   [                             |
| atrix_handler6numberENSt6size_tE) | cudaq::sample_result::expectation |
| -                                 |     (C++                          |
| [cudaq::matrix_handler::operator= |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     fun                           | tml#_CPPv4NK5cudaq13sample_result |
| ction)](api/languages/cpp_api.htm | 11expectationEKNSt11string_viewE) |
| l#_CPPv4I0_NSt11enable_if_tIXaant | -   [c                            |
| NSt7is_sameI1T14matrix_handlerE5v | udaq::sample_result::get_marginal |
| alueENSt12is_base_of_vI16operator |     (C++                          |
| _handler1TEEEbEEEN5cudaq14matrix_ |     function)](api/languages/cpp_ |
| handleraSER14matrix_handlerRK1T), | api.html#_CPPv4NK5cudaq13sample_r |
|     [\[1\]](api/languages         | esult12get_marginalERKNSt6vectorI |
| /cpp_api.html#_CPPv4N5cudaq14matr | NSt6size_tEEEKNSt11string_viewE), |
| ix_handleraSERK14matrix_handler), |     [\[1\]](api/languages/cpp_    |
|     [\[2\]](api/language          | api.html#_CPPv4NK5cudaq13sample_r |
| s/cpp_api.html#_CPPv4N5cudaq14mat | esult12get_marginalERRKNSt6vector |
| rix_handleraSERR14matrix_handler) | INSt6size_tEEEKNSt11string_viewE) |
| -   [                             | -   [cuda                         |
| cudaq::matrix_handler::operator== | q::sample_result::get_total_shots |
|     (C++                          |     (C++                          |
|     function)](api/languages      |     function)](api/langua         |
| /cpp_api.html#_CPPv4NK5cudaq14mat | ges/cpp_api.html#_CPPv4NK5cudaq13 |
| rix_handlereqERK14matrix_handler) | sample_result15get_total_shotsEv) |
| -                                 | -   [cuda                         |
|    [cudaq::matrix_handler::parity | q::sample_result::has_even_parity |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     fun                           |
| ges/cpp_api.html#_CPPv4N5cudaq14m | ction)](api/languages/cpp_api.htm |
| atrix_handler6parityENSt6size_tE) | l#_CPPv4N5cudaq13sample_result15h |
| -                                 | as_even_parityENSt11string_viewE) |
|  [cudaq::matrix_handler::position | -   [cuda                         |
|     (C++                          | q::sample_result::has_expectation |
|     function)](api/language       |     (C++                          |
| s/cpp_api.html#_CPPv4N5cudaq14mat |     funct                         |
| rix_handler8positionENSt6size_tE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::                      | _CPPv4NK5cudaq13sample_result15ha |
| matrix_handler::remove_definition | s_expectationEKNSt11string_viewE) |
|     (C++                          | -   [cu                           |
|     fu                            | daq::sample_result::most_probable |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq14matrix_handler1 |     fun                           |
| 7remove_definitionERKNSt6stringE) | ction)](api/languages/cpp_api.htm |
| -                                 | l#_CPPv4NK5cudaq13sample_result13 |
|   [cudaq::matrix_handler::squeeze | most_probableEKNSt11string_viewE) |
|     (C++                          | -                                 |
|     function)](api/languag        | [cudaq::sample_result::operator+= |
| es/cpp_api.html#_CPPv4N5cudaq14ma |     (C++                          |
| trix_handler7squeezeENSt6size_tE) |     function)](api/langua         |
| -   [cudaq::m                     | ges/cpp_api.html#_CPPv4N5cudaq13s |
| atrix_handler::to_diagonal_matrix | ample_resultpLERK13sample_result) |
|     (C++                          | -                                 |
|     function)](api/lang           |  [cudaq::sample_result::operator= |
| uages/cpp_api.html#_CPPv4NK5cudaq |     (C++                          |
| 14matrix_handler18to_diagonal_mat |     function)](api/langua         |
| rixERNSt13unordered_mapINSt6size_ | ges/cpp_api.html#_CPPv4N5cudaq13s |
| tENSt7int64_tEEERKNSt13unordered_ | ample_resultaSERR13sample_result) |
| mapINSt6stringENSt7complexIdEEEE) | -                                 |
| -                                 | [cudaq::sample_result::operator== |
| [cudaq::matrix_handler::to_matrix |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     function)                     | es/cpp_api.html#_CPPv4NK5cudaq13s |
| ](api/languages/cpp_api.html#_CPP | ample_resulteqERK13sample_result) |
| v4NK5cudaq14matrix_handler9to_mat | -   [                             |
| rixERNSt13unordered_mapINSt6size_ | cudaq::sample_result::probability |
| tENSt7int64_tEEERKNSt13unordered_ |     (C++                          |
| mapINSt6stringENSt7complexIdEEEE) |     function)](api/lan            |
| -                                 | guages/cpp_api.html#_CPPv4NK5cuda |
| [cudaq::matrix_handler::to_string | q13sample_result11probabilityENSt |
|     (C++                          | 11string_viewEKNSt11string_viewE) |
|     function)](api/               | -   [cud                          |
| languages/cpp_api.html#_CPPv4NK5c | aq::sample_result::register_names |
| udaq14matrix_handler9to_stringEb) |     (C++                          |
| -                                 |     function)](api/langu          |
| [cudaq::matrix_handler::unique_id | ages/cpp_api.html#_CPPv4NK5cudaq1 |
|     (C++                          | 3sample_result14register_namesEv) |
|     function)](api/               | -                                 |
| languages/cpp_api.html#_CPPv4NK5c |    [cudaq::sample_result::reorder |
| udaq14matrix_handler9unique_idEv) |     (C++                          |
| -   [cudaq:                       |     function)](api/langua         |
| :matrix_handler::\~matrix_handler | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     (C++                          | ample_result7reorderERKNSt6vector |
|     functi                        | INSt6size_tEEEKNSt11string_viewE) |
| on)](api/languages/cpp_api.html#_ | -   [cu                           |
| CPPv4N5cudaq14matrix_handlerD0Ev) | daq::sample_result::sample_result |
| -   [cudaq::matrix_op (C++        |     (C++                          |
|     type)](api/languages/cpp_a    |     func                          |
| pi.html#_CPPv4N5cudaq9matrix_opE) | tion)](api/languages/cpp_api.html |
| -   [cudaq::matrix_op_term (C++   | #_CPPv4N5cudaq13sample_result13sa |
|                                   | mple_resultERK15ExecutionResult), |
|  type)](api/languages/cpp_api.htm |     [\[1\]](api/la                |
| l#_CPPv4N5cudaq14matrix_op_termE) | nguages/cpp_api.html#_CPPv4N5cuda |
| -                                 | q13sample_result13sample_resultER |
|    [cudaq::mdiag_operator_handler | KNSt6vectorI15ExecutionResultEE), |
|     (C++                          |                                   |
|     class)](                      |  [\[2\]](api/languages/cpp_api.ht |
| api/languages/cpp_api.html#_CPPv4 | ml#_CPPv4N5cudaq13sample_result13 |
| N5cudaq22mdiag_operator_handlerE) | sample_resultERR13sample_result), |
| -   [cudaq::mpi (C++              |     [                             |
|     type)](api/languages          | \[3\]](api/languages/cpp_api.html |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | #_CPPv4N5cudaq13sample_result13sa |
| -   [cudaq::mpi::all_gather (C++  | mple_resultERR15ExecutionResult), |
|     fu                            |     [\[4\]](api/lan               |
| nction)](api/languages/cpp_api.ht | guages/cpp_api.html#_CPPv4N5cudaq |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | 13sample_result13sample_resultEdR |
| RNSt6vectorIdEERKNSt6vectorIdEE), | KNSt6vectorI15ExecutionResultEE), |
|                                   |     [\[5\]](api/lan               |
|   [\[1\]](api/languages/cpp_api.h | guages/cpp_api.html#_CPPv4N5cudaq |
| tml#_CPPv4N5cudaq3mpi10all_gather | 13sample_result13sample_resultEv) |
| ERNSt6vectorIiEERKNSt6vectorIiEE) | -                                 |
| -   [cudaq::mpi::all_reduce (C++  |  [cudaq::sample_result::serialize |
|                                   |     (C++                          |
|  function)](api/languages/cpp_api |     function)](api                |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | /languages/cpp_api.html#_CPPv4NK5 |
| reduceE1TRK1TRK14BinaryFunction), | cudaq13sample_result9serializeEv) |
|     [\[1\]](api/langu             | -   [cudaq::sample_result::size   |
| ages/cpp_api.html#_CPPv4I00EN5cud |     (C++                          |
| aq3mpi10all_reduceE1TRK1TRK4Func) |     function)](api/languages/c    |
| -   [cudaq::mpi::broadcast (C++   | pp_api.html#_CPPv4NK5cudaq13sampl |
|     function)](api/               | e_result4sizeEKNSt11string_viewE) |
| languages/cpp_api.html#_CPPv4N5cu | -   [cudaq::sample_result::to_map |
| daq3mpi9broadcastERNSt6stringEi), |     (C++                          |
|     [\[1\]](api/la                |     function)](api/languages/cpp  |
| nguages/cpp_api.html#_CPPv4N5cuda | _api.html#_CPPv4NK5cudaq13sample_ |
| q3mpi9broadcastERNSt6vectorIdEEi) | result6to_mapEKNSt11string_viewE) |
| -   [cudaq::mpi::finalize (C++    | -   [cuda                         |
|     f                             | q::sample_result::\~sample_result |
| unction)](api/languages/cpp_api.h |     (C++                          |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) |     funct                         |
| -   [cudaq::mpi::initialize (C++  | ion)](api/languages/cpp_api.html# |
|     function                      | _CPPv4N5cudaq13sample_resultD0Ev) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::scalar_callback (C++  |
| Pv4N5cudaq3mpi10initializeEiPPc), |     c                             |
|     [                             | lass)](api/languages/cpp_api.html |
| \[1\]](api/languages/cpp_api.html | #_CPPv4N5cudaq15scalar_callbackE) |
| #_CPPv4N5cudaq3mpi10initializeEv) | -   [c                            |
| -   [cudaq::mpi::is_initialized   | udaq::scalar_callback::operator() |
|     (C++                          |     (C++                          |
|     function                      |     function)](api/language       |
| )](api/languages/cpp_api.html#_CP | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| Pv4N5cudaq3mpi14is_initializedEv) | alar_callbackclERKNSt13unordered_ |
| -   [cudaq::mpi::num_ranks (C++   | mapINSt6stringENSt7complexIdEEEE) |
|     fu                            | -   [                             |
| nction)](api/languages/cpp_api.ht | cudaq::scalar_callback::operator= |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     (C++                          |
| -   [cudaq::mpi::rank (C++        |     function)](api/languages/c    |
|                                   | pp_api.html#_CPPv4N5cudaq15scalar |
|    function)](api/languages/cpp_a | _callbackaSERK15scalar_callback), |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) |     [\[1\]](api/languages/        |
| -   [cudaq::noise_model (C++      | cpp_api.html#_CPPv4N5cudaq15scala |
|                                   | r_callbackaSERR15scalar_callback) |
|    class)](api/languages/cpp_api. | -   [cudaq:                       |
| html#_CPPv4N5cudaq11noise_modelE) | :scalar_callback::scalar_callback |
| -   [cudaq::n                     |     (C++                          |
| oise_model::add_all_qubit_channel |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4I0_NSt11ena |
|     function)](api                | ble_if_tINSt16is_invocable_r_vINS |
| /languages/cpp_api.html#_CPPv4IDp | t7complexIdEE8CallableRKNSt13unor |
| EN5cudaq11noise_model21add_all_qu | dered_mapINSt6stringENSt7complexI |
| bit_channelEvRK13kraus_channeli), | dEEEEEEbEEEN5cudaq15scalar_callba |
|     [\[1\]](api/langua            | ck15scalar_callbackERR8Callable), |
| ges/cpp_api.html#_CPPv4N5cudaq11n |     [\[1\                         |
| oise_model21add_all_qubit_channel | ]](api/languages/cpp_api.html#_CP |
| ERKNSt6stringERK13kraus_channeli) | Pv4N5cudaq15scalar_callback15scal |
| -                                 | ar_callbackERK15scalar_callback), |
|  [cudaq::noise_model::add_channel |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     funct                         | PPv4N5cudaq15scalar_callback15sca |
| ion)](api/languages/cpp_api.html# | lar_callbackERR15scalar_callback) |
| _CPPv4IDpEN5cudaq11noise_model11a | -   [cudaq::scalar_operator (C++  |
| dd_channelEvRK15PredicateFuncTy), |     c                             |
|     [\[1\]](api/languages/cpp_    | lass)](api/languages/cpp_api.html |
| api.html#_CPPv4IDpEN5cudaq11noise | #_CPPv4N5cudaq15scalar_operatorE) |
| _model11add_channelEvRKNSt6vector | -                                 |
| INSt6size_tEEERK13kraus_channel), | [cudaq::scalar_operator::evaluate |
|     [\[2\]](ap                    |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |                                   |
| cudaq11noise_model11add_channelER |    function)](api/languages/cpp_a |
| KNSt6stringERK15PredicateFuncTy), | pi.html#_CPPv4NK5cudaq15scalar_op |
|                                   | erator8evaluateERKNSt13unordered_ |
| [\[3\]](api/languages/cpp_api.htm | mapINSt6stringENSt7complexIdEEEE) |
| l#_CPPv4N5cudaq11noise_model11add | -   [cudaq::scalar_ope            |
| _channelERKNSt6stringERKNSt6vecto | rator::get_parameter_descriptions |
| rINSt6size_tEEERK13kraus_channel) |     (C++                          |
| -   [cudaq::noise_model::empty    |     f                             |
|     (C++                          | unction)](api/languages/cpp_api.h |
|     function                      | tml#_CPPv4NK5cudaq15scalar_operat |
| )](api/languages/cpp_api.html#_CP | or26get_parameter_descriptionsEv) |
| Pv4NK5cudaq11noise_model5emptyEv) | -   [cu                           |
| -                                 | daq::scalar_operator::is_constant |
| [cudaq::noise_model::get_channels |     (C++                          |
|     (C++                          |     function)](api/lang           |
|     function)](api/l              | uages/cpp_api.html#_CPPv4NK5cudaq |
| anguages/cpp_api.html#_CPPv4I0ENK | 15scalar_operator11is_constantEv) |
| 5cudaq11noise_model12get_channels | -   [c                            |
| ENSt6vectorI13kraus_channelEERKNS | udaq::scalar_operator::operator\* |
| t6vectorINSt6size_tEEERKNSt6vecto |     (C++                          |
| rINSt6size_tEEERKNSt6vectorIdEE), |     function                      |
|     [\[1\]](api/languages/cpp_a   | )](api/languages/cpp_api.html#_CP |
| pi.html#_CPPv4NK5cudaq11noise_mod | Pv4N5cudaq15scalar_operatormlENSt |
| el12get_channelsERKNSt6stringERKN | 7complexIdEERK15scalar_operator), |
| St6vectorINSt6size_tEEERKNSt6vect |     [\[1\                         |
| orINSt6size_tEEERKNSt6vectorIdEE) | ]](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatormlENSt |
|  [cudaq::noise_model::noise_model | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     function)](api                | p_api.html#_CPPv4N5cudaq15scalar_ |
| /languages/cpp_api.html#_CPPv4N5c | operatormlEdRK15scalar_operator), |
| udaq11noise_model11noise_modelEv) |     [\[3\]](api/languages/cp      |
| -   [cu                           | p_api.html#_CPPv4N5cudaq15scalar_ |
| daq::noise_model::PredicateFuncTy | operatormlEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     type)](api/la                 | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| nguages/cpp_api.html#_CPPv4N5cuda | alar_operatormlENSt7complexIdEE), |
| q11noise_model15PredicateFuncTyE) |     [\[5\]](api/languages/cpp     |
| -   [cud                          | _api.html#_CPPv4NKR5cudaq15scalar |
| aq::noise_model::register_channel | _operatormlERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     function)](api/languages      | (api/languages/cpp_api.html#_CPPv |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | 4NKR5cudaq15scalar_operatormlEd), |
| noise_model16register_channelEvv) |     [\[7\]](api/language          |
| -   [cudaq::                      | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| noise_model::requires_constructor | alar_operatormlENSt7complexIdEE), |
|     (C++                          |     [\[8\]](api/languages/cp      |
|     type)](api/languages/cp       | p_api.html#_CPPv4NO5cudaq15scalar |
| p_api.html#_CPPv4I0DpEN5cudaq11no | _operatormlERK15scalar_operator), |
| ise_model20requires_constructorE) |     [\[9\                         |
| -   [cudaq::noise_model_type (C++ | ]](api/languages/cpp_api.html#_CP |
|     e                             | Pv4NO5cudaq15scalar_operatormlEd) |
| num)](api/languages/cpp_api.html# | -   [cu                           |
| _CPPv4N5cudaq16noise_model_typeE) | daq::scalar_operator::operator\*= |
| -   [cudaq::no                    |     (C++                          |
| ise_model_type::amplitude_damping |     function)](api/languag        |
|     (C++                          | es/cpp_api.html#_CPPv4N5cudaq15sc |
|     enumerator)](api/languages    | alar_operatormLENSt7complexIdEE), |
| /cpp_api.html#_CPPv4N5cudaq16nois |     [\[1\]](api/languages/c       |
| e_model_type17amplitude_dampingE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::noise_mode            | _operatormLERK15scalar_operator), |
| l_type::amplitude_damping_channel |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     e                             | PPv4N5cudaq15scalar_operatormLEd) |
| numerator)](api/languages/cpp_api | -   [                             |
| .html#_CPPv4N5cudaq16noise_model_ | cudaq::scalar_operator::operator+ |
| type25amplitude_damping_channelE) |     (C++                          |
| -   [cudaq::n                     |     function                      |
| oise_model_type::bit_flip_channel | )](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatorplENSt |
|     enumerator)](api/language     | 7complexIdEERK15scalar_operator), |
| s/cpp_api.html#_CPPv4N5cudaq16noi |     [\[1\                         |
| se_model_type16bit_flip_channelE) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::                      | Pv4N5cudaq15scalar_operatorplENSt |
| noise_model_type::depolarization1 | 7complexIdEERR15scalar_operator), |
|     (C++                          |     [\[2\]](api/languages/cp      |
|     enumerator)](api/languag      | p_api.html#_CPPv4N5cudaq15scalar_ |
| es/cpp_api.html#_CPPv4N5cudaq16no | operatorplEdRK15scalar_operator), |
| ise_model_type15depolarization1E) |     [\[3\]](api/languages/cp      |
| -   [cudaq::                      | p_api.html#_CPPv4N5cudaq15scalar_ |
| noise_model_type::depolarization2 | operatorplEdRR15scalar_operator), |
|     (C++                          |     [\[4\]](api/languages         |
|     enumerator)](api/languag      | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| es/cpp_api.html#_CPPv4N5cudaq16no | alar_operatorplENSt7complexIdEE), |
| ise_model_type15depolarization2E) |     [\[5\]](api/languages/cpp     |
| -   [cudaq::noise_m               | _api.html#_CPPv4NKR5cudaq15scalar |
| odel_type::depolarization_channel | _operatorplERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|   enumerator)](api/languages/cpp_ | 4NKR5cudaq15scalar_operatorplEd), |
| api.html#_CPPv4N5cudaq16noise_mod |     [\[7\]]                       |
| el_type22depolarization_channelE) | (api/languages/cpp_api.html#_CPPv |
| -                                 | 4NKR5cudaq15scalar_operatorplEv), |
|  [cudaq::noise_model_type::pauli1 |     [\[8\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|     enumerator)](a                | alar_operatorplENSt7complexIdEE), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[9\]](api/languages/cp      |
| 5cudaq16noise_model_type6pauli1E) | p_api.html#_CPPv4NO5cudaq15scalar |
| -                                 | _operatorplERK15scalar_operator), |
|  [cudaq::noise_model_type::pauli2 |     [\[10\]                       |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     enumerator)](a                | v4NO5cudaq15scalar_operatorplEd), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[11\                        |
| 5cudaq16noise_model_type6pauli2E) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq                        | Pv4NO5cudaq15scalar_operatorplEv) |
| ::noise_model_type::phase_damping | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator+= |
|     enumerator)](api/langu        |     (C++                          |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     function)](api/languag        |
| noise_model_type13phase_dampingE) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -   [cudaq::noi                   | alar_operatorpLENSt7complexIdEE), |
| se_model_type::phase_flip_channel |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     enumerator)](api/languages/   | _operatorpLERK15scalar_operator), |
| cpp_api.html#_CPPv4N5cudaq16noise |     [\[2                          |
| _model_type18phase_flip_channelE) | \]](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq15scalar_operatorpLEd) |
| [cudaq::noise_model_type::unknown | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator- |
|     enumerator)](ap               |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function                      |
| cudaq16noise_model_type7unknownE) | )](api/languages/cpp_api.html#_CP |
| -                                 | Pv4N5cudaq15scalar_operatormiENSt |
| [cudaq::noise_model_type::x_error | 7complexIdEERK15scalar_operator), |
|     (C++                          |     [\[1\                         |
|     enumerator)](ap               | ]](api/languages/cpp_api.html#_CP |
| i/languages/cpp_api.html#_CPPv4N5 | Pv4N5cudaq15scalar_operatormiENSt |
| cudaq16noise_model_type7x_errorE) | 7complexIdEERR15scalar_operator), |
| -                                 |     [\[2\]](api/languages/cp      |
| [cudaq::noise_model_type::y_error | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRK15scalar_operator), |
|     enumerator)](ap               |     [\[3\]](api/languages/cp      |
| i/languages/cpp_api.html#_CPPv4N5 | p_api.html#_CPPv4N5cudaq15scalar_ |
| cudaq16noise_model_type7y_errorE) | operatormiEdRR15scalar_operator), |
| -                                 |     [\[4\]](api/languages         |
| [cudaq::noise_model_type::z_error | /cpp_api.html#_CPPv4NKR5cudaq15sc |
|     (C++                          | alar_operatormiENSt7complexIdEE), |
|     enumerator)](ap               |     [\[5\]](api/languages/cpp     |
| i/languages/cpp_api.html#_CPPv4N5 | _api.html#_CPPv4NKR5cudaq15scalar |
| cudaq16noise_model_type7z_errorE) | _operatormiERK15scalar_operator), |
| -   [cudaq::num_available_gpus    |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     function                      | 4NKR5cudaq15scalar_operatormiEd), |
| )](api/languages/cpp_api.html#_CP |     [\[7\]]                       |
| Pv4N5cudaq18num_available_gpusEv) | (api/languages/cpp_api.html#_CPPv |
| -   [cudaq::observe (C++          | 4NKR5cudaq15scalar_operatormiEv), |
|     function)]                    |     [\[8\]](api/language          |
| (api/languages/cpp_api.html#_CPPv | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| 4I00DpEN5cudaq7observeENSt6vector | alar_operatormiENSt7complexIdEE), |
| I14observe_resultEERR13QuantumKer |     [\[9\]](api/languages/cp      |
| nelRK15SpinOpContainerDpRR4Args), | p_api.html#_CPPv4NO5cudaq15scalar |
|     [\[1\]](api/languages/cpp_ap  | _operatormiERK15scalar_operator), |
| i.html#_CPPv4I0DpEN5cudaq7observe |     [\[10\]                       |
| E14observe_resultNSt6size_tERR13Q | ](api/languages/cpp_api.html#_CPP |
| uantumKernelRK7spin_opDpRR4Args), | v4NO5cudaq15scalar_operatormiEd), |
|     [\[                           |     [\[11\                        |
| 2\]](api/languages/cpp_api.html#_ | ]](api/languages/cpp_api.html#_CP |
| CPPv4I0DpEN5cudaq7observeE14obser | Pv4NO5cudaq15scalar_operatormiEv) |
| ve_resultRK15observe_optionsRR13Q | -   [c                            |
| uantumKernelRK7spin_opDpRR4Args), | udaq::scalar_operator::operator-= |
|     [\[3\]](api/lang              |     (C++                          |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     function)](api/languag        |
| udaq7observeE14observe_resultRR13 | es/cpp_api.html#_CPPv4N5cudaq15sc |
| QuantumKernelRK7spin_opDpRR4Args) | alar_operatormIENSt7complexIdEE), |
| -   [cudaq::observe_options (C++  |     [\[1\]](api/languages/c       |
|     st                            | pp_api.html#_CPPv4N5cudaq15scalar |
| ruct)](api/languages/cpp_api.html | _operatormIERK15scalar_operator), |
| #_CPPv4N5cudaq15observe_optionsE) |     [\[2                          |
| -   [cudaq::observe_result (C++   | \]](api/languages/cpp_api.html#_C |
|                                   | PPv4N5cudaq15scalar_operatormIEd) |
| class)](api/languages/cpp_api.htm | -   [                             |
| l#_CPPv4N5cudaq14observe_resultE) | cudaq::scalar_operator::operator/ |
| -                                 |     (C++                          |
|    [cudaq::observe_result::counts |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function)](api/languages/c    | Pv4N5cudaq15scalar_operatordvENSt |
| pp_api.html#_CPPv4N5cudaq14observ | 7complexIdEERK15scalar_operator), |
| e_result6countsERK12spin_op_term) |     [\[1\                         |
| -   [cudaq::observe_result::dump  | ]](api/languages/cpp_api.html#_CP |
|     (C++                          | Pv4N5cudaq15scalar_operatordvENSt |
|     function)                     | 7complexIdEERR15scalar_operator), |
| ](api/languages/cpp_api.html#_CPP |     [\[2\]](api/languages/cp      |
| v4N5cudaq14observe_result4dumpEv) | p_api.html#_CPPv4N5cudaq15scalar_ |
| -   [c                            | operatordvEdRK15scalar_operator), |
| udaq::observe_result::expectation |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq15scalar_ |
|                                   | operatordvEdRR15scalar_operator), |
| function)](api/languages/cpp_api. |     [\[4\]](api/languages         |
| html#_CPPv4N5cudaq14observe_resul | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| t11expectationERK12spin_op_term), | alar_operatordvENSt7complexIdEE), |
|     [\[1\]](api/la                |     [\[5\]](api/languages/cpp     |
| nguages/cpp_api.html#_CPPv4N5cuda | _api.html#_CPPv4NKR5cudaq15scalar |
| q14observe_result11expectationEv) | _operatordvERK15scalar_operator), |
| -   [cuda                         |     [\[6\]]                       |
| q::observe_result::id_coefficient | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatordvEd), |
|     function)](api/langu          |     [\[7\]](api/language          |
| ages/cpp_api.html#_CPPv4N5cudaq14 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| observe_result14id_coefficientEv) | alar_operatordvENSt7complexIdEE), |
| -   [cuda                         |     [\[8\]](api/languages/cp      |
| q::observe_result::observe_result | p_api.html#_CPPv4NO5cudaq15scalar |
|     (C++                          | _operatordvERK15scalar_operator), |
|                                   |     [\[9\                         |
|   function)](api/languages/cpp_ap | ]](api/languages/cpp_api.html#_CP |
| i.html#_CPPv4N5cudaq14observe_res | Pv4NO5cudaq15scalar_operatordvEd) |
| ult14observe_resultEdRK7spin_op), | -   [c                            |
|     [\[1\]](a                     | udaq::scalar_operator::operator/= |
| pi/languages/cpp_api.html#_CPPv4N |     (C++                          |
| 5cudaq14observe_result14observe_r |     function)](api/languag        |
| esultEdRK7spin_op13sample_result) | es/cpp_api.html#_CPPv4N5cudaq15sc |
| -                                 | alar_operatordVENSt7complexIdEE), |
|  [cudaq::observe_result::operator |     [\[1\]](api/languages/c       |
|     double (C++                   | pp_api.html#_CPPv4N5cudaq15scalar |
|     functio                       | _operatordVERK15scalar_operator), |
| n)](api/languages/cpp_api.html#_C |     [\[2                          |
| PPv4N5cudaq14observe_resultcvdEv) | \]](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq15scalar_operatordVEd) |
|  [cudaq::observe_result::raw_data | -   [                             |
|     (C++                          | cudaq::scalar_operator::operator= |
|     function)](ap                 |     (C++                          |
| i/languages/cpp_api.html#_CPPv4N5 |     function)](api/languages/c    |
| cudaq14observe_result8raw_dataEv) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::operator_handler (C++ | _operatoraSERK15scalar_operator), |
|     cl                            |     [\[1\]](api/languages/        |
| ass)](api/languages/cpp_api.html# | cpp_api.html#_CPPv4N5cudaq15scala |
| _CPPv4N5cudaq16operator_handlerE) | r_operatoraSERR15scalar_operator) |
| -   [cudaq::optimizable_function  | -   [c                            |
|     (C++                          | udaq::scalar_operator::operator== |
|     class)                        |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     function)](api/languages/c    |
| v4N5cudaq20optimizable_functionE) | pp_api.html#_CPPv4NK5cudaq15scala |
| -   [cudaq::optimization_result   | r_operatoreqERK15scalar_operator) |
|     (C++                          | -   [cudaq:                       |
|     type                          | :scalar_operator::scalar_operator |
| )](api/languages/cpp_api.html#_CP |     (C++                          |
| Pv4N5cudaq19optimization_resultE) |     func                          |
| -   [cudaq::optimizer (C++        | tion)](api/languages/cpp_api.html |
|     class)](api/languages/cpp_a   | #_CPPv4N5cudaq15scalar_operator15 |
| pi.html#_CPPv4N5cudaq9optimizerE) | scalar_operatorENSt7complexIdEE), |
| -   [cudaq::optimizer::optimize   |     [\[1\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|                                   | scalar_operator15scalar_operatorE |
|  function)](api/languages/cpp_api | RK15scalar_callbackRRNSt13unorder |
| .html#_CPPv4N5cudaq9optimizer8opt | ed_mapINSt6stringENSt6stringEEE), |
| imizeEKiRR20optimizable_function) |     [\[2\                         |
| -   [cu                           | ]](api/languages/cpp_api.html#_CP |
| daq::optimizer::requiresGradients | Pv4N5cudaq15scalar_operator15scal |
|     (C++                          | ar_operatorERK15scalar_operator), |
|     function)](api/la             |     [\[3\]](api/langu             |
| nguages/cpp_api.html#_CPPv4N5cuda | ages/cpp_api.html#_CPPv4N5cudaq15 |
| q9optimizer17requiresGradientsEv) | scalar_operator15scalar_operatorE |
| -   [cudaq::orca (C++             | RR15scalar_callbackRRNSt13unorder |
|     type)](api/languages/         | ed_mapINSt6stringENSt6stringEEE), |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     [\[4\                         |
| -   [cudaq::orca::sample (C++     | ]](api/languages/cpp_api.html#_CP |
|     function)](api/languages/c    | Pv4N5cudaq15scalar_operator15scal |
| pp_api.html#_CPPv4N5cudaq4orca6sa | ar_operatorERR15scalar_operator), |
| mpleERNSt6vectorINSt6size_tEEERNS |     [\[5\]](api/language          |
| t6vectorINSt6size_tEEERNSt6vector | s/cpp_api.html#_CPPv4N5cudaq15sca |
| IdEERNSt6vectorIdEEiNSt6size_tE), | lar_operator15scalar_operatorEd), |
|     [\[1\]]                       |     [\[6\]](api/languag           |
| (api/languages/cpp_api.html#_CPPv | es/cpp_api.html#_CPPv4N5cudaq15sc |
| 4N5cudaq4orca6sampleERNSt6vectorI | alar_operator15scalar_operatorEv) |
| NSt6size_tEEERNSt6vectorINSt6size | -   [                             |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | cudaq::scalar_operator::to_matrix |
| -   [cudaq::orca::sample_async    |     (C++                          |
|     (C++                          |                                   |
|                                   |   function)](api/languages/cpp_ap |
| function)](api/languages/cpp_api. | i.html#_CPPv4NK5cudaq15scalar_ope |
| html#_CPPv4N5cudaq4orca12sample_a | rator9to_matrixERKNSt13unordered_ |
| syncERNSt6vectorINSt6size_tEEERNS | mapINSt6stringENSt7complexIdEEEE) |
| t6vectorINSt6size_tEEERNSt6vector | -   [                             |
| IdEERNSt6vectorIdEEiNSt6size_tE), | cudaq::scalar_operator::to_string |
|     [\[1\]](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/l              |
| q4orca12sample_asyncERNSt6vectorI | anguages/cpp_api.html#_CPPv4NK5cu |
| NSt6size_tEEERNSt6vectorINSt6size | daq15scalar_operator9to_stringEv) |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | -   [cudaq::s                     |
| -   [cudaq::OrcaRemoteRESTQPU     | calar_operator::\~scalar_operator |
|     (C++                          |     (C++                          |
|     cla                           |     functio                       |
| ss)](api/languages/cpp_api.html#_ | n)](api/languages/cpp_api.html#_C |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | PPv4N5cudaq15scalar_operatorD0Ev) |
| -   [cudaq::pauli1 (C++           | -   [cudaq::set_noise (C++        |
|     class)](api/languages/cp      |     function)](api/langu          |
| p_api.html#_CPPv4N5cudaq6pauli1E) | ages/cpp_api.html#_CPPv4N5cudaq9s |
| -                                 | et_noiseERKN5cudaq11noise_modelE) |
|    [cudaq::pauli1::num_parameters | -   [cudaq::set_random_seed (C++  |
|     (C++                          |     function)](api/               |
|     member)]                      | languages/cpp_api.html#_CPPv4N5cu |
| (api/languages/cpp_api.html#_CPPv | daq15set_random_seedENSt6size_tE) |
| 4N5cudaq6pauli114num_parametersE) | -   [cudaq::simulation_precision  |
| -   [cudaq::pauli1::num_targets   |     (C++                          |
|     (C++                          |     enum)                         |
|     membe                         | ](api/languages/cpp_api.html#_CPP |
| r)](api/languages/cpp_api.html#_C | v4N5cudaq20simulation_precisionE) |
| PPv4N5cudaq6pauli111num_targetsE) | -   [                             |
| -   [cudaq::pauli1::pauli1 (C++   | cudaq::simulation_precision::fp32 |
|     function)](api/languages/cpp_ |     (C++                          |
| api.html#_CPPv4N5cudaq6pauli16pau |     enumerator)](api              |
| li1ERKNSt6vectorIN5cudaq4realEEE) | /languages/cpp_api.html#_CPPv4N5c |
| -   [cudaq::pauli2 (C++           | udaq20simulation_precision4fp32E) |
|     class)](api/languages/cp      | -   [                             |
| p_api.html#_CPPv4N5cudaq6pauli2E) | cudaq::simulation_precision::fp64 |
| -                                 |     (C++                          |
|    [cudaq::pauli2::num_parameters |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)]                      | udaq20simulation_precision4fp64E) |
| (api/languages/cpp_api.html#_CPPv | -   [cudaq::SimulationState (C++  |
| 4N5cudaq6pauli214num_parametersE) |     c                             |
| -   [cudaq::pauli2::num_targets   | lass)](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq15SimulationStateE) |
|     membe                         | -   [                             |
| r)](api/languages/cpp_api.html#_C | cudaq::SimulationState::precision |
| PPv4N5cudaq6pauli211num_targetsE) |     (C++                          |
| -   [cudaq::pauli2::pauli2 (C++   |     enum)](api                    |
|     function)](api/languages/cpp_ | /languages/cpp_api.html#_CPPv4N5c |
| api.html#_CPPv4N5cudaq6pauli26pau | udaq15SimulationState9precisionE) |
| li2ERKNSt6vectorIN5cudaq4realEEE) | -   [cudaq:                       |
|                                   | :SimulationState::precision::fp32 |
|                                   |     (C++                          |
|                                   |     enumerator)](api/lang         |
|                                   | uages/cpp_api.html#_CPPv4N5cudaq1 |
|                                   | 5SimulationState9precision4fp32E) |
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
