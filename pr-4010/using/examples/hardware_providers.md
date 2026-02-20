::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../index.html){.icon .icon-home}

::: version
pr-4010
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
    -   [Using Quantum Hardware Providers](#){.current .reference
        .internal}
        -   [Amazon Braket](#amazon-braket){.reference .internal}
        -   [Anyon Technologies](#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](#infleqtion){.reference .internal}
        -   [IonQ](#ionq){.reference .internal}
        -   [IQM](#iqm){.reference .internal}
        -   [OQC](#oqc){.reference .internal}
        -   [ORCA Computing](#orca-computing){.reference .internal}
        -   [Pasqal](#pasqal){.reference .internal}
        -   [Quantinuum](#quantinuum){.reference .internal}
        -   [Quantum Circuits, Inc.](#quantum-circuits-inc){.reference
            .internal}
        -   [Quantum Machines](#quantum-machines){.reference .internal}
        -   [QuEra Computing](#quera-computing){.reference .internal}
    -   [Dynamics Examples](dynamics_examples.html){.reference
        .internal}
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
            -   [Exercise 2 - Adding additional jump operators
                [\\(L_i\\)]{.math .notranslate
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
            -   [Section 3 - Heisenberg Model with a time-varying
                magnetic
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
        -   [5. Compuare
            results](../../applications/python/qsci.html#5.-Compuare-results){.reference
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
    -   [Anderson Impurity Model ground state solver on Infleqtion's
        Sqale](../../applications/python/logical_aim_sqale.html){.reference
        .internal}
        -   [Performing logical Variational Quantum Eigensolver (VQE)
            with
            CUDA-QX](../../applications/python/logical_aim_sqale.html#Performing-logical-Variational-Quantum-Eigensolver-(VQE)-with-CUDA-QX){.reference
            .internal}
        -   [Constructing circuits in the [`[[4,2,2]]`{.docutils
            .literal .notranslate}]{.pre}
            encoding](../../applications/python/logical_aim_sqale.html#Constructing-circuits-in-the-%5B%5B4,2,2%5D%5D-encoding){.reference
            .internal}
        -   [Setting up submission and decoding
            workflow](../../applications/python/logical_aim_sqale.html#Setting-up-submission-and-decoding-workflow){.reference
            .internal}
        -   [Running a CUDA-Q noisy
            simulation](../../applications/python/logical_aim_sqale.html#Running-a-CUDA-Q-noisy-simulation){.reference
            .internal}
        -   [Running logical AIM on Infleqtion's
            hardware](../../applications/python/logical_aim_sqale.html#Running-logical-AIM-on-Infleqtion's-hardware){.reference
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
    -   [Quantum
        Transformer](../../applications/python/quantum_transformer.html){.reference
        .internal}
        -   [Installation](../../applications/python/quantum_transformer.html#Installation){.reference
            .internal}
        -   [Algorithm and
            Example](../../applications/python/quantum_transformer.html#Algorithm-and-Example){.reference
            .internal}
            -   [Creating the self-attention
                circuits](../../applications/python/quantum_transformer.html#Creating-the-self-attention-circuits){.reference
                .internal}
        -   [Usage](../../applications/python/quantum_transformer.html#Usage){.reference
            .internal}
            -   [Model
                Training](../../applications/python/quantum_transformer.html#Model-Training){.reference
                .internal}
            -   [Generating
                Molecules](../../applications/python/quantum_transformer.html#Generating-Molecules){.reference
                .internal}
            -   [Attention
                Maps](../../applications/python/quantum_transformer.html#Attention-Maps){.reference
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
            -   [The SKQD Algorithm: Matrix Construction
                Details](../../applications/python/skqd.html#The-SKQD-Algorithm:-Matrix-Construction-Details){.reference
                .internal}
        -   [Results Analysis and
            Convergence](../../applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](../../applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
        -   [GPU Acceleration for
            Postprocessing](../../applications/python/skqd.html#GPU-Acceleration-for-Postprocessing){.reference
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
            -   [Submission from
                C++](../backends/cloud/braket.html#submission-from-c){.reference
                .internal}
            -   [Submission from
                Python](../backends/cloud/braket.html#submission-from-python){.reference
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
-   Using Quantum Hardware Providers
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](../../examples/python/performance_optimizations.html "Optimizing Performance"){.btn
.btn-neutral .float-left accesskey="p"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](dynamics_examples.html "CUDA-Q Dynamics"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#using-quantum-hardware-providers .section}
# Using Quantum Hardware Providers[¶](#using-quantum-hardware-providers "Permalink to this heading"){.headerlink}

CUDA-Q contains support for using a set of hardware providers (Amazon
Braket, Infleqtion, IonQ, IQM, OQC, ORCA Computing, Quantinuum, and
QuEra Computing). For more information about executing quantum kernels
on different hardware backends, please take a look at
[[hardware]{.doc}](../backends/hardware.html){.reference .internal}.

::: {#amazon-braket .section}
[]{#amazon-braket-examples}

## Amazon Braket[¶](#amazon-braket "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on Amazon Braket's
backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq

    # NOTE: Amazon Braket credentials must be set before running this program.
    # Amazon Braket costs apply.
    cudaq.set_target("braket")

    # The default device is SV1, state vector simulator. Users may choose any of
    # the available devices by supplying its `ARN` with the `machine` parameter.
    # For example,
    # ```
    # cudaq.set_target("braket", machine="arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
    # ```


    # Create the kernel we'd like to execute
    @cudaq.kernel
    def kernel():
        qvector = cudaq.qvector(2)
        h(qvector[0])
        x.ctrl(qvector[0], qvector[1])


    # Execute and print out the results.

    # Option A:
    # By using the asynchronous `cudaq.sample_async`, the remaining
    # classical code will be executed while the job is being handled
    # by Amazon Braket.
    async_results = cudaq.sample_async(kernel)
    # ... more classical code to run ...

    async_counts = async_results.get()
    print(async_counts)

    # Option B:
    # By using the synchronous `cudaq.sample`, the execution of
    # any remaining classical code in the file will occur only
    # after the job has been returned from Amazon Braket.
    counts = cudaq.sample(kernel)
    print(counts)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target braket braket.cpp -o out.x && ./out.x
    // ```
    // This will submit the job to the Amazon Braket state vector simulator
    // (default). Alternatively, users can choose any of the available devices by
    // specifying its `ARN` with the `--braket-machine`, e.g.,
    // ```
    // nvq++ --target braket --braket-machine \
    // "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet" braket.cpp -o out.x
    // ./out.x
    // ```
    // Assumes a valid set of credentials have been set prior to execution.

    #include <cudaq.h>
    #include <fstream>

    // Define a simple quantum kernel to execute on Amazon Braket.
    struct ghz {
      // Maximally entangled state between 5 qubits.
      auto operator()() __qpu__ {
        cudaq::qvector q(5);
        h(q[0]);
        for (int i = 0; i < 4; i++) {
          x<cudaq::ctrl>(q[i], q[i + 1]);
        }
        mz(q);
      }
    };

    int main() {
      // Submit asynchronously (e.g., continue executing
      // code in the file until the job has been returned).
      auto future = cudaq::sample_async(ghz{});
      // ... classical code to execute in the meantime ...

      // Get the results of the read in future.
      auto async_counts = future.get();
      async_counts.dump();

      // OR: Submit synchronously (e.g., wait for the job
      // result to be returned before proceeding).
      auto counts = cudaq::sample(ghz{});
      counts.dump();
    }
:::
:::
:::
:::
:::

::: {#anyon-technologies .section}
[]{#anyon-examples}

## Anyon Technologies[¶](#anyon-technologies "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on Anyon's backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq

    # You only have to set the target once! No need to redefine it
    # for every execution call on your kernel.
    # To use different targets in the same file, you must update
    # it via another call to `cudaq.set_target()`

    # To use the Anyon target you will need to set up credentials in `~/.anyon_config`
    # The configuration file should contain your Anyon Technologies username and password:
    # credentials: {"username":"<username>","password":"<password>"}

    # Set the target to the default QPU
    cudaq.set_target("anyon")

    # You can specify a specific machine via the machine parameter:
    # ```
    # cudaq.set_target("anyon", machine="telegraph-8q")
    # ```
    # or for the larger system:
    # ```
    # cudaq.set_target("anyon", machine="berkeley-25q")
    # ```


    # Create the kernel we'd like to execute on Anyon.
    @cudaq.kernel
    def ghz():
        """Maximally entangled state between 5 qubits."""
        q = cudaq.qvector(5)
        h(q[0])
        for i in range(4):
            x.ctrl(q[i], q[i + 1])
        return mz(q)


    # Execute on Anyon and print out the results.

    # Option A (recommended):
    # By using the asynchronous `cudaq.sample_async`, the remaining
    # classical code will be executed while the job is being handled
    # remotely on Anyon's superconducting QPU. This is ideal for
    # longer running jobs.
    future = cudaq.sample_async(ghz)
    # ... classical optimization code can run while job executes ...

    # Can write the future to file:
    with open("future.txt", "w") as outfile:
        print(future, file=outfile)

    # Then come back and read it in later.
    with open("future.txt", "r") as infile:
        restored_future = cudaq.AsyncSampleResult(infile.read())

    # Get the results of the restored future.
    async_counts = restored_future.get()
    print("Asynchronous results:")
    async_counts.dump()

    # Option B:
    # By using the synchronous `cudaq.sample`, the kernel
    # will be executed on Anyon and the calling thread will be blocked
    # until the results are returned.
    counts = cudaq.sample(ghz)
    print("\nSynchronous results:")
    counts.dump()
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target anyon anyon.cpp -o out.x && ./out.x
    // ```
    // This will submit the job to Anyon's default superconducting QPU.
    // You can specify a specific machine via the `--anyon-machine` flag:
    // ```
    // nvq++ --target anyon --anyon-machine telegraph-8q anyon.cpp -o out.x &&
    // ./out.x
    // ```
    // or for the larger system:
    // ```
    // nvq++ --target anyon --anyon-machine berkeley-25q anyon.cpp -o out.x &&
    // ./out.x
    // ```
    //
    // To use this target you will need to set up credentials in `~/.anyon_config`
    // The configuration file should contain your Anyon Technologies username and
    // password:
    // ```
    // credential:<username>:<password>
    // ```

    #include <cudaq.h>
    #include <fstream>

    // Define a quantum kernel to execute on Anyon backend.
    struct ghz {
      // Maximally entangled state between 5 qubits.
      auto operator()() __qpu__ {
        cudaq::qvector q(5);
        h(q[0]);
        for (int i = 0; i < 4; i++) {
          x<cudaq::ctrl>(q[i], q[i + 1]);
        }
        mz(q);
      }
    };

    int main() {

      // Submit asynchronously
      auto future = cudaq::sample_async(ghz{});

      // ... classical optimization code can run while job executes ...

      // Can write the future to file:
      {
        std::ofstream out("saveMe.json");
        out << future;
      }

      // Then come back and read it in later.
      cudaq::async_result<cudaq::sample_result> readIn;
      std::ifstream in("saveMe.json");
      in >> readIn;

      // Get the results of the read in future.
      auto async_counts = readIn.get();
      async_counts.dump();

      // OR: Submit to synchronously
      auto counts = cudaq::sample(ghz{});
      counts.dump();

      return 0;
    }
:::
:::
:::
:::
:::

::: {#infleqtion .section}
[]{#infleqtion-examples}

## Infleqtion[¶](#infleqtion "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on Infleqtion's
backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq

    # You only have to set the target once! No need to redefine it
    # for every execution call on your kernel.
    # To use different targets in the same file, you must update
    # it via another call to `cudaq.set_target()`
    cudaq.set_target("infleqtion")


    # Create the kernel we'd like to execute on Infleqtion.
    @cudaq.kernel
    def kernel():
        qvector = cudaq.qvector(2)
        h(qvector[0])
        x.ctrl(qvector[0], qvector[1])


    # Note: All measurements must be terminal when performing the sampling.

    # Execute on Infleqtion and print out the results.

    # Option A (recommended):
    # By using the asynchronous `cudaq.sample_async`, the remaining
    # classical code will be executed while the job is being handled
    # by the Superstaq API. This is ideal when submitting via a queue
    # over the cloud.
    async_results = cudaq.sample_async(kernel)
    # ... more classical code to run ...

    # We can either retrieve the results later in the program with
    # ```
    # async_counts = async_results.get()
    # ```
    # or we can also write the job reference (`async_results`) to
    # a file and load it later or from a different process.
    file = open("future.txt", "w")
    file.write(str(async_results))
    file.close()

    # We can later read the file content and retrieve the job
    # information and results.
    same_file = open("future.txt", "r")
    retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

    counts = retrieved_async_results.get()
    print(counts)

    # Option B:
    # By using the synchronous `cudaq.sample`, the execution of
    # any remaining classical code in the file will occur only
    # after the job has been returned from Superstaq.
    counts = cudaq.sample(kernel)
    print(counts)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target infleqtion infleqtion.cpp -o out.x && ./out.x
    // ```
    // This will submit the job to the ideal simulator for Infleqtion,
    // `cq_sqale_simulator` (default). Alternatively, we can enable hardware noise
    // model simulation by specifying `noise-sim` to the flag `--infleqtion-method`,
    // e.g.,
    // ```
    // nvq++ --target infleqtion --infleqtion-machine cq_sqale_qpu
    // --infleqtion-method noise-sim infleqtion.cpp -o out.x && ./out.x
    // ```
    // where "noise-sim" instructs Superstaq to perform a noisy emulation of the
    // QPU. An ideal dry-run execution on the QPU may be performed by passing
    // `dry-run` to the `--infleqtion-method` flag, e.g.,
    // ```
    // nvq++ --target infleqtion --infleqtion-machine cq_sqale_qpu
    // --infleqtion-method dry-run infleqtion.cpp -o out.x && ./out.x
    // ```
    // Note: If targeting ideal cloud simulation,
    // `--infleqtion-machine cq_sqale_simulator` is optional since it is the
    // default configuration if not provided.

    #include <cudaq.h>
    #include <fstream>

    // Define a simple quantum kernel to execute on Infleqtion backends.
    struct ghz {
      // Maximally entangled state between 5 qubits.
      auto operator()() __qpu__ {
        cudaq::qvector q(5);
        h(q[0]);
        for (int i = 0; i < 4; i++) {
          x<cudaq::ctrl>(q[i], q[i + 1]);
        }
        mz(q);
      }
    };

    int main() {
      // Submit to Infleqtion asynchronously (e.g., continue executing
      // code in the file until the job has been returned).
      auto future = cudaq::sample_async(ghz{});
      // ... classical code to execute in the meantime ...

      // Can write the future to file:
      {
        std::ofstream out("saveMe.json");
        out << future;
      }

      // Then come back and read it in later.
      cudaq::async_result<cudaq::sample_result> readIn;
      std::ifstream in("saveMe.json");
      in >> readIn;

      // Get the results of the read in future.
      auto async_counts = readIn.get();
      async_counts.dump();

      // OR: Submit to Infleqtion synchronously (e.g., wait for the job
      // result to be returned before proceeding).
      auto counts = cudaq::sample(ghz{});
      counts.dump();
    }
:::
:::
:::
:::
:::

::: {#ionq .section}
[]{#ionq-examples}

## IonQ[¶](#ionq "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on IonQ's backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq

    # You only have to set the target once! No need to redefine it
    # for every execution call on your kernel.
    # To use different targets in the same file, you must update
    # it via another call to `cudaq.set_target()`
    cudaq.set_target("ionq")


    # Create the kernel we'd like to execute on IonQ.
    @cudaq.kernel
    def kernel():
        qvector = cudaq.qvector(2)
        h(qvector[0])
        x.ctrl(qvector[0], qvector[1])


    # Note: All qubits will be measured at the end upon performing
    # the sampling. You may encounter a pre-flight error on IonQ
    # backends if you include explicit measurements.

    # Execute on IonQ and print out the results.

    # Option A:
    # By using the asynchronous `cudaq.sample_async`, the remaining
    # classical code will be executed while the job is being handled
    # by IonQ. This is ideal when submitting via a queue over
    # the cloud.
    async_results = cudaq.sample_async(kernel)
    # ... more classical code to run ...

    # We can either retrieve the results later in the program with
    # ```
    # async_counts = async_results.get()
    # ```
    # or we can also write the job reference (`async_results`) to
    # a file and load it later or from a different process.
    file = open("future.txt", "w")
    file.write(str(async_results))
    file.close()

    # We can later read the file content and retrieve the job
    # information and results.
    same_file = open("future.txt", "r")
    retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

    counts = retrieved_async_results.get()
    print(counts)

    # Option B:
    # By using the synchronous `cudaq.sample`, the execution of
    # any remaining classical code in the file will occur only
    # after the job has been returned from IonQ.
    counts = cudaq.sample(kernel)
    print(counts)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target ionq ionq.cpp -o out.x && ./out.x
    // ```
    // This will submit the job to the IonQ ideal simulator target (default).
    // Alternatively, we can enable hardware noise model simulation by specifying
    // the `--ionq-noise-model`, e.g.,
    // ```
    // nvq++ --target ionq --ionq-machine simulator --ionq-noise-model aria-1
    // ionq.cpp -o out.x && ./out.x
    // ```
    // where we set the noise model to mimic the 'aria-1' hardware device.
    // Please refer to your IonQ Cloud dashboard for the list of simulator noise
    // models.
    // Note: `--ionq-machine simulator` is  optional since 'simulator' is the
    // default configuration if not provided. Assumes a valid set of credentials
    // have been stored.

    #include <cudaq.h>
    #include <fstream>

    // Define a simple quantum kernel to execute on IonQ.
    struct ghz {
      // Maximally entangled state between 5 qubits.
      auto operator()() __qpu__ {
        cudaq::qvector q(5);
        h(q[0]);
        for (int i = 0; i < 4; i++) {
          x<cudaq::ctrl>(q[i], q[i + 1]);
        }
        mz(q);
      }
    };

    int main() {
      // Submit to IonQ asynchronously (e.g., continue executing
      // code in the file until the job has been returned).
      auto future = cudaq::sample_async(ghz{});
      // ... classical code to execute in the meantime ...

      // Can write the future to file:
      {
        std::ofstream out("saveMe.json");
        out << future;
      }

      // Then come back and read it in later.
      cudaq::async_result<cudaq::sample_result> readIn;
      std::ifstream in("saveMe.json");
      in >> readIn;

      // Get the results of the read in future.
      auto async_counts = readIn.get();
      async_counts.dump();

      // OR: Submit to IonQ synchronously (e.g., wait for the job
      // result to be returned before proceeding).
      auto counts = cudaq::sample(ghz{});
      counts.dump();
    }
:::
:::
:::
:::
:::

::: {#iqm .section}
[]{#iqm-examples}

## IQM[¶](#iqm "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on IQM's backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq

    # You only have to set the target once! No need to redefine it
    # for every execution call on your kernel.
    # To use different targets in the same file, you must update
    # it via another call to `cudaq.set_target()`
    cudaq.set_target("iqm", url="http://localhost/")

    # Crystal_5 QPU architecture:
    #       QB1
    #        |
    # QB2 - QB3 - QB4
    #        |
    #       QB5


    # Create the kernel we'd like to execute on IQM.
    @cudaq.kernel
    def kernel():
        qvector = cudaq.qvector(5)
        h(qvector[2])  # QB3
        x.ctrl(qvector[2], qvector[0])
        mz(qvector)


    # Execute on IQM Server and print out the results.

    # Option A:
    # By using the asynchronous `cudaq.sample_async`, the remaining
    # classical code will be executed while the job is being handled
    # by IQM Server. This is ideal when submitting via a queue over
    # the cloud.
    async_results = cudaq.sample_async(kernel)
    # ... more classical code to run ...

    # We can either retrieve the results later in the program with
    # ```
    # async_counts = async_results.get()
    # ```
    # or we can also write the job reference (`async_results`) to
    # a file and load it later or from a different process.
    file = open("future.txt", "w")
    file.write(str(async_results))
    file.close()

    # We can later read the file content and retrieve the job
    # information and results.
    same_file = open("future.txt", "r")
    retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

    counts = retrieved_async_results.get()
    print(counts)

    # Option B:
    # By using the synchronous `cudaq.sample`, the execution of
    # any remaining classical code in the file will occur only
    # after the job has been returned from IQM Server.
    counts = cudaq.sample(kernel)
    print(counts)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target iqm iqm.cpp -o out.x && ./out.x
    // ```
    // Assumes a valid set of credentials have been stored.

    #include <cudaq.h>
    #include <fstream>

    // Define a simple quantum kernel to execute on IQM Server.
    struct crystal_5_ghz {
      // Maximally entangled state between 5 qubits on Crystal_5 QPU.
      //       QB1
      //        |
      // QB2 - QB3 - QB4
      //        |
      //       QB5

      void operator()() __qpu__ {
        cudaq::qvector q(5);
        h(q[0]);

        // Note that the CUDA-Q compiler will automatically generate the
        // necessary instructions to swap qubits to satisfy the required
        // connectivity constraints for the Crystal_5 QPU. In this program, that
        // means that despite QB1 not being physically connected to QB2, the user
        // can still perform joint operations q[0] and q[1] because the compiler
        // will automatically (and transparently) inject the necessary swap
        // instructions to execute the user's program without the user having to
        // worry about the physical constraints.
        for (int i = 0; i < 4; i++) {
          x<cudaq::ctrl>(q[i], q[i + 1]);
        }
        mz(q);
      }
    };

    int main() {
      // Submit to IQM Server asynchronously. E.g, continue executing
      // code in the file until the job has been returned.
      auto future = cudaq::sample_async(crystal_5_ghz{});
      // ... classical code to execute in the meantime ...

      // Can write the future to file:
      {
        std::ofstream out("saveMe.json");
        out << future;
      }

      // Then come back and read it in later.
      cudaq::async_result<cudaq::sample_result> readIn;
      std::ifstream in("saveMe.json");
      in >> readIn;

      // Get the results of the read in future.
      auto async_counts = readIn.get();
      async_counts.dump();

      // OR: Submit to IQM Server synchronously. E.g, wait for the job
      // result to be returned before proceeding.
      auto counts = cudaq::sample(crystal_5_ghz{});
      counts.dump();
    }
:::
:::
:::
:::
:::

::: {#oqc .section}
[]{#oqc-examples}

## OQC[¶](#oqc "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on OQC's backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq

    # You only have to set the target once! No need to redefine it
    # for every execution call on your kernel.
    # To use different targets in the same file, you must update
    # it via another call to `cudaq.set_target()`

    # To use the OQC target you will need to set the following environment variables
    # OQC_URL
    # OQC_EMAIL
    # OQC_PASSWORD
    # To setup an account, contact oqc_qcaas_support@oxfordquantumcircuits.com

    cudaq.set_target("oqc")


    # Create the kernel we'd like to execute on OQC.
    @cudaq.kernel
    def kernel():
        qvector = cudaq.qvector(2)
        h(qvector[0])
        x.ctrl(qvector[0], qvector[1])
        mz(qvector)


    # Option A:
    # By using the asynchronous `cudaq.sample_async`, the remaining
    # classical code will be executed while the job is being handled
    # by OQC. This is ideal when submitting via a queue over
    # the cloud.
    async_results = cudaq.sample_async(kernel)
    # ... more classical code to run ...

    # We can either retrieve the results later in the program with
    # ```
    # async_counts = async_results.get()
    # ```
    # or we can also write the job reference (`async_results`) to
    # a file and load it later or from a different process.
    file = open("future.txt", "w")
    file.write(str(async_results))
    file.close()

    # We can later read the file content and retrieve the job
    # information and results.
    same_file = open("future.txt", "r")
    retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

    counts = retrieved_async_results.get()
    print(counts)

    # Option B:
    # By using the synchronous `cudaq.sample`, the execution of
    # any remaining classical code in the file will occur only
    # after the job has been returned from OQC.
    counts = cudaq.sample(kernel)
    print(counts)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target oqc oqc.cpp -o out.x && ./out.x
    // ```
    // This will submit the job to the OQC platform. You can also specify
    // the machine to use via the `--oqc-machine` flag:
    // ```
    // nvq++ --target oqc --oqc-machine lucy oqc.cpp -o out.x && ./out.x
    // ```
    // The default is the 8 qubit Lucy device. You can set this to be either
    // `toshiko` or `lucy` via this flag.
    //
    // To use the OQC target you will need to set the following environment
    // variables: OQC_URL OQC_EMAIL OQC_PASSWORD To setup an account, contact
    // oqc_qcaas_support@oxfordquantumcircuits.com

    #include <cudaq.h>
    #include <fstream>

    // Define a simple quantum kernel to execute on OQC backends.
    struct bell_state {
      auto operator()() __qpu__ {
        cudaq::qvector q(2);
        h(q[0]);
        x<cudaq::ctrl>(q[0], q[1]);
      }
    };

    int main() {
      // Submit to OQC asynchronously (e.g., continue executing
      // code in the file until the job has been returned).
      auto future = cudaq::sample_async(bell_state{});
      // ... classical code to execute in the meantime ...

      // Can write the future to file:
      {
        std::ofstream out("future.json");
        out << future;
      }

      // Then come back and read it in later.
      cudaq::async_result<cudaq::sample_result> readIn;
      std::ifstream in("future.json");
      in >> readIn;

      // Get the results of the read in future.
      auto async_counts = readIn.get();
      async_counts.dump();

      // OR: Submit to OQC synchronously (e.g., wait for the job
      // result to be returned before proceeding).
      auto counts = cudaq::sample(bell_state{});
      counts.dump();
    }
:::
:::
:::
:::
:::

::: {#orca-computing .section}
[]{#orca-examples}

## ORCA Computing[¶](#orca-computing "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on ORCA Computing's
backends.

ORCA Computing's PT Series implement the boson sampling model of quantum
computation, in which multiple photons are interfered with each other
within a network of beam splitters, and photon detectors measure where
the photons leave this network.

The following image shows the schematic of a Time Bin Interferometer
(TBI) boson sampling experiment that runs on ORCA Computing's backends.
A TBI uses optical delay lines with reconfigurable coupling parameters.
A TBI can be represented by a circuit diagram, like the one below, where
this illustration example corresponds to 4 photons in 8 modes sent into
alternating time-bins in a circuit composed of two delay lines in
series.

[![](../../_images/orca_tbi.png){.align-center
style="width: 400px;"}](../../_images/orca_tbi.png){.reference .internal
.image-reference}

The parameters needed to define the time bin interferometer are the the
input state, the loop lengths, beam splitter angles, and optionally the
phase shifter angles, and the number of samples. The *input state* is
the initial state of the photons in the time bin interferometer, the
left-most entry corresponds to the first mode entering the loop. The
*loop lengths* are the lengths of the different loops in the time bin
interferometer. The *beam splitter angles* and the phase shifter angles
are controllable parameters of the time bin interferometer.

This experiment is performed on ORCA's backends by the code below.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    import time

    import numpy as np
    import os
    # You only have to set the target once! No need to redefine it
    # for every execution call on your kernel.
    # To use different targets in the same file, you must update
    # it via another call to `cudaq.set_target()`

    # To use the ORCA Computing target you will need to set the ORCA_ACCESS_URL
    # environment variable or pass a URL.
    orca_url = os.getenv("ORCA_ACCESS_URL", "http://localhost/sample")

    cudaq.set_target("orca", url=orca_url)

    # A time-bin boson sampling experiment: An input state of 4 indistinguishable
    # photons mixed with 4 vacuum states across 8 time bins (modes) enter the
    # time bin interferometer (TBI). The interferometer is composed of two loops
    # each with a beam splitter (and optionally with a corresponding phase
    # shifter). Each photon can either be stored in a loop to interfere with the
    # next photon or exit the loop to be measured. Since there are 8 time bins
    # and 2 loops, there is a total of 14 beam splitters (and optionally 14 phase
    # shifters) in the interferometer, which is the number of controllable
    # parameters.

    # half of 8 time bins is filled with a single photon and the other half is
    # filled with the vacuum state (empty)
    input_state = [1, 0, 1, 0, 1, 0, 1, 0]

    # The time bin interferometer in this example has two loops, each of length 1
    loop_lengths = [1, 1]

    # Calculate the number of beam splitters and phase shifters
    n_beam_splitters = len(loop_lengths) * len(input_state) - sum(loop_lengths)

    # beam splitter angles
    bs_angles = np.linspace(np.pi / 3, np.pi / 6, n_beam_splitters)

    # Optionally, we can also specify the phase shifter angles, if the system
    # includes phase shifters
    # ```
    # ps_angles = np.linspace(np.pi / 3, np.pi / 5, n_beam_splitters)
    # ```

    # we can also set number of requested samples
    n_samples = 10000

    # Option A:
    # By using the synchronous `cudaq.orca.sample`, the execution of
    # any remaining classical code in the file will occur only
    # after the job has been returned from ORCA Server.
    print("Submitting to ORCA Server synchronously")
    counts = cudaq.orca.sample(input_state, loop_lengths, bs_angles, n_samples)

    # If the system includes phase shifters, the phase shifter angles can be
    # included in the call
    # ```
    # counts = cudaq.orca.sample(input_state, loop_lengths, bs_angles, ps_angles,
    #                            n_samples)
    # ```

    # Print the results
    print(counts)

    # Option B:
    # By using the asynchronous `cudaq.orca.sample_async`, the remaining
    # classical code will be executed while the job is being handled
    # by Orca. This is ideal when submitting via a queue over
    # the cloud.
    print("Submitting to ORCA Server asynchronously")
    async_results = cudaq.orca.sample_async(input_state, loop_lengths, bs_angles,
                                            n_samples)
    # ... more classical code to run ...

    # We can either retrieve the results later in the program with
    # ```
    # async_counts = async_results.get()
    # ```
    # or we can also write the job reference (`async_results`) to
    # a file and load it later or from a different process.
    file = open("future.txt", "w")
    file.write(str(async_results))
    file.close()

    # We can later read the file content and retrieve the job
    # information and results.
    time.sleep(0.2)  # wait for the job to be processed
    same_file = open("future.txt", "r")
    retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

    counts = retrieved_async_results.get()
    print(counts)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target orca --orca-url $ORCA_ACCESS_URL orca.cpp -o out.x && ./out.x
    // ```
    // To use the ORCA Computing target you will need to set the ORCA_ACCESS_URL
    // environment variable or pass the URL to the `--orca-url` flag.

    #include <chrono>
    #include <cudaq.h>
    #include <cudaq/orca.h>
    #include <fstream>
    #include <iostream>
    #include <thread>

    int main() {
      using namespace std::this_thread;     // sleep_for, sleep_until
      using namespace std::chrono_literals; // `ns`, `us`, `ms`, `s`, `h`, etc.

      // A time-bin boson sampling experiment: An input state of 4 indistinguishable
      // photons mixed with 4 vacuum states across 8 time bins (modes) enter the
      // time bin interferometer (TBI). The interferometer is composed of two loops
      // each with a beam splitter (and optionally with a corresponding phase
      // shifter). Each photon can either be stored in a loop to interfere with the
      // next photon or exit the loop to be measured. Since there are 8 time bins
      // and 2 loops, there is a total of 14 beam splitters (and optionally 14 phase
      // shifters) in the interferometer, which is the number of controllable
      // parameters.

      // half of 8 time bins is filled with a single photon and the other half is
      // filled with the vacuum state (empty)
      std::vector<std::size_t> input_state = {1, 0, 1, 0, 1, 0, 1, 0};

      // The time bin interferometer in this example has two loops, each of length 1
      std::vector<std::size_t> loop_lengths = {1, 1};

      // helper variables to calculate the number of beam splitters and phase
      // shifters needed in the TBI
      std::size_t sum_loop_lengths{std::accumulate(
          loop_lengths.begin(), loop_lengths.end(), static_cast<std::size_t>(0))};
      const std::size_t n_loops = loop_lengths.size();
      const std::size_t n_modes = input_state.size();
      const std::size_t n_beam_splitters = n_loops * n_modes - sum_loop_lengths;

      // beam splitter angles (created as a linear spaced vector of angles)
      std::vector<double> bs_angles =
          cudaq::linspace(M_PI / 3, M_PI / 6, n_beam_splitters);

      // Optionally, we can also specify the phase shifter angles (created as a
      // linear spaced vector of angles), if the system includes phase shifters
      // ```
      // std::vector<double> ps_angles = cudaq::linspace(M_PI / 3, M_PI / 5,
      // n_beam_splitters);
      // ```

      // we can also set number of requested samples
      int n_samples = 10000;

      // Submit to ORCA synchronously (e.g., wait for the job result to be
      // returned before proceeding with the rest of the execution).
      std::cout << "Submitting to ORCA Server synchronously" << std::endl;
      auto counts =
          cudaq::orca::sample(input_state, loop_lengths, bs_angles, n_samples);

      // Print the results
      counts.dump();

      // If the system includes phase shifters, the phase shifter angles can be
      // included in the call

      // ```
      // auto counts = cudaq::orca::sample(input_state, loop_lengths, bs_angles,
      //                                   ps_angles, n_samples);
      // ```

      // Alternatively we can submit to ORCA asynchronously (e.g., continue
      // executing code in the file until the job has been returned).
      std::cout << "Submitting to ORCA Server asynchronously" << std::endl;
      auto async_results = cudaq::orca::sample_async(input_state, loop_lengths,
                                                     bs_angles, n_samples);

      // Can write the future to file:
      {
        std::ofstream out("saveMe.json");
        out << async_results;
      }

      // Then come back and read it in later.
      cudaq::async_result<cudaq::sample_result> readIn;
      std::ifstream in("saveMe.json");
      in >> readIn;

      sleep_for(200ms); // wait for the job to be processed
      // Get the results of the read in future.
      auto async_counts = readIn.get();
      async_counts.dump();

      return 0;
    }
:::
:::
:::
:::
:::

::: {#pasqal .section}
[]{#pasqal-examples}

## Pasqal[¶](#pasqal "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on Pasqal's backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    from cudaq.operators import RydbergHamiltonian, ScalarOperator
    from cudaq.dynamics import Schedule

    # This example illustrates how to use Pasqal's EMU_MPS emulator over Pasqal's cloud via CUDA-Q.
    #
    # To obtain the authentication token for the cloud  we recommend logging in with
    # Pasqal's Python SDK. See our documentation https://docs.pasqal.com/cloud/ for more.
    #
    # Contact Pasqal at help@pasqal.com or through https://community.pasqal.com for assistance.
    #
    # Visit the documentation portal, https://docs.pasqal.com/, to find further
    # documentation on Pasqal's devices, emulators and the cloud platform.
    #
    # For more details on the EMU_MPS emulator see the documentation of the open-source
    # package: https://pasqal-io.github.io/emulators/latest/emu_mps/.
    from pasqal_cloud import SDK
    import os

    # We recommend leaving the password empty in an interactive session as you will be
    # prompted to enter it securely via the command line interface.
    sdk = SDK(
        username=os.environ.get("PASQAL_USERNAME"),
        password=os.environ.get("PASQAL_PASSWORD", None),
    )

    os.environ["PASQAL_AUTH_TOKEN"] = str(sdk.user_token())

    # It is also mandatory to specify the project against which the execution will be billed.
    # Uncomment this line to set it from Python, or export it as an environment variable
    # prior to execution. You can find your projects here: https://portal.pasqal.cloud/projects.
    # ```
    # os.environ['PASQAL_PROJECT_ID'] = 'your project id'
    # ```

    # Set the target including specifying optional arguments like target machine
    cudaq.set_target("pasqal",
                     machine=os.environ.get("PASQAL_MACHINE_TARGET", "EMU_MPS"))

    # ```
    ## To target QPU set FRESNEL as the machine, see our cloud portal for latest machine names
    # cudaq.set_target("pasqal", machine="FRESNEL")
    # ```

    # Define the 2-dimensional atom arrangement
    a = 5e-6
    register = [(a, 0), (2 * a, 0), (3 * a, 0)]
    time_ramp = 0.000001
    time_max = 0.000003
    # Times for the piece-wise linear waveforms
    steps = [0.0, time_ramp, time_max - time_ramp, time_max]
    schedule = Schedule(steps, ["t"])
    # Rabi frequencies at each step
    omega_max = 1000000
    delta_end = 1000000
    delta_start = 0.0
    omega = ScalarOperator(lambda t: omega_max
                           if time_ramp < t.real < time_max else 0.0)
    # Global phase at each step
    phi = ScalarOperator.const(0.0)
    # Global detuning at each step
    delta = ScalarOperator(lambda t: delta_end
                           if time_ramp < t.real < time_max else delta_start)

    async_result = cudaq.evolve_async(RydbergHamiltonian(atom_sites=register,
                                                         amplitude=omega,
                                                         phase=phi,
                                                         delta_global=delta),
                                      schedule=schedule,
                                      shots_count=100).get()
    async_result.dump()

    ## Sample result
    # ```
    # {'001': 16, '010': 23, '100': 19, '000': 42}
    # ```
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target pasqal pasqal.cpp -o out.x
    // ./out.x
    // ```
    // Assumes a valid set of credentials (`PASQAL_AUTH_TOKEN`, `PASQAL_PROJECT_ID`)
    // have been set.

    #include "cudaq/algorithms/evolve.h"
    #include "cudaq/algorithms/integrator.h"
    #include "cudaq/operators.h"
    #include "cudaq/schedule.h"
    #include <cmath>
    #include <map>
    #include <vector>

    // This example illustrates how to use `Pasqal's` EMU_MPS emulator over
    // `Pasqal's` cloud via CUDA-Q. Contact Pasqal at help@pasqal.com or through
    // https://community.pasqal.com for assistance.

    int main() {
      // Topology initialization
      const double a = 5e-6;
      std::vector<std::pair<double, double>> register_sites;
      register_sites.push_back(std::make_pair(a, 0.0));
      register_sites.push_back(std::make_pair(2 * a, 0.0));
      register_sites.push_back(std::make_pair(3 * a, 0.0));

      // Simulation Timing
      const double time_ramp = 0.000001; // seconds
      const double time_max = 0.000003;  // seconds
      const double omega_max = 1000000;  // rad/sec
      const double delta_end = 1000000;
      const double delta_start = 0.0;

      std::vector<std::complex<double>> steps = {0.0, time_ramp,
                                                 time_max - time_ramp, time_max};
      cudaq::schedule schedule(steps, {"t"}, {});

      // Basic Rydberg Hamiltonian
      auto omega = cudaq::scalar_operator(
          [time_ramp, time_max,
           omega_max](const std::unordered_map<std::string, std::complex<double>>
                          &parameters) {
            double t = std::real(parameters.at("t"));
            return std::complex<double>(
                (t > time_ramp && t < time_max) ? omega_max : 0.0, 0.0);
          });

      auto phi = cudaq::scalar_operator(0.0);

      auto delta = cudaq::scalar_operator(
          [time_ramp, time_max, delta_start,
           delta_end](const std::unordered_map<std::string, std::complex<double>>
                          &parameters) {
            double t = std::real(parameters.at("t"));
            return std::complex<double>(
                (t > time_ramp && t < time_max) ? delta_end : delta_start, 0.0);
          });

      auto hamiltonian =
          cudaq::rydberg_hamiltonian(register_sites, omega, phi, delta);

      // Evolve the system
      auto result = cudaq::evolve(hamiltonian, schedule, 100);
      result.sampling_result->dump();

      return 0;
    }
:::
:::
:::
:::
:::

::: {#quantinuum .section}
[]{#quantinuum-examples}

## Quantinuum[¶](#quantinuum "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on Quantinuum's
backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    import os

    # You only have to set the target once! No need to redefine it for every
    # execution call on your kernel.
    # By default, we will submit to the Quantinuum syntax checker.
    ## NOTE: It is mandatory to specify the Nexus project by name or ID.
    # Update and un-comment the line below.
    # ```
    # cudaq.set_target("quantinuum", project="nexus_project")
    # ```
    # Or use environment variable
    # ```
    # os.environ["QUANTINUUM_NEXUS_PROJECT"] = "nexus_project"
    # ```
    cudaq.set_target("quantinuum",
                     project=os.environ.get("QUANTINUUM_NEXUS_PROJECT", None))


    # Create the kernel we'd like to execute on Quantinuum.
    @cudaq.kernel
    def kernel():
        qvector = cudaq.qvector(2)
        h(qvector[0])
        x.ctrl(qvector[0], qvector[1])


    # Submit to Quantinuum's endpoint and confirm the program is valid.

    # Option A:
    # By using the synchronous `cudaq.sample`, the execution of
    # any remaining classical code in the file will occur only
    # after the job has been executed by the Quantinuum service.
    # We will use the synchronous call to submit to the syntax
    # checker to confirm the validity of the program.
    syntax_check = cudaq.sample(kernel)
    if (syntax_check):
        print("Syntax check passed! Kernel is ready for submission.")

    # Now we can update the target to the Quantinuum emulator and
    # execute our program.
    cudaq.set_target("quantinuum",
                     machine="H2-1E",
                     project=os.environ.get("QUANTINUUM_NEXUS_PROJECT", None))

    # Option B:
    # By using the asynchronous `cudaq.sample_async`, the remaining
    # classical code will be executed while the job is being handled
    # by Quantinuum. This is ideal when submitting via a queue over
    # the cloud.
    async_results = cudaq.sample_async(kernel)
    # ... more classical code to run ...

    # We can either retrieve the results later in the program with
    # ```
    # async_counts = async_results.get()
    # ```
    # or we can also write the job reference (`async_results`) to
    # a file and load it later or from a different process.
    file = open("future.txt", "w")
    file.write(str(async_results))
    file.close()

    # We can later read the file content and retrieve the job
    # information and results.
    same_file = open("future.txt", "r")
    retrieved_async_results = cudaq.AsyncSampleResult(str(same_file.read()))

    counts = retrieved_async_results.get()
    print(counts)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target quantinuum --quantinuum-machine H2-1E --quantinuum-project \
    // <nexus_project> quantinuum.cpp  -o out.x && ./out.x
    // ```
    // Assumes a valid set of credentials have been stored.
    // To first confirm the correctness of the program locally,
    // Add a --emulate to the `nvq++` command above.

    #include <cudaq.h>
    #include <fstream>

    // Define a simple quantum kernel to execute on Quantinuum.
    struct ghz {
      // Maximally entangled state between 5 qubits.
      auto operator()() __qpu__ {
        cudaq::qvector q(5);
        h(q[0]);
        for (int i = 0; i < 4; i++) {
          x<cudaq::ctrl>(q[i], q[i + 1]);
        }
      }
    };

    int main() {
      // Submit to Quantinuum asynchronously (e.g., continue executing
      // code in the file until the job has been returned).
      auto future = cudaq::sample_async(ghz{});
      // ... classical code to execute in the meantime ...

      // Can write the future to file:
      {
        std::ofstream out("saveMe.json");
        out << future;
      }

      // Then come back and read it in later.
      cudaq::async_result<cudaq::sample_result> readIn;
      std::ifstream in("saveMe.json");
      in >> readIn;

      // Get the results of the read in future.
      auto async_counts = readIn.get();
      async_counts.dump();

      // OR: Submit to Quantinuum synchronously (e.g., wait for the job
      // result to be returned before proceeding).
      auto counts = cudaq::sample(ghz{});
      counts.dump();
    }
:::
:::
:::
:::
:::

::: {#quantum-circuits-inc .section}
[]{#quantum-circuits-examples}

## Quantum Circuits, Inc.[¶](#quantum-circuits-inc "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on Quantum Circuits'
backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq

    # Make sure to export or otherwise present your user token via the environment,
    # e.g., using export:
    # ```
    # export QCI_AUTH_TOKEN="your token here"
    # ```
    #
    # The example will run on QCI's AquSim simulator by default.

    cudaq.set_target("qci")


    @cudaq.kernel
    def teleportation():

        # Initialize a three qubit quantum circuit
        qubits = cudaq.qvector(3)

        # Random quantum state on qubit 0.
        rx(3.14, qubits[0])
        ry(2.71, qubits[0])
        rz(6.62, qubits[0])

        # Create a maximally entangled state on qubits 1 and 2.
        h(qubits[1])
        cx(qubits[1], qubits[2])

        cx(qubits[0], qubits[1])

        h(qubits[0])
        m1 = mz(qubits[0])
        m2 = mz(qubits[1])

        if m1 == 1:
            z(qubits[2])

        if m2 == 1:
            x(qubits[2])

        mz(qubits)


    print(cudaq.sample(teleportation))
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile with
    // ```
    // nvq++ teleport.cpp --target qci -o teleport.x
    // ```
    //
    // Make sure to export or otherwise present your user token via the environment,
    // e.g., using export:
    // ```
    // export QCI_AUTH_TOKEN="your token here"
    // ```
    //
    // Then run against the Seeker or AquSim with:
    // ```
    // ./teleport.x
    // ```

    #include <cudaq.h>
    #include <iostream>

    struct teleportation {
      auto operator()() __qpu__ {
        // Initialize a three qubit quantum circuit
        cudaq::qvector qubits(3);

        // Random quantum state on qubit 0.
        rx(3.14, qubits[0]);
        ry(2.71, qubits[0]);
        rz(6.62, qubits[0]);

        // Create a maximally entangled state on qubits 1 and 2.
        h(qubits[1]);
        cx(qubits[1], qubits[2]);

        cx(qubits[0], qubits[1]);
        h(qubits[0]);

        if (mz(qubits[0])) {
          z(qubits[2]);
        }

        if (mz(qubits[1])) {
          x(qubits[2]);
        }

        /// NOTE: If the return statement is changed to `mz(qubits)`, the program
        /// fails. Ref: https://github.com/NVIDIA/cuda-quantum/issues/3708
        return mz(qubits[2]);
      }
    };

    int main() {
      auto results = cudaq::run(20, teleportation{});
      std::cout << "Measurement results of the teleported qubit:\n[ ";
      for (auto r : results)
        std::cout << r << " ";
      std::cout << "]\n";
      return 0;
    }
:::
:::
:::
:::
:::

::: {#quantum-machines .section}
[]{#quantum-machines-examples}

## Quantum Machines[¶](#quantum-machines "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on Quantum Machines'
backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    import math

    # The default executor is mock, use executor name to run on another backend (real or simulator).
    # Configure the address of the QOperator server in the `url` argument, and set the `api_key`.
    cudaq.set_target("quantum_machines",
                     url="http://host.docker.internal:8080",
                     api_key="1234567890",
                     executor="mock")

    qubit_count = 5


    # Maximally entangled state between 5 qubits
    @cudaq.kernel
    def all_h():
        qvector = cudaq.qvector(qubit_count)

        for i in range(qubit_count - 1):
            h(qvector[i])

        s(qvector[0])
        r1(math.pi / 2, qvector[1])
        mz(qvector)


    # Submit synchronously
    cudaq.sample(all_h).dump()
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target quantum_machines quantum_machines.cpp -o out.x && ./out.x
    // ```
    // This will submit the job to the Quantum Machines OPX available in the address
    // provider by `--quantum-machines-url`. By default, the action runs a on a mock
    // executor. To execute or a real QPU please note the executor name by
    // `--quantum-machines-executor`.
    // ```
    // nvq++ --target quantum_machines --quantum-machines-url
    // "https://iqcc.qoperator.qm.co" \
    //  --quantum-machines-executor iqcc quantum_machines.cpp -o out.x
    // ./out.x
    // ```
    // Assumes a valid set of credentials have been set prior to execution.

    #include "math.h"
    #include <cudaq.h>
    #include <fstream>

    // Define a simple quantum kernel to execute on Quantum Machines OPX.
    struct all_h {
      // Maximally entangled state between 5 qubits.
      auto operator()() __qpu__ {
        cudaq::qvector q(5);
        for (int i = 0; i < 4; i++) {
          h(q[i]);
        }
        s(q[0]);
        r1(M_PI / 2, q[1]);
        mz(q);
      }
    };

    int main() {
      // Submit asynchronously (e.g., continue executing code in the file until
      // the job has been returned).
      auto future = cudaq::sample_async(all_h{});
      // ... classical code to execute in the meantime ...

      // Get the results of the read in future.
      auto async_counts = future.get();
      async_counts.dump();

      // OR: Submit synchronously (e.g., wait for the job
      // result to be returned before proceeding).
      auto counts = cudaq::sample(all_h{});
      counts.dump();
    }
:::
:::
:::
:::
:::

::: {#quera-computing .section}
[]{#quera-examples}

## QuEra Computing[¶](#quera-computing "Permalink to this heading"){.headerlink}

The following code illustrates how to run kernels on QuEra's backends.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    from cudaq.operators import RydbergHamiltonian, ScalarOperator
    from cudaq.dynamics import Schedule
    import numpy as np

    ## NOTE: QuEra Aquila system is available via Amazon Braket.
    # Credentials must be set before running this program.
    # Amazon Braket costs apply.

    # This example illustrates how to use QuEra's Aquila device on Braket with CUDA-Q.
    # It is a CUDA-Q implementation of the getting started materials for Braket available here:
    # https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html

    cudaq.set_target("quera")

    # Define the 2-dimensional atom arrangement
    a = 5.7e-6
    register = []
    register.append(tuple(np.array([0.5, 0.5 + 1 / np.sqrt(2)]) * a))
    register.append(tuple(np.array([0.5 + 1 / np.sqrt(2), 0.5]) * a))
    register.append(tuple(np.array([0.5 + 1 / np.sqrt(2), -0.5]) * a))
    register.append(tuple(np.array([0.5, -0.5 - 1 / np.sqrt(2)]) * a))
    register.append(tuple(np.array([-0.5, -0.5 - 1 / np.sqrt(2)]) * a))
    register.append(tuple(np.array([-0.5 - 1 / np.sqrt(2), -0.5]) * a))
    register.append(tuple(np.array([-0.5 - 1 / np.sqrt(2), 0.5]) * a))
    register.append(tuple(np.array([-0.5, 0.5 + 1 / np.sqrt(2)]) * a))

    time_max = 4e-6  # seconds
    time_ramp = 1e-7  # seconds
    omega_max = 6300000.0  # rad / sec
    delta_start = -5 * omega_max
    delta_end = 5 * omega_max

    # Times for the piece-wise linear waveforms
    steps = [0.0, time_ramp, time_max - time_ramp, time_max]
    schedule = Schedule(steps, ["t"])
    # Rabi frequencies at each step
    omega = ScalarOperator(lambda t: omega_max
                           if time_ramp < t.real < time_max else 0.0)
    # Global phase at each step
    phi = ScalarOperator.const(0.0)
    # Global detuning at each step
    delta = ScalarOperator(lambda t: delta_end
                           if time_ramp < t.real < time_max else delta_start)

    async_result = cudaq.evolve_async(RydbergHamiltonian(atom_sites=register,
                                                         amplitude=omega,
                                                         phase=phi,
                                                         delta_global=delta),
                                      schedule=schedule,
                                      shots_count=10).get()
    async_result.dump()

    ## Sample result
    # ```
    # {
    #   __global__ : { 12121222:1 21202221:1 21212121:2 21212122:1 21221212:1 21221221:2 22121221:1 22221221:1 }
    #    post_sequence : { 01010111:1 10101010:2 10101011:1 10101110:1 10110101:1 10110110:2 11010110:1 11110110:1 }
    #    pre_sequence : { 11101111:1 11111111:9 }
    # }
    # ```

    ## Interpreting result
    # `pre_sequence` has the measurement bits, one for each atomic site, before the
    # quantum evolution is run. The count is aggregated across shots. The value is
    # 0 if site is empty, 1 if site is filled.
    # `post_sequence` has the measurement bits, one for each atomic site, at the
    # end of the quantum evolution. The count is aggregated across shots. The value
    # is 0 if atom is in Rydberg state or site is empty, 1 if atom is in ground
    # state.
    # `__global__` has the aggregate of the state counts from all the successful
    # shots. The value is 0 if site is empty, 1 if atom is in Rydberg state (up
    # state spin) and 2 if atom is in ground state (down state spin).
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ --target quera quera_basic.cpp -o out.x
    // ./out.x
    // ```
    // Assumes a valid set of credentials have been stored.

    #include "cudaq/algorithms/evolve.h"
    #include "cudaq/algorithms/integrator.h"
    #include "cudaq/schedule.h"
    #include <cmath>
    #include <map>
    #include <vector>

    // NOTE: QuEra Aquila system is available via Amazon Braket.
    // Credentials must be set before running this program.
    // Amazon Braket costs apply.

    // This example illustrates how to use QuEra's Aquila device on Braket with
    // CUDA-Q. It is a CUDA-Q implementation of the getting started materials for
    // Braket available here:
    // https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html

    int main() {
      // Topology initialization
      const double a = 5.7e-6;
      std::vector<std::pair<double, double>> register_sites;

      auto make_coord = [a](double x, double y) {
        return std::make_pair(x * a, y * a);
      };

      register_sites.push_back(make_coord(0.5, 0.5 + 1.0 / std::sqrt(2)));
      register_sites.push_back(make_coord(0.5 + 1.0 / std::sqrt(2), 0.5));
      register_sites.push_back(make_coord(0.5 + 1.0 / std::sqrt(2), -0.5));
      register_sites.push_back(make_coord(0.5, -0.5 - 1.0 / std::sqrt(2)));
      register_sites.push_back(make_coord(-0.5, -0.5 - 1.0 / std::sqrt(2)));
      register_sites.push_back(make_coord(-0.5 - 1.0 / std::sqrt(2), -0.5));
      register_sites.push_back(make_coord(-0.5 - 1.0 / std::sqrt(2), 0.5));
      register_sites.push_back(make_coord(-0.5, 0.5 + 1.0 / std::sqrt(2)));

      // Simulation Timing
      const double time_max = 4e-6;   // seconds
      const double time_ramp = 1e-7;  // seconds
      const double omega_max = 6.3e6; // rad/sec
      const double delta_start = -5 * omega_max;
      const double delta_end = 5 * omega_max;

      std::vector<std::complex<double>> steps = {0.0, time_ramp,
                                                 time_max - time_ramp, time_max};
      cudaq::schedule schedule(steps, {"t"}, {});

      // Basic Rydberg Hamiltonian
      auto omega = cudaq::scalar_operator(
          [time_ramp, time_max,
           omega_max](const std::unordered_map<std::string, std::complex<double>>
                          &parameters) {
            double t = std::real(parameters.at("t"));
            return std::complex<double>(
                (t > time_ramp && t < time_max) ? omega_max : 0.0, 0.0);
          });

      auto phi = cudaq::scalar_operator(0.0);

      auto delta = cudaq::scalar_operator(
          [time_ramp, time_max, delta_start,
           delta_end](const std::unordered_map<std::string, std::complex<double>>
                          &parameters) {
            double t = std::real(parameters.at("t"));
            return std::complex<double>(
                (t > time_ramp && t < time_max) ? delta_end : delta_start, 0.0);
          });

      auto hamiltonian =
          cudaq::rydberg_hamiltonian(register_sites, omega, phi, delta);

      // Evolve the system
      auto result = cudaq::evolve_async(hamiltonian, schedule, 10).get();
      result.sampling_result->dump();

      return 0;
    }
:::
:::
:::
:::
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](../../examples/python/performance_optimizations.html "Optimizing Performance"){.btn
.btn-neutral .float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](dynamics_examples.html "CUDA-Q Dynamics"){.btn
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
