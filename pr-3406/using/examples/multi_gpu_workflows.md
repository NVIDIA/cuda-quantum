::: {.wy-grid-for-nav}
::: {.wy-side-scroll}
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../index.html){.icon .icon-home}

::: {.version}
pr-3406
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
    -   [Multi-GPU Workflows](#){.current .reference .internal}
        -   [From CPU to GPU](#from-cpu-to-gpu){.reference .internal}
        -   [Pooling the memory of multiple GPUs (`mgpu`{.code .docutils
            .literal
            .notranslate})](#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs (`mqpu`{.code
            .docutils .literal
            .notranslate})](#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit Batching](#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends (`remote-mqpu`{.code .docutils
            .literal
            .notranslate})](#multi-qpu-other-backends-remote-mqpu){.reference
            .internal}
    -   [Optimizers &
        Gradients](../../examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [Built in CUDA-Q Optimizers and
            Gradients](../../examples/python/optimizers_gradients.html#Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
        -   [Third-Party
            Optimizers](../../examples/python/optimizers_gradients.html#Third-Party-Optimizers){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../../examples/python/optimizers_gradients.html#Parallel-Parameter-Shift-Gradients){.reference
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
                `collapse_operators`{.docutils .literal
                .notranslate}](../../examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
                .internal}
            -   [Exercise 2 - Adding additional jump operators
                [\\(L\_i\\)]{.math .notranslate
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
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H\_2\\)]{.math
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
        -   [Using `Sample`{.docutils .literal .notranslate} to perform
            the Hadamard
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
        -   [Constructing circuits in the `[[4,2,2]]`{.docutils .literal
            .notranslate}
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
            -   [3. `Compute overlap`{.docutils .literal
                .notranslate}](../../applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
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
        -   [The problem Hamiltonian [\\(H\_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](../../applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A\_j\\)]{.math .notranslate
            .nohighlight}:](../../applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H\_C,A\_j\]\\)]{.math .notranslate
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
            [\\(A\_i\\)]{.math .notranslate
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
        -   [NVIDIA Quantum Cloud
            (nvqc)](../backends/cloud/nvqc.html){.reference .internal}
            -   [Quick
                Start](../backends/cloud/nvqc.html#quick-start){.reference
                .internal}
            -   [Simulator Backend
                Selection](../backends/cloud/nvqc.html#simulator-backend-selection){.reference
                .internal}
            -   [Multiple
                GPUs](../backends/cloud/nvqc.html#multiple-gpus){.reference
                .internal}
            -   [Multiple QPUs Asynchronous
                Execution](../backends/cloud/nvqc.html#multiple-qpus-asynchronous-execution){.reference
                .internal}
            -   [FAQ](../backends/cloud/nvqc.html#faq){.reference
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
            -   [`CMakeLists.txt`{.docutils .literal
                .notranslate}](../extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](../extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent `CMakeLists.txt`{.docutils .literal
                .notranslate}](../extending/backend.html#update-parent-cmakelists-txt){.reference
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
        -   [`CircuitSimulator`{.code .docutils .literal
            .notranslate}](../extending/nvqir_simulator.html#circuitsimulator){.reference
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
            -   [3.1. `cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. `cudaq::qubit`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](../../specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](../../specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. `cudaq::spin_op`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](../../specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on `cudaq::qubit`{.code .docutils
                .literal
                .notranslate}](../../specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
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
            -   [12.1. `cudaq::sample`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. `cudaq::run`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. `cudaq::observe`{.code .docutils .literal
                .notranslate}](../../specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. `cudaq::optimizer`{.code .docutils .literal
                .notranslate} (deprecated, functionality moved to CUDA-Q
                libraries)](../../specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. `cudaq::gradient`{.code .docutils .literal
                .notranslate} (deprecated, functionality moved to CUDA-Q
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
            -   [`make_kernel()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [`PyKernel`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [`Kernel`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [`PyKernelDecorator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [`kernel()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](../../api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [`sample()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [`sample_async()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [`run()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [`run_async()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [`observe()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [`observe_async()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [`get_state()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [`get_state_async()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [`vqe()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [`draw()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [`translate()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [`estimate_resources()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
        -   [Backend
            Configuration](../../api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [`has_target()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [`get_target()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [`get_targets()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [`set_target()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [`reset_target()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [`set_noise()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [`unset_noise()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [`register_set_target_callback()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [`unregister_set_target_callback()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [`cudaq.apply_noise()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [`initialize_cudaq()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [`num_available_gpus()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [`set_random_seed()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](../../api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [`evolve()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [`evolve_async()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [`Schedule`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [`BaseIntegrator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [`InitialState`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [`InitialStateType`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [`IntermediateResultSave`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](../../api/languages/python_api.html#operators){.reference
            .internal}
            -   [`OperatorSum`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [`ProductOperator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [`ElementaryOperator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [`ScalarOperator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [`RydbergHamiltonian`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [`SuperOperator`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [`operators.define()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [`operators.instantiate()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.operators.instantiate){.reference
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
            -   [`SimulationPrecision`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [`Target`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [`State`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [`Tensor`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [`QuakeValue`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [`qubit`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [`qreg`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [`qvector`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [`ComplexMatrix`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [`SampleResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [`AsyncSampleResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [`ObserveResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [`AsyncObserveResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [`AsyncStateResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [`OptimizationResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [`EvolveResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [`AsyncEvolveResult`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [`Resources`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.Resources){.reference
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
            -   [`initialize()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [`rank()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [`num_ranks()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [`all_gather()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [`broadcast()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [`is_initialized()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [`finalize()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](../../api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [`sample()`{.docutils .literal
                .notranslate}](../../api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
    -   [Quantum Operations](../../api/default_ops.html){.reference
        .internal}
        -   [Unitary Operations on
            Qubits](../../api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [`x`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#x){.reference
                .internal}
            -   [`y`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#y){.reference
                .internal}
            -   [`z`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#z){.reference
                .internal}
            -   [`h`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#h){.reference
                .internal}
            -   [`r1`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#r1){.reference
                .internal}
            -   [`rx`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#rx){.reference
                .internal}
            -   [`ry`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#ry){.reference
                .internal}
            -   [`rz`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#rz){.reference
                .internal}
            -   [`s`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#s){.reference
                .internal}
            -   [`t`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#t){.reference
                .internal}
            -   [`swap`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#swap){.reference
                .internal}
            -   [`u3`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#u3){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](../../api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](../../api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [`mz`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#mz){.reference
                .internal}
            -   [`mx`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#mx){.reference
                .internal}
            -   [`my`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](../../api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](../../api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [`create`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#create){.reference
                .internal}
            -   [`annihilate`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#annihilate){.reference
                .internal}
            -   [`phase_shift`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#phase-shift){.reference
                .internal}
            -   [`beam_splitter`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [`mz`{.code .docutils .literal
                .notranslate}](../../api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](../../versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](../../index.html)

::: {.wy-nav-content}
::: {.rst-content}
::: {role="navigation" aria-label="Page navigation"}
-   [](../../index.html){.icon .icon-home}
-   [CUDA-Q by Example](examples.html)
-   Multi-GPU Workflows
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](expectation_values.html "Computing Expectation Values"){.btn
.btn-neutral .float-left} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](../../examples/python/optimizers_gradients.html "Optimizers and Gradients"){.btn
.btn-neutral .float-right}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#multi-gpu-workflows .section}
Multi-GPU Workflows[¶](#multi-gpu-workflows "Permalink to this heading"){.headerlink}
=====================================================================================

There are many backends available with CUDA-Q that enable seamless
switching between GPUs, QPUs and CPUs and also allow for workflows
involving multiple architectures working in tandem. This page will walk
through the simple steps to accelerate any quantum circuit simulation
with a GPU and how to scale large simulations using multi-GPU multi-node
capabilities.

::: {#from-cpu-to-gpu .section}
From CPU to GPU[¶](#from-cpu-to-gpu "Permalink to this heading"){.headerlink}
-----------------------------------------------------------------------------

The code below defines a kernel that creates a GHZ state using
[\\(N\\)]{.math .notranslate .nohighlight} qubits.

::: {.highlight-python .notranslate}
::: {.highlight}
    import cudaq


    @cudaq.kernel
    def ghz_state(qubit_count: int):
        qubits = cudaq.qvector(qubit_count)
        h(qubits[0])
        for i in range(1, qubit_count):
            cx(qubits[0], qubits[i])
        mz(qubits)


    def sample_ghz_state(qubit_count, target):
        """A function that will sample a variable sized GHZ state."""
        cudaq.set_target(target)
        result = cudaq.sample(ghz_state, qubit_count, shots_count=1000)
        return result
:::
:::

You can run a state vector simulation using your CPU with the
`qpp-cpu`{.code .docutils .literal .notranslate} backend. This is
helpful for debugging code and testing small circuits.

::: {.highlight-python .notranslate}
::: {.highlight}
    cpu_result = sample_ghz_state(qubit_count=2, target="qpp-cpu")
    cpu_result.dump()
:::
:::

::: {.highlight-default .notranslate}
::: {.highlight}
    { 00:475 11:525 }
:::
:::

As the number of qubits increases to even modest size, the CPU
simulation will become impractically slow. By switching to the
`nvidia`{.code .docutils .literal .notranslate} backend, you can
accelerate the same code on a single GPU and achieve a speedup of up to
**425x**. If you have a GPU available, this the default backend to
ensure maximum productivity.

::: {.highlight-python .notranslate}
::: {.highlight}
    if cudaq.num_available_gpus() > 0:
        gpu_result = sample_ghz_state(qubit_count=25, target="nvidia")
        gpu_result.dump()
:::
:::

::: {.highlight-default .notranslate}
::: {.highlight}
    { 0000000000000000000000000:510 1111111111111111111111111:490 }
:::
:::
:::

::: {#pooling-the-memory-of-multiple-gpus-mgpu .section}
Pooling the memory of multiple GPUs (`mgpu`{.code .docutils .literal .notranslate})[¶](#pooling-the-memory-of-multiple-gpus-mgpu "Permalink to this heading"){.headerlink}
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

As `N`{.code .docutils .literal .notranslate} gets larger, the size of
the state vector that needs to be stored in memory increases
exponentially. The state vector has [\\(2\^N\\)]{.math .notranslate
.nohighlight} elements, each a complex number requiring 8 bytes. This
means a 30 qubit simulation requires roughly 8 GB. Adding a few more
qubits will quickly exceed the memory of as single GPU. The `mqpu`{.code
.docutils .literal .notranslate} backend solved this problem by pooling
the memory of multiple GPUs across multiple nodes to perform a single
state vector simulation.

![](../../_images/mgpu.png)

If you have multiple GPUs, you can use the following command to run the
simulation across [\\(n\\)]{.math .notranslate .nohighlight} GPUs.

`mpiexec -np n python3 program.py --target nvidia --target-option mgpu`{.code
.docutils .literal .notranslate}

This code will execute in an MPI context and provide additional memory
to simulation much larger state vectors.

You can also set `cudaq.set_target('nvidia', option='mgpu')`{.code
.docutils .literal .notranslate} within the file to select the target.
:::

::: {#parallel-execution-over-multiple-qpus-mqpu .section}
Parallel execution over multiple QPUs (`mqpu`{.code .docutils .literal .notranslate})[¶](#parallel-execution-over-multiple-qpus-mqpu "Permalink to this heading"){.headerlink}
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

::: {#batching-hamiltonian-terms .section}
### Batching Hamiltonian Terms[¶](#batching-hamiltonian-terms "Permalink to this heading"){.headerlink}

Multiple GPUs can also come in handy for cases where applications might
benefit from multiple QPUs running in parallel. The `mqpu`{.code
.docutils .literal .notranslate} backend uses multiple GPUs to simulate
QPUs so you can accelerate quantum applications with parallelization.

![](../../_images/mqpu.png)

The most simple example is Hamiltonian Batching. In this case, an
expectation value of a large Hamiltonian is distributed across multiple
simulated QPUs, where each QPUs evaluates a subset of the Hamiltonian
terms.

![](../../_images/hsplit.png)

The code below evaluates the expectation value of a random 100000 term
Hamiltonian. A standard `observe`{.code .docutils .literal .notranslate}
call will run the program on a single GPU. Adding the argument
`execution=cudaq.parallel.thread`{.code .docutils .literal .notranslate}
or `execution=cudaq.parallel.mpi`{.code .docutils .literal .notranslate}
will automatically distribute the Hamiltonian terms across multiple GPUs
on a single node or multiple GPUs on multiple nodes, respectively.

The code is executed with
`mpiexec -np n python3 program.py --target nvidia --target-option mqpu`{.code
.docutils .literal .notranslate} where [\\(n\\)]{.math .notranslate
.nohighlight} is the number of GPUs available.

::: {.highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin

    if cudaq.num_available_gpus() == 0:
        print("This example requires a GPU to run. No GPU detected.")
        exit(0)

    cudaq.set_target("nvidia", option="mqpu")
    cudaq.mpi.initialize()

    qubit_count = 15
    term_count = 100000


    @cudaq.kernel
    def kernel(n_qubits: int):

        qubits = cudaq.qvector(n_qubits)

        h(qubits[0])
        for i in range(1, n_qubits):
            x.ctrl(qubits[0], qubits[i])


    # Create a random Hamiltonian
    hamiltonian = cudaq.SpinOperator.random(qubit_count, term_count)

    # The observe calls allows calculation of the the expectation value of the Hamiltonian with respect to a specified kernel.

    # Single node, single GPU.
    result = cudaq.observe(kernel, hamiltonian, qubit_count)
    result.expectation()

    # If multiple GPUs/ QPUs are available, the computation can parallelize with the addition of an argument in the observe call.

    # Single node, multi-GPU.
    result = cudaq.observe(kernel,
                           hamiltonian,
                           qubit_count,
                           execution=cudaq.parallel.thread)
    result.expectation()

    # Multi-node, multi-GPU.
    result = cudaq.observe(kernel,
                           hamiltonian,
                           qubit_count,
                           execution=cudaq.parallel.mpi)
    result.expectation()

    cudaq.mpi.finalize()
:::
:::
:::

::: {#circuit-batching .section}
### Circuit Batching[¶](#circuit-batching "Permalink to this heading"){.headerlink}

A second way to leverage the `mqpu`{.code .docutils .literal
.notranslate} backend is to batch circuit evaluations across multiple
simulated QPUs.

![](../../_images/circsplit.png)

One example where circuit batching is helpful might be evaluating a
parameterized circuit many times with different parameters. The code
below prepares a list of 10000 parameter sets for a 5 qubit circuit.

::: {.highlight-python .notranslate}
::: {.highlight}
    import cudaq
    from cudaq import spin
    import numpy as np

    if cudaq.num_available_gpus() == 0:
        print("This example requires a GPU to run. No GPU detected.")
        exit(0)

    np.random.seed(1)
    cudaq.set_target("nvidia")

    qubit_count = 5
    sample_count = 10000
    h = spin.z(0)
    parameter_count = qubit_count

    # prepare 10000 different input parameter sets.
    parameters = np.random.default_rng(13).uniform(low=0,
                                                   high=1,
                                                   size=(sample_count,
                                                         parameter_count))


    @cudaq.kernel
    def kernel(params: list[float]):

        qubits = cudaq.qvector(5)

        for i in range(5):
            rx(params[i], qubits[i])
:::
:::

All of these circuits can be broadcast through a single
:code"`observe`{.code .docutils .literal .notranslate} call and run by
default on a single GPU. The code below times this entire process.

::: {.highlight-python .notranslate}
::: {.highlight}
    import time

    start_time = time.time()
    cudaq.observe(kernel, h, parameters)
    end_time = time.time()
    print(end_time - start_time)
:::
:::

::: {.highlight-default .notranslate}
::: {.highlight}
    3.185340642929077
:::
:::

This can be greatly accelerated by batching the circuits on multiple
QPUs. The first step is to slice the large list of parameters unto
smaller arrays. The example below divides by four, in preparation to run
on four GPUs.

::: {.highlight-python .notranslate}
::: {.highlight}
    print('There are', parameters.shape[0], 'parameter sets to execute')

    xi = np.split(
        parameters,
        4)  # Split the parameters into 4 arrays since 4 GPUs are available.

    print('Split parameters into', len(xi), 'batches of', xi[0].shape[0], ',',
          xi[1].shape[0], ',', xi[2].shape[0], ',', xi[3].shape[0])
:::
:::

::: {.highlight-default .notranslate}
::: {.highlight}
    There are now 10000 parameter sets split into 4 batches of 2500 , 2500 , 2500 , 2500
:::
:::

As the results are run asynchronously, they need to be stored in a list
(`asyncresults`{.code .docutils .literal .notranslate}) and retrieved
later with the `get`{.code .docutils .literal .notranslate} command. The
following loops over the parameter batches, and the sets of parameters
in each batch. The parameter sets are provided as inputs to
`observe_async`{.code .docutils .literal .notranslate} along with
specification of a `qpu_id`{.code .docutils .literal .notranslate} which
designates the GPU (of the four available) which will run computation. A
speedup of up to 4x can be expected with results varying by problem
size.

::: {.highlight-python .notranslate}
::: {.highlight}
    # Timing the execution on a single GPU vs 4 GPUs,
    # one will see a nearly 4x performance improvement if 4 GPUs are available.

    cudaq.set_target("nvidia", option="mqpu")
    asyncresults = []
    num_gpus = cudaq.num_available_gpus()

    start_time = time.time()
    for i in range(len(xi)):
        for j in range(xi[i].shape[0]):
            qpu_id = i * num_gpus // len(xi)
            asyncresults.append(
                cudaq.observe_async(kernel, h, xi[i][j, :], qpu_id=qpu_id))
    result = [res.get() for res in asyncresults]
    end_time = time.time()
    print(end_time - start_time)
:::
:::

::: {.highlight-default .notranslate}
::: {.highlight}
    1.1754660606384277
:::
:::
:::
:::

::: {#multi-qpu-other-backends-remote-mqpu .section}
Multi-QPU + Other Backends (`remote-mqpu`{.code .docutils .literal .notranslate})[¶](#multi-qpu-other-backends-remote-mqpu "Permalink to this heading"){.headerlink}
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

The `mqpu`{.code .docutils .literal .notranslate} backend can be
extended so that each parallel simulated QPU run backends other than
`nvidia`{.code .docutils .literal .notranslate}. This provides a way to
simulate larger scale circuits and execute parallel algorithms. This
accomplished by launching remotes servers which each simulated a QPU.
The code example below demonstrates this using the `tensornet-mps`{.code
.docutils .literal .notranslate} backend which allows sampling of a 40
qubit circuit too larger for state vector simulation. In this case, the
target is specified as `remote-mqpu`{.code .docutils .literal
.notranslate} while an additional `backend`{.code .docutils .literal
.notranslate} is specified for the simulator used for each QPU.

The default approach uses one GPU per QPU and can both launch and close
each server automatically. This is accomplished by specifying
`auto_launch`{.code .docutils .literal .notranslate} and
:code"`url`{.code .docutils .literal .notranslate} within
`cudaq.set_target`{.code .docutils .literal .notranslate}. Running the
script below will then sample the 40 qubit circuit using two QPUs each
running `tensornet-mps`{.code .docutils .literal .notranslate}.

::: {.highlight-python .notranslate}
::: {.highlight}
    import cudaq

    backend = 'tensornet-mps'

    servers = '2'

    @cudaq.kernel
    def kernel(controls_count: int):
        controls = cudaq.qvector(controls_count)
        targets = cudaq.qvector(40)
        # Place controls in superposition state.
        h(controls)
        for target in range(40):
            x.ctrl(controls, targets[target])
        # Measure.
        mz(controls)
        mz(targets)

    # Set the target to execute on and query the number of QPUs in the system;
    # The number of QPUs is equal to the number of (auto-)launched server instances.
    cudaq.set_target("remote-mqpu",
                     backend=backend,
                     auto_launch=str(servers) if servers.isdigit() else "",
                     url="" if servers.isdigit() else servers)
    qpu_count = cudaq.get_target().num_qpus()
    print("Number of virtual QPUs:", qpu_count)

    # We will launch asynchronous sampling tasks,
    # and will store the results as a future we can query at some later point.
    # Each QPU (indexed by an unique Id) is associated with a remote REST server.
    count_futures = []
    for i in range(qpu_count):

        result = cudaq.sample_async(kernel, i + 1, qpu_id=i)
        count_futures.append(result)
    print("Sampling jobs launched for asynchronous processing.")

    # Go do other work, asynchronous execution of sample tasks on-going.
    # Get the results, note future::get() will kick off a wait
    # if the results are not yet available.
    for idx in range(len(count_futures)):
        counts = count_futures[idx].get()
        print(counts)
:::
:::

`remote-mqpu`{.code .docutils .literal .notranslate} can also be used
with `mqpu`{.code .docutils .literal .notranslate}, allowing each QPU to
be simulated by multiple GPUs. This requires manual preparation of the
servers and detailed instructions are in the [[remote multi-QPU
platform]{.std
.std-ref}](../backends/sims/mqpusims.html#remote-mqpu-platform){.reference
.internal} section of the docs.
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](expectation_values.html "Computing Expectation Values"){.btn
.btn-neutral .float-left} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](../../examples/python/optimizers_gradients.html "Optimizers and Gradients"){.btn
.btn-neutral .float-right}
:::

------------------------------------------------------------------------

::: {role="contentinfo"}
© Copyright 2025, NVIDIA Corporation & Affiliates.
:::

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme) provided by
[Read the Docs](https://readthedocs.org).
:::
:::
:::
:::
