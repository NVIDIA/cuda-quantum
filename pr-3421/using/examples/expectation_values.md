::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../index.html){.icon .icon-home}

::: version
pr-3421
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
    -   [Computing Expectation Values](#){.current .reference .internal}
        -   [Parallelizing across Multiple
            Processors](#parallelizing-across-multiple-processors){.reference
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
-   Computing Expectation Values
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](executing_kernels.html "Executing Kernels"){.btn .btn-neutral
.float-left accesskey="p"} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](multi_gpu_workflows.html "Multi-GPU Workflows"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#computing-expectation-values .section}
# Computing Expectation Values[¶](#computing-expectation-values "Permalink to this heading"){.headerlink}

CUDA-Q provides generic library functions enabling one to compute
expectation values of quantum spin operators with respect to a
parameterized CUDA-Q kernel. Let's take a look at an example of this:

::: {.tab-set .docutils}
C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // nvq++ expectation_values.cpp -o d2.x && ./d2.x
    // ```

    #include <cudaq.h>
    #include <cudaq/algorithm.h>

    // The example here shows a simple use case for the `cudaq::observe`
    // function in computing expected values of provided spin_ops.

    struct ansatz {
      auto operator()(double theta) __qpu__ {
        cudaq::qvector q(2);
        x(q[0]);
        ry(theta, q[1]);
        x<cudaq::ctrl>(q[1], q[0]);
      }
    };

    int main() {

      // Build up your spin op algebraically
      cudaq::spin_op h =
          5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
          2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
          .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

      // Observe takes the kernel, the spin_op, and the concrete
      // parameters for the kernel
      double energy = cudaq::observe(ansatz{}, h, .59);
      printf("Energy is %lf\n", energy);
      return 0;
    }
:::
:::

Here we define a parameterized CUDA-Q kernel, a callable type named
[`ansatz`{.code .docutils .literal .notranslate}]{.pre} that takes as
input a single angle [`theta`{.code .docutils .literal
.notranslate}]{.pre}. This angle becomes the argument of a single
[`ry`{.code .docutils .literal .notranslate}]{.pre} rotation.

In host code, we define a Hamiltonian operator via the CUDA-Q
[`spin_op`{.code .docutils .literal .notranslate}]{.pre} type. CUDA-Q
provides a generic function [`cudaq::observe`{.code .docutils .literal
.notranslate}]{.pre}. This function takes as input three terms. The
first two terms are a parameterized kernel and the [`spin_op`{.code
.docutils .literal .notranslate}]{.pre} whose expectation value we wish
to compute. The last term contains the runtime parameters at which we
evaluate the parameterized kernel.

The return type of this function is an [`cudaq::observe_result`{.code
.docutils .literal .notranslate}]{.pre} which contains all the data from
the execution, but is trivially convertible to a double, resulting in
the expectation value we are interested in.

To compile and execute this code, we run the following:

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ expectation_values.cpp -o exp_vals.x
    ./exp_vals.x
:::
:::
:::
:::

::: {#parallelizing-across-multiple-processors .section}
## Parallelizing across Multiple Processors[¶](#parallelizing-across-multiple-processors "Permalink to this heading"){.headerlink}

One typical use case of [[multi-processor platforms]{.std
.std-ref}](../backends/sims/mqpusims.html#mqpu-platform){.reference
.internal} is to distribute the expectation value computations of a
multi-term Hamiltonian across multiple virtual QPUs.

The following shows an example using the [`nvidia-mqpu`{.code .docutils
.literal .notranslate}]{.pre} platform:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    from cudaq import spin

    cudaq.set_target("nvidia", option="mqpu")
    target = cudaq.get_target()
    num_qpus = target.num_qpus()
    print("Number of QPUs:", num_qpus)


    # Define spin ansatz.
    @cudaq.kernel
    def kernel(angle: float):
        qvector = cudaq.qvector(2)
        x(qvector[0])
        ry(angle, qvector[1])
        x.ctrl(qvector[1], qvector[0])


    # Define spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    exp_val = cudaq.observe(kernel,
                            hamiltonian,
                            0.59,
                            execution=cudaq.parallel.thread).expectation()
    print("Expectation value: ", exp_val)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      cudaq::spin_op h =
          5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
          2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
          .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

      // Get the quantum_platform singleton
      auto &platform = cudaq::get_platform();

      // Query the number of QPUs in the system
      auto num_qpus = platform.num_qpus();
      printf("Number of QPUs: %zu\n", num_qpus);

      auto ansatz = [](double theta) __qpu__ {
        cudaq::qubit q, r;
        x(q);
        ry(theta, r);
        x<cudaq::ctrl>(r, q);
      };

      double result = cudaq::observe<cudaq::parallel::thread>(ansatz, h, 0.59);
      printf("Expectation value: %lf\n", result);
:::
:::

One can then target the [`nvidia-mqpu`{.code .docutils .literal
.notranslate}]{.pre} platform by executing the following commands:

::: {.highlight-console .notranslate}
::: highlight
    nvq++ observe_mqpu.cpp -target nvidia-mqpu
    ./a.out
:::
:::
:::
:::

In the above code snippets, since the Hamiltonian contains four
non-identity terms, there are four quantum circuits that need to be
executed in order to compute the expectation value of that Hamiltonian
and given the quantum state prepared by the ansatz kernel. When the
[`nvidia-mqpu`{.code .docutils .literal .notranslate}]{.pre} platform is
selected, these circuits will be distributed across all available QPUs.
The final expectation value result is computed from all QPU execution
results.
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](executing_kernels.html "Executing Kernels"){.btn .btn-neutral
.float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](multi_gpu_workflows.html "Multi-GPU Workflows"){.btn
.btn-neutral .float-right accesskey="n" rel="next"}
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
