::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../../index.html){.icon .icon-home}

::: version
pr-3535
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](../../quick_start.html){.reference .internal}
    -   [Install
        CUDA-Q](../../quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](../../quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q
        Academic](../../quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](../../basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q
        Kernel?](../../basics/kernel_intro.html){.reference .internal}
    -   [Building your first CUDA-Q
        Program](../../basics/build_kernel.html){.reference .internal}
    -   [Running your first CUDA-Q
        Program](../../basics/run_kernel.html){.reference .internal}
        -   [Sample](../../basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](../../basics/run_kernel.html#run){.reference
            .internal}
        -   [Observe](../../basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](../../basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](../../basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](../../basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
-   [Examples](../../examples/examples.html){.reference .internal}
    -   [Introduction](../../examples/introduction.html){.reference
        .internal}
    -   [Building
        Kernels](../../examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](../../examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](../../examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](../../examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](../../examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](../../examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](../../examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](../../examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](../../examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](../../examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](../../examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](../../examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](../../examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](../../examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring
        Kernels](../../examples/measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](../../examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](../../../examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](../../../examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](../../../examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing
        Kernels](../../examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](../../examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](../../examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](../../examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](../../examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](../../examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](../../examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](../../examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](../../examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](../../examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](../../examples/expectation_values.html){.reference
        .internal}
        -   [Parallelizing across Multiple
            Processors](../../examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](../../examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](../../examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../../examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../../examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](../../examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](../../examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends ([`remote-mqpu`{.code .docutils
            .literal
            .notranslate}]{.pre})](../../examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
            .internal}
    -   [Optimizers &
        Gradients](../../../examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [Built in CUDA-Q Optimizers and
            Gradients](../../../examples/python/optimizers_gradients.html#Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
        -   [Third-Party
            Optimizers](../../../examples/python/optimizers_gradients.html#Third-Party-Optimizers){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../../../examples/python/optimizers_gradients.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](../../../examples/python/noisy_simulations.html){.reference
        .internal}
    -   [Constructing
        Operators](../../examples/operators.html){.reference .internal}
        -   [Constructing Spin
            Operators](../../examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](../../examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](../../../examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](../../../examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](../../examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](../../examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](../../examples/hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](../../examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](../../examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](../../examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](../../examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](../../examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](../../examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](../../examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum Circuits,
            Inc.](../../examples/hardware_providers.html#quantum-circuits-inc){.reference
            .internal}
        -   [Quantum
            Machines](../../examples/hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](../../examples/hardware_providers.html#quera-computing){.reference
            .internal}
    -   [Dynamics
        Examples](../../examples/dynamics_examples.html){.reference
        .internal}
        -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
            Model)](../../../examples/python/dynamics/dynamics_intro_1.html){.reference
            .internal}
            -   [Why dynamics simulations vs. circuit
                simulations?](../../../examples/python/dynamics/dynamics_intro_1.html#Why-dynamics-simulations-vs.-circuit-simulations?){.reference
                .internal}
            -   [Functionality](../../../examples/python/dynamics/dynamics_intro_1.html#Functionality){.reference
                .internal}
            -   [Performance](../../../examples/python/dynamics/dynamics_intro_1.html#Performance){.reference
                .internal}
            -   [Section 1 - Simulating the Jaynes-Cummings
                Hamiltonian](../../../examples/python/dynamics/dynamics_intro_1.html#Section-1---Simulating-the-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Exercise 1 - Simulating a many-photon Jaynes-Cummings
                Hamiltonian](../../../examples/python/dynamics/dynamics_intro_1.html#Exercise-1---Simulating-a-many-photon-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Section 2 - Simulating open quantum systems with the
                [`collapse_operators`{.docutils .literal
                .notranslate}]{.pre}](../../../examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
                .internal}
            -   [Exercise 2 - Adding additional jump operators
                [\\(L_i\\)]{.math .notranslate
                .nohighlight}](../../../examples/python/dynamics/dynamics_intro_1.html#Exercise-2---Adding-additional-jump-operators-L_i){.reference
                .internal}
            -   [Section 3 - Many qubits coupled to the
                resonator](../../../examples/python/dynamics/dynamics_intro_1.html#Section-3---Many-qubits-coupled-to-the-resonator){.reference
                .internal}
        -   [Introduction to CUDA-Q Dynamics (Time Dependent
            Hamiltonians)](../../../examples/python/dynamics/dynamics_intro_2.html){.reference
            .internal}
            -   [The Landau-Zener
                model](../../../examples/python/dynamics/dynamics_intro_2.html#The-Landau-Zener-model){.reference
                .internal}
            -   [Section 1 - Implementing time dependent
                terms](../../../examples/python/dynamics/dynamics_intro_2.html#Section-1---Implementing-time-dependent-terms){.reference
                .internal}
            -   [Section 2 - Implementing custom
                operators](../../../examples/python/dynamics/dynamics_intro_2.html#Section-2---Implementing-custom-operators){.reference
                .internal}
            -   [Section 3 - Heisenberg Model with a time-varying
                magnetic
                field](../../../examples/python/dynamics/dynamics_intro_2.html#Section-3---Heisenberg-Model-with-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 1 - Define a time-varying magnetic
                field](../../../examples/python/dynamics/dynamics_intro_2.html#Exercise-1---Define-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 2
                (Optional)](../../../examples/python/dynamics/dynamics_intro_2.html#Exercise-2-(Optional)){.reference
                .internal}
        -   [Superconducting
            Qubits](../../../examples/python/dynamics/superconducting.html){.reference
            .internal}
            -   [Cavity
                QED](../../../examples/python/dynamics/superconducting.html#Cavity-QED){.reference
                .internal}
            -   [Cross
                Resonance](../../../examples/python/dynamics/superconducting.html#Cross-Resonance){.reference
                .internal}
            -   [Transmon
                Resonator](../../../examples/python/dynamics/superconducting.html#Transmon-Resonator){.reference
                .internal}
        -   [Spin
            Qubits](../../../examples/python/dynamics/spinqubits.html){.reference
            .internal}
            -   [Silicon Spin
                Qubit](../../../examples/python/dynamics/spinqubits.html#Silicon-Spin-Qubit){.reference
                .internal}
            -   [Heisenberg
                Model](../../../examples/python/dynamics/spinqubits.html#Heisenberg-Model){.reference
                .internal}
        -   [Trapped Ion
            Qubits](../../../examples/python/dynamics/iontrap.html){.reference
            .internal}
            -   [GHZ
                state](../../../examples/python/dynamics/iontrap.html#GHZ-state){.reference
                .internal}
        -   [Control](../../../examples/python/dynamics/control.html){.reference
            .internal}
            -   [Gate
                Calibration](../../../examples/python/dynamics/control.html#Gate-Calibration){.reference
                .internal}
            -   [Pulse](../../../examples/python/dynamics/control.html#Pulse){.reference
                .internal}
            -   [Qubit
                Control](../../../examples/python/dynamics/control.html#Qubit-Control){.reference
                .internal}
            -   [Qubit
                Dynamics](../../../examples/python/dynamics/control.html#Qubit-Dynamics){.reference
                .internal}
            -   [Landau-Zenner](../../../examples/python/dynamics/control.html#Landau-Zenner){.reference
                .internal}
-   [Applications](../../applications.html){.reference .internal}
    -   [Max-Cut with
        QAOA](../../../applications/python/qaoa.html){.reference
        .internal}
    -   [Molecular docking via
        DC-QAOA](../../../applications/python/digitized_counterdiabatic_qaoa.html){.reference
        .internal}
        -   [Setting up the Molecular Docking
            Problem](../../../applications/python/digitized_counterdiabatic_qaoa.html#Setting-up-the-Molecular-Docking-Problem){.reference
            .internal}
        -   [CUDA-Q
            Implementation](../../../applications/python/digitized_counterdiabatic_qaoa.html#CUDA-Q-Implementation){.reference
            .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](../../../applications/python/krylov.html){.reference
        .internal}
        -   [Setup](../../../applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](../../../applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](../../../applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Quantum-Selected Configuration Interaction
        (QSCI)](../../../applications/python/qsci.html){.reference
        .internal}
        -   [0. Problem
            definition](../../../applications/python/qsci.html#0.-Problem-definition){.reference
            .internal}
        -   [1. Prepare an Approximate Quantum
            State](../../../applications/python/qsci.html#1.-Prepare-an-Approximate-Quantum-State){.reference
            .internal}
        -   [2 Quantum Sampling to Select
            Configuration](../../../applications/python/qsci.html#2-Quantum-Sampling-to-Select-Configuration){.reference
            .internal}
        -   [3. Classical Diagonalization on the Selected
            Subspace](../../../applications/python/qsci.html#3.-Classical-Diagonalization-on-the-Selected-Subspace){.reference
            .internal}
        -   [5. Compuare
            results](../../../applications/python/qsci.html#5.-Compuare-results){.reference
            .internal}
        -   [Reference](../../../applications/python/qsci.html#Reference){.reference
            .internal}
    -   [Bernstein-Vazirani
        Algorithm](../../../applications/python/bernstein_vazirani.html){.reference
        .internal}
        -   [Classical
            case](../../../applications/python/bernstein_vazirani.html#Classical-case){.reference
            .internal}
        -   [Quantum
            case](../../../applications/python/bernstein_vazirani.html#Quantum-case){.reference
            .internal}
        -   [Implementing in
            CUDA-Q](../../../applications/python/bernstein_vazirani.html#Implementing-in-CUDA-Q){.reference
            .internal}
    -   [Cost
        Minimization](../../../applications/python/cost_minimization.html){.reference
        .internal}
    -   [Deutsch's
        Algorithm](../../../applications/python/deutsch_algorithm.html){.reference
        .internal}
        -   [XOR [\\(\\oplus\\)]{.math .notranslate
            .nohighlight}](../../../applications/python/deutsch_algorithm.html#XOR-\oplus){.reference
            .internal}
        -   [Quantum
            oracles](../../../applications/python/deutsch_algorithm.html#Quantum-oracles){.reference
            .internal}
        -   [Phase
            oracle](../../../applications/python/deutsch_algorithm.html#Phase-oracle){.reference
            .internal}
        -   [Quantum
            parallelism](../../../applications/python/deutsch_algorithm.html#Quantum-parallelism){.reference
            .internal}
        -   [Deutsch's
            Algorithm:](../../../applications/python/deutsch_algorithm.html#Deutsch's-Algorithm:){.reference
            .internal}
    -   [Divisive Clustering With Coresets Using
        CUDA-Q](../../../applications/python/divisive_clustering_coresets.html){.reference
        .internal}
        -   [Data
            preprocessing](../../../applications/python/divisive_clustering_coresets.html#Data-preprocessing){.reference
            .internal}
        -   [Quantum
            functions](../../../applications/python/divisive_clustering_coresets.html#Quantum-functions){.reference
            .internal}
        -   [Divisive Clustering
            Function](../../../applications/python/divisive_clustering_coresets.html#Divisive-Clustering-Function){.reference
            .internal}
        -   [QAOA
            Implementation](../../../applications/python/divisive_clustering_coresets.html#QAOA-Implementation){.reference
            .internal}
        -   [Scaling simulations with
            CUDA-Q](../../../applications/python/divisive_clustering_coresets.html#Scaling-simulations-with-CUDA-Q){.reference
            .internal}
    -   [Hybrid Quantum Neural
        Networks](../../../applications/python/hybrid_quantum_neural_networks.html){.reference
        .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](../../../applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](../../../applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using [`Sample`{.docutils .literal .notranslate}]{.pre} to
            perform the Hadamard
            test](../../../applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](../../../applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](../../../applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Anderson Impurity Model ground state solver on Infleqtion's
        Sqale](../../../applications/python/logical_aim_sqale.html){.reference
        .internal}
        -   [Performing logical Variational Quantum Eigensolver (VQE)
            with
            CUDA-QX](../../../applications/python/logical_aim_sqale.html#Performing-logical-Variational-Quantum-Eigensolver-(VQE)-with-CUDA-QX){.reference
            .internal}
        -   [Constructing circuits in the [`[[4,2,2]]`{.docutils
            .literal .notranslate}]{.pre}
            encoding](../../../applications/python/logical_aim_sqale.html#Constructing-circuits-in-the-%5B%5B4,2,2%5D%5D-encoding){.reference
            .internal}
        -   [Setting up submission and decoding
            workflow](../../../applications/python/logical_aim_sqale.html#Setting-up-submission-and-decoding-workflow){.reference
            .internal}
        -   [Running a CUDA-Q noisy
            simulation](../../../applications/python/logical_aim_sqale.html#Running-a-CUDA-Q-noisy-simulation){.reference
            .internal}
        -   [Running logical AIM on Infleqtion's
            hardware](../../../applications/python/logical_aim_sqale.html#Running-logical-AIM-on-Infleqtion's-hardware){.reference
            .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](../../../applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](../../../applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](../../../applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](../../../applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](../../../applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](../../../applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](../../../applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](../../../applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. [`Compute`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`overlap`{.docutils .literal
                .notranslate}]{.pre}](../../../applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](../../../applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](../../../applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](../../../applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](../../../applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](../../../applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](../../../applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](../../../applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](../../../applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](../../../applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum Fourier
        Transform](../../../applications/python/quantum_fourier_transform.html){.reference
        .internal}
        -   [Quantum Fourier Transform
            revisited](../../../applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform-revisited){.reference
            .internal}
    -   [Quantum
        Teleporation](../../../applications/python/quantum_teleportation.html){.reference
        .internal}
        -   [Teleportation
            explained](../../../applications/python/quantum_teleportation.html#Teleportation-explained){.reference
            .internal}
    -   [Quantum
        Volume](../../../applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](../../../applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](../../../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](../../../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](../../../applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Compiling Unitaries Using Diffusion
        Models](../../../applications/python/unitary_compilation_diffusion_models.html){.reference
        .internal}
        -   [Diffusion model
            pipeline](../../../applications/python/unitary_compilation_diffusion_models.html#Diffusion-model-pipeline){.reference
            .internal}
        -   [Setup and load
            models](../../../applications/python/unitary_compilation_diffusion_models.html#Setup-and-load-models){.reference
            .internal}
            -   [Load discrete
                model](../../../applications/python/unitary_compilation_diffusion_models.html#Load-discrete-model){.reference
                .internal}
            -   [Load continuous
                model](../../../applications/python/unitary_compilation_diffusion_models.html#Load-continuous-model){.reference
                .internal}
            -   [Create helper
                functions](../../../applications/python/unitary_compilation_diffusion_models.html#Create-helper-functions){.reference
                .internal}
        -   [Unitary
            compilation](../../../applications/python/unitary_compilation_diffusion_models.html#Unitary-compilation){.reference
            .internal}
            -   [Random
                unitary](../../../applications/python/unitary_compilation_diffusion_models.html#Random-unitary){.reference
                .internal}
            -   [Discrete
                model](../../../applications/python/unitary_compilation_diffusion_models.html#Discrete-model){.reference
                .internal}
            -   [Continuous
                model](../../../applications/python/unitary_compilation_diffusion_models.html#Continuous-model){.reference
                .internal}
            -   [Quantum Fourier
                transform](../../../applications/python/unitary_compilation_diffusion_models.html#Quantum-Fourier-transform){.reference
                .internal}
            -   [XXZ-Hamiltonian
                evolution](../../../applications/python/unitary_compilation_diffusion_models.html#XXZ-Hamiltonian-evolution){.reference
                .internal}
        -   [Choosing the circuit you
            need](../../../applications/python/unitary_compilation_diffusion_models.html#Choosing-the-circuit-you-need){.reference
            .internal}
    -   [VQE with gradients, active spaces, and gate
        fusion](../../../applications/python/vqe_advanced.html){.reference
        .internal}
        -   [The Basics of
            VQE](../../../applications/python/vqe_advanced.html#The-Basics-of-VQE){.reference
            .internal}
        -   [Installing/Loading Relevant
            Packages](../../../applications/python/vqe_advanced.html#Installing/Loading-Relevant-Packages){.reference
            .internal}
        -   [Implementing VQE in
            CUDA-Q](../../../applications/python/vqe_advanced.html#Implementing-VQE-in-CUDA-Q){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../../../applications/python/vqe_advanced.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
        -   [Using an Active
            Space](../../../applications/python/vqe_advanced.html#Using-an-Active-Space){.reference
            .internal}
        -   [Gate Fusion for Larger
            Circuits](../../../applications/python/vqe_advanced.html#Gate-Fusion-for-Larger-Circuits){.reference
            .internal}
    -   [Quantum
        Transformer](../../../applications/python/quantum_transformer.html){.reference
        .internal}
        -   [Installation](../../../applications/python/quantum_transformer.html#Installation){.reference
            .internal}
        -   [Algorithm and
            Example](../../../applications/python/quantum_transformer.html#Algorithm-and-Example){.reference
            .internal}
            -   [Creating the self-attention
                circuits](../../../applications/python/quantum_transformer.html#Creating-the-self-attention-circuits){.reference
                .internal}
        -   [Usage](../../../applications/python/quantum_transformer.html#Usage){.reference
            .internal}
            -   [Model
                Training](../../../applications/python/quantum_transformer.html#Model-Training){.reference
                .internal}
            -   [Generating
                Molecules](../../../applications/python/quantum_transformer.html#Generating-Molecules){.reference
                .internal}
            -   [Attention
                Maps](../../../applications/python/quantum_transformer.html#Attention-Maps){.reference
                .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](../../../applications/python/afqmc.html){.reference
        .internal}
        -   [Hamiltonian preparation for
            VQE](../../../applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](../../../applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](../../../applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](../../../applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](../../../applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](../../../applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [ADAPT-QAOA
        algorithm](../../../applications/python/adapt_qaoa.html){.reference
        .internal}
        -   [Simulation
            input:](../../../applications/python/adapt_qaoa.html#Simulation-input:){.reference
            .internal}
        -   [The problem Hamiltonian [\\(H_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](../../../applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A_j\\)]{.math .notranslate
            .nohighlight}:](../../../applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H_C,A_j\]\\)]{.math .notranslate
            .nohighlight}:](../../../applications/python/adapt_qaoa.html#The-commutator-%5BH_C,A_j%5D:){.reference
            .internal}
        -   [Beginning of ADAPT-QAOA
            iteration:](../../../applications/python/adapt_qaoa.html#Beginning-of-ADAPT-QAOA-iteration:){.reference
            .internal}
    -   [ADAPT-VQE
        algorithm](../../../applications/python/adapt_vqe.html){.reference
        .internal}
        -   [Classical
            pre-processing](../../../applications/python/adapt_vqe.html#Classical-pre-processing){.reference
            .internal}
        -   [Jordan
            Wigner:](../../../applications/python/adapt_vqe.html#Jordan-Wigner:){.reference
            .internal}
        -   [UCCSD operator
            pool](../../../applications/python/adapt_vqe.html#UCCSD-operator-pool){.reference
            .internal}
            -   [Single
                excitation](../../../applications/python/adapt_vqe.html#Single-excitation){.reference
                .internal}
            -   [Double
                excitation](../../../applications/python/adapt_vqe.html#Double-excitation){.reference
                .internal}
        -   [Commutator \[[\\(H\\)]{.math .notranslate .nohighlight},
            [\\(A_i\\)]{.math .notranslate
            .nohighlight}\]](../../../applications/python/adapt_vqe.html#Commutator-%5BH,-A_i%5D){.reference
            .internal}
        -   [Reference
            State:](../../../applications/python/adapt_vqe.html#Reference-State:){.reference
            .internal}
        -   [Quantum
            kernels:](../../../applications/python/adapt_vqe.html#Quantum-kernels:){.reference
            .internal}
        -   [Beginning of
            ADAPT-VQE:](../../../applications/python/adapt_vqe.html#Beginning-of-ADAPT-VQE:){.reference
            .internal}
    -   [Quantum edge
        detection](../../../applications/python/edge_detection.html){.reference
        .internal}
        -   [Image](../../../applications/python/edge_detection.html#Image){.reference
            .internal}
        -   [Quantum Probability Image Encoding
            (QPIE):](../../../applications/python/edge_detection.html#Quantum-Probability-Image-Encoding-(QPIE):){.reference
            .internal}
            -   [Below we show how to encode an image using QPIE in
                cudaq.](../../../applications/python/edge_detection.html#Below-we-show-how-to-encode-an-image-using-QPIE-in-cudaq.){.reference
                .internal}
        -   [Flexible Representation of Quantum Images
            (FRQI):](../../../applications/python/edge_detection.html#Flexible-Representation-of-Quantum-Images-(FRQI):){.reference
            .internal}
            -   [Building the FRQI
                State:](../../../applications/python/edge_detection.html#Building-the-FRQI-State:){.reference
                .internal}
        -   [Quantum Hadamard Edge Detection
            (QHED)](../../../applications/python/edge_detection.html#Quantum-Hadamard-Edge-Detection-(QHED)){.reference
            .internal}
            -   [Post-processing](../../../applications/python/edge_detection.html#Post-processing){.reference
                .internal}
    -   [Factoring Integers With Shor's
        Algorithm](../../../applications/python/shors.html){.reference
        .internal}
        -   [Shor's
            algorithm](../../../applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](../../../applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](../../../applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](../../../applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](../../../applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](../../../applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](../../../applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](../../../applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
            -   [(a) Generate the molecular Hamiltonian using Restricted
                Hartree Fock molecular
                orbitals](../../../applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Restricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(b) Generate the molecular Hamiltonian using
                Unrestricted Hartree Fock molecular
                orbitals](../../../applications/python/generate_fermionic_ham.html#(b)-Generate-the-molecular-Hamiltonian-using-Unrestricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(a) Generate the active space hamiltonian using RHF
                molecular
                orbitals.](../../../applications/python/generate_fermionic_ham.html#(a)-Generate-the-active-space-hamiltonian-using-RHF-molecular-orbitals.){.reference
                .internal}
            -   [(b) Generate the active space Hamiltonian using the
                natural orbitals computed from MP2
                simulation](../../../applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
                .internal}
            -   [(c) Generate the active space Hamiltonian computed from
                the CASSCF molecular
                orbitals](../../../applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
                .internal}
            -   [(d) Generate the electronic Hamiltonian using
                ROHF](../../../applications/python/generate_fermionic_ham.html#(d)-Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
            -   [(e) Generate electronic Hamiltonian using
                UHF](../../../applications/python/generate_fermionic_ham.html#(e)-Generate-electronic-Hamiltonian-using-UHF){.reference
                .internal}
    -   [Grover's
        Algorithm](../../../applications/python/grovers.html){.reference
        .internal}
        -   [Overview](../../../applications/python/grovers.html#Overview){.reference
            .internal}
        -   [Problem](../../../applications/python/grovers.html#Problem){.reference
            .internal}
        -   [Structure of Grover's
            Algorithm](../../../applications/python/grovers.html#Structure-of-Grover's-Algorithm){.reference
            .internal}
            -   [Step 1:
                Preparation](../../../applications/python/grovers.html#Step-1:-Preparation){.reference
                .internal}
            -   [Good and Bad
                States](../../../applications/python/grovers.html#Good-and-Bad-States){.reference
                .internal}
            -   [Step 2: Oracle
                application](../../../applications/python/grovers.html#Step-2:-Oracle-application){.reference
                .internal}
            -   [Step 3: Amplitude
                amplification](../../../applications/python/grovers.html#Step-3:-Amplitude-amplification){.reference
                .internal}
            -   [Steps 4 and 5: Iteration and
                measurement](../../../applications/python/grovers.html#Steps-4-and-5:-Iteration-and-measurement){.reference
                .internal}
    -   [Quantum
        PageRank](../../../applications/python/quantum_pagerank.html){.reference
        .internal}
        -   [Problem
            Definition](../../../applications/python/quantum_pagerank.html#Problem-Definition){.reference
            .internal}
        -   [Simulating Quantum PageRank by CUDA-Q
            dynamics](../../../applications/python/quantum_pagerank.html#Simulating-Quantum-PageRank-by-CUDA-Q-dynamics){.reference
            .internal}
        -   [Breakdown of
            Terms](../../../applications/python/quantum_pagerank.html#Breakdown-of-Terms){.reference
            .internal}
    -   [The UCCSD Wavefunction
        ansatz](../../../applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](../../../applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](../../../applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](../../../applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](../../../applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](../../../applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](../../../applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [QM/MM simulation: VQE within a Polarizable Embedded
        Framework.](../../../applications/python/qm_mm_pe.html){.reference
        .internal}
        -   [Key
            concepts:](../../../applications/python/qm_mm_pe.html#Key-concepts:){.reference
            .internal}
        -   [PE-VQE-SCF Algorithm
            Steps](../../../applications/python/qm_mm_pe.html#PE-VQE-SCF-Algorithm-Steps){.reference
            .internal}
            -   [Step 1: Initialize (Classical
                pre-processing)](../../../applications/python/qm_mm_pe.html#Step-1:-Initialize-(Classical-pre-processing)){.reference
                .internal}
            -   [Step 2: Build the
                Hamiltonian](../../../applications/python/qm_mm_pe.html#Step-2:-Build-the-Hamiltonian){.reference
                .internal}
            -   [Step 3: Run
                VQE](../../../applications/python/qm_mm_pe.html#Step-3:-Run-VQE){.reference
                .internal}
            -   [Step 4: Update
                Environment](../../../applications/python/qm_mm_pe.html#Step-4:-Update-Environment){.reference
                .internal}
            -   [Step 5: Self-Consistency
                Loop](../../../applications/python/qm_mm_pe.html#Step-5:-Self-Consistency-Loop){.reference
                .internal}
            -   [Requirments:](../../../applications/python/qm_mm_pe.html#Requirments:){.reference
                .internal}
            -   [Example 1: LiH with 2 water
                molecules.](../../../applications/python/qm_mm_pe.html#Example-1:-LiH-with-2-water-molecules.){.reference
                .internal}
            -   [VQE, update environment, and scf
                loop.](../../../applications/python/qm_mm_pe.html#VQE,-update-environment,-and-scf-loop.){.reference
                .internal}
            -   [Example 2: NH3 with 46 water molecule using active
                space.](../../../applications/python/qm_mm_pe.html#Example-2:-NH3-with-46-water-molecule-using-active-space.){.reference
                .internal}
    -   [Sample-Based Krylov Quantum Diagonalization
        (SKQD)](../../../applications/python/skqd.html){.reference
        .internal}
        -   [Why
            SKQD?](../../../applications/python/skqd.html#Why-SKQD?){.reference
            .internal}
        -   [Setup and
            Imports](../../../applications/python/skqd.html#Setup-and-Imports){.reference
            .internal}
        -   [Understanding Krylov
            Subspaces](../../../applications/python/skqd.html#Understanding-Krylov-Subspaces){.reference
            .internal}
            -   [What is a Krylov
                Subspace?](../../../applications/python/skqd.html#What-is-a-Krylov-Subspace?){.reference
                .internal}
            -   [The SKQD
                Algorithm](../../../applications/python/skqd.html#The-SKQD-Algorithm){.reference
                .internal}
        -   [Krylov State Generation via Repeated
            Evolution](../../../applications/python/skqd.html#Krylov-State-Generation-via-Repeated-Evolution){.reference
            .internal}
        -   [Quantum Measurements and
            Sampling](../../../applications/python/skqd.html#Quantum-Measurements-and-Sampling){.reference
            .internal}
            -   [The Sampling
                Process](../../../applications/python/skqd.html#The-Sampling-Process){.reference
                .internal}
        -   [Classical Post-Processing and
            Diagonalization](../../../applications/python/skqd.html#Classical-Post-Processing-and-Diagonalization){.reference
            .internal}
            -   [The SKQD Algorithm: Matrix Construction
                Details](../../../applications/python/skqd.html#The-SKQD-Algorithm:-Matrix-Construction-Details){.reference
                .internal}
        -   [Results Analysis and
            Convergence](../../../applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](../../../applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
    -   [Entanglement Accelerates Quantum
        Simulation](../../../applications/python/entanglement_acc_hamiltonian_simulation.html){.reference
        .internal}
        -   [2. Model
            Definition](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.-Model-Definition){.reference
            .internal}
            -   [2.1 Initial product
                state](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.1-Initial-product-state){.reference
                .internal}
            -   [2.2 QIMF
                Hamiltonian](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.2-QIMF-Hamiltonian){.reference
                .internal}
            -   [2.3 First-Order Trotter Formula
                (PF1)](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.3-First-Order-Trotter-Formula-(PF1)){.reference
                .internal}
            -   [2.4 PF1 step for the QIMF
                partition](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.4-PF1-step-for-the-QIMF-partition){.reference
                .internal}
            -   [2.5 Hamiltonian
                helpers](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#2.5-Hamiltonian-helpers){.reference
                .internal}
        -   [3. Entanglement
            metrics](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#3.-Entanglement-metrics){.reference
            .internal}
        -   [4. Simulation
            workflow](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#4.-Simulation-workflow){.reference
            .internal}
            -   [4.1 Single-step Trotter
                error](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#4.1-Single-step-Trotter-error){.reference
                .internal}
            -   [4.2 Dual trajectory
                update](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#4.2-Dual-trajectory-update){.reference
                .internal}
        -   [5. Reproducing the paper's Figure
            1a](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#5.-Reproducing-the-paper’s-Figure-1a){.reference
            .internal}
            -   [5.1 Visualising the joint
                behaviour](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#5.1-Visualising-the-joint-behaviour){.reference
                .internal}
            -   [5.2 Interpreting the
                result](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#5.2-Interpreting-the-result){.reference
                .internal}
        -   [6. References and further
            reading](../../../applications/python/entanglement_acc_hamiltonian_simulation.html#6.-References-and-further-reading){.reference
            .internal}
-   [Backends](../backends.html){.reference .internal}
    -   [Circuit Simulation](../simulators.html){.reference .internal}
        -   [State Vector Simulators](#){.current .reference .internal}
            -   [CPU](#cpu){.reference .internal}
            -   [Single-GPU](#single-gpu){.reference .internal}
            -   [Multi-GPU multi-node](#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network Simulators](tnsims.html){.reference
            .internal}
            -   [Multi-GPU
                multi-node](tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](tnsims.html#fermioniq){.reference .internal}
        -   [Multi-QPU Simulators](mqpusims.html){.reference .internal}
            -   [Simulate Multiple QPUs in
                Parallel](mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](mqpusims.html#multi-qpu-other-backends){.reference
                .internal}
        -   [Noisy Simulators](noisy.html){.reference .internal}
            -   [Trajectory Noisy
                Simulation](noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density Matrix](noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](noisy.html#stim){.reference .internal}
        -   [Photonics Simulators](photonics.html){.reference .internal}
            -   [orca-photonics](photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware (QPUs)](../hardware.html){.reference
        .internal}
        -   [Ion Trap QPUs](../hardware/iontrap.html){.reference
            .internal}
            -   [IonQ](../hardware/iontrap.html#ionq){.reference
                .internal}
            -   [Quantinuum](../hardware/iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting
            QPUs](../hardware/superconducting.html){.reference
            .internal}
            -   [Anyon Technologies/Anyon
                Computing](../hardware/superconducting.html#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](../hardware/superconducting.html#iqm){.reference
                .internal}
            -   [OQC](../hardware/superconducting.html#oqc){.reference
                .internal}
            -   [Quantum Circuits,
                Inc.](../hardware/superconducting.html#quantum-circuits-inc){.reference
                .internal}
        -   [Neutral Atom QPUs](../hardware/neutralatom.html){.reference
            .internal}
            -   [Infleqtion](../hardware/neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](../hardware/neutralatom.html#pasqal){.reference
                .internal}
            -   [QuEra
                Computing](../hardware/neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic QPUs](../hardware/photonic.html){.reference
            .internal}
            -   [ORCA
                Computing](../hardware/photonic.html#orca-computing){.reference
                .internal}
        -   [Quantum Control
            Systems](../hardware/qcontrol.html){.reference .internal}
            -   [Quantum
                Machines](../hardware/qcontrol.html#quantum-machines){.reference
                .internal}
    -   [Dynamics Simulation](../dynamics_backends.html){.reference
        .internal}
    -   [Cloud](../cloud.html){.reference .internal}
        -   [Amazon Braket (braket)](../cloud/braket.html){.reference
            .internal}
            -   [Setting
                Credentials](../cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submission from
                C++](../cloud/braket.html#submission-from-c){.reference
                .internal}
            -   [Submission from
                Python](../cloud/braket.html#submission-from-python){.reference
                .internal}
        -   [NVIDIA Quantum Cloud (nvqc)](../cloud/nvqc.html){.reference
            .internal}
            -   [Quick Start](../cloud/nvqc.html#quick-start){.reference
                .internal}
            -   [Simulator Backend
                Selection](../cloud/nvqc.html#simulator-backend-selection){.reference
                .internal}
            -   [Multiple
                GPUs](../cloud/nvqc.html#multiple-gpus){.reference
                .internal}
            -   [Multiple QPUs Asynchronous
                Execution](../cloud/nvqc.html#multiple-qpus-asynchronous-execution){.reference
                .internal}
            -   [FAQ](../cloud/nvqc.html#faq){.reference .internal}
-   [Dynamics](../../dynamics.html){.reference .internal}
    -   [Quick Start](../../dynamics.html#quick-start){.reference
        .internal}
    -   [Operator](../../dynamics.html#operator){.reference .internal}
    -   [Time-Dependent
        Dynamics](../../dynamics.html#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](../../dynamics.html#super-operator-representation){.reference
        .internal}
    -   [Numerical
        Integrators](../../dynamics.html#numerical-integrators){.reference
        .internal}
    -   [Batch
        simulation](../../dynamics.html#batch-simulation){.reference
        .internal}
    -   [Multi-GPU Multi-Node
        Execution](../../dynamics.html#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](../../dynamics.html#examples){.reference .internal}
-   [CUDA-QX](../../cudaqx/cudaqx.html){.reference .internal}
    -   [CUDA-Q
        Solvers](../../cudaqx/cudaqx.html#cuda-q-solvers){.reference
        .internal}
    -   [CUDA-Q QEC](../../cudaqx/cudaqx.html#cuda-q-qec){.reference
        .internal}
-   [Installation](../../install/install.html){.reference .internal}
    -   [Local
        Installation](../../install/local_installation.html){.reference
        .internal}
        -   [Introduction](../../install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](../../install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](../../install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](../../install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](../../install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](../../install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](../../install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](../../install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](../../install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](../../install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](../../install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](../../install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](../../install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](../../install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](../../install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](../../install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](../../install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](../../install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](../../install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](../../install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](../../install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](../../install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](../../install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
        -   [Next
            Steps](../../install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](../../install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](../../install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](../../install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](../../install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](../../install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](../../install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](../../install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](../../install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](../../install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](../../install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](../../install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](../../integration/integration.html){.reference
    .internal}
    -   [Downstream CMake
        Integration](../../integration/cmake_app.html){.reference
        .internal}
    -   [Combining CUDA with
        CUDA-Q](../../integration/cuda_gpu.html){.reference .internal}
    -   [Integrating with Third-Party
        Libraries](../../integration/libraries.html){.reference
        .internal}
        -   [Calling a CUDA-Q library from
            C++](../../integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](../../integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](../../integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](../../extending/extending.html){.reference .internal}
    -   [Add a new Hardware
        Backend](../../extending/backend.html){.reference .internal}
        -   [Overview](../../extending/backend.html#overview){.reference
            .internal}
        -   [Server Helper
            Implementation](../../extending/backend.html#server-helper-implementation){.reference
            .internal}
            -   [Directory
                Structure](../../extending/backend.html#directory-structure){.reference
                .internal}
            -   [Server Helper
                Class](../../extending/backend.html#server-helper-class){.reference
                .internal}
            -   [[`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](../../extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](../../extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent [`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](../../extending/backend.html#update-parent-cmakelists-txt){.reference
                .internal}
        -   [Testing](../../extending/backend.html#testing){.reference
            .internal}
            -   [Unit
                Tests](../../extending/backend.html#unit-tests){.reference
                .internal}
            -   [Mock
                Server](../../extending/backend.html#mock-server){.reference
                .internal}
            -   [Python
                Tests](../../extending/backend.html#python-tests){.reference
                .internal}
            -   [Integration
                Tests](../../extending/backend.html#integration-tests){.reference
                .internal}
        -   [Documentation](../../extending/backend.html#documentation){.reference
            .internal}
        -   [Example
            Usage](../../extending/backend.html#example-usage){.reference
            .internal}
        -   [Code
            Review](../../extending/backend.html#code-review){.reference
            .internal}
        -   [Maintaining a
            Backend](../../extending/backend.html#maintaining-a-backend){.reference
            .internal}
        -   [Conclusion](../../extending/backend.html#conclusion){.reference
            .internal}
    -   [Create a new NVQIR
        Simulator](../../extending/nvqir_simulator.html){.reference
        .internal}
        -   [[`CircuitSimulator`{.code .docutils .literal
            .notranslate}]{.pre}](../../extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](../../extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
    -   [Working with CUDA-Q
        IR](../../extending/cudaq_ir.html){.reference .internal}
    -   [Create an MLIR Pass for
        CUDA-Q](../../extending/mlir_pass.html){.reference .internal}
-   [Specifications](../../../specification/index.html){.reference
    .internal}
    -   [Language
        Specification](../../../specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](../../../specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](../../../specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](../../../specification/cudaq/types.html){.reference
            .internal}
            -   [3.1. [`cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. [`cudaq::qubit`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](../../../specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](../../../specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. [`cudaq::spin_op`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](../../../specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on [`cudaq::qubit`{.code .docutils
                .literal
                .notranslate}]{.pre}](../../../specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](../../../specification/cudaq/kernels.html){.reference
            .internal}
        -   [7. Sub-circuit
            Synthesis](../../../specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](../../../specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](../../../specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](../../../specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](../../../specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11.
            Platform](../../../specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](../../../specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. [`cudaq::sample`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. [`cudaq::run`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. [`cudaq::observe`{.code .docutils .literal
                .notranslate}]{.pre}](../../../specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. [`cudaq::optimizer`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../../../specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. [`cudaq::gradient`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../../../specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](../../../specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](../../../specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](../../../specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](../../../specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](../../../specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](../../../specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](../../../specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](../../../specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](../../../specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](../../../specification/quake-dialect.html#motivation){.reference
            .internal}
-   [API Reference](../../../api/api.html){.reference .internal}
    -   [C++ API](../../../api/languages/cpp_api.html){.reference
        .internal}
        -   [Operators](../../../api/languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](../../../api/languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](../../../api/languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](../../../api/languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](../../../api/languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](../../../api/languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Platform](../../../api/languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](../../../api/languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](../../../api/languages/cpp_api.html#namespaces){.reference
            .internal}
    -   [Python API](../../../api/languages/python_api.html){.reference
        .internal}
        -   [Program
            Construction](../../../api/languages/python_api.html#program-construction){.reference
            .internal}
            -   [[`make_kernel()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [[`PyKernel`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [[`Kernel`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [[`PyKernelDecorator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [[`kernel()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](../../../api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [[`sample_async()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [[`run()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [[`run_async()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [[`observe()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [[`observe_async()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [[`get_state()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [[`get_state_async()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [[`vqe()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [[`draw()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [[`translate()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [[`estimate_resources()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
        -   [Backend
            Configuration](../../../api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [[`has_target()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [[`get_target()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [[`get_targets()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [[`set_target()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [[`reset_target()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [[`set_noise()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [[`unset_noise()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [[`register_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [[`unregister_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [[`cudaq.apply_noise()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [[`initialize_cudaq()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [[`num_available_gpus()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [[`set_random_seed()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](../../../api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [[`evolve()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [[`evolve_async()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [[`Schedule`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [[`BaseIntegrator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [[`InitialState`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [[`InitialStateType`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [[`IntermediateResultSave`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](../../../api/languages/python_api.html#operators){.reference
            .internal}
            -   [[`OperatorSum`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [[`ProductOperator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [[`ElementaryOperator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [[`ScalarOperator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [[`RydbergHamiltonian`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [[`SuperOperator`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [[`operators.define()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [[`operators.instantiate()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](../../../api/languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](../../../api/languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](../../../api/languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](../../../api/languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](../../../api/languages/python_api.html#data-types){.reference
            .internal}
            -   [[`SimulationPrecision`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [[`Target`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [[`State`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [[`Tensor`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [[`QuakeValue`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [[`qubit`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [[`qreg`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [[`qvector`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [[`ComplexMatrix`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [[`SampleResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [[`AsyncSampleResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [[`ObserveResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [[`AsyncObserveResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [[`AsyncStateResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [[`OptimizationResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [[`EvolveResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [[`AsyncEvolveResult`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [[`Resources`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.Resources){.reference
                .internal}
            -   [Optimizers](../../../api/languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](../../../api/languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](../../../api/languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](../../../api/languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [[`initialize()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [[`rank()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [[`num_ranks()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [[`all_gather()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [[`broadcast()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [[`is_initialized()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [[`finalize()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](../../../api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](../../../api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
    -   [Quantum Operations](../../../api/default_ops.html){.reference
        .internal}
        -   [Unitary Operations on
            Qubits](../../../api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [[`x`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#x){.reference
                .internal}
            -   [[`y`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#y){.reference
                .internal}
            -   [[`z`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#z){.reference
                .internal}
            -   [[`h`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#h){.reference
                .internal}
            -   [[`r1`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#r1){.reference
                .internal}
            -   [[`rx`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#rx){.reference
                .internal}
            -   [[`ry`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#ry){.reference
                .internal}
            -   [[`rz`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#rz){.reference
                .internal}
            -   [[`s`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#s){.reference
                .internal}
            -   [[`t`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#t){.reference
                .internal}
            -   [[`swap`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#swap){.reference
                .internal}
            -   [[`u3`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#u3){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](../../../api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](../../../api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#mz){.reference
                .internal}
            -   [[`mx`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#mx){.reference
                .internal}
            -   [[`my`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](../../../api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](../../../api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [[`create`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#create){.reference
                .internal}
            -   [[`annihilate`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#annihilate){.reference
                .internal}
            -   [[`phase_shift`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#phase-shift){.reference
                .internal}
            -   [[`beam_splitter`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](../../../api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](../../../versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](../../../index.html)

::: wy-nav-content
::: rst-content
::: {role="navigation" aria-label="Page navigation"}
-   [](../../../index.html){.icon .icon-home aria-label="Home"}
-   [CUDA-Q Backends](../backends.html)
-   [CUDA-Q Circuit Simulation Backends](../simulators.html)
-   State Vector Simulators
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](../simulators.html "CUDA-Q Circuit Simulation Backends"){.btn
.btn-neutral .float-left accesskey="p"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](tnsims.html "Tensor Network Simulators"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#state-vector-simulators .section}
# State Vector Simulators[¶](#state-vector-simulators "Permalink to this heading"){.headerlink}

::: {#cpu .section}
## CPU[¶](#cpu "Permalink to this heading"){.headerlink}

[]{#openmp-cpu-only}The [`qpp-cpu`{.code .docutils .literal
.notranslate}]{.pre} backend backend provides a state vector simulator
based on the CPU-only, OpenMP threaded
[Q++](https://github.com/softwareqinc/qpp){.reference .external}
library. This backend is good for basic testing and experimentation with
just a few qubits, but performs poorly for all but the smallest
simulation and is the default target when running on CPU-only systems.

To execute a program on the [`qpp-cpu`{.code .docutils .literal
.notranslate}]{.pre} target even if a GPU-accelerated backend is
available,

use the following commands:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-bash .notranslate}
::: highlight
    python3 program.py [...] --target qpp-cpu
:::
:::

The target can also be defined in the application code by calling

::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target('qpp-cpu')
:::
:::

If a target is set in the application code, this target will override
the [`--target`{.code .docutils .literal .notranslate}]{.pre} command
line flag given during program invocation.
:::

C++

::: {.tab-content .docutils}
::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target qpp-cpu program.cpp [...] -o program.x
    ./program.x
:::
:::
:::
:::
:::

::: {#single-gpu .section}
## Single-GPU[¶](#single-gpu "Permalink to this heading"){.headerlink}

[]{#default-simulator}[]{#cuquantum-single-gpu}The [`nvidia`{.code
.docutils .literal .notranslate}]{.pre} backend provides a state vector
simulator accelerated with - the [`cuStateVec`{.code .docutils .literal
.notranslate}]{.pre} library. The [cuStateVec
documentation](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html){.reference
.external} provides a detailed explanation for how the simulations are
performed on the GPU.

The [`nvidia`{.code .docutils .literal .notranslate}]{.pre} target
supports multiple configurable options including specification of
floating point precision.

To execute a program on the [`nvidia`{.code .docutils .literal
.notranslate}]{.pre} backend, use the following commands:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
Single Precision (Default):

::: {.highlight-bash .notranslate}
::: highlight
    python3 program.py [...] --target nvidia --target-option fp32
:::
:::

Double Precision:

::: {.highlight-bash .notranslate}
::: highlight
    python3 program.py [...] --target nvidia --target-option fp64
:::
:::

The target can also be defined in the application code by calling

::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target('nvidia', option = 'fp64')
:::
:::

If a target is set in the application code, this target will override
the [`--target`{.code .docutils .literal .notranslate}]{.pre} command
line flag given during program invocation.
:::

C++

::: {.tab-content .docutils}
Single Precision (Default):

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target nvidia --target-option fp32 program.cpp [...] -o program.x
    ./program.x
:::
:::

Double Precision (Default):

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target nvidia --target-option fp64 program.cpp [...] -o program.x
    ./program.x
:::
:::
:::
:::

::: {.admonition .note}
Note

This backend requires an NVIDIA GPU and CUDA runtime libraries. If you
do not have these dependencies installed, you may encounter an error
stating [`Invalid`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`simulator`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`requested`{.code .docutils .literal .notranslate}]{.pre}.
See the section [[Dependencies and Compatibility]{.std
.std-ref}](../../install/local_installation.html#dependencies-and-compatibility){.reference
.internal} for more information about how to install dependencies.
:::

In the single-GPU mode, the [`nvidia`{.code .docutils .literal
.notranslate}]{.pre} backend provides the following environment variable
options. Any environment variables must be set prior to setting the
target or running "[`import`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`cudaq`{.code .docutils .literal .notranslate}]{.pre}". It
is worth drawing attention to gate fusion, a powerful tool for improving
simulation performance which is discussed in greater detail
[here](https://nvidia.github.io/cuda-quantum/latest/examples/python/performance_optimizations.html){.reference
.external}.

+-------------+--------------------+-----------------------------------+
| Option      | Value              | Description                       |
+-------------+--------------------+-----------------------------------+
| [`C         | positive integer   | The max number of qubits used for |
| UDAQ_FUSION |                    | gate fusion. The default value    |
| _MAX_QUBITS |                    | depends on [GPU Compute           |
| `{.docutils |                    | Capability](https://developer     |
| .literal    |                    | .nvidia.com/cuda-gpus){.reference |
| .notransl   |                    | .external} (CC) and the floating  |
| ate}]{.pre} |                    | point precision selected for the  |
|             |                    | simulator as specified            |
|             |                    | [[here]{.std                      |
|             |                    | .std-ref                          |
|             |                    | }](#gate-fusion-table){.reference |
|             |                    | .internal}.                       |
+-------------+--------------------+-----------------------------------+
| [`CUDA      | integer greater    | The max number of qubits used for |
| Q_FUSION_DI | than or equal to   | diagonal gate fusion. The default |
| AGONAL_GATE | -1                 | value is set to [`-1`{.code       |
| _MAX_QUBITS |                    | .docutils .literal                |
| `{.docutils |                    | .notranslate}]{.pre} and the      |
| .literal    |                    | fusion size will be automatically |
| .notransl   |                    | adjusted for the better           |
| ate}]{.pre} |                    | performance. If 0, the gate       |
|             |                    | fusion for diagonal gates is      |
|             |                    | disabled.                         |
+-------------+--------------------+-----------------------------------+
| [`CUDAQ_F   | positive integer   | Number of CPU threads used for    |
| USION_NUM_H |                    | circuit processing. The default   |
| OST_THREADS |                    | value is [`8`{.code .docutils     |
| `{.docutils |                    | .literal .notranslate}]{.pre}.    |
| .literal    |                    |                                   |
| .notransl   |                    |                                   |
| ate}]{.pre} |                    |                                   |
+-------------+--------------------+-----------------------------------+
| [`C         | non-negative       | CPU memory size (in GB) allowed   |
| UDAQ_MAX_CP | integer, or        | for state-vector migration.       |
| U_MEMORY_GB | [`NONE`{.code      | [`NONE`{.code .docutils .literal  |
| `{.docutils | .docutils .literal | .notranslate}]{.pre} means        |
| .literal    | .n                 | unlimited (up to physical memory  |
| .notransl   | otranslate}]{.pre} | constraints). Default is 0GB      |
| ate}]{.pre} |                    | (disabled, variable is not set to |
|             |                    | any value).                       |
+-------------+--------------------+-----------------------------------+
| [`C         | positive integer,  | GPU memory (in GB) allowed for    |
| UDAQ_MAX_GP | or [`NONE`{.code   | on-device state-vector            |
| U_MEMORY_GB | .docutils .literal | allocation. As the state-vector   |
| `{.docutils | .n                 | size exceeds this limit, host     |
| .literal    | otranslate}]{.pre} | memory will be utilized for       |
| .notransl   |                    | migration. [`NONE`{.code          |
| ate}]{.pre} |                    | .docutils .literal                |
|             |                    | .notranslate}]{.pre} means        |
|             |                    | unlimited (up to physical memory  |
|             |                    | constraints). This is the         |
|             |                    | default.                          |
+-------------+--------------------+-----------------------------------+
| [`CUD       | [`TRUE`{.code      | \[Blackwell (compute capability   |
| AQ_ALLOW_FP | .docutils .literal | 10.0+) only\] Enable or disable   |
| 32_EMULATED | .n                 | floating point math emulation. If |
| `{.docutils | otranslate}]{.pre} | enabled, allows [`FP32`{.code     |
| .literal    | ([`1`{.code        | .docutils .literal                |
| .notransl   | .docutils .literal | .notranslate}]{.pre} emulation    |
| ate}]{.pre} | .no                | kernels using [`BFloat16`{.code   |
|             | translate}]{.pre}, | .docutils .literal                |
|             | [`ON`{.code        | .notranslate}]{.pre}              |
|             | .docutils .literal | ([`BF16`{.code .docutils .literal |
|             | .no                | .notranslate}]{.pre}) whenever    |
|             | translate}]{.pre}) | possible. Enabled by default.     |
|             | or [`FALSE`{.code  |                                   |
|             | .docutils .literal |                                   |
|             | .n                 |                                   |
|             | otranslate}]{.pre} |                                   |
|             | ([`0`{.code        |                                   |
|             | .docutils .literal |                                   |
|             | .no                |                                   |
|             | translate}]{.pre}, |                                   |
|             | [`OFF`{.code       |                                   |
|             | .docutils .literal |                                   |
|             | .no                |                                   |
|             | translate}]{.pre}) |                                   |
+-------------+--------------------+-----------------------------------+
| [`CUDAQ_ENA | [`TRUE`{.code      | Enable or disable [CUDA memory    |
| BLE_MEMPOOL | .docutils .literal | pool](https://de                  |
| `{.docutils | .n                 | veloper.nvidia.com/blog/using-cud |
| .literal    | otranslate}]{.pre} | a-stream-ordered-memory-allocator |
| .notransl   | ([`1`{.code        | -part-1/#memory_pools){.reference |
| ate}]{.pre} | .docutils .literal | .external} for state vector       |
|             | .no                | allocation/deallocation. Enabled  |
|             | translate}]{.pre}, | by default.                       |
|             | [`ON`{.code        |                                   |
|             | .docutils .literal |                                   |
|             | .no                |                                   |
|             | translate}]{.pre}) |                                   |
|             | or [`FALSE`{.code  |                                   |
|             | .docutils .literal |                                   |
|             | .n                 |                                   |
|             | otranslate}]{.pre} |                                   |
|             | ([`0`{.code        |                                   |
|             | .docutils .literal |                                   |
|             | .no                |                                   |
|             | translate}]{.pre}, |                                   |
|             | [`OFF`{.code       |                                   |
|             | .docutils .literal |                                   |
|             | .no                |                                   |
|             | translate}]{.pre}) |                                   |
+-------------+--------------------+-----------------------------------+

: [**Environment variable options supported in single-GPU
mode**]{.caption-text}[¶](#id1 "Permalink to this table"){.headerlink}

::: deprecated
[Deprecated since version 0.8: ]{.versionmodified .deprecated}The
[`nvidia-fp64`{.code .docutils .literal .notranslate}]{.pre} targets,
which is equivalent setting the [`fp64`{.code .docutils .literal
.notranslate}]{.pre} option on the [`nvidia`{.code .docutils .literal
.notranslate}]{.pre} target, is deprecated and will be removed in a
future release.
:::

::: {.admonition .note}
Note

In host-device simulation, [`CUDAQ_MAX_CPU_MEMORY_GB`{.code .docutils
.literal .notranslate}]{.pre} is not 0, the backend automatically
switching between inner product (default) and operator matrix-based
methods for expectation calculations ([`cudaq::observe`{.code .docutils
.literal .notranslate}]{.pre}) depending on whether a clone of the state
can be allocated or not.

For example, when [`CUDAQ_MAX_GPU_MEMORY_GB`{.code .docutils .literal
.notranslate}]{.pre} is unconstrained, the quantum state vector would
consume all device memory before utilizing host memory. Thus, the
backend would fall back to the operator matrix-based approach as cloning
the state is not possible. For performance reason, only Pauli operator
matrices of up to 8 qubits (identity padding not included) are allowed
in this mode. This constrain can be relaxed by setting the
[`CUDAQ_MATRIX_EXP_VAL_MAX_SIZE`{.code .docutils .literal
.notranslate}]{.pre} environment variable. Users would need to take into
account the full operator matrix size when increasing this setting.
:::
:::

::: {#multi-gpu-multi-node .section}
## Multi-GPU multi-node[¶](#multi-gpu-multi-node "Permalink to this heading"){.headerlink}

The [`nvidia`{.code .docutils .literal .notranslate}]{.pre} backend also
provides a state vector simulator accelerated with the
[`cuStateVec`{.code .docutils .literal .notranslate}]{.pre} library with
support for Multi-GPU, Multi-node distribution of the state vector.

This backend is necessary to scale applications that require a state
vector that cannot fit on a single GPU memory.

The multi-node multi-GPU simulator expects to run within an MPI context.
To execute a program on the multi-node multi-GPU NVIDIA target, use the
following commands (adjust the value of the [`-np`{.code .docutils
.literal .notranslate}]{.pre} flag as needed to reflect available GPU
resources on your system):

See the [Divisive
Clustering](https://nvidia.github.io/cuda-quantum/latest/applications/python/divisive_clustering_coresets.html){.reference
.external} application to see how this backend can be used in practice.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
Double precision simulation:

::: {.highlight-bash .notranslate}
::: highlight
    mpiexec -np 2 python3 program.py [...] --target nvidia --target-option fp64,mgpu
:::
:::

Single precision simulation:

::: {.highlight-bash .notranslate}
::: highlight
    mpiexec -np 2 python3 program.py [...] --target nvidia --target-option fp32,mgpu
:::
:::

::: {.admonition .note}
Note

If you installed CUDA-Q via [`pip`{.code .docutils .literal
.notranslate}]{.pre}, you will need to install the necessary MPI
dependencies separately; please follow the instructions for installing
dependencies in the [Project
Description](https://pypi.org/project/cuda-quantum/#description){.reference
.external}.
:::

In addition to using MPI in the simulator, you can use it in your
application code by installing
[mpi4py](https://mpi4py.readthedocs.io/){.reference .external}, and
invoking the program with the command

::: {.highlight-bash .notranslate}
::: highlight
    mpiexec -np 2 python3 -m mpi4py program.py [...] --target nvidia --target-option fp64,mgpu
:::
:::

The target can also be defined in the application code by calling

::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target('nvidia', option='mgpu,fp64')
:::
:::

If a target is set in the application code, this target will override
the [`--target`{.code .docutils .literal .notranslate}]{.pre} command
line flag given during program invocation.

::: {.admonition .note}
Note

-   The order of the option settings are interchangeable. For example,
    [`cudaq.set_target('nvidia',`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`option='mgpu,fp64')`{.code .docutils .literal
    .notranslate}]{.pre} is equivalent to
    [`cudaq.set_target('nvidia',`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`option='fp64,mgpu')`{.code .docutils .literal
    .notranslate}]{.pre}.

-   The [`nvidia`{.code .docutils .literal .notranslate}]{.pre} target
    has single-precision as the default setting. Thus, using
    [`option='mgpu'`{.code .docutils .literal .notranslate}]{.pre}
    implies that [`option='mgpu,fp32'`{.code .docutils .literal
    .notranslate}]{.pre}.
:::
:::

C++

::: {.tab-content .docutils}
Double precision simulation:

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target nvidia  --target-option mgpu,fp64 program.cpp [...] -o program.x
    mpiexec -np 2 ./program.x
:::
:::

Single precision simulation:

::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target nvidia  --target-option mgpu,fp32 program.cpp [...] -o program.x
    mpiexec -np 2 ./program.x
:::
:::
:::
:::

::: {.admonition .note}
Note

This backend requires an NVIDIA GPU, CUDA runtime libraries, as well as
an MPI installation. If you do not have these dependencies installed,
you may encounter either an error stating [`invalid`{.code .docutils
.literal .notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`simulator`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`requested`{.code .docutils .literal .notranslate}]{.pre}
(missing CUDA libraries), or an error along the lines of [`failed`{.code
.docutils .literal .notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`to`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`launch`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`kernel`{.code .docutils .literal .notranslate}]{.pre}
(missing MPI installation). See the section [[Dependencies and
Compatibility]{.std
.std-ref}](../../install/local_installation.html#dependencies-and-compatibility){.reference
.internal} for more information about how to install dependencies.

The number of processes and nodes should be always power-of-2.

Host-device state vector migration is also supported in the multi-GPU
multi-node configuration.
:::

In addition to those environment variable options supported in the
single-GPU mode, the [`nvidia`{.code .docutils .literal
.notranslate}]{.pre} backend provides the following environment variable
options particularly for the multi-node multi-GPU configuration. Any
environment variables must be set prior to setting the target or running
"[`import`{.code .docutils .literal .notranslate}]{.pre}` `{.code
.docutils .literal .notranslate}[`cudaq`{.code .docutils .literal
.notranslate}]{.pre}".

+-------------+--------------------+-----------------------------------+
| Option      | Value              | Description                       |
+-------------+--------------------+-----------------------------------+
| [`CUDAQ_M   | string             | The shared library name for       |
| GPU_LIB_MPI |                    | inter-process communication. The  |
| `{.docutils |                    | default value is                  |
| .literal    |                    | [`libmpi.so`{.code .docutils      |
| .notransl   |                    | .literal .notranslate}]{.pre}.    |
| ate}]{.pre} |                    |                                   |
+-------------+--------------------+-----------------------------------+
| [`CUDAQ     | [`AUTO`{.code      | Selecting [`cuStateVec`{.code     |
| _MGPU_COMM_ | .docutils .literal | .docutils .literal                |
| PLUGIN_TYPE | .no                | .notranslate}]{.pre}              |
| `{.docutils | translate}]{.pre}, | [`CommPlugin`{.code .docutils     |
| .literal    | [`EXTERNAL`{.code  | .literal .notranslate}]{.pre} for |
| .notransl   | .docutils .literal | inter-process communication. The  |
| ate}]{.pre} | .no                | default is [`AUTO`{.code          |
|             | translate}]{.pre}, | .docutils .literal                |
|             | [`OpenMPI`{.code   | .notranslate}]{.pre}. If          |
|             | .docutils .literal | [`EXTERNAL`{.code .docutils       |
|             | .no                | .literal .notranslate}]{.pre} is  |
|             | translate}]{.pre}, | selected,                         |
|             | or [`MPICH`{.code  | [`CUDAQ_MGPU_LIB_MPI`{.code       |
|             | .docutils .literal | .docutils .literal                |
|             | .n                 | .notranslate}]{.pre} should point |
|             | otranslate}]{.pre} | to an implementation of           |
|             |                    | [`cuStateVec`{.code .docutils     |
|             |                    | .literal .notranslate}]{.pre}     |
|             |                    | [`CommPlugin`{.code .docutils     |
|             |                    | .literal .notranslate}]{.pre}     |
|             |                    | interface.                        |
+-------------+--------------------+-----------------------------------+
| [`CUD       | positive integer   | The qubit count threshold where   |
| AQ_MGPU_NQU |                    | state vector distribution is      |
| BITS_THRESH |                    | activated. Below this threshold,  |
| `{.docutils |                    | simulation is performed as        |
| .literal    |                    | independent (non-distributed)     |
| .notransl   |                    | tasks across all MPI processes    |
| ate}]{.pre} |                    | for optimal performance. Default  |
|             |                    | is 25.                            |
+-------------+--------------------+-----------------------------------+
| [`CUDA      | positive integer   | The max number of qubits used for |
| Q_MGPU_FUSE |                    | gate fusion. The default value    |
| `{.docutils |                    | depends on [GPU Compute           |
| .literal    |                    | Capability](https://developer     |
| .notransl   |                    | .nvidia.com/cuda-gpus){.reference |
| ate}]{.pre} |                    | .external} (CC) and the floating  |
|             |                    | point precision selected for the  |
|             |                    | simulator as specified            |
|             |                    | [[here]{.std                      |
|             |                    | .std-ref                          |
|             |                    | }](#gate-fusion-table){.reference |
|             |                    | .internal}.                       |
+-------------+--------------------+-----------------------------------+
| [`CUDA      | positive integer   | Specify the number of GPUs that   |
| Q_MGPU_P2P_ |                    | can communicate by using          |
| DEVICE_BITS |                    | GPUDirect P2P. Default value is 0 |
| `{.docutils |                    | (P2P communication is disabled).  |
| .literal    |                    |                                   |
| .notransl   |                    |                                   |
| ate}]{.pre} |                    |                                   |
+-------------+--------------------+-----------------------------------+
| [`CUDAQ     | [`MNNVL`{.code     | Automatically set the number of   |
| _GPU_FABRIC | .docutils .literal | P2P device bits based on the      |
| `{.docutils | .no                | total number of processes when    |
| .literal    | translate}]{.pre}, | multi-node NVLink ([`MNNVL`{.code |
| .notransl   | [`NVL`{.code       | .docutils .literal                |
| ate}]{.pre} | .docutils .literal | .notranslate}]{.pre}) is          |
|             | .no                | selected; or the number of        |
|             | translate}]{.pre}, | processes per node when NVLink    |
|             | [`NONE`{.code      | ([`NVL`{.code .docutils .literal  |
|             | .docutils .literal | .notranslate}]{.pre}) is          |
|             | .no                | selected; or disable P2P (with    |
|             | translate}]{.pre}, | [`NONE`{.code .docutils .literal  |
|             | or NVLink domain   | .notranslate}]{.pre}); or a       |
|             | size (power of 2   | specific NVLink domain size.      |
|             | integer)           |                                   |
+-------------+--------------------+-----------------------------------+
| [`C         | comma-separated    | Specify the network structure     |
| UDAQ_GLOBAL | list of positive   | (faster to slower). For example,  |
| _INDEX_BITS | integers           | assuming a 32 MPI processes       |
| `{.docutils |                    | simulation, whereby the network   |
| .literal    |                    | topology is divided into 4 groups |
| .notransl   |                    | of 8 processes, which have faster |
| ate}]{.pre} |                    | communication network between     |
|             |                    | them. In this case, the           |
|             |                    | [`CUDAQ_GLOBAL_INDEX_BITS`{.code  |
|             |                    | .docutils .literal                |
|             |                    | .notranslate}]{.pre} environment  |
|             |                    | variable can be set to            |
|             |                    | [`3,2`{.code .docutils .literal   |
|             |                    | .notranslate}]{.pre}. The first   |
|             |                    | [`3`{.code .docutils .literal     |
|             |                    | .notranslate}]{.pre}              |
|             |                    | ([`log2(8)`{.code .docutils       |
|             |                    | .literal .notranslate}]{.pre})    |
|             |                    | represents **8** processes with   |
|             |                    | fast communication within the     |
|             |                    | group and the second [`2`{.code   |
|             |                    | .docutils .literal                |
|             |                    | .notranslate}]{.pre} represents   |
|             |                    | the **4** groups (8 processes     |
|             |                    | each) in those total 32           |
|             |                    | processes. The sum of all         |
|             |                    | elements in this list is          |
|             |                    | [`5`{.code .docutils .literal     |
|             |                    | .notranslate}]{.pre},             |
|             |                    | corresponding to the total number |
|             |                    | of MPI processes ([`2^5`{.code    |
|             |                    | .docutils .literal                |
|             |                    | .notranslate}]{.pre}` `{.code     |
|             |                    | .docutils .literal                |
|             |                    | .notranslate}[`=`{.code .docutils |
|             |                    | .literal                          |
|             |                    | .notranslate}]{.pre}` `{.code     |
|             |                    | .docutils .literal                |
|             |                    | .notranslate}[`32`{.code          |
|             |                    | .docutils .literal                |
|             |                    | .notranslate}]{.pre}). If none    |
|             |                    | specified, the global index bits  |
|             |                    | are set based on P2P device bits. |
+-------------+--------------------+-----------------------------------+
| [`          | positive integer   | Specify host-device memory        |
| CUDAQ_HOST_ |                    | migration w.r.t. the network      |
| DEVICE_MIGR |                    | structure. If provided, this      |
| ATION_LEVEL |                    | setting determines the position   |
| `{.docutils |                    | to insert the number of migration |
| .literal    |                    | index bits to the                 |
| .notransl   |                    | [`CUDAQ_GLOBAL_INDEX_BITS`{.code  |
| ate}]{.pre} |                    | .docutils .literal                |
|             |                    | .notranslate}]{.pre} list. By     |
|             |                    | default, if not set, the number   |
|             |                    | of migration index bits (CPU-GPU  |
|             |                    | data transfers) is appended to    |
|             |                    | the end of the array of index     |
|             |                    | bits (aka, state vector           |
|             |                    | distribution scheme). This        |
|             |                    | default behavior is optimized for |
|             |                    | systems with fast GPU-GPU         |
|             |                    | interconnects (NVLink,            |
|             |                    | InfiniBand, etc.)                 |
+-------------+--------------------+-----------------------------------+
| [`CUDAQ_DAT | positive integer   | Specify the temporary buffer size |
| A_TRANSFER_ | greater than or    | ([`1`{.code .docutils .literal    |
| BUFFER_BITS | equal to 24        | .notranslate}]{.pre}` `{.code     |
| `{.docutils |                    | .docutils .literal                |
| .literal    |                    | .notranslate}[`<<`{.code          |
| .notransl   |                    | .docutils .literal                |
| ate}]{.pre} |                    | .notranslate}]{.pre}` `{.code     |
|             |                    | .docutils .literal                |
|             |                    | .notranslate}[`CUDAQ              |
|             |                    | _DATA_TRANSFER_BUFFER_BITS`{.code |
|             |                    | .docutils .literal                |
|             |                    | .notranslate}]{.pre} bytes) for   |
|             |                    | inter-node data transfer. The     |
|             |                    | default is set to 26 (64 MB). The |
|             |                    | minimum allowed value is 24 (16   |
|             |                    | MB). Depending on systems,        |
|             |                    | setting a larger value to         |
|             |                    | [`CUDAQ                           |
|             |                    | _DATA_TRANSFER_BUFFER_BITS`{.code |
|             |                    | .docutils .literal                |
|             |                    | .notranslate}]{.pre} can          |
|             |                    | accelerate inter-node data        |
|             |                    | transfers.                        |
+-------------+--------------------+-----------------------------------+

: [**Additional environment variable options for multi-node multi-GPU
mode**]{.caption-text}[¶](#id2 "Permalink to this table"){.headerlink}

::: deprecated
[Deprecated since version 0.8: ]{.versionmodified .deprecated}The
[`nvidia-mgpu`{.code .docutils .literal .notranslate}]{.pre} backend,
which is equivalent to the multi-node multi-GPU double-precision option
([`mgpu,fp64`{.code .docutils .literal .notranslate}]{.pre}) of the
[`nvidia`{.code .docutils .literal .notranslate}]{.pre} is deprecated
and will be removed in a future release.
:::

[]{#gate-fusion-table}

+-------------+--------------------+-----------------------------------+
| Compute     | GPU                | Default Gate Fusion Size          |
| Capability  |                    |                                   |
+-------------+--------------------+-----------------------------------+
| 8.0         | NVIDIA A100        | 4 ([`fp32`{.code .docutils        |
|             |                    | .literal .notranslate}]{.pre}) or |
|             |                    | 5 ([`fp64`{.code .docutils        |
|             |                    | .literal .notranslate}]{.pre})    |
+-------------+--------------------+-----------------------------------+
| 9.0         | NVIDIA H100, H200, | 5 ([`fp32`{.code .docutils        |
|             | GH200              | .literal .notranslate}]{.pre}) or |
|             |                    | 6 ([`fp64`{.code .docutils        |
|             |                    | .literal .notranslate}]{.pre})    |
+-------------+--------------------+-----------------------------------+
| 10.0        | NVIDIA GB200, B200 | 5 ([`fp32`{.code .docutils        |
|             |                    | .literal .notranslate}]{.pre}) or |
|             |                    | 4 ([`fp64`{.code .docutils        |
|             |                    | .literal .notranslate}]{.pre})    |
+-------------+--------------------+-----------------------------------+
| 10.3        | NVIDIA B300        | 5 ([`fp32`{.code .docutils        |
|             |                    | .literal .notranslate}]{.pre}) or |
|             |                    | 1 ([`fp64`{.code .docutils        |
|             |                    | .literal .notranslate}]{.pre})    |
+-------------+--------------------+-----------------------------------+
| Others      |                    | 4 ([`fp32`{.code .docutils        |
|             |                    | .literal .notranslate}]{.pre} and |
|             |                    | [`fp64`{.code .docutils .literal  |
|             |                    | .notranslate}]{.pre})             |
+-------------+--------------------+-----------------------------------+

: [**Default Gate Fusion
Size**]{.caption-text}[¶](#id3 "Permalink to this table"){.headerlink}

The above configuration options of the [`nvidia`{.code .docutils
.literal .notranslate}]{.pre} backend can be tuned to reduce your
simulation runtimes. One of the performance improvements is to fuse
multiple gates together during runtime. For example, [`x(qubit0)`{.code
.docutils .literal .notranslate}]{.pre} and [`x(qubit1)`{.code .docutils
.literal .notranslate}]{.pre} can be fused together into a single 4x4
matrix operation on the state vector rather than 2 separate 2x2 matrix
operations on the state vector. This fusion reduces memory bandwidth on
the GPU because the state vector is transferred into and out of memory
fewer times. By default, up to 4 gates are fused together for single-GPU
simulations, and up to 6 gates are fused together for multi-GPU
simulations. The number of gates fused can **significantly** affect
performance of some circuits, so users can override the default fusion
level by setting the setting [`CUDAQ_MGPU_FUSE`{.code .docutils .literal
.notranslate}]{.pre} environment variable to another integer value as
shown below.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-bash .notranslate}
::: highlight
    CUDAQ_MGPU_FUSE=5 mpiexec -np 2 python3 program.py [...] --target nvidia --target-option mgpu,fp64
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-bash .notranslate}
::: highlight
    nvq++ --target nvidia --target-option mgpu,fp64 program.cpp [...] -o program.x
    CUDAQ_MGPU_FUSE=5 mpiexec -np 2 ./program.x
:::
:::
:::
:::

::: {.admonition .note}
Note

On multi-node systems without [`MNNVL`{.code .docutils .literal
.notranslate}]{.pre} support, the [`nvidia`{.code .docutils .literal
.notranslate}]{.pre} target in [`mgpu`{.code .docutils .literal
.notranslate}]{.pre} mode may fail to allocate memory. Users can disable
[`MNNVL`{.code .docutils .literal .notranslate}]{.pre} fabric-based
memory sharing by setting the environment variable
[`UBACKEND_USE_FABRIC_HANDLE=0`{.code .docutils .literal
.notranslate}]{.pre}.
:::
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](../simulators.html "CUDA-Q Circuit Simulation Backends"){.btn
.btn-neutral .float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](tnsims.html "Tensor Network Simulators"){.btn
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
