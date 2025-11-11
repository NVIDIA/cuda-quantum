::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../../index.html){.icon .icon-home}

::: version
pr-3592
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
        -   [State Vector Simulators](../sims/svsims.html){.reference
            .internal}
            -   [CPU](../sims/svsims.html#cpu){.reference .internal}
            -   [Single-GPU](../sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](../sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network Simulators](../sims/tnsims.html){.reference
            .internal}
            -   [Multi-GPU
                multi-node](../sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](../sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](../sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU Simulators](../sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](../sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](../sims/mqpusims.html#multi-qpu-other-backends){.reference
                .internal}
        -   [Noisy Simulators](../sims/noisy.html){.reference .internal}
            -   [Trajectory Noisy
                Simulation](../sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](../sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](../sims/noisy.html#stim){.reference .internal}
        -   [Photonics Simulators](../sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](../sims/photonics.html#orca-photonics){.reference
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
        -   [Amazon Braket (braket)](braket.html){.reference .internal}
            -   [Setting
                Credentials](braket.html#setting-credentials){.reference
                .internal}
            -   [Submission from
                C++](braket.html#submission-from-c){.reference
                .internal}
            -   [Submission from
                Python](braket.html#submission-from-python){.reference
                .internal}
        -   [NVIDIA Quantum Cloud (nvqc)](#){.current .reference
            .internal}
            -   [Quick Start](#quick-start){.reference .internal}
            -   [Simulator Backend
                Selection](#simulator-backend-selection){.reference
                .internal}
            -   [Multiple GPUs](#multiple-gpus){.reference .internal}
            -   [Multiple QPUs Asynchronous
                Execution](#multiple-qpus-asynchronous-execution){.reference
                .internal}
            -   [FAQ](#faq){.reference .internal}
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
-   [CUDA-Q Cloud Backends](../cloud.html)
-   NVIDIA Quantum Cloud
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](braket.html "Amazon Braket"){.btn .btn-neutral .float-left
accesskey="p"} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](../../dynamics.html "Dynamics Simulation"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#nvidia-quantum-cloud .section}
# NVIDIA Quantum Cloud[¶](#nvidia-quantum-cloud "Permalink to this heading"){.headerlink}

NVIDIA Quantum Cloud (NVQC) offers universal access to the world's most
powerful computing platform, for every quantum researcher to do their
life's work. To learn more about NVQC visit this
[link](https://www.nvidia.com/en-us/solutions/quantum-computing/cloud){.reference
.external}.

Apply for early access
[here](https://developer.nvidia.com/quantum-cloud-early-access-join){.reference
.external}. Access to the Quantum Cloud early access program requires an
NVIDIA Developer account.

::: {#quick-start .section}
## Quick Start[¶](#quick-start "Permalink to this heading"){.headerlink}

Once you have been approved for an early access to NVQC, you will be
able to follow these instructions to use it.

1\. Follow the instructions in your NVQC Early Access welcome email to
obtain an API Key for NVQC. You can also find the instructions
[here](https://developer.nvidia.com/quantum-cloud-early-access-members){.reference
.external} (link available only for approved users)

2.  Set the environment variable [`NVQC_API_KEY`{.code .docutils
    .literal .notranslate}]{.pre} to the API Key obtained above.

> <div>
>
> ::: {.highlight-console .notranslate}
> ::: highlight
>     export NVQC_API_KEY=<your NVQC API key>
> :::
> :::
>
> </div>

You may wish to persist that environment variable between bash sessions,
e.g., by adding it to your [`$HOME/.bashrc`{.code .docutils .literal
.notranslate}]{.pre} file.

3.  Run your first NVQC example

The following is a typical CUDA-Q kernel example. By selecting the
[`nvqc`{.code .docutils .literal .notranslate}]{.pre} target, the
quantum circuit simulation will run on NVQC in the cloud, rather than
running locally.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq

    cudaq.set_target("nvqc")
    num_qubits = 25
    # Define a simple quantum kernel to execute on NVQC.
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(num_qubits)
    # Maximally entangled state between 25 qubits.
    kernel.h(qubits[0])
    for i in range(num_qubits - 1):
        kernel.cx(qubits[i], qubits[i + 1])
    kernel.mz(qubits)

    counts = cudaq.sample(kernel)
    print(counts)
:::
:::

::: {.highlight-console .notranslate}
::: highlight
    [2024-03-14 19:26:31.438] Submitting jobs to NVQC service with 1 GPU(s). Max execution time: 3600 seconds (excluding queue wait time).

    ================ NVQC Device Info ================
    GPU Device Name: "NVIDIA H100 80GB HBM3"
    CUDA Driver Version / Runtime Version: 12.2 / 12.0
    Total global memory (GB): 79.1
    Memory Clock Rate (MHz): 2619.000
    GPU Clock Rate (MHz): 1980.000
    ==================================================
    { 1111111111111111111111111:486 0000000000000000000000000:514 }
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    #include <cudaq.h>

    // Define a simple quantum kernel to execute on NVQC.
    struct ghz {
      // Maximally entangled state between 25 qubits.
      auto operator()() __qpu__ {
        constexpr int NUM_QUBITS = 25;
        cudaq::qvector q(NUM_QUBITS);
        h(q[0]);
        for (int i = 0; i < NUM_QUBITS - 1; i++) {
          x<cudaq::ctrl>(q[i], q[i + 1]);
        }
        auto result = mz(q);
      }
    };

    int main() {
      auto counts = cudaq::sample(ghz{});
      counts.dump();
    }
:::
:::

The code above is saved in [`nvqc_intro.cpp`{.code .docutils .literal
.notranslate}]{.pre} and compiled with the following command, targeting
the [`nvqc`{.code .docutils .literal .notranslate}]{.pre} platform

::: {.highlight-console .notranslate}
::: highlight
    nvq++ nvqc_intro.cpp -o nvqc_intro.x --target nvqc
    ./nvqc_intro.x

    [2024-03-14 19:25:05.545] Submitting jobs to NVQC service with 1 GPU(s). Max execution time: 3600 seconds (excluding queue wait time).

    ================ NVQC Device Info ================
    GPU Device Name: "NVIDIA H100 80GB HBM3"
    CUDA Driver Version / Runtime Version: 12.2 / 12.0
    Total global memory (GB): 79.1
    Memory Clock Rate (MHz): 2619.000
    GPU Clock Rate (MHz): 1980.000
    ==================================================
    {
    __global__ : { 1111111111111111111111111:487 0000000000000000000000000:513 }
    result : { 1111111111111111111111111:487 0000000000000000000000000:513 }
    }
:::
:::
:::
:::
:::

::: {#simulator-backend-selection .section}
## Simulator Backend Selection[¶](#simulator-backend-selection "Permalink to this heading"){.headerlink}

NVQC hosts all CUDA-Q simulator backends (see [[simulators]{.std
.std-ref}](../simulators.html#simulators){.reference .internal}). You
may use the NVQC [`backend`{.code .docutils .literal
.notranslate}]{.pre} (Python) or [`--nvqc-backend`{.code .docutils
.literal .notranslate}]{.pre} (C++) option to select the simulator to be
used by the service.

For example, to request the [`tensornet`{.code .docutils .literal
.notranslate}]{.pre} simulator backend, the user can do the following
for C++ or Python.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target("nvqc", backend="tensornet")
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-console .notranslate}
::: highlight
    nvq++ nvqc_sample.cpp -o nvqc_sample.x --target nvqc --nvqc-backend tensornet
:::
:::
:::
:::

::: {.admonition .note}
Note

By default, the single-GPU single-precision [`custatevec-fp32`{.code
.docutils .literal .notranslate}]{.pre} simulator backend will be
selected if backend information is not specified.
:::
:::

::: {#multiple-gpus .section}
## Multiple GPUs[¶](#multiple-gpus "Permalink to this heading"){.headerlink}

Some CUDA-Q simulator backends are capable of multi-GPU distribution as
detailed in [[simulators]{.std
.std-ref}](../simulators.html#simulators){.reference .internal}. For
example, the [`nvidia-mgpu`{.code .docutils .literal
.notranslate}]{.pre} backend can partition and distribute state vector
simulation to multiple GPUs to simulate a larger number of qubits, whose
state vector size grows beyond the memory size of a single GPU.

To select a specific number of GPUs on the NVQC managed service, the
following [`ngpus`{.code .docutils .literal .notranslate}]{.pre}
(Python) or [`--nvqc-ngpus`{.code .docutils .literal
.notranslate}]{.pre} (C++) option can be used.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target("nvqc", backend="nvidia-mgpu", ngpus=4)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-console .notranslate}
::: highlight
    nvq++ nvqc_sample.cpp -o nvqc_sample.x --target nvqc --nvqc-backend nvidia-mgpu --nvqc-ngpus 4
:::
:::
:::
:::

::: {.admonition .note}
Note

If your NVQC subscription does not contain service instances that have
the specified number of GPUs, you may encounter the following error.

::: {.highlight-console .notranslate}
::: highlight
    Unable to find NVQC deployment with 16 GPUs.
    Available deployments have {1, 2, 4, 8} GPUs.
    Please check your `ngpus` value (Python) or `--nvqc-ngpus` value (C++).
:::
:::
:::

::: {.admonition .note}
Note

Not all simulator backends are capable of utilizing multiple GPUs. When
requesting a multiple-GPU service with a single-GPU simulator backend,
you might encounter the following log message:

::: {.highlight-console .notranslate}
::: highlight
    The requested backend simulator (custatevec-fp32) is not capable of using all 4 GPUs requested.
    Only one GPU will be used for simulation.
    Please refer to CUDA-Q documentation for a list of multi-GPU capable simulator backends.
:::
:::

Consider removing the [`ngpus`{.code .docutils .literal
.notranslate}]{.pre} value (Python) or [`--nvqc-ngpus`{.code .docutils
.literal .notranslate}]{.pre} value (C++) to use the default of 1 GPU if
you don't need to use a multi-GPU backend to better utilize NVQC
resources.

Please refer to the table below for a list of backend simulator names
along with its multi-GPU capability.

+--------------+---------------------------------------+------+------+
| Name         | Description                           | GPU  | M    |
|              |                                       | Acc  | ulti |
|              |                                       | eler | -GPU |
|              |                                       | ated |      |
+==============+=======================================+======+======+
| [`qpp`{.code | CPU-only state vector simulator       | no   | no   |
| .docutils    |                                       |      |      |
| .literal     |                                       |      |      |
| .notrans     |                                       |      |      |
| late}]{.pre} |                                       |      |      |
+--------------+---------------------------------------+------+------+
| [`dm`{.code  | CPU-only density matrix simulator     | no   | no   |
| .docutils    |                                       |      |      |
| .literal     |                                       |      |      |
| .notrans     |                                       |      |      |
| late}]{.pre} |                                       |      |      |
+--------------+---------------------------------------+------+------+
| [`custatevec | Single-precision [`cuStateVec`{.code  | yes  | no   |
| -fp32`{.code | .docutils .literal                    |      |      |
| .docutils    | .notranslate}]{.pre} simulator        |      |      |
| .literal     |                                       |      |      |
| .notrans     |                                       |      |      |
| late}]{.pre} |                                       |      |      |
+--------------+---------------------------------------+------+------+
| [`custatevec | Double-precision [`cuStateVec`{.code  | yes  | no   |
| -fp64`{.code | .docutils .literal                    |      |      |
| .docutils    | .notranslate}]{.pre} simulator        |      |      |
| .literal     |                                       |      |      |
| .notrans     |                                       |      |      |
| late}]{.pre} |                                       |      |      |
+--------------+---------------------------------------+------+------+
| [`tens       | Double-precision [`cuTensorNet`{.code | yes  | yes  |
| ornet`{.code | .docutils .literal                    |      |      |
| .docutils    | .notranslate}]{.pre} full tensor      |      |      |
| .literal     | network contraction simulator         |      |      |
| .notrans     |                                       |      |      |
| late}]{.pre} |                                       |      |      |
+--------------+---------------------------------------+------+------+
| [`tensorne   | Double-precision [`cuTensorNet`{.code | yes  | no   |
| t-mps`{.code | .docutils .literal                    |      |      |
| .docutils    | .notranslate}]{.pre} matrix-product   |      |      |
| .literal     | state simulator                       |      |      |
| .notrans     |                                       |      |      |
| late}]{.pre} |                                       |      |      |
+--------------+---------------------------------------+------+------+
| [`nvidia     | Double-precision [`cuStateVec`{.code  | yes  | yes  |
| -mgpu`{.code | .docutils .literal                    |      |      |
| .docutils    | .notranslate}]{.pre} multi-GPU        |      |      |
| .literal     | simulator                             |      |      |
| .notrans     |                                       |      |      |
| late}]{.pre} |                                       |      |      |
+--------------+---------------------------------------+------+------+

: [Simulator
Backends]{.caption-text}[¶](#id1 "Permalink to this table"){.headerlink}
:::
:::

::: {#multiple-qpus-asynchronous-execution .section}
## Multiple QPUs Asynchronous Execution[¶](#multiple-qpus-asynchronous-execution "Permalink to this heading"){.headerlink}

NVQC provides scalable QPU virtualization services, whereby clients can
submit asynchronous jobs simultaneously to NVQC. These jobs are handled
by a pool of service worker instances.

For example, in the following code snippet, using the [`nqpus`{.code
.docutils .literal .notranslate}]{.pre} (Python) or
[`--nvqc-nqpus`{.code .docutils .literal .notranslate}]{.pre} (C++)
configuration option, the user instantiates 3 virtual QPU instances to
submit simulation jobs to NVQC calculating the expectation value along
with parameter-shift gradients simultaneously.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    from cudaq import spin
    import math

    # Use NVQC with 3 virtual QPUs
    cudaq.set_target("nvqc", nqpus=3)

    print("Number of QPUs:", cudaq.get_target().num_qpus())
    # Create the parameterized ansatz
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    # Define its spin Hamiltonian.
    hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
                   2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
                   6.125 * spin.z(1))


    def opt_gradient(parameter_vector):
        # Evaluate energy and gradient on different remote QPUs
        # (i.e., concurrent job submissions to NVQC)
        energy_future = cudaq.observe_async(kernel,
                                            hamiltonian,
                                            parameter_vector[0],
                                            qpu_id=0)
        plus_future = cudaq.observe_async(kernel,
                                          hamiltonian,
                                          parameter_vector[0] + 0.5 * math.pi,
                                          qpu_id=1)
        minus_future = cudaq.observe_async(kernel,
                                           hamiltonian,
                                           parameter_vector[0] - 0.5 * math.pi,
                                           qpu_id=2)
        return (energy_future.get().expectation(), [
            (plus_future.get().expectation() - minus_future.get().expectation()) /
            2.0
        ])


    optimizer = cudaq.optimizers.LBFGS()
    optimal_value, optimal_parameters = optimizer.optimize(1, opt_gradient)
    print("Ground state energy =", optimal_value)
    print("Optimal parameters =", optimal_parameters)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    #include <cudaq.h>
    #include <cudaq/algorithm.h>
    #include <cudaq/gradients.h>
    #include <cudaq/optimizers.h>
    #include <iostream>

    int main() {
      cudaq::spin_op h =
          5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
          2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
          .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

      auto [ansatz, theta] = cudaq::make_kernel<double>();
      auto q = ansatz.qalloc();
      auto r = ansatz.qalloc();
      ansatz.x(q);
      ansatz.ry(theta, r);
      ansatz.x<cudaq::ctrl>(r, q);

      // Run VQE with a gradient-based optimizer.
      // Delegate cost function and gradient computation across different NVQC-based
      // QPUs.
      // Note: this needs to be compiled with `--nvqc-nqpus 3` create 3 virtual
      // QPUs.
      cudaq::optimizers::lbfgs optimizer;
      auto [opt_val, opt_params] = optimizer.optimize(
          /*dim=*/1, /*opt_function*/ [&](const std::vector<double> &params,
                                          std::vector<double> &grads) {
            // Queue asynchronous jobs to do energy evaluations across multiple QPUs
            auto energy_future =
                cudaq::observe_async(/*qpu_id=*/0, ansatz, h, params[0]);
            const double paramShift = M_PI_2;
            auto plus_future = cudaq::observe_async(/*qpu_id=*/1, ansatz, h,
                                                    params[0] + paramShift);
            auto minus_future = cudaq::observe_async(/*qpu_id=*/2, ansatz, h,
                                                     params[0] - paramShift);
            grads[0] = (plus_future.get().expectation() -
                        minus_future.get().expectation()) /
                       2.0;
            return energy_future.get().expectation();
          });
      std::cout << "Minimum energy = " << opt_val << " (expected -1.74886).\n";
    }
:::
:::

The code above is saved in [`nvqc_vqe.cpp`{.code .docutils .literal
.notranslate}]{.pre} and compiled with the following command, targeting
the [`nvqc`{.code .docutils .literal .notranslate}]{.pre} platform with
3 virtual QPUs.

::: {.highlight-console .notranslate}
::: highlight
    nvq++ nvqc_vqe.cpp -o nvqc_vqe.x --target nvqc --nvqc-nqpus 3
    ./nvqc_vqe.x
:::
:::
:::
:::

::: {.admonition .note}
Note

The NVQC managed-service has a pool of worker instances processing
incoming requests on a first-come-first-serve basis. Thus, the
attainable speedup using multiple virtual QPUs vs. sequential execution
on a single QPU depends on the NVQC service load. For example, if the
number of free workers is greater than the number of requested virtual
QPUs, a linear (ideal) speedup could be achieved. On the other hand, if
all the service workers are busy, multi-QPU distribution may not deliver
any substantial speedup.
:::
:::

::: {#faq .section}
## FAQ[¶](#faq "Permalink to this heading"){.headerlink}

1.  How do I get more information about my NVQC API submission?

The environment variable [`NVQC_LOG_LEVEL`{.code .docutils .literal
.notranslate}]{.pre} can be used to turn on and off certain logs. There
are three levels:

-   Info ([`info`{.code .docutils .literal .notranslate}]{.pre}): basic
    information about NVQC is logged to the console. This is the
    default.

-   Off ([`off`{.code .docutils .literal .notranslate}]{.pre} or
    [`0`{.code .docutils .literal .notranslate}]{.pre}): disable all
    NVQC logging.

-   Trace: ([`trace`{.code .docutils .literal .notranslate}]{.pre}): log
    additional information for each NVQC job execution (including
    timing)

2.  I want to persist my API key to a configuration file.

You may persist your NVQC API Key to a credential configuration file in
lieu of using the [`NVQC_API_KEY`{.code .docutils .literal
.notranslate}]{.pre} environment variable. The configuration file can be
generated as follows, replacing the [`api_key`{.code .docutils .literal
.notranslate}]{.pre} value with your NVQC API Key.

::: {.highlight-bash .notranslate}
::: highlight
    echo "key: <api_key>" >> $HOME/.nvqc_config
:::
:::
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](braket.html "Amazon Braket"){.btn .btn-neutral .float-left
accesskey="p" rel="prev"} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](../../dynamics.html "Dynamics Simulation"){.btn
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
