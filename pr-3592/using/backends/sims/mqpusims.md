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
        -   [State Vector Simulators](svsims.html){.reference .internal}
            -   [CPU](svsims.html#cpu){.reference .internal}
            -   [Single-GPU](svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](svsims.html#multi-gpu-multi-node){.reference
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
        -   [Multi-QPU Simulators](#){.current .reference .internal}
            -   [Simulate Multiple QPUs in
                Parallel](#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](#multi-qpu-other-backends){.reference
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
-   Multiple QPUs
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](tnsims.html "Tensor Network Simulators"){.btn .btn-neutral
.float-left accesskey="p"} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](noisy.html "Noisy Simulators"){.btn .btn-neutral
.float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#multiple-qpus .section}
# Multiple QPUs[¶](#multiple-qpus "Permalink to this heading"){.headerlink}

The CUDA-Q machine model elucidates the various devices considered in
the broader quantum-classical compute node context. Programmers will
have one or many host CPUs, zero or many NVIDIA GPUs, a classical QPU
control space, and the quantum register itself. Moreover, the
[[specification]{.doc}](../../../specification/cudaq/platform.html){.reference
.internal} notes that the underlying platform may expose multiple QPUs.
In the near-term, this will be unlikely with physical QPU
instantiations, but the availability of GPU-based circuit simulators on
NVIDIA multi-GPU architectures does give one an opportunity to think
about programming such a multi-QPU architecture in the near-term. CUDA-Q
starts by enabling one to query information about the underlying quantum
platform via the [`quantum_platform`{.code .docutils .literal
.notranslate}]{.pre} abstraction. This type exposes a
[`num_qpus()`{.code .docutils .literal .notranslate}]{.pre} method that
can be used to query the number of available QPUs for asynchronous
CUDA-Q kernel and [`cudaq::`{.code .docutils .literal
.notranslate}]{.pre} function invocations. Each available QPU is
assigned a logical index, and programmers can launch specific
asynchronous function invocations targeting a desired QPU.

::: {#simulate-multiple-qpus-in-parallel .section}
[]{#mqpu-platform}

## Simulate Multiple QPUs in Parallel[¶](#simulate-multiple-qpus-in-parallel "Permalink to this heading"){.headerlink}

In the multi-QPU mode ([`mqpu`{.code .docutils .literal
.notranslate}]{.pre} option), the NVIDIA backend provides a simulated
QPU for every available NVIDIA GPU on the underlying system. Each QPU is
simulated via a [`cuStateVec`{.code .docutils .literal
.notranslate}]{.pre} simulator backend as defined by the NVIDIA backend.
For more information about using multiple GPUs to simulate each virtual
QPU, or using a different backend for virtual QPUs, please see [[remote
MQPU platform]{.std .std-ref}](#remote-mqpu-platform){.reference
.internal}. This target enables asynchronous parallel execution of
quantum kernel tasks.

Here is a simple example demonstrating its usage.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq

    cudaq.set_target("nvidia", option="mqpu")
    target = cudaq.get_target()
    qpu_count = target.num_qpus()
    print("Number of QPUs:", qpu_count)


    @cudaq.kernel
    def kernel(qubit_count: int):
        qvector = cudaq.qvector(qubit_count)
        # Place qubits in superposition state.
        h(qvector)
        # Measure.
        mz(qvector)


    count_futures = []
    for qpu in range(qpu_count):
        count_futures.append(cudaq.sample_async(kernel, 5, qpu_id=qpu))

    for counts in count_futures:
        print(counts.get())
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      auto kernelToBeSampled = [](int runtimeParam) __qpu__ {
        cudaq::qvector q(runtimeParam);
        h(q);
        mz(q);
      };

      // Get the quantum_platform singleton
      auto &platform = cudaq::get_platform();

      // Query the number of QPUs in the system
      auto num_qpus = platform.num_qpus();
      printf("Number of QPUs: %zu\n", num_qpus);
      // We will launch asynchronous sampling tasks
      // and will store the results immediately as a future
      // we can query at some later point
      std::vector<cudaq::async_sample_result> countFutures;
      for (std::size_t i = 0; i < num_qpus; i++) {
        countFutures.emplace_back(
            cudaq::sample_async(i, kernelToBeSampled, 5 /*runtimeParam*/));
      }

      //
      // Go do other work, asynchronous execution of sample tasks on-going
      //

      // Get the results, note future::get() will kick off a wait
      // if the results are not yet available.
      for (auto &counts : countFutures) {
        counts.get().dump();
      }
:::
:::

One can specify the target multi-QPU architecture with the
[`--target`{.code .docutils .literal .notranslate}]{.pre} flag:

::: {.highlight-console .notranslate}
::: highlight
    nvq++ sample_async.cpp --target nvidia --target-option mqpu
    ./a.out
:::
:::
:::
:::

CUDA-Q exposes asynchronous versions of the default [`cudaq`{.code
.docutils .literal .notranslate}]{.pre} algorithmic primitive functions
like [`sample`{.code .docutils .literal .notranslate}]{.pre},
[`observe`{.code .docutils .literal .notranslate}]{.pre}, and
[`get_state`{.code .docutils .literal .notranslate}]{.pre} (e.g.,
[`sample_async`{.code .docutils .literal .notranslate}]{.pre} function
in the above code snippets).

Depending on the number of GPUs available on the system, the
[`nvidia`{.code .docutils .literal .notranslate}]{.pre} multi-QPU
platform will create the same number of virtual QPU instances. For
example, on a system with 4 GPUs, the above code will distribute the
four sampling tasks among those [`GPUEmulatedQPU`{.code .docutils
.literal .notranslate}]{.pre} instances.

The results might look like the following 4 different random samplings:

::: {.highlight-console .notranslate}
::: highlight
    Number of QPUs: 4
    { 10011:28 01100:28 ... }
    { 10011:37 01100:25 ... }
    { 10011:29 01100:25 ... }
    { 10011:33 01100:30 ... }
:::
:::

::: {.admonition .note}
Note

By default, the [`nvidia`{.code .docutils .literal .notranslate}]{.pre}
multi-QPU platform will utilize all available GPUs (number of QPUs
instances is equal to the number of GPUs). To specify the number QPUs to
be instantiated, one can set the [`CUDAQ_MQPU_NGPUS`{.code .docutils
.literal .notranslate}]{.pre} environment variable. For example, use
[`export`{.code .docutils .literal .notranslate}]{.pre}` `{.code
.docutils .literal .notranslate}[`CUDAQ_MQPU_NGPUS=2`{.code .docutils
.literal .notranslate}]{.pre} to specify that only 2 QPUs (GPUs) are
needed.
:::

Since the underlying [`GPUEmulatedQPU`{.code .docutils .literal
.notranslate}]{.pre} is a simulator backend, we can also retrieve the
state vector from each QPU via the [`cudaq::get_state_async`{.code
.docutils .literal .notranslate}]{.pre} (C++) or
[`cudaq.get_state_async`{.code .docutils .literal .notranslate}]{.pre}
(Python) as shown in the bellow code snippets.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq

    cudaq.set_target("nvidia", option="mqpu")
    target = cudaq.get_target()
    qpu_count = target.num_qpus()
    print("Number of QPUs:", qpu_count)


    @cudaq.kernel
    def kernel():
        qvector = cudaq.qvector(5)
        # Place qubits in GHZ State
        h(qvector[0])
        for qubit in range(4):
            x.ctrl(qvector[qubit], qvector[qubit + 1])


    state_futures = []
    for qpu in range(qpu_count):
        state_futures.append(cudaq.get_state_async(kernel, qpu_id=qpu))

    for state in state_futures:
        print(state.get())
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      auto kernelToRun = [](int runtimeParam) __qpu__ {
        cudaq::qvector q(runtimeParam);
        h(q[0]);
        for (int i = 0; i < runtimeParam - 1; ++i)
          x<cudaq::ctrl>(q[i], q[i + 1]);
      };

      // Get the quantum_platform singleton
      auto &platform = cudaq::get_platform();

      // Query the number of QPUs in the system
      auto num_qpus = platform.num_qpus();
      printf("Number of QPUs: %zu\n", num_qpus);
      // We will launch asynchronous tasks
      // and will store the results immediately as a future
      // we can query at some later point
      std::vector<cudaq::async_state_result> stateFutures;
      for (std::size_t i = 0; i < num_qpus; i++) {
        stateFutures.emplace_back(
            cudaq::get_state_async(i, kernelToRun, 5 /*runtimeParam*/));
      }

      //
      // Go do other work, asynchronous execution of tasks on-going
      //

      // Get the results, note future::get() will kick off a wait
      // if the results are not yet available.
      for (auto &state : stateFutures) {
        state.get().dump();
      }
:::
:::

One can specify the target multi-QPU architecture with the
[`--target`{.code .docutils .literal .notranslate}]{.pre} flag:

::: {.highlight-console .notranslate}
::: highlight
    nvq++ get_state_async.cpp --target nvidia --target-option mqpu
    ./a.out
:::
:::
:::
:::

See the [Hadamard Test
notebook](https://nvidia.github.io/cuda-quantum/latest/applications/python/hadamard_test.html){.reference
.external} for an application that leverages the [`mqpu`{.code .docutils
.literal .notranslate}]{.pre} backend.

::: deprecated
[Deprecated since version 0.8: ]{.versionmodified .deprecated}The
[`nvidia-mqpu`{.code .docutils .literal .notranslate}]{.pre} and
[`nvidia-mqpu-fp64`{.code .docutils .literal .notranslate}]{.pre}
targets, which are equivalent to the multi-QPU options
[`mqpu,fp32`{.code .docutils .literal .notranslate}]{.pre} and
[`mqpu,fp64`{.code .docutils .literal .notranslate}]{.pre},
respectively, of the [`nvidia`{.code .docutils .literal
.notranslate}]{.pre} target, are deprecated and will be removed in a
future release.
:::

::: {#parallel-distribution-mode .section}
### Parallel distribution mode[¶](#parallel-distribution-mode "Permalink to this heading"){.headerlink}

The CUDA-Q [`nvidia`{.code .docutils .literal .notranslate}]{.pre}
multi-QPU platform supports two modes of parallel distribution of
expectation value computation:

-   MPI: distribute the expectation value computations across available
    MPI ranks and GPUs for each Hamiltonian term.

-   Thread: distribute the expectation value computations among
    available GPUs via standard C++ threads (each thread handles one
    GPU).

For instance, if all GPUs are available on a single node, thread-based
parallel distribution ([`cudaq::parallel::thread`{.code .docutils
.literal .notranslate}]{.pre} in C++ or [`cudaq.parallel.thread`{.code
.docutils .literal .notranslate}]{.pre} in Python, as shown in the above
example) is sufficient. On the other hand, if one wants to distribute
the tasks across GPUs on multiple nodes, e.g., on a compute cluster, MPI
distribution mode should be used.

An example of MPI distribution mode usage in both C++ and Python is
given below:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    from cudaq import spin

    cudaq.mpi.initialize()
    cudaq.set_target("nvidia", option="mqpu")


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

    exp_val = cudaq.observe(kernel, hamiltonian, 0.59,
                            execution=cudaq.parallel.mpi).expectation()
    if cudaq.mpi.rank() == 0:
        print("Expectation value: ", exp_val)

    cudaq.mpi.finalize()
:::
:::

::: {.highlight-console .notranslate}
::: highlight
    mpiexec -np <N> python3 file.py
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      cudaq::mpi::initialize();
      cudaq::spin_op h =
          5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
          2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
          .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

      auto ansatz = [](double theta) __qpu__ {
        cudaq::qubit q, r;
        x(q);
        ry(theta, r);
        x<cudaq::ctrl>(r, q);
      };

      double result = cudaq::observe<cudaq::parallel::mpi>(ansatz, h, 0.59);
      if (cudaq::mpi::rank() == 0)
        printf("Expectation value: %lf\n", result);
      cudaq::mpi::finalize();
:::
:::

::: {.highlight-console .notranslate}
::: highlight
    nvq++ file.cpp --target nvidia --target-option mqpu
    mpiexec -np <N> a.out
:::
:::
:::
:::

In the above example, the parallel distribution mode was set to
[`mpi`{.code .docutils .literal .notranslate}]{.pre} using
[`cudaq::parallel::mpi`{.code .docutils .literal .notranslate}]{.pre} in
C++ or [`cudaq.parallel.mpi`{.code .docutils .literal
.notranslate}]{.pre} in Python. CUDA-Q provides MPI utility functions to
initialize, finalize, or query (rank, size, etc.) the MPI runtime. Last
but not least, the compiled executable (C++) or Python script needs to
be launched with an appropriate MPI command, e.g., [`mpiexec`{.code
.docutils .literal .notranslate}]{.pre}, [`mpirun`{.code .docutils
.literal .notranslate}]{.pre}, [`srun`{.code .docutils .literal
.notranslate}]{.pre}, etc.
:::
:::

::: {#multi-qpu-other-backends .section}
## Multi-QPU + Other Backends[¶](#multi-qpu-other-backends "Permalink to this heading"){.headerlink}

As shown in the above examples, the multi-QPU NVIDIA platform enables
multi-QPU distribution whereby each QPU is simulated by a [[single
NVIDIA GPU]{.std .std-ref}](svsims.html#cuquantum-single-gpu){.reference
.internal}. To run multi-QPU workloads on different simulator backends,
one can use the [`remote-mqpu`{.code .docutils .literal
.notranslate}]{.pre} platform, which encapsulates simulated QPUs as
independent HTTP REST server instances. The following code illustrates
how to launch asynchronous sampling tasks on multiple virtual QPUs, each
simulated by a [`tensornet`{.code .docutils .literal
.notranslate}]{.pre} simulator backend.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
        # Specified as program input, e.g.
        # ```
        # backend = "tensornet"; servers = "2"
        # ```
        backend = args.backend
        servers = args.servers

        # Define a kernel to be sampled.
        @cudaq.kernel
        def kernel(controls_count: int):
            controls = cudaq.qvector(controls_count)
            targets = cudaq.qvector(2)
            # Place controls in superposition state.
            h(controls)
            for target in range(2):
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
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      // Define a kernel to be sampled.
      auto [kernel, nrControls] = cudaq::make_kernel<int>();
      auto controls = kernel.qalloc(nrControls);
      auto targets = kernel.qalloc(2);
      kernel.h(controls);
      for (std::size_t tidx = 0; tidx < 2; ++tidx) {
        kernel.x<cudaq::ctrl>(controls, targets[tidx]);
      }
      kernel.mz(controls);
      kernel.mz(targets);

      // Query the number of QPUs in the system;
      // The number of QPUs is equal to the number of (auto-)launched server
      // instances.
      auto &platform = cudaq::get_platform();
      auto num_qpus = platform.num_qpus();
      printf("Number of QPUs: %zu\n", num_qpus);

      // We will launch asynchronous sampling tasks,
      // and will store the results as a future we can query at some later point.
      // Each QPU (indexed by an unique Id) is associated with a remote REST server.
      std::vector<cudaq::async_sample_result> countFutures;
      for (std::size_t i = 0; i < num_qpus; i++) {
        countFutures.emplace_back(cudaq::sample_async(
            /*qpuId=*/i, kernel, /*nrControls=*/i + 1));
      }

      // Go do other work, asynchronous execution of sample tasks on-going
      // Get the results, note future::get() will kick off a wait
      // if the results are not yet available.
      for (auto &counts : countFutures) {
        counts.get().dump();
      }
:::
:::

The code above is saved in [`sample_async.cpp`{.code .docutils .literal
.notranslate}]{.pre} and compiled with the following command, targeting
the [`remote-mqpu`{.code .docutils .literal .notranslate}]{.pre}
platform:

::: {.highlight-console .notranslate}
::: highlight
    nvq++ sample_async.cpp -o sample_async.x --target remote-mqpu --remote-mqpu-backend tensornet --remote-mqpu-auto-launch 2
    ./sample_async.x
:::
:::
:::
:::

In the above code snippets, the [`remote-mqpu`{.code .docutils .literal
.notranslate}]{.pre} platform was used in the auto-launch mode, whereby
a specific number of server instances, i.e., virtual QPUs, are launched
on the local machine in the background. The remote QPU daemon service,
[`cudaq-qpud`{.code .docutils .literal .notranslate}]{.pre}, will also
be shut down automatically at the end of the session.

::: {.admonition .note}
Note

By default, auto launching daemon services do not support MPI
parallelism. Hence, using the [`nvidia-mgpu`{.code .docutils .literal
.notranslate}]{.pre} backend to simulate each virtual QPU requires
manually launching each server instance. How to do that is explained in
the rest of this section.
:::

To customize how many and which GPUs are used for simulating each
virtual QPU, one can launch each server manually. For instance, on a
machine with 8 NVIDIA GPUs, one may wish to partition those GPUs into 4
virtual QPU instances, each manages 2 GPUs. To do so, first launch a
[`cudaq-qpud`{.code .docutils .literal .notranslate}]{.pre} server for
each virtual QPU:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-bash .notranslate}
::: highlight
    # Use cudaq-qpud.py wrapper script to automatically find dependencies for the Python wheel configuration.
    cudaq_location=`python3 -m pip show cudaq | grep -e 'Location: .*$'`
    qpud_py="${cudaq_location#Location: }/bin/cudaq-qpud.py"
    CUDA_VISIBLE_DEVICES=0,1 mpiexec -np 2 python3 "$qpud_py" --port <QPU 1 TCP/IP port number>
    CUDA_VISIBLE_DEVICES=2,3 mpiexec -np 2 python3 "$qpud_py" --port <QPU 2 TCP/IP port number>
    CUDA_VISIBLE_DEVICES=4,5 mpiexec -np 2 python3 "$qpud_py" --port <QPU 3 TCP/IP port number>
    CUDA_VISIBLE_DEVICES=6,7 mpiexec -np 2 python3 "$qpud_py" --port <QPU 4 TCP/IP port number>
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-bash .notranslate}
::: highlight
    # It is assumed that your $LD_LIBRARY_PATH is able to find all the necessary dependencies.
    CUDA_VISIBLE_DEVICES=0,1 mpiexec -np 2 cudaq-qpud --port <QPU 1 TCP/IP port number>
    CUDA_VISIBLE_DEVICES=2,3 mpiexec -np 2 cudaq-qpud --port <QPU 2 TCP/IP port number>
    CUDA_VISIBLE_DEVICES=4,5 mpiexec -np 2 cudaq-qpud --port <QPU 3 TCP/IP port number>
    CUDA_VISIBLE_DEVICES=6,7 mpiexec -np 2 cudaq-qpud --port <QPU 4 TCP/IP port number>
:::
:::
:::
:::

In the above code snippet, four [`nvidia-mgpu`{.code .docutils .literal
.notranslate}]{.pre} daemons are started in MPI context via the
[`mpiexec`{.code .docutils .literal .notranslate}]{.pre} launcher. This
activates MPI runtime environment required by the [`nvidia-mgpu`{.code
.docutils .literal .notranslate}]{.pre} backend. Each QPU daemon is
assigned a unique TCP/IP port number via the [`--port`{.code .docutils
.literal .notranslate}]{.pre} command-line option. The
[`CUDA_VISIBLE_DEVICES`{.code .docutils .literal .notranslate}]{.pre}
environment variable restricts the GPU devices that each QPU daemon sees
so that it targets specific GPUs.

With these invocations, each virtual QPU is locally addressable at the
URL [`localhost:<port>`{.code .docutils .literal .notranslate}]{.pre}.

::: {.admonition .warning}
Warning

There is no authentication required to communicate with this server app.
Hence, please make sure to either (1) use a non-public TCP/IP port for
internal use or (2) use firewalls or other security mechanisms to manage
user access.
:::

User code can then target these QPUs for multi-QPU workloads, such as
asynchronous sample or observe shown above for the multi-QPU NVIDIA
platform platform.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    cudaq.set_target("remote-mqpu", url="localhost:<port1>,localhost:<port2>,localhost:<port3>,localhost:<port4>", backend="nvidia-mgpu")
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-console .notranslate}
::: highlight
    nvq++ distributed.cpp --target remote-mqpu --remote-mqpu-url localhost:<port1>,localhost:<port2>,localhost:<port3>,localhost:<port4> --remote-mqpu-backend nvidia-mgpu
:::
:::
:::
:::

Each URL is treated as an independent QPU, hence the number of QPUs
([`num_qpus()`{.code .docutils .literal .notranslate}]{.pre}) is equal
to the number of URLs provided. The multi-node multi-GPU simulator
backend ([`nvidia-mgpu`{.code .docutils .literal .notranslate}]{.pre})
is requested via the [`--remote-mqpu-backend`{.code .docutils .literal
.notranslate}]{.pre} command-line option.

::: {.admonition .note}
Note

The requested backend ([`nvidia-mgpu`{.code .docutils .literal
.notranslate}]{.pre}) will be executed inside the context of the QPU
daemon service, thus inherits its GPU resource allocation (two GPUs per
backend simulator instance).
:::

::: {#supported-kernel-arguments .section}
### Supported Kernel Arguments[¶](#supported-kernel-arguments "Permalink to this heading"){.headerlink}

The platform serializes kernel invocation to QPU daemons via REST APIs.
Please refer to the [Open API Docs](../../openapi.html){.reference
.external} for the latest API information. Runtime arguments are
serialized into a flat memory buffer ([`args`{.code .docutils .literal
.notranslate}]{.pre} field of the request JSON). For more information
about argument type serialization, please see [[the table below]{.std
.std-ref}](#type-serialization-table){.reference .internal}.

When using a remote backend to simulate each virtual QPU, by default, we
currently do not support passing complex data structures, such as nested
vectors or class objects, or other kernels as arguments to the entry
point kernels. These type limitations only apply to the **entry-point**
kernel and not when passing arguments to other quantum kernels.

Support for the full range of argument types within CUDA-Q can be
enabled by compiling the code with the [`--enable-mlir`{.code .docutils
.literal .notranslate}]{.pre} option. This flag forces quantum kernels
to be compiled with the CUDA-Q MLIR-based compiler. As a result, runtime
arguments can be resolved by the CUDA Quantum compiler infrastructure to
support wider range of argument types. However, certain language
constructs within quantum kernels may not yet be fully supported.

[]{#type-serialization-table}

+----------------------+----------------------+----------------------+
| Data type            | Example              | Serialization        |
+======================+======================+======================+
| Trivial type         | [`int`{.code         | Byte data (via       |
| (occupies a          | .docutils .literal   | [`memcpy`{.code      |
| contiguous memory    | .                    | .docutils .literal   |
| area)                | notranslate}]{.pre}, | .                    |
|                      | [`std::size_t`{.code | notranslate}]{.pre}) |
|                      | .docutils .literal   |                      |
|                      | .                    |                      |
|                      | notranslate}]{.pre}, |                      |
|                      | [`double`{.code      |                      |
|                      | .docutils .literal   |                      |
|                      | .                    |                      |
|                      | notranslate}]{.pre}, |                      |
|                      | etc.                 |                      |
+----------------------+----------------------+----------------------+
| [`std::vector`{.code | [`std                | Total vector size in |
| .docutils .literal   | ::vector<int>`{.code | bytes as a 64-bit    |
| .notranslate}]{.pre} | .docutils .literal   | integer followed by  |
| of trivial type      | .                    | serialized data of   |
|                      | notranslate}]{.pre}, | all vector elements. |
|                      | [`std::v             |                      |
|                      | ector<double>`{.code |                      |
|                      | .docutils .literal   |                      |
|                      | .                    |                      |
|                      | notranslate}]{.pre}, |                      |
|                      | etc.                 |                      |
+----------------------+----------------------+----------------------+
| [`cuda               | [`cudaq::pauli       | Same as              |
| q::pauli_word`{.code | _word("IXIZ")`{.code | [`std:               |
| .docutils .literal   | .docutils .literal   | :vector<char>`{.code |
| .notranslate}]{.pre} | .notranslate}]{.pre} | .docutils .literal   |
|                      |                      | .                    |
|                      |                      | notranslate}]{.pre}: |
|                      |                      | total vector size in |
|                      |                      | bytes as a 64-bit    |
|                      |                      | integer followed by  |
|                      |                      | serialized data of   |
|                      |                      | all characters.      |
+----------------------+----------------------+----------------------+
| Single-level nested  | [`std::vector<std:   | Number of top-level  |
| [`std::vector`{.code | :vector<int>>`{.code | elements (as a       |
| .docutils .literal   | .docutils .literal   | 64-bit integer)      |
| .notranslate}]{.pre} | .                    | followed sizes in    |
| of supported         | notranslate}]{.pre}, | bytes of element     |
| [`std::vector`{.code | [`std::vector<cudaq  | vectors (as a        |
| .docutils .literal   | ::pauli_word>`{.code | contiguous array of  |
| .notranslate}]{.pre} | .docutils .literal   | 64-bit integers)     |
| types                | .                    | then serialized data |
|                      | notranslate}]{.pre}, | of the inner         |
|                      | etc.                 | vectors.             |
+----------------------+----------------------+----------------------+

: [Kernel argument
serialization]{.caption-text}[¶](#id3 "Permalink to this table"){.headerlink}

For CUDA-Q kernels that return a value, the remote platform supports
returning simple data types of [`bool`{.code .docutils .literal
.notranslate}]{.pre}, integral (e.g., [`int`{.code .docutils .literal
.notranslate}]{.pre} or [`std::size_t`{.code .docutils .literal
.notranslate}]{.pre}), and floating-point types ([`float`{.code
.docutils .literal .notranslate}]{.pre} or [`double`{.code .docutils
.literal .notranslate}]{.pre}) when MLIR-based compilation is enabled
([`--enable-mlir`{.code .docutils .literal .notranslate}]{.pre}).
:::

::: {#accessing-simulated-quantum-state .section}
### Accessing Simulated Quantum State[¶](#accessing-simulated-quantum-state "Permalink to this heading"){.headerlink}

The remote [`MQPU`{.code .docutils .literal .notranslate}]{.pre}
platform supports accessing simulator backend's state vector via the
[`cudaq::get_state`{.code .docutils .literal .notranslate}]{.pre} (C++)
or [`cudaq.get_state`{.code .docutils .literal .notranslate}]{.pre}
(Python) APIs, similar to local simulator backends.

State data can be retrieved as a full state vector or as individual
basis states' amplitudes. The later is designed for large quantum
states, which incurred data transfer overheads.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    state = cudaq.get_state(kernel)
    amplitudes = state.amplitudes(['0000', '1111'])
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    auto state = cudaq::get_state(kernel)
    auto amplitudes = state.amplitudes({{0, 0, 0, 0}, {1, 1, 1, 1}});
:::
:::
:::
:::

In the above example, the amplitudes of the two requested states are
returned.

For C++ quantum kernels
[[\[]{.fn-bracket}\*[\]]{.fn-bracket}](#id2){#id1 .footnote-reference
.brackets role="doc-noteref"} compiled with the CUDA-Q MLIR-based
compiler and Python kernels, state accessor is evaluated in a
just-in-time/on-demand manner, and hence can be customize to users'
need.

For instance, in the above amplitude access example, if the state vector
is very large, e.g., multi-GPU distributed state vectors or
tensor-network encoded quantum states, the full state vector will not be
retrieved when [`get_state`{.code .docutils .literal
.notranslate}]{.pre} is called. Instead, when the [`amplitudes`{.code
.docutils .literal .notranslate}]{.pre} accessor is called, a specific
amplitude calculation request will be sent to the server. Thus, only the
amplitudes of those basis states will be computed and returned.

Similarly, for state overlap calculation, if deferred state evaluation
is available (Python/MLIR-based compiler) for both of the operand
quantum states, a custom overlap calculation request will be constructed
and sent to the server. Only the final overlap result will be returned,
thereby eliminating back-and-forth state data transfers.

[[\[]{.fn-bracket}[\*](#id1){role="doc-backlink"}[\]]{.fn-bracket}]{.label}

Only C++ quantum kernels whose names are available via run-time type
information (RTTI) are supported. For example, quantum kernels expressed
as named [`struct`{.code .docutils .literal .notranslate}]{.pre} are
supported but not standalone functions. Kernels that do not have
deferred state evaluation support will perform synchronous
[`get_state`{.code .docutils .literal .notranslate}]{.pre}, whereby the
full state vector is returned from the server immediately.
:::
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](tnsims.html "Tensor Network Simulators"){.btn .btn-neutral
.float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](noisy.html "Noisy Simulators"){.btn .btn-neutral
.float-right accesskey="n" rel="next"}
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
