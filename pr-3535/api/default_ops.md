::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../index.html){.icon .icon-home}

::: version
pr-3535
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](../using/quick_start.html){.reference .internal}
    -   [Install
        CUDA-Q](../using/quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](../using/quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q
        Academic](../using/quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](../using/basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q
        Kernel?](../using/basics/kernel_intro.html){.reference
        .internal}
    -   [Building your first CUDA-Q
        Program](../using/basics/build_kernel.html){.reference
        .internal}
    -   [Running your first CUDA-Q
        Program](../using/basics/run_kernel.html){.reference .internal}
        -   [Sample](../using/basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](../using/basics/run_kernel.html#run){.reference
            .internal}
        -   [Observe](../using/basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](../using/basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](../using/basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](../using/basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
-   [Examples](../using/examples/examples.html){.reference .internal}
    -   [Introduction](../using/examples/introduction.html){.reference
        .internal}
    -   [Building
        Kernels](../using/examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](../using/examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](../using/examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](../using/examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](../using/examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](../using/examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](../using/examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](../using/examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](../using/examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](../using/examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](../using/examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](../using/examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](../using/examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](../using/examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring
        Kernels](../using/examples/measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](../using/examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](../examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](../examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](../examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing
        Kernels](../using/examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](../using/examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](../using/examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](../using/examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](../using/examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](../using/examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](../using/examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](../using/examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](../using/examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](../using/examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](../using/examples/expectation_values.html){.reference
        .internal}
        -   [Parallelizing across Multiple
            Processors](../using/examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](../using/examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](../using/examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../using/examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../using/examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](../using/examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](../using/examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends ([`remote-mqpu`{.code .docutils
            .literal
            .notranslate}]{.pre})](../using/examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
            .internal}
    -   [Optimizers &
        Gradients](../examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [Built in CUDA-Q Optimizers and
            Gradients](../examples/python/optimizers_gradients.html#Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
        -   [Third-Party
            Optimizers](../examples/python/optimizers_gradients.html#Third-Party-Optimizers){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../examples/python/optimizers_gradients.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](../examples/python/noisy_simulations.html){.reference
        .internal}
    -   [Constructing
        Operators](../using/examples/operators.html){.reference
        .internal}
        -   [Constructing Spin
            Operators](../using/examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](../using/examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](../examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](../examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](../using/examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](../using/examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](../using/examples/hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](../using/examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](../using/examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](../using/examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](../using/examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](../using/examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](../using/examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](../using/examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum Circuits,
            Inc.](../using/examples/hardware_providers.html#quantum-circuits-inc){.reference
            .internal}
        -   [Quantum
            Machines](../using/examples/hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](../using/examples/hardware_providers.html#quera-computing){.reference
            .internal}
    -   [Dynamics
        Examples](../using/examples/dynamics_examples.html){.reference
        .internal}
        -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
            Model)](../examples/python/dynamics/dynamics_intro_1.html){.reference
            .internal}
            -   [Why dynamics simulations vs. circuit
                simulations?](../examples/python/dynamics/dynamics_intro_1.html#Why-dynamics-simulations-vs.-circuit-simulations?){.reference
                .internal}
            -   [Functionality](../examples/python/dynamics/dynamics_intro_1.html#Functionality){.reference
                .internal}
            -   [Performance](../examples/python/dynamics/dynamics_intro_1.html#Performance){.reference
                .internal}
            -   [Section 1 - Simulating the Jaynes-Cummings
                Hamiltonian](../examples/python/dynamics/dynamics_intro_1.html#Section-1---Simulating-the-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Exercise 1 - Simulating a many-photon Jaynes-Cummings
                Hamiltonian](../examples/python/dynamics/dynamics_intro_1.html#Exercise-1---Simulating-a-many-photon-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Section 2 - Simulating open quantum systems with the
                [`collapse_operators`{.docutils .literal
                .notranslate}]{.pre}](../examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
                .internal}
            -   [Exercise 2 - Adding additional jump operators
                [\\(L_i\\)]{.math .notranslate
                .nohighlight}](../examples/python/dynamics/dynamics_intro_1.html#Exercise-2---Adding-additional-jump-operators-L_i){.reference
                .internal}
            -   [Section 3 - Many qubits coupled to the
                resonator](../examples/python/dynamics/dynamics_intro_1.html#Section-3---Many-qubits-coupled-to-the-resonator){.reference
                .internal}
        -   [Introduction to CUDA-Q Dynamics (Time Dependent
            Hamiltonians)](../examples/python/dynamics/dynamics_intro_2.html){.reference
            .internal}
            -   [The Landau-Zener
                model](../examples/python/dynamics/dynamics_intro_2.html#The-Landau-Zener-model){.reference
                .internal}
            -   [Section 1 - Implementing time dependent
                terms](../examples/python/dynamics/dynamics_intro_2.html#Section-1---Implementing-time-dependent-terms){.reference
                .internal}
            -   [Section 2 - Implementing custom
                operators](../examples/python/dynamics/dynamics_intro_2.html#Section-2---Implementing-custom-operators){.reference
                .internal}
            -   [Section 3 - Heisenberg Model with a time-varying
                magnetic
                field](../examples/python/dynamics/dynamics_intro_2.html#Section-3---Heisenberg-Model-with-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 1 - Define a time-varying magnetic
                field](../examples/python/dynamics/dynamics_intro_2.html#Exercise-1---Define-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 2
                (Optional)](../examples/python/dynamics/dynamics_intro_2.html#Exercise-2-(Optional)){.reference
                .internal}
        -   [Superconducting
            Qubits](../examples/python/dynamics/superconducting.html){.reference
            .internal}
            -   [Cavity
                QED](../examples/python/dynamics/superconducting.html#Cavity-QED){.reference
                .internal}
            -   [Cross
                Resonance](../examples/python/dynamics/superconducting.html#Cross-Resonance){.reference
                .internal}
            -   [Transmon
                Resonator](../examples/python/dynamics/superconducting.html#Transmon-Resonator){.reference
                .internal}
        -   [Spin
            Qubits](../examples/python/dynamics/spinqubits.html){.reference
            .internal}
            -   [Silicon Spin
                Qubit](../examples/python/dynamics/spinqubits.html#Silicon-Spin-Qubit){.reference
                .internal}
            -   [Heisenberg
                Model](../examples/python/dynamics/spinqubits.html#Heisenberg-Model){.reference
                .internal}
        -   [Trapped Ion
            Qubits](../examples/python/dynamics/iontrap.html){.reference
            .internal}
            -   [GHZ
                state](../examples/python/dynamics/iontrap.html#GHZ-state){.reference
                .internal}
        -   [Control](../examples/python/dynamics/control.html){.reference
            .internal}
            -   [Gate
                Calibration](../examples/python/dynamics/control.html#Gate-Calibration){.reference
                .internal}
            -   [Pulse](../examples/python/dynamics/control.html#Pulse){.reference
                .internal}
            -   [Qubit
                Control](../examples/python/dynamics/control.html#Qubit-Control){.reference
                .internal}
            -   [Qubit
                Dynamics](../examples/python/dynamics/control.html#Qubit-Dynamics){.reference
                .internal}
            -   [Landau-Zenner](../examples/python/dynamics/control.html#Landau-Zenner){.reference
                .internal}
-   [Applications](../using/applications.html){.reference .internal}
    -   [Max-Cut with QAOA](../applications/python/qaoa.html){.reference
        .internal}
    -   [Molecular docking via
        DC-QAOA](../applications/python/digitized_counterdiabatic_qaoa.html){.reference
        .internal}
        -   [Setting up the Molecular Docking
            Problem](../applications/python/digitized_counterdiabatic_qaoa.html#Setting-up-the-Molecular-Docking-Problem){.reference
            .internal}
        -   [CUDA-Q
            Implementation](../applications/python/digitized_counterdiabatic_qaoa.html#CUDA-Q-Implementation){.reference
            .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](../applications/python/krylov.html){.reference
        .internal}
        -   [Setup](../applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](../applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](../applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Quantum-Selected Configuration Interaction
        (QSCI)](../applications/python/qsci.html){.reference .internal}
        -   [0. Problem
            definition](../applications/python/qsci.html#0.-Problem-definition){.reference
            .internal}
        -   [1. Prepare an Approximate Quantum
            State](../applications/python/qsci.html#1.-Prepare-an-Approximate-Quantum-State){.reference
            .internal}
        -   [2 Quantum Sampling to Select
            Configuration](../applications/python/qsci.html#2-Quantum-Sampling-to-Select-Configuration){.reference
            .internal}
        -   [3. Classical Diagonalization on the Selected
            Subspace](../applications/python/qsci.html#3.-Classical-Diagonalization-on-the-Selected-Subspace){.reference
            .internal}
        -   [5. Compuare
            results](../applications/python/qsci.html#5.-Compuare-results){.reference
            .internal}
        -   [Reference](../applications/python/qsci.html#Reference){.reference
            .internal}
    -   [Bernstein-Vazirani
        Algorithm](../applications/python/bernstein_vazirani.html){.reference
        .internal}
        -   [Classical
            case](../applications/python/bernstein_vazirani.html#Classical-case){.reference
            .internal}
        -   [Quantum
            case](../applications/python/bernstein_vazirani.html#Quantum-case){.reference
            .internal}
        -   [Implementing in
            CUDA-Q](../applications/python/bernstein_vazirani.html#Implementing-in-CUDA-Q){.reference
            .internal}
    -   [Cost
        Minimization](../applications/python/cost_minimization.html){.reference
        .internal}
    -   [Deutsch's
        Algorithm](../applications/python/deutsch_algorithm.html){.reference
        .internal}
        -   [XOR [\\(\\oplus\\)]{.math .notranslate
            .nohighlight}](../applications/python/deutsch_algorithm.html#XOR-\oplus){.reference
            .internal}
        -   [Quantum
            oracles](../applications/python/deutsch_algorithm.html#Quantum-oracles){.reference
            .internal}
        -   [Phase
            oracle](../applications/python/deutsch_algorithm.html#Phase-oracle){.reference
            .internal}
        -   [Quantum
            parallelism](../applications/python/deutsch_algorithm.html#Quantum-parallelism){.reference
            .internal}
        -   [Deutsch's
            Algorithm:](../applications/python/deutsch_algorithm.html#Deutsch's-Algorithm:){.reference
            .internal}
    -   [Divisive Clustering With Coresets Using
        CUDA-Q](../applications/python/divisive_clustering_coresets.html){.reference
        .internal}
        -   [Data
            preprocessing](../applications/python/divisive_clustering_coresets.html#Data-preprocessing){.reference
            .internal}
        -   [Quantum
            functions](../applications/python/divisive_clustering_coresets.html#Quantum-functions){.reference
            .internal}
        -   [Divisive Clustering
            Function](../applications/python/divisive_clustering_coresets.html#Divisive-Clustering-Function){.reference
            .internal}
        -   [QAOA
            Implementation](../applications/python/divisive_clustering_coresets.html#QAOA-Implementation){.reference
            .internal}
        -   [Scaling simulations with
            CUDA-Q](../applications/python/divisive_clustering_coresets.html#Scaling-simulations-with-CUDA-Q){.reference
            .internal}
    -   [Hybrid Quantum Neural
        Networks](../applications/python/hybrid_quantum_neural_networks.html){.reference
        .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](../applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](../applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using [`Sample`{.docutils .literal .notranslate}]{.pre} to
            perform the Hadamard
            test](../applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](../applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](../applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Anderson Impurity Model ground state solver on Infleqtion's
        Sqale](../applications/python/logical_aim_sqale.html){.reference
        .internal}
        -   [Performing logical Variational Quantum Eigensolver (VQE)
            with
            CUDA-QX](../applications/python/logical_aim_sqale.html#Performing-logical-Variational-Quantum-Eigensolver-(VQE)-with-CUDA-QX){.reference
            .internal}
        -   [Constructing circuits in the [`[[4,2,2]]`{.docutils
            .literal .notranslate}]{.pre}
            encoding](../applications/python/logical_aim_sqale.html#Constructing-circuits-in-the-%5B%5B4,2,2%5D%5D-encoding){.reference
            .internal}
        -   [Setting up submission and decoding
            workflow](../applications/python/logical_aim_sqale.html#Setting-up-submission-and-decoding-workflow){.reference
            .internal}
        -   [Running a CUDA-Q noisy
            simulation](../applications/python/logical_aim_sqale.html#Running-a-CUDA-Q-noisy-simulation){.reference
            .internal}
        -   [Running logical AIM on Infleqtion's
            hardware](../applications/python/logical_aim_sqale.html#Running-logical-AIM-on-Infleqtion's-hardware){.reference
            .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](../applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](../applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](../applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](../applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](../applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](../applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](../applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](../applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. [`Compute`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`overlap`{.docutils .literal
                .notranslate}]{.pre}](../applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](../applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](../applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](../applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](../applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](../applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](../applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](../applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](../applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](../applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum Fourier
        Transform](../applications/python/quantum_fourier_transform.html){.reference
        .internal}
        -   [Quantum Fourier Transform
            revisited](../applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform-revisited){.reference
            .internal}
    -   [Quantum
        Teleporation](../applications/python/quantum_teleportation.html){.reference
        .internal}
        -   [Teleportation
            explained](../applications/python/quantum_teleportation.html#Teleportation-explained){.reference
            .internal}
    -   [Quantum
        Volume](../applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](../applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](../applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](../applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Compiling Unitaries Using Diffusion
        Models](../applications/python/unitary_compilation_diffusion_models.html){.reference
        .internal}
        -   [Diffusion model
            pipeline](../applications/python/unitary_compilation_diffusion_models.html#Diffusion-model-pipeline){.reference
            .internal}
        -   [Setup and load
            models](../applications/python/unitary_compilation_diffusion_models.html#Setup-and-load-models){.reference
            .internal}
            -   [Load discrete
                model](../applications/python/unitary_compilation_diffusion_models.html#Load-discrete-model){.reference
                .internal}
            -   [Load continuous
                model](../applications/python/unitary_compilation_diffusion_models.html#Load-continuous-model){.reference
                .internal}
            -   [Create helper
                functions](../applications/python/unitary_compilation_diffusion_models.html#Create-helper-functions){.reference
                .internal}
        -   [Unitary
            compilation](../applications/python/unitary_compilation_diffusion_models.html#Unitary-compilation){.reference
            .internal}
            -   [Random
                unitary](../applications/python/unitary_compilation_diffusion_models.html#Random-unitary){.reference
                .internal}
            -   [Discrete
                model](../applications/python/unitary_compilation_diffusion_models.html#Discrete-model){.reference
                .internal}
            -   [Continuous
                model](../applications/python/unitary_compilation_diffusion_models.html#Continuous-model){.reference
                .internal}
            -   [Quantum Fourier
                transform](../applications/python/unitary_compilation_diffusion_models.html#Quantum-Fourier-transform){.reference
                .internal}
            -   [XXZ-Hamiltonian
                evolution](../applications/python/unitary_compilation_diffusion_models.html#XXZ-Hamiltonian-evolution){.reference
                .internal}
        -   [Choosing the circuit you
            need](../applications/python/unitary_compilation_diffusion_models.html#Choosing-the-circuit-you-need){.reference
            .internal}
    -   [VQE with gradients, active spaces, and gate
        fusion](../applications/python/vqe_advanced.html){.reference
        .internal}
        -   [The Basics of
            VQE](../applications/python/vqe_advanced.html#The-Basics-of-VQE){.reference
            .internal}
        -   [Installing/Loading Relevant
            Packages](../applications/python/vqe_advanced.html#Installing/Loading-Relevant-Packages){.reference
            .internal}
        -   [Implementing VQE in
            CUDA-Q](../applications/python/vqe_advanced.html#Implementing-VQE-in-CUDA-Q){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](../applications/python/vqe_advanced.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
        -   [Using an Active
            Space](../applications/python/vqe_advanced.html#Using-an-Active-Space){.reference
            .internal}
        -   [Gate Fusion for Larger
            Circuits](../applications/python/vqe_advanced.html#Gate-Fusion-for-Larger-Circuits){.reference
            .internal}
    -   [Quantum
        Transformer](../applications/python/quantum_transformer.html){.reference
        .internal}
        -   [Installation](../applications/python/quantum_transformer.html#Installation){.reference
            .internal}
        -   [Algorithm and
            Example](../applications/python/quantum_transformer.html#Algorithm-and-Example){.reference
            .internal}
            -   [Creating the self-attention
                circuits](../applications/python/quantum_transformer.html#Creating-the-self-attention-circuits){.reference
                .internal}
        -   [Usage](../applications/python/quantum_transformer.html#Usage){.reference
            .internal}
            -   [Model
                Training](../applications/python/quantum_transformer.html#Model-Training){.reference
                .internal}
            -   [Generating
                Molecules](../applications/python/quantum_transformer.html#Generating-Molecules){.reference
                .internal}
            -   [Attention
                Maps](../applications/python/quantum_transformer.html#Attention-Maps){.reference
                .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](../applications/python/afqmc.html){.reference .internal}
        -   [Hamiltonian preparation for
            VQE](../applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](../applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](../applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](../applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](../applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](../applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [ADAPT-QAOA
        algorithm](../applications/python/adapt_qaoa.html){.reference
        .internal}
        -   [Simulation
            input:](../applications/python/adapt_qaoa.html#Simulation-input:){.reference
            .internal}
        -   [The problem Hamiltonian [\\(H_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](../applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A_j\\)]{.math .notranslate
            .nohighlight}:](../applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H_C,A_j\]\\)]{.math .notranslate
            .nohighlight}:](../applications/python/adapt_qaoa.html#The-commutator-%5BH_C,A_j%5D:){.reference
            .internal}
        -   [Beginning of ADAPT-QAOA
            iteration:](../applications/python/adapt_qaoa.html#Beginning-of-ADAPT-QAOA-iteration:){.reference
            .internal}
    -   [ADAPT-VQE
        algorithm](../applications/python/adapt_vqe.html){.reference
        .internal}
        -   [Classical
            pre-processing](../applications/python/adapt_vqe.html#Classical-pre-processing){.reference
            .internal}
        -   [Jordan
            Wigner:](../applications/python/adapt_vqe.html#Jordan-Wigner:){.reference
            .internal}
        -   [UCCSD operator
            pool](../applications/python/adapt_vqe.html#UCCSD-operator-pool){.reference
            .internal}
            -   [Single
                excitation](../applications/python/adapt_vqe.html#Single-excitation){.reference
                .internal}
            -   [Double
                excitation](../applications/python/adapt_vqe.html#Double-excitation){.reference
                .internal}
        -   [Commutator \[[\\(H\\)]{.math .notranslate .nohighlight},
            [\\(A_i\\)]{.math .notranslate
            .nohighlight}\]](../applications/python/adapt_vqe.html#Commutator-%5BH,-A_i%5D){.reference
            .internal}
        -   [Reference
            State:](../applications/python/adapt_vqe.html#Reference-State:){.reference
            .internal}
        -   [Quantum
            kernels:](../applications/python/adapt_vqe.html#Quantum-kernels:){.reference
            .internal}
        -   [Beginning of
            ADAPT-VQE:](../applications/python/adapt_vqe.html#Beginning-of-ADAPT-VQE:){.reference
            .internal}
    -   [Quantum edge
        detection](../applications/python/edge_detection.html){.reference
        .internal}
        -   [Image](../applications/python/edge_detection.html#Image){.reference
            .internal}
        -   [Quantum Probability Image Encoding
            (QPIE):](../applications/python/edge_detection.html#Quantum-Probability-Image-Encoding-(QPIE):){.reference
            .internal}
            -   [Below we show how to encode an image using QPIE in
                cudaq.](../applications/python/edge_detection.html#Below-we-show-how-to-encode-an-image-using-QPIE-in-cudaq.){.reference
                .internal}
        -   [Flexible Representation of Quantum Images
            (FRQI):](../applications/python/edge_detection.html#Flexible-Representation-of-Quantum-Images-(FRQI):){.reference
            .internal}
            -   [Building the FRQI
                State:](../applications/python/edge_detection.html#Building-the-FRQI-State:){.reference
                .internal}
        -   [Quantum Hadamard Edge Detection
            (QHED)](../applications/python/edge_detection.html#Quantum-Hadamard-Edge-Detection-(QHED)){.reference
            .internal}
            -   [Post-processing](../applications/python/edge_detection.html#Post-processing){.reference
                .internal}
    -   [Factoring Integers With Shor's
        Algorithm](../applications/python/shors.html){.reference
        .internal}
        -   [Shor's
            algorithm](../applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](../applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](../applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](../applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](../applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](../applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](../applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](../applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
            -   [(a) Generate the molecular Hamiltonian using Restricted
                Hartree Fock molecular
                orbitals](../applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Restricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(b) Generate the molecular Hamiltonian using
                Unrestricted Hartree Fock molecular
                orbitals](../applications/python/generate_fermionic_ham.html#(b)-Generate-the-molecular-Hamiltonian-using-Unrestricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(a) Generate the active space hamiltonian using RHF
                molecular
                orbitals.](../applications/python/generate_fermionic_ham.html#(a)-Generate-the-active-space-hamiltonian-using-RHF-molecular-orbitals.){.reference
                .internal}
            -   [(b) Generate the active space Hamiltonian using the
                natural orbitals computed from MP2
                simulation](../applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
                .internal}
            -   [(c) Generate the active space Hamiltonian computed from
                the CASSCF molecular
                orbitals](../applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
                .internal}
            -   [(d) Generate the electronic Hamiltonian using
                ROHF](../applications/python/generate_fermionic_ham.html#(d)-Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
            -   [(e) Generate electronic Hamiltonian using
                UHF](../applications/python/generate_fermionic_ham.html#(e)-Generate-electronic-Hamiltonian-using-UHF){.reference
                .internal}
    -   [Grover's
        Algorithm](../applications/python/grovers.html){.reference
        .internal}
        -   [Overview](../applications/python/grovers.html#Overview){.reference
            .internal}
        -   [Problem](../applications/python/grovers.html#Problem){.reference
            .internal}
        -   [Structure of Grover's
            Algorithm](../applications/python/grovers.html#Structure-of-Grover's-Algorithm){.reference
            .internal}
            -   [Step 1:
                Preparation](../applications/python/grovers.html#Step-1:-Preparation){.reference
                .internal}
            -   [Good and Bad
                States](../applications/python/grovers.html#Good-and-Bad-States){.reference
                .internal}
            -   [Step 2: Oracle
                application](../applications/python/grovers.html#Step-2:-Oracle-application){.reference
                .internal}
            -   [Step 3: Amplitude
                amplification](../applications/python/grovers.html#Step-3:-Amplitude-amplification){.reference
                .internal}
            -   [Steps 4 and 5: Iteration and
                measurement](../applications/python/grovers.html#Steps-4-and-5:-Iteration-and-measurement){.reference
                .internal}
    -   [Quantum
        PageRank](../applications/python/quantum_pagerank.html){.reference
        .internal}
        -   [Problem
            Definition](../applications/python/quantum_pagerank.html#Problem-Definition){.reference
            .internal}
        -   [Simulating Quantum PageRank by CUDA-Q
            dynamics](../applications/python/quantum_pagerank.html#Simulating-Quantum-PageRank-by-CUDA-Q-dynamics){.reference
            .internal}
        -   [Breakdown of
            Terms](../applications/python/quantum_pagerank.html#Breakdown-of-Terms){.reference
            .internal}
    -   [The UCCSD Wavefunction
        ansatz](../applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](../applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](../applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](../applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](../applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](../applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](../applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [QM/MM simulation: VQE within a Polarizable Embedded
        Framework.](../applications/python/qm_mm_pe.html){.reference
        .internal}
        -   [Key
            concepts:](../applications/python/qm_mm_pe.html#Key-concepts:){.reference
            .internal}
        -   [PE-VQE-SCF Algorithm
            Steps](../applications/python/qm_mm_pe.html#PE-VQE-SCF-Algorithm-Steps){.reference
            .internal}
            -   [Step 1: Initialize (Classical
                pre-processing)](../applications/python/qm_mm_pe.html#Step-1:-Initialize-(Classical-pre-processing)){.reference
                .internal}
            -   [Step 2: Build the
                Hamiltonian](../applications/python/qm_mm_pe.html#Step-2:-Build-the-Hamiltonian){.reference
                .internal}
            -   [Step 3: Run
                VQE](../applications/python/qm_mm_pe.html#Step-3:-Run-VQE){.reference
                .internal}
            -   [Step 4: Update
                Environment](../applications/python/qm_mm_pe.html#Step-4:-Update-Environment){.reference
                .internal}
            -   [Step 5: Self-Consistency
                Loop](../applications/python/qm_mm_pe.html#Step-5:-Self-Consistency-Loop){.reference
                .internal}
            -   [Requirments:](../applications/python/qm_mm_pe.html#Requirments:){.reference
                .internal}
            -   [Example 1: LiH with 2 water
                molecules.](../applications/python/qm_mm_pe.html#Example-1:-LiH-with-2-water-molecules.){.reference
                .internal}
            -   [VQE, update environment, and scf
                loop.](../applications/python/qm_mm_pe.html#VQE,-update-environment,-and-scf-loop.){.reference
                .internal}
            -   [Example 2: NH3 with 46 water molecule using active
                space.](../applications/python/qm_mm_pe.html#Example-2:-NH3-with-46-water-molecule-using-active-space.){.reference
                .internal}
    -   [Sample-Based Krylov Quantum Diagonalization
        (SKQD)](../applications/python/skqd.html){.reference .internal}
        -   [Why
            SKQD?](../applications/python/skqd.html#Why-SKQD?){.reference
            .internal}
        -   [Setup and
            Imports](../applications/python/skqd.html#Setup-and-Imports){.reference
            .internal}
        -   [Understanding Krylov
            Subspaces](../applications/python/skqd.html#Understanding-Krylov-Subspaces){.reference
            .internal}
            -   [What is a Krylov
                Subspace?](../applications/python/skqd.html#What-is-a-Krylov-Subspace?){.reference
                .internal}
            -   [The SKQD
                Algorithm](../applications/python/skqd.html#The-SKQD-Algorithm){.reference
                .internal}
        -   [Krylov State Generation via Repeated
            Evolution](../applications/python/skqd.html#Krylov-State-Generation-via-Repeated-Evolution){.reference
            .internal}
        -   [Quantum Measurements and
            Sampling](../applications/python/skqd.html#Quantum-Measurements-and-Sampling){.reference
            .internal}
            -   [The Sampling
                Process](../applications/python/skqd.html#The-Sampling-Process){.reference
                .internal}
        -   [Classical Post-Processing and
            Diagonalization](../applications/python/skqd.html#Classical-Post-Processing-and-Diagonalization){.reference
            .internal}
            -   [The SKQD Algorithm: Matrix Construction
                Details](../applications/python/skqd.html#The-SKQD-Algorithm:-Matrix-Construction-Details){.reference
                .internal}
        -   [Results Analysis and
            Convergence](../applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](../applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
    -   [Entanglement Accelerates Quantum
        Simulation](../applications/python/entanglement_acc_hamiltonian_simulation.html){.reference
        .internal}
        -   [2. Model
            Definition](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.-Model-Definition){.reference
            .internal}
            -   [2.1 Initial product
                state](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.1-Initial-product-state){.reference
                .internal}
            -   [2.2 QIMF
                Hamiltonian](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.2-QIMF-Hamiltonian){.reference
                .internal}
            -   [2.3 First-Order Trotter Formula
                (PF1)](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.3-First-Order-Trotter-Formula-(PF1)){.reference
                .internal}
            -   [2.4 PF1 step for the QIMF
                partition](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.4-PF1-step-for-the-QIMF-partition){.reference
                .internal}
            -   [2.5 Hamiltonian
                helpers](../applications/python/entanglement_acc_hamiltonian_simulation.html#2.5-Hamiltonian-helpers){.reference
                .internal}
        -   [3. Entanglement
            metrics](../applications/python/entanglement_acc_hamiltonian_simulation.html#3.-Entanglement-metrics){.reference
            .internal}
        -   [4. Simulation
            workflow](../applications/python/entanglement_acc_hamiltonian_simulation.html#4.-Simulation-workflow){.reference
            .internal}
            -   [4.1 Single-step Trotter
                error](../applications/python/entanglement_acc_hamiltonian_simulation.html#4.1-Single-step-Trotter-error){.reference
                .internal}
            -   [4.2 Dual trajectory
                update](../applications/python/entanglement_acc_hamiltonian_simulation.html#4.2-Dual-trajectory-update){.reference
                .internal}
        -   [5. Reproducing the paper's Figure
            1a](../applications/python/entanglement_acc_hamiltonian_simulation.html#5.-Reproducing-the-paper’s-Figure-1a){.reference
            .internal}
            -   [5.1 Visualising the joint
                behaviour](../applications/python/entanglement_acc_hamiltonian_simulation.html#5.1-Visualising-the-joint-behaviour){.reference
                .internal}
            -   [5.2 Interpreting the
                result](../applications/python/entanglement_acc_hamiltonian_simulation.html#5.2-Interpreting-the-result){.reference
                .internal}
        -   [6. References and further
            reading](../applications/python/entanglement_acc_hamiltonian_simulation.html#6.-References-and-further-reading){.reference
            .internal}
-   [Backends](../using/backends/backends.html){.reference .internal}
    -   [Circuit
        Simulation](../using/backends/simulators.html){.reference
        .internal}
        -   [State Vector
            Simulators](../using/backends/sims/svsims.html){.reference
            .internal}
            -   [CPU](../using/backends/sims/svsims.html#cpu){.reference
                .internal}
            -   [Single-GPU](../using/backends/sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](../using/backends/sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network
            Simulators](../using/backends/sims/tnsims.html){.reference
            .internal}
            -   [Multi-GPU
                multi-node](../using/backends/sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](../using/backends/sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](../using/backends/sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU
            Simulators](../using/backends/sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](../using/backends/sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](../using/backends/sims/mqpusims.html#multi-qpu-other-backends){.reference
                .internal}
        -   [Noisy
            Simulators](../using/backends/sims/noisy.html){.reference
            .internal}
            -   [Trajectory Noisy
                Simulation](../using/backends/sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](../using/backends/sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](../using/backends/sims/noisy.html#stim){.reference
                .internal}
        -   [Photonics
            Simulators](../using/backends/sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](../using/backends/sims/photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware
        (QPUs)](../using/backends/hardware.html){.reference .internal}
        -   [Ion Trap
            QPUs](../using/backends/hardware/iontrap.html){.reference
            .internal}
            -   [IonQ](../using/backends/hardware/iontrap.html#ionq){.reference
                .internal}
            -   [Quantinuum](../using/backends/hardware/iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting
            QPUs](../using/backends/hardware/superconducting.html){.reference
            .internal}
            -   [Anyon Technologies/Anyon
                Computing](../using/backends/hardware/superconducting.html#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](../using/backends/hardware/superconducting.html#iqm){.reference
                .internal}
            -   [OQC](../using/backends/hardware/superconducting.html#oqc){.reference
                .internal}
            -   [Quantum Circuits,
                Inc.](../using/backends/hardware/superconducting.html#quantum-circuits-inc){.reference
                .internal}
        -   [Neutral Atom
            QPUs](../using/backends/hardware/neutralatom.html){.reference
            .internal}
            -   [Infleqtion](../using/backends/hardware/neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](../using/backends/hardware/neutralatom.html#pasqal){.reference
                .internal}
            -   [QuEra
                Computing](../using/backends/hardware/neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic
            QPUs](../using/backends/hardware/photonic.html){.reference
            .internal}
            -   [ORCA
                Computing](../using/backends/hardware/photonic.html#orca-computing){.reference
                .internal}
        -   [Quantum Control
            Systems](../using/backends/hardware/qcontrol.html){.reference
            .internal}
            -   [Quantum
                Machines](../using/backends/hardware/qcontrol.html#quantum-machines){.reference
                .internal}
    -   [Dynamics
        Simulation](../using/backends/dynamics_backends.html){.reference
        .internal}
    -   [Cloud](../using/backends/cloud.html){.reference .internal}
        -   [Amazon Braket
            (braket)](../using/backends/cloud/braket.html){.reference
            .internal}
            -   [Setting
                Credentials](../using/backends/cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submission from
                C++](../using/backends/cloud/braket.html#submission-from-c){.reference
                .internal}
            -   [Submission from
                Python](../using/backends/cloud/braket.html#submission-from-python){.reference
                .internal}
        -   [NVIDIA Quantum Cloud
            (nvqc)](../using/backends/cloud/nvqc.html){.reference
            .internal}
            -   [Quick
                Start](../using/backends/cloud/nvqc.html#quick-start){.reference
                .internal}
            -   [Simulator Backend
                Selection](../using/backends/cloud/nvqc.html#simulator-backend-selection){.reference
                .internal}
            -   [Multiple
                GPUs](../using/backends/cloud/nvqc.html#multiple-gpus){.reference
                .internal}
            -   [Multiple QPUs Asynchronous
                Execution](../using/backends/cloud/nvqc.html#multiple-qpus-asynchronous-execution){.reference
                .internal}
            -   [FAQ](../using/backends/cloud/nvqc.html#faq){.reference
                .internal}
-   [Dynamics](../using/dynamics.html){.reference .internal}
    -   [Quick Start](../using/dynamics.html#quick-start){.reference
        .internal}
    -   [Operator](../using/dynamics.html#operator){.reference
        .internal}
    -   [Time-Dependent
        Dynamics](../using/dynamics.html#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](../using/dynamics.html#super-operator-representation){.reference
        .internal}
    -   [Numerical
        Integrators](../using/dynamics.html#numerical-integrators){.reference
        .internal}
    -   [Batch
        simulation](../using/dynamics.html#batch-simulation){.reference
        .internal}
    -   [Multi-GPU Multi-Node
        Execution](../using/dynamics.html#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](../using/dynamics.html#examples){.reference
        .internal}
-   [CUDA-QX](../using/cudaqx/cudaqx.html){.reference .internal}
    -   [CUDA-Q
        Solvers](../using/cudaqx/cudaqx.html#cuda-q-solvers){.reference
        .internal}
    -   [CUDA-Q QEC](../using/cudaqx/cudaqx.html#cuda-q-qec){.reference
        .internal}
-   [Installation](../using/install/install.html){.reference .internal}
    -   [Local
        Installation](../using/install/local_installation.html){.reference
        .internal}
        -   [Introduction](../using/install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](../using/install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](../using/install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](../using/install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](../using/install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](../using/install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](../using/install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](../using/install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](../using/install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](../using/install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](../using/install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](../using/install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](../using/install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](../using/install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](../using/install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](../using/install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](../using/install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](../using/install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](../using/install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](../using/install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](../using/install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](../using/install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](../using/install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
        -   [Next
            Steps](../using/install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](../using/install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](../using/install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](../using/install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](../using/install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](../using/install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](../using/install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](../using/install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](../using/install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](../using/install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](../using/install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](../using/install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](../using/integration/integration.html){.reference
    .internal}
    -   [Downstream CMake
        Integration](../using/integration/cmake_app.html){.reference
        .internal}
    -   [Combining CUDA with
        CUDA-Q](../using/integration/cuda_gpu.html){.reference
        .internal}
    -   [Integrating with Third-Party
        Libraries](../using/integration/libraries.html){.reference
        .internal}
        -   [Calling a CUDA-Q library from
            C++](../using/integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](../using/integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](../using/integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](../using/extending/extending.html){.reference .internal}
    -   [Add a new Hardware
        Backend](../using/extending/backend.html){.reference .internal}
        -   [Overview](../using/extending/backend.html#overview){.reference
            .internal}
        -   [Server Helper
            Implementation](../using/extending/backend.html#server-helper-implementation){.reference
            .internal}
            -   [Directory
                Structure](../using/extending/backend.html#directory-structure){.reference
                .internal}
            -   [Server Helper
                Class](../using/extending/backend.html#server-helper-class){.reference
                .internal}
            -   [[`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](../using/extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](../using/extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent [`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](../using/extending/backend.html#update-parent-cmakelists-txt){.reference
                .internal}
        -   [Testing](../using/extending/backend.html#testing){.reference
            .internal}
            -   [Unit
                Tests](../using/extending/backend.html#unit-tests){.reference
                .internal}
            -   [Mock
                Server](../using/extending/backend.html#mock-server){.reference
                .internal}
            -   [Python
                Tests](../using/extending/backend.html#python-tests){.reference
                .internal}
            -   [Integration
                Tests](../using/extending/backend.html#integration-tests){.reference
                .internal}
        -   [Documentation](../using/extending/backend.html#documentation){.reference
            .internal}
        -   [Example
            Usage](../using/extending/backend.html#example-usage){.reference
            .internal}
        -   [Code
            Review](../using/extending/backend.html#code-review){.reference
            .internal}
        -   [Maintaining a
            Backend](../using/extending/backend.html#maintaining-a-backend){.reference
            .internal}
        -   [Conclusion](../using/extending/backend.html#conclusion){.reference
            .internal}
    -   [Create a new NVQIR
        Simulator](../using/extending/nvqir_simulator.html){.reference
        .internal}
        -   [[`CircuitSimulator`{.code .docutils .literal
            .notranslate}]{.pre}](../using/extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](../using/extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
    -   [Working with CUDA-Q
        IR](../using/extending/cudaq_ir.html){.reference .internal}
    -   [Create an MLIR Pass for
        CUDA-Q](../using/extending/mlir_pass.html){.reference .internal}
-   [Specifications](../specification/index.html){.reference .internal}
    -   [Language Specification](../specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](../specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](../specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](../specification/cudaq/types.html){.reference
            .internal}
            -   [3.1. [`cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. [`cudaq::qubit`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](../specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](../specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. [`cudaq::spin_op`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](../specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on [`cudaq::qubit`{.code .docutils
                .literal
                .notranslate}]{.pre}](../specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](../specification/cudaq/kernels.html){.reference
            .internal}
        -   [7. Sub-circuit
            Synthesis](../specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](../specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](../specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](../specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](../specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11.
            Platform](../specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](../specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. [`cudaq::sample`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. [`cudaq::run`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. [`cudaq::observe`{.code .docutils .literal
                .notranslate}]{.pre}](../specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. [`cudaq::optimizer`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. [`cudaq::gradient`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](../specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](../specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](../specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](../specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](../specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](../specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](../specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](../specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](../specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](../specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](../specification/quake-dialect.html#motivation){.reference
            .internal}
-   [API Reference](api.html){.reference .internal}
    -   [C++ API](languages/cpp_api.html){.reference .internal}
        -   [Operators](languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Platform](languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](languages/cpp_api.html#namespaces){.reference
            .internal}
    -   [Python API](languages/python_api.html){.reference .internal}
        -   [Program
            Construction](languages/python_api.html#program-construction){.reference
            .internal}
            -   [[`make_kernel()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [[`PyKernel`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [[`Kernel`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [[`PyKernelDecorator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [[`kernel()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [[`sample_async()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [[`run()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [[`run_async()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [[`observe()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [[`observe_async()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [[`get_state()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [[`get_state_async()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [[`vqe()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [[`draw()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [[`translate()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [[`estimate_resources()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
        -   [Backend
            Configuration](languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [[`has_target()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [[`get_target()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [[`get_targets()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [[`set_target()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [[`reset_target()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [[`set_noise()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [[`unset_noise()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [[`register_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [[`unregister_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [[`cudaq.apply_noise()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [[`initialize_cudaq()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [[`num_available_gpus()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [[`set_random_seed()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](languages/python_api.html#dynamics){.reference
            .internal}
            -   [[`evolve()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [[`evolve_async()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [[`Schedule`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [[`BaseIntegrator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [[`InitialState`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [[`InitialStateType`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [[`IntermediateResultSave`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](languages/python_api.html#operators){.reference
            .internal}
            -   [[`OperatorSum`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [[`ProductOperator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [[`ElementaryOperator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [[`ScalarOperator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [[`RydbergHamiltonian`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [[`SuperOperator`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [[`operators.define()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [[`operators.instantiate()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](languages/python_api.html#data-types){.reference
            .internal}
            -   [[`SimulationPrecision`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [[`Target`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [[`State`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [[`Tensor`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [[`QuakeValue`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [[`qubit`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [[`qreg`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [[`qvector`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [[`ComplexMatrix`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [[`SampleResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [[`AsyncSampleResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [[`ObserveResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [[`AsyncObserveResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [[`AsyncStateResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [[`OptimizationResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [[`EvolveResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [[`AsyncEvolveResult`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [[`Resources`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.Resources){.reference
                .internal}
            -   [Optimizers](languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [[`initialize()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [[`rank()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [[`num_ranks()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [[`all_gather()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [[`broadcast()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [[`is_initialized()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [[`finalize()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
    -   [Quantum Operations](#){.current .reference .internal}
        -   [Unitary Operations on
            Qubits](#unitary-operations-on-qubits){.reference .internal}
            -   [[`x`{.code .docutils .literal
                .notranslate}]{.pre}](#x){.reference .internal}
            -   [[`y`{.code .docutils .literal
                .notranslate}]{.pre}](#y){.reference .internal}
            -   [[`z`{.code .docutils .literal
                .notranslate}]{.pre}](#z){.reference .internal}
            -   [[`h`{.code .docutils .literal
                .notranslate}]{.pre}](#h){.reference .internal}
            -   [[`r1`{.code .docutils .literal
                .notranslate}]{.pre}](#r1){.reference .internal}
            -   [[`rx`{.code .docutils .literal
                .notranslate}]{.pre}](#rx){.reference .internal}
            -   [[`ry`{.code .docutils .literal
                .notranslate}]{.pre}](#ry){.reference .internal}
            -   [[`rz`{.code .docutils .literal
                .notranslate}]{.pre}](#rz){.reference .internal}
            -   [[`s`{.code .docutils .literal
                .notranslate}]{.pre}](#s){.reference .internal}
            -   [[`t`{.code .docutils .literal
                .notranslate}]{.pre}](#t){.reference .internal}
            -   [[`swap`{.code .docutils .literal
                .notranslate}]{.pre}](#swap){.reference .internal}
            -   [[`u3`{.code .docutils .literal
                .notranslate}]{.pre}](#u3){.reference .internal}
        -   [Adjoint and Controlled
            Operations](#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on Qubits](#measurements-on-qubits){.reference
            .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](#mz){.reference .internal}
            -   [[`mx`{.code .docutils .literal
                .notranslate}]{.pre}](#mx){.reference .internal}
            -   [[`my`{.code .docutils .literal
                .notranslate}]{.pre}](#my){.reference .internal}
        -   [User-Defined Custom
            Operations](#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](#photonic-operations-on-qudits){.reference
            .internal}
            -   [[`create`{.code .docutils .literal
                .notranslate}]{.pre}](#create){.reference .internal}
            -   [[`annihilate`{.code .docutils .literal
                .notranslate}]{.pre}](#annihilate){.reference .internal}
            -   [[`phase_shift`{.code .docutils .literal
                .notranslate}]{.pre}](#phase-shift){.reference
                .internal}
            -   [[`beam_splitter`{.code .docutils .literal
                .notranslate}]{.pre}](#beam-splitter){.reference
                .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](#id1){.reference .internal}
-   [Other Versions](../versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](../index.html)

::: wy-nav-content
::: rst-content
::: {role="navigation" aria-label="Page navigation"}
-   [](../index.html){.icon .icon-home aria-label="Home"}
-   [Code documentation](api.html)
-   Quantum Operations
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](languages/python_api.html "CUDA-Q Python API"){.btn
.btn-neutral .float-left accesskey="p"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](../versions.html "CUDA-Q Versions"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#quantum-operations .section}
# Quantum Operations[¶](#quantum-operations "Permalink to this heading"){.headerlink}

CUDA-Q provides a default set of quantum operations on qubits. These
operations can be used to define custom kernels and libraries. Since the
set of quantum intrinsic operations natively supported on a specific
target depends on the backends architecture, the [`nvq++`{.code
.docutils .literal .notranslate}]{.pre} compiler automatically
decomposes the default operations into the appropriate set of intrinsic
operations for that target.

The sections [Unitary Operations on
Qubits](#unitary-operations-on-qubits){.reference .internal} and
[Measurements on Qubits](#measurements-on-qubits){.reference .internal}
list the default set of quantum operations on qubits.

Operations that implement unitary transformations of the quantum state
are templated. The template argument allows to invoke the adjoint and
controlled version of the quantum transformation, see the section on
[Adjoint and Controlled
Operations](#adjoint-and-controlled-operations){.reference .internal}.

CUDA-Q additionally provides overloads to support broadcasting of
single-qubit operations across a vector of qubits. For example,
[`x(cudaq::qvector<>&)`{.code .docutils .literal .notranslate}]{.pre}
flips the state of each qubit in the provided [`cudaq::qvector`{.code
.docutils .literal .notranslate}]{.pre}.

::: {#unitary-operations-on-qubits .section}
## Unitary Operations on Qubits[¶](#unitary-operations-on-qubits "Permalink to this heading"){.headerlink}

::: {#x .section}
### [`x`{.code .docutils .literal .notranslate}]{.pre}[¶](#x "Permalink to this heading"){.headerlink}

This operation implements the transformation defined by the Pauli-X
matrix. It is also known as the quantum version of a [`NOT`{.code
.docutils .literal .notranslate}]{.pre}-gate.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # X = | 0  1 |
    #     | 1  0 |
    x(qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // X = | 0  1 |
    //     | 1  0 |
    x(qubit);
:::
:::
:::
:::
:::

::: {#y .section}
### [`y`{.code .docutils .literal .notranslate}]{.pre}[¶](#y "Permalink to this heading"){.headerlink}

This operation implements the transformation defined by the Pauli-Y
matrix.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # Y = | 0  -i |
    #     | i   0 |
    y(qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // Y = | 0  -i |
    //     | i   0 |
    y(qubit);
:::
:::
:::
:::
:::

::: {#z .section}
### [`z`{.code .docutils .literal .notranslate}]{.pre}[¶](#z "Permalink to this heading"){.headerlink}

This operation implements the transformation defined by the Pauli-Z
matrix.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # Z = | 1   0 |
    #     | 0  -1 |
    z(qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // Z = | 1   0 |
    //     | 0  -1 |
    z(qubit);
:::
:::
:::
:::
:::

::: {#h .section}
### [`h`{.code .docutils .literal .notranslate}]{.pre}[¶](#h "Permalink to this heading"){.headerlink}

This operation is a rotation by π about the X+Z axis, and enables one to
create a superposition of computational basis states.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # H = (1 / sqrt(2)) * | 1   1 |
    #                     | 1  -1 |
    h(qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // H = (1 / sqrt(2)) * | 1   1 |
    //                     | 1  -1 |
    h(qubit);
:::
:::
:::
:::
:::

::: {#r1 .section}
### [`r1`{.code .docutils .literal .notranslate}]{.pre}[¶](#r1 "Permalink to this heading"){.headerlink}

This operation is an arbitrary rotation about the [`|1>`{.code .docutils
.literal .notranslate}]{.pre} state.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # R1(λ) = | 1     0    |
    #         | 0  exp(iλ) |
    r1(math.pi, qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // R1(λ) = | 1     0    |
    //         | 0  exp(iλ) |
    r1(std::numbers::pi, qubit);
:::
:::
:::
:::
:::

::: {#rx .section}
### [`rx`{.code .docutils .literal .notranslate}]{.pre}[¶](#rx "Permalink to this heading"){.headerlink}

This operation is an arbitrary rotation about the X axis.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # Rx(θ) = |  cos(θ/2)  -isin(θ/2) |
    #         | -isin(θ/2)  cos(θ/2)  |
    rx(math.pi, qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // Rx(θ) = |  cos(θ/2)  -isin(θ/2) |
    //         | -isin(θ/2)  cos(θ/2)  |
    rx(std::numbers::pi, qubit);
:::
:::
:::
:::
:::

::: {#ry .section}
### [`ry`{.code .docutils .literal .notranslate}]{.pre}[¶](#ry "Permalink to this heading"){.headerlink}

This operation is an arbitrary rotation about the Y axis.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # Ry(θ) = | cos(θ/2)  -sin(θ/2) |
    #         | sin(θ/2)   cos(θ/2) |
    ry(math.pi, qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // Ry(θ) = | cos(θ/2)  -sin(θ/2) |
    //         | sin(θ/2)   cos(θ/2) |
    ry(std::numbers::pi, qubit);
:::
:::
:::
:::
:::

::: {#rz .section}
### [`rz`{.code .docutils .literal .notranslate}]{.pre}[¶](#rz "Permalink to this heading"){.headerlink}

This operation is an arbitrary rotation about the Z axis.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # Rz(λ) = | exp(-iλ/2)      0     |
    #         |     0       exp(iλ/2) |
    rz(math.pi, qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // Rz(λ) = | exp(-iλ/2)      0     |
    //         |     0       exp(iλ/2) |
    rz(std::numbers::pi, qubit);
:::
:::
:::
:::
:::

::: {#s .section}
### [`s`{.code .docutils .literal .notranslate}]{.pre}[¶](#s "Permalink to this heading"){.headerlink}

This operation applies to its target a rotation by π/2 about the Z axis.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # S = | 1   0 |
    #     | 0   i |
    s(qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // S = | 1   0 |
    //     | 0   i |
    s(qubit);
:::
:::
:::
:::
:::

::: {#t .section}
### [`t`{.code .docutils .literal .notranslate}]{.pre}[¶](#t "Permalink to this heading"){.headerlink}

This operation applies to its target a π/4 rotation about the Z axis.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # T = | 1      0     |
    #     | 0  exp(iπ/4) |
    t(qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // T = | 1      0     |
    //     | 0  exp(iπ/4) |
    t(qubit);
:::
:::
:::
:::
:::

::: {#swap .section}
### [`swap`{.code .docutils .literal .notranslate}]{.pre}[¶](#swap "Permalink to this heading"){.headerlink}

This operation swaps the states of two qubits.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit_1, qubit_2 = cudaq.qubit(), cudaq.qubit()

    # Apply the unitary transformation
    # Swap = | 1 0 0 0 |
    #        | 0 0 1 0 |
    #        | 0 1 0 0 |
    #        | 0 0 0 1 |
    swap(qubit_1, qubit_2)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit_1, qubit_2;

    // Apply the unitary transformation
    // Swap = | 1 0 0 0 |
    //        | 0 0 1 0 |
    //        | 0 1 0 0 |
    //        | 0 0 0 1 |
    swap(qubit_1, qubit_2);
:::
:::
:::
:::
:::

::: {#u3 .section}
### [`u3`{.code .docutils .literal .notranslate}]{.pre}[¶](#u3 "Permalink to this heading"){.headerlink}

This operation applies the universal three-parameters operator to target
qubit. The three parameters are Euler angles - theta (θ), phi (φ), and
lambda (λ).

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # U3(θ,φ,λ) = | cos(θ/2)            -exp(iλ) * sin(θ/2)       |
    #             | exp(iφ) * sin(θ/2)   exp(i(λ + φ)) * cos(θ/2) |
    u3(np.pi, np.pi, np.pi / 2, q)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;

    // Apply the unitary transformation
    // U3(θ,φ,λ) = | cos(θ/2)            -exp(iλ) * sin(θ/2)       |
    //             | exp(iφ) * sin(θ/2)   exp(i(λ + φ)) * cos(θ/2) |
    u3(M_PI, M_PI, M_PI_2, q);
:::
:::
:::
:::
:::
:::

::: {#adjoint-and-controlled-operations .section}
## Adjoint and Controlled Operations[¶](#adjoint-and-controlled-operations "Permalink to this heading"){.headerlink}

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
The [`adj`{.code .docutils .literal .notranslate}]{.pre} method of any
gate can be used to invoke the
[adjoint](https://en.wikipedia.org/wiki/Conjugate_transpose){.reference
.external} transformation:

::: {.highlight-python .notranslate}
::: highlight
    # Create a kernel and allocate a qubit in a |0> state.
    qubit = cudaq.qubit()

    # Apply the unitary transformation defined by the matrix
    # T = | 1      0     |
    #     | 0  exp(iπ/4) |
    # to the state of the qubit `q`:
    t(qubit)

    # Apply its adjoint transformation defined by the matrix
    # T† = | 1      0     |
    #      | 0  exp(-iπ/4) |
    t.adj(qubit)
    # `qubit` is now again in the initial state |0>.
:::
:::

The [`ctrl`{.code .docutils .literal .notranslate}]{.pre} method of any
gate can be used to apply the transformation conditional on the state of
one or more control qubits, see also this [Wikipedia
entry](https://en.wikipedia.org/wiki/Quantum_logic_gate#Controlled_gates){.reference
.external}.

::: {.highlight-python .notranslate}
::: highlight
    # Create a kernel and allocate qubits in a |0> state.
    ctrl_1, ctrl_2, target = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
    # Create a superposition.
    h(ctrl_1)
    # `ctrl_1` is now in a state (|0> + |1>) / √2.

    # Apply the unitary transformation
    # | 1  0  0  0 |
    # | 0  1  0  0 |
    # | 0  0  0  1 |
    # | 0  0  1  0 |
    x.ctrl(ctrl_1, ctrl_2)
    # `ctrl_1` and `ctrl_2` are in a state (|00> + |11>) / √2.

    # Set the state of `target` to |1>:
    x(target)
    # Apply the transformation T only if both
    # control qubits are in a |1> state:
    t.ctrl([ctrl_1, ctrl_2], target)
    # The qubits ctrl_1, ctrl_2, and target are now in a state
    # (|000> + exp(iπ/4)|111>) / √2.
:::
:::
:::

C++

::: {.tab-content .docutils}
The template argument [`cudaq::adj`{.code .docutils .literal
.notranslate}]{.pre} can be used to invoke the
[adjoint](https://en.wikipedia.org/wiki/Conjugate_transpose){.reference
.external} transformation:

::: {.highlight-cpp .notranslate}
::: highlight
    // Allocate a qubit in a |0> state.
    cudaq::qubit qubit;

    // Apply the unitary transformation defined by the matrix
    // T = | 1      0     |
    //     | 0  exp(iπ/4) |
    // to the state of the qubit `q`:
    t(qubit);

    // Apply its adjoint transformation defined by the matrix
    // T† = | 1      0     |
    //      | 0  exp(-iπ/4) |
    t<cudaq::adj>(qubit);
    // Qubit `q` is now again in the initial state |0>.
:::
:::

The template argument [`cudaq::ctrl`{.code .docutils .literal
.notranslate}]{.pre} can be used to apply the transformation conditional
on the state of one or more control qubits, see also this [Wikipedia
entry](https://en.wikipedia.org/wiki/Quantum_logic_gate#Controlled_gates){.reference
.external}.

::: {.highlight-cpp .notranslate}
::: highlight
    // Allocate qubits in a |0> state.
    cudaq::qubit ctrl_1, ctrl_2, target;
    // Create a superposition.
    h(ctrl_1);
    // Qubit ctrl_1 is now in a state (|0> + |1>) / √2.

    // Apply the unitary transformation
    // | 1  0  0  0 |
    // | 0  1  0  0 |
    // | 0  0  0  1 |
    // | 0  0  1  0 |
    x<cudaq::ctrl>(ctrl_1, ctrl_2);
    // The qubits ctrl_1 and ctrl_2 are in a state (|00> + |11>) / √2.

    // Set the state of `target` to |1>:
    x(target);
    // Apply the transformation T only if both
    // control qubits are in a |1> state:
    t<cudaq::ctrl>(ctrl_1, ctrl_2, target);
    // The qubits ctrl_1, ctrl_2, and target are now in a state
    // (|000> + exp(iπ/4)|111>) / √2.
:::
:::
:::
:::

Following common convention, by default the transformation is applied to
the target qubit(s) if all control qubits are in a [`|1>`{.code
.docutils .literal .notranslate}]{.pre} state. However, that behavior
can be changed to instead apply the transformation when a control qubit
is in a [`|0>`{.code .docutils .literal .notranslate}]{.pre} state by
negating the polarity of the control qubit. The syntax for negating the
polarity is the not-operator preceding the control qubit:

::: {.tab-set .docutils}
C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit c, q;
    h(c);
    x<cudaq::ctrl>(!c, q);
    // The qubits c and q are in a state (|01> + |10>) / √2.
:::
:::
:::
:::

This notation is only supported in the context of applying a controlled
operation and is only valid for control qubits. For example, negating
either of the target qubits in the [`swap`{.code .docutils .literal
.notranslate}]{.pre} operation is not allowed. Negating the polarity of
control qubits is similarly supported when using [`cudaq::control`{.code
.docutils .literal .notranslate}]{.pre} to conditionally apply a custom
quantum kernel.
:::

::: {#measurements-on-qubits .section}
## Measurements on Qubits[¶](#measurements-on-qubits "Permalink to this heading"){.headerlink}

::: {#mz .section}
### [`mz`{.code .docutils .literal .notranslate}]{.pre}[¶](#mz "Permalink to this heading"){.headerlink}

This operation measures a qubit with respect to the computational basis,
i.e., it projects the state of that qubit onto the eigenvectors of the
Pauli-Z matrix. This is a non-linear transformation, and no template
overloads are available.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()
    mz(qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;
    mz(qubit);
:::
:::
:::
:::
:::

::: {#mx .section}
### [`mx`{.code .docutils .literal .notranslate}]{.pre}[¶](#mx "Permalink to this heading"){.headerlink}

This operation measures a qubit with respect to the Pauli-X basis, i.e.,
it projects the state of that qubit onto the eigenvectors of the Pauli-X
matrix. This is a non-linear transformation, and no template overloads
are available.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()
    mx(qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;
    mx(qubit);
:::
:::
:::
:::
:::

::: {#my .section}
### [`my`{.code .docutils .literal .notranslate}]{.pre}[¶](#my "Permalink to this heading"){.headerlink}

This operation measures a qubit with respect to the Pauli-Y basis, i.e.,
it projects the state of that qubit onto the eigenvectors of the Pauli-Y
matrix. This is a non-linear transformation, and no template overloads
are available.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit = cudaq.qubit()
    kernel.my(qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qubit qubit;
    my(qubit);
:::
:::
:::
:::
:::
:::

::: {#user-defined-custom-operations .section}
## User-Defined Custom Operations[¶](#user-defined-custom-operations "Permalink to this heading"){.headerlink}

Users can define a custom quantum operation by its unitary matrix. First
use the API to register a custom operation, outside of a CUDA-Q kernel.
Then the operation can be used within a CUDA-Q kernel like any of the
built-in operations defined above. Custom operations are supported on
qubits only ([`qudit`{.code .docutils .literal .notranslate}]{.pre} with
[`level`{.code .docutils .literal .notranslate}]{.pre}` `{.code
.docutils .literal .notranslate}[`=`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal .notranslate}[`2`{.code
.docutils .literal .notranslate}]{.pre}).

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
The [`cudaq.register_operation`{.code .docutils .literal
.notranslate}]{.pre} API accepts an identifier string for the custom
operation and its unitary matrix. The matrix can be a [`list`{.code
.docutils .literal .notranslate}]{.pre} or [`numpy`{.code .docutils
.literal .notranslate}]{.pre} array of complex numbers. A 1D matrix is
interpreted as row-major.

::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    import numpy as np

    cudaq.register_operation("custom_h", 1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))

    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        custom_h(qubits[0])
        custom_x.ctrl(qubits[0], qubits[1])

    cudaq.sample(bell).dump()
:::
:::
:::

C++

::: {.tab-content .docutils}
The macro [`CUDAQ_REGISTER_OPERATION`{.code .docutils .literal
.notranslate}]{.pre} accepts a unique name for the operation, the number
of target qubits, the number of rotation parameters (can be 0), and the
unitary matrix as a 1D row-major [`std::vector<complex>`{.code .docutils
.literal .notranslate}]{.pre} representation.

::: {.highlight-cpp .notranslate}
::: highlight
    #include <cudaq.h>

    CUDAQ_REGISTER_OPERATION(custom_h, 1, 0,
                            {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2})

    CUDAQ_REGISTER_OPERATION(custom_x, 1, 0, {0, 1, 1, 0})

    __qpu__ void bell_pair() {
        cudaq::qubit q, r;
        custom_h(q);
        custom_x<cudaq::ctrl>(q, r);
    }

    int main() {
        auto counts = cudaq::sample(bell_pair);
        for (auto &[bits, count] : counts) {
            printf("%s\n", bits.data());
        }
    }
:::
:::
:::
:::

For multi-qubit operations, the matrix is interpreted with MSB qubit
ordering, i.e. big-endian convention. The following example shows two
different custom operations, each operating on 2 qubits.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    import numpy as np

    # Create and test a custom CNOT operation.
    cudaq.register_operation("my_cnot", np.array([1, 0, 0, 0,
                                                  0, 1, 0, 0,
                                                  0, 0, 0, 1,
                                                  0, 0, 1, 0]))

    @cudaq.kernel
    def bell_pair():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        my_cnot(qubits[0], qubits[1]) # `my_cnot(control, target)`


    cudaq.sample(bell_pair).dump() # prints { 11:500 00:500 } (exact numbers will be random)


    # Construct a custom unitary matrix for X on the first qubit and Y
    # on the second qubit.
    X = np.array([[0,  1 ], [1 , 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    XY = np.kron(X, Y)

    # Register the custom operation
    cudaq.register_operation("my_XY", XY)

    @cudaq.kernel
    def custom_xy_test():
        qubits = cudaq.qvector(2)
        my_XY(qubits[0], qubits[1])
        y(qubits[1]) # undo the prior Y gate on qubit 1


    cudaq.sample(custom_xy_test).dump() # prints { 10:1000 }
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    #include <cudaq.h>

    CUDAQ_REGISTER_OPERATION(MyCNOT, 2, 0,
                             {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0});

    CUDAQ_REGISTER_OPERATION(
        MyXY, 2, 0,
        {0, 0, 0, {0, -1}, 0, 0, {0, 1}, 0, 0, {0, -1}, 0, 0, {0, 1}, 0, 0, 0});

    __qpu__ void bell_pair() {
      cudaq::qubit q, r;
      h(q);
      MyCNOT(q, r); // MyCNOT(control, target)
    }

    __qpu__ void custom_xy_test() {
      cudaq::qubit q, r;
      MyXY(q, r);
      y(r); // undo the prior Y gate on qubit 1
    }

    int main() {
      auto counts = cudaq::sample(bell_pair);
      counts.dump(); // prints { 11:500 00:500 } (exact numbers will be random)

      counts = cudaq::sample(custom_xy_test);
      counts.dump(); // prints { 10:1000 }
    }
:::
:::
:::
:::

::: {.admonition .note}
Note

When a custom operation is used on hardware backends, it is synthesized
to a set of native quantum operations. Currently, only 1-qubit and
2-qubit custom operations are supported on hardware backends.
:::
:::

::: {#photonic-operations-on-qudits .section}
## Photonic Operations on Qudits[¶](#photonic-operations-on-qudits "Permalink to this heading"){.headerlink}

These operations are valid only on the [`orca-photonics`{.code .docutils
.literal .notranslate}]{.pre} target which does not support the quantum
operations above.

::: {#create .section}
### [`create`{.code .docutils .literal .notranslate}]{.pre}[¶](#create "Permalink to this heading"){.headerlink}

This operation increments the number of photons in a qumode up to a
maximum value defined by the qudit level that represents the qumode. If
it is applied to a qumode where the number of photons is already at the
maximum value, the operation has no effect.

[\\(C\|0\\rangle → \|1\\rangle, C\|1\\rangle → \|2\\rangle, C\|2\\rangle
→ \|3\\rangle, \\cdots, C\|d\\rangle → \|d\\rangle\\)]{.math
.notranslate .nohighlight} where [\\(d\\)]{.math .notranslate
.nohighlight} is the qudit level.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    q = qudit(3)
    create(q)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qvector<3> q(1);
    create(q[0]);
:::
:::
:::
:::
:::

::: {#annihilate .section}
### [`annihilate`{.code .docutils .literal .notranslate}]{.pre}[¶](#annihilate "Permalink to this heading"){.headerlink}

This operation reduces the number of photons in a qumode up to a minimum
value of 0 representing the vacuum state. If it is applied to a qumode
where the number of photons is already at the minimum value 0, the
operation has no effect.

[\\(A\|0\\rangle → \|0\\rangle, A\|1\\rangle → \|0\\rangle, A\|2\\rangle
→ \|1\\rangle, \\cdots, A\|d\\rangle → \|d-1\\rangle\\)]{.math
.notranslate .nohighlight} where [\\(d\\)]{.math .notranslate
.nohighlight} is the qudit level.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    q = qudit(3)
    annihilate(q)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qvector<3> q(1);
    annihilate(q[0]);
:::
:::
:::
:::
:::

::: {#phase-shift .section}
### [`phase_shift`{.code .docutils .literal .notranslate}]{.pre}[¶](#phase-shift "Permalink to this heading"){.headerlink}

A phase shifter adds a phase [\\(\\phi\\)]{.math .notranslate
.nohighlight} on a qumode. For the annihilation ([\\(a_1\\)]{.math
.notranslate .nohighlight}) and creation operators
([\\(a_1\^\\dagger\\)]{.math .notranslate .nohighlight}) of a qumode,
the phase shift operator is defined by

::: {.math .notranslate .nohighlight}
\\\[P(\\phi) = \\exp\\left(i \\phi a_1\^\\dagger a_1 \\right)\\\]
:::

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    q = qudit(4)
    phase_shift(q, 0.17)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qvector<4> q(1);
    phase_shift(q[0], 0.17);
:::
:::
:::
:::
:::

::: {#beam-splitter .section}
### [`beam_splitter`{.code .docutils .literal .notranslate}]{.pre}[¶](#beam-splitter "Permalink to this heading"){.headerlink}

Beam splitters act on two qumodes together and it is parameterized by a
single angle [\\(\\theta\\)]{.math .notranslate .nohighlight}, relating
to reflectivity. For the annihilation ([\\(a_1\\)]{.math .notranslate
.nohighlight} and [\\(a_2\\)]{.math .notranslate .nohighlight}) and
creation operators ([\\(a_1\^\\dagger\\)]{.math .notranslate
.nohighlight} and [\\(a_2\^\\dagger\\)]{.math .notranslate
.nohighlight}) of two qumodes, the beam splitter operator is defined by

::: {.math .notranslate .nohighlight}
\\\[B(\\theta) = \\exp\\left\[i \\theta (a_1\^\\dagger a_2 + a_1
a_2\^\\dagger) \\right\]\\\]
:::

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    q = [qudit(3) for _ in range(2)]
    beam_splitter(q[0], q[1], 0.34)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qvector<3> q(2);
    beam_splitter(q[0], q[1], 0.34);
:::
:::
:::
:::
:::

::: {#id1 .section}
### [`mz`{.code .docutils .literal .notranslate}]{.pre}[¶](#id1 "Permalink to this heading"){.headerlink}

This operation returns the measurement results of the input qumode(s).

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qumodes = [qudit(3) for _ in range(2)]
    mz(qumodes)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    cudaq::qvector<3> qumodes(2);
    mz(qumodes);
:::
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
Previous](languages/python_api.html "CUDA-Q Python API"){.btn
.btn-neutral .float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](../versions.html "CUDA-Q Versions"){.btn
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
