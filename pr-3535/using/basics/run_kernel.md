::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../index.html){.icon .icon-home}

::: version
pr-3535
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
-   [Basics](basics.html){.reference .internal}
    -   [What is a CUDA-Q Kernel?](kernel_intro.html){.reference
        .internal}
    -   [Building your first CUDA-Q
        Program](build_kernel.html){.reference .internal}
    -   [Running your first CUDA-Q Program](#){.current .reference
        .internal}
        -   [Sample](#sample){.reference .internal}
        -   [Run](#run){.reference .internal}
        -   [Observe](#observe){.reference .internal}
        -   [Running on a GPU](#running-on-a-gpu){.reference .internal}
    -   [Troubleshooting](troubleshooting.html){.reference .internal}
        -   [Debugging and Verbose Simulation
            Output](troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
-   [Examples](../examples/examples.html){.reference .internal}
    -   [Introduction](../examples/introduction.html){.reference
        .internal}
    -   [Building Kernels](../examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](../examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](../examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](../examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](../examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](../examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](../examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](../examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](../examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](../examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](../examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](../examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](../examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](../examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring
        Kernels](../examples/measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](../examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
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
    -   [Executing
        Kernels](../examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](../examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](../examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](../examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](../examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](../examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](../examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](../examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](../examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](../examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](../examples/expectation_values.html){.reference
        .internal}
        -   [Parallelizing across Multiple
            Processors](../examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](../examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](../examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](../examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](../examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](../examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends ([`remote-mqpu`{.code .docutils
            .literal
            .notranslate}]{.pre})](../examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
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
    -   [Constructing Operators](../examples/operators.html){.reference
        .internal}
        -   [Constructing Spin
            Operators](../examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](../examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](../../examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](../../examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](../examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](../examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](../examples/hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](../examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](../examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](../examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](../examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](../examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](../examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](../examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum Circuits,
            Inc.](../examples/hardware_providers.html#quantum-circuits-inc){.reference
            .internal}
        -   [Quantum
            Machines](../examples/hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](../examples/hardware_providers.html#quera-computing){.reference
            .internal}
    -   [Dynamics
        Examples](../examples/dynamics_examples.html){.reference
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
        -   [Setup and
            Imports](../../applications/python/skqd.html#Setup-and-Imports){.reference
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
-   [CUDA-Q Basics](basics.html)
-   Running your first CUDA-Q Program
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](build_kernel.html "Building your first CUDA-Q Program"){.btn
.btn-neutral .float-left accesskey="p"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](troubleshooting.html "Troubleshooting"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#running-your-first-cuda-q-program .section}
# Running your first CUDA-Q Program[¶](#running-your-first-cuda-q-program "Permalink to this heading"){.headerlink}

Now that you have defined your first quantum kernel, let's look at
different options for how to execute it. In CUDA-Q, quantum circuits are
stored as quantum kernels. For estimating the probability distribution
of a measured quantum state in a circuit, we use the [`sample`{.docutils
.literal .notranslate}]{.pre} function call, for analyzing individual
return values from multiple executions, we use the [`run`{.docutils
.literal .notranslate}]{.pre} function call, and for computing the
expectation value of a quantum state with a given observable, we use the
[`observe`{.docutils .literal .notranslate}]{.pre} function call.

::: {#sample .section}
## Sample[¶](#sample "Permalink to this heading"){.headerlink}

Quantum states collapse upon measurement and hence need to be sampled
many times to gather statistics. The CUDA-Q [`sample`{.code .docutils
.literal .notranslate}]{.pre} call enables this.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
The [[`cudaq.sample()`{.xref .py .py-func .docutils .literal
.notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.sample "cudaq.sample"){.reference
.internal} method takes a kernel and its arguments as inputs, and
returns a [[`cudaq.SampleResult`{.xref .py .py-class .docutils .literal
.notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.SampleResult "cudaq.SampleResult"){.reference
.internal}.
:::

C++

::: {.tab-content .docutils}
The [`cudaq::sample`{.code .docutils .literal .notranslate}]{.pre}
method takes a kernel and its arguments as inputs, and returns a
[`cudaq::SampleResult`{.code .docutils .literal .notranslate}]{.pre}.
:::
:::

This result dictionary contains the distribution of measured states for
the system.

Continuing with the GHZ kernel defined in [[Building Your First CUDA-Q
Program]{.doc}](build_kernel.html){.reference .internal}, we will set
the concrete value of our [`qubit_count`{.code .docutils .literal
.notranslate}]{.pre} to be two.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    qubit_count = 2
    print(cudaq.draw(kernel, qubit_count))
    results = cudaq.sample(kernel, qubit_count)
    # Should see a roughly 50/50 distribution between the |00> and
    # |11> states. Example: {00: 505  11: 495}
    print("Measurement distribution:" + str(results))
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    int main() {

      int qubit_count = 2;
      auto result_0 = cudaq::sample(kernel, /* kernel args */ qubit_count);
      // Should see a roughly 50/50 distribution between the |00> and
      // |11> states. Example: {00: 505  11: 495}
      result_0.dump();
:::
:::
:::
:::

The code above can be run like any other program:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
Assuming the program is saved in the file [`sample.py`{.code .docutils
.literal .notranslate}]{.pre}, we can execute it with the command

::: {.highlight-console .notranslate}
::: highlight
    python3 sample.py
:::
:::
:::

C++

::: {.tab-content .docutils}
Assuming the program is saved in the file [`sample.cpp`{.code .docutils
.literal .notranslate}]{.pre}, we can now compile this file with the
[`nvq++`{.code .docutils .literal .notranslate}]{.pre} toolchain, and
then run the compiled executable.

::: {.highlight-console .notranslate}
::: highlight
    nvq++ sample.cpp
    ./a.out
:::
:::
:::
:::

By default, [`sample`{.code .docutils .literal .notranslate}]{.pre}
produces an ensemble of 1000 shots. This can be changed by specifying an
integer argument for the [`shots_count`{.code .docutils .literal
.notranslate}]{.pre}.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    # With an increased shots count, we will still see the same 50/50 distribution,
    # but now with 10,000 total measurements instead of the default 1000.
    # Example: {00: 5005  11: 4995}
    results = cudaq.sample(kernel, qubit_count, shots_count=10000)
    print("Measurement distribution:" + str(results))
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      // With an increased shots count, we will still see the same 50/50
      // distribution, but now with 10,000 total measurements instead of the default
      // 1000. Example: {00: 5005  11: 4995}
      int shots_count = 10000;
      auto result_1 = cudaq::sample(shots_count, kernel, qubit_count);
      result_1.dump();
:::
:::
:::
:::

Note that there is a subtle difference between how sample is executed
with the target device set to a simulator or with the target device set
to a QPU. When run on a simulator, the quantum state is built once and
then sampled repeatedly, where the number of samples is defined by
[`shots_count`{.code .docutils .literal .notranslate}]{.pre}. When
executed on quantum hardware, the quantum state collapses upon
measurement and hence needs to be rebuilt every time to collect a
sample.

A variety of methods can be used to extract useful information from a
[`SampleResult`{.code .docutils .literal .notranslate}]{.pre}. For
example, to return the most probable measurement and its respective
probability:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    most_probable_result = results.most_probable()
    probability = results.probability(most_probable_result)
    print("Most probable result: " + most_probable_result)
    print("Measured with probability " + str(probability), end='\n\n')
:::
:::

See the [[API
specification]{.doc}](../../api/languages/python_api.html){.reference
.internal} for further information.
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      std::cout << result_1.most_probable() << "\n"; // prints: `00`
      std::cout << result_1.probability(result_1.most_probable())
                << "\n"; // prints: `0.5005`
    }
:::
:::

See the [[API
specification]{.doc}](../../api/languages/cpp_api.html){.reference
.internal} for further information.
:::
:::

Sampling a distribution can be a time intensive task. An asynchronous
version of sample exists and can be useful to parallelize your
application. Asynchronous programming is a technique that enables your
program to start a potentially long-running task and still be able to be
responsive to other events while that task runs, rather than having to
wait until that task has finished. Once that task has finished, your
program is presented with the result.

Asynchronous execution allows to easily parallelize execution of
multiple kernels on a multi-processor platform. Such a platform is
available, for example, by choosing the target [`nvidia-mqpu`{.code
.docutils .literal .notranslate}]{.pre}:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    @cudaq.kernel
    def kernel2(qubit_count: int):
        # Allocate our qubits.
        qvector = cudaq.qvector(qubit_count)
        # Place all qubits in a uniform superposition.
        h(qvector)
        # Measure the qubits.
        mz(qvector)


    num_gpus = cudaq.num_available_gpus()
    if num_gpus > 1:
        # Set the target to include multiple virtual QPUs.
        cudaq.set_target("nvidia", option="mqpu")
        # Asynchronous execution on multiple virtual QPUs, each simulated by an NVIDIA GPU.
        result_1 = cudaq.sample_async(kernel,
                                      qubit_count,
                                      shots_count=1000,
                                      qpu_id=0)
        result_2 = cudaq.sample_async(kernel2,
                                      qubit_count,
                                      shots_count=1000,
                                      qpu_id=1)
    else:
        # Schedule for execution on the same virtual QPU.
        result_1 = cudaq.sample_async(kernel,
                                      qubit_count,
                                      shots_count=1000,
                                      qpu_id=0)
        result_2 = cudaq.sample_async(kernel2,
                                      qubit_count,
                                      shots_count=1000,
                                      qpu_id=0)

    print("Measurement distribution for kernel:" + str(result_1.get()))
    print("Measurement distribution for kernel2:" + str(result_2.get()))
:::
:::
:::
:::

::: {.admonition .note}
Note

This kind of parallelization is most effective if you actually have
multiple QPUs or GPUs available. Otherwise, the sampling will still have
to execute sequentially due to resource constraints.
:::

More information about parallelizing execution can be found on the
[[Simulate Multiple QPUs in Parallel]{.std
.std-ref}](../backends/sims/mqpusims.html#mqpu-platform){.reference
.internal} page.
:::

::: {#run .section}
## Run[¶](#run "Permalink to this heading"){.headerlink}

The [`run`{.code .docutils .literal .notranslate}]{.pre} method executes
a quantum kernel multiple times and returns each individual result.
Unlike [`sample`{.code .docutils .literal .notranslate}]{.pre}, which
collects measurement statistics as counts, [`run`{.code .docutils
.literal .notranslate}]{.pre} preserves each individual return value
from each execution. This is useful when you need to analyze the
distribution of returned values which may not be possible from just
aggregated measurement counts. Additionally, the [`run`{.code .docutils
.literal .notranslate}]{.pre} method also supports returning various
types of values from the quantum kernel, including scalar types (bool,
int, float and their variants) and user-defined data structures.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
The [`cudaq.run`{.docutils .literal .notranslate}]{.pre} method takes a
kernel and its arguments as inputs and returns a list containing the
result values from each execution. The kernel must return a non-void
value.
:::

C++

::: {.tab-content .docutils}
The [`cudaq::run`{.docutils .literal .notranslate}]{.pre} method takes a
kernel and its arguments as inputs and returns a [`std::vector`{.code
.docutils .literal .notranslate}]{.pre} containing the result values
from each execution. The kernel must return a non-void value.
:::
:::

Below is an example of a quantum kernel that creates a GHZ state,
measures all qubits, and returns the total count of qubits in state
[\\(\|1\\rangle\\)]{.math .notranslate .nohighlight}:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq


    # Define a quantum kernel that returns an integer
    @cudaq.kernel
    def ghz_kernel(qubit_count: int) -> int:
        # Allocate qubits
        qubits = cudaq.qvector(qubit_count)

        # Create GHZ state
        h(qubits[0])
        for i in range(1, qubit_count):
            x.ctrl(qubits[0], qubits[i])

        # Measure and count the number of qubits in state |1⟩
        result = 0
        for i in range(qubit_count):
            if mz(qubits[i]):
                result += 1

        return result


    # Execute the kernel multiple times and collect individual results
    qubit_count = 3
    results = cudaq.run(ghz_kernel, qubit_count, shots_count=10)
    print(f"Executed {len(results)} shots")
    print(f"Results: {results}")
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    #include <algorithm>
    #include <cudaq.h>
    #include <iostream>
    #include <map>
    #include <numeric>

    // Define a quantum kernel that returns an integer
    __qpu__ int ghz_kernel(int qubit_count) {
      // Allocate qubits
      cudaq::qvector qubits(qubit_count);

      // Create GHZ state
      h(qubits[0]);
      for (int i = 1; i < qubit_count; ++i) {
        x<cudaq::ctrl>(qubits[0], qubits[i]);
      }

      // Measure and count the number of qubits in state |1⟩
      int result = 0;
      for (int i = 0; i < qubit_count; ++i) {
        if (mz(qubits[i])) {
          result += 1;
        }
      }

      return result;
    }

    int main() {
      // Execute the kernel multiple times and collect individual results
      int qubit_count = 3;
      auto results = cudaq::run(10, ghz_kernel, qubit_count);

      std::cout << "Executed " << results.size() << " shots\n";
      std::cout << "Results: ";
      for (auto result : results) {
        std::cout << result << " ";
      }
      std::cout << "\n";
:::
:::
:::
:::

The code above will execute the kernel multiple times (defined by
[`shots_count`{.code .docutils .literal .notranslate}]{.pre}) and return
a list of individual results. By default, the [`shots_count`{.code
.docutils .literal .notranslate}]{.pre} for [`run`{.code .docutils
.literal .notranslate}]{.pre} is 100.

You can process the results to get statistics or other insights:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    # Count occurrences of each result
    value_counts = {}
    for value in results:
        value_counts[value] = value_counts.get(value, 0) + 1

    print("\nCounts of each result:")
    for value, count in sorted(value_counts.items()):
        print(f"Result {value}: {count} times")

    # Analyze patterns in the results
    zero_count = results.count(0)
    full_count = results.count(qubit_count)
    other_count = len(results) - zero_count - full_count
    print(f"\nGHZ state analysis:")
    print(
        f"  All qubits in |0⟩: {zero_count} times ({zero_count/len(results)*100:.1f}%)"
    )
    print(
        f"  All qubits in |1⟩: {full_count} times ({full_count/len(results)*100:.1f}%)"
    )
    print(
        f"  Other states: {other_count} times ({other_count/len(results)*100:.1f}%)"
    )
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      // Count occurrences of each result
      std::map<int, int> value_counts;
      for (auto value : results) {
        value_counts[value]++;
      }

      std::cout << "\nCounts of each result:\n";
      for (auto &[value, count] : value_counts) {
        std::cout << "Result " << value << ": " << count << " times\n";
      }

      // Analyze patterns in the results
      int zero_count = std::count(results.begin(), results.end(), 0);
      int full_count = std::count(results.begin(), results.end(), qubit_count);
      int other_count = results.size() - zero_count - full_count;

      std::cout << "\nGHZ state analysis:\n";
      std::cout << "  All qubits in |0⟩: " << zero_count << " times ("
                << (float)zero_count / results.size() * 100.0 << "%)\n";
      std::cout << "  All qubits in |1⟩: " << full_count << " times ("
                << (float)full_count / results.size() * 100.0 << "%)\n";
      std::cout << "  Other states: " << other_count << " times ("
                << (float)other_count / results.size() * 100.0 << "%)\n";
:::
:::
:::
:::

::: {.admonition .note}
Note

Currently, [`run`{.code .docutils .literal .notranslate}]{.pre} supports
kernels returning scalar types (bool, int, float) and custom data
structures.
:::

::: {.admonition .note}
Note

When using custom data structures, they must be defined with
[`slots=True`{.code .docutils .literal .notranslate}]{.pre} in Python or
as simple aggregates in C++.
:::

Similar to [`sample_async`{.code .docutils .literal
.notranslate}]{.pre}, the [`run`{.code .docutils .literal
.notranslate}]{.pre} API also supports asynchronous execution through
[`run_async`{.code .docutils .literal .notranslate}]{.pre}. This is
particularly useful for parallelizing execution of multiple kernels on a
multi-processor platform:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    # Define a simple kernel for asynchronous execution
    @cudaq.kernel
    def simple_kernel(theta: float) -> bool:
        q = cudaq.qubit()
        rx(theta, q)
        return mz(q)


    # Check if we have multiple GPUs
    num_gpus = cudaq.num_available_gpus()
    if num_gpus > 1:
        # Set the target to include multiple virtual QPUs
        cudaq.set_target("nvidia", option="mqpu")

        # Run kernels asynchronously with different parameters
        future1 = cudaq.run_async(simple_kernel, 0.0, shots_count=100, qpu_id=0)
        future2 = cudaq.run_async(simple_kernel, 3.14159, shots_count=100, qpu_id=1)
    else:
        # Schedule for execution on the same virtual QPU, defaulting to `qpu_id=0`
        future1 = cudaq.run_async(simple_kernel, 0.0, shots_count=100)
        future2 = cudaq.run_async(simple_kernel, 3.14159, shots_count=100)

    # Get results when ready
    results1 = future1.get()
    results2 = future2.get()

    # Analyze the results
    print("\nAsynchronous execution results:")
    true_count1 = sum(1 for res in results1 if res)
    true_count2 = sum(1 for res in results2 if res)
    print(f"Kernel with theta=0.0: {true_count1}/100 times measured |1⟩")
    print(f"Kernel with theta=π: {true_count2}/100 times measured |1⟩")
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      // Define a simple kernel for async execution
      auto simple_kernel = [](float theta) __qpu__ -> bool {
        cudaq::qubit q;
        rx(theta, q);
        return mz(q);
      };

      // Check if we have multiple QPUs available
      // Note: In C++ API, we would check this differently
      // Here we'll use the target setting directly
      bool has_multiple_qpus = false;

      if (has_multiple_qpus) {
        // Set the target to include multiple virtual QPUs
        // In a real application, this would involve proper target configuration

        // Run kernels asynchronously with different parameters
        auto future1 = cudaq::run_async(0, 100, simple_kernel, 0.0);
        auto future2 = cudaq::run_async(1, 100, simple_kernel, 3.14159);

        // Get results when ready
        auto results1 = future1.get();
        auto results2 = future2.get();

        // Analyze the results
        std::cout << "\nAsynchronous execution results:\n";
        int true_count1 = std::count(results1.begin(), results1.end(), true);
        int true_count2 = std::count(results2.begin(), results2.end(), true);

        std::cout << "Kernel with theta=0.0: " << true_count1
                  << "/100 times measured |1⟩\n";
        std::cout << "Kernel with theta=π: " << true_count2
                  << "/100 times measured |1⟩\n";
      } else {
        // Schedule for execution on the same QPU
        auto future1 = cudaq::run_async(0, 100, simple_kernel, 0.0);
        auto future2 = cudaq::run_async(0, 100, simple_kernel, 3.14159);

        // Get results when ready
        auto results1 = future1.get();
        auto results2 = future2.get();

        // Analyze the results
        std::cout << "\nAsynchronous execution results:\n";
        int true_count1 = std::count(results1.begin(), results1.end(), true);
        int true_count2 = std::count(results2.begin(), results2.end(), true);

        std::cout << "Kernel with theta=0.0: " << true_count1
                  << "/100 times measured |1⟩\n";
        std::cout << "Kernel with theta=π: " << true_count2
                  << "/100 times measured |1⟩\n";
      }
:::
:::
:::
:::

More information about parallelizing execution can be found at the
[[Simulate Multiple QPUs in Parallel]{.std
.std-ref}](../backends/sims/mqpusims.html#mqpu-platform){.reference
.internal} page.

::: {.admonition .note}
Note

Currently, [`run`{.code .docutils .literal .notranslate}]{.pre} and
[`run_async`{.code .docutils .literal .notranslate}]{.pre} are only
supported on simulator targets.
:::
:::

::: {#observe .section}
## Observe[¶](#observe "Permalink to this heading"){.headerlink}

The observe function allows us to calculate expectation values for a
defined quantum operator, that is the value of
[\\(\\bra{\\psi}H\\ket{\\psi}\\)]{.math .notranslate .nohighlight},
where [\\(H\\)]{.math .notranslate .nohighlight} is the desired operator
and [\\(\\ket{\\psi}\\)]{.math .notranslate .nohighlight} is the quantum
state after executing a given kernel.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
The [[`cudaq.observe()`{.xref .py .py-func .docutils .literal
.notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.observe "cudaq.observe"){.reference
.internal} method takes a kernel and its arguments as inputs, along with
a [[`cudaq.operators.spin.SpinOperator`{.xref .py .py-class .docutils
.literal
.notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.operators.spin.SpinOperator "cudaq.operators.spin.SpinOperator"){.reference
.internal}.

Using the [`cudaq.spin`{.code .docutils .literal .notranslate}]{.pre}
module, operators may be defined as a linear combination of Pauli
strings. Functions, such as [[`cudaq.spin.i()`{.xref .py .py-func
.docutils .literal
.notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.spin.i "cudaq.spin.i"){.reference
.internal}, [[`cudaq.spin.x()`{.xref .py .py-func .docutils .literal
.notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.spin.x "cudaq.spin.x"){.reference
.internal}, [[`cudaq.spin.y()`{.xref .py .py-func .docutils .literal
.notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.spin.y "cudaq.spin.y"){.reference
.internal}, [[`cudaq.spin.z()`{.xref .py .py-func .docutils .literal
.notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.spin.z "cudaq.spin.z"){.reference
.internal} may be used to construct more complex spin Hamiltonians on
multiple qubits.
:::

C++

::: {.tab-content .docutils}
The [`cudaq::observe`{.code .docutils .literal .notranslate}]{.pre}
method takes a kernel and its arguments as inputs, along with a
[`cudaq::spin_op`{.code .docutils .literal .notranslate}]{.pre}.

Operators may be defined as a linear combination of Pauli strings.
Functions, such as [`cudaq::spin_op::i`{.code .docutils .literal
.notranslate}]{.pre}, [`cudaq::spin_op::x`{.code .docutils .literal
.notranslate}]{.pre}, [`cudaq::spin_op::y`{.code .docutils .literal
.notranslate}]{.pre}, [`cudaq::spin_op::z`{.code .docutils .literal
.notranslate}]{.pre} may be used to construct more complex spin
Hamiltonians on multiple qubits.
:::
:::

Below is an example of a spin operator object consisting of a
[`Z(0)`{.code .docutils .literal .notranslate}]{.pre} operator, or a
Pauli Z-operator on the qubit zero. This is followed by the construction
of a kernel with a single qubit in an equal superposition. The
Hamiltonian is printed to confirm that it has been constructed properly.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import cudaq
    from cudaq import spin

    operator = spin.z(0)
    print(operator)  # prints: [1+0j] Z


    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        h(qubit)
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
    #include <cudaq.h>
    #include <cudaq/algorithm.h>

    #include <iostream>

    __qpu__ void kernel() {
      cudaq::qubit qubit;
      h(qubit);
    }

    int main() {
      auto spin_operator = cudaq::spin_op::z(0);
      std::cout << spin_operator.to_string() << "\n";
:::
:::
:::
:::

The [`observe`{.code .docutils .literal .notranslate}]{.pre} function
takes a kernel, any kernel arguments, and a spin operator as inputs and
produces an [`ObserveResult`{.code .docutils .literal
.notranslate}]{.pre} object. The expectation value can be printed using
the [`expectation`{.code .docutils .literal .notranslate}]{.pre} method.

::: {.admonition .note}
Note

It is important to exclude a measurement in the kernel, otherwise the
expectation value will be determined from a collapsed classical state.
For this example, the expected result of 0.0 is produced.
:::

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    result = cudaq.observe(kernel, operator)
    print(result.expectation())  # prints: 0.0
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      auto result_0 = cudaq::observe(kernel, spin_operator);
      // Expectation value of kernel with respect to single `Z` term
      // should print: 0.0
      std::cout << "<kernel | spin_operator | kernel> = " << result_0.expectation()
                << "\n";
:::
:::
:::
:::

Unlike [`sample`{.code .docutils .literal .notranslate}]{.pre}, the
default [`shots_count`{.code .docutils .literal .notranslate}]{.pre} for
[`observe`{.code .docutils .literal .notranslate}]{.pre} is 1. This
result is deterministic and equivalent to the expectation value in the
limit of infinite shots. To produce an approximate expectation value
from sampling, [`shots_count`{.code .docutils .literal
.notranslate}]{.pre} can be specified to any integer.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    result = cudaq.observe(kernel, operator, shots_count=1000)
    print(result.expectation())  # prints non-zero value
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-cpp .notranslate}
::: highlight
      auto result_1 = cudaq::observe(1000, kernel, spin_operator);
      // Expectation value of kernel with respect to single `Z` term,
      // but instead of a single deterministic execution of the kernel,
      // we sample over 1000 shots. We should now print an expectation
      // value that is close to, but not quite, zero.
      // Example: 0.025
      std::cout << "<kernel | spin_operator | kernel> = " << result_1.expectation()
                << "\n";
    }
:::
:::
:::
:::

Similar to [`sample_async`{.code .docutils .literal .notranslate}]{.pre}
above, observe also supports asynchronous execution. More information
about parallelizing execution can be found at the [[Simulate Multiple
QPUs in Parallel]{.std
.std-ref}](../backends/sims/mqpusims.html#mqpu-platform){.reference
.internal} page.
:::

::: {#running-on-a-gpu .section}
## Running on a GPU[¶](#running-on-a-gpu "Permalink to this heading"){.headerlink}

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
Using [[`cudaq.set_target()`{.xref .py .py-func .docutils .literal
.notranslate}]{.pre}](../../api/languages/python_api.html#cudaq.set_target "cudaq.set_target"){.reference
.internal}, different targets can be specified for kernel execution.
:::

C++

::: {.tab-content .docutils}
Using the [`--target`{.code .docutils .literal .notranslate}]{.pre}
argument to [`nvq++`{.code .docutils .literal .notranslate}]{.pre},
different targets can be specified for kernel execution.
:::
:::

If a local GPU is detected, the target will default to [`nvidia`{.code
.docutils .literal .notranslate}]{.pre}. Otherwise, the CPU-based
simulation target, [`qpp-cpu`{.code .docutils .literal
.notranslate}]{.pre}, will be selected.

We will demonstrate the benefits of using a GPU by sampling our GHZ
kernel with 25 qubits and a [`shots_count`{.code .docutils .literal
.notranslate}]{.pre} of 1 million. Using a GPU accelerates this task by
more than 35x. To learn about all of the available targets and ways to
accelerate kernel execution, visit the
[[Backends]{.doc}](../backends/backends.html){.reference .internal}
page.

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-python .notranslate}
::: highlight
    import sys
    import timeit

    # Will time the execution of our sample call.
    code_to_time = 'cudaq.sample(kernel, qubit_count, shots_count=1000000)'
    qubit_count = int(sys.argv[1]) if 1 < len(sys.argv) else 25

    # Execute on CPU backend.
    cudaq.set_target('qpp-cpu')
    print('CPU time')  # Example: 27.57462 s.
    print(timeit.timeit(stmt=code_to_time, globals=globals(), number=1))

    if cudaq.num_available_gpus() > 0:
        # Execute on GPU backend.
        cudaq.set_target('nvidia')
        print('GPU time')  # Example: 0.773286 s.
        print(timeit.timeit(stmt=code_to_time, globals=globals(), number=1))
:::
:::
:::

C++

::: {.tab-content .docutils}
To compare the performance, we can create a simple timing script that
isolates just the call to [`cudaq::sample`{.code .docutils .literal
.notranslate}]{.pre}. We are still using the same GHZ kernel as earlier,
but the following modification is made to the main function:

::: {.highlight-cpp .notranslate}
::: highlight
    int main(int argc, char *argv[]) {
      auto qubit_count = 1 < argc ? atoi(argv[1]) : 25;
      auto shots_count = 1000000;
      auto start = std::chrono::high_resolution_clock::now();

      // Timing just the sample execution.
      auto result = cudaq::sample(shots_count, kernel, qubit_count);

      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double>(stop - start);
      std::cout << "It took " << duration.count() << " seconds.\n";
    }
:::
:::

First we execute on the CPU backend:

::: {.highlight-console .notranslate}
::: highlight
    nvq++ --target=qpp-cpu sample.cpp
    ./a.out
:::
:::

seeing an output of the order: [`It`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`took`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`22.8337`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`seconds.`{.docutils .literal .notranslate}]{.pre}

Now we can execute on the GPU enabled backend:

::: {.highlight-console .notranslate}
::: highlight
    nvq++ --target=nvidia sample.cpp
    ./a.out
:::
:::

seeing an output of the order: [`It`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`took`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`3.18988`{.docutils .literal
.notranslate}]{.pre}` `{.docutils .literal
.notranslate}[`seconds.`{.docutils .literal .notranslate}]{.pre}
:::
:::
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](build_kernel.html "Building your first CUDA-Q Program"){.btn
.btn-neutral .float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](troubleshooting.html "Troubleshooting"){.btn
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
