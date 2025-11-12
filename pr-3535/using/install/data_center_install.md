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
-   [Installation](install.html){.reference .internal}
    -   [Local Installation](local_installation.html){.reference
        .internal}
        -   [Introduction](local_installation.html#introduction){.reference
            .internal}
            -   [Docker](local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX Cloud](local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](local_installation.html#dependencies-and-compatibility){.reference
            .internal}
        -   [Next Steps](local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center Installation](#){.current .reference .internal}
        -   [Prerequisites](#prerequisites){.reference .internal}
        -   [Build Dependencies](#build-dependencies){.reference
            .internal}
            -   [CUDA](#cuda){.reference .internal}
            -   [Toolchain](#toolchain){.reference .internal}
        -   [Building CUDA-Q](#building-cuda-q){.reference .internal}
        -   [Python Support](#python-support){.reference .internal}
        -   [C++ Support](#c-support){.reference .internal}
        -   [Installation on the
            Host](#installation-on-the-host){.reference .internal}
            -   [CUDA Runtime
                Libraries](#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](#mpi){.reference .internal}
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
-   [Installation Guide](install.html)
-   Installation from Source
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](local_installation.html "Local Installation"){.btn
.btn-neutral .float-left accesskey="p"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](../integration/integration.html "Integration with other Software Tools"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#installation-from-source .section}
# Installation from Source[¶](#installation-from-source "Permalink to this heading"){.headerlink}

In most cases, you should not need to build CUDA-Q from source. For the
best experience, we recommend using a container runtime to avoid
conflicts with other software tools installed on the system. Note that
[Singularity](https://docs.sylabs.io/guides/2.6/user-guide/faq.html#what-is-so-special-about-singularity){.reference
.external} or [Docker rootless
mode](https://docs.docker.com/engine/security/rootless/){.reference
.external} address common issue or concerns that are often the
motivation for avoiding the use of containers. Singularity, for example,
can be installed in a user folder and its installation does not require
admin permissions; see [[this section]{.std
.std-ref}](local_installation.html#install-singularity-image){.reference
.internal} for more detailed instructions on how to do that. Our
installation guide also contains instructions for how to [[connect an
IDE]{.std
.std-ref}](local_installation.html#local-development-with-vscode){.reference
.internal} to a running container.

If you do not want use a container runtime, we also provide pre-built
binaries for using CUDA-Q with C++, and Python wheels for using CUDA-Q
with Python. These binaries and wheels are built following the
instructions in this guide and should work for you as long as your
system meets the compatibility requirements listed under
[[Prerequisites]{.std
.std-ref}](#compatibility-prebuilt-binaries){.reference .internal}. To
install the pre-built binaries, please follow the instructions
[[here]{.std
.std-ref}](local_installation.html#install-prebuilt-binaries){.reference
.internal}. To install the Python wheels, please follow the instructions
[[here]{.std
.std-ref}](local_installation.html#install-python-wheels){.reference
.internal}.

If your system is not listed as supported by our official packages, e.g.
because you would like to use CUDA-Q on an operating system that uses an
older C standard library, please follow this guide carefully without
skipping any steps to build and install CUDA-Q from source. The rest of
this guide details system requirements during the build and after
installation, and walks through the installation steps.

::: {.admonition .note}
Note

CUDA-Q contains some components that are only included as pre-built
binaries and not part of our open source repository. We are working on
either open-sourcing these components or making them available as
separate downloads in the future. Even without these components, almost
all features of CUDA-Q will be enabled in a source build, though some
pieces may be less performant. At this time, the [[multi-GPU state
vector simulator]{.std
.std-ref}](../backends/sims/svsims.html#nvidia-mgpu-backend){.reference
.internal} backend will not be included if you build CUDA-Q from source.
:::

::: {#prerequisites .section}
[]{#compatibility-prebuilt-binaries}

## Prerequisites[¶](#prerequisites "Permalink to this heading"){.headerlink}

The following pre-requisites need to be satisfied both on the build
system and on the host system, that is the system where the built CUDA-Q
binaries will be installed and used.

-   Linux operating system. The instructions in this guide have been
    validated with the [AlmaLinux 8
    image](https://hub.docker.com/u/almalinux){.reference .external}
    that serves as the base image for the [manylinux_2_28
    image](https://github.com/pypa/manylinux){.reference .external}, and
    should work for the operating systems CentOS 8, Debian 11 and 12,
    Fedora 41, OpenSUSE/SLED/SLES 15.5 and 15.6, RHEL 8 and 9, Rocky 8
    and 9, Ubuntu 24.04 and 22.04. Other operating systems may work, but
    have not been tested.

-   [Bash](https://www.gnu.org/software/bash/){.reference .external}
    shell. The CUDA-Q build, install and run scripts expect to use
    [`/bin/bash`{.code .docutils .literal .notranslate}]{.pre}.

-   [GNU C library](https://www.gnu.org/software/libc/){.reference
    .external}. Make sure that the version on the host system is the
    same one or newer than the version on the build system. Our own
    builds use version 2.28.

-   CPU with either x86-64 (x86-64-v3 architecture and newer) or ARM64
    (ARM v8-A architecture and newer). Other architectures may work but
    are not tested and may require adjustments to the build
    instructions.

-   Needed **only on the host** system: NVIDIA GPU with Volta, Turing,
    Ampere, Ada, or Hopper architecture and [Compute
    Capability](https://developer.nvidia.com/cuda-gpus){.reference
    .external} 7+. Make sure you have the latest
    [drivers](https://www.nvidia.com/download/index.aspx){.reference
    .external} installed for your GPU, and double check that the driver
    version listed by the [`nvidia-smi`{.code .docutils .literal
    .notranslate}]{.pre} command is 470.57.02 or newer. You do *not*
    need to have a GPU available on the build system; the CUDA compiler
    needed for the build can be installed and used without a GPU.

We strongly recommend using a virtual environment for the build that
includes *only* the tools and dependencies listed in this guide. If you
have additional software installed, you will need to make sure that the
build is linking against the correct libraries and versions.
:::

::: {#build-dependencies .section}
## Build Dependencies[¶](#build-dependencies "Permalink to this heading"){.headerlink}

In addition to the prerequisites listed above, you will need to install
the following prerequisites in your build environment prior to
proceeding with the build as described in the subsequent sections:

-   Python version 3.10 or newer: If you intend to build CUDA-Q with
    Python support, make sure the Python version on the build system
    matches the version on the host system. If you intend to only build
    the C++ support for CUDA-Q, the Python interpreter is required only
    for some of the LLVM build scripts and the Python version used for
    the build does not have to match the version on the host system.

-   Common tools: [`wget`{.code .docutils .literal .notranslate}]{.pre},
    [`git`{.code .docutils .literal .notranslate}]{.pre}, [`unzip`{.code
    .docutils .literal .notranslate}]{.pre}. The commands in the rest of
    this guide assume that these tools are present on the build system,
    but they can be replaced by other alternatives (such as, for
    example, manually going to a web page and downloading a
    file/folder).

The above prerequisites are no longer needed once CUDA-Q is built and do
not need to be present on the host system.

::: {.admonition .note}
Note

The CUDA-Q build scripts and the commands listed in the rest of this
document assume you are using [`bash`{.code .docutils .literal
.notranslate}]{.pre} as the shell for your build.
:::

In addition to installing the needed build dependencies listed above,
make sure to set the following environment variables prior to
proceeding:

::: {.highlight-bash .notranslate}
::: highlight
    export CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
    export CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum
    export CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor
    export LLVM_INSTALL_PREFIX=/usr/local/llvm
    export BLAS_INSTALL_PREFIX=/usr/local/blas
    export ZLIB_INSTALL_PREFIX=/usr/local/zlib
    export OPENSSL_INSTALL_PREFIX=/usr/local/openssl
    export CURL_INSTALL_PREFIX=/usr/local/curl
    export AWS_INSTALL_PREFIX=/usr/local/aws
:::
:::

These environment variables *must* be set during the build. We strongly
recommend that their value is set to a path that does *not* already
exist; this will ensure that these components are built/installed as
needed when building CUDA-Q. The configured paths can be chosen freely,
but the paths specified during the build are also where the
corresponding libraries will be installed on the host system. We are
working on making this more flexible in the future.

::: {.admonition .note}
Note

Please do **not** set [`LLVM_INSTALL_PREFIX`{.code .docutils .literal
.notranslate}]{.pre} to an existing directory; To avoid compatibility
issues, it is important to use the same compiler to build the LLVM/MLIR
dependencies from source as is later used to build CUDA-Q itself.
:::

::: {.admonition .note}
Note

If you are setting the [`CURL_INSTALL_PREFIX`{.code .docutils .literal
.notranslate}]{.pre} variable to an existing CURL installation (not
recommended), please make sure the command [`curl`{.code .docutils
.literal .notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`--version`{.code .docutils .literal .notranslate}]{.pre}
lists HTTP and HTTPS as supported protocols. If these protocols are not
listed, please instead set the [`CURL_INSTALL_PREFIX`{.code .docutils
.literal .notranslate}]{.pre} variable to a path that does *not* exist.
In that case, a suitable library will be automatically built from source
as part of building CUDA-Q.
:::

If you deviate from the instructions below for installing one of the
dependencies and instead install it, for example, via package manager,
you will need to make sure that the installation path matches the path
you set for the corresponding environment variable(s).

::: {#cuda .section}
### CUDA[¶](#cuda "Permalink to this heading"){.headerlink}

Building CUDA-Q requires a full installation of the CUDA toolkit. **You
can install the CUDA toolkit and use the CUDA compiler without having a
GPU.** The instructions are tested using version 12.6 and 13.0, but
other CUDA 12 or 13 versions should work, as long as the CUDA runtime
version on the host system matches the CUDA version used for the build,
and the installed driver on the host system supports that CUDA version.
We recommend using the latest CUDA version that is supported by the
driver on the host system.

Download a suitable [CUDA
version](https://developer.nvidia.com/cuda-toolkit-archive){.reference
.external} following the installation guide for your platform in the
online documentation linked on that page.

Within the tested AlmaLinux 8 environment, for example, the following
commands install CUDA 12.6:

::: {.highlight-bash .notranslate}
::: highlight
    CUDA_VERSION=${CUDA_VERSION:-12.6}
    CUDA_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cuda/repos
    # Go to the url above, set the variables below to a suitable distribution
    # and subfolder for your platform, and uncomment the line below.
    # DISTRIBUTION=rhel8 CUDA_ARCH_FOLDER=x86_64

    dnf config-manager --add-repo "${CUDA_DOWNLOAD_URL}/${DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${DISTRIBUTION}.repo"
    dnf install -y --nobest --setopt=install_weak_deps=False \
        cuda-toolkit-$(echo ${CUDA_VERSION} | tr . -)
    # custatevec is now linked to `libnvidia-ml.so.1`, which is provided in the NVIDIA driver.
    # For build on non-GPU systems, we also need to install the driver. 
    dnf install -y --nobest --setopt=install_weak_deps=False nvidia-driver-libs    
:::
:::
:::

::: {#toolchain .section}
### Toolchain[¶](#toolchain "Permalink to this heading"){.headerlink}

The compiler toolchain used for the build must be a supported [CUDA host
compiler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-host-compilers){.reference
.external} for the installed CUDA version. The following instructions
have been tested with
[GCC-11](https://gcc.gnu.org/index.html){.reference .external}. Other
toolchains may be supported but have not been tested.

Within the tested AlmaLinux 8 environment, for example, the following
commands install GCC 11:

::: {.highlight-bash .notranslate}
::: highlight
    GCC_VERSION=${GCC_VERSION:-11}
    dnf install -y --nobest --setopt=install_weak_deps=False \
        gcc-toolset-${GCC_VERSION}
    # Enabling the toolchain globally is only needed for debug builds
    # to ensure that the correct assembler is picked to process debug symbols.
    enable_script=`find / -path '*gcc*' -path '*'$GCC_VERSIONS'*' -name enable`
    if [ -n "$enable_script" ]; then
        . "$enable_script"
    fi
:::
:::

Independent on which compiler toolchain you installed, set the following
environment variables to point to the respective compilers on your build
system:

::: {.highlight-bash .notranslate}
::: highlight
    export GCC_TOOLCHAIN=/opt/rh/gcc-toolset-11/root/usr/
    export CXX="${GCC_TOOLCHAIN}/bin/g++"
    export CC="${GCC_TOOLCHAIN}/bin/gcc"
    export CUDACXX=/usr/local/cuda/bin/nvcc
    export CUDAHOSTCXX="${GCC_TOOLCHAIN}/bin/g++"
:::
:::

-   The variables [`CC`{.code .docutils .literal .notranslate}]{.pre}
    and [`CXX`{.code .docutils .literal .notranslate}]{.pre} *must* be
    set for the CUDA-Q build.

-   To use GPU-acceleration in CUDA-Q, make sure to set [`CUDACXX`{.code
    .docutils .literal .notranslate}]{.pre} to your CUDA compiler, and
    [`CUDAHOSTCXX`{.code .docutils .literal .notranslate}]{.pre} to the
    CUDA compatible host compiler you are using. If the CUDA compiler is
    not found when building CUDA-Q, some components and backends will be
    omitted automatically during the build.
:::
:::

::: {#building-cuda-q .section}
## Building CUDA-Q[¶](#building-cuda-q "Permalink to this heading"){.headerlink}

This installation guide has been written for a specific version/commit
of CUDA-Q. Make sure to obtain the source code for that version. Clone
the CUDA-Q [GitHub
repository](https://github.com/NVIDIA/cuda-quantum){.reference
.external} and checkout the appropriate branch, tag, or commit. Note
that the build scripts assume that they are run from within a git
repository, and merely downloading the source code as ZIP archive hence
will not work.

Please follow the instructions in the respective subsection(s) to build
the necessary components for using CUDA-Q from C++ and/or Python. After
the build, check that the GPU-accelerated components have been built by
confirming that the file [`nvidia.config`{.code .docutils .literal
.notranslate}]{.pre} exists in the
[`$CUDAQ_INSTALL_PREFIX/targets`{.code .docutils .literal
.notranslate}]{.pre} folder. We also recommend checking the build log
printed to the console to confirm that all desired components have been
built.

::: {.admonition .note}
Note

The CUDA-Q build will compile or omit optional components automatically
depending on whether the necessary pre-requisites are found in the build
environment. If you see a message that a component has been skipped,
and/or the CUDA compiler is not properly detected, make sure you
followed the instructions for installing the necessary prerequisites and
build dependencies, and have set the necessary environment variables as
described in this document.
:::
:::

::: {#python-support .section}
[]{#cudaq-python-from-source}

## Python Support[¶](#python-support "Permalink to this heading"){.headerlink}

The most convenient way to enable Python support within CUDA-Q is to
build a [wheel](https://pythonwheels.com/){.reference .external} that
can then easily be installed using [`pip`{.code .docutils .literal
.notranslate}]{.pre}. To ensure the wheel can be installed on the host
system, make sure to use the same Python version for the build as the
one that is installed on the host system. To build a CUDA-Q Python
wheel, you will need to install the following additional Python-specific
tools:

-   Python development headers: The development headers for your Python
    version are installed in the way as you installed Python itself. If
    you installed Python via the package manager for your system, you
    may need to install an additional package to get the development
    headers. The package name is usually your python version followed by
    either a [`-dev`{.code .docutils .literal .notranslate}]{.pre} or
    [`-devel`{.code .docutils .literal .notranslate}]{.pre} suffix. If
    you are using a [Conda
    environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html){.reference
    .external}, the necessary headers should already be installed.

-   Pip package manager: Make sure the [`pip`{.code .docutils .literal
    .notranslate}]{.pre} module is enable for your Python version, and
    that your [`pip`{.code .docutils .literal .notranslate}]{.pre}
    version is 24 or newer. We refer to the Python
    [documentation](https://pip.pypa.io/en/stable/installation/){.reference
    .external} for more information about installing/enabling
    [`pip`{.code .docutils .literal .notranslate}]{.pre}.

-   Python modules: Install the additional modules [`numpy`{.code
    .docutils .literal .notranslate}]{.pre}, [`build`{.code .docutils
    .literal .notranslate}]{.pre}, [`auditwheel`{.code .docutils
    .literal .notranslate}]{.pre}, and [`patchelf`{.code .docutils
    .literal .notranslate}]{.pre} for your Python version, e.g.
    [`python3`{.code .docutils .literal .notranslate}]{.pre}` `{.code
    .docutils .literal .notranslate}[`-m`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`pip`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`install`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`numpy`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`build`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`auditwheel`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`patchelf`{.code .docutils .literal
    .notranslate}]{.pre}.

::: {.admonition .note}
Note

The wheel build by default is configured to depend on CUDA 13. To build
a wheel for CUDA 12, you need to copy the [`pyproject.toml.cu12`{.code
.docutils .literal .notranslate}]{.pre} file as [`pyproject.toml`{.code
.docutils .literal .notranslate}]{.pre}.
:::

From within the folder where you cloned the CUDA-Q repository, run the
following command to build the CUDA-Q Python wheel:

::: {.highlight-bash .notranslate}
::: highlight
    LLVM_PROJECTS='clang;flang;lld;mlir;python-bindings;openmp;runtimes' \
    bash scripts/install_prerequisites.sh -t llvm && \
    CC="$LLVM_INSTALL_PREFIX/bin/clang" \
    CXX="$LLVM_INSTALL_PREFIX/bin/clang++" \
    FC="$LLVM_INSTALL_PREFIX/bin/flang-new" \
    python3 -m build --wheel
:::
:::

::: {.admonition .note}
Note

A version identifier will be automatically assigned to the wheel based
on the commit history. You can manually override this detection to give
a more descriptive identifier by setting the environment variable
[`SETUPTOOLS_SCM_PRETEND_VERSION`{.code .docutils .literal
.notranslate}]{.pre} to the desired value before building the wheel.
:::

After the initial build,
[auditwheel](https://github.com/pypa/auditwheel){.reference .external}
is used to include dependencies in the wheel, if necessary, and
correctly label the wheel. We recommend not including the CUDA runtime
libraries and instead install them separately on the host system
following the instructions in the next section. The following command
builds the final wheel, not including CUDA dependencies:

::: {.highlight-bash .notranslate}
::: highlight
    CUDAQ_WHEEL="$(find . -name 'cuda_quantum*.whl')" && \
    MANYLINUX_PLATFORM="$(echo ${CUDAQ_WHEEL} | grep -o '[a-z]*linux_[^\.]*' | sed -re 's/^linux_/manylinux_2_28_/')" && \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/_skbuild/lib" \ 
    python3 -m auditwheel -v repair ${CUDAQ_WHEEL} \
        --plat ${MANYLINUX_PLATFORM} \
        --exclude libcublas.so.11 \
        --exclude libcublasLt.so.11 \
        --exclude libcurand.so.10 \
        --exclude libcusolver.so.11 \
        --exclude libcusparse.so.11 \
        --exclude libcutensor.so.2 \
        --exclude libcutensornet.so.2 \
        --exclude libcustatevec.so.1 \
        --exclude libcudensitymat.so.0 \
        --exclude libcudart.so.11.0 \
        --exclude libnvToolsExt.so.1 \
        --exclude libnvidia-ml.so.1 \
        --exclude libcuda.so.1
:::
:::

The command above will create a new wheel in the [`wheelhouse`{.code
.docutils .literal .notranslate}]{.pre} folder. This wheel can be
installed on any [compatible
platform](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/){.reference
.external}.

::: {.admonition .note}
Note

You can confirm that the wheel is indeed compatible with your host
platform by checking that the wheel tag (i.e. the file name ending of
the [`.whl`{.code .docutils .literal .notranslate}]{.pre} file) is
listed under "Compatible Tags" when running the command [`python3`{.code
.docutils .literal .notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`-m`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`pip`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`debug`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`--verbose`{.code .docutils .literal .notranslate}]{.pre}
on the host.
:::
:::

::: {#c-support .section}
[]{#cudaq-cpp-from-source}

## C++ Support[¶](#c-support "Permalink to this heading"){.headerlink}

From within the folder where you cloned the CUDA-Q repository, run the
following command to build CUDA-Q:

::: {.highlight-bash .notranslate}
::: highlight
    CUDAQ_ENABLE_STATIC_LINKING=TRUE \
    CUDAQ_REQUIRE_OPENMP=TRUE \
    CUDAQ_WERROR=TRUE \
    CUDAQ_PYTHON_SUPPORT=OFF \
    LLVM_PROJECTS='clang;flang;lld;mlir;openmp;runtimes' \
    bash scripts/build_cudaq.sh -t llvm -v
:::
:::

Note that [`lld`{.code .docutils .literal .notranslate}]{.pre} is
primarily needed when the build or host system does not already have an
existing default linker on its path; CUDA-Q supports the same linkers as
[`clang`{.code .docutils .literal .notranslate}]{.pre} does.

To easily migrate the built binaries to the host system, we recommend
creating a [self-extracting archive](https://makeself.io/){.reference
.external}. To do so, download the [makeself
script(s)](https://github.com/megastep/makeself){.reference .external}
and move the necessary files to install into a separate folder using the
command

::: {.highlight-bash .notranslate}
::: highlight
    mkdir -p cuda_quantum_assets/llvm/bin && \
    mkdir -p cuda_quantum_assets/llvm/lib && \
    mkdir -p cuda_quantum_assets/llvm/include && \
    mv "${LLVM_INSTALL_PREFIX}/bin/"clang* cuda_quantum_assets/llvm/bin/ && \
    mv cuda_quantum_assets/llvm/bin/clang-format* "${LLVM_INSTALL_PREFIX}/bin/" && \
    mv "${LLVM_INSTALL_PREFIX}/bin/llc" cuda_quantum_assets/llvm/bin/llc && \
    mv "${LLVM_INSTALL_PREFIX}/bin/lld" cuda_quantum_assets/llvm/bin/lld && \
    mv "${LLVM_INSTALL_PREFIX}/bin/ld.lld" cuda_quantum_assets/llvm/bin/ld.lld && \
    mv "${LLVM_INSTALL_PREFIX}/lib/"* cuda_quantum_assets/llvm/lib/ && \
    mv "${LLVM_INSTALL_PREFIX}/include/"* cuda_quantum_assets/llvm/include/ && \
    mv "${CUTENSOR_INSTALL_PREFIX}" cuda_quantum_assets && \
    mv "${CUQUANTUM_INSTALL_PREFIX}" cuda_quantum_assets && \
    mv "${CUDAQ_INSTALL_PREFIX}/build_config.xml" cuda_quantum_assets/build_config.xml && \
    mv "${CUDAQ_INSTALL_PREFIX}" cuda_quantum_assets
:::
:::

You can then create a self-extracting archive with the command

::: {.highlight-bash .notranslate}
::: highlight
    ./makeself.sh --gzip --sha256 --license cuda_quantum_assets/cudaq/LICENSE \
        cuda_quantum_assets install_cuda_quantum.$(uname -m) \
        "CUDA-Q toolkit for heterogeneous quantum-classical workflows" \
        bash cudaq/migrate_assets.sh -t /opt/nvidia/cudaq
:::
:::
:::

::: {#installation-on-the-host .section}
## Installation on the Host[¶](#installation-on-the-host "Permalink to this heading"){.headerlink}

Make sure your host system satisfies the
[Prerequisites](#prerequisites){.reference .internal} listed above.

-   To use CUDA-Q with Python, you should have a working Python
    installation on the host system, including the [`pip`{.code
    .docutils .literal .notranslate}]{.pre} package manager.

-   To use CUDA-Q with C++, you should make sure that you have the
    necessary development headers of the C standard library installed.
    You can check this by searching for [`features.h`{.code .docutils
    .literal .notranslate}]{.pre}, commonly found in
    [`/usr/include/`{.code .docutils .literal .notranslate}]{.pre}. You
    can install the necessary headers via package manager (usually the
    package name is called something like [`glibc-devel`{.code .docutils
    .literal .notranslate}]{.pre} or [`libc6-devel`{.code .docutils
    .literal .notranslate}]{.pre}). These headers are also included with
    any installation of GCC.

If you followed the instructions for building the [[CUDA-Q Python
wheel]{.std .std-ref}](#cudaq-python-from-source){.reference .internal},
copy the built [`.whl`{.code .docutils .literal .notranslate}]{.pre}
file to the host system, and install it using [`pip`{.code .docutils
.literal .notranslate}]{.pre}; e.g.

::: {.highlight-bash .notranslate}
::: highlight
    pip install cuda_quantum*.whl
:::
:::

To install the necessary CUDA and MPI dependencies for some of the
components, you can either follow the instructions on
[PyPI.org](https://pypi.org/project/cudaq/){.reference .external},
replacing [`pip`{.code .docutils .literal .notranslate}]{.pre}` `{.code
.docutils .literal .notranslate}[`install`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`cudaq`{.code .docutils .literal .notranslate}]{.pre} with
the command above, or you can follow the instructions in the remaining
sections of this document to customize and better optimize them for your
host system.

If you followed the instructions for building the [[CUDA-Q C++
tools]{.std .std-ref}](#cudaq-cpp-from-source){.reference .internal},
copy the [`install_cuda_quantum`{.code .docutils .literal
.notranslate}]{.pre} file that you created to the host system, and
install it by running the commands

::: {.highlight-bash .notranslate}
::: highlight
    sudo bash install_cuda_quantum.$(uname -m) --accept
    . /opt/nvidia/cudaq/set_env.sh
:::
:::

This will extract the built assets and move them to the correct
locations. The [`set_env.sh`{.code .docutils .literal
.notranslate}]{.pre} script in [`/opt/nvidia/cudaq`{.code .docutils
.literal .notranslate}]{.pre} defines the necessary environment
variables to use CUDA-Q. To avoid having to set them manually every time
a new shell is opened, we highly recommend adding the following lines to
the [`/etc/profile`{.code .docutils .literal .notranslate}]{.pre} file:

::: {.highlight-bash .notranslate}
::: highlight
    if [ -f /opt/nvidia/cudaq/set_env.sh ];
      . /opt/nvidia/cudaq/set_env.sh
    fi
:::
:::

::: {.admonition .note}
Note

CUDA-Q as built following the instructions above includes and uses the
LLVM C++ standard library. This will not interfere with any other C++
standard library you may have on your system. Pre-built external
libraries, you may want to use with CUDA-Q, such as specific optimizers
for example, have a C API to ensure compatibility across different
versions of the C++ standard library and will work with CUDA-Q without
issues. The same is true for all distributed CUDA libraries. To build
you own CUDA libraries that can be used with CUDA-Q, please take a look
at [[Using CUDA and CUDA-Q in a
Project]{.doc}](../integration/cuda_gpu.html){.reference .internal}.
:::

The remaining sections in this document list additional runtime
dependencies that are not included in the migrated assets and are needed
to use some of the CUDA-Q features and components.

::: {#cuda-runtime-libraries .section}
### CUDA Runtime Libraries[¶](#cuda-runtime-libraries "Permalink to this heading"){.headerlink}

To use GPU-acceleration in CUDA-Q you will need to install the necessary
CUDA runtime libraries. Their version (at least the version major) needs
to match the version used for the build. While not necessary, we
recommend installing the complete CUDA toolkit like you did for the
CUDA-Q build. If you prefer to only install the minimal set of runtime
libraries, the following commands, for example, install the necessary
packages for the AlmaLinux 8 environment:

::: {.highlight-bash .notranslate}
::: highlight
    CUDA_VERSION=${CUDA_VERSION:-12.6}
    CUDA_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cuda/repos
    # Go to the url above, set the variables below to a suitable distribution
    # and subfolder for your platform, and uncomment the line below.
    # DISTRIBUTION=rhel8 CUDA_ARCH_FOLDER=x86_64

    version_suffix=$(echo ${CUDA_VERSION} | tr . -)
    dnf config-manager --add-repo "${CUDA_DOWNLOAD_URL}/${DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${DISTRIBUTION}.repo"
    dnf install -y --nobest --setopt=install_weak_deps=False \
        cuda-cudart-${version_suffix} \
        cuda-nvrtc-${version_suffix} \
        libcusolver-${version_suffix} \
        libcusparse-${version_suffix} \
        libcublas-${version_suffix} \
        libcurand-${version_suffix}
    if [ $(echo ${CUDA_VERSION} | cut -d . -f1) -gt 11 ]; then 
        dnf install -y --nobest --setopt=install_weak_deps=False \
            libnvjitlink-${version_suffix}
    fi
:::
:::
:::

::: {#mpi .section}
### MPI[¶](#mpi "Permalink to this heading"){.headerlink}

To work with all CUDA-Q backends, a CUDA-aware MPI installation is
required. If you do not have an existing CUDA-aware MPI installation,
you can build one from source. To do so, in addition to the CUDA runtime
libraries listed above you will need to install the CUDA runtime
development package ([`cuda-cudart-devel-${version_suffix}`{.code
.docutils .literal .notranslate}]{.pre} or
[`cuda-cudart-dev-${version_suffix}`{.code .docutils .literal
.notranslate}]{.pre}, depending on your distribution).

The following commands build a sufficient CUDA-aware OpenMPI
installation. To make best use of MPI, we recommend a more fully
featured installation including additional configurations that fit your
host system. The commands below assume you have the necessary
prerequisites for the OpenMPI build installed on the build system.
Within the tested AlmaLinux 8 environment, for example, the packages
[`autoconf`{.code .docutils .literal .notranslate}]{.pre},
[`libtool`{.code .docutils .literal .notranslate}]{.pre}, [`flex`{.code
.docutils .literal .notranslate}]{.pre}, and [`make`{.code .docutils
.literal .notranslate}]{.pre} need to be installed.

::: {.highlight-bash .notranslate}
::: highlight
    OPENMPI_VERSION=4.1.4
    OPENMPI_DOWNLOAD_URL=https://github.com/open-mpi/ompi

    wget "${OPENMPI_DOWNLOAD_URL}/archive/v${OPENMPI_VERSION}.tar.gz" -O /tmp/openmpi.tar.gz
    mkdir -p ~/.openmpi-src && tar xf /tmp/openmpi.tar.gz --strip-components 1 -C ~/.openmpi-src
    rm -rf /tmp/openmpi.tar.gz && cd ~/.openmpi-src
    ./autogen.pl 
    LDFLAGS=-Wl,--as-needed ./configure \
        --prefix=/usr/local/openmpi \
        --disable-getpwuid --disable-static \
        --disable-debug --disable-mem-debug --disable-event-debug \
        --disable-mem-profile --disable-memchecker \
        --without-verbs \
        --with-cuda=/usr/local/cuda
    make -j$(nproc) 
    make -j$(nproc) install
    cd - && rm -rf ~/.openmpi-src
:::
:::

Confirm that you have a suitable MPI implementation installed. For
OpenMPI and MPICH, for example, this can be done by compiling and
running the following program:

::: {.highlight-cpp .notranslate}
::: highlight
    // Compile and run with:
    // ```
    // mpic++ mpi_cuda_check.cpp -o check.x && mpiexec -np 1 ./check.x
    // ```

    #include "mpi.h"
    #if __has_include("mpi-ext.h")
    #include "mpi-ext.h"
    #endif
    #include <stdio.h>

    int main(int argc, char *argv[]) {
      MPI_Init(&argc, &argv);
      int exit_code;
      if (MPIX_Query_cuda_support()) {
        printf("CUDA-aware MPI installation.\n");
        exit_code = 0;
      } else {
        printf("Missing CUDA support.\n");
        exit_code = 1;
      }
      MPI_Finalize();
      return exit_code;
    }
:::
:::

::: {.admonition .note}
Note

If you are encountering an error similar to "The value of the MCA
parameter [`plm_rsh_agent`{.code .docutils .literal .notranslate}]{.pre}
was set to a path that could not be found", please make sure you have an
SSH Client installed or update the MCA parameter to another suitable
agent. MPI uses
[SSH](https://en.wikipedia.org/wiki/Secure_Shell){.reference .external}
or [RSH](https://en.wikipedia.org/wiki/Remote_Shell){.reference
.external} to communicate with each node unless another resource
manager, such as
[SLURM](https://slurm.schedmd.com/overview.html){.reference .external},
is used.
:::

Different MPI implementations are supported via a plugin infrastructure
in CUDA-Q. Once you have a CUDA-aware MPI installation on your host
system, you can configure CUDA-Q to use it by activating the necessary
plugin. Plugins for OpenMPI and MPICH are included in CUDA-Q and can be
activated by setting the environment variable [`MPI_PATH`{.code
.docutils .literal .notranslate}]{.pre} to the MPI installation folder
and then running the command

::: {.highlight-bash .notranslate}
::: highlight
    bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh"
:::
:::

::: {.admonition .note}
Note

To activate the MPI plugin for the Python support, replace
[`${CUDA_QUANTUM_PATH}`{.code .docutils .literal .notranslate}]{.pre}
with the path that is listed under "Location" when you run the command
[`pip`{.code .docutils .literal .notranslate}]{.pre}` `{.code .docutils
.literal .notranslate}[`show`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`cuda-quantum`{.code .docutils .literal
.notranslate}]{.pre}.
:::

If you use a different MPI implementation than OpenMPI or MPICH, you
will need to implement the necessary plugin interface yourself prior to
activating the plugin with the command above.
:::
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](local_installation.html "Local Installation"){.btn
.btn-neutral .float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](../integration/integration.html "Integration with other Software Tools"){.btn
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
