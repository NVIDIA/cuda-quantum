::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](../../index.html){.icon .icon-home}

::: version
pr-3592
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
    -   [Local Installation](#){.current .reference .internal}
        -   [Introduction](#introduction){.reference .internal}
            -   [Docker](#docker){.reference .internal}
            -   [Known Blackwell
                Issues](#known-blackwell-issues){.reference .internal}
            -   [Singularity](#singularity){.reference .internal}
            -   [Python wheels](#python-wheels){.reference .internal}
            -   [Pre-built binaries](#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](#development-with-vs-code){.reference .internal}
            -   [Using a Docker
                container](#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](#connecting-to-a-remote-host){.reference .internal}
            -   [Developing with Remote
                Tunnels](#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](#remote-access-via-ssh){.reference .internal}
        -   [DGX Cloud](#dgx-cloud){.reference .internal}
            -   [Get Started](#get-started){.reference .internal}
            -   [Use JupyterLab](#use-jupyterlab){.reference .internal}
            -   [Use VS Code](#use-vs-code){.reference .internal}
        -   [Additional CUDA Tools](#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](#installation-via-pypi){.reference .internal}
            -   [Installation In Container
                Images](#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](#distributed-computing-with-mpi){.reference .internal}
        -   [Updating CUDA-Q](#updating-cuda-q){.reference .internal}
        -   [Dependencies and
            Compatibility](#dependencies-and-compatibility){.reference
            .internal}
        -   [Next Steps](#next-steps){.reference .internal}
    -   [Data Center Installation](data_center_install.html){.reference
        .internal}
        -   [Prerequisites](data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](data_center_install.html#python-support){.reference
            .internal}
        -   [C++ Support](data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](data_center_install.html#mpi){.reference
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
-   [Installation Guide](install.html)
-   Local Installation
-   

::: {.rst-breadcrumbs-buttons role="navigation" aria-label="Sequential page navigation"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](install.html "Installation Guide"){.btn .btn-neutral
.float-left accesskey="p"} [Next []{.fa .fa-arrow-circle-right
aria-hidden="true"}](data_center_install.html "Installation from Source"){.btn
.btn-neutral .float-right accesskey="n"}
:::

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
::: {#local-installation .section}
# Local Installation[¶](#local-installation "Permalink to this heading"){.headerlink}

::: {#introduction .section}
## Introduction[¶](#introduction "Permalink to this heading"){.headerlink}

This guide walks through how to [[install CUDA-Q]{.std
.std-ref}](install.html#install-cuda-quantum){.reference .internal} on
your system, and how to set up [[VS Code for local development]{.std
.std-ref}](#local-development-with-vscode){.reference .internal}. The
section on [[connecting to a remote host]{.std
.std-ref}](#connect-to-remote){.reference .internal} contains some
guidance for application development on a remote host where CUDA-Q is
installed.

The following sections contain instructions for how to install CUDA-Q on
your machine using

-   [[Docker]{.std .std-ref}](#install-docker-image){.reference
    .internal}: A fully featured CUDA-Q installation including all C++
    and Python tools is available as a
    [Docker](https://docs.docker.com/get-started/overview/){.reference
    .external} image.

-   [[Singularity]{.std
    .std-ref}](#install-singularity-image){.reference .internal}: A
    [Singularity](https://docs.sylabs.io/guides/latest/user-guide/introduction.html){.reference
    .external} container can easily be created based on our Docker
    images.

-   [[PyPI]{.std .std-ref}](#install-python-wheels){.reference
    .internal}: Additionally, we distribute pre-built Python wheels via
    [PyPI](https://pypi.org){.reference .external}.

-   [[Pre-built binaries]{.std
    .std-ref}](#install-prebuilt-binaries){.reference .internal}: We
    also provide pre-built C++ binaries, bundled as [self-extracting
    archive](https://makeself.io/){.reference .external}, that work
    across a range of Linux operating systems.

If you would like to build CUDA-Q from source to deploy on an HPC system
without relying on a container runtime, please follow the instructions
for [[Installation from
Source]{.doc}](data_center_install.html){.reference .internal}. If, on
the other hand, you want to contribute to the development of CUDA-Q
itself and hence want to build a custom version of CUDA-Q from source,
follow the instructions on the [CUDA-Q GitHub
repository](https://github.com/NVIDIA/cuda-quantum/blob/main/Building.md){.reference
.external} instead.

If you are unsure which option suits you best, we recommend using our
[[Docker image]{.std .std-ref}](#install-docker-image){.reference
.internal} to develop your applications in a controlled environment that
does not depend on, or interfere with, other software that is installed
on your system.

::: {#docker .section}
[]{#install-docker-image}

### Docker[¶](#docker "Permalink to this heading"){.headerlink}

To download and use our Docker images, you will need to install and
launch the Docker engine. If you do not already have Docker installed on
your system, you can get it by downloading and installing [Docker
Desktop](https://docs.docker.com/get-docker/){.reference .external}. If
you do not have the necessary administrator permissions to install
software on your machine, take a look at the section below on how to use
[Singularity](#singularity){.reference .internal} instead.

Docker images for all CUDA-Q releases are available on the [NGC
Container
Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum){.reference
.external}. In addition to publishing [stable
releases](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags){.reference
.external}, we also publish Docker images whenever we update certain
branches on our [GitHub
repository](https://github.com/NVIDIA/cuda-quantum){.reference
.external}. These images are published in our [nightly channel on
NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nightly/containers/cuda-quantum/tags){.reference
.external}. To download the latest version on the main branch of our
GitHub repository, built to work with CUDA 12, for example, use the
command

::: {.highlight-console .notranslate}
::: highlight
    docker pull nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
:::
:::

Early prototypes for features we are considering can be tried out by
using the image tags starting with [`experimental`{.code .docutils
.literal .notranslate}]{.pre}. The [`README`{.code .docutils .literal
.notranslate}]{.pre} in the [`/home/cudaq`{.code .docutils .literal
.notranslate}]{.pre} folder in the container gives more details about
the feature. We welcome and appreciate your feedback about these early
prototypes; how popular they are will help inform whether we should
include them in future releases.

Once you have downloaded an image, the container can be run using the
command

::: {.highlight-console .notranslate}
::: highlight
    docker run -it --name cuda-quantum nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
:::
:::

Replace the image name and/or tag in the command above, if necessary,
with the one you want to use. This will give you terminal access to the
created container. To enable support for GPU-accelerated backends, you
will need to pass the [`--gpus`{.code .docutils .literal
.notranslate}]{.pre} flag when launching the container, for example:

::: {.highlight-console .notranslate}
::: highlight
    docker run -it --gpus all --name cuda-quantum nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
:::
:::

::: {.admonition .note}
Note

This command will fail if you do not have a suitable NVIDIA GPU
available, or if your driver version is insufficient. To improve
compatibility with older drivers, you may need to install the [NVIDIA
container
toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html){.reference
.external}.
:::

You can stop and exit the container by typing the command [`exit`{.code
.docutils .literal .notranslate}]{.pre}. If you did not specify
[`--rm`{.code .docutils .literal .notranslate}]{.pre} flag when
launching the container, the container still exists after exiting, as
well as any changes you made in it. You can get back to it using the
command [`docker`{.code .docutils .literal .notranslate}]{.pre}` `{.code
.docutils .literal .notranslate}[`start`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`-i`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`cuda-quantum`{.code .docutils .literal
.notranslate}]{.pre}. You can delete an existing container and any
changes you made using [`docker`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`rm`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`-v`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`cuda-quantum`{.code .docutils .literal
.notranslate}]{.pre}.

When working with Docker images, the files inside the container are not
visible outside the container environment. To facilitate application
development with, for example, debugging, code completion, hover
information, and so on, please take a look at the section on
[[Development with VS Code]{.std
.std-ref}](#docker-in-vscode){.reference .internal}.

Alternatively, it is possible, but not recommended, to launch an SSH
server inside the container environment and connect an IDE using SSH. To
do so, make sure you have generated a suitable RSA key pair; if your
[`~/.ssh/`{.code .docutils .literal .notranslate}]{.pre} folder does not
already contain the files [`id_rsa.pub`{.code .docutils .literal
.notranslate}]{.pre} and [`id.rsa`{.code .docutils .literal
.notranslate}]{.pre}, follow the instructions for generating a new SSH
key on [this
page](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent){.reference
.external}. You can then launch the container and connect to it via SSH
by executing the following commands:

::: {.highlight-console .notranslate}
::: highlight
    docker run -itd --gpus all --name cuda-quantum -p 2222:22 nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
    docker exec cuda-quantum bash -c "sudo apt-get install -y --no-install-recommends openssh-server"
    docker exec cuda-quantum bash -c "sudo sed -i -E "s/#?\s*UsePAM\s+.+/UsePAM yes/g" /etc/ssh/sshd_config"
    docker cp ~/.ssh/id_rsa.pub cuda-quantum:/home/cudaq/.ssh/authorized_keys
    docker exec -d cuda-quantum bash -c "sudo -E /usr/sbin/sshd -D"
    ssh cudaq@localhost -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null
:::
:::
:::

::: {#known-blackwell-issues .section}
[]{#id1}

### Known Blackwell Issues[¶](#known-blackwell-issues "Permalink to this heading"){.headerlink}

There are some known Blackwell issues when using CUDA-Q.

::: {#blackwell-cuda-dependencies .admonition .note}
Note

If you are using CUDA 12.8 on Blackwell, you may need to install
additional dependencies to use the python wheels.

If you see the following error:

::: {.highlight-console .notranslate}
::: highlight
    cupy_backends.cuda.api.driver.CUDADriverError: CUDA_ERROR_NO_BINARY_FOR_GPU: no kernel image is available for execution on the device
:::
:::

You may need to install the more updated python wheels.

::: {.highlight-console .notranslate}
::: highlight
    pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12 nvidia-curand-cu12
:::
:::
:::

::: {#blackwell-torch-dependences .admonition .note}
Note

If you are attempting to use torch integrators on Blackwell, you will
need to install the nightly torch version.

::: {.highlight-console .notranslate}
::: highlight
    python3 -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
:::
:::

With this new version of torch, you may see:

::: {.highlight-console .notranslate}
::: highlight
    Module 'torch' was found, but when imported by pytest it raised:
    ImportError('/home/cudaq/.local/lib/python3.10/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkCreate_12_8, version libnvJitLink.so.12')
:::
:::

This may be caused by an incorrectly linked shared object. If you
encounter this, try adding the shared object to the LD_LIBRARY_PATH:

::: {.highlight-console .notranslate}
::: highlight
    export LD_LIBRARY_PATH=$(pip show nvidia-nvjitlink-cu12 | sed -nE 's/Location: (.*)$/\1/p')/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
:::
:::
:::
:::

::: {#singularity .section}
[]{#install-singularity-image}

### Singularity[¶](#singularity "Permalink to this heading"){.headerlink}

You can use
[Singularity](https://github.com/sylabs/singularity){.reference
.external} to run a CUDA-Q container in a folder without needing
administrator permissions. If you do not already have Singularity
installed, you can build a relocatable installation from source. To do
so on Linux or WSL, make sure you have the [necessary
prerequisites](https://docs.sylabs.io/guides/4.0/user-guide/quick_start.html#prerequisites){.reference
.external} installed, download a suitable version of the [go
toolchain](https://docs.sylabs.io/guides/4.0/user-guide/quick_start.html#install-go){.reference
.external}, and make sure the [`go`{.code .docutils .literal
.notranslate}]{.pre} binaries are on your [`PATH`{.code .docutils
.literal .notranslate}]{.pre}. You can then build Singularity with the
commands

::: {.highlight-console .notranslate}
::: highlight
    wget https://github.com/sylabs/singularity/releases/download/v4.0.1/singularity-ce-4.0.1.tar.gz
    tar -xzf singularity-ce-4.0.1.tar.gz singularity-ce-4.0.1/ && rm singularity-ce-4.0.1.tar.gz && cd singularity-ce-4.0.1/
    ./mconfig --without-suid --prefix="$HOME/.local/singularity"
    make -C ./builddir && make -C ./builddir install && cd .. && rm -rf singularity-ce-4.0.1/
    echo 'PATH="$PATH:$HOME/.local/singularity/bin/"' >> ~/.profile && source ~/.profile
:::
:::

For more information about using Singularity on other systems, take a
look at the [admin
guide](https://docs.sylabs.io/guides/4.0/admin-guide/installation.html#installation-on-windows-or-mac){.reference
.external}.

Once you have singularity installed, create a file
[`cuda-quantum.def`{.code .docutils .literal .notranslate}]{.pre} with
the following content:

::: {.highlight-console .notranslate}
::: highlight
    Bootstrap: docker
    From: nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest

    %runscript
        mount devpts /dev/pts -t devpts
        cp -r /home/cudaq/* .
        bash
:::
:::

Replace the image name and/or tag in the [`From`{.code .docutils
.literal .notranslate}]{.pre} line, if necessary, with the one you want
to use; In addition to publishing [stable
releases](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum/tags){.reference
.external}, we also publish Docker images whenever we update certain
branches on our [GitHub
repository](https://github.com/NVIDIA/cuda-quantum){.reference
.external}. These images are published in our [nightly channel on
NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nightly/containers/cuda-quantum/tags){.reference
.external}. Early prototypes for features we are considering can be
tried out by using the image tags starting with [`experimental`{.code
.docutils .literal .notranslate}]{.pre}. We welcome and appreciate your
feedback about these early prototypes; how popular they are will help
inform whether we should include them in future releases.

You can then create a CUDA-Q container by running the following
commands:

::: {.highlight-console .notranslate}
::: highlight
    singularity build --fakeroot cuda-quantum.sif cuda-quantum.def
    singularity run --writable --fakeroot cuda-quantum.sif
:::
:::

In addition to the files in your current folder, you should now see a
[`README`{.code .docutils .literal .notranslate}]{.pre} file, as well as
examples and tutorials. To enable support for GPU-accelerated backends,
you will need to pass the the [`--nv`{.code .docutils .literal
.notranslate}]{.pre} flag when running the container:

::: {.highlight-console .notranslate}
::: highlight
    singularity run --writable --fakeroot --nv cuda-quantum.sif
    nvidia-smi
:::
:::

The output of the command above lists the GPUs that are visible and
accessible in the container environment.

::: {.admonition .note}
Note

If you do not see any GPUs listed in the output of [`nvidia-smi`{.code
.docutils .literal .notranslate}]{.pre}, it means the container
environment is unable to access a suitable NVIDIA GPU. This can happen
if your driver version is insufficient, or if you are working on WSL. To
improve compatibility with older drivers, or to enable GPU support on
WSL, please install the [NVIDIA container
toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html){.reference
.external}, and update the singularity configuration to set [`use`{.code
.docutils .literal .notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`nvidia-container-cli`{.code .docutils .literal
.notranslate}]{.pre} to [`yes`{.code .docutils .literal
.notranslate}]{.pre} and configure the correct
[`nvidia-container-cli`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`path`{.code .docutils .literal .notranslate}]{.pre}. The
two commands below use [`sed`{.code .docutils .literal
.notranslate}]{.pre} to do that:

::: {.highlight-console .notranslate}
::: highlight
    sed -i 's/use nvidia-container-cli = no/use nvidia-container-cli = yes/' "$HOME/.local/singularity/etc/singularity/singularity.conf"
    sed -i 's/# nvidia-container-cli path =/nvidia-container-cli path = \/usr\/bin\/nvidia-container-cli/' "$HOME/.local/singularity/etc/singularity/singularity.conf"
:::
:::
:::

You can exit the container environment by typing the command
[`exit`{.code .docutils .literal .notranslate}]{.pre}. Any changes you
made will still be visible after you exit the container, and you can
re-enable the container environment at any time using the [`run`{.code
.docutils .literal .notranslate}]{.pre} command above.

To facilitate application development with, for example, debugging, code
completion, hover information, and so on, please take a look at the
section on [[Development with VS Code]{.std
.std-ref}](#singularity-in-vscode){.reference .internal}.
:::

::: {#python-wheels .section}
[]{#install-python-wheels}

### Python wheels[¶](#python-wheels "Permalink to this heading"){.headerlink}

CUDA-Q Python wheels are available on
[PyPI.org](https://pypi.org/project/cudaq/){.reference .external}.
Installation instructions can be found in the [project
description](https://pypi.org/project/cudaq/#description){.reference
.external}. For more information about available versions and
documentation, see [[CUDA-Q
Releases]{.doc}](../../releases.html){.reference .internal}.

There are currently no source distributions available on PyPI, but you
can download the source code for the latest version of the CUDA-Q Python
wheels from our [GitHub
repository](https://github.com/NVIDIA/cuda-quantum){.reference
.external}. The source code for previous versions can be downloaded from
the respective [GitHub
Release](https://github.com/NVIDIA/cuda-quantum/releases){.reference
.external}.

At this time, wheels are distributed for Linux operating systems only.
If your platform is not [[officially supported]{.std
.std-ref}](#dependencies-and-compatibility){.reference .internal} and
[`pip`{.code .docutils .literal .notranslate}]{.pre} does not find a
compatible wheel to install, you can build your own wheel from source
following the instructions here: [[Installation from
Source]{.doc}](data_center_install.html){.reference .internal}.

To build the CUDA-Q Python API for the purpose of contributing to our
[GitHub repository](https://github.com/NVIDIA/cuda-quantum){.reference
.external}, follow the instructions for [Setting up your
Environment](https://github.com/NVIDIA/cuda-quantum/blob/main/Dev_Setup.md){.reference
.external}, and then run the following commands in the repository root:

::: {.highlight-console .notranslate}
::: highlight
    bash scripts/install_prerequisites.sh
    pip install . --user
:::
:::
:::

::: {#pre-built-binaries .section}
[]{#install-prebuilt-binaries}

### Pre-built binaries[¶](#pre-built-binaries "Permalink to this heading"){.headerlink}

Starting with the 0.6.0 release, we provide pre-built binaries for using
CUDA-Q with C++. Support for using CUDA-Q with Python can be installed
side-by-side with the pre-built binaries for C++ by following the
instructions on [PyPI.org](https://pypi.org/project/cudaq/){.reference
.external}. The pre-built binaries work across a range of Linux
operating systems listed under [[Dependencies and Compatibility]{.std
.std-ref}](#dependencies-and-compatibility){.reference .internal}.

Before installing our pre-built binaries, please make sure that your
operating system is using the [GNU C
library](https://www.gnu.org/software/libc/){.reference .external}
version 2.28 or newer. You can confirm this by checking the output of
the command [`ldd`{.code .docutils .literal
.notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`--version`{.code .docutils .literal .notranslate}]{.pre}.
If this command does not exist, or shows an older version than 2.28,
please double check that your operating system is listed as
[[supported]{.std .std-ref}](#dependencies-and-compatibility){.reference
.internal}. If you use an operating system with an older GNU C library
version, you will need to build the installer from source following the
instructions in [[Installation from
Source]{.doc}](data_center_install.html){.reference .internal}.

You can download the [`install_cuda_quantum`{.code .docutils .literal
.notranslate}]{.pre} file for your processor architecture from the
assets of the respective [GitHub
release](https://github.com/NVIDIA/cuda-quantum/releases){.reference
.external}. The installer is a [self-extracting
archive](https://makeself.io/){.reference .external} that contains the
pre-built binaries as well as a script to move them to the correct
locations. You will need [`bash`{.code .docutils .literal
.notranslate}]{.pre}, [`tar`{.code .docutils .literal
.notranslate}]{.pre}, and [`gzip`{.code .docutils .literal
.notranslate}]{.pre} (usually already installed on most Linux
distributions) to run the installer. The installation location of CUDA-Q
is not currently configurable and using the installer hence requires
admin privileges on the system. We may revise that in the future; please
see and upvote the corresponding [GitHub
issue](https://github.com/NVIDIA/cuda-quantum/issues/1075){.reference
.external}.

To install CUDA-Q, execute the command

::: {.highlight-bash .notranslate}
::: highlight
    MPI_PATH=/usr/local/openmpi \
    sudo -E bash install_cuda_quantum*.$(uname -m) --accept && . /etc/profile
:::
:::

::: {.admonition .note}
Note

To use GPU-accelerated backends, you will need to install the necessary
CUDA runtime libraries. For more information see the corresponding
section on [[Additional CUDA Tools]{.std
.std-ref}](#cuda-dependencies-prebuilt-binaries){.reference .internal}.
:::

The installation ensures that the necessary environment variables for
using the CUDA-Q toolchain are set upon login for all POSIX shells.
Confirm that the [`nvq++`{.code .docutils .literal .notranslate}]{.pre}
command is found. If it is not, please make sure to set the environment
variables defined by the [`set_env.sh`{.code .docutils .literal
.notranslate}]{.pre} script in the CUDA-Q installation folder (usually
[`/usr/local/cudaq`{.code .docutils .literal .notranslate}]{.pre} or
[`/opt/nvidia/cudaq`{.code .docutils .literal .notranslate}]{.pre}).

If an MPI installation is available in the directory defined by
[`MPI_PATH`{.code .docutils .literal .notranslate}]{.pre}, the installer
automatically enables MPI support in CUDA-Q. If you do not have MPI
installed on your system, you can simply leave that path empty, and
CUDA-Q will be installed without MPI support. If you install MPI at a
later point in time, you can activate the MPI support in CUDA Quantum by
setting the [`MPI_PATH`{.code .docutils .literal .notranslate}]{.pre}
variable to its installation location and executing the commands

::: {.highlight-console .notranslate}
::: highlight
    MPI_PATH=/usr/local/openmpi # update this path as needed
    bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh"
:::
:::

::: {.admonition .note}
Note

Please make sure that you have the necessary development headers of the
C standard library installed. You can check this by searching for
[`features.h`{.code .docutils .literal .notranslate}]{.pre}, commonly
found in [`/usr/include/`{.code .docutils .literal .notranslate}]{.pre}.
You can install the necessary headers via package manager (usually the
package name is called something like [`glibc-devel`{.code .docutils
.literal .notranslate}]{.pre} or [`libc6-dev`{.code .docutils .literal
.notranslate}]{.pre}). These headers are also included with any
installation of GCC.
:::
:::
:::

::: {#development-with-vs-code .section}
[]{#local-development-with-vscode}

## Development with VS Code[¶](#development-with-vs-code "Permalink to this heading"){.headerlink}

To facilitate application development with, for example, debugging, code
completion, hover information, and so on, we recommend using [VS
Code](https://code.visualstudio.com/){.reference .external}. VS Code
provides a seamless development experience on all platforms, and is also
available without installation via web browser. This sections describes
how to connect VS Code to a running container on your machine. The
section on [[connecting to a remote host]{.std
.std-ref}](#connect-to-remote){.reference .internal} contains
information on how to set up your development environment when accessing
CUDA-Q on a remote host instead.

::: {#using-a-docker-container .section}
[]{#docker-in-vscode}

### Using a Docker container[¶](#using-a-docker-container "Permalink to this heading"){.headerlink}

Before connecting VS Code, open a terminal/shell, and start the CUDA-Q
Docker container following the instructions in the [[section above]{.std
.std-ref}](#install-docker-image){.reference .internal}.

If you have a local installation of [VS
Code](https://code.visualstudio.com/){.reference .external} you can
connect to the running container using the [Dev
Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers){.reference
.external} extension. If you want to use VS Code in the web browser,
please follow the instructions in the section [Developing with Remote
Tunnels](#developing-with-remote-tunnels){.reference .internal} instead.

After installing the [Dev
Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers){.reference
.external} extension, launch VS Code, open the Command Palette with
[`Ctrl+Shift+P`{.code .docutils .literal .notranslate}]{.pre}, and enter
"Dev Containers: Attach to Running Container". You should see and select
the running [`cuda-quantum`{.code .docutils .literal
.notranslate}]{.pre} container in the list. After the window reloaded,
enter "File: Open Folder" in the Command Palette to open the
[`/home/cudaq/`{.code .docutils .literal .notranslate}]{.pre} folder.

To run the examples, open the Command Palette and enter "View: Show
Terminal" to launch an integrated terminal. You are now all set to [[get
started]{.std .std-ref}](#post-installation){.reference .internal} with
CUDA-Q development.
:::

::: {#using-a-singularity-container .section}
[]{#singularity-in-vscode}

### Using a Singularity container[¶](#using-a-singularity-container "Permalink to this heading"){.headerlink}

If you have a GitHub or Microsoft account, we recommend that you connect
to a CUDA-Q container using tunnels. To do so, launch a CUDA-Q
Singularity container following the instructions in the [[section
above]{.std .std-ref}](#install-singularity-image){.reference
.internal}, and then follow the instructions in the section [Developing
with Remote Tunnels](#developing-with-remote-tunnels){.reference
.internal}.

If you cannot use tunnels, you need a local installation of [VS
Code](https://code.visualstudio.com/){.reference .external} and you need
to install the [Remote -
SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh){.reference
.external} extension. Make sure you also have a suitable SSH key pair;
if your [`~/.ssh/`{.code .docutils .literal .notranslate}]{.pre} folder
does not already contain the files [`id_rsa.pub`{.code .docutils
.literal .notranslate}]{.pre} and [`id.rsa`{.code .docutils .literal
.notranslate}]{.pre}, follow the instructions for generating a new SSH
key on [this
page](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent){.reference
.external}.

To connect VS Code to a running CUDA-Q container, the most convenient
setup is to install and run an SSH server in the Singularity container.
Open a terminal/shell in a separate window, and enter the following
commands to create a suitable sandbox:

::: {.highlight-console .notranslate}
::: highlight
    singularity build --sandbox cuda-quantum-sandbox cuda-quantum.sif
    singularity exec --writable --fakeroot cuda-quantum-sandbox \
      apt-get install -y --no-install-recommends openssh-server
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
:::
:::

You can launch this sandbox by entering the commands below. Please see
the [Singularity](#singularity){.reference .internal} section above for
more information about how to get the [`cuda-quantum.sif`{.code
.docutils .literal .notranslate}]{.pre} image, and how to enable
GPU-acceleration with the [`--nv`{.code .docutils .literal
.notranslate}]{.pre} flag.

::: {.highlight-console .notranslate}
::: highlight
    singularity run --writable --fakeroot --nv --network-args="portmap=22:2222/tcp" cuda-quantum-sandbox
    /usr/sbin/sshd -D -p 2222 -E sshd_output.txt
:::
:::

::: {.admonition .note}
Note

Make sure to use a free port. You can check if the SSH server is ready
and listening by looking at the log in [`sshd_output.txt`{.code
.docutils .literal .notranslate}]{.pre}. If the port is already in use,
you can replace the number [`2222`{.code .docutils .literal
.notranslate}]{.pre} by any free TCP port in the range
[`1025-65535`{.code .docutils .literal .notranslate}]{.pre} in all
commands.
:::

Entering [`Ctrl+C`{.code .docutils .literal .notranslate}]{.pre}
followed by [`exit`{.code .docutils .literal .notranslate}]{.pre} will
stop the running container. You can re-start it at any time by entering
the two commands above. While the container is running, open the Command
Palette in VS Code with [`Ctrl+Shift+P`{.code .docutils .literal
.notranslate}]{.pre}, enter "Remote-SSH: Add new SSH Host", and enter
the following SSH command:

::: {.highlight-console .notranslate}
::: highlight
    ssh root@localhost -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null
:::
:::

::: {.admonition .note}
Note

If you are working on Windows and are building and running the
Singularity container in WSL, make sure to copy the used SSH keys to the
Windows partition, such that VS Code can connect with the expected key.
Alternatively, add the used public key to the
[`/root/.ssh/authorized_keys`{.code .docutils .literal
.notranslate}]{.pre} file in the Singularity container.
:::

You can then connect to the host by opening the Command Palette,
entering "Remote SSH: Connect Current Window to Host", and choosing the
newly created host. After the window reloaded, enter "File: Open Folder"
in the Command Palette to open the desired folder.

To run the examples, open the Command Palette and enter "View: Show
Terminal" to launch an integrated terminal. You are now all set to [[get
started]{.std .std-ref}](#post-installation){.reference .internal} with
CUDA-Q development.
:::
:::

::: {#connecting-to-a-remote-host .section}
[]{#connect-to-remote}

## Connecting to a Remote Host[¶](#connecting-to-a-remote-host "Permalink to this heading"){.headerlink}

Depending on the setup on the remote host, there are a couple of
different options for developing CUDA-Q applications.

-   If a CUDA-Q container is running on the remote host, and you have a
    GitHub or Microsoft account, take a look at [Developing with Remote
    Tunnels](#developing-with-remote-tunnels){.reference .internal}.
    This works for both Docker and Singularity containers on the remote
    host, and should also work for other containers.

-   If you cannot use tunnels, or if you want to work with an existing
    CUDA-Q installation without using a container, take a look at
    [Remote Access via SSH](#remote-access-via-ssh){.reference
    .internal} instead.

::: {#developing-with-remote-tunnels .section}
[]{#connect-vscode-via-tunnel}

### Developing with Remote Tunnels[¶](#developing-with-remote-tunnels "Permalink to this heading"){.headerlink}

[Remote access via
tunnel](https://code.visualstudio.com/blogs/2022/12/07/remote-even-better){.reference
.external} can easily be enabled with the [VS Code
CLI](https://code.visualstudio.com/docs/editor/command-line){.reference
.external}. This allows to connect either a local installation of [VS
Code](https://code.visualstudio.com/){.reference .external}, or the [VS
Code Web UI](https://vscode.dev/){.reference .external}, to a running
CUDA-Q container on the same or a different machine.

Creating a secure connection requires authenticating with the same
GitHub or Microsoft account on each end. Once authenticated, an SSH
connection over the tunnel provides end-to-end encryption. To download
the VS Code CLI, if necessary, and create a tunnel, execute the
following command in the running CUDA-Q container, and follow the
instructions to authenticate:

::: {.highlight-console .notranslate}
::: highlight
    vscode-setup tunnel --name cuda-quantum-remote --accept-server-license-terms
:::
:::

You can then either [open VS Code in a web
browser](https://vscode.dev/tunnel/cuda-quantum-remote/home/cudaq/){.reference
.external}, or connect a local installation of VS Code. To connect a
local installation of VS Code, make sure you have the [Remote -
Tunnels](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-server){.reference
.external} extension installed, then open the Command Palette with
[`Ctrl+Shift+P`{.code .docutils .literal .notranslate}]{.pre}, enter
"Remote Tunnels: Connect to Tunnel", and enter
[`cuda-quantum-remote`{.code .docutils .literal .notranslate}]{.pre}.
After the window reloaded, enter "File: Open Folder" in the Command
Palette to open the [`/home/cudaq/`{.code .docutils .literal
.notranslate}]{.pre} folder.

You should see a pop up asking if you want to install the recommended
extensions. Selecting to install them will configure VS Code with
extensions for working with C++, Python, and Jupyter. You can always see
the list of recommended extensions that aren't installed yet by clicking
on the "Extensions" icon in the sidebar and navigating to the
"Recommended" tab.
:::

::: {#remote-access-via-ssh .section}
### Remote Access via SSH[¶](#remote-access-via-ssh "Permalink to this heading"){.headerlink}

To facilitate application development with, for example, debugging, code
completion, hover information, and so on, you can connect a local
installation of [VS Code](https://code.visualstudio.com/){.reference
.external} to a remote host via SSH.

::: {.admonition .note}
Note

For the best user experience, we recommend to launch a CUDA-Q container
on the remote host, and then connect [[VS Code using tunnels]{.std
.std-ref}](#connect-vscode-via-tunnel){.reference .internal}. If a
connection via tunnel is not possible, this section describes using SSH
instead.
:::

To do so, make sure you have [Remote -
SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh){.reference
.external} extension installed. Open the Command Palette with
[`Ctrl+Shift+P`{.code .docutils .literal .notranslate}]{.pre}, enter
"Remote-SSH: Add new SSH Host", and enter the SSH command to connect to
your account on the remote host. You can then connect to the host by
opening the Command Palette, entering "Remote SSH: Connect Current
Window to Host", and choosing the newly created host.

When prompted, choose Linux as the operating system, and enter your
password. After the window reloaded, enter "File: Open Folder" in the
Command Palette to open the desired folder. Our GitHub repository
contains a folder with VS Code configurations including a list of
recommended extensions for working with CUDA-Q; you can copy [these
configurations](https://github.com/NVIDIA/cuda-quantum/tree/main/docker/release/config/.vscode){.reference
.external} into the a folder named [`.vscode`{.code .docutils .literal
.notranslate}]{.pre} in your workspace to use them.

If you want to work with an existing CUDA-Q installation on the remote
host, you are all set. Alternatively, you can use Singularity to build
and run a container following the instructions in [[this section]{.std
.std-ref}](#install-singularity-image){.reference .internal}. Once the
[`cuda-quantum.sif`{.code .docutils .literal .notranslate}]{.pre} image
is built and available in your home directory on the remote host, you
can update your VS Code configuration to enable/improve completion,
hover information, and other development tools within the container.

To do so, open the Command Palette and enter "Remote-SSH: Open SSH
Configuration File". Add a new entry to that file with the command to
launch the container, and edit the configuration of the remote host,
titled [`remote-host`{.code .docutils .literal .notranslate}]{.pre} in
the snippets below, to add a new identifier:

::: {.highlight-console .notranslate}
::: highlight
    Host cuda-quantum~*
      RemoteCommand singularity run --writable --fakeroot --nv ~/cuda-quantum.sif
      RequestTTY yes

    Host remote-host cuda-quantum~remote-host
      HostName ...
      ...
:::
:::

You will need to edit a couple of VS Code setting to make use of the
newly defined remote command; open the Command Palette, enter
"Preferences: Open User Settings (JSON)", and add or update the
following configurations:

::: {.highlight-console .notranslate}
::: highlight
    "remote.SSH.enableRemoteCommand": true,
    "remote.SSH.useLocalServer": true,
    "remote.SSH.remoteServerListenOnSocket": false,
    "remote.SSH.connectTimeout": 120,
    "remote.SSH.serverInstallPath": {
        "cuda-quantum~remote-host": "~/.vscode-container/cuda-quantum",
    },
:::
:::

After saving the changes, you should now be able to select
[`cuda-quantum~remote-host`{.code .docutils .literal
.notranslate}]{.pre} as the host when connecting via SSH, which will
launch the CUDA-Q container and connect VS Code to it.

::: {.admonition .note}
Note

If the connection to [`cuda-quantum~remote-host`{.code .docutils
.literal .notranslate}]{.pre} fails, you may need to specify the full
path to the [`singularity`{.code .docutils .literal .notranslate}]{.pre}
executable on the remote host, since environment variables, and
specifically the configured [`PATH`{.code .docutils .literal
.notranslate}]{.pre} may be different during launch than in your user
account.
:::
:::
:::

::: {#dgx-cloud .section}
## DGX Cloud[¶](#dgx-cloud "Permalink to this heading"){.headerlink}

If you are using [DGX
Cloud](https://www.nvidia.com/en-us/data-center/dgx-cloud/){.reference
.external}, you can easily use it to run CUDA-Q applications. While
submitting jobs to DGX Cloud directly from within CUDA-Q is not (yet)
supported, you can use the NGC CLI to launch and interact with workloads
in DGX Cloud. The following sections detail how to do that, and how to
connect JupyterLab and/or VS Code to a running CUDA-Q job in DGX Cloud.

::: {#get-started .section}
[]{#dgx-cloud-setup}

### Get Started[¶](#get-started "Permalink to this heading"){.headerlink}

To get started with DGX Cloud, you can [request access
here](https://www.nvidia.com/en-us/data-center/dgx-cloud/trial/){.reference
.external}. Once you have access, [sign
in](https://ngc.nvidia.com/signin){.reference .external} to your
account, and [generate an API
key](https://ngc.nvidia.com/setup/api-key){.reference .external}.
[Install the NGC
CLI](https://ngc.nvidia.com/setup/installers/cli){.reference .external}
and configure it with

::: {.highlight-console .notranslate}
::: highlight
    ngc config set
:::
:::

entering the API key you just generated when prompted, and configure
other settings as appropriate.

::: {.admonition .note}
Note

The rest of this section assumes you have CLI version 3.33.0. If you
have an older version installed, you can upgrade to the latest version
using the command

::: {.highlight-console .notranslate}
::: highlight
    ngc version upgrade 3.33.0
:::
:::

See also the [NGC CLI
documentation](https://docs.ngc.nvidia.com/cli/index.html){.reference
.external} for more information about available commands.
:::

You can see all information about available compute resources and ace
instances with the command

::: {.highlight-console .notranslate}
::: highlight
    ngc base-command ace list
:::
:::

Confirm that you can submit a job with the command

::: {.highlight-console .notranslate}
::: highlight
    ngc base-command job run \
      --name Job-001 --total-runtime 60s \
      --image nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest --result /results \
      --ace <ace_name> --instance <instance_name> \
      --commandline 'echo "Hello from DGX Cloud!"'
:::
:::

replacing [`<ace_name>`{.code .docutils .literal .notranslate}]{.pre}
and [`<instance_name>`{.code .docutils .literal .notranslate}]{.pre}
with the name of the ace and instance you want to execute the job on.
You should now see that job listed when you run the command

::: {.highlight-console .notranslate}
::: highlight
    ngc base-command job list
:::
:::

Once it has completed you can download the job results using the command

::: {.highlight-console .notranslate}
::: highlight
    ngc base-command result download <job_id>
:::
:::

replacing [`<job_id>`{.code .docutils .literal .notranslate}]{.pre} with
the id of the job you just submitted. You should see a new folder named
[`<job_id>`{.code .docutils .literal .notranslate}]{.pre} with the job
log that contains the output "Hello from DGX Cloud!".

For more information about how to use the NGC CLI to interact with DGX
Cloud, we refer to the [NGC CLI
documentation](https://docs.ngc.nvidia.com/cli/index.html){.reference
.external}.
:::

::: {#use-jupyterlab .section}
### Use JupyterLab[¶](#use-jupyterlab "Permalink to this heading"){.headerlink}

Once you can [[run jobs on DGX Cloud]{.std
.std-ref}](#dgx-cloud-setup){.reference .internal}, you can launch an
interactive job to use CUDA-Q with
[JupyterLab](https://jupyterlab.readthedocs.io/en/latest/){.reference
.external} running on DGX Cloud:

::: {.highlight-console .notranslate}
::: highlight
    ngc base-command job run \
      --name Job-interactive-001 --total-runtime 600s \
      --image nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest --result /results \
      --ace <ace_name> --instance <instance_name> \
      --port 8888 --commandline 'jupyter-lab-setup <my-custom-token> --port=8888'
:::
:::

Replace [`<my-custom-token>`{.code .docutils .literal
.notranslate}]{.pre} in the command above with a custom token that you
can freely choose. You will use this token to authenticate with
JupyterLab; Go to the [job
portal](https://bc.ngc.nvidia.com/jobs){.reference .external}, click on
the job you just launched, and click on the link under "URL/Hostname" in
Service Mapped Ports.

::: {.admonition .note}
Note

It may take a couple of minutes for DGX Cloud to launch and for the URL
to become active, even after it appears in the Service Mapped Ports
section; if you encounter a "404: Not Found" error, be patient and try
again in a couple of minutes.
:::

Once this URL opens, you should see the JupyterLab authentication page;
enter the token you selected above to get access to the running CUDA-Q
container. On the left you should see a folder with tutorials. Happy
coding!
:::

::: {#use-vs-code .section}
### Use VS Code[¶](#use-vs-code "Permalink to this heading"){.headerlink}

Once you can [[run jobs on DGX Cloud]{.std
.std-ref}](#dgx-cloud-setup){.reference .internal}, you can launch an
interactive job to use CUDA-Q with a local installation of [VS
Code](https://code.visualstudio.com/){.reference .external}, or the [VS
Code Web UI](https://vscode.dev/){.reference .external}, running on DGX
Cloud:

::: {.highlight-console .notranslate}
::: highlight
    ngc base-command job run \
      --name Job-interactive-001 --total-runtime 600s \
      --image nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest --result /results \
      --ace <ace_name> --instance <instance_name> \
      --commandline 'vscode-setup tunnel --name cuda-quantum-dgx --accept-server-license-terms'
:::
:::

Go to the [job portal](https://bc.ngc.nvidia.com/jobs){.reference
.external}, click on the job you just launched, and select the "Log"
tab. Once the job is running, you should see instructions there for how
to connect to the device the job is running on. These instructions
include a link to open and the code to enter on that page; follow the
instructions to authenticate. Once you have authenticated, you can
either [open VS Code in a web
browser](https://vscode.dev/tunnel/cuda-quantum-dgx/home/cudaq/){.reference
.external}, or connect a local installation of VS Code. To connect a
local installation of VS Code, make sure you have the [Remote -
Tunnels](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-server){.reference
.external} extension installed, then open the Command Palette with
[`Ctrl+Shift+P`{.code .docutils .literal .notranslate}]{.pre}, enter
"Remote Tunnels: Connect to Tunnel", and enter
[`cuda-quantum-remote`{.code .docutils .literal .notranslate}]{.pre}.
After the window reloaded, enter "File: Open Folder" in the Command
Palette to open the [`/home/cudaq/`{.code .docutils .literal
.notranslate}]{.pre} folder.

You should see a pop up asking if you want to install the recommended
extensions. Selecting to install them will configure VS Code with
extensions for working with C++, Python, and Jupyter. You can always see
the list of recommended extensions that aren't installed yet by clicking
on the "Extensions" icon in the sidebar and navigating to the
"Recommended" tab.

If you enter "View: Show Explorer" in the Command Palette, you should
see a folder with tutorials and examples to help you get started. Take a
look at [Next Steps](#next-steps){.reference .internal} to dive into
CUDA-Q development.
:::
:::

::: {#additional-cuda-tools .section}
[]{#id8}

## Additional CUDA Tools[¶](#additional-cuda-tools "Permalink to this heading"){.headerlink}

CUDA-Q makes use of GPU-acceleration in certain backends and components.
Depending on how you installed CUDA-Q, you may need to install certain
CUDA libraries separately to take advantage of these.

::: {#installation-via-pypi .section}
### Installation via PyPI[¶](#installation-via-pypi "Permalink to this heading"){.headerlink}

If you installed CUDA-Q via
[PyPI](https://pypi.org/project/cudaq/){.reference .external}, please
follow the installation instructions there to install the necessary CUDA
dependencies.
:::

::: {#installation-in-container-images .section}
### Installation In Container Images[¶](#installation-in-container-images "Permalink to this heading"){.headerlink}

If you are using the CUDA-Q container image, the image already contains
all necessary runtime libraries to use all CUDA-Q components. To take
advantage of GPU-acceleration, make sure to enable GPU support when you
launch the container, that is pass the [`--gpus`{.code .docutils
.literal .notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`all`{.code .docutils .literal .notranslate}]{.pre} flag
when launching the container with Docker and the [`--nv`{.code .docutils
.literal .notranslate}]{.pre} flag when launching the container with
Singularity.

Note that the image does not contain all development dependencies for
CUDA, such as, for example the [`nvcc`{.code .docutils .literal
.notranslate}]{.pre} compiler. You can install all CUDA development
dependencies by running the command

::: {.highlight-console .notranslate}
::: highlight
    sudo apt-get install cuda-toolkit-12.0
:::
:::

inside the container. Make sure the toolkit version you install matches
the CUDA runtime installation in the container. Most Python packages
that use GPU-acceleration, such as for example
[CuPy](https://cupy.dev){.reference .external}, require an existing CUDA
installation. After installing the [`cuda-toolkit-12.0`{.code .docutils
.literal .notranslate}]{.pre} you can install CuPy for CUDA 12 with the
command

::: {.highlight-console .notranslate}
::: highlight
    python3 -m pip install cupy-cuda12x
:::
:::
:::

::: {#installing-pre-built-binaries .section}
[]{#cuda-dependencies-prebuilt-binaries}

### Installing Pre-built Binaries[¶](#installing-pre-built-binaries "Permalink to this heading"){.headerlink}

If you installed pre-built binaries for CUDA-Q, you will need to install
the necessary CUDA runtime libraries to use GPU-acceleration in CUDA-Q.
If you prefer to only install the minimal set of runtime libraries, the
following commands, for example, install the necessary packages for RHEL
8:

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

More detailed instructions for your platform can be found in the online
documentation linked for that [CUDA
version](https://developer.nvidia.com/cuda-toolkit-archive){.reference
.external}. Please make sure to install CUDA version 12.0 or newer, and
confirm that your [GPU
driver](https://www.nvidia.com/download/index.aspx){.reference
.external} supports that version. While the above packages are
sufficient to use GPU-acceleration within CUDA-Q, we recommend
installing the complete CUDA toolkit ([`cuda-toolkit-12-0`{.code
.docutils .literal .notranslate}]{.pre}) that also includes the
[`nvcc`{.code .docutils .literal .notranslate}]{.pre} compiler. A
separate CUDA-Q installer is available for CUDA 12, built against
version 12.6, and for CUDA 13, built against version 13.0.
:::
:::

::: {#distributed-computing-with-mpi .section}
[]{#id9}

## Distributed Computing with MPI[¶](#distributed-computing-with-mpi "Permalink to this heading"){.headerlink}

CUDA-Q supports the Message Passing Interface (MPI) parallelism via a
plugin interface. It is possible to activate or replace such an MPI
plugin without re-installing or re-compiling CUDA-Q. MPI calls via
CUDA-Q API for C++ and Python will be delegated to the currently
activated plugin at runtime.

::: {.tab-set .docutils}
Built-in MPI Support

::: {.tab-content .docutils}
The [[CUDA-Q Docker image]{.std
.std-ref}](#install-docker-image){.reference .internal} is shipped with
a pre-built MPI plugin based on an optimized OpenMPI installation
included in the image. No action is required to use this plugin. We
recommend using this plugin unless the container host has an existing
MPI implementation other than OpenMPI.

If you are not using the Docker image, or are using the image on a
system that has a vendor-optimized MPI library pre-installed, please
follow the instructions in the "Custom MPI Support" tab to enable MPI
support.
:::

Custom MPI Support

::: {.tab-content .docutils}
If you are not using the Docker image, or are using the image on a
system that has a vendor-optimized MPI library pre-installed, CUDA-Q can
be configured to use the local MPI installation by manually activating a
suitable plugin post-installation. To do so,

-   Make sure the environment variable [`CUDA_QUANTUM_PATH`{.code
    .docutils .literal .notranslate}]{.pre} points to the CUDA-Q
    installation directory. If you installed CUDA-Q using the
    [`installer`{.code .docutils .literal .notranslate}]{.pre}` `{.code
    .docutils .literal .notranslate}[`<install-prebuilt-binaries>`{.code
    .docutils .literal .notranslate}]{.pre}, or if you are using the
    CUDA-Q container image, this variable should already be defined. If
    you installed the CUDA-Q [`Python`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`wheels`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`<install-python-wheels>`{.code .docutils .literal
    .notranslate}]{.pre}, set this variable to the directory listed
    under "Location" when you run the command [`pip`{.code .docutils
    .literal .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`show`{.code .docutils .literal
    .notranslate}]{.pre}` `{.code .docutils .literal
    .notranslate}[`cudaq`{.code .docutils .literal .notranslate}]{.pre}.

-   Set the environment variable [`MPI_PATH`{.code .docutils .literal
    .notranslate}]{.pre} to the location of your MPI installation. In
    particular, [`${MPI_PATH}/include`{.code .docutils .literal
    .notranslate}]{.pre} is expected to contain the [`mpi.h`{.code
    .docutils .literal .notranslate}]{.pre} header and
    [`${MPI_PATH}/lib64`{.code .docutils .literal .notranslate}]{.pre}
    or [`${MPI_PATH}/lib`{.code .docutils .literal .notranslate}]{.pre}
    is expected to contain [`libmpi.so`{.code .docutils .literal
    .notranslate}]{.pre}.

-   Execute the following command to complete the activation:

    ::: {.highlight-console .notranslate}
    ::: highlight
        bash $CUDA_QUANTUM_PATH/distributed_interfaces/activate_custom_mpi.sh
    :::
    :::

::: {.admonition .note}
Note

HPC data centers often have a vendor-optimized MPI library pre-installed
on their system. If you are using our container images, installing that
MPI implementation in the container and manually activating the plugin
following the steps above ensure the best performance, and guarantee
compatibility when MPI injection into a container occurs.
:::

Manually activating an MPI plugin replaces any existing plugin; After
the initial activation, the newly built
[`libcudaq_distributed_interface_mpi.so`{.code .docutils .literal
.notranslate}]{.pre} in the installation directory will subsequently
always be used to handle CUDA-Q MPI calls.

::: {.admonition .note}
Note

Executing the activation script from the CUDA-Q installation directory
requires *write* permissions to that directory. If you do not have the
necessary permissions, copy the [`distributed_interfaces`{.code
.docutils .literal .notranslate}]{.pre} sub-directory to a local
location and execute the activation script from there.

In this scenario, since the activated plugin
([`libcudaq_distributed_interface_mpi.so`{.code .docutils .literal
.notranslate}]{.pre}) is outside the CUDA-Q installation, you must set
the environment variable [`$CUDAQ_MPI_COMM_LIB`{.code .docutils .literal
.notranslate}]{.pre} to the path of that shared library. This is done
automatically when executing that activation script, but you may wish to
persist that environment variable between bash sessions, e.g., by adding
it to the [`.bashrc`{.code .docutils .literal .notranslate}]{.pre} file.
:::
:::
:::
:::

::: {#updating-cuda-q .section}
[]{#updating-cuda-quantum}

## Updating CUDA-Q[¶](#updating-cuda-q "Permalink to this heading"){.headerlink}

If you installed the CUDA-Q Python wheels, you can update to the latest
release by running the command

::: {.highlight-console .notranslate}
::: highlight
    python3 -m pip install --upgrade cudaq
:::
:::

::: {.admonition .note}
Note

Please check if you have an existing installation of the
[`cuda-quantum`{.code .docutils .literal .notranslate}]{.pre},
[`cudaq-quantum-cu12`{.code .docutils .literal .notranslate}]{.pre}, or
[`cuda-quantum-cu13`{.code .docutils .literal .notranslate}]{.pre}
package, and uninstall it prior to installing [`cudaq`{.code .docutils
.literal .notranslate}]{.pre}. The [`cudaq`{.code .docutils .literal
.notranslate}]{.pre} package supersedes the [`cuda-quantum`{.code
.docutils .literal .notranslate}]{.pre} package and will install a
suitable binary distribution (either [`cuda-quantum-cu12`{.code
.docutils .literal .notranslate}]{.pre} or [`cuda-quantum-cu13`{.code
.docutils .literal .notranslate}]{.pre}) for your system. Multiple
versions of a CUDA-Q binary distribution will conflict with each other
and not work properly.
:::

If you previously installed the CUDA-Q pre-built binaries, you should
first uninstall your current CUDA-Q installation before installing the
new version using the installer. To uninstall your current CUDA-Q
version, run the command

::: {.highlight-console .notranslate}
::: highlight
    sudo bash "${CUDA_QUANTUM_PATH}/uninstall.sh" -y
:::
:::

The [`uninstall.sh`{.code .docutils .literal .notranslate}]{.pre} script
is generated during installation, and will remove all files and folders
that were created as part of the installation, whether they were
modified in the meantime or not. It does not remove any additional files
that existed prior to the installation or that you have added to the
installation location since then. You can then download and install the
new version of CUDA-Q following the instructions [[above]{.std
.std-ref}](#install-prebuilt-binaries){.reference .internal}.
:::

::: {#dependencies-and-compatibility .section}
[]{#id10}

## Dependencies and Compatibility[¶](#dependencies-and-compatibility "Permalink to this heading"){.headerlink}

CUDA-Q can be used to compile and run quantum programs on a CPU-only
system, but a GPU is highly recommended and necessary to use the
GPU-based simulators, see also [[CUDA-Q Circuit Simulation
Backends]{.doc}](../backends/simulators.html){.reference .internal}.

The supported CPUs include x86_64 (x86-64-v3 architecture and newer) and
ARM64 (ARM v8-A architecture and newer).

::: {.admonition .note}
Note

Some of the components included in the CUDA-Q Python wheels depend on an
existing CUDA installation on your system. For more information about
installing the CUDA-Q Python wheels, take a look at [[this section]{.std
.std-ref}](#install-python-wheels){.reference .internal}.
:::

The following table summarizes the required components.

+--------------------------+--------------------------------------------+
| CPU architectures        | x86_64, ARM64                              |
+--------------------------+--------------------------------------------+
| Operating System         | Linux                                      |
+--------------------------+--------------------------------------------+
| Tested Distributions     | CentOS 8; Debian 11, 12; Fedora 41;        |
|                          | OpenSUSE/SLED/SLES 15.5, 15.6; RHEL 8, 9;  |
|                          | Rocky 8, 9; Ubuntu 22.04, 24.04            |
+--------------------------+--------------------------------------------+
| Python versions          | 3.10+                                      |
+--------------------------+--------------------------------------------+

: [Supported
Systems]{.caption-text}[¶](#id11 "Permalink to this table"){.headerlink}

+--------------------------+--------------------------------------------+
| GPU Architectures        | Turing, Ampere, Ada, Hopper, Blackwell     |
|                          | (Blackwell supported for CUDA 13.x only)   |
+--------------------------+--------------------------------------------+
| NVIDIA GPU with Compute  | 7.5+                                       |
| Capability               |                                            |
+--------------------------+--------------------------------------------+
| CUDA                     | -   12.x (Driver 525.60.13+) -- For GPUs   |
|                          |     that support CUDA Forward              |
|                          |     Compatibility                          |
|                          |                                            |
|                          | -   12.6+ (Driver 560.35.05+) -- For all   |
|                          |     GPUs with supported architecture       |
|                          |                                            |
|                          | -   13.x (Driver 580.65.06+)               |
+--------------------------+--------------------------------------------+

: [Requirements for GPU
Simulation]{.caption-text}[¶](#id12 "Permalink to this table"){.headerlink}

Detailed information about supported drivers for different CUDA versions
and be found
[here](https://docs.nvidia.com/deploy/cuda-compatibility/){.reference
.external}. For more information on GPU forward capabilities, please
refer to [this
page](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html){.reference
.external}.

::: {.admonition .note}
Note

Tegra devices (Jetson) are not supported in CUDA-Q at this time.

For more information, please refer to [Binary Compatibility
documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#binary-compatibility){.reference
.external}.
:::
:::

::: {#next-steps .section}
[]{#post-installation}

## Next Steps[¶](#next-steps "Permalink to this heading"){.headerlink}

You can now compile and/or run the C++ and Python examples using the
terminal. To open a terminal in VS Code, open the Command Palette with
[`Ctrl+Shift+P`{.code .docutils .literal .notranslate}]{.pre} and enter
"View: Show Terminal".

![](../../_images/getToWork.png)

The CUDA-Q image contains a folder with examples and tutorials in the
[`/home/cudaq`{.code .docutils .literal .notranslate}]{.pre} directory.
These examples are provided to get you started with CUDA-Q and
understanding the programming and execution model. If you are not using
a container image, you can find these examples on our [GitHub
repository](https://github.com/NVIDIA/cuda-quantum){.reference
.external}.

Let's start by running a simple program to validate your installation.
The samples contain an implementation of a [Bernstein-Vazirani
algorithm](https://en.wikipedia.org/wiki/Bernstein%E2%80%93Vazirani_algorithm){.reference
.external}. To run the example, execute the command:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-console .notranslate}
::: highlight
    python examples/python/bernstein_vazirani.py --size 5
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-console .notranslate}
::: highlight
    nvq++ examples/cpp/algorithms/bernstein_vazirani.cpp && ./a.out
:::
:::
:::
:::

This will execute the program on the [[default simulator]{.std
.std-ref}](../backends/sims/svsims.html#default-simulator){.reference
.internal}, which will use GPU-acceleration if a suitable GPU has been
detected. To confirm that the GPU acceleration works, you can increase
the size of the secret string, and pass the target as a command line
argument:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-console .notranslate}
::: highlight
    python examples/python/bernstein_vazirani.py --size 25 --target nvidia
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-console .notranslate}
::: highlight
    nvq++ examples/cpp/algorithms/bernstein_vazirani.cpp -DSIZE=25 --target nvidia && ./a.out
:::
:::
:::
:::

This program should complete fairly quickly. Depending on the available
memory on your GPU, you can set the size of the secret string to up to
28-32 when running on the [`nvidia`{.code .docutils .literal
.notranslate}]{.pre} target.

::: {.admonition .note}
Note

If you get an error that the CUDA driver version is insufficient or no
GPU has been detected, check that you have enabled GPU support when
launching the container by passing the [`--gpus`{.code .docutils
.literal .notranslate}]{.pre}` `{.code .docutils .literal
.notranslate}[`all`{.code .docutils .literal .notranslate}]{.pre} flag
(for [[Docker]{.std .std-ref}](#install-docker-image){.reference
.internal}) or the [`--nv`{.code .docutils .literal .notranslate}]{.pre}
flag (for [[Singularity]{.std
.std-ref}](#install-singularity-image){.reference .internal}). If you
are not running a container, you can execute the command
[`nvidia-smi`{.code .docutils .literal .notranslate}]{.pre} to confirm
your setup; if the command is unknown or fails, you do not have a GPU or
do not have a driver installed. If the command succeeds, please confirm
that your CUDA and driver version matches the [[supported versions]{.std
.std-ref}](#dependencies-and-compatibility){.reference .internal}.
:::

Let's compare that to using only your CPU:

::: {.tab-set .docutils}
Python

::: {.tab-content .docutils}
::: {.highlight-console .notranslate}
::: highlight
    python examples/python/bernstein_vazirani.py --size 25 --target qpp-cpu
:::
:::
:::

C++

::: {.tab-content .docutils}
::: {.highlight-console .notranslate}
::: highlight
    nvq++ examples/cpp/algorithms/bernstein_vazirani.cpp -DSIZE=25 --target qpp-cpu && ./a.out
:::
:::
:::
:::

When you execute this command, the program simply seems to hang; that is
because it takes a long time for the CPU-only backend to simulate 28+
qubits! Cancel the execution with [`Ctrl+C`{.code .docutils .literal
.notranslate}]{.pre}.

You are now all set to start developing quantum applications using
CUDA-Q! Please proceed to
[[Basics]{.doc}](../basics/basics.html){.reference .internal} for an
introduction to the fundamental features of CUDA-Q.
:::
:::
:::
:::

::: {.rst-footer-buttons role="navigation" aria-label="Footer"}
[[]{.fa .fa-arrow-circle-left aria-hidden="true"}
Previous](install.html "Installation Guide"){.btn .btn-neutral
.float-left accesskey="p" rel="prev"} [Next []{.fa
.fa-arrow-circle-right
aria-hidden="true"}](data_center_install.html "Installation from Source"){.btn
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
