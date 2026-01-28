::: wy-grid-for-nav
::: wy-side-scroll
::: {.wy-side-nav-search style="background: #76b900"}
[NVIDIA CUDA-Q](index.html){.icon .icon-home}

::: version
pr-3736
:::

::: {role="search"}
:::
:::

::: {.wy-menu .wy-menu-vertical spy="affix" role="navigation" aria-label="Navigation menu"}
[Contents]{.caption-text}

-   [Quick Start](using/quick_start.html){.reference .internal}
    -   [Install
        CUDA-Q](using/quick_start.html#install-cuda-q){.reference
        .internal}
    -   [Validate your
        Installation](using/quick_start.html#validate-your-installation){.reference
        .internal}
    -   [CUDA-Q
        Academic](using/quick_start.html#cuda-q-academic){.reference
        .internal}
-   [Basics](using/basics/basics.html){.reference .internal}
    -   [What is a CUDA-Q
        Kernel?](using/basics/kernel_intro.html){.reference .internal}
    -   [Building your first CUDA-Q
        Program](using/basics/build_kernel.html){.reference .internal}
    -   [Running your first CUDA-Q
        Program](using/basics/run_kernel.html){.reference .internal}
        -   [Sample](using/basics/run_kernel.html#sample){.reference
            .internal}
        -   [Run](using/basics/run_kernel.html#run){.reference
            .internal}
        -   [Observe](using/basics/run_kernel.html#observe){.reference
            .internal}
        -   [Running on a
            GPU](using/basics/run_kernel.html#running-on-a-gpu){.reference
            .internal}
    -   [Troubleshooting](using/basics/troubleshooting.html){.reference
        .internal}
        -   [Debugging and Verbose Simulation
            Output](using/basics/troubleshooting.html#debugging-and-verbose-simulation-output){.reference
            .internal}
-   [Examples](using/examples/examples.html){.reference .internal}
    -   [Introduction](using/examples/introduction.html){.reference
        .internal}
    -   [Building
        Kernels](using/examples/building_kernels.html){.reference
        .internal}
        -   [Defining
            Kernels](using/examples/building_kernels.html#defining-kernels){.reference
            .internal}
        -   [Initializing
            states](using/examples/building_kernels.html#initializing-states){.reference
            .internal}
        -   [Applying
            Gates](using/examples/building_kernels.html#applying-gates){.reference
            .internal}
        -   [Controlled
            Operations](using/examples/building_kernels.html#controlled-operations){.reference
            .internal}
        -   [Multi-Controlled
            Operations](using/examples/building_kernels.html#multi-controlled-operations){.reference
            .internal}
        -   [Adjoint
            Operations](using/examples/building_kernels.html#adjoint-operations){.reference
            .internal}
        -   [Custom
            Operations](using/examples/building_kernels.html#custom-operations){.reference
            .internal}
        -   [Building Kernels with
            Kernels](using/examples/building_kernels.html#building-kernels-with-kernels){.reference
            .internal}
        -   [Parameterized
            Kernels](using/examples/building_kernels.html#parameterized-kernels){.reference
            .internal}
    -   [Quantum
        Operations](using/examples/quantum_operations.html){.reference
        .internal}
        -   [Quantum
            States](using/examples/quantum_operations.html#quantum-states){.reference
            .internal}
        -   [Quantum
            Gates](using/examples/quantum_operations.html#quantum-gates){.reference
            .internal}
        -   [Measurements](using/examples/quantum_operations.html#measurements){.reference
            .internal}
    -   [Measuring
        Kernels](using/examples/measuring_kernels.html){.reference
        .internal}
        -   [Mid-circuit Measurement and Conditional
            Logic](using/examples/measuring_kernels.html#mid-circuit-measurement-and-conditional-logic){.reference
            .internal}
    -   [Visualizing
        Kernels](examples/python/visualization.html){.reference
        .internal}
        -   [Qubit
            Visualization](examples/python/visualization.html#Qubit-Visualization){.reference
            .internal}
        -   [Kernel
            Visualization](examples/python/visualization.html#Kernel-Visualization){.reference
            .internal}
    -   [Executing
        Kernels](using/examples/executing_kernels.html){.reference
        .internal}
        -   [Sample](using/examples/executing_kernels.html#sample){.reference
            .internal}
            -   [Sample
                Asynchronous](using/examples/executing_kernels.html#sample-asynchronous){.reference
                .internal}
        -   [Run](using/examples/executing_kernels.html#run){.reference
            .internal}
            -   [Return Custom Data
                Types](using/examples/executing_kernels.html#return-custom-data-types){.reference
                .internal}
            -   [Run
                Asynchronous](using/examples/executing_kernels.html#run-asynchronous){.reference
                .internal}
        -   [Observe](using/examples/executing_kernels.html#observe){.reference
            .internal}
            -   [Observe
                Asynchronous](using/examples/executing_kernels.html#observe-asynchronous){.reference
                .internal}
        -   [Get
            State](using/examples/executing_kernels.html#get-state){.reference
            .internal}
            -   [Get State
                Asynchronous](using/examples/executing_kernels.html#get-state-asynchronous){.reference
                .internal}
    -   [Computing Expectation
        Values](using/examples/expectation_values.html){.reference
        .internal}
        -   [Parallelizing across Multiple
            Processors](using/examples/expectation_values.html#parallelizing-across-multiple-processors){.reference
            .internal}
    -   [Multi-GPU
        Workflows](using/examples/multi_gpu_workflows.html){.reference
        .internal}
        -   [From CPU to
            GPU](using/examples/multi_gpu_workflows.html#from-cpu-to-gpu){.reference
            .internal}
        -   [Pooling the memory of multiple GPUs ([`mgpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](using/examples/multi_gpu_workflows.html#pooling-the-memory-of-multiple-gpus-mgpu){.reference
            .internal}
        -   [Parallel execution over multiple QPUs ([`mqpu`{.code
            .docutils .literal
            .notranslate}]{.pre})](using/examples/multi_gpu_workflows.html#parallel-execution-over-multiple-qpus-mqpu){.reference
            .internal}
            -   [Batching Hamiltonian
                Terms](using/examples/multi_gpu_workflows.html#batching-hamiltonian-terms){.reference
                .internal}
            -   [Circuit
                Batching](using/examples/multi_gpu_workflows.html#circuit-batching){.reference
                .internal}
        -   [Multi-QPU + Other Backends ([`remote-mqpu`{.code .docutils
            .literal
            .notranslate}]{.pre})](using/examples/multi_gpu_workflows.html#multi-qpu-other-backends-remote-mqpu){.reference
            .internal}
    -   [Optimizers &
        Gradients](examples/python/optimizers_gradients.html){.reference
        .internal}
        -   [Built in CUDA-Q Optimizers and
            Gradients](examples/python/optimizers_gradients.html#Built-in-CUDA-Q-Optimizers-and-Gradients){.reference
            .internal}
        -   [Third-Party
            Optimizers](examples/python/optimizers_gradients.html#Third-Party-Optimizers){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](examples/python/optimizers_gradients.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
    -   [Noisy
        Simulations](examples/python/noisy_simulations.html){.reference
        .internal}
    -   [Constructing
        Operators](using/examples/operators.html){.reference .internal}
        -   [Constructing Spin
            Operators](using/examples/operators.html#constructing-spin-operators){.reference
            .internal}
        -   [Pauli Words and Exponentiating Pauli
            Words](using/examples/operators.html#pauli-words-and-exponentiating-pauli-words){.reference
            .internal}
    -   [Performance
        Optimizations](examples/python/performance_optimizations.html){.reference
        .internal}
        -   [Gate
            Fusion](examples/python/performance_optimizations.html#Gate-Fusion){.reference
            .internal}
    -   [Using Quantum Hardware
        Providers](using/examples/hardware_providers.html){.reference
        .internal}
        -   [Amazon
            Braket](using/examples/hardware_providers.html#amazon-braket){.reference
            .internal}
        -   [Anyon
            Technologies](using/examples/hardware_providers.html#anyon-technologies){.reference
            .internal}
        -   [Infleqtion](using/examples/hardware_providers.html#infleqtion){.reference
            .internal}
        -   [IonQ](using/examples/hardware_providers.html#ionq){.reference
            .internal}
        -   [IQM](using/examples/hardware_providers.html#iqm){.reference
            .internal}
        -   [OQC](using/examples/hardware_providers.html#oqc){.reference
            .internal}
        -   [ORCA
            Computing](using/examples/hardware_providers.html#orca-computing){.reference
            .internal}
        -   [Pasqal](using/examples/hardware_providers.html#pasqal){.reference
            .internal}
        -   [Quantinuum](using/examples/hardware_providers.html#quantinuum){.reference
            .internal}
        -   [Quantum Circuits,
            Inc.](using/examples/hardware_providers.html#quantum-circuits-inc){.reference
            .internal}
        -   [Quantum
            Machines](using/examples/hardware_providers.html#quantum-machines){.reference
            .internal}
        -   [QuEra
            Computing](using/examples/hardware_providers.html#quera-computing){.reference
            .internal}
    -   [Dynamics
        Examples](using/examples/dynamics_examples.html){.reference
        .internal}
        -   [Introduction to CUDA-Q Dynamics (Jaynes-Cummings
            Model)](examples/python/dynamics/dynamics_intro_1.html){.reference
            .internal}
            -   [Why dynamics simulations vs. circuit
                simulations?](examples/python/dynamics/dynamics_intro_1.html#Why-dynamics-simulations-vs.-circuit-simulations?){.reference
                .internal}
            -   [Functionality](examples/python/dynamics/dynamics_intro_1.html#Functionality){.reference
                .internal}
            -   [Performance](examples/python/dynamics/dynamics_intro_1.html#Performance){.reference
                .internal}
            -   [Section 1 - Simulating the Jaynes-Cummings
                Hamiltonian](examples/python/dynamics/dynamics_intro_1.html#Section-1---Simulating-the-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Exercise 1 - Simulating a many-photon Jaynes-Cummings
                Hamiltonian](examples/python/dynamics/dynamics_intro_1.html#Exercise-1---Simulating-a-many-photon-Jaynes-Cummings-Hamiltonian){.reference
                .internal}
            -   [Section 2 - Simulating open quantum systems with the
                [`collapse_operators`{.docutils .literal
                .notranslate}]{.pre}](examples/python/dynamics/dynamics_intro_1.html#Section-2---Simulating-open-quantum-systems-with-the-collapse_operators){.reference
                .internal}
            -   [Exercise 2 - Adding additional jump operators
                [\\(L_i\\)]{.math .notranslate
                .nohighlight}](examples/python/dynamics/dynamics_intro_1.html#Exercise-2---Adding-additional-jump-operators-L_i){.reference
                .internal}
            -   [Section 3 - Many qubits coupled to the
                resonator](examples/python/dynamics/dynamics_intro_1.html#Section-3---Many-qubits-coupled-to-the-resonator){.reference
                .internal}
        -   [Introduction to CUDA-Q Dynamics (Time Dependent
            Hamiltonians)](examples/python/dynamics/dynamics_intro_2.html){.reference
            .internal}
            -   [The Landau-Zener
                model](examples/python/dynamics/dynamics_intro_2.html#The-Landau-Zener-model){.reference
                .internal}
            -   [Section 1 - Implementing time dependent
                terms](examples/python/dynamics/dynamics_intro_2.html#Section-1---Implementing-time-dependent-terms){.reference
                .internal}
            -   [Section 2 - Implementing custom
                operators](examples/python/dynamics/dynamics_intro_2.html#Section-2---Implementing-custom-operators){.reference
                .internal}
            -   [Section 3 - Heisenberg Model with a time-varying
                magnetic
                field](examples/python/dynamics/dynamics_intro_2.html#Section-3---Heisenberg-Model-with-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 1 - Define a time-varying magnetic
                field](examples/python/dynamics/dynamics_intro_2.html#Exercise-1---Define-a-time-varying-magnetic-field){.reference
                .internal}
            -   [Exercise 2
                (Optional)](examples/python/dynamics/dynamics_intro_2.html#Exercise-2-(Optional)){.reference
                .internal}
        -   [Superconducting
            Qubits](examples/python/dynamics/superconducting.html){.reference
            .internal}
            -   [Cavity
                QED](examples/python/dynamics/superconducting.html#Cavity-QED){.reference
                .internal}
            -   [Cross
                Resonance](examples/python/dynamics/superconducting.html#Cross-Resonance){.reference
                .internal}
            -   [Transmon
                Resonator](examples/python/dynamics/superconducting.html#Transmon-Resonator){.reference
                .internal}
        -   [Spin
            Qubits](examples/python/dynamics/spinqubits.html){.reference
            .internal}
            -   [Silicon Spin
                Qubit](examples/python/dynamics/spinqubits.html#Silicon-Spin-Qubit){.reference
                .internal}
            -   [Heisenberg
                Model](examples/python/dynamics/spinqubits.html#Heisenberg-Model){.reference
                .internal}
        -   [Trapped Ion
            Qubits](examples/python/dynamics/iontrap.html){.reference
            .internal}
            -   [GHZ
                state](examples/python/dynamics/iontrap.html#GHZ-state){.reference
                .internal}
        -   [Control](examples/python/dynamics/control.html){.reference
            .internal}
            -   [Gate
                Calibration](examples/python/dynamics/control.html#Gate-Calibration){.reference
                .internal}
            -   [Pulse](examples/python/dynamics/control.html#Pulse){.reference
                .internal}
            -   [Qubit
                Control](examples/python/dynamics/control.html#Qubit-Control){.reference
                .internal}
            -   [Qubit
                Dynamics](examples/python/dynamics/control.html#Qubit-Dynamics){.reference
                .internal}
            -   [Landau-Zenner](examples/python/dynamics/control.html#Landau-Zenner){.reference
                .internal}
-   [Applications](using/applications.html){.reference .internal}
    -   [Max-Cut with QAOA](applications/python/qaoa.html){.reference
        .internal}
    -   [Molecular docking via
        DC-QAOA](applications/python/digitized_counterdiabatic_qaoa.html){.reference
        .internal}
        -   [Setting up the Molecular Docking
            Problem](applications/python/digitized_counterdiabatic_qaoa.html#Setting-up-the-Molecular-Docking-Problem){.reference
            .internal}
        -   [CUDA-Q
            Implementation](applications/python/digitized_counterdiabatic_qaoa.html#CUDA-Q-Implementation){.reference
            .internal}
    -   [Multi-reference Quantum Krylov Algorithm - [\\(H_2\\)]{.math
        .notranslate .nohighlight}
        Molecule](applications/python/krylov.html){.reference .internal}
        -   [Setup](applications/python/krylov.html#Setup){.reference
            .internal}
        -   [Computing the matrix
            elements](applications/python/krylov.html#Computing-the-matrix-elements){.reference
            .internal}
        -   [Determining the ground state energy of the
            subspace](applications/python/krylov.html#Determining-the-ground-state-energy-of-the-subspace){.reference
            .internal}
    -   [Quantum-Selected Configuration Interaction
        (QSCI)](applications/python/qsci.html){.reference .internal}
        -   [0. Problem
            definition](applications/python/qsci.html#0.-Problem-definition){.reference
            .internal}
        -   [1. Prepare an Approximate Quantum
            State](applications/python/qsci.html#1.-Prepare-an-Approximate-Quantum-State){.reference
            .internal}
        -   [2 Quantum Sampling to Select
            Configuration](applications/python/qsci.html#2-Quantum-Sampling-to-Select-Configuration){.reference
            .internal}
        -   [3. Classical Diagonalization on the Selected
            Subspace](applications/python/qsci.html#3.-Classical-Diagonalization-on-the-Selected-Subspace){.reference
            .internal}
        -   [5. Compuare
            results](applications/python/qsci.html#5.-Compuare-results){.reference
            .internal}
        -   [Reference](applications/python/qsci.html#Reference){.reference
            .internal}
    -   [Bernstein-Vazirani
        Algorithm](applications/python/bernstein_vazirani.html){.reference
        .internal}
        -   [Classical
            case](applications/python/bernstein_vazirani.html#Classical-case){.reference
            .internal}
        -   [Quantum
            case](applications/python/bernstein_vazirani.html#Quantum-case){.reference
            .internal}
        -   [Implementing in
            CUDA-Q](applications/python/bernstein_vazirani.html#Implementing-in-CUDA-Q){.reference
            .internal}
    -   [Cost
        Minimization](applications/python/cost_minimization.html){.reference
        .internal}
    -   [Deutsch's
        Algorithm](applications/python/deutsch_algorithm.html){.reference
        .internal}
        -   [XOR [\\(\\oplus\\)]{.math .notranslate
            .nohighlight}](applications/python/deutsch_algorithm.html#XOR-\oplus){.reference
            .internal}
        -   [Quantum
            oracles](applications/python/deutsch_algorithm.html#Quantum-oracles){.reference
            .internal}
        -   [Phase
            oracle](applications/python/deutsch_algorithm.html#Phase-oracle){.reference
            .internal}
        -   [Quantum
            parallelism](applications/python/deutsch_algorithm.html#Quantum-parallelism){.reference
            .internal}
        -   [Deutsch's
            Algorithm:](applications/python/deutsch_algorithm.html#Deutsch's-Algorithm:){.reference
            .internal}
    -   [Divisive Clustering With Coresets Using
        CUDA-Q](applications/python/divisive_clustering_coresets.html){.reference
        .internal}
        -   [Data
            preprocessing](applications/python/divisive_clustering_coresets.html#Data-preprocessing){.reference
            .internal}
        -   [Quantum
            functions](applications/python/divisive_clustering_coresets.html#Quantum-functions){.reference
            .internal}
        -   [Divisive Clustering
            Function](applications/python/divisive_clustering_coresets.html#Divisive-Clustering-Function){.reference
            .internal}
        -   [QAOA
            Implementation](applications/python/divisive_clustering_coresets.html#QAOA-Implementation){.reference
            .internal}
        -   [Scaling simulations with
            CUDA-Q](applications/python/divisive_clustering_coresets.html#Scaling-simulations-with-CUDA-Q){.reference
            .internal}
    -   [Hybrid Quantum Neural
        Networks](applications/python/hybrid_quantum_neural_networks.html){.reference
        .internal}
    -   [Using the Hadamard Test to Determine Quantum Krylov Subspace
        Decomposition Matrix
        Elements](applications/python/hadamard_test.html){.reference
        .internal}
        -   [Numerical result as a
            reference:](applications/python/hadamard_test.html#Numerical-result-as-a-reference:){.reference
            .internal}
        -   [Using [`Sample`{.docutils .literal .notranslate}]{.pre} to
            perform the Hadamard
            test](applications/python/hadamard_test.html#Using-Sample-to-perform-the-Hadamard-test){.reference
            .internal}
        -   [Multi-GPU evaluation of QKSD matrix elements using the
            Hadamard
            Test](applications/python/hadamard_test.html#Multi-GPU-evaluation-of-QKSD-matrix-elements-using-the-Hadamard-Test){.reference
            .internal}
            -   [Classically Diagonalize the Subspace
                Matrix](applications/python/hadamard_test.html#Classically-Diagonalize-the-Subspace-Matrix){.reference
                .internal}
    -   [Anderson Impurity Model ground state solver on Infleqtion's
        Sqale](applications/python/logical_aim_sqale.html){.reference
        .internal}
        -   [Performing logical Variational Quantum Eigensolver (VQE)
            with
            CUDA-QX](applications/python/logical_aim_sqale.html#Performing-logical-Variational-Quantum-Eigensolver-(VQE)-with-CUDA-QX){.reference
            .internal}
        -   [Constructing circuits in the [`[[4,2,2]]`{.docutils
            .literal .notranslate}]{.pre}
            encoding](applications/python/logical_aim_sqale.html#Constructing-circuits-in-the-%5B%5B4,2,2%5D%5D-encoding){.reference
            .internal}
        -   [Setting up submission and decoding
            workflow](applications/python/logical_aim_sqale.html#Setting-up-submission-and-decoding-workflow){.reference
            .internal}
        -   [Running a CUDA-Q noisy
            simulation](applications/python/logical_aim_sqale.html#Running-a-CUDA-Q-noisy-simulation){.reference
            .internal}
        -   [Running logical AIM on Infleqtion's
            hardware](applications/python/logical_aim_sqale.html#Running-logical-AIM-on-Infleqtion's-hardware){.reference
            .internal}
    -   [Spin-Hamiltonian Simulation Using
        CUDA-Q](applications/python/hamiltonian_simulation.html){.reference
        .internal}
        -   [Introduction](applications/python/hamiltonian_simulation.html#Introduction){.reference
            .internal}
            -   [Heisenberg
                Hamiltonian](applications/python/hamiltonian_simulation.html#Heisenberg-Hamiltonian){.reference
                .internal}
            -   [Transverse Field Ising Model
                (TFIM)](applications/python/hamiltonian_simulation.html#Transverse-Field-Ising-Model-(TFIM)){.reference
                .internal}
            -   [Time Evolution and Trotter
                Decomposition](applications/python/hamiltonian_simulation.html#Time-Evolution-and-Trotter-Decomposition){.reference
                .internal}
        -   [Key
            steps](applications/python/hamiltonian_simulation.html#Key-steps){.reference
            .internal}
            -   [1. Prepare initial
                state](applications/python/hamiltonian_simulation.html#1.-Prepare-initial-state){.reference
                .internal}
            -   [2. Hamiltonian
                Trotterization](applications/python/hamiltonian_simulation.html#2.-Hamiltonian-Trotterization){.reference
                .internal}
            -   [3. [`Compute`{.docutils .literal
                .notranslate}]{.pre}` `{.docutils .literal
                .notranslate}[`overlap`{.docutils .literal
                .notranslate}]{.pre}](applications/python/hamiltonian_simulation.html#3.-Compute-overlap){.reference
                .internal}
            -   [4. Construct Heisenberg
                Hamiltonian](applications/python/hamiltonian_simulation.html#4.-Construct-Heisenberg-Hamiltonian){.reference
                .internal}
            -   [5. Construct TFIM
                Hamiltonian](applications/python/hamiltonian_simulation.html#5.-Construct-TFIM-Hamiltonian){.reference
                .internal}
            -   [6. Extract coefficients and Pauli
                words](applications/python/hamiltonian_simulation.html#6.-Extract-coefficients-and-Pauli-words){.reference
                .internal}
        -   [Main
            code](applications/python/hamiltonian_simulation.html#Main-code){.reference
            .internal}
        -   [Visualization of probablity over
            time](applications/python/hamiltonian_simulation.html#Visualization-of-probablity-over-time){.reference
            .internal}
        -   [Expectation value over
            time:](applications/python/hamiltonian_simulation.html#Expectation-value-over-time:){.reference
            .internal}
        -   [Visualization of expectation over
            time](applications/python/hamiltonian_simulation.html#Visualization-of-expectation-over-time){.reference
            .internal}
        -   [Additional
            information](applications/python/hamiltonian_simulation.html#Additional-information){.reference
            .internal}
        -   [Relevant
            references](applications/python/hamiltonian_simulation.html#Relevant-references){.reference
            .internal}
    -   [Quantum Fourier
        Transform](applications/python/quantum_fourier_transform.html){.reference
        .internal}
        -   [Quantum Fourier Transform
            revisited](applications/python/quantum_fourier_transform.html#Quantum-Fourier-Transform-revisited){.reference
            .internal}
    -   [Quantum
        Teleporation](applications/python/quantum_teleportation.html){.reference
        .internal}
        -   [Teleportation
            explained](applications/python/quantum_teleportation.html#Teleportation-explained){.reference
            .internal}
    -   [Quantum
        Volume](applications/python/quantum_volume.html){.reference
        .internal}
    -   [Readout Error
        Mitigation](applications/python/readout_error_mitigation.html){.reference
        .internal}
        -   [Inverse confusion matrix from single-qubit noise
            model](applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-single-qubit-noise-model){.reference
            .internal}
        -   [Inverse confusion matrix from k local confusion
            matrices](applications/python/readout_error_mitigation.html#Inverse-confusion-matrix-from-k-local-confusion-matrices){.reference
            .internal}
        -   [Inverse of full confusion
            matrix](applications/python/readout_error_mitigation.html#Inverse-of-full-confusion-matrix){.reference
            .internal}
    -   [Compiling Unitaries Using Diffusion
        Models](applications/python/unitary_compilation_diffusion_models.html){.reference
        .internal}
        -   [Diffusion model
            pipeline](applications/python/unitary_compilation_diffusion_models.html#Diffusion-model-pipeline){.reference
            .internal}
        -   [Setup and load
            models](applications/python/unitary_compilation_diffusion_models.html#Setup-and-load-models){.reference
            .internal}
            -   [Load discrete
                model](applications/python/unitary_compilation_diffusion_models.html#Load-discrete-model){.reference
                .internal}
            -   [Load continuous
                model](applications/python/unitary_compilation_diffusion_models.html#Load-continuous-model){.reference
                .internal}
            -   [Create helper
                functions](applications/python/unitary_compilation_diffusion_models.html#Create-helper-functions){.reference
                .internal}
        -   [Unitary
            compilation](applications/python/unitary_compilation_diffusion_models.html#Unitary-compilation){.reference
            .internal}
            -   [Random
                unitary](applications/python/unitary_compilation_diffusion_models.html#Random-unitary){.reference
                .internal}
            -   [Discrete
                model](applications/python/unitary_compilation_diffusion_models.html#Discrete-model){.reference
                .internal}
            -   [Continuous
                model](applications/python/unitary_compilation_diffusion_models.html#Continuous-model){.reference
                .internal}
            -   [Quantum Fourier
                transform](applications/python/unitary_compilation_diffusion_models.html#Quantum-Fourier-transform){.reference
                .internal}
            -   [XXZ-Hamiltonian
                evolution](applications/python/unitary_compilation_diffusion_models.html#XXZ-Hamiltonian-evolution){.reference
                .internal}
        -   [Choosing the circuit you
            need](applications/python/unitary_compilation_diffusion_models.html#Choosing-the-circuit-you-need){.reference
            .internal}
    -   [VQE with gradients, active spaces, and gate
        fusion](applications/python/vqe_advanced.html){.reference
        .internal}
        -   [The Basics of
            VQE](applications/python/vqe_advanced.html#The-Basics-of-VQE){.reference
            .internal}
        -   [Installing/Loading Relevant
            Packages](applications/python/vqe_advanced.html#Installing/Loading-Relevant-Packages){.reference
            .internal}
        -   [Implementing VQE in
            CUDA-Q](applications/python/vqe_advanced.html#Implementing-VQE-in-CUDA-Q){.reference
            .internal}
        -   [Parallel Parameter Shift
            Gradients](applications/python/vqe_advanced.html#Parallel-Parameter-Shift-Gradients){.reference
            .internal}
        -   [Using an Active
            Space](applications/python/vqe_advanced.html#Using-an-Active-Space){.reference
            .internal}
        -   [Gate Fusion for Larger
            Circuits](applications/python/vqe_advanced.html#Gate-Fusion-for-Larger-Circuits){.reference
            .internal}
    -   [Quantum
        Transformer](applications/python/quantum_transformer.html){.reference
        .internal}
        -   [Installation](applications/python/quantum_transformer.html#Installation){.reference
            .internal}
        -   [Algorithm and
            Example](applications/python/quantum_transformer.html#Algorithm-and-Example){.reference
            .internal}
            -   [Creating the self-attention
                circuits](applications/python/quantum_transformer.html#Creating-the-self-attention-circuits){.reference
                .internal}
        -   [Usage](applications/python/quantum_transformer.html#Usage){.reference
            .internal}
            -   [Model
                Training](applications/python/quantum_transformer.html#Model-Training){.reference
                .internal}
            -   [Generating
                Molecules](applications/python/quantum_transformer.html#Generating-Molecules){.reference
                .internal}
            -   [Attention
                Maps](applications/python/quantum_transformer.html#Attention-Maps){.reference
                .internal}
    -   [Quantum Enhanced Auxiliary Field Quantum Monte
        Carlo](applications/python/afqmc.html){.reference .internal}
        -   [Hamiltonian preparation for
            VQE](applications/python/afqmc.html#Hamiltonian-preparation-for-VQE){.reference
            .internal}
        -   [Run VQE with
            CUDA-Q](applications/python/afqmc.html#Run-VQE-with-CUDA-Q){.reference
            .internal}
        -   [Auxiliary Field Quantum Monte Carlo
            (AFQMC)](applications/python/afqmc.html#Auxiliary-Field-Quantum-Monte-Carlo-(AFQMC)){.reference
            .internal}
        -   [Preparation of the molecular
            Hamiltonian](applications/python/afqmc.html#Preparation-of-the-molecular-Hamiltonian){.reference
            .internal}
        -   [Preparation of the trial wave
            function](applications/python/afqmc.html#Preparation-of-the-trial-wave-function){.reference
            .internal}
        -   [Setup of the AFQMC
            parameters](applications/python/afqmc.html#Setup-of-the-AFQMC-parameters){.reference
            .internal}
    -   [ADAPT-QAOA
        algorithm](applications/python/adapt_qaoa.html){.reference
        .internal}
        -   [Simulation
            input:](applications/python/adapt_qaoa.html#Simulation-input:){.reference
            .internal}
        -   [The problem Hamiltonian [\\(H_C\\)]{.math .notranslate
            .nohighlight} of the max-cut
            graph:](applications/python/adapt_qaoa.html#The-problem-Hamiltonian-H_C-of-the-max-cut-graph:){.reference
            .internal}
        -   [Th operator pool [\\(A_j\\)]{.math .notranslate
            .nohighlight}:](applications/python/adapt_qaoa.html#Th-operator-pool-A_j:){.reference
            .internal}
        -   [The commutator [\\(\[H_C,A_j\]\\)]{.math .notranslate
            .nohighlight}:](applications/python/adapt_qaoa.html#The-commutator-%5BH_C,A_j%5D:){.reference
            .internal}
        -   [Beginning of ADAPT-QAOA
            iteration:](applications/python/adapt_qaoa.html#Beginning-of-ADAPT-QAOA-iteration:){.reference
            .internal}
    -   [ADAPT-VQE
        algorithm](applications/python/adapt_vqe.html){.reference
        .internal}
        -   [Classical
            pre-processing](applications/python/adapt_vqe.html#Classical-pre-processing){.reference
            .internal}
        -   [Jordan
            Wigner:](applications/python/adapt_vqe.html#Jordan-Wigner:){.reference
            .internal}
        -   [UCCSD operator
            pool](applications/python/adapt_vqe.html#UCCSD-operator-pool){.reference
            .internal}
            -   [Single
                excitation](applications/python/adapt_vqe.html#Single-excitation){.reference
                .internal}
            -   [Double
                excitation](applications/python/adapt_vqe.html#Double-excitation){.reference
                .internal}
        -   [Commutator \[[\\(H\\)]{.math .notranslate .nohighlight},
            [\\(A_i\\)]{.math .notranslate
            .nohighlight}\]](applications/python/adapt_vqe.html#Commutator-%5BH,-A_i%5D){.reference
            .internal}
        -   [Reference
            State:](applications/python/adapt_vqe.html#Reference-State:){.reference
            .internal}
        -   [Quantum
            kernels:](applications/python/adapt_vqe.html#Quantum-kernels:){.reference
            .internal}
        -   [Beginning of
            ADAPT-VQE:](applications/python/adapt_vqe.html#Beginning-of-ADAPT-VQE:){.reference
            .internal}
    -   [Quantum edge
        detection](applications/python/edge_detection.html){.reference
        .internal}
        -   [Image](applications/python/edge_detection.html#Image){.reference
            .internal}
        -   [Quantum Probability Image Encoding
            (QPIE):](applications/python/edge_detection.html#Quantum-Probability-Image-Encoding-(QPIE):){.reference
            .internal}
            -   [Below we show how to encode an image using QPIE in
                cudaq.](applications/python/edge_detection.html#Below-we-show-how-to-encode-an-image-using-QPIE-in-cudaq.){.reference
                .internal}
        -   [Flexible Representation of Quantum Images
            (FRQI):](applications/python/edge_detection.html#Flexible-Representation-of-Quantum-Images-(FRQI):){.reference
            .internal}
            -   [Building the FRQI
                State:](applications/python/edge_detection.html#Building-the-FRQI-State:){.reference
                .internal}
        -   [Quantum Hadamard Edge Detection
            (QHED)](applications/python/edge_detection.html#Quantum-Hadamard-Edge-Detection-(QHED)){.reference
            .internal}
            -   [Post-processing](applications/python/edge_detection.html#Post-processing){.reference
                .internal}
    -   [Factoring Integers With Shor's
        Algorithm](applications/python/shors.html){.reference .internal}
        -   [Shor's
            algorithm](applications/python/shors.html#Shor's-algorithm){.reference
            .internal}
            -   [Solving the order-finding problem
                classically](applications/python/shors.html#Solving-the-order-finding-problem-classically){.reference
                .internal}
            -   [Solving the order-finding problem with a quantum
                algorithm](applications/python/shors.html#Solving-the-order-finding-problem-with-a-quantum-algorithm){.reference
                .internal}
            -   [Determining the order from the measurement results of
                the phase
                kernel](applications/python/shors.html#Determining-the-order-from-the-measurement-results-of-the-phase-kernel){.reference
                .internal}
            -   [Postscript](applications/python/shors.html#Postscript){.reference
                .internal}
    -   [Generating the electronic
        Hamiltonian](applications/python/generate_fermionic_ham.html){.reference
        .internal}
        -   [Second Quantized
            formulation.](applications/python/generate_fermionic_ham.html#Second-Quantized-formulation.){.reference
            .internal}
            -   [Computational
                Implementation](applications/python/generate_fermionic_ham.html#Computational-Implementation){.reference
                .internal}
            -   [(a) Generate the molecular Hamiltonian using Restricted
                Hartree Fock molecular
                orbitals](applications/python/generate_fermionic_ham.html#(a)-Generate-the-molecular-Hamiltonian-using-Restricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(b) Generate the molecular Hamiltonian using
                Unrestricted Hartree Fock molecular
                orbitals](applications/python/generate_fermionic_ham.html#(b)-Generate-the-molecular-Hamiltonian-using-Unrestricted-Hartree-Fock-molecular-orbitals){.reference
                .internal}
            -   [(a) Generate the active space hamiltonian using RHF
                molecular
                orbitals.](applications/python/generate_fermionic_ham.html#(a)-Generate-the-active-space-hamiltonian-using-RHF-molecular-orbitals.){.reference
                .internal}
            -   [(b) Generate the active space Hamiltonian using the
                natural orbitals computed from MP2
                simulation](applications/python/generate_fermionic_ham.html#(b)-Generate-the-active-space-Hamiltonian-using-the-natural-orbitals-computed-from-MP2-simulation){.reference
                .internal}
            -   [(c) Generate the active space Hamiltonian computed from
                the CASSCF molecular
                orbitals](applications/python/generate_fermionic_ham.html#(c)-Generate-the-active-space-Hamiltonian-computed-from-the-CASSCF-molecular-orbitals){.reference
                .internal}
            -   [(d) Generate the electronic Hamiltonian using
                ROHF](applications/python/generate_fermionic_ham.html#(d)-Generate-the-electronic-Hamiltonian-using-ROHF){.reference
                .internal}
            -   [(e) Generate electronic Hamiltonian using
                UHF](applications/python/generate_fermionic_ham.html#(e)-Generate-electronic-Hamiltonian-using-UHF){.reference
                .internal}
    -   [Grover's
        Algorithm](applications/python/grovers.html){.reference
        .internal}
        -   [Overview](applications/python/grovers.html#Overview){.reference
            .internal}
        -   [Problem](applications/python/grovers.html#Problem){.reference
            .internal}
        -   [Structure of Grover's
            Algorithm](applications/python/grovers.html#Structure-of-Grover's-Algorithm){.reference
            .internal}
            -   [Step 1:
                Preparation](applications/python/grovers.html#Step-1:-Preparation){.reference
                .internal}
            -   [Good and Bad
                States](applications/python/grovers.html#Good-and-Bad-States){.reference
                .internal}
            -   [Step 2: Oracle
                application](applications/python/grovers.html#Step-2:-Oracle-application){.reference
                .internal}
            -   [Step 3: Amplitude
                amplification](applications/python/grovers.html#Step-3:-Amplitude-amplification){.reference
                .internal}
            -   [Steps 4 and 5: Iteration and
                measurement](applications/python/grovers.html#Steps-4-and-5:-Iteration-and-measurement){.reference
                .internal}
    -   [Quantum
        PageRank](applications/python/quantum_pagerank.html){.reference
        .internal}
        -   [Problem
            Definition](applications/python/quantum_pagerank.html#Problem-Definition){.reference
            .internal}
        -   [Simulating Quantum PageRank by CUDA-Q
            dynamics](applications/python/quantum_pagerank.html#Simulating-Quantum-PageRank-by-CUDA-Q-dynamics){.reference
            .internal}
        -   [Breakdown of
            Terms](applications/python/quantum_pagerank.html#Breakdown-of-Terms){.reference
            .internal}
    -   [The UCCSD Wavefunction
        ansatz](applications/python/uccsd_wf_ansatz.html){.reference
        .internal}
        -   [What is
            UCCSD?](applications/python/uccsd_wf_ansatz.html#What-is-UCCSD?){.reference
            .internal}
        -   [Implementation in Quantum
            Computing](applications/python/uccsd_wf_ansatz.html#Implementation-in-Quantum-Computing){.reference
            .internal}
        -   [Run
            VQE](applications/python/uccsd_wf_ansatz.html#Run-VQE){.reference
            .internal}
        -   [Challenges and
            consideration](applications/python/uccsd_wf_ansatz.html#Challenges-and-consideration){.reference
            .internal}
    -   [Approximate State Preparation using MPS Sequential
        Encoding](applications/python/mps_encoding.html){.reference
        .internal}
        -   [Ran's
            approach](applications/python/mps_encoding.html#Ran's-approach){.reference
            .internal}
    -   [QM/MM simulation: VQE within a Polarizable Embedded
        Framework.](applications/python/qm_mm_pe.html){.reference
        .internal}
        -   [Key
            concepts:](applications/python/qm_mm_pe.html#Key-concepts:){.reference
            .internal}
        -   [PE-VQE-SCF Algorithm
            Steps](applications/python/qm_mm_pe.html#PE-VQE-SCF-Algorithm-Steps){.reference
            .internal}
            -   [Step 1: Initialize (Classical
                pre-processing)](applications/python/qm_mm_pe.html#Step-1:-Initialize-(Classical-pre-processing)){.reference
                .internal}
            -   [Step 2: Build the
                Hamiltonian](applications/python/qm_mm_pe.html#Step-2:-Build-the-Hamiltonian){.reference
                .internal}
            -   [Step 3: Run
                VQE](applications/python/qm_mm_pe.html#Step-3:-Run-VQE){.reference
                .internal}
            -   [Step 4: Update
                Environment](applications/python/qm_mm_pe.html#Step-4:-Update-Environment){.reference
                .internal}
            -   [Step 5: Self-Consistency
                Loop](applications/python/qm_mm_pe.html#Step-5:-Self-Consistency-Loop){.reference
                .internal}
            -   [Requirments:](applications/python/qm_mm_pe.html#Requirments:){.reference
                .internal}
            -   [Example 1: LiH with 2 water
                molecules.](applications/python/qm_mm_pe.html#Example-1:-LiH-with-2-water-molecules.){.reference
                .internal}
            -   [VQE, update environment, and scf
                loop.](applications/python/qm_mm_pe.html#VQE,-update-environment,-and-scf-loop.){.reference
                .internal}
            -   [Example 2: NH3 with 46 water molecule using active
                space.](applications/python/qm_mm_pe.html#Example-2:-NH3-with-46-water-molecule-using-active-space.){.reference
                .internal}
    -   [Sample-Based Krylov Quantum Diagonalization
        (SKQD)](applications/python/skqd.html){.reference .internal}
        -   [Why
            SKQD?](applications/python/skqd.html#Why-SKQD?){.reference
            .internal}
        -   [Understanding Krylov
            Subspaces](applications/python/skqd.html#Understanding-Krylov-Subspaces){.reference
            .internal}
            -   [What is a Krylov
                Subspace?](applications/python/skqd.html#What-is-a-Krylov-Subspace?){.reference
                .internal}
            -   [The SKQD
                Algorithm](applications/python/skqd.html#The-SKQD-Algorithm){.reference
                .internal}
        -   [Problem Setup: 22-Qubit Heisenberg
            Model](applications/python/skqd.html#Problem-Setup:-22-Qubit-Heisenberg-Model){.reference
            .internal}
        -   [Krylov State Generation via Repeated
            Evolution](applications/python/skqd.html#Krylov-State-Generation-via-Repeated-Evolution){.reference
            .internal}
        -   [Quantum Measurements and
            Sampling](applications/python/skqd.html#Quantum-Measurements-and-Sampling){.reference
            .internal}
            -   [The Sampling
                Process](applications/python/skqd.html#The-Sampling-Process){.reference
                .internal}
        -   [Classical Post-Processing and
            Diagonalization](applications/python/skqd.html#Classical-Post-Processing-and-Diagonalization){.reference
            .internal}
            -   [The SKQD Algorithm: Matrix Construction
                Details](applications/python/skqd.html#The-SKQD-Algorithm:-Matrix-Construction-Details){.reference
                .internal}
        -   [Results Analysis and
            Convergence](applications/python/skqd.html#Results-Analysis-and-Convergence){.reference
            .internal}
            -   [What to
                Expect:](applications/python/skqd.html#What-to-Expect:){.reference
                .internal}
        -   [GPU Acceleration for
            Postprocessing](applications/python/skqd.html#GPU-Acceleration-for-Postprocessing){.reference
            .internal}
    -   [Entanglement Accelerates Quantum
        Simulation](applications/python/entanglement_acc_hamiltonian_simulation.html){.reference
        .internal}
        -   [2. Model
            Definition](applications/python/entanglement_acc_hamiltonian_simulation.html#2.-Model-Definition){.reference
            .internal}
            -   [2.1 Initial product
                state](applications/python/entanglement_acc_hamiltonian_simulation.html#2.1-Initial-product-state){.reference
                .internal}
            -   [2.2 QIMF
                Hamiltonian](applications/python/entanglement_acc_hamiltonian_simulation.html#2.2-QIMF-Hamiltonian){.reference
                .internal}
            -   [2.3 First-Order Trotter Formula
                (PF1)](applications/python/entanglement_acc_hamiltonian_simulation.html#2.3-First-Order-Trotter-Formula-(PF1)){.reference
                .internal}
            -   [2.4 PF1 step for the QIMF
                partition](applications/python/entanglement_acc_hamiltonian_simulation.html#2.4-PF1-step-for-the-QIMF-partition){.reference
                .internal}
            -   [2.5 Hamiltonian
                helpers](applications/python/entanglement_acc_hamiltonian_simulation.html#2.5-Hamiltonian-helpers){.reference
                .internal}
        -   [3. Entanglement
            metrics](applications/python/entanglement_acc_hamiltonian_simulation.html#3.-Entanglement-metrics){.reference
            .internal}
        -   [4. Simulation
            workflow](applications/python/entanglement_acc_hamiltonian_simulation.html#4.-Simulation-workflow){.reference
            .internal}
            -   [4.1 Single-step Trotter
                error](applications/python/entanglement_acc_hamiltonian_simulation.html#4.1-Single-step-Trotter-error){.reference
                .internal}
            -   [4.2 Dual trajectory
                update](applications/python/entanglement_acc_hamiltonian_simulation.html#4.2-Dual-trajectory-update){.reference
                .internal}
        -   [5. Reproducing the paper's Figure
            1a](applications/python/entanglement_acc_hamiltonian_simulation.html#5.-Reproducing-the-paper’s-Figure-1a){.reference
            .internal}
            -   [5.1 Visualising the joint
                behaviour](applications/python/entanglement_acc_hamiltonian_simulation.html#5.1-Visualising-the-joint-behaviour){.reference
                .internal}
            -   [5.2 Interpreting the
                result](applications/python/entanglement_acc_hamiltonian_simulation.html#5.2-Interpreting-the-result){.reference
                .internal}
        -   [6. References and further
            reading](applications/python/entanglement_acc_hamiltonian_simulation.html#6.-References-and-further-reading){.reference
            .internal}
-   [Backends](using/backends/backends.html){.reference .internal}
    -   [Circuit Simulation](using/backends/simulators.html){.reference
        .internal}
        -   [State Vector
            Simulators](using/backends/sims/svsims.html){.reference
            .internal}
            -   [CPU](using/backends/sims/svsims.html#cpu){.reference
                .internal}
            -   [Single-GPU](using/backends/sims/svsims.html#single-gpu){.reference
                .internal}
            -   [Multi-GPU
                multi-node](using/backends/sims/svsims.html#multi-gpu-multi-node){.reference
                .internal}
        -   [Tensor Network
            Simulators](using/backends/sims/tnsims.html){.reference
            .internal}
            -   [Multi-GPU
                multi-node](using/backends/sims/tnsims.html#multi-gpu-multi-node){.reference
                .internal}
            -   [Matrix product
                state](using/backends/sims/tnsims.html#matrix-product-state){.reference
                .internal}
            -   [Fermioniq](using/backends/sims/tnsims.html#fermioniq){.reference
                .internal}
        -   [Multi-QPU
            Simulators](using/backends/sims/mqpusims.html){.reference
            .internal}
            -   [Simulate Multiple QPUs in
                Parallel](using/backends/sims/mqpusims.html#simulate-multiple-qpus-in-parallel){.reference
                .internal}
            -   [Multi-QPU + Other
                Backends](using/backends/sims/mqpusims.html#multi-qpu-other-backends){.reference
                .internal}
        -   [Noisy
            Simulators](using/backends/sims/noisy.html){.reference
            .internal}
            -   [Trajectory Noisy
                Simulation](using/backends/sims/noisy.html#trajectory-noisy-simulation){.reference
                .internal}
            -   [Density
                Matrix](using/backends/sims/noisy.html#density-matrix){.reference
                .internal}
            -   [Stim](using/backends/sims/noisy.html#stim){.reference
                .internal}
        -   [Photonics
            Simulators](using/backends/sims/photonics.html){.reference
            .internal}
            -   [orca-photonics](using/backends/sims/photonics.html#orca-photonics){.reference
                .internal}
    -   [Quantum Hardware
        (QPUs)](using/backends/hardware.html){.reference .internal}
        -   [Ion Trap
            QPUs](using/backends/hardware/iontrap.html){.reference
            .internal}
            -   [IonQ](using/backends/hardware/iontrap.html#ionq){.reference
                .internal}
            -   [Quantinuum](using/backends/hardware/iontrap.html#quantinuum){.reference
                .internal}
        -   [Superconducting
            QPUs](using/backends/hardware/superconducting.html){.reference
            .internal}
            -   [Anyon Technologies/Anyon
                Computing](using/backends/hardware/superconducting.html#anyon-technologies-anyon-computing){.reference
                .internal}
            -   [IQM](using/backends/hardware/superconducting.html#iqm){.reference
                .internal}
            -   [OQC](using/backends/hardware/superconducting.html#oqc){.reference
                .internal}
            -   [Quantum Circuits,
                Inc.](using/backends/hardware/superconducting.html#quantum-circuits-inc){.reference
                .internal}
        -   [Neutral Atom
            QPUs](using/backends/hardware/neutralatom.html){.reference
            .internal}
            -   [Infleqtion](using/backends/hardware/neutralatom.html#infleqtion){.reference
                .internal}
            -   [Pasqal](using/backends/hardware/neutralatom.html#pasqal){.reference
                .internal}
            -   [QuEra
                Computing](using/backends/hardware/neutralatom.html#quera-computing){.reference
                .internal}
        -   [Photonic
            QPUs](using/backends/hardware/photonic.html){.reference
            .internal}
            -   [ORCA
                Computing](using/backends/hardware/photonic.html#orca-computing){.reference
                .internal}
        -   [Quantum Control
            Systems](using/backends/hardware/qcontrol.html){.reference
            .internal}
            -   [Quantum
                Machines](using/backends/hardware/qcontrol.html#quantum-machines){.reference
                .internal}
    -   [Dynamics
        Simulation](using/backends/dynamics_backends.html){.reference
        .internal}
    -   [Cloud](using/backends/cloud.html){.reference .internal}
        -   [Amazon Braket
            (braket)](using/backends/cloud/braket.html){.reference
            .internal}
            -   [Setting
                Credentials](using/backends/cloud/braket.html#setting-credentials){.reference
                .internal}
            -   [Submission from
                C++](using/backends/cloud/braket.html#submission-from-c){.reference
                .internal}
            -   [Submission from
                Python](using/backends/cloud/braket.html#submission-from-python){.reference
                .internal}
-   [Dynamics](using/dynamics.html){.reference .internal}
    -   [Quick Start](using/dynamics.html#quick-start){.reference
        .internal}
    -   [Operator](using/dynamics.html#operator){.reference .internal}
    -   [Time-Dependent
        Dynamics](using/dynamics.html#time-dependent-dynamics){.reference
        .internal}
    -   [Super-operator
        Representation](using/dynamics.html#super-operator-representation){.reference
        .internal}
    -   [Numerical
        Integrators](using/dynamics.html#numerical-integrators){.reference
        .internal}
    -   [Batch
        simulation](using/dynamics.html#batch-simulation){.reference
        .internal}
    -   [Multi-GPU Multi-Node
        Execution](using/dynamics.html#multi-gpu-multi-node-execution){.reference
        .internal}
    -   [Examples](using/dynamics.html#examples){.reference .internal}
-   [CUDA-QX](using/cudaqx/cudaqx.html){.reference .internal}
    -   [CUDA-Q
        Solvers](using/cudaqx/cudaqx.html#cuda-q-solvers){.reference
        .internal}
    -   [CUDA-Q QEC](using/cudaqx/cudaqx.html#cuda-q-qec){.reference
        .internal}
-   [Installation](using/install/install.html){.reference .internal}
    -   [Local
        Installation](using/install/local_installation.html){.reference
        .internal}
        -   [Introduction](using/install/local_installation.html#introduction){.reference
            .internal}
            -   [Docker](using/install/local_installation.html#docker){.reference
                .internal}
            -   [Known Blackwell
                Issues](using/install/local_installation.html#known-blackwell-issues){.reference
                .internal}
            -   [Singularity](using/install/local_installation.html#singularity){.reference
                .internal}
            -   [Python
                wheels](using/install/local_installation.html#python-wheels){.reference
                .internal}
            -   [Pre-built
                binaries](using/install/local_installation.html#pre-built-binaries){.reference
                .internal}
        -   [Development with VS
            Code](using/install/local_installation.html#development-with-vs-code){.reference
            .internal}
            -   [Using a Docker
                container](using/install/local_installation.html#using-a-docker-container){.reference
                .internal}
            -   [Using a Singularity
                container](using/install/local_installation.html#using-a-singularity-container){.reference
                .internal}
        -   [Connecting to a Remote
            Host](using/install/local_installation.html#connecting-to-a-remote-host){.reference
            .internal}
            -   [Developing with Remote
                Tunnels](using/install/local_installation.html#developing-with-remote-tunnels){.reference
                .internal}
            -   [Remote Access via
                SSH](using/install/local_installation.html#remote-access-via-ssh){.reference
                .internal}
        -   [DGX
            Cloud](using/install/local_installation.html#dgx-cloud){.reference
            .internal}
            -   [Get
                Started](using/install/local_installation.html#get-started){.reference
                .internal}
            -   [Use
                JupyterLab](using/install/local_installation.html#use-jupyterlab){.reference
                .internal}
            -   [Use VS
                Code](using/install/local_installation.html#use-vs-code){.reference
                .internal}
        -   [Additional CUDA
            Tools](using/install/local_installation.html#additional-cuda-tools){.reference
            .internal}
            -   [Installation via
                PyPI](using/install/local_installation.html#installation-via-pypi){.reference
                .internal}
            -   [Installation In Container
                Images](using/install/local_installation.html#installation-in-container-images){.reference
                .internal}
            -   [Installing Pre-built
                Binaries](using/install/local_installation.html#installing-pre-built-binaries){.reference
                .internal}
        -   [Distributed Computing with
            MPI](using/install/local_installation.html#distributed-computing-with-mpi){.reference
            .internal}
        -   [Updating
            CUDA-Q](using/install/local_installation.html#updating-cuda-q){.reference
            .internal}
        -   [Dependencies and
            Compatibility](using/install/local_installation.html#dependencies-and-compatibility){.reference
            .internal}
        -   [Next
            Steps](using/install/local_installation.html#next-steps){.reference
            .internal}
    -   [Data Center
        Installation](using/install/data_center_install.html){.reference
        .internal}
        -   [Prerequisites](using/install/data_center_install.html#prerequisites){.reference
            .internal}
        -   [Build
            Dependencies](using/install/data_center_install.html#build-dependencies){.reference
            .internal}
            -   [CUDA](using/install/data_center_install.html#cuda){.reference
                .internal}
            -   [Toolchain](using/install/data_center_install.html#toolchain){.reference
                .internal}
        -   [Building
            CUDA-Q](using/install/data_center_install.html#building-cuda-q){.reference
            .internal}
        -   [Python
            Support](using/install/data_center_install.html#python-support){.reference
            .internal}
        -   [C++
            Support](using/install/data_center_install.html#c-support){.reference
            .internal}
        -   [Installation on the
            Host](using/install/data_center_install.html#installation-on-the-host){.reference
            .internal}
            -   [CUDA Runtime
                Libraries](using/install/data_center_install.html#cuda-runtime-libraries){.reference
                .internal}
            -   [MPI](using/install/data_center_install.html#mpi){.reference
                .internal}
-   [Integration](using/integration/integration.html){.reference
    .internal}
    -   [Downstream CMake
        Integration](using/integration/cmake_app.html){.reference
        .internal}
    -   [Combining CUDA with
        CUDA-Q](using/integration/cuda_gpu.html){.reference .internal}
    -   [Integrating with Third-Party
        Libraries](using/integration/libraries.html){.reference
        .internal}
        -   [Calling a CUDA-Q library from
            C++](using/integration/libraries.html#calling-a-cuda-q-library-from-c){.reference
            .internal}
        -   [Calling an C++ library from
            CUDA-Q](using/integration/libraries.html#calling-an-c-library-from-cuda-q){.reference
            .internal}
        -   [Interfacing between binaries compiled with a different
            toolchains](using/integration/libraries.html#interfacing-between-binaries-compiled-with-a-different-toolchains){.reference
            .internal}
-   [Extending](using/extending/extending.html){.reference .internal}
    -   [Add a new Hardware
        Backend](using/extending/backend.html){.reference .internal}
        -   [Overview](using/extending/backend.html#overview){.reference
            .internal}
        -   [Server Helper
            Implementation](using/extending/backend.html#server-helper-implementation){.reference
            .internal}
            -   [Directory
                Structure](using/extending/backend.html#directory-structure){.reference
                .internal}
            -   [Server Helper
                Class](using/extending/backend.html#server-helper-class){.reference
                .internal}
            -   [[`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](using/extending/backend.html#cmakelists-txt){.reference
                .internal}
        -   [Target
            Configuration](using/extending/backend.html#target-configuration){.reference
            .internal}
            -   [Update Parent [`CMakeLists.txt`{.docutils .literal
                .notranslate}]{.pre}](using/extending/backend.html#update-parent-cmakelists-txt){.reference
                .internal}
        -   [Testing](using/extending/backend.html#testing){.reference
            .internal}
            -   [Unit
                Tests](using/extending/backend.html#unit-tests){.reference
                .internal}
            -   [Mock
                Server](using/extending/backend.html#mock-server){.reference
                .internal}
            -   [Python
                Tests](using/extending/backend.html#python-tests){.reference
                .internal}
            -   [Integration
                Tests](using/extending/backend.html#integration-tests){.reference
                .internal}
        -   [Documentation](using/extending/backend.html#documentation){.reference
            .internal}
        -   [Example
            Usage](using/extending/backend.html#example-usage){.reference
            .internal}
        -   [Code
            Review](using/extending/backend.html#code-review){.reference
            .internal}
        -   [Maintaining a
            Backend](using/extending/backend.html#maintaining-a-backend){.reference
            .internal}
        -   [Conclusion](using/extending/backend.html#conclusion){.reference
            .internal}
    -   [Create a new NVQIR
        Simulator](using/extending/nvqir_simulator.html){.reference
        .internal}
        -   [[`CircuitSimulator`{.code .docutils .literal
            .notranslate}]{.pre}](using/extending/nvqir_simulator.html#circuitsimulator){.reference
            .internal}
        -   [Let's see this in
            action](using/extending/nvqir_simulator.html#let-s-see-this-in-action){.reference
            .internal}
    -   [Working with CUDA-Q
        IR](using/extending/cudaq_ir.html){.reference .internal}
    -   [Create an MLIR Pass for
        CUDA-Q](using/extending/mlir_pass.html){.reference .internal}
-   [Specifications](specification/index.html){.reference .internal}
    -   [Language Specification](specification/cudaq.html){.reference
        .internal}
        -   [1. Machine
            Model](specification/cudaq/machine_model.html){.reference
            .internal}
        -   [2. Namespace and
            Standard](specification/cudaq/namespace.html){.reference
            .internal}
        -   [3. Quantum
            Types](specification/cudaq/types.html){.reference .internal}
            -   [3.1. [`cudaq::qudit<Levels>`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/types.html#cudaq-qudit-levels){.reference
                .internal}
            -   [3.2. [`cudaq::qubit`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/types.html#cudaq-qubit){.reference
                .internal}
            -   [3.3. Quantum
                Containers](specification/cudaq/types.html#quantum-containers){.reference
                .internal}
        -   [4. Quantum
            Operators](specification/cudaq/operators.html){.reference
            .internal}
            -   [4.1. [`cudaq::spin_op`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/operators.html#cudaq-spin-op){.reference
                .internal}
        -   [5. Quantum
            Operations](specification/cudaq/operations.html){.reference
            .internal}
            -   [5.1. Operations on [`cudaq::qubit`{.code .docutils
                .literal
                .notranslate}]{.pre}](specification/cudaq/operations.html#operations-on-cudaq-qubit){.reference
                .internal}
        -   [6. Quantum
            Kernels](specification/cudaq/kernels.html){.reference
            .internal}
        -   [7. Sub-circuit
            Synthesis](specification/cudaq/synthesis.html){.reference
            .internal}
        -   [8. Control
            Flow](specification/cudaq/control_flow.html){.reference
            .internal}
        -   [9. Just-in-Time Kernel
            Creation](specification/cudaq/dynamic_kernels.html){.reference
            .internal}
        -   [10. Quantum
            Patterns](specification/cudaq/patterns.html){.reference
            .internal}
            -   [10.1.
                Compute-Action-Uncompute](specification/cudaq/patterns.html#compute-action-uncompute){.reference
                .internal}
        -   [11. Platform](specification/cudaq/platform.html){.reference
            .internal}
        -   [12. Algorithmic
            Primitives](specification/cudaq/algorithmic_primitives.html){.reference
            .internal}
            -   [12.1. [`cudaq::sample`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/algorithmic_primitives.html#cudaq-sample){.reference
                .internal}
            -   [12.2. [`cudaq::run`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/algorithmic_primitives.html#cudaq-run){.reference
                .internal}
            -   [12.3. [`cudaq::observe`{.code .docutils .literal
                .notranslate}]{.pre}](specification/cudaq/algorithmic_primitives.html#cudaq-observe){.reference
                .internal}
            -   [12.4. [`cudaq::optimizer`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](specification/cudaq/algorithmic_primitives.html#cudaq-optimizer-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
            -   [12.5. [`cudaq::gradient`{.code .docutils .literal
                .notranslate}]{.pre} (deprecated, functionality moved to
                CUDA-Q
                libraries)](specification/cudaq/algorithmic_primitives.html#cudaq-gradient-deprecated-functionality-moved-to-cuda-q-libraries){.reference
                .internal}
        -   [13. Example
            Programs](specification/cudaq/examples.html){.reference
            .internal}
            -   [13.1. Hello World - Simple Bell
                State](specification/cudaq/examples.html#hello-world-simple-bell-state){.reference
                .internal}
            -   [13.2. GHZ State Preparation and
                Sampling](specification/cudaq/examples.html#ghz-state-preparation-and-sampling){.reference
                .internal}
            -   [13.3. Quantum Phase
                Estimation](specification/cudaq/examples.html#quantum-phase-estimation){.reference
                .internal}
            -   [13.4. Deuteron Binding Energy Parameter
                Sweep](specification/cudaq/examples.html#deuteron-binding-energy-parameter-sweep){.reference
                .internal}
            -   [13.5. Grover's
                Algorithm](specification/cudaq/examples.html#grover-s-algorithm){.reference
                .internal}
            -   [13.6. Iterative Phase
                Estimation](specification/cudaq/examples.html#iterative-phase-estimation){.reference
                .internal}
    -   [Quake
        Specification](specification/quake-dialect.html){.reference
        .internal}
        -   [General
            Introduction](specification/quake-dialect.html#general-introduction){.reference
            .internal}
        -   [Motivation](specification/quake-dialect.html#motivation){.reference
            .internal}
-   [API Reference](api/api.html){.reference .internal}
    -   [C++ API](api/languages/cpp_api.html){.reference .internal}
        -   [Operators](api/languages/cpp_api.html#operators){.reference
            .internal}
        -   [Quantum](api/languages/cpp_api.html#quantum){.reference
            .internal}
        -   [Common](api/languages/cpp_api.html#common){.reference
            .internal}
        -   [Noise
            Modeling](api/languages/cpp_api.html#noise-modeling){.reference
            .internal}
        -   [Kernel
            Builder](api/languages/cpp_api.html#kernel-builder){.reference
            .internal}
        -   [Algorithms](api/languages/cpp_api.html#algorithms){.reference
            .internal}
        -   [Platform](api/languages/cpp_api.html#platform){.reference
            .internal}
        -   [Utilities](api/languages/cpp_api.html#utilities){.reference
            .internal}
        -   [Namespaces](api/languages/cpp_api.html#namespaces){.reference
            .internal}
    -   [Python API](api/languages/python_api.html){.reference
        .internal}
        -   [Program
            Construction](api/languages/python_api.html#program-construction){.reference
            .internal}
            -   [[`make_kernel()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.make_kernel){.reference
                .internal}
            -   [[`PyKernel`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.PyKernel){.reference
                .internal}
            -   [[`Kernel`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Kernel){.reference
                .internal}
            -   [[`PyKernelDecorator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.PyKernelDecorator){.reference
                .internal}
            -   [[`kernel()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.kernel){.reference
                .internal}
        -   [Kernel
            Execution](api/languages/python_api.html#kernel-execution){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.sample){.reference
                .internal}
            -   [[`sample_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.sample_async){.reference
                .internal}
            -   [[`run()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.run){.reference
                .internal}
            -   [[`run_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.run_async){.reference
                .internal}
            -   [[`observe()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.observe){.reference
                .internal}
            -   [[`observe_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.observe_async){.reference
                .internal}
            -   [[`get_state()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_state){.reference
                .internal}
            -   [[`get_state_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_state_async){.reference
                .internal}
            -   [[`vqe()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.vqe){.reference
                .internal}
            -   [[`draw()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.draw){.reference
                .internal}
            -   [[`translate()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.translate){.reference
                .internal}
            -   [[`estimate_resources()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.estimate_resources){.reference
                .internal}
        -   [Backend
            Configuration](api/languages/python_api.html#backend-configuration){.reference
            .internal}
            -   [[`has_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.has_target){.reference
                .internal}
            -   [[`get_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_target){.reference
                .internal}
            -   [[`get_targets()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.get_targets){.reference
                .internal}
            -   [[`set_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.set_target){.reference
                .internal}
            -   [[`reset_target()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.reset_target){.reference
                .internal}
            -   [[`set_noise()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.set_noise){.reference
                .internal}
            -   [[`unset_noise()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.unset_noise){.reference
                .internal}
            -   [[`register_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.register_set_target_callback){.reference
                .internal}
            -   [[`unregister_set_target_callback()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.unregister_set_target_callback){.reference
                .internal}
            -   [[`cudaq.apply_noise()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.cudaq.apply_noise){.reference
                .internal}
            -   [[`initialize_cudaq()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.initialize_cudaq){.reference
                .internal}
            -   [[`num_available_gpus()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.num_available_gpus){.reference
                .internal}
            -   [[`set_random_seed()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.set_random_seed){.reference
                .internal}
        -   [Dynamics](api/languages/python_api.html#dynamics){.reference
            .internal}
            -   [[`evolve()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.evolve){.reference
                .internal}
            -   [[`evolve_async()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.evolve_async){.reference
                .internal}
            -   [[`Schedule`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Schedule){.reference
                .internal}
            -   [[`BaseIntegrator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.dynamics.integrator.BaseIntegrator){.reference
                .internal}
            -   [[`InitialState`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.dynamics.helpers.InitialState){.reference
                .internal}
            -   [[`InitialStateType`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.InitialStateType){.reference
                .internal}
            -   [[`IntermediateResultSave`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.IntermediateResultSave){.reference
                .internal}
        -   [Operators](api/languages/python_api.html#operators){.reference
            .internal}
            -   [[`OperatorSum`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.OperatorSum){.reference
                .internal}
            -   [[`ProductOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.ProductOperator){.reference
                .internal}
            -   [[`ElementaryOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.ElementaryOperator){.reference
                .internal}
            -   [[`ScalarOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.ScalarOperator){.reference
                .internal}
            -   [[`RydbergHamiltonian`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.RydbergHamiltonian){.reference
                .internal}
            -   [[`SuperOperator`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.SuperOperator){.reference
                .internal}
            -   [[`operators.define()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.define){.reference
                .internal}
            -   [[`operators.instantiate()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.operators.instantiate){.reference
                .internal}
            -   [Spin
                Operators](api/languages/python_api.html#spin-operators){.reference
                .internal}
            -   [Fermion
                Operators](api/languages/python_api.html#fermion-operators){.reference
                .internal}
            -   [Boson
                Operators](api/languages/python_api.html#boson-operators){.reference
                .internal}
            -   [General
                Operators](api/languages/python_api.html#general-operators){.reference
                .internal}
        -   [Data
            Types](api/languages/python_api.html#data-types){.reference
            .internal}
            -   [[`SimulationPrecision`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.SimulationPrecision){.reference
                .internal}
            -   [[`Target`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Target){.reference
                .internal}
            -   [[`State`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.State){.reference
                .internal}
            -   [[`Tensor`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Tensor){.reference
                .internal}
            -   [[`QuakeValue`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.QuakeValue){.reference
                .internal}
            -   [[`qubit`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.qubit){.reference
                .internal}
            -   [[`qreg`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.qreg){.reference
                .internal}
            -   [[`qvector`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.qvector){.reference
                .internal}
            -   [[`ComplexMatrix`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.ComplexMatrix){.reference
                .internal}
            -   [[`SampleResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.SampleResult){.reference
                .internal}
            -   [[`AsyncSampleResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncSampleResult){.reference
                .internal}
            -   [[`ObserveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.ObserveResult){.reference
                .internal}
            -   [[`AsyncObserveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncObserveResult){.reference
                .internal}
            -   [[`AsyncStateResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncStateResult){.reference
                .internal}
            -   [[`OptimizationResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.OptimizationResult){.reference
                .internal}
            -   [[`EvolveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.EvolveResult){.reference
                .internal}
            -   [[`AsyncEvolveResult`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.AsyncEvolveResult){.reference
                .internal}
            -   [[`Resources`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.Resources){.reference
                .internal}
            -   [Optimizers](api/languages/python_api.html#optimizers){.reference
                .internal}
            -   [Gradients](api/languages/python_api.html#gradients){.reference
                .internal}
            -   [Noisy
                Simulation](api/languages/python_api.html#noisy-simulation){.reference
                .internal}
        -   [MPI
            Submodule](api/languages/python_api.html#mpi-submodule){.reference
            .internal}
            -   [[`initialize()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.initialize){.reference
                .internal}
            -   [[`rank()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.rank){.reference
                .internal}
            -   [[`num_ranks()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.num_ranks){.reference
                .internal}
            -   [[`all_gather()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.all_gather){.reference
                .internal}
            -   [[`broadcast()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.broadcast){.reference
                .internal}
            -   [[`is_initialized()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.is_initialized){.reference
                .internal}
            -   [[`finalize()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.mpi.finalize){.reference
                .internal}
        -   [ORCA
            Submodule](api/languages/python_api.html#orca-submodule){.reference
            .internal}
            -   [[`sample()`{.docutils .literal
                .notranslate}]{.pre}](api/languages/python_api.html#cudaq.orca.sample){.reference
                .internal}
    -   [Quantum Operations](api/default_ops.html){.reference .internal}
        -   [Unitary Operations on
            Qubits](api/default_ops.html#unitary-operations-on-qubits){.reference
            .internal}
            -   [[`x`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#x){.reference
                .internal}
            -   [[`y`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#y){.reference
                .internal}
            -   [[`z`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#z){.reference
                .internal}
            -   [[`h`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#h){.reference
                .internal}
            -   [[`r1`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#r1){.reference
                .internal}
            -   [[`rx`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#rx){.reference
                .internal}
            -   [[`ry`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#ry){.reference
                .internal}
            -   [[`rz`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#rz){.reference
                .internal}
            -   [[`s`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#s){.reference
                .internal}
            -   [[`t`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#t){.reference
                .internal}
            -   [[`swap`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#swap){.reference
                .internal}
            -   [[`u3`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#u3){.reference
                .internal}
        -   [Adjoint and Controlled
            Operations](api/default_ops.html#adjoint-and-controlled-operations){.reference
            .internal}
        -   [Measurements on
            Qubits](api/default_ops.html#measurements-on-qubits){.reference
            .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#mz){.reference
                .internal}
            -   [[`mx`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#mx){.reference
                .internal}
            -   [[`my`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#my){.reference
                .internal}
        -   [User-Defined Custom
            Operations](api/default_ops.html#user-defined-custom-operations){.reference
            .internal}
        -   [Photonic Operations on
            Qudits](api/default_ops.html#photonic-operations-on-qudits){.reference
            .internal}
            -   [[`create`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#create){.reference
                .internal}
            -   [[`annihilate`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#annihilate){.reference
                .internal}
            -   [[`phase_shift`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#phase-shift){.reference
                .internal}
            -   [[`beam_splitter`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#beam-splitter){.reference
                .internal}
            -   [[`mz`{.code .docutils .literal
                .notranslate}]{.pre}](api/default_ops.html#id1){.reference
                .internal}
-   [Other Versions](versions.html){.reference .internal}
:::
:::

::: {.section .wy-nav-content-wrap toggle="wy-nav-shift"}
[NVIDIA CUDA-Q](index.html)

::: wy-nav-content
::: rst-content
::: {role="navigation" aria-label="Page navigation"}
-   [](index.html){.icon .icon-home aria-label="Home"}
-   Index
-   

------------------------------------------------------------------------
:::

::: {.document role="main" itemscope="itemscope" itemtype="http://schema.org/Article"}
::: {itemprop="articleBody"}
# Index

::: genindex-jumpbox
[**\_**](#_) \| [**A**](#A) \| [**B**](#B) \| [**C**](#C) \| [**D**](#D)
\| [**E**](#E) \| [**F**](#F) \| [**G**](#G) \| [**H**](#H) \|
[**I**](#I) \| [**K**](#K) \| [**L**](#L) \| [**M**](#M) \| [**N**](#N)
\| [**O**](#O) \| [**P**](#P) \| [**Q**](#Q) \| [**R**](#R) \|
[**S**](#S) \| [**T**](#T) \| [**U**](#U) \| [**V**](#V) \| [**X**](#X)
\| [**Y**](#Y) \| [**Z**](#Z)
:::

## \_ {#_}

+-----------------------------------+-----------------------------------+
| -   [\_\_add\_\_()                | -   [\_\_iter\_\_()               |
|     (cudaq.QuakeValue             |     (cudaq.SampleResult           |
|                                   |     m                             |
|   method)](api/languages/python_a | ethod)](api/languages/python_api. |
| pi.html#cudaq.QuakeValue.__add__) | html#cudaq.SampleResult.__iter__) |
| -   [\_\_call\_\_()               | -   [\_\_len\_\_()                |
|     (cudaq.PyKernelDecorator      |     (cudaq.SampleResult           |
|     method                        |                                   |
| )](api/languages/python_api.html# | method)](api/languages/python_api |
| cudaq.PyKernelDecorator.__call__) | .html#cudaq.SampleResult.__len__) |
| -   [\_\_getitem\_\_()            | -   [\_\_mul\_\_()                |
|     (cudaq.ComplexMatrix          |     (cudaq.QuakeValue             |
|     metho                         |                                   |
| d)](api/languages/python_api.html |   method)](api/languages/python_a |
| #cudaq.ComplexMatrix.__getitem__) | pi.html#cudaq.QuakeValue.__mul__) |
|     -   [(cudaq.KrausChannel      | -   [\_\_neg\_\_()                |
|         meth                      |     (cudaq.QuakeValue             |
| od)](api/languages/python_api.htm |                                   |
| l#cudaq.KrausChannel.__getitem__) |   method)](api/languages/python_a |
|     -   [(cudaq.QuakeValue        | pi.html#cudaq.QuakeValue.__neg__) |
|         me                        | -   [\_\_radd\_\_()               |
| thod)](api/languages/python_api.h |     (cudaq.QuakeValue             |
| tml#cudaq.QuakeValue.__getitem__) |                                   |
|     -   [(cudaq.SampleResult      |  method)](api/languages/python_ap |
|         meth                      | i.html#cudaq.QuakeValue.__radd__) |
| od)](api/languages/python_api.htm | -   [\_\_rmul\_\_()               |
| l#cudaq.SampleResult.__getitem__) |     (cudaq.QuakeValue             |
| -   [\_\_init\_\_()               |                                   |
|                                   |  method)](api/languages/python_ap |
|    (cudaq.AmplitudeDampingChannel | i.html#cudaq.QuakeValue.__rmul__) |
|     method)](api                  | -   [\_\_rsub\_\_()               |
| /languages/python_api.html#cudaq. |     (cudaq.QuakeValue             |
| AmplitudeDampingChannel.__init__) |                                   |
|     -   [(cudaq.BitFlipChannel    |  method)](api/languages/python_ap |
|         met                       | i.html#cudaq.QuakeValue.__rsub__) |
| hod)](api/languages/python_api.ht | -   [\_\_str\_\_()                |
| ml#cudaq.BitFlipChannel.__init__) |     (cudaq.ComplexMatrix          |
|                                   |     m                             |
| -   [(cudaq.DepolarizationChannel | ethod)](api/languages/python_api. |
|         method)](a                | html#cudaq.ComplexMatrix.__str__) |
| pi/languages/python_api.html#cuda |     -   [(cudaq.PyKernelDecorator |
| q.DepolarizationChannel.__init__) |         metho                     |
|     -   [(cudaq.NoiseModel        | d)](api/languages/python_api.html |
|                                   | #cudaq.PyKernelDecorator.__str__) |
|  method)](api/languages/python_ap | -   [\_\_sub\_\_()                |
| i.html#cudaq.NoiseModel.__init__) |     (cudaq.QuakeValue             |
|     -   [(c                       |                                   |
| udaq.operators.RydbergHamiltonian |   method)](api/languages/python_a |
|         method)](api/lang         | pi.html#cudaq.QuakeValue.__sub__) |
| uages/python_api.html#cudaq.opera |                                   |
| tors.RydbergHamiltonian.__init__) |                                   |
|     -   [(cudaq.PhaseFlipChannel  |                                   |
|         metho                     |                                   |
| d)](api/languages/python_api.html |                                   |
| #cudaq.PhaseFlipChannel.__init__) |                                   |
+-----------------------------------+-----------------------------------+

## A {#A}

+-----------------------------------+-----------------------------------+
| -   [add_all_qubit_channel()      | -   [append() (cudaq.KrausChannel |
|     (cudaq.NoiseModel             |                                   |
|     method)](api                  |  method)](api/languages/python_ap |
| /languages/python_api.html#cudaq. | i.html#cudaq.KrausChannel.append) |
| NoiseModel.add_all_qubit_channel) | -   [argument_count               |
| -   [add_channel()                |     (cudaq.PyKernel               |
|     (cudaq.NoiseModel             |     attrib                        |
|     me                            | ute)](api/languages/python_api.ht |
| thod)](api/languages/python_api.h | ml#cudaq.PyKernel.argument_count) |
| tml#cudaq.NoiseModel.add_channel) | -   [arguments (cudaq.PyKernel    |
| -   [all_gather() (in module      |     a                             |
|                                   | ttribute)](api/languages/python_a |
|    cudaq.mpi)](api/languages/pyth | pi.html#cudaq.PyKernel.arguments) |
| on_api.html#cudaq.mpi.all_gather) | -   [as_pauli()                   |
| -   [amplitude() (cudaq.State     |     (cudaq.o                      |
|     method)](api/languages/pytho  | perators.spin.SpinOperatorElement |
| n_api.html#cudaq.State.amplitude) |     method)](api/languages/       |
| -   [AmplitudeDampingChannel      | python_api.html#cudaq.operators.s |
|     (class in                     | pin.SpinOperatorElement.as_pauli) |
|     cu                            | -   [AsyncEvolveResult (class in  |
| daq)](api/languages/python_api.ht |     cudaq)](api/languages/python_ |
| ml#cudaq.AmplitudeDampingChannel) | api.html#cudaq.AsyncEvolveResult) |
| -   [amplitudes() (cudaq.State    | -   [AsyncObserveResult (class in |
|     method)](api/languages/python |                                   |
| _api.html#cudaq.State.amplitudes) |    cudaq)](api/languages/python_a |
| -   [annihilate() (in module      | pi.html#cudaq.AsyncObserveResult) |
|     c                             | -   [AsyncSampleResult (class in  |
| udaq.boson)](api/languages/python |     cudaq)](api/languages/python_ |
| _api.html#cudaq.boson.annihilate) | api.html#cudaq.AsyncSampleResult) |
|     -   [(in module               | -   [AsyncStateResult (class in   |
|         cudaq                     |     cudaq)](api/languages/python  |
| .fermion)](api/languages/python_a | _api.html#cudaq.AsyncStateResult) |
| pi.html#cudaq.fermion.annihilate) |                                   |
+-----------------------------------+-----------------------------------+

## B {#B}

+-----------------------------------+-----------------------------------+
| -   [BaseIntegrator (class in     | -   [BosonOperator (class in      |
|                                   |     cudaq.operators.boson)](      |
| cudaq.dynamics.integrator)](api/l | api/languages/python_api.html#cud |
| anguages/python_api.html#cudaq.dy | aq.operators.boson.BosonOperator) |
| namics.integrator.BaseIntegrator) | -   [BosonOperatorElement (class  |
| -   [beta_reduction()             |     in                            |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](api                  |   cudaq.operators.boson)](api/lan |
| /languages/python_api.html#cudaq. | guages/python_api.html#cudaq.oper |
| PyKernelDecorator.beta_reduction) | ators.boson.BosonOperatorElement) |
| -   [BitFlipChannel (class in     | -   [BosonOperatorTerm (class in  |
|     cudaq)](api/languages/pyth    |     cudaq.operators.boson)](api/  |
| on_api.html#cudaq.BitFlipChannel) | languages/python_api.html#cudaq.o |
|                                   | perators.boson.BosonOperatorTerm) |
|                                   | -   [broadcast() (in module       |
|                                   |     cudaq.mpi)](api/languages/pyt |
|                                   | hon_api.html#cudaq.mpi.broadcast) |
+-----------------------------------+-----------------------------------+

## C {#C}

+-----------------------------------+-----------------------------------+
| -   [canonicalize()               | -   [cudaq::pauli1::pauli1 (C++   |
|     (cu                           |     function)](api/languages/cpp_ |
| daq.operators.boson.BosonOperator | api.html#_CPPv4N5cudaq6pauli16pau |
|     method)](api/languages        | li1ERKNSt6vectorIN5cudaq4realEEE) |
| /python_api.html#cudaq.operators. | -   [cudaq::pauli2 (C++           |
| boson.BosonOperator.canonicalize) |     class)](api/languages/cp      |
|     -   [(cudaq.                  | p_api.html#_CPPv4N5cudaq6pauli2E) |
| operators.boson.BosonOperatorTerm | -                                 |
|                                   |    [cudaq::pauli2::num_parameters |
|        method)](api/languages/pyt |     (C++                          |
| hon_api.html#cudaq.operators.boso |     member)]                      |
| n.BosonOperatorTerm.canonicalize) | (api/languages/cpp_api.html#_CPPv |
|     -   [(cudaq.                  | 4N5cudaq6pauli214num_parametersE) |
| operators.fermion.FermionOperator | -   [cudaq::pauli2::num_targets   |
|                                   |     (C++                          |
|        method)](api/languages/pyt |     membe                         |
| hon_api.html#cudaq.operators.ferm | r)](api/languages/cpp_api.html#_C |
| ion.FermionOperator.canonicalize) | PPv4N5cudaq6pauli211num_targetsE) |
|     -   [(cudaq.oper              | -   [cudaq::pauli2::pauli2 (C++   |
| ators.fermion.FermionOperatorTerm |     function)](api/languages/cpp_ |
|                                   | api.html#_CPPv4N5cudaq6pauli26pau |
|    method)](api/languages/python_ | li2ERKNSt6vectorIN5cudaq4realEEE) |
| api.html#cudaq.operators.fermion. | -   [cudaq::phase_damping (C++    |
| FermionOperatorTerm.canonicalize) |                                   |
|     -                             |  class)](api/languages/cpp_api.ht |
|  [(cudaq.operators.MatrixOperator | ml#_CPPv4N5cudaq13phase_dampingE) |
|         method)](api/lang         | -   [cud                          |
| uages/python_api.html#cudaq.opera | aq::phase_damping::num_parameters |
| tors.MatrixOperator.canonicalize) |     (C++                          |
|     -   [(c                       |     member)](api/lan              |
| udaq.operators.MatrixOperatorTerm | guages/cpp_api.html#_CPPv4N5cudaq |
|         method)](api/language     | 13phase_damping14num_parametersE) |
| s/python_api.html#cudaq.operators | -   [                             |
| .MatrixOperatorTerm.canonicalize) | cudaq::phase_damping::num_targets |
|     -   [(                        |     (C++                          |
| cudaq.operators.spin.SpinOperator |     member)](api/                 |
|         method)](api/languag      | languages/cpp_api.html#_CPPv4N5cu |
| es/python_api.html#cudaq.operator | daq13phase_damping11num_targetsE) |
| s.spin.SpinOperator.canonicalize) | -   [cudaq::phase_flip_channel    |
|     -   [(cuda                    |     (C++                          |
| q.operators.spin.SpinOperatorTerm |     clas                          |
|         method)](api/languages/p  | s)](api/languages/cpp_api.html#_C |
| ython_api.html#cudaq.operators.sp | PPv4N5cudaq18phase_flip_channelE) |
| in.SpinOperatorTerm.canonicalize) | -   [cudaq::p                     |
| -   [canonicalized() (in module   | hase_flip_channel::num_parameters |
|     cuda                          |     (C++                          |
| q.boson)](api/languages/python_ap |     member)](api/language         |
| i.html#cudaq.boson.canonicalized) | s/cpp_api.html#_CPPv4N5cudaq18pha |
|     -   [(in module               | se_flip_channel14num_parametersE) |
|         cudaq.fe                  | -   [cudaq                        |
| rmion)](api/languages/python_api. | ::phase_flip_channel::num_targets |
| html#cudaq.fermion.canonicalized) |     (C++                          |
|     -   [(in module               |     member)](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq18 |
|        cudaq.operators.custom)](a | phase_flip_channel11num_targetsE) |
| pi/languages/python_api.html#cuda | -   [cudaq::product_op (C++       |
| q.operators.custom.canonicalized) |                                   |
|     -   [(in module               |  class)](api/languages/cpp_api.ht |
|         cu                        | ml#_CPPv4I0EN5cudaq10product_opE) |
| daq.spin)](api/languages/python_a | -   [cudaq::product_op::begin     |
| pi.html#cudaq.spin.canonicalized) |     (C++                          |
| -   [CentralDifference (class in  |     functio                       |
|     cudaq.gradients)              | n)](api/languages/cpp_api.html#_C |
| ](api/languages/python_api.html#c | PPv4NK5cudaq10product_op5beginEv) |
| udaq.gradients.CentralDifference) | -                                 |
| -   [clear() (cudaq.Resources     |  [cudaq::product_op::canonicalize |
|     method)](api/languages/pytho  |     (C++                          |
| n_api.html#cudaq.Resources.clear) |     func                          |
|     -   [(cudaq.SampleResult      | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq10product_op12canon |
|   method)](api/languages/python_a | icalizeERKNSt3setINSt6size_tEEE), |
| pi.html#cudaq.SampleResult.clear) |     [\[1\]](api                   |
| -   [COBYLA (class in             | /languages/cpp_api.html#_CPPv4N5c |
|     cudaq.o                       | udaq10product_op12canonicalizeEv) |
| ptimizers)](api/languages/python_ | -   [                             |
| api.html#cudaq.optimizers.COBYLA) | cudaq::product_op::const_iterator |
| -   [coefficient                  |     (C++                          |
|     (cudaq.                       |     struct)](api/                 |
| operators.boson.BosonOperatorTerm | languages/cpp_api.html#_CPPv4N5cu |
|     property)](api/languages/py   | daq10product_op14const_iteratorE) |
| thon_api.html#cudaq.operators.bos | -   [cudaq::product_o             |
| on.BosonOperatorTerm.coefficient) | p::const_iterator::const_iterator |
|     -   [(cudaq.oper              |     (C++                          |
| ators.fermion.FermionOperatorTerm |     fu                            |
|                                   | nction)](api/languages/cpp_api.ht |
|   property)](api/languages/python | ml#_CPPv4N5cudaq10product_op14con |
| _api.html#cudaq.operators.fermion | st_iterator14const_iteratorEPK10p |
| .FermionOperatorTerm.coefficient) | roduct_opI9HandlerTyENSt6size_tE) |
|     -   [(c                       | -   [cudaq::produ                 |
| udaq.operators.MatrixOperatorTerm | ct_op::const_iterator::operator!= |
|         property)](api/languag    |     (C++                          |
| es/python_api.html#cudaq.operator |     fun                           |
| s.MatrixOperatorTerm.coefficient) | ction)](api/languages/cpp_api.htm |
|     -   [(cuda                    | l#_CPPv4NK5cudaq10product_op14con |
| q.operators.spin.SpinOperatorTerm | st_iteratorneERK14const_iterator) |
|         property)](api/languages/ | -   [cudaq::produ                 |
| python_api.html#cudaq.operators.s | ct_op::const_iterator::operator\* |
| pin.SpinOperatorTerm.coefficient) |     (C++                          |
| -   [col_count                    |     function)](api/lang           |
|     (cudaq.KrausOperator          | uages/cpp_api.html#_CPPv4NK5cudaq |
|     prope                         | 10product_op14const_iteratormlEv) |
| rty)](api/languages/python_api.ht | -   [cudaq::produ                 |
| ml#cudaq.KrausOperator.col_count) | ct_op::const_iterator::operator++ |
| -   [ComplexMatrix (class in      |     (C++                          |
|     cudaq)](api/languages/pyt     |     function)](api/lang           |
| hon_api.html#cudaq.ComplexMatrix) | uages/cpp_api.html#_CPPv4N5cudaq1 |
| -   [compute()                    | 0product_op14const_iteratorppEi), |
|     (                             |     [\[1\]](api/lan               |
| cudaq.gradients.CentralDifference | guages/cpp_api.html#_CPPv4N5cudaq |
|     method)](api/la               | 10product_op14const_iteratorppEv) |
| nguages/python_api.html#cudaq.gra | -   [cudaq::produc                |
| dients.CentralDifference.compute) | t_op::const_iterator::operator\-- |
|     -   [(                        |     (C++                          |
| cudaq.gradients.ForwardDifference |     function)](api/lang           |
|         method)](api/la           | uages/cpp_api.html#_CPPv4N5cudaq1 |
| nguages/python_api.html#cudaq.gra | 0product_op14const_iteratormmEi), |
| dients.ForwardDifference.compute) |     [\[1\]](api/lan               |
|     -                             | guages/cpp_api.html#_CPPv4N5cudaq |
|  [(cudaq.gradients.ParameterShift | 10product_op14const_iteratormmEv) |
|         method)](api              | -   [cudaq::produc                |
| /languages/python_api.html#cudaq. | t_op::const_iterator::operator-\> |
| gradients.ParameterShift.compute) |     (C++                          |
| -   [const()                      |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|   (cudaq.operators.ScalarOperator | 10product_op14const_iteratorptEv) |
|     class                         | -   [cudaq::produ                 |
|     method)](a                    | ct_op::const_iterator::operator== |
| pi/languages/python_api.html#cuda |     (C++                          |
| q.operators.ScalarOperator.const) |     fun                           |
| -   [copy()                       | ction)](api/languages/cpp_api.htm |
|     (cu                           | l#_CPPv4NK5cudaq10product_op14con |
| daq.operators.boson.BosonOperator | st_iteratoreqERK14const_iterator) |
|     method)](api/l                | -   [cudaq::product_op::degrees   |
| anguages/python_api.html#cudaq.op |     (C++                          |
| erators.boson.BosonOperator.copy) |     function)                     |
|     -   [(cudaq.                  | ](api/languages/cpp_api.html#_CPP |
| operators.boson.BosonOperatorTerm | v4NK5cudaq10product_op7degreesEv) |
|         method)](api/langu        | -   [cudaq::product_op::dump (C++ |
| ages/python_api.html#cudaq.operat |     functi                        |
| ors.boson.BosonOperatorTerm.copy) | on)](api/languages/cpp_api.html#_ |
|     -   [(cudaq.                  | CPPv4NK5cudaq10product_op4dumpEv) |
| operators.fermion.FermionOperator | -   [cudaq::product_op::end (C++  |
|         method)](api/langu        |     funct                         |
| ages/python_api.html#cudaq.operat | ion)](api/languages/cpp_api.html# |
| ors.fermion.FermionOperator.copy) | _CPPv4NK5cudaq10product_op3endEv) |
|     -   [(cudaq.oper              | -   [c                            |
| ators.fermion.FermionOperatorTerm | udaq::product_op::get_coefficient |
|         method)](api/languages    |     (C++                          |
| /python_api.html#cudaq.operators. |     function)](api/lan            |
| fermion.FermionOperatorTerm.copy) | guages/cpp_api.html#_CPPv4NK5cuda |
|     -                             | q10product_op15get_coefficientEv) |
|  [(cudaq.operators.MatrixOperator | -                                 |
|         method)](                 |   [cudaq::product_op::get_term_id |
| api/languages/python_api.html#cud |     (C++                          |
| aq.operators.MatrixOperator.copy) |     function)](api                |
|     -   [(c                       | /languages/cpp_api.html#_CPPv4NK5 |
| udaq.operators.MatrixOperatorTerm | cudaq10product_op11get_term_idEv) |
|         method)](api/             | -                                 |
| languages/python_api.html#cudaq.o |   [cudaq::product_op::is_identity |
| perators.MatrixOperatorTerm.copy) |     (C++                          |
|     -   [(                        |     function)](api                |
| cudaq.operators.spin.SpinOperator | /languages/cpp_api.html#_CPPv4NK5 |
|         method)](api              | cudaq10product_op11is_identityEv) |
| /languages/python_api.html#cudaq. | -   [cudaq::product_op::num_ops   |
| operators.spin.SpinOperator.copy) |     (C++                          |
|     -   [(cuda                    |     function)                     |
| q.operators.spin.SpinOperatorTerm | ](api/languages/cpp_api.html#_CPP |
|         method)](api/lan          | v4NK5cudaq10product_op7num_opsEv) |
| guages/python_api.html#cudaq.oper | -                                 |
| ators.spin.SpinOperatorTerm.copy) |    [cudaq::product_op::operator\* |
| -   [count() (cudaq.Resources     |     (C++                          |
|     method)](api/languages/pytho  |     function)](api/languages/     |
| n_api.html#cudaq.Resources.count) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
|     -   [(cudaq.SampleResult      | oduct_opmlE10product_opI1TERK15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|   method)](api/languages/python_a |     [\[1\]](api/languages/        |
| pi.html#cudaq.SampleResult.count) | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| -   [count_controls()             | oduct_opmlE10product_opI1TERK15sc |
|     (cudaq.Resources              | alar_operatorRR10product_opI1TE), |
|     meth                          |     [\[2\]](api/languages/        |
| od)](api/languages/python_api.htm | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| l#cudaq.Resources.count_controls) | oduct_opmlE10product_opI1TERR15sc |
| -   [counts()                     | alar_operatorRK10product_opI1TE), |
|     (cudaq.ObserveResult          |     [\[3\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4I0EN5cudaq10pr |
| method)](api/languages/python_api | oduct_opmlE10product_opI1TERR15sc |
| .html#cudaq.ObserveResult.counts) | alar_operatorRR10product_opI1TE), |
| -   [create() (in module          |     [\[4\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|    cudaq.boson)](api/languages/py | 5cudaq10product_opmlE6sum_opI1TER |
| thon_api.html#cudaq.boson.create) | K15scalar_operatorRK6sum_opI1TE), |
|     -   [(in module               |     [\[5\]](api/                  |
|         c                         | languages/cpp_api.html#_CPPv4I0EN |
| udaq.fermion)](api/languages/pyth | 5cudaq10product_opmlE6sum_opI1TER |
| on_api.html#cudaq.fermion.create) | K15scalar_operatorRR6sum_opI1TE), |
| -   [csr_spmatrix (C++            |     [\[6\]](api/                  |
|     type)](api/languages/c        | languages/cpp_api.html#_CPPv4I0EN |
| pp_api.html#_CPPv412csr_spmatrix) | 5cudaq10product_opmlE6sum_opI1TER |
| -   cudaq                         | R15scalar_operatorRK6sum_opI1TE), |
|     -   [module](api/langua       |     [\[7\]](api/                  |
| ges/python_api.html#module-cudaq) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq (C++                   | 5cudaq10product_opmlE6sum_opI1TER |
|     type)](api/lan                | R15scalar_operatorRR6sum_opI1TE), |
| guages/cpp_api.html#_CPPv45cudaq) |     [\[8\]](api/languages         |
| -   [cudaq.apply_noise() (in      | /cpp_api.html#_CPPv4NK5cudaq10pro |
|     module                        | duct_opmlERK6sum_opI9HandlerTyE), |
|     cudaq)](api/languages/python_ |     [\[9\]](api/languages/cpp_a   |
| api.html#cudaq.cudaq.apply_noise) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   cudaq.boson                   | opmlERK10product_opI9HandlerTyE), |
|     -   [module](api/languages/py |     [\[10\]](api/language         |
| thon_api.html#module-cudaq.boson) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -   cudaq.fermion                 | roduct_opmlERK15scalar_operator), |
|                                   |     [\[11\]](api/languages/cpp_a  |
|   -   [module](api/languages/pyth | pi.html#_CPPv4NKR5cudaq10product_ |
| on_api.html#module-cudaq.fermion) | opmlERR10product_opI9HandlerTyE), |
| -   cudaq.operators.custom        |     [\[12\]](api/language         |
|     -   [mo                       | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| dule](api/languages/python_api.ht | roduct_opmlERR15scalar_operator), |
| ml#module-cudaq.operators.custom) |     [\[13\]](api/languages/cpp_   |
| -   cudaq.spin                    | api.html#_CPPv4NO5cudaq10product_ |
|     -   [module](api/languages/p  | opmlERK10product_opI9HandlerTyE), |
| ython_api.html#module-cudaq.spin) |     [\[14\]](api/languag          |
| -   [cudaq::amplitude_damping     | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opmlERK15scalar_operator), |
|     cla                           |     [\[15\]](api/languages/cpp_   |
| ss)](api/languages/cpp_api.html#_ | api.html#_CPPv4NO5cudaq10product_ |
| CPPv4N5cudaq17amplitude_dampingE) | opmlERR10product_opI9HandlerTyE), |
| -                                 |     [\[16\]](api/langua           |
| [cudaq::amplitude_damping_channel | ges/cpp_api.html#_CPPv4NO5cudaq10 |
|     (C++                          | product_opmlERR15scalar_operator) |
|     class)](api                   | -                                 |
| /languages/cpp_api.html#_CPPv4N5c |   [cudaq::product_op::operator\*= |
| udaq25amplitude_damping_channelE) |     (C++                          |
| -   [cudaq::amplitud              |     function)](api/languages/cpp  |
| e_damping_channel::num_parameters | _api.html#_CPPv4N5cudaq10product_ |
|     (C++                          | opmLERK10product_opI9HandlerTyE), |
|     member)](api/languages/cpp_a  |     [\[1\]](api/langua            |
| pi.html#_CPPv4N5cudaq25amplitude_ | ges/cpp_api.html#_CPPv4N5cudaq10p |
| damping_channel14num_parametersE) | roduct_opmLERK15scalar_operator), |
| -   [cudaq::ampli                 |     [\[2\]](api/languages/cp      |
| tude_damping_channel::num_targets | p_api.html#_CPPv4N5cudaq10product |
|     (C++                          | _opmLERR10product_opI9HandlerTyE) |
|     member)](api/languages/cp     | -   [cudaq::product_op::operator+ |
| p_api.html#_CPPv4N5cudaq25amplitu |     (C++                          |
| de_damping_channel11num_targetsE) |     function)](api/langu          |
| -   [cudaq::AnalogRemoteRESTQPU   | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opplE6sum_opI1TERK15sc |
|     class                         | alar_operatorRK10product_opI1TE), |
| )](api/languages/cpp_api.html#_CP |     [\[1\]](api/                  |
| Pv4N5cudaq19AnalogRemoteRESTQPUE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::apply_noise (C++      | 5cudaq10product_opplE6sum_opI1TER |
|     function)](api/               | K15scalar_operatorRK6sum_opI1TE), |
| languages/cpp_api.html#_CPPv4I0Dp |     [\[2\]](api/langu             |
| EN5cudaq11apply_noiseEvDpRR4Args) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -   [cudaq::async_result (C++     | q10product_opplE6sum_opI1TERK15sc |
|     c                             | alar_operatorRR10product_opI1TE), |
| lass)](api/languages/cpp_api.html |     [\[3\]](api/                  |
| #_CPPv4I0EN5cudaq12async_resultE) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::async_result::get     | 5cudaq10product_opplE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRR6sum_opI1TE), |
|     functi                        |     [\[4\]](api/langu             |
| on)](api/languages/cpp_api.html#_ | ages/cpp_api.html#_CPPv4I0EN5cuda |
| CPPv4N5cudaq12async_result3getEv) | q10product_opplE6sum_opI1TERR15sc |
| -   [cudaq::async_sample_result   | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[5\]](api/                  |
|     type                          | languages/cpp_api.html#_CPPv4I0EN |
| )](api/languages/cpp_api.html#_CP | 5cudaq10product_opplE6sum_opI1TER |
| Pv4N5cudaq19async_sample_resultE) | R15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::BaseRemoteRESTQPU     |     [\[6\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     cla                           | q10product_opplE6sum_opI1TERR15sc |
| ss)](api/languages/cpp_api.html#_ | alar_operatorRR10product_opI1TE), |
| CPPv4N5cudaq17BaseRemoteRESTQPUE) |     [\[7\]](api/                  |
| -                                 | languages/cpp_api.html#_CPPv4I0EN |
|    [cudaq::BaseRemoteSimulatorQPU | 5cudaq10product_opplE6sum_opI1TER |
|     (C++                          | R15scalar_operatorRR6sum_opI1TE), |
|     class)](                      |     [\[8\]](api/languages/cpp_a   |
| api/languages/cpp_api.html#_CPPv4 | pi.html#_CPPv4NKR5cudaq10product_ |
| N5cudaq22BaseRemoteSimulatorQPUE) | opplERK10product_opI9HandlerTyE), |
| -   [cudaq::bit_flip_channel (C++ |     [\[9\]](api/language          |
|     cl                            | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| ass)](api/languages/cpp_api.html# | roduct_opplERK15scalar_operator), |
| _CPPv4N5cudaq16bit_flip_channelE) |     [\[10\]](api/languages/       |
| -   [cudaq:                       | cpp_api.html#_CPPv4NKR5cudaq10pro |
| :bit_flip_channel::num_parameters | duct_opplERK6sum_opI9HandlerTyE), |
|     (C++                          |     [\[11\]](api/languages/cpp_a  |
|     member)](api/langua           | pi.html#_CPPv4NKR5cudaq10product_ |
| ges/cpp_api.html#_CPPv4N5cudaq16b | opplERR10product_opI9HandlerTyE), |
| it_flip_channel14num_parametersE) |     [\[12\]](api/language         |
| -   [cud                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| aq::bit_flip_channel::num_targets | roduct_opplERR15scalar_operator), |
|     (C++                          |     [\[13\]](api/languages/       |
|     member)](api/lan              | cpp_api.html#_CPPv4NKR5cudaq10pro |
| guages/cpp_api.html#_CPPv4N5cudaq | duct_opplERR6sum_opI9HandlerTyE), |
| 16bit_flip_channel11num_targetsE) |     [\[                           |
| -   [cudaq::boson_handler (C++    | 14\]](api/languages/cpp_api.html# |
|                                   | _CPPv4NKR5cudaq10product_opplEv), |
|  class)](api/languages/cpp_api.ht |     [\[15\]](api/languages/cpp_   |
| ml#_CPPv4N5cudaq13boson_handlerE) | api.html#_CPPv4NO5cudaq10product_ |
| -   [cudaq::boson_op (C++         | opplERK10product_opI9HandlerTyE), |
|     type)](api/languages/cpp_     |     [\[16\]](api/languag          |
| api.html#_CPPv4N5cudaq8boson_opE) | es/cpp_api.html#_CPPv4NO5cudaq10p |
| -   [cudaq::boson_op_term (C++    | roduct_opplERK15scalar_operator), |
|                                   |     [\[17\]](api/languages        |
|   type)](api/languages/cpp_api.ht | /cpp_api.html#_CPPv4NO5cudaq10pro |
| ml#_CPPv4N5cudaq13boson_op_termE) | duct_opplERK6sum_opI9HandlerTyE), |
| -   [cudaq::CodeGenConfig (C++    |     [\[18\]](api/languages/cpp_   |
|                                   | api.html#_CPPv4NO5cudaq10product_ |
| struct)](api/languages/cpp_api.ht | opplERR10product_opI9HandlerTyE), |
| ml#_CPPv4N5cudaq13CodeGenConfigE) |     [\[19\]](api/languag          |
| -   [cudaq::commutation_relations | es/cpp_api.html#_CPPv4NO5cudaq10p |
|     (C++                          | roduct_opplERR15scalar_operator), |
|     struct)]                      |     [\[20\]](api/languages        |
| (api/languages/cpp_api.html#_CPPv | /cpp_api.html#_CPPv4NO5cudaq10pro |
| 4N5cudaq21commutation_relationsE) | duct_opplERR6sum_opI9HandlerTyE), |
| -   [cudaq::complex (C++          |     [                             |
|     type)](api/languages/cpp      | \[21\]](api/languages/cpp_api.htm |
| _api.html#_CPPv4N5cudaq7complexE) | l#_CPPv4NO5cudaq10product_opplEv) |
| -   [cudaq::complex_matrix (C++   | -   [cudaq::product_op::operator- |
|                                   |     (C++                          |
| class)](api/languages/cpp_api.htm |     function)](api/langu          |
| l#_CPPv4N5cudaq14complex_matrixE) | ages/cpp_api.html#_CPPv4I0EN5cuda |
| -                                 | q10product_opmiE6sum_opI1TERK15sc |
|   [cudaq::complex_matrix::adjoint | alar_operatorRK10product_opI1TE), |
|     (C++                          |     [\[1\]](api/                  |
|     function)](a                  | languages/cpp_api.html#_CPPv4I0EN |
| pi/languages/cpp_api.html#_CPPv4N | 5cudaq10product_opmiE6sum_opI1TER |
| 5cudaq14complex_matrix7adjointEv) | K15scalar_operatorRK6sum_opI1TE), |
| -   [cudaq::                      |     [\[2\]](api/langu             |
| complex_matrix::diagonal_elements | ages/cpp_api.html#_CPPv4I0EN5cuda |
|     (C++                          | q10product_opmiE6sum_opI1TERK15sc |
|     function)](api/languages      | alar_operatorRR10product_opI1TE), |
| /cpp_api.html#_CPPv4NK5cudaq14com |     [\[3\]](api/                  |
| plex_matrix17diagonal_elementsEi) | languages/cpp_api.html#_CPPv4I0EN |
| -   [cudaq::complex_matrix::dump  | 5cudaq10product_opmiE6sum_opI1TER |
|     (C++                          | K15scalar_operatorRR6sum_opI1TE), |
|     function)](api/language       |     [\[4\]](api/langu             |
| s/cpp_api.html#_CPPv4NK5cudaq14co | ages/cpp_api.html#_CPPv4I0EN5cuda |
| mplex_matrix4dumpERNSt7ostreamE), | q10product_opmiE6sum_opI1TERR15sc |
|     [\[1\]]                       | alar_operatorRK10product_opI1TE), |
| (api/languages/cpp_api.html#_CPPv |     [\[5\]](api/                  |
| 4NK5cudaq14complex_matrix4dumpEv) | languages/cpp_api.html#_CPPv4I0EN |
| -   [c                            | 5cudaq10product_opmiE6sum_opI1TER |
| udaq::complex_matrix::eigenvalues | R15scalar_operatorRK6sum_opI1TE), |
|     (C++                          |     [\[6\]](api/langu             |
|     function)](api/lan            | ages/cpp_api.html#_CPPv4I0EN5cuda |
| guages/cpp_api.html#_CPPv4NK5cuda | q10product_opmiE6sum_opI1TERR15sc |
| q14complex_matrix11eigenvaluesEv) | alar_operatorRR10product_opI1TE), |
| -   [cu                           |     [\[7\]](api/                  |
| daq::complex_matrix::eigenvectors | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq10product_opmiE6sum_opI1TER |
|     function)](api/lang           | R15scalar_operatorRR6sum_opI1TE), |
| uages/cpp_api.html#_CPPv4NK5cudaq |     [\[8\]](api/languages/cpp_a   |
| 14complex_matrix12eigenvectorsEv) | pi.html#_CPPv4NKR5cudaq10product_ |
| -   [c                            | opmiERK10product_opI9HandlerTyE), |
| udaq::complex_matrix::exponential |     [\[9\]](api/language          |
|     (C++                          | s/cpp_api.html#_CPPv4NKR5cudaq10p |
|     function)](api/la             | roduct_opmiERK15scalar_operator), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\[10\]](api/languages/       |
| q14complex_matrix11exponentialEv) | cpp_api.html#_CPPv4NKR5cudaq10pro |
| -                                 | duct_opmiERK6sum_opI9HandlerTyE), |
|  [cudaq::complex_matrix::identity |     [\[11\]](api/languages/cpp_a  |
|     (C++                          | pi.html#_CPPv4NKR5cudaq10product_ |
|     function)](api/languages      | opmiERR10product_opI9HandlerTyE), |
| /cpp_api.html#_CPPv4N5cudaq14comp |     [\[12\]](api/language         |
| lex_matrix8identityEKNSt6size_tE) | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| -                                 | roduct_opmiERR15scalar_operator), |
| [cudaq::complex_matrix::kronecker |     [\[13\]](api/languages/       |
|     (C++                          | cpp_api.html#_CPPv4NKR5cudaq10pro |
|     function)](api/lang           | duct_opmiERR6sum_opI9HandlerTyE), |
| uages/cpp_api.html#_CPPv4I00EN5cu |     [\[                           |
| daq14complex_matrix9kroneckerE14c | 14\]](api/languages/cpp_api.html# |
| omplex_matrix8Iterable8Iterable), | _CPPv4NKR5cudaq10product_opmiEv), |
|     [\[1\]](api/l                 |     [\[15\]](api/languages/cpp_   |
| anguages/cpp_api.html#_CPPv4N5cud | api.html#_CPPv4NO5cudaq10product_ |
| aq14complex_matrix9kroneckerERK14 | opmiERK10product_opI9HandlerTyE), |
| complex_matrixRK14complex_matrix) |     [\[16\]](api/languag          |
| -   [cudaq::c                     | es/cpp_api.html#_CPPv4NO5cudaq10p |
| omplex_matrix::minimal_eigenvalue | roduct_opmiERK15scalar_operator), |
|     (C++                          |     [\[17\]](api/languages        |
|     function)](api/languages/     | /cpp_api.html#_CPPv4NO5cudaq10pro |
| cpp_api.html#_CPPv4NK5cudaq14comp | duct_opmiERK6sum_opI9HandlerTyE), |
| lex_matrix18minimal_eigenvalueEv) |     [\[18\]](api/languages/cpp_   |
| -   [                             | api.html#_CPPv4NO5cudaq10product_ |
| cudaq::complex_matrix::operator() | opmiERR10product_opI9HandlerTyE), |
|     (C++                          |     [\[19\]](api/languag          |
|     function)](api/languages/cpp  | es/cpp_api.html#_CPPv4NO5cudaq10p |
| _api.html#_CPPv4N5cudaq14complex_ | roduct_opmiERR15scalar_operator), |
| matrixclENSt6size_tENSt6size_tE), |     [\[20\]](api/languages        |
|     [\[1\]](api/languages/cpp     | /cpp_api.html#_CPPv4NO5cudaq10pro |
| _api.html#_CPPv4NK5cudaq14complex | duct_opmiERR6sum_opI9HandlerTyE), |
| _matrixclENSt6size_tENSt6size_tE) |     [                             |
| -   [                             | \[21\]](api/languages/cpp_api.htm |
| cudaq::complex_matrix::operator\* | l#_CPPv4NO5cudaq10product_opmiEv) |
|     (C++                          | -   [cudaq::product_op::operator/ |
|     function)](api/langua         |     (C++                          |
| ges/cpp_api.html#_CPPv4N5cudaq14c |     function)](api/language       |
| omplex_matrixmlEN14complex_matrix | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| 10value_typeERK14complex_matrix), | roduct_opdvERK15scalar_operator), |
|     [\[1\]                        |     [\[1\]](api/language          |
| ](api/languages/cpp_api.html#_CPP | s/cpp_api.html#_CPPv4NKR5cudaq10p |
| v4N5cudaq14complex_matrixmlERK14c | roduct_opdvERR15scalar_operator), |
| omplex_matrixRK14complex_matrix), |     [\[2\]](api/languag           |
|                                   | es/cpp_api.html#_CPPv4NO5cudaq10p |
|  [\[2\]](api/languages/cpp_api.ht | roduct_opdvERK15scalar_operator), |
| ml#_CPPv4N5cudaq14complex_matrixm |     [\[3\]](api/langua            |
| lERK14complex_matrixRKNSt6vectorI | ges/cpp_api.html#_CPPv4NO5cudaq10 |
| N14complex_matrix10value_typeEEE) | product_opdvERR15scalar_operator) |
| -                                 | -                                 |
| [cudaq::complex_matrix::operator+ |    [cudaq::product_op::operator/= |
|     (C++                          |     (C++                          |
|     function                      |     function)](api/langu          |
| )](api/languages/cpp_api.html#_CP | ages/cpp_api.html#_CPPv4N5cudaq10 |
| Pv4N5cudaq14complex_matrixplERK14 | product_opdVERK15scalar_operator) |
| complex_matrixRK14complex_matrix) | -   [cudaq::product_op::operator= |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::operator- |     function)](api/la             |
|     (C++                          | nguages/cpp_api.html#_CPPv4I0_NSt |
|     function                      | 11enable_if_tIXaantNSt7is_sameI1T |
| )](api/languages/cpp_api.html#_CP | 9HandlerTyE5valueENSt16is_constru |
| Pv4N5cudaq14complex_matrixmiERK14 | ctibleI9HandlerTy1TE5valueEEbEEEN |
| complex_matrixRK14complex_matrix) | 5cudaq10product_opaSER10product_o |
| -   [cu                           | pI9HandlerTyERK10product_opI1TE), |
| daq::complex_matrix::operator\[\] |     [\[1\]](api/languages/cpp     |
|     (C++                          | _api.html#_CPPv4N5cudaq10product_ |
|                                   | opaSERK10product_opI9HandlerTyE), |
|  function)](api/languages/cpp_api |     [\[2\]](api/languages/cp      |
| .html#_CPPv4N5cudaq14complex_matr | p_api.html#_CPPv4N5cudaq10product |
| ixixERKNSt6vectorINSt6size_tEEE), | _opaSERR10product_opI9HandlerTyE) |
|     [\[1\]](api/languages/cpp_api | -                                 |
| .html#_CPPv4NK5cudaq14complex_mat |    [cudaq::product_op::operator== |
| rixixERKNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq::complex_matrix::power |     function)](api/languages/cpp  |
|     (C++                          | _api.html#_CPPv4NK5cudaq10product |
|     function)]                    | _opeqERK10product_opI9HandlerTyE) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq14complex_matrix5powerEi) |  [cudaq::product_op::operator\[\] |
| -                                 |     (C++                          |
|  [cudaq::complex_matrix::set_zero |     function)](ap                 |
|     (C++                          | i/languages/cpp_api.html#_CPPv4NK |
|     function)](ap                 | 5cudaq10product_opixENSt6size_tE) |
| i/languages/cpp_api.html#_CPPv4N5 | -                                 |
| cudaq14complex_matrix8set_zeroEv) |    [cudaq::product_op::product_op |
| -                                 |     (C++                          |
| [cudaq::complex_matrix::to_string |     function)](api/languages/c    |
|     (C++                          | pp_api.html#_CPPv4I0_NSt11enable_ |
|     function)](api/               | if_tIXaaNSt7is_sameI9HandlerTy14m |
| languages/cpp_api.html#_CPPv4NK5c | atrix_handlerE5valueEaantNSt7is_s |
| udaq14complex_matrix9to_stringEv) | ameI1T9HandlerTyE5valueENSt16is_c |
| -   [                             | onstructibleI9HandlerTy1TE5valueE |
| cudaq::complex_matrix::value_type | EbEEEN5cudaq10product_op10product |
|     (C++                          | _opERK10product_opI1TERKN14matrix |
|     type)](api/                   | _handler20commutation_behaviorE), |
| languages/cpp_api.html#_CPPv4N5cu |                                   |
| daq14complex_matrix10value_typeE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [cudaq::contrib (C++          | ml#_CPPv4I0_NSt11enable_if_tIXaan |
|     type)](api/languages/cpp      | tNSt7is_sameI1T9HandlerTyE5valueE |
| _api.html#_CPPv4N5cudaq7contribE) | NSt16is_constructibleI9HandlerTy1 |
| -   [cudaq::contrib::draw (C++    | TE5valueEEbEEEN5cudaq10product_op |
|     function)                     | 10product_opERK10product_opI1TE), |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4I0DpEN5cudaq7contrib4drawENSt6s |   [\[2\]](api/languages/cpp_api.h |
| tringERR13QuantumKernelDpRR4Args) | tml#_CPPv4N5cudaq10product_op10pr |
| -                                 | oduct_opENSt6size_tENSt6size_tE), |
| [cudaq::contrib::get_unitary_cmat |     [\[3\]](api/languages/cp      |
|     (C++                          | p_api.html#_CPPv4N5cudaq10product |
|     function)](api/languages/cp   | _op10product_opENSt7complexIdEE), |
| p_api.html#_CPPv4I0DpEN5cudaq7con |     [\[4\]](api/l                 |
| trib16get_unitary_cmatE14complex_ | anguages/cpp_api.html#_CPPv4N5cud |
| matrixRR13QuantumKernelDpRR4Args) | aq10product_op10product_opERK10pr |
| -   [cudaq::CusvState (C++        | oduct_opI9HandlerTyENSt6size_tE), |
|                                   |     [\[5\]](api/l                 |
|    class)](api/languages/cpp_api. | anguages/cpp_api.html#_CPPv4N5cud |
| html#_CPPv4I0EN5cudaq9CusvStateE) | aq10product_op10product_opERR10pr |
| -   [cudaq::depolarization1 (C++  | oduct_opI9HandlerTyENSt6size_tE), |
|     c                             |     [\[6\]](api/languages         |
| lass)](api/languages/cpp_api.html | /cpp_api.html#_CPPv4N5cudaq10prod |
| #_CPPv4N5cudaq15depolarization1E) | uct_op10product_opERR9HandlerTy), |
| -   [cudaq::depolarization2 (C++  |     [\[7\]](ap                    |
|     c                             | i/languages/cpp_api.html#_CPPv4N5 |
| lass)](api/languages/cpp_api.html | cudaq10product_op10product_opEd), |
| #_CPPv4N5cudaq15depolarization2E) |     [\[8\]](a                     |
| -   [cudaq:                       | pi/languages/cpp_api.html#_CPPv4N |
| :depolarization2::depolarization2 | 5cudaq10product_op10product_opEv) |
|     (C++                          | -   [cuda                         |
|     function)](api/languages/cp   | q::product_op::to_diagonal_matrix |
| p_api.html#_CPPv4N5cudaq15depolar |     (C++                          |
| ization215depolarization2EK4real) |     function)](api/               |
| -   [cudaq                        | languages/cpp_api.html#_CPPv4NK5c |
| ::depolarization2::num_parameters | udaq10product_op18to_diagonal_mat |
|     (C++                          | rixENSt13unordered_mapINSt6size_t |
|     member)](api/langu            | ENSt7int64_tEEERKNSt13unordered_m |
| ages/cpp_api.html#_CPPv4N5cudaq15 | apINSt6stringENSt7complexIdEEEEb) |
| depolarization214num_parametersE) | -   [cudaq::product_op::to_matrix |
| -   [cu                           |     (C++                          |
| daq::depolarization2::num_targets |     funct                         |
|     (C++                          | ion)](api/languages/cpp_api.html# |
|     member)](api/la               | _CPPv4NK5cudaq10product_op9to_mat |
| nguages/cpp_api.html#_CPPv4N5cuda | rixENSt13unordered_mapINSt6size_t |
| q15depolarization211num_targetsE) | ENSt7int64_tEEERKNSt13unordered_m |
| -                                 | apINSt6stringENSt7complexIdEEEEb) |
|    [cudaq::depolarization_channel | -   [cu                           |
|     (C++                          | daq::product_op::to_sparse_matrix |
|     class)](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](ap                 |
| N5cudaq22depolarization_channelE) | i/languages/cpp_api.html#_CPPv4NK |
| -   [cudaq::depol                 | 5cudaq10product_op16to_sparse_mat |
| arization_channel::num_parameters | rixENSt13unordered_mapINSt6size_t |
|     (C++                          | ENSt7int64_tEEERKNSt13unordered_m |
|     member)](api/languages/cp     | apINSt6stringENSt7complexIdEEEEb) |
| p_api.html#_CPPv4N5cudaq22depolar | -   [cudaq::product_op::to_string |
| ization_channel14num_parametersE) |     (C++                          |
| -   [cudaq::de                    |     function)](                   |
| polarization_channel::num_targets | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq10product_op9to_stringEv) |
|     member)](api/languages        | -                                 |
| /cpp_api.html#_CPPv4N5cudaq22depo |  [cudaq::product_op::\~product_op |
| larization_channel11num_targetsE) |     (C++                          |
| -   [cudaq::details (C++          |     fu                            |
|     type)](api/languages/cpp      | nction)](api/languages/cpp_api.ht |
| _api.html#_CPPv4N5cudaq7detailsE) | ml#_CPPv4N5cudaq10product_opD0Ev) |
| -   [cudaq::details::future (C++  | -   [cudaq::QPU (C++              |
|                                   |     class)](api/languages         |
|  class)](api/languages/cpp_api.ht | /cpp_api.html#_CPPv4N5cudaq3QPUE) |
| ml#_CPPv4N5cudaq7details6futureE) | -   [cudaq::QPU::beginExecution   |
| -                                 |     (C++                          |
|   [cudaq::details::future::future |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     functio                       | Pv4N5cudaq3QPU14beginExecutionEv) |
| n)](api/languages/cpp_api.html#_C | -   [cuda                         |
| PPv4N5cudaq7details6future6future | q::QPU::configureExecutionContext |
| ERNSt6vectorI3JobEERNSt6stringERN |     (C++                          |
| St3mapINSt6stringENSt6stringEEE), |     funct                         |
|     [\[1\]](api/lang              | ion)](api/languages/cpp_api.html# |
| uages/cpp_api.html#_CPPv4N5cudaq7 | _CPPv4NK5cudaq3QPU25configureExec |
| details6future6futureERR6future), | utionContextER16ExecutionContext) |
|     [\[2\]]                       | -   [cudaq::QPU::endExecution     |
| (api/languages/cpp_api.html#_CPPv |     (C++                          |
| 4N5cudaq7details6future6futureEv) |     functi                        |
| -   [cu                           | on)](api/languages/cpp_api.html#_ |
| daq::details::kernel_builder_base | CPPv4N5cudaq3QPU12endExecutionEv) |
|     (C++                          | -   [cudaq::QPU::enqueue (C++     |
|     class)](api/l                 |     function)](ap                 |
| anguages/cpp_api.html#_CPPv4N5cud | i/languages/cpp_api.html#_CPPv4N5 |
| aq7details19kernel_builder_baseE) | cudaq3QPU7enqueueER11QuantumTask) |
| -   [cudaq::details::             | -   [cud                          |
| kernel_builder_base::operator\<\< | aq::QPU::finalizeExecutionContext |
|     (C++                          |     (C++                          |
|     function)](api/langua         |     func                          |
| ges/cpp_api.html#_CPPv4N5cudaq7de | tion)](api/languages/cpp_api.html |
| tails19kernel_builder_baselsERNSt | #_CPPv4NK5cudaq3QPU24finalizeExec |
| 7ostreamERK19kernel_builder_base) | utionContextER16ExecutionContext) |
| -   [                             | -   [cudaq::QPU::getConnectivity  |
| cudaq::details::KernelBuilderType |     (C++                          |
|     (C++                          |     function)                     |
|     class)](api                   | ](api/languages/cpp_api.html#_CPP |
| /languages/cpp_api.html#_CPPv4N5c | v4N5cudaq3QPU15getConnectivityEv) |
| udaq7details17KernelBuilderTypeE) | -                                 |
| -   [cudaq::d                     | [cudaq::QPU::getExecutionThreadId |
| etails::KernelBuilderType::create |     (C++                          |
|     (C++                          |     function)](api/               |
|     function)                     | languages/cpp_api.html#_CPPv4NK5c |
| ](api/languages/cpp_api.html#_CPP | udaq3QPU20getExecutionThreadIdEv) |
| v4N5cudaq7details17KernelBuilderT | -   [cudaq::QPU::getNumQubits     |
| ype6createEPN4mlir11MLIRContextE) |     (C++                          |
| -   [cudaq::details::Ker          |     functi                        |
| nelBuilderType::KernelBuilderType | on)](api/languages/cpp_api.html#_ |
|     (C++                          | CPPv4N5cudaq3QPU12getNumQubitsEv) |
|     function)](api/lang           | -   [                             |
| uages/cpp_api.html#_CPPv4N5cudaq7 | cudaq::QPU::getRemoteCapabilities |
| details17KernelBuilderType17Kerne |     (C++                          |
| lBuilderTypeERRNSt8functionIFN4ml |     function)](api/l              |
| ir4TypeEPN4mlir11MLIRContextEEEE) | anguages/cpp_api.html#_CPPv4NK5cu |
| -   [cudaq::diag_matrix_callback  | daq3QPU21getRemoteCapabilitiesEv) |
|     (C++                          | -   [cudaq::QPU::isEmulated (C++  |
|     class)                        |     func                          |
| ](api/languages/cpp_api.html#_CPP | tion)](api/languages/cpp_api.html |
| v4N5cudaq20diag_matrix_callbackE) | #_CPPv4N5cudaq3QPU10isEmulatedEv) |
| -   [cudaq::dyn (C++              | -   [cudaq::QPU::isSimulator (C++ |
|     member)](api/languages        |     funct                         |
| /cpp_api.html#_CPPv4N5cudaq3dynE) | ion)](api/languages/cpp_api.html# |
| -   [cudaq::ExecutionContext (C++ | _CPPv4N5cudaq3QPU11isSimulatorEv) |
|     cl                            | -   [cudaq::QPU::launchKernel     |
| ass)](api/languages/cpp_api.html# |     (C++                          |
| _CPPv4N5cudaq16ExecutionContextE) |     function)](api/               |
| -   [cudaq                        | languages/cpp_api.html#_CPPv4N5cu |
| ::ExecutionContext::amplitudeMaps | daq3QPU12launchKernelERKNSt6strin |
|     (C++                          | gE15KernelThunkTypePvNSt8uint64_t |
|     member)](api/langu            | ENSt8uint64_tERKNSt6vectorIPvEE), |
| ages/cpp_api.html#_CPPv4N5cudaq16 |                                   |
| ExecutionContext13amplitudeMapsE) |  [\[1\]](api/languages/cpp_api.ht |
| -   [c                            | ml#_CPPv4N5cudaq3QPU12launchKerne |
| udaq::ExecutionContext::asyncExec | lERKNSt6stringERKNSt6vectorIPvEE) |
|     (C++                          | -   [cudaq::QPU::onRandomSeedSet  |
|     member)](api/                 |     (C++                          |
| languages/cpp_api.html#_CPPv4N5cu |     function)](api/lang           |
| daq16ExecutionContext9asyncExecE) | uages/cpp_api.html#_CPPv4N5cudaq3 |
| -   [cud                          | QPU15onRandomSeedSetENSt6size_tE) |
| aq::ExecutionContext::asyncResult | -   [cudaq::QPU::QPU (C++         |
|     (C++                          |     functio                       |
|     member)](api/lan              | n)](api/languages/cpp_api.html#_C |
| guages/cpp_api.html#_CPPv4N5cudaq | PPv4N5cudaq3QPU3QPUENSt6size_tE), |
| 16ExecutionContext11asyncResultE) |                                   |
| -   [cudaq:                       |  [\[1\]](api/languages/cpp_api.ht |
| :ExecutionContext::batchIteration | ml#_CPPv4N5cudaq3QPU3QPUERR3QPU), |
|     (C++                          |     [\[2\]](api/languages/cpp_    |
|     member)](api/langua           | api.html#_CPPv4N5cudaq3QPU3QPUEv) |
| ges/cpp_api.html#_CPPv4N5cudaq16E | -   [cudaq::QPU::setId (C++       |
| xecutionContext14batchIterationE) |     function                      |
| -   [cudaq::E                     | )](api/languages/cpp_api.html#_CP |
| xecutionContext::canHandleObserve | Pv4N5cudaq3QPU5setIdENSt6size_tE) |
|     (C++                          | -   [cudaq::QPU::setShots (C++    |
|     member)](api/language         |     f                             |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | unction)](api/languages/cpp_api.h |
| cutionContext16canHandleObserveE) | tml#_CPPv4N5cudaq3QPU8setShotsEi) |
| -   [cudaq::E                     | -   [cudaq:                       |
| xecutionContext::ExecutionContext | :QPU::supportsConditionalFeedback |
|     (C++                          |     (C++                          |
|     func                          |     function)](api/langua         |
| tion)](api/languages/cpp_api.html | ges/cpp_api.html#_CPPv4N5cudaq3QP |
| #_CPPv4N5cudaq16ExecutionContext1 | U27supportsConditionalFeedbackEv) |
| 6ExecutionContextERKNSt6stringE), | -   [cudaq::                      |
|     [\[1\]](api/languages/        | QPU::supportsExplicitMeasurements |
| cpp_api.html#_CPPv4N5cudaq16Execu |     (C++                          |
| tionContext16ExecutionContextERKN |     function)](api/languag        |
| St6stringENSt6size_tENSt6size_tE) | es/cpp_api.html#_CPPv4N5cudaq3QPU |
| -   [cudaq::E                     | 28supportsExplicitMeasurementsEv) |
| xecutionContext::expectationValue | -   [cudaq::QPU::\~QPU (C++       |
|     (C++                          |     function)](api/languages/cp   |
|     member)](api/language         | p_api.html#_CPPv4N5cudaq3QPUD0Ev) |
| s/cpp_api.html#_CPPv4N5cudaq16Exe | -   [cudaq::QPUState (C++         |
| cutionContext16expectationValueE) |     class)](api/languages/cpp_    |
| -   [cudaq::Execu                 | api.html#_CPPv4N5cudaq8QPUStateE) |
| tionContext::explicitMeasurements | -   [cudaq::qreg (C++             |
|     (C++                          |     class)](api/lan               |
|     member)](api/languages/cp     | guages/cpp_api.html#_CPPv4I_NSt6s |
| p_api.html#_CPPv4N5cudaq16Executi | ize_tE_NSt6size_tEEN5cudaq4qregE) |
| onContext20explicitMeasurementsE) | -   [cudaq::qreg::back (C++       |
| -   [cuda                         |     function)                     |
| q::ExecutionContext::futureResult | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4backENSt6size_tE), |
|     member)](api/lang             |     [\[1\]](api/languages/cpp_ap  |
| uages/cpp_api.html#_CPPv4N5cudaq1 | i.html#_CPPv4N5cudaq4qreg4backEv) |
| 6ExecutionContext12futureResultE) | -   [cudaq::qreg::begin (C++      |
| -   [cudaq::ExecutionContext      |                                   |
| ::hasConditionalsOnMeasureResults |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4N5cudaq4qreg5beginEv) |
|     mem                           | -   [cudaq::qreg::clear (C++      |
| ber)](api/languages/cpp_api.html# |                                   |
| _CPPv4N5cudaq16ExecutionContext31 |  function)](api/languages/cpp_api |
| hasConditionalsOnMeasureResultsE) | .html#_CPPv4N5cudaq4qreg5clearEv) |
| -   [cudaq::Executi               | -   [cudaq::qreg::front (C++      |
| onContext::invocationResultBuffer |     function)]                    |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     member)](api/languages/cpp_   | 4N5cudaq4qreg5frontENSt6size_tE), |
| api.html#_CPPv4N5cudaq16Execution |     [\[1\]](api/languages/cpp_api |
| Context22invocationResultBufferE) | .html#_CPPv4N5cudaq4qreg5frontEv) |
| -                                 | -   [cudaq::qreg::operator\[\]    |
|  [cudaq::ExecutionContext::jitEng |     (C++                          |
|     (C++                          |     functi                        |
|     member)](a                    | on)](api/languages/cpp_api.html#_ |
| pi/languages/cpp_api.html#_CPPv4N | CPPv4N5cudaq4qregixEKNSt6size_tE) |
| 5cudaq16ExecutionContext6jitEngE) | -   [cudaq::qreg::qreg (C++       |
| -   [cu                           |     function)                     |
| daq::ExecutionContext::kernelName | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq4qreg4qregENSt6size_tE), |
|     member)](api/la               |     [\[1\]](api/languages/cpp_ap  |
| nguages/cpp_api.html#_CPPv4N5cuda | i.html#_CPPv4N5cudaq4qreg4qregEv) |
| q16ExecutionContext10kernelNameE) | -   [cudaq::qreg::size (C++       |
| -   [cud                          |                                   |
| aq::ExecutionContext::kernelTrace |  function)](api/languages/cpp_api |
|     (C++                          | .html#_CPPv4NK5cudaq4qreg4sizeEv) |
|     member)](api/lan              | -   [cudaq::qreg::slice (C++      |
| guages/cpp_api.html#_CPPv4N5cudaq |     function)](api/langu          |
| 16ExecutionContext11kernelTraceE) | ages/cpp_api.html#_CPPv4N5cudaq4q |
| -   [cudaq:                       | reg5sliceENSt6size_tENSt6size_tE) |
| :ExecutionContext::msm_dimensions | -   [cudaq::qreg::value_type (C++ |
|     (C++                          |                                   |
|     member)](api/langua           | type)](api/languages/cpp_api.html |
| ges/cpp_api.html#_CPPv4N5cudaq16E | #_CPPv4N5cudaq4qreg10value_typeE) |
| xecutionContext14msm_dimensionsE) | -   [cudaq::qspan (C++            |
| -   [cudaq::                      |     class)](api/lang              |
| ExecutionContext::msm_prob_err_id | uages/cpp_api.html#_CPPv4I_NSt6si |
|     (C++                          | ze_tE_NSt6size_tEEN5cudaq5qspanE) |
|     member)](api/languag          | -   [cudaq::QuakeValue (C++       |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     class)](api/languages/cpp_api |
| ecutionContext15msm_prob_err_idE) | .html#_CPPv4N5cudaq10QuakeValueE) |
| -   [cudaq::Ex                    | -   [cudaq::Q                     |
| ecutionContext::msm_probabilities | uakeValue::canValidateNumElements |
|     (C++                          |     (C++                          |
|     member)](api/languages        |     function)](api/languages      |
| /cpp_api.html#_CPPv4N5cudaq16Exec | /cpp_api.html#_CPPv4N5cudaq10Quak |
| utionContext17msm_probabilitiesE) | eValue22canValidateNumElementsEv) |
| -                                 | -                                 |
|    [cudaq::ExecutionContext::name |  [cudaq::QuakeValue::constantSize |
|     (C++                          |     (C++                          |
|     member)]                      |     function)](api                |
| (api/languages/cpp_api.html#_CPPv | /languages/cpp_api.html#_CPPv4N5c |
| 4N5cudaq16ExecutionContext4nameE) | udaq10QuakeValue12constantSizeEv) |
| -   [cu                           | -   [cudaq::QuakeValue::dump (C++ |
| daq::ExecutionContext::noiseModel |     function)](api/lan            |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     member)](api/la               | 10QuakeValue4dumpERNSt7ostreamE), |
| nguages/cpp_api.html#_CPPv4N5cuda |     [\                            |
| q16ExecutionContext10noiseModelE) | [1\]](api/languages/cpp_api.html# |
| -   [cudaq::Exe                   | _CPPv4N5cudaq10QuakeValue4dumpEv) |
| cutionContext::numberTrajectories | -   [cudaq                        |
|     (C++                          | ::QuakeValue::getRequiredElements |
|     member)](api/languages/       |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq16Execu |     function)](api/langua         |
| tionContext18numberTrajectoriesE) | ges/cpp_api.html#_CPPv4N5cudaq10Q |
| -   [c                            | uakeValue19getRequiredElementsEv) |
| udaq::ExecutionContext::optResult | -   [cudaq::QuakeValue::getValue  |
|     (C++                          |     (C++                          |
|     member)](api/                 |     function)]                    |
| languages/cpp_api.html#_CPPv4N5cu | (api/languages/cpp_api.html#_CPPv |
| daq16ExecutionContext9optResultE) | 4NK5cudaq10QuakeValue8getValueEv) |
| -   [cudaq::Execu                 | -   [cudaq::QuakeValue::inverse   |
| tionContext::overlapComputeStates |     (C++                          |
|     (C++                          |     function)                     |
|     member)](api/languages/cp     | ](api/languages/cpp_api.html#_CPP |
| p_api.html#_CPPv4N5cudaq16Executi | v4NK5cudaq10QuakeValue7inverseEv) |
| onContext20overlapComputeStatesE) | -   [cudaq::QuakeValue::isStdVec  |
| -   [cudaq                        |     (C++                          |
| ::ExecutionContext::overlapResult |     function)                     |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     member)](api/langu            | v4N5cudaq10QuakeValue8isStdVecEv) |
| ages/cpp_api.html#_CPPv4N5cudaq16 | -                                 |
| ExecutionContext13overlapResultE) |    [cudaq::QuakeValue::operator\* |
| -                                 |     (C++                          |
|   [cudaq::ExecutionContext::qpuId |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](                     | udaq10QuakeValuemlE10QuakeValue), |
| api/languages/cpp_api.html#_CPPv4 |                                   |
| N5cudaq16ExecutionContext5qpuIdE) | [\[1\]](api/languages/cpp_api.htm |
| -   [cudaq                        | l#_CPPv4N5cudaq10QuakeValuemlEKd) |
| ::ExecutionContext::registerNames | -   [cudaq::QuakeValue::operator+ |
|     (C++                          |     (C++                          |
|     member)](api/langu            |     function)](api                |
| ages/cpp_api.html#_CPPv4N5cudaq16 | /languages/cpp_api.html#_CPPv4N5c |
| ExecutionContext13registerNamesE) | udaq10QuakeValueplE10QuakeValue), |
| -   [cu                           |     [                             |
| daq::ExecutionContext::reorderIdx | \[1\]](api/languages/cpp_api.html |
|     (C++                          | #_CPPv4N5cudaq10QuakeValueplEKd), |
|     member)](api/la               |                                   |
| nguages/cpp_api.html#_CPPv4N5cuda | [\[2\]](api/languages/cpp_api.htm |
| q16ExecutionContext10reorderIdxE) | l#_CPPv4N5cudaq10QuakeValueplEKi) |
| -                                 | -   [cudaq::QuakeValue::operator- |
|  [cudaq::ExecutionContext::result |     (C++                          |
|     (C++                          |     function)](api                |
|     member)](a                    | /languages/cpp_api.html#_CPPv4N5c |
| pi/languages/cpp_api.html#_CPPv4N | udaq10QuakeValuemiE10QuakeValue), |
| 5cudaq16ExecutionContext6resultE) |     [                             |
| -                                 | \[1\]](api/languages/cpp_api.html |
|   [cudaq::ExecutionContext::shots | #_CPPv4N5cudaq10QuakeValuemiEKd), |
|     (C++                          |     [                             |
|     member)](                     | \[2\]](api/languages/cpp_api.html |
| api/languages/cpp_api.html#_CPPv4 | #_CPPv4N5cudaq10QuakeValuemiEKi), |
| N5cudaq16ExecutionContext5shotsE) |                                   |
| -   [cudaq::                      | [\[3\]](api/languages/cpp_api.htm |
| ExecutionContext::simulationState | l#_CPPv4NK5cudaq10QuakeValuemiEv) |
|     (C++                          | -   [cudaq::QuakeValue::operator/ |
|     member)](api/languag          |     (C++                          |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     function)](api                |
| ecutionContext15simulationStateE) | /languages/cpp_api.html#_CPPv4N5c |
| -                                 | udaq10QuakeValuedvE10QuakeValue), |
|    [cudaq::ExecutionContext::spin |                                   |
|     (C++                          | [\[1\]](api/languages/cpp_api.htm |
|     member)]                      | l#_CPPv4N5cudaq10QuakeValuedvEKd) |
| (api/languages/cpp_api.html#_CPPv | -                                 |
| 4N5cudaq16ExecutionContext4spinE) |  [cudaq::QuakeValue::operator\[\] |
| -   [cudaq::                      |     (C++                          |
| ExecutionContext::totalIterations |     function)](api                |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|     member)](api/languag          | udaq10QuakeValueixEKNSt6size_tE), |
| es/cpp_api.html#_CPPv4N5cudaq16Ex |     [\[1\]](api/                  |
| ecutionContext15totalIterationsE) | languages/cpp_api.html#_CPPv4N5cu |
| -   [cudaq::ExecutionResult (C++  | daq10QuakeValueixERK10QuakeValue) |
|     st                            | -                                 |
| ruct)](api/languages/cpp_api.html |    [cudaq::QuakeValue::QuakeValue |
| #_CPPv4N5cudaq15ExecutionResultE) |     (C++                          |
| -   [cud                          |     function)](api/languag        |
| aq::ExecutionResult::appendResult | es/cpp_api.html#_CPPv4N5cudaq10Qu |
|     (C++                          | akeValue10QuakeValueERN4mlir20Imp |
|     functio                       | licitLocOpBuilderEN4mlir5ValueE), |
| n)](api/languages/cpp_api.html#_C |     [\[1\]                        |
| PPv4N5cudaq15ExecutionResult12app | ](api/languages/cpp_api.html#_CPP |
| endResultENSt6stringENSt6size_tE) | v4N5cudaq10QuakeValue10QuakeValue |
| -   [cu                           | ERN4mlir20ImplicitLocOpBuilderEd) |
| daq::ExecutionResult::deserialize | -   [cudaq::QuakeValue::size (C++ |
|     (C++                          |     funct                         |
|     function)                     | ion)](api/languages/cpp_api.html# |
| ](api/languages/cpp_api.html#_CPP | _CPPv4N5cudaq10QuakeValue4sizeEv) |
| v4N5cudaq15ExecutionResult11deser | -   [cudaq::QuakeValue::slice     |
| ializeERNSt6vectorINSt6size_tEEE) |     (C++                          |
| -   [cudaq:                       |     function)](api/languages/cpp_ |
| :ExecutionResult::ExecutionResult | api.html#_CPPv4N5cudaq10QuakeValu |
|     (C++                          | e5sliceEKNSt6size_tEKNSt6size_tE) |
|     functio                       | -   [cudaq::quantum_platform (C++ |
| n)](api/languages/cpp_api.html#_C |     cl                            |
| PPv4N5cudaq15ExecutionResult15Exe | ass)](api/languages/cpp_api.html# |
| cutionResultE16CountsDictionary), | _CPPv4N5cudaq16quantum_platformE) |
|     [\[1\]](api/lan               | -   [cudaq:                       |
| guages/cpp_api.html#_CPPv4N5cudaq | :quantum_platform::beginExecution |
| 15ExecutionResult15ExecutionResul |     (C++                          |
| tE16CountsDictionaryNSt6stringE), |     function)](api/languag        |
|     [\[2\                         | es/cpp_api.html#_CPPv4N5cudaq16qu |
| ]](api/languages/cpp_api.html#_CP | antum_platform14beginExecutionEv) |
| Pv4N5cudaq15ExecutionResult15Exec | -   [cudaq::quantum_pl            |
| utionResultE16CountsDictionaryd), | atform::configureExecutionContext |
|                                   |     (C++                          |
|    [\[3\]](api/languages/cpp_api. |     function)](api/lang           |
| html#_CPPv4N5cudaq15ExecutionResu | uages/cpp_api.html#_CPPv4NK5cudaq |
| lt15ExecutionResultENSt6stringE), | 16quantum_platform25configureExec |
|     [\[4\                         | utionContextER16ExecutionContext) |
| ]](api/languages/cpp_api.html#_CP | -   [cuda                         |
| Pv4N5cudaq15ExecutionResult15Exec | q::quantum_platform::connectivity |
| utionResultERK15ExecutionResult), |     (C++                          |
|     [\[5\]](api/language          |     function)](api/langu          |
| s/cpp_api.html#_CPPv4N5cudaq15Exe | ages/cpp_api.html#_CPPv4N5cudaq16 |
| cutionResult15ExecutionResultEd), | quantum_platform12connectivityEv) |
|     [\[6\]](api/languag           | -   [cuda                         |
| es/cpp_api.html#_CPPv4N5cudaq15Ex | q::quantum_platform::endExecution |
| ecutionResult15ExecutionResultEv) |     (C++                          |
| -   [                             |     function)](api/langu          |
| cudaq::ExecutionResult::operator= | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     (C++                          | quantum_platform12endExecutionEv) |
|     function)](api/languages/     | -   [cudaq::q                     |
| cpp_api.html#_CPPv4N5cudaq15Execu | uantum_platform::enqueueAsyncTask |
| tionResultaSERK15ExecutionResult) |     (C++                          |
| -   [c                            |     function)](api/languages/     |
| udaq::ExecutionResult::operator== | cpp_api.html#_CPPv4N5cudaq16quant |
|     (C++                          | um_platform16enqueueAsyncTaskEKNS |
|     function)](api/languages/c    | t6size_tER19KernelExecutionTask), |
| pp_api.html#_CPPv4NK5cudaq15Execu |     [\[1\]](api/languag           |
| tionResulteqERK15ExecutionResult) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [cud                          | antum_platform16enqueueAsyncTaskE |
| aq::ExecutionResult::registerName | KNSt6size_tERNSt8functionIFvvEEE) |
|     (C++                          | -   [cudaq::quantum_p             |
|     member)](api/lan              | latform::finalizeExecutionContext |
| guages/cpp_api.html#_CPPv4N5cudaq |     (C++                          |
| 15ExecutionResult12registerNameE) |     function)](api/languages/c    |
| -   [cudaq                        | pp_api.html#_CPPv4NK5cudaq16quant |
| ::ExecutionResult::sequentialData | um_platform24finalizeExecutionCon |
|     (C++                          | textERN5cudaq16ExecutionContextE) |
|     member)](api/langu            | -   [cudaq::qua                   |
| ages/cpp_api.html#_CPPv4N5cudaq15 | ntum_platform::get_codegen_config |
| ExecutionResult14sequentialDataE) |     (C++                          |
| -   [                             |     function)](api/languages/c    |
| cudaq::ExecutionResult::serialize | pp_api.html#_CPPv4N5cudaq16quantu |
|     (C++                          | m_platform18get_codegen_configEv) |
|     function)](api/l              | -   [c                            |
| anguages/cpp_api.html#_CPPv4NK5cu | udaq::quantum_platform::get_noise |
| daq15ExecutionResult9serializeEv) |     (C++                          |
| -   [cudaq::fermion_handler (C++  |     function)](api/languages/c    |
|     c                             | pp_api.html#_CPPv4N5cudaq16quantu |
| lass)](api/languages/cpp_api.html | m_platform9get_noiseENSt6size_tE) |
| #_CPPv4N5cudaq15fermion_handlerE) | -   [cudaq:                       |
| -   [cudaq::fermion_op (C++       | :quantum_platform::get_num_qubits |
|     type)](api/languages/cpp_api  |     (C++                          |
| .html#_CPPv4N5cudaq10fermion_opE) |                                   |
| -   [cudaq::fermion_op_term (C++  | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq16quantum_plat |
| type)](api/languages/cpp_api.html | form14get_num_qubitsENSt6size_tE) |
| #_CPPv4N5cudaq15fermion_op_termE) | -   [cudaq::quantum_              |
| -   [cudaq::FermioniqBaseQPU (C++ | platform::get_remote_capabilities |
|     cl                            |     (C++                          |
| ass)](api/languages/cpp_api.html# |     function)                     |
| _CPPv4N5cudaq16FermioniqBaseQPUE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::get_state (C++        | v4NK5cudaq16quantum_platform23get |
|                                   | _remote_capabilitiesENSt6size_tE) |
|    function)](api/languages/cpp_a | -   [cudaq::qua                   |
| pi.html#_CPPv4I0DpEN5cudaq9get_st | ntum_platform::get_runtime_target |
| ateEDaRR13QuantumKernelDpRR4Args) |     (C++                          |
| -   [cudaq::gradient (C++         |     function)](api/languages/cp   |
|     class)](api/languages/cpp_    | p_api.html#_CPPv4NK5cudaq16quantu |
| api.html#_CPPv4N5cudaq8gradientE) | m_platform18get_runtime_targetEv) |
| -   [cudaq::gradient::clone (C++  | -   [cuda                         |
|     fun                           | q::quantum_platform::getLogStream |
| ction)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq8gradient5cloneEv) |     function)](api/langu          |
| -   [cudaq::gradient::compute     | ages/cpp_api.html#_CPPv4N5cudaq16 |
|     (C++                          | quantum_platform12getLogStreamEv) |
|     function)](api/language       | -   [cud                          |
| s/cpp_api.html#_CPPv4N5cudaq8grad | aq::quantum_platform::is_emulated |
| ient7computeERKNSt6vectorIdEERKNS |     (C++                          |
| t8functionIFdNSt6vectorIdEEEEEd), |                                   |
|     [\[1\]](ap                    |    function)](api/languages/cpp_a |
| i/languages/cpp_api.html#_CPPv4N5 | pi.html#_CPPv4NK5cudaq16quantum_p |
| cudaq8gradient7computeERKNSt6vect | latform11is_emulatedENSt6size_tE) |
| orIdEERNSt6vectorIdEERK7spin_opd) | -   [c                            |
| -   [cudaq::gradient::gradient    | udaq::quantum_platform::is_remote |
|     (C++                          |     (C++                          |
|     function)](api/lang           |     function)](api/languages/cp   |
| uages/cpp_api.html#_CPPv4I00EN5cu | p_api.html#_CPPv4NK5cudaq16quantu |
| daq8gradient8gradientER7KernelT), | m_platform9is_remoteENSt6size_tE) |
|                                   | -   [cuda                         |
|    [\[1\]](api/languages/cpp_api. | q::quantum_platform::is_simulator |
| html#_CPPv4I00EN5cudaq8gradient8g |     (C++                          |
| radientER7KernelTRR10ArgsMapper), |                                   |
|     [\[2\                         |   function)](api/languages/cpp_ap |
| ]](api/languages/cpp_api.html#_CP | i.html#_CPPv4NK5cudaq16quantum_pl |
| Pv4I00EN5cudaq8gradient8gradientE | atform12is_simulatorENSt6size_tE) |
| RR13QuantumKernelRR10ArgsMapper), | -   [c                            |
|     [\[3                          | udaq::quantum_platform::launchVQE |
| \]](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4N5cudaq8gradient8gradientERRN |     function)](                   |
| St8functionIFvNSt6vectorIdEEEEE), | api/languages/cpp_api.html#_CPPv4 |
|     [\[                           | N5cudaq16quantum_platform9launchV |
| 4\]](api/languages/cpp_api.html#_ | QEEKNSt6stringEPKvPN5cudaq8gradie |
| CPPv4N5cudaq8gradient8gradientEv) | ntERKN5cudaq7spin_opERN5cudaq9opt |
| -   [cudaq::gradient::setArgs     | imizerEKiKNSt6size_tENSt6size_tE) |
|     (C++                          | -   [cudaq:                       |
|     fu                            | :quantum_platform::list_platforms |
| nction)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4I0DpEN5cudaq8gradient7se |     function)](api/languag        |
| tArgsEvR13QuantumKernelDpRR4Args) | es/cpp_api.html#_CPPv4N5cudaq16qu |
| -   [cudaq::gradient::setKernel   | antum_platform14list_platformsEv) |
|     (C++                          | -                                 |
|     function)](api/languages/c    |    [cudaq::quantum_platform::name |
| pp_api.html#_CPPv4I0EN5cudaq8grad |     (C++                          |
| ient9setKernelEvR13QuantumKernel) |     function)](a                  |
| -   [cud                          | pi/languages/cpp_api.html#_CPPv4N |
| aq::gradients::central_difference | K5cudaq16quantum_platform4nameEv) |
|     (C++                          | -   [                             |
|     class)](api/la                | cudaq::quantum_platform::num_qpus |
| nguages/cpp_api.html#_CPPv4N5cuda |     (C++                          |
| q9gradients18central_differenceE) |     function)](api/l              |
| -   [cudaq::gra                   | anguages/cpp_api.html#_CPPv4NK5cu |
| dients::central_difference::clone | daq16quantum_platform8num_qpusEv) |
|     (C++                          | -   [cudaq::                      |
|     function)](api/languages      | quantum_platform::onRandomSeedSet |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18central_difference5cloneEv) |                                   |
| -   [cudaq::gradi                 | function)](api/languages/cpp_api. |
| ents::central_difference::compute | html#_CPPv4N5cudaq16quantum_platf |
|     (C++                          | orm15onRandomSeedSetENSt6size_tE) |
|     function)](                   | -   [cud                          |
| api/languages/cpp_api.html#_CPPv4 | aq::quantum_platform::reset_noise |
| N5cudaq9gradients18central_differ |     (C++                          |
| ence7computeERKNSt6vectorIdEERKNS |     function)](api/languages/cpp_ |
| t8functionIFdNSt6vectorIdEEEEEd), | api.html#_CPPv4N5cudaq16quantum_p |
|                                   | latform11reset_noiseENSt6size_tE) |
|   [\[1\]](api/languages/cpp_api.h | -   [cudaq:                       |
| tml#_CPPv4N5cudaq9gradients18cent | :quantum_platform::resetLogStream |
| ral_difference7computeERKNSt6vect |     (C++                          |
| orIdEERNSt6vectorIdEERK7spin_opd) |     function)](api/languag        |
| -   [cudaq::gradie                | es/cpp_api.html#_CPPv4N5cudaq16qu |
| nts::central_difference::gradient | antum_platform14resetLogStreamEv) |
|     (C++                          | -   [c                            |
|     functio                       | udaq::quantum_platform::set_noise |
| n)](api/languages/cpp_api.html#_C |     (C++                          |
| PPv4I00EN5cudaq9gradients18centra |     function                      |
| l_difference8gradientER7KernelT), | )](api/languages/cpp_api.html#_CP |
|     [\[1\]](api/langua            | Pv4N5cudaq16quantum_platform9set_ |
| ges/cpp_api.html#_CPPv4I00EN5cuda | noiseEPK11noise_modelNSt6size_tE) |
| q9gradients18central_difference8g | -   [cuda                         |
| radientER7KernelTRR10ArgsMapper), | q::quantum_platform::setLogStream |
|     [\[2\]](api/languages/cpp_    |     (C++                          |
| api.html#_CPPv4I00EN5cudaq9gradie |                                   |
| nts18central_difference8gradientE |  function)](api/languages/cpp_api |
| RR13QuantumKernelRR10ArgsMapper), | .html#_CPPv4N5cudaq16quantum_plat |
|     [\[3\]](api/languages/cpp     | form12setLogStreamERNSt7ostreamE) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::q                     |
| 18central_difference8gradientERRN | uantum_platform::setTargetBackend |
| St8functionIFvNSt6vectorIdEEEEE), |     (C++                          |
|     [\[4\]](api/languages/cp      |     fun                           |
| p_api.html#_CPPv4N5cudaq9gradient | ction)](api/languages/cpp_api.htm |
| s18central_difference8gradientEv) | l#_CPPv4N5cudaq16quantum_platform |
| -   [cud                          | 16setTargetBackendERKNSt6stringE) |
| aq::gradients::forward_difference | -   [cudaq::quantum_platfo        |
|     (C++                          | rm::supports_conditional_feedback |
|     class)](api/la                |     (C++                          |
| nguages/cpp_api.html#_CPPv4N5cuda |     function)](api/               |
| q9gradients18forward_differenceE) | languages/cpp_api.html#_CPPv4NK5c |
| -   [cudaq::gra                   | udaq16quantum_platform29supports_ |
| dients::forward_difference::clone | conditional_feedbackENSt6size_tE) |
|     (C++                          | -   [cudaq::quantum_platfor       |
|     function)](api/languages      | m::supports_explicit_measurements |
| /cpp_api.html#_CPPv4N5cudaq9gradi |     (C++                          |
| ents18forward_difference5cloneEv) |     function)](api/l              |
| -   [cudaq::gradi                 | anguages/cpp_api.html#_CPPv4NK5cu |
| ents::forward_difference::compute | daq16quantum_platform30supports_e |
|     (C++                          | xplicit_measurementsENSt6size_tE) |
|     function)](                   | -   [cudaq::quantum_pla           |
| api/languages/cpp_api.html#_CPPv4 | tform::supports_task_distribution |
| N5cudaq9gradients18forward_differ |     (C++                          |
| ence7computeERKNSt6vectorIdEERKNS |     fu                            |
| t8functionIFdNSt6vectorIdEEEEEd), | nction)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4NK5cudaq16quantum_platfo |
|   [\[1\]](api/languages/cpp_api.h | rm26supports_task_distributionEv) |
| tml#_CPPv4N5cudaq9gradients18forw | -   [cudaq::quantum               |
| ard_difference7computeERKNSt6vect | _platform::with_execution_context |
| orIdEERNSt6vectorIdEERK7spin_opd) |     (C++                          |
| -   [cudaq::gradie                |     function)                     |
| nts::forward_difference::gradient | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4I0DpEN5cudaq16quantum_platform2 |
|     functio                       | 2with_execution_contextEDaR16Exec |
| n)](api/languages/cpp_api.html#_C | utionContextRR8CallableDpRR4Args) |
| PPv4I00EN5cudaq9gradients18forwar | -   [cudaq::QuantumTask (C++      |
| d_difference8gradientER7KernelT), |     type)](api/languages/cpp_api. |
|     [\[1\]](api/langua            | html#_CPPv4N5cudaq11QuantumTaskE) |
| ges/cpp_api.html#_CPPv4I00EN5cuda | -   [cudaq::qubit (C++            |
| q9gradients18forward_difference8g |     type)](api/languages/c        |
| radientER7KernelTRR10ArgsMapper), | pp_api.html#_CPPv4N5cudaq5qubitE) |
|     [\[2\]](api/languages/cpp_    | -   [cudaq::QubitConnectivity     |
| api.html#_CPPv4I00EN5cudaq9gradie |     (C++                          |
| nts18forward_difference8gradientE |     ty                            |
| RR13QuantumKernelRR10ArgsMapper), | pe)](api/languages/cpp_api.html#_ |
|     [\[3\]](api/languages/cpp     | CPPv4N5cudaq17QubitConnectivityE) |
| _api.html#_CPPv4N5cudaq9gradients | -   [cudaq::QubitEdge (C++        |
| 18forward_difference8gradientERRN |     type)](api/languages/cpp_a    |
| St8functionIFvNSt6vectorIdEEEEE), | pi.html#_CPPv4N5cudaq9QubitEdgeE) |
|     [\[4\]](api/languages/cp      | -   [cudaq::qudit (C++            |
| p_api.html#_CPPv4N5cudaq9gradient |     clas                          |
| s18forward_difference8gradientEv) | s)](api/languages/cpp_api.html#_C |
| -   [                             | PPv4I_NSt6size_tEEN5cudaq5quditE) |
| cudaq::gradients::parameter_shift | -   [cudaq::qudit::qudit (C++     |
|     (C++                          |                                   |
|     class)](api                   | function)](api/languages/cpp_api. |
| /languages/cpp_api.html#_CPPv4N5c | html#_CPPv4N5cudaq5qudit5quditEv) |
| udaq9gradients15parameter_shiftE) | -   [cudaq::qvector (C++          |
| -   [cudaq::                      |     class)                        |
| gradients::parameter_shift::clone | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4I_NSt6size_tEEN5cudaq7qvectorE) |
|     function)](api/langua         | -   [cudaq::qvector::back (C++    |
| ges/cpp_api.html#_CPPv4N5cudaq9gr |     function)](a                  |
| adients15parameter_shift5cloneEv) | pi/languages/cpp_api.html#_CPPv4N |
| -   [cudaq::gr                    | 5cudaq7qvector4backENSt6size_tE), |
| adients::parameter_shift::compute |                                   |
|     (C++                          |   [\[1\]](api/languages/cpp_api.h |
|     function                      | tml#_CPPv4N5cudaq7qvector4backEv) |
| )](api/languages/cpp_api.html#_CP | -   [cudaq::qvector::begin (C++   |
| Pv4N5cudaq9gradients15parameter_s |     fu                            |
| hift7computeERKNSt6vectorIdEERKNS | nction)](api/languages/cpp_api.ht |
| t8functionIFdNSt6vectorIdEEEEEd), | ml#_CPPv4N5cudaq7qvector5beginEv) |
|     [\[1\]](api/languages/cpp_ap  | -   [cudaq::qvector::clear (C++   |
| i.html#_CPPv4N5cudaq9gradients15p |     fu                            |
| arameter_shift7computeERKNSt6vect | nction)](api/languages/cpp_api.ht |
| orIdEERNSt6vectorIdEERK7spin_opd) | ml#_CPPv4N5cudaq7qvector5clearEv) |
| -   [cudaq::gra                   | -   [cudaq::qvector::end (C++     |
| dients::parameter_shift::gradient |                                   |
|     (C++                          | function)](api/languages/cpp_api. |
|     func                          | html#_CPPv4N5cudaq7qvector3endEv) |
| tion)](api/languages/cpp_api.html | -   [cudaq::qvector::front (C++   |
| #_CPPv4I00EN5cudaq9gradients15par |     function)](ap                 |
| ameter_shift8gradientER7KernelT), | i/languages/cpp_api.html#_CPPv4N5 |
|     [\[1\]](api/lan               | cudaq7qvector5frontENSt6size_tE), |
| guages/cpp_api.html#_CPPv4I00EN5c |                                   |
| udaq9gradients15parameter_shift8g |  [\[1\]](api/languages/cpp_api.ht |
| radientER7KernelTRR10ArgsMapper), | ml#_CPPv4N5cudaq7qvector5frontEv) |
|     [\[2\]](api/languages/c       | -   [cudaq::qvector::operator=    |
| pp_api.html#_CPPv4I00EN5cudaq9gra |     (C++                          |
| dients15parameter_shift8gradientE |     functio                       |
| RR13QuantumKernelRR10ArgsMapper), | n)](api/languages/cpp_api.html#_C |
|     [\[3\]](api/languages/        | PPv4N5cudaq7qvectoraSERK7qvector) |
| cpp_api.html#_CPPv4N5cudaq9gradie | -   [cudaq::qvector::operator\[\] |
| nts15parameter_shift8gradientERRN |     (C++                          |
| St8functionIFvNSt6vectorIdEEEEE), |     function)                     |
|     [\[4\]](api/languages         | ](api/languages/cpp_api.html#_CPP |
| /cpp_api.html#_CPPv4N5cudaq9gradi | v4N5cudaq7qvectorixEKNSt6size_tE) |
| ents15parameter_shift8gradientEv) | -   [cudaq::qvector::qvector (C++ |
| -   [cudaq::kernel_builder (C++   |     function)](api/               |
|     clas                          | languages/cpp_api.html#_CPPv4N5cu |
| s)](api/languages/cpp_api.html#_C | daq7qvector7qvectorENSt6size_tE), |
| PPv4IDpEN5cudaq14kernel_builderE) |     [\[1\]](a                     |
| -   [c                            | pi/languages/cpp_api.html#_CPPv4N |
| udaq::kernel_builder::constantVal | 5cudaq7qvector7qvectorERK5state), |
|     (C++                          |     [\[2\]](api                   |
|     function)](api/la             | /languages/cpp_api.html#_CPPv4N5c |
| nguages/cpp_api.html#_CPPv4N5cuda | udaq7qvector7qvectorERK7qvector), |
| q14kernel_builder11constantValEd) |     [\[3\]](ap                    |
| -   [cu                           | i/languages/cpp_api.html#_CPPv4N5 |
| daq::kernel_builder::getArguments | cudaq7qvector7qvectorERR7qvector) |
|     (C++                          | -   [cudaq::qvector::size (C++    |
|     function)](api/lan            |     fu                            |
| guages/cpp_api.html#_CPPv4N5cudaq | nction)](api/languages/cpp_api.ht |
| 14kernel_builder12getArgumentsEv) | ml#_CPPv4NK5cudaq7qvector4sizeEv) |
| -   [cu                           | -   [cudaq::qvector::slice (C++   |
| daq::kernel_builder::getNumParams |     function)](api/language       |
|     (C++                          | s/cpp_api.html#_CPPv4N5cudaq7qvec |
|     function)](api/lan            | tor5sliceENSt6size_tENSt6size_tE) |
| guages/cpp_api.html#_CPPv4N5cudaq | -   [cudaq::qvector::value_type   |
| 14kernel_builder12getNumParamsEv) |     (C++                          |
| -   [c                            |     typ                           |
| udaq::kernel_builder::isArgStdVec | e)](api/languages/cpp_api.html#_C |
|     (C++                          | PPv4N5cudaq7qvector10value_typeE) |
|     function)](api/languages/cp   | -   [cudaq::qview (C++            |
| p_api.html#_CPPv4N5cudaq14kernel_ |     clas                          |
| builder11isArgStdVecENSt6size_tE) | s)](api/languages/cpp_api.html#_C |
| -   [cuda                         | PPv4I_NSt6size_tEEN5cudaq5qviewE) |
| q::kernel_builder::kernel_builder | -   [cudaq::qview::back (C++      |
|     (C++                          |     function)                     |
|     function)](api/languages/cpp_ | ](api/languages/cpp_api.html#_CPP |
| api.html#_CPPv4N5cudaq14kernel_bu | v4N5cudaq5qview4backENSt6size_tE) |
| ilder14kernel_builderERNSt6vector | -   [cudaq::qview::begin (C++     |
| IN7details17KernelBuilderTypeEEE) |                                   |
| -   [cudaq::kernel_builder::name  | function)](api/languages/cpp_api. |
|     (C++                          | html#_CPPv4N5cudaq5qview5beginEv) |
|     function)                     | -   [cudaq::qview::end (C++       |
| ](api/languages/cpp_api.html#_CPP |                                   |
| v4N5cudaq14kernel_builder4nameEv) |   function)](api/languages/cpp_ap |
| -                                 | i.html#_CPPv4N5cudaq5qview3endEv) |
|    [cudaq::kernel_builder::qalloc | -   [cudaq::qview::front (C++     |
|     (C++                          |     function)](                   |
|     function)](api/language       | api/languages/cpp_api.html#_CPPv4 |
| s/cpp_api.html#_CPPv4N5cudaq14ker | N5cudaq5qview5frontENSt6size_tE), |
| nel_builder6qallocE10QuakeValue), |                                   |
|     [\[1\]](api/language          |    [\[1\]](api/languages/cpp_api. |
| s/cpp_api.html#_CPPv4N5cudaq14ker | html#_CPPv4N5cudaq5qview5frontEv) |
| nel_builder6qallocEKNSt6size_tE), | -   [cudaq::qview::operator\[\]   |
|     [\[2                          |     (C++                          |
| \]](api/languages/cpp_api.html#_C |     functio                       |
| PPv4N5cudaq14kernel_builder6qallo | n)](api/languages/cpp_api.html#_C |
| cERNSt6vectorINSt7complexIdEEEE), | PPv4N5cudaq5qviewixEKNSt6size_tE) |
|     [\[3\]](                      | -   [cudaq::qview::qview (C++     |
| api/languages/cpp_api.html#_CPPv4 |     functio                       |
| N5cudaq14kernel_builder6qallocEv) | n)](api/languages/cpp_api.html#_C |
| -   [cudaq::kernel_builder::swap  | PPv4I0EN5cudaq5qview5qviewERR1R), |
|     (C++                          |     [\[1                          |
|     function)](api/language       | \]](api/languages/cpp_api.html#_C |
| s/cpp_api.html#_CPPv4I00EN5cudaq1 | PPv4N5cudaq5qview5qviewERK5qview) |
| 4kernel_builder4swapEvRK10QuakeVa | -   [cudaq::qview::size (C++      |
| lueRK10QuakeValueRK10QuakeValue), |                                   |
|                                   | function)](api/languages/cpp_api. |
| [\[1\]](api/languages/cpp_api.htm | html#_CPPv4NK5cudaq5qview4sizeEv) |
| l#_CPPv4I00EN5cudaq14kernel_build | -   [cudaq::qview::slice (C++     |
| er4swapEvRKNSt6vectorI10QuakeValu |     function)](api/langua         |
| eEERK10QuakeValueRK10QuakeValue), | ges/cpp_api.html#_CPPv4N5cudaq5qv |
|                                   | iew5sliceENSt6size_tENSt6size_tE) |
| [\[2\]](api/languages/cpp_api.htm | -   [cudaq::qview::value_type     |
| l#_CPPv4N5cudaq14kernel_builder4s |     (C++                          |
| wapERK10QuakeValueRK10QuakeValue) |     t                             |
| -   [cudaq::KernelExecutionTask   | ype)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4N5cudaq5qview10value_typeE) |
|     type                          | -   [cudaq::range (C++            |
| )](api/languages/cpp_api.html#_CP |     fun                           |
| Pv4N5cudaq19KernelExecutionTaskE) | ction)](api/languages/cpp_api.htm |
| -   [cudaq::KernelThunkResultType | l#_CPPv4I0EN5cudaq5rangeENSt6vect |
|     (C++                          | orI11ElementTypeEE11ElementType), |
|     struct)]                      |     [\[1\]](api/languages/cpp_    |
| (api/languages/cpp_api.html#_CPPv | api.html#_CPPv4I0EN5cudaq5rangeEN |
| 4N5cudaq21KernelThunkResultTypeE) | St6vectorI11ElementTypeEE11Elemen |
| -   [cudaq::KernelThunkType (C++  | tType11ElementType11ElementType), |
|                                   |     [                             |
| type)](api/languages/cpp_api.html | \[2\]](api/languages/cpp_api.html |
| #_CPPv4N5cudaq15KernelThunkTypeE) | #_CPPv4N5cudaq5rangeENSt6size_tE) |
| -   [cudaq::kraus_channel (C++    | -   [cudaq::real (C++             |
|                                   |     type)](api/languages/         |
|  class)](api/languages/cpp_api.ht | cpp_api.html#_CPPv4N5cudaq4realE) |
| ml#_CPPv4N5cudaq13kraus_channelE) | -   [cudaq::registry (C++         |
| -   [cudaq::kraus_channel::empty  |     type)](api/languages/cpp_     |
|     (C++                          | api.html#_CPPv4N5cudaq8registryE) |
|     function)]                    | -                                 |
| (api/languages/cpp_api.html#_CPPv |  [cudaq::registry::RegisteredType |
| 4NK5cudaq13kraus_channel5emptyEv) |     (C++                          |
| -   [cudaq::kraus_c               |     class)](api/                  |
| hannel::generateUnitaryParameters | languages/cpp_api.html#_CPPv4I0EN |
|     (C++                          | 5cudaq8registry14RegisteredTypeE) |
|                                   | -   [cudaq::RemoteCapabilities    |
|    function)](api/languages/cpp_a |     (C++                          |
| pi.html#_CPPv4N5cudaq13kraus_chan |     struc                         |
| nel25generateUnitaryParametersEv) | t)](api/languages/cpp_api.html#_C |
| -                                 | PPv4N5cudaq18RemoteCapabilitiesE) |
|    [cudaq::kraus_channel::get_ops | -   [cudaq::Remo                  |
|     (C++                          | teCapabilities::isRemoteSimulator |
|     function)](a                  |     (C++                          |
| pi/languages/cpp_api.html#_CPPv4N |     member)](api/languages/c      |
| K5cudaq13kraus_channel7get_opsEv) | pp_api.html#_CPPv4N5cudaq18Remote |
| -   [cudaq::                      | Capabilities17isRemoteSimulatorE) |
| kraus_channel::is_unitary_mixture | -   [cudaq::Remot                 |
|     (C++                          | eCapabilities::RemoteCapabilities |
|     function)](api/languages      |     (C++                          |
| /cpp_api.html#_CPPv4NK5cudaq13kra |     function)](api/languages/cpp  |
| us_channel18is_unitary_mixtureEv) | _api.html#_CPPv4N5cudaq18RemoteCa |
| -   [cu                           | pabilities18RemoteCapabilitiesEb) |
| daq::kraus_channel::kraus_channel | -   [cudaq:                       |
|     (C++                          | :RemoteCapabilities::stateOverlap |
|     function)](api/lang           |     (C++                          |
| uages/cpp_api.html#_CPPv4IDpEN5cu |     member)](api/langua           |
| daq13kraus_channel13kraus_channel | ges/cpp_api.html#_CPPv4N5cudaq18R |
| EDpRRNSt16initializer_listI1TEE), | emoteCapabilities12stateOverlapE) |
|                                   | -                                 |
|  [\[1\]](api/languages/cpp_api.ht |   [cudaq::RemoteCapabilities::vqe |
| ml#_CPPv4N5cudaq13kraus_channel13 |     (C++                          |
| kraus_channelERK13kraus_channel), |     member)](                     |
|     [\[2\]                        | api/languages/cpp_api.html#_CPPv4 |
| ](api/languages/cpp_api.html#_CPP | N5cudaq18RemoteCapabilities3vqeE) |
| v4N5cudaq13kraus_channel13kraus_c | -   [cudaq::RemoteSimulationState |
| hannelERKNSt6vectorI8kraus_opEE), |     (C++                          |
|     [\[3\]                        |     class)]                       |
| ](api/languages/cpp_api.html#_CPP | (api/languages/cpp_api.html#_CPPv |
| v4N5cudaq13kraus_channel13kraus_c | 4N5cudaq21RemoteSimulationStateE) |
| hannelERRNSt6vectorI8kraus_opEE), | -   [cudaq::Resources (C++        |
|     [\[4\]](api/lan               |     class)](api/languages/cpp_a   |
| guages/cpp_api.html#_CPPv4N5cudaq | pi.html#_CPPv4N5cudaq9ResourcesE) |
| 13kraus_channel13kraus_channelEv) | -   [cudaq::run (C++              |
| -                                 |     function)]                    |
| [cudaq::kraus_channel::noise_type | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4I0DpEN5cudaq3runENSt6vectorINSt1 |
|     member)](api                  | 5invoke_result_tINSt7decay_tI13Qu |
| /languages/cpp_api.html#_CPPv4N5c | antumKernelEEDpNSt7decay_tI4ARGSE |
| udaq13kraus_channel10noise_typeE) | EEEEENSt6size_tERN5cudaq11noise_m |
| -                                 | odelERR13QuantumKernelDpRR4ARGS), |
|  [cudaq::kraus_channel::operator= |     [\[1\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4I0DpEN5cu |
|     function)](api/langua         | daq3runENSt6vectorINSt15invoke_re |
| ges/cpp_api.html#_CPPv4N5cudaq13k | sult_tINSt7decay_tI13QuantumKerne |
| raus_channelaSERK13kraus_channel) | lEEDpNSt7decay_tI4ARGSEEEEEENSt6s |
| -   [c                            | ize_tERR13QuantumKernelDpRR4ARGS) |
| udaq::kraus_channel::operator\[\] | -   [cudaq::run_async (C++        |
|     (C++                          |     functio                       |
|     function)](api/l              | n)](api/languages/cpp_api.html#_C |
| anguages/cpp_api.html#_CPPv4N5cud | PPv4I0DpEN5cudaq9run_asyncENSt6fu |
| aq13kraus_channelixEKNSt6size_tE) | tureINSt6vectorINSt15invoke_resul |
| -                                 | t_tINSt7decay_tI13QuantumKernelEE |
| [cudaq::kraus_channel::parameters | DpNSt7decay_tI4ARGSEEEEEEEENSt6si |
|     (C++                          | ze_tENSt6size_tERN5cudaq11noise_m |
|     member)](api                  | odelERR13QuantumKernelDpRR4ARGS), |
| /languages/cpp_api.html#_CPPv4N5c |     [\[1\]](api/la                |
| udaq13kraus_channel10parametersE) | nguages/cpp_api.html#_CPPv4I0DpEN |
| -   [cu                           | 5cudaq9run_asyncENSt6futureINSt6v |
| daq::kraus_channel::probabilities | ectorINSt15invoke_result_tINSt7de |
|     (C++                          | cay_tI13QuantumKernelEEDpNSt7deca |
|     member)](api/la               | y_tI4ARGSEEEEEEEENSt6size_tENSt6s |
| nguages/cpp_api.html#_CPPv4N5cuda | ize_tERR13QuantumKernelDpRR4ARGS) |
| q13kraus_channel13probabilitiesE) | -   [cudaq::RuntimeTarget (C++    |
| -                                 |                                   |
|  [cudaq::kraus_channel::push_back | struct)](api/languages/cpp_api.ht |
|     (C++                          | ml#_CPPv4N5cudaq13RuntimeTargetE) |
|     function)](api/langua         | -   [cudaq::sample (C++           |
| ges/cpp_api.html#_CPPv4N5cudaq13k |     function)](api/languages/c    |
| raus_channel9push_backE8kraus_op) | pp_api.html#_CPPv4I0DpEN5cudaq6sa |
| -   [cudaq::kraus_channel::size   | mpleE13sample_resultRK14sample_op |
|     (C++                          | tionsRR13QuantumKernelDpRR4Args), |
|     function)                     |     [\[1\                         |
| ](api/languages/cpp_api.html#_CPP | ]](api/languages/cpp_api.html#_CP |
| v4NK5cudaq13kraus_channel4sizeEv) | Pv4I0DpEN5cudaq6sampleE13sample_r |
| -   [                             | esultRR13QuantumKernelDpRR4Args), |
| cudaq::kraus_channel::unitary_ops |     [\                            |
|     (C++                          | [2\]](api/languages/cpp_api.html# |
|     member)](api/                 | _CPPv4I0DpEN5cudaq6sampleEDaNSt6s |
| languages/cpp_api.html#_CPPv4N5cu | ize_tERR13QuantumKernelDpRR4Args) |
| daq13kraus_channel11unitary_opsE) | -   [cudaq::sample_options (C++   |
| -   [cudaq::kraus_op (C++         |     s                             |
|     struct)](api/languages/cpp_   | truct)](api/languages/cpp_api.htm |
| api.html#_CPPv4N5cudaq8kraus_opE) | l#_CPPv4N5cudaq14sample_optionsE) |
| -   [cudaq::kraus_op::adjoint     | -   [cudaq::sample_result (C++    |
|     (C++                          |                                   |
|     functi                        |  class)](api/languages/cpp_api.ht |
| on)](api/languages/cpp_api.html#_ | ml#_CPPv4N5cudaq13sample_resultE) |
| CPPv4NK5cudaq8kraus_op7adjointEv) | -   [cudaq::sample_result::append |
| -   [cudaq::kraus_op::data (C++   |     (C++                          |
|                                   |     function)](api/languages/cpp_ |
|  member)](api/languages/cpp_api.h | api.html#_CPPv4N5cudaq13sample_re |
| tml#_CPPv4N5cudaq8kraus_op4dataE) | sult6appendERK15ExecutionResultb) |
| -   [cudaq::kraus_op::kraus_op    | -   [cudaq::sample_result::begin  |
|     (C++                          |     (C++                          |
|     func                          |     function)]                    |
| tion)](api/languages/cpp_api.html | (api/languages/cpp_api.html#_CPPv |
| #_CPPv4I0EN5cudaq8kraus_op8kraus_ | 4N5cudaq13sample_result5beginEv), |
| opERRNSt16initializer_listI1TEE), |     [\[1\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|  [\[1\]](api/languages/cpp_api.ht | 4NK5cudaq13sample_result5beginEv) |
| ml#_CPPv4N5cudaq8kraus_op8kraus_o | -   [cudaq::sample_result::cbegin |
| pENSt6vectorIN5cudaq7complexEEE), |     (C++                          |
|     [\[2\]](api/l                 |     function)](                   |
| anguages/cpp_api.html#_CPPv4N5cud | api/languages/cpp_api.html#_CPPv4 |
| aq8kraus_op8kraus_opERK8kraus_op) | NK5cudaq13sample_result6cbeginEv) |
| -   [cudaq::kraus_op::nCols (C++  | -   [cudaq::sample_result::cend   |
|                                   |     (C++                          |
| member)](api/languages/cpp_api.ht |     function)                     |
| ml#_CPPv4N5cudaq8kraus_op5nColsE) | ](api/languages/cpp_api.html#_CPP |
| -   [cudaq::kraus_op::nRows (C++  | v4NK5cudaq13sample_result4cendEv) |
|                                   | -   [cudaq::sample_result::clear  |
| member)](api/languages/cpp_api.ht |     (C++                          |
| ml#_CPPv4N5cudaq8kraus_op5nRowsE) |     function)                     |
| -   [cudaq::kraus_op::operator=   | ](api/languages/cpp_api.html#_CPP |
|     (C++                          | v4N5cudaq13sample_result5clearEv) |
|     function)                     | -   [cudaq::sample_result::count  |
| ](api/languages/cpp_api.html#_CPP |     (C++                          |
| v4N5cudaq8kraus_opaSERK8kraus_op) |     function)](                   |
| -   [cudaq::kraus_op::precision   | api/languages/cpp_api.html#_CPPv4 |
|     (C++                          | NK5cudaq13sample_result5countENSt |
|     memb                          | 11string_viewEKNSt11string_viewE) |
| er)](api/languages/cpp_api.html#_ | -   [                             |
| CPPv4N5cudaq8kraus_op9precisionE) | cudaq::sample_result::deserialize |
| -   [cudaq::matrix_callback (C++  |     (C++                          |
|     c                             |     functio                       |
| lass)](api/languages/cpp_api.html | n)](api/languages/cpp_api.html#_C |
| #_CPPv4N5cudaq15matrix_callbackE) | PPv4N5cudaq13sample_result11deser |
| -   [cudaq::matrix_handler (C++   | ializeERNSt6vectorINSt6size_tEEE) |
|                                   | -   [cudaq::sample_result::dump   |
| class)](api/languages/cpp_api.htm |     (C++                          |
| l#_CPPv4N5cudaq14matrix_handlerE) |     function)](api/languag        |
| -   [cudaq::mat                   | es/cpp_api.html#_CPPv4NK5cudaq13s |
| rix_handler::commutation_behavior | ample_result4dumpERNSt7ostreamE), |
|     (C++                          |     [\[1\]                        |
|     struct)](api/languages/       | ](api/languages/cpp_api.html#_CPP |
| cpp_api.html#_CPPv4N5cudaq14matri | v4NK5cudaq13sample_result4dumpEv) |
| x_handler20commutation_behaviorE) | -   [cudaq::sample_result::end    |
| -                                 |     (C++                          |
|    [cudaq::matrix_handler::define |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     function)](a                  | Pv4N5cudaq13sample_result3endEv), |
| pi/languages/cpp_api.html#_CPPv4N |     [\[1\                         |
| 5cudaq14matrix_handler6defineENSt | ]](api/languages/cpp_api.html#_CP |
| 6stringENSt6vectorINSt7int64_tEEE | Pv4NK5cudaq13sample_result3endEv) |
| RR15matrix_callbackRKNSt13unorder | -   [                             |
| ed_mapINSt6stringENSt6stringEEE), | cudaq::sample_result::expectation |
|                                   |     (C++                          |
| [\[1\]](api/languages/cpp_api.htm |     f                             |
| l#_CPPv4N5cudaq14matrix_handler6d | unction)](api/languages/cpp_api.h |
| efineENSt6stringENSt6vectorINSt7i | tml#_CPPv4NK5cudaq13sample_result |
| nt64_tEEERR15matrix_callbackRR20d | 11expectationEKNSt11string_viewE) |
| iag_matrix_callbackRKNSt13unorder | -   [c                            |
| ed_mapINSt6stringENSt6stringEEE), | udaq::sample_result::get_marginal |
|     [\[2\]](                      |     (C++                          |
| api/languages/cpp_api.html#_CPPv4 |     function)](api/languages/cpp_ |
| N5cudaq14matrix_handler6defineENS | api.html#_CPPv4NK5cudaq13sample_r |
| t6stringENSt6vectorINSt7int64_tEE | esult12get_marginalERKNSt6vectorI |
| ERR15matrix_callbackRRNSt13unorde | NSt6size_tEEEKNSt11string_viewE), |
| red_mapINSt6stringENSt6stringEEE) |     [\[1\]](api/languages/cpp_    |
| -                                 | api.html#_CPPv4NK5cudaq13sample_r |
|   [cudaq::matrix_handler::degrees | esult12get_marginalERRKNSt6vector |
|     (C++                          | INSt6size_tEEEKNSt11string_viewE) |
|     function)](ap                 | -   [cuda                         |
| i/languages/cpp_api.html#_CPPv4NK | q::sample_result::get_total_shots |
| 5cudaq14matrix_handler7degreesEv) |     (C++                          |
| -                                 |     function)](api/langua         |
|  [cudaq::matrix_handler::displace | ges/cpp_api.html#_CPPv4NK5cudaq13 |
|     (C++                          | sample_result15get_total_shotsEv) |
|     function)](api/language       | -   [cuda                         |
| s/cpp_api.html#_CPPv4N5cudaq14mat | q::sample_result::has_even_parity |
| rix_handler8displaceENSt6size_tE) |     (C++                          |
| -   [cudaq::matrix                |     fun                           |
| _handler::get_expected_dimensions | ction)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4N5cudaq13sample_result15h |
|                                   | as_even_parityENSt11string_viewE) |
|    function)](api/languages/cpp_a | -   [cuda                         |
| pi.html#_CPPv4NK5cudaq14matrix_ha | q::sample_result::has_expectation |
| ndler23get_expected_dimensionsEv) |     (C++                          |
| -   [cudaq::matrix_ha             |     funct                         |
| ndler::get_parameter_descriptions | ion)](api/languages/cpp_api.html# |
|     (C++                          | _CPPv4NK5cudaq13sample_result15ha |
|                                   | s_expectationEKNSt11string_viewE) |
| function)](api/languages/cpp_api. | -   [cu                           |
| html#_CPPv4NK5cudaq14matrix_handl | daq::sample_result::most_probable |
| er26get_parameter_descriptionsEv) |     (C++                          |
| -   [c                            |     fun                           |
| udaq::matrix_handler::instantiate | ction)](api/languages/cpp_api.htm |
|     (C++                          | l#_CPPv4NK5cudaq13sample_result13 |
|     function)](a                  | most_probableEKNSt11string_viewE) |
| pi/languages/cpp_api.html#_CPPv4N | -                                 |
| 5cudaq14matrix_handler11instantia | [cudaq::sample_result::operator+= |
| teENSt6stringERKNSt6vectorINSt6si |     (C++                          |
| ze_tEEERK20commutation_behavior), |     function)](api/langua         |
|     [\[1\]](                      | ges/cpp_api.html#_CPPv4N5cudaq13s |
| api/languages/cpp_api.html#_CPPv4 | ample_resultpLERK13sample_result) |
| N5cudaq14matrix_handler11instanti | -                                 |
| ateENSt6stringERRNSt6vectorINSt6s |  [cudaq::sample_result::operator= |
| ize_tEEERK20commutation_behavior) |     (C++                          |
| -   [cuda                         |     function)](api/langua         |
| q::matrix_handler::matrix_handler | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     (C++                          | ample_resultaSERR13sample_result) |
|     function)](api/languag        | -                                 |
| es/cpp_api.html#_CPPv4I0_NSt11ena | [cudaq::sample_result::operator== |
| ble_if_tINSt12is_base_of_vI16oper |     (C++                          |
| ator_handler1TEEbEEEN5cudaq14matr |     function)](api/languag        |
| ix_handler14matrix_handlerERK1T), | es/cpp_api.html#_CPPv4NK5cudaq13s |
|     [\[1\]](ap                    | ample_resulteqERK13sample_result) |
| i/languages/cpp_api.html#_CPPv4I0 | -   [                             |
| _NSt11enable_if_tINSt12is_base_of | cudaq::sample_result::probability |
| _vI16operator_handler1TEEbEEEN5cu |     (C++                          |
| daq14matrix_handler14matrix_handl |     function)](api/lan            |
| erERK1TRK20commutation_behavior), | guages/cpp_api.html#_CPPv4NK5cuda |
|     [\[2\]](api/languages/cpp_ap  | q13sample_result11probabilityENSt |
| i.html#_CPPv4N5cudaq14matrix_hand | 11string_viewEKNSt11string_viewE) |
| ler14matrix_handlerENSt6size_tE), | -   [cud                          |
|     [\[3\]](api/                  | aq::sample_result::register_names |
| languages/cpp_api.html#_CPPv4N5cu |     (C++                          |
| daq14matrix_handler14matrix_handl |     function)](api/langu          |
| erENSt6stringERKNSt6vectorINSt6si | ages/cpp_api.html#_CPPv4NK5cudaq1 |
| ze_tEEERK20commutation_behavior), | 3sample_result14register_namesEv) |
|     [\[4\]](api/                  | -                                 |
| languages/cpp_api.html#_CPPv4N5cu |    [cudaq::sample_result::reorder |
| daq14matrix_handler14matrix_handl |     (C++                          |
| erENSt6stringERRNSt6vectorINSt6si |     function)](api/langua         |
| ze_tEEERK20commutation_behavior), | ges/cpp_api.html#_CPPv4N5cudaq13s |
|     [\                            | ample_result7reorderERKNSt6vector |
| [5\]](api/languages/cpp_api.html# | INSt6size_tEEEKNSt11string_viewE) |
| _CPPv4N5cudaq14matrix_handler14ma | -   [cu                           |
| trix_handlerERK14matrix_handler), | daq::sample_result::sample_result |
|     [                             |     (C++                          |
| \[6\]](api/languages/cpp_api.html |     func                          |
| #_CPPv4N5cudaq14matrix_handler14m | tion)](api/languages/cpp_api.html |
| atrix_handlerERR14matrix_handler) | #_CPPv4N5cudaq13sample_result13sa |
| -                                 | mple_resultERK15ExecutionResult), |
|  [cudaq::matrix_handler::momentum |     [\[1\]](api/la                |
|     (C++                          | nguages/cpp_api.html#_CPPv4N5cuda |
|     function)](api/language       | q13sample_result13sample_resultER |
| s/cpp_api.html#_CPPv4N5cudaq14mat | KNSt6vectorI15ExecutionResultEE), |
| rix_handler8momentumENSt6size_tE) |                                   |
| -                                 |  [\[2\]](api/languages/cpp_api.ht |
|    [cudaq::matrix_handler::number | ml#_CPPv4N5cudaq13sample_result13 |
|     (C++                          | sample_resultERR13sample_result), |
|     function)](api/langua         |     [                             |
| ges/cpp_api.html#_CPPv4N5cudaq14m | \[3\]](api/languages/cpp_api.html |
| atrix_handler6numberENSt6size_tE) | #_CPPv4N5cudaq13sample_result13sa |
| -                                 | mple_resultERR15ExecutionResult), |
| [cudaq::matrix_handler::operator= |     [\[4\]](api/lan               |
|     (C++                          | guages/cpp_api.html#_CPPv4N5cudaq |
|     fun                           | 13sample_result13sample_resultEdR |
| ction)](api/languages/cpp_api.htm | KNSt6vectorI15ExecutionResultEE), |
| l#_CPPv4I0_NSt11enable_if_tIXaant |     [\[5\]](api/lan               |
| NSt7is_sameI1T14matrix_handlerE5v | guages/cpp_api.html#_CPPv4N5cudaq |
| alueENSt12is_base_of_vI16operator | 13sample_result13sample_resultEv) |
| _handler1TEEEbEEEN5cudaq14matrix_ | -                                 |
| handleraSER14matrix_handlerRK1T), |  [cudaq::sample_result::serialize |
|     [\[1\]](api/languages         |     (C++                          |
| /cpp_api.html#_CPPv4N5cudaq14matr |     function)](api                |
| ix_handleraSERK14matrix_handler), | /languages/cpp_api.html#_CPPv4NK5 |
|     [\[2\]](api/language          | cudaq13sample_result9serializeEv) |
| s/cpp_api.html#_CPPv4N5cudaq14mat | -   [cudaq::sample_result::size   |
| rix_handleraSERR14matrix_handler) |     (C++                          |
| -   [                             |     function)](api/languages/c    |
| cudaq::matrix_handler::operator== | pp_api.html#_CPPv4NK5cudaq13sampl |
|     (C++                          | e_result4sizeEKNSt11string_viewE) |
|     function)](api/languages      | -   [cudaq::sample_result::to_map |
| /cpp_api.html#_CPPv4NK5cudaq14mat |     (C++                          |
| rix_handlereqERK14matrix_handler) |     function)](api/languages/cpp  |
| -                                 | _api.html#_CPPv4NK5cudaq13sample_ |
|    [cudaq::matrix_handler::parity | result6to_mapEKNSt11string_viewE) |
|     (C++                          | -   [cuda                         |
|     function)](api/langua         | q::sample_result::\~sample_result |
| ges/cpp_api.html#_CPPv4N5cudaq14m |     (C++                          |
| atrix_handler6parityENSt6size_tE) |     funct                         |
| -                                 | ion)](api/languages/cpp_api.html# |
|  [cudaq::matrix_handler::position | _CPPv4N5cudaq13sample_resultD0Ev) |
|     (C++                          | -   [cudaq::scalar_callback (C++  |
|     function)](api/language       |     c                             |
| s/cpp_api.html#_CPPv4N5cudaq14mat | lass)](api/languages/cpp_api.html |
| rix_handler8positionENSt6size_tE) | #_CPPv4N5cudaq15scalar_callbackE) |
| -   [cudaq::                      | -   [c                            |
| matrix_handler::remove_definition | udaq::scalar_callback::operator() |
|     (C++                          |     (C++                          |
|     fu                            |     function)](api/language       |
| nction)](api/languages/cpp_api.ht | s/cpp_api.html#_CPPv4NK5cudaq15sc |
| ml#_CPPv4N5cudaq14matrix_handler1 | alar_callbackclERKNSt13unordered_ |
| 7remove_definitionERKNSt6stringE) | mapINSt6stringENSt7complexIdEEEE) |
| -                                 | -   [                             |
|   [cudaq::matrix_handler::squeeze | cudaq::scalar_callback::operator= |
|     (C++                          |     (C++                          |
|     function)](api/languag        |     function)](api/languages/c    |
| es/cpp_api.html#_CPPv4N5cudaq14ma | pp_api.html#_CPPv4N5cudaq15scalar |
| trix_handler7squeezeENSt6size_tE) | _callbackaSERK15scalar_callback), |
| -   [cudaq::m                     |     [\[1\]](api/languages/        |
| atrix_handler::to_diagonal_matrix | cpp_api.html#_CPPv4N5cudaq15scala |
|     (C++                          | r_callbackaSERR15scalar_callback) |
|     function)](api/lang           | -   [cudaq:                       |
| uages/cpp_api.html#_CPPv4NK5cudaq | :scalar_callback::scalar_callback |
| 14matrix_handler18to_diagonal_mat |     (C++                          |
| rixERNSt13unordered_mapINSt6size_ |     function)](api/languag        |
| tENSt7int64_tEEERKNSt13unordered_ | es/cpp_api.html#_CPPv4I0_NSt11ena |
| mapINSt6stringENSt7complexIdEEEE) | ble_if_tINSt16is_invocable_r_vINS |
| -                                 | t7complexIdEE8CallableRKNSt13unor |
| [cudaq::matrix_handler::to_matrix | dered_mapINSt6stringENSt7complexI |
|     (C++                          | dEEEEEEbEEEN5cudaq15scalar_callba |
|     function)                     | ck15scalar_callbackERR8Callable), |
| ](api/languages/cpp_api.html#_CPP |     [\[1\                         |
| v4NK5cudaq14matrix_handler9to_mat | ]](api/languages/cpp_api.html#_CP |
| rixERNSt13unordered_mapINSt6size_ | Pv4N5cudaq15scalar_callback15scal |
| tENSt7int64_tEEERKNSt13unordered_ | ar_callbackERK15scalar_callback), |
| mapINSt6stringENSt7complexIdEEEE) |     [\[2                          |
| -                                 | \]](api/languages/cpp_api.html#_C |
| [cudaq::matrix_handler::to_string | PPv4N5cudaq15scalar_callback15sca |
|     (C++                          | lar_callbackERR15scalar_callback) |
|     function)](api/               | -   [cudaq::scalar_operator (C++  |
| languages/cpp_api.html#_CPPv4NK5c |     c                             |
| udaq14matrix_handler9to_stringEb) | lass)](api/languages/cpp_api.html |
| -                                 | #_CPPv4N5cudaq15scalar_operatorE) |
| [cudaq::matrix_handler::unique_id | -                                 |
|     (C++                          | [cudaq::scalar_operator::evaluate |
|     function)](api/               |     (C++                          |
| languages/cpp_api.html#_CPPv4NK5c |                                   |
| udaq14matrix_handler9unique_idEv) |    function)](api/languages/cpp_a |
| -   [cudaq:                       | pi.html#_CPPv4NK5cudaq15scalar_op |
| :matrix_handler::\~matrix_handler | erator8evaluateERKNSt13unordered_ |
|     (C++                          | mapINSt6stringENSt7complexIdEEEE) |
|     functi                        | -   [cudaq::scalar_ope            |
| on)](api/languages/cpp_api.html#_ | rator::get_parameter_descriptions |
| CPPv4N5cudaq14matrix_handlerD0Ev) |     (C++                          |
| -   [cudaq::matrix_op (C++        |     f                             |
|     type)](api/languages/cpp_a    | unction)](api/languages/cpp_api.h |
| pi.html#_CPPv4N5cudaq9matrix_opE) | tml#_CPPv4NK5cudaq15scalar_operat |
| -   [cudaq::matrix_op_term (C++   | or26get_parameter_descriptionsEv) |
|                                   | -   [cu                           |
|  type)](api/languages/cpp_api.htm | daq::scalar_operator::is_constant |
| l#_CPPv4N5cudaq14matrix_op_termE) |     (C++                          |
| -                                 |     function)](api/lang           |
|    [cudaq::mdiag_operator_handler | uages/cpp_api.html#_CPPv4NK5cudaq |
|     (C++                          | 15scalar_operator11is_constantEv) |
|     class)](                      | -   [c                            |
| api/languages/cpp_api.html#_CPPv4 | udaq::scalar_operator::operator\* |
| N5cudaq22mdiag_operator_handlerE) |     (C++                          |
| -   [cudaq::mpi (C++              |     function                      |
|     type)](api/languages          | )](api/languages/cpp_api.html#_CP |
| /cpp_api.html#_CPPv4N5cudaq3mpiE) | Pv4N5cudaq15scalar_operatormlENSt |
| -   [cudaq::mpi::all_gather (C++  | 7complexIdEERK15scalar_operator), |
|     fu                            |     [\[1\                         |
| nction)](api/languages/cpp_api.ht | ]](api/languages/cpp_api.html#_CP |
| ml#_CPPv4N5cudaq3mpi10all_gatherE | Pv4N5cudaq15scalar_operatormlENSt |
| RNSt6vectorIdEERKNSt6vectorIdEE), | 7complexIdEERR15scalar_operator), |
|                                   |     [\[2\]](api/languages/cp      |
|   [\[1\]](api/languages/cpp_api.h | p_api.html#_CPPv4N5cudaq15scalar_ |
| tml#_CPPv4N5cudaq3mpi10all_gather | operatormlEdRK15scalar_operator), |
| ERNSt6vectorIiEERKNSt6vectorIiEE) |     [\[3\]](api/languages/cp      |
| -   [cudaq::mpi::all_reduce (C++  | p_api.html#_CPPv4N5cudaq15scalar_ |
|                                   | operatormlEdRR15scalar_operator), |
|  function)](api/languages/cpp_api |     [\[4\]](api/languages         |
| .html#_CPPv4I00EN5cudaq3mpi10all_ | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| reduceE1TRK1TRK14BinaryFunction), | alar_operatormlENSt7complexIdEE), |
|     [\[1\]](api/langu             |     [\[5\]](api/languages/cpp     |
| ages/cpp_api.html#_CPPv4I00EN5cud | _api.html#_CPPv4NKR5cudaq15scalar |
| aq3mpi10all_reduceE1TRK1TRK4Func) | _operatormlERK15scalar_operator), |
| -   [cudaq::mpi::broadcast (C++   |     [\[6\]]                       |
|     function)](api/               | (api/languages/cpp_api.html#_CPPv |
| languages/cpp_api.html#_CPPv4N5cu | 4NKR5cudaq15scalar_operatormlEd), |
| daq3mpi9broadcastERNSt6stringEi), |     [\[7\]](api/language          |
|     [\[1\]](api/la                | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| nguages/cpp_api.html#_CPPv4N5cuda | alar_operatormlENSt7complexIdEE), |
| q3mpi9broadcastERNSt6vectorIdEEi) |     [\[8\]](api/languages/cp      |
| -   [cudaq::mpi::finalize (C++    | p_api.html#_CPPv4NO5cudaq15scalar |
|     f                             | _operatormlERK15scalar_operator), |
| unction)](api/languages/cpp_api.h |     [\[9\                         |
| tml#_CPPv4N5cudaq3mpi8finalizeEv) | ]](api/languages/cpp_api.html#_CP |
| -   [cudaq::mpi::initialize (C++  | Pv4NO5cudaq15scalar_operatormlEd) |
|     function                      | -   [cu                           |
| )](api/languages/cpp_api.html#_CP | daq::scalar_operator::operator\*= |
| Pv4N5cudaq3mpi10initializeEiPPc), |     (C++                          |
|     [                             |     function)](api/languag        |
| \[1\]](api/languages/cpp_api.html | es/cpp_api.html#_CPPv4N5cudaq15sc |
| #_CPPv4N5cudaq3mpi10initializeEv) | alar_operatormLENSt7complexIdEE), |
| -   [cudaq::mpi::is_initialized   |     [\[1\]](api/languages/c       |
|     (C++                          | pp_api.html#_CPPv4N5cudaq15scalar |
|     function                      | _operatormLERK15scalar_operator), |
| )](api/languages/cpp_api.html#_CP |     [\[2                          |
| Pv4N5cudaq3mpi14is_initializedEv) | \]](api/languages/cpp_api.html#_C |
| -   [cudaq::mpi::num_ranks (C++   | PPv4N5cudaq15scalar_operatormLEd) |
|     fu                            | -   [                             |
| nction)](api/languages/cpp_api.ht | cudaq::scalar_operator::operator+ |
| ml#_CPPv4N5cudaq3mpi9num_ranksEv) |     (C++                          |
| -   [cudaq::mpi::rank (C++        |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|    function)](api/languages/cpp_a | Pv4N5cudaq15scalar_operatorplENSt |
| pi.html#_CPPv4N5cudaq3mpi4rankEv) | 7complexIdEERK15scalar_operator), |
| -   [cudaq::noise_model (C++      |     [\[1\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
|    class)](api/languages/cpp_api. | Pv4N5cudaq15scalar_operatorplENSt |
| html#_CPPv4N5cudaq11noise_modelE) | 7complexIdEERR15scalar_operator), |
| -   [cudaq::n                     |     [\[2\]](api/languages/cp      |
| oise_model::add_all_qubit_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatorplEdRK15scalar_operator), |
|     function)](api                |     [\[3\]](api/languages/cp      |
| /languages/cpp_api.html#_CPPv4IDp | p_api.html#_CPPv4N5cudaq15scalar_ |
| EN5cudaq11noise_model21add_all_qu | operatorplEdRR15scalar_operator), |
| bit_channelEvRK13kraus_channeli), |     [\[4\]](api/languages         |
|     [\[1\]](api/langua            | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| ges/cpp_api.html#_CPPv4N5cudaq11n | alar_operatorplENSt7complexIdEE), |
| oise_model21add_all_qubit_channel |     [\[5\]](api/languages/cpp     |
| ERKNSt6stringERK13kraus_channeli) | _api.html#_CPPv4NKR5cudaq15scalar |
| -                                 | _operatorplERK15scalar_operator), |
|  [cudaq::noise_model::add_channel |     [\[6\]]                       |
|     (C++                          | (api/languages/cpp_api.html#_CPPv |
|     funct                         | 4NKR5cudaq15scalar_operatorplEd), |
| ion)](api/languages/cpp_api.html# |     [\[7\]]                       |
| _CPPv4IDpEN5cudaq11noise_model11a | (api/languages/cpp_api.html#_CPPv |
| dd_channelEvRK15PredicateFuncTy), | 4NKR5cudaq15scalar_operatorplEv), |
|     [\[1\]](api/languages/cpp_    |     [\[8\]](api/language          |
| api.html#_CPPv4IDpEN5cudaq11noise | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| _model11add_channelEvRKNSt6vector | alar_operatorplENSt7complexIdEE), |
| INSt6size_tEEERK13kraus_channel), |     [\[9\]](api/languages/cp      |
|     [\[2\]](ap                    | p_api.html#_CPPv4NO5cudaq15scalar |
| i/languages/cpp_api.html#_CPPv4N5 | _operatorplERK15scalar_operator), |
| cudaq11noise_model11add_channelER |     [\[10\]                       |
| KNSt6stringERK15PredicateFuncTy), | ](api/languages/cpp_api.html#_CPP |
|                                   | v4NO5cudaq15scalar_operatorplEd), |
| [\[3\]](api/languages/cpp_api.htm |     [\[11\                        |
| l#_CPPv4N5cudaq11noise_model11add | ]](api/languages/cpp_api.html#_CP |
| _channelERKNSt6stringERKNSt6vecto | Pv4NO5cudaq15scalar_operatorplEv) |
| rINSt6size_tEEERK13kraus_channel) | -   [c                            |
| -   [cudaq::noise_model::empty    | udaq::scalar_operator::operator+= |
|     (C++                          |     (C++                          |
|     function                      |     function)](api/languag        |
| )](api/languages/cpp_api.html#_CP | es/cpp_api.html#_CPPv4N5cudaq15sc |
| Pv4NK5cudaq11noise_model5emptyEv) | alar_operatorpLENSt7complexIdEE), |
| -                                 |     [\[1\]](api/languages/c       |
| [cudaq::noise_model::get_channels | pp_api.html#_CPPv4N5cudaq15scalar |
|     (C++                          | _operatorpLERK15scalar_operator), |
|     function)](api/l              |     [\[2                          |
| anguages/cpp_api.html#_CPPv4I0ENK | \]](api/languages/cpp_api.html#_C |
| 5cudaq11noise_model12get_channels | PPv4N5cudaq15scalar_operatorpLEd) |
| ENSt6vectorI13kraus_channelEERKNS | -   [                             |
| t6vectorINSt6size_tEEERKNSt6vecto | cudaq::scalar_operator::operator- |
| rINSt6size_tEEERKNSt6vectorIdEE), |     (C++                          |
|     [\[1\]](api/languages/cpp_a   |     function                      |
| pi.html#_CPPv4NK5cudaq11noise_mod | )](api/languages/cpp_api.html#_CP |
| el12get_channelsERKNSt6stringERKN | Pv4N5cudaq15scalar_operatormiENSt |
| St6vectorINSt6size_tEEERKNSt6vect | 7complexIdEERK15scalar_operator), |
| orINSt6size_tEEERKNSt6vectorIdEE) |     [\[1\                         |
| -                                 | ]](api/languages/cpp_api.html#_CP |
|  [cudaq::noise_model::noise_model | Pv4N5cudaq15scalar_operatormiENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     function)](api                |     [\[2\]](api/languages/cp      |
| /languages/cpp_api.html#_CPPv4N5c | p_api.html#_CPPv4N5cudaq15scalar_ |
| udaq11noise_model11noise_modelEv) | operatormiEdRK15scalar_operator), |
| -   [cu                           |     [\[3\]](api/languages/cp      |
| daq::noise_model::PredicateFuncTy | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatormiEdRR15scalar_operator), |
|     type)](api/la                 |     [\[4\]](api/languages         |
| nguages/cpp_api.html#_CPPv4N5cuda | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| q11noise_model15PredicateFuncTyE) | alar_operatormiENSt7complexIdEE), |
| -   [cud                          |     [\[5\]](api/languages/cpp     |
| aq::noise_model::register_channel | _api.html#_CPPv4NKR5cudaq15scalar |
|     (C++                          | _operatormiERK15scalar_operator), |
|     function)](api/languages      |     [\[6\]]                       |
| /cpp_api.html#_CPPv4I00EN5cudaq11 | (api/languages/cpp_api.html#_CPPv |
| noise_model16register_channelEvv) | 4NKR5cudaq15scalar_operatormiEd), |
| -   [cudaq::                      |     [\[7\]]                       |
| noise_model::requires_constructor | (api/languages/cpp_api.html#_CPPv |
|     (C++                          | 4NKR5cudaq15scalar_operatormiEv), |
|     type)](api/languages/cp       |     [\[8\]](api/language          |
| p_api.html#_CPPv4I0DpEN5cudaq11no | s/cpp_api.html#_CPPv4NO5cudaq15sc |
| ise_model20requires_constructorE) | alar_operatormiENSt7complexIdEE), |
| -   [cudaq::noise_model_type (C++ |     [\[9\]](api/languages/cp      |
|     e                             | p_api.html#_CPPv4NO5cudaq15scalar |
| num)](api/languages/cpp_api.html# | _operatormiERK15scalar_operator), |
| _CPPv4N5cudaq16noise_model_typeE) |     [\[10\]                       |
| -   [cudaq::no                    | ](api/languages/cpp_api.html#_CPP |
| ise_model_type::amplitude_damping | v4NO5cudaq15scalar_operatormiEd), |
|     (C++                          |     [\[11\                        |
|     enumerator)](api/languages    | ]](api/languages/cpp_api.html#_CP |
| /cpp_api.html#_CPPv4N5cudaq16nois | Pv4NO5cudaq15scalar_operatormiEv) |
| e_model_type17amplitude_dampingE) | -   [c                            |
| -   [cudaq::noise_mode            | udaq::scalar_operator::operator-= |
| l_type::amplitude_damping_channel |     (C++                          |
|     (C++                          |     function)](api/languag        |
|     e                             | es/cpp_api.html#_CPPv4N5cudaq15sc |
| numerator)](api/languages/cpp_api | alar_operatormIENSt7complexIdEE), |
| .html#_CPPv4N5cudaq16noise_model_ |     [\[1\]](api/languages/c       |
| type25amplitude_damping_channelE) | pp_api.html#_CPPv4N5cudaq15scalar |
| -   [cudaq::n                     | _operatormIERK15scalar_operator), |
| oise_model_type::bit_flip_channel |     [\[2                          |
|     (C++                          | \]](api/languages/cpp_api.html#_C |
|     enumerator)](api/language     | PPv4N5cudaq15scalar_operatormIEd) |
| s/cpp_api.html#_CPPv4N5cudaq16noi | -   [                             |
| se_model_type16bit_flip_channelE) | cudaq::scalar_operator::operator/ |
| -   [cudaq::                      |     (C++                          |
| noise_model_type::depolarization1 |     function                      |
|     (C++                          | )](api/languages/cpp_api.html#_CP |
|     enumerator)](api/languag      | Pv4N5cudaq15scalar_operatordvENSt |
| es/cpp_api.html#_CPPv4N5cudaq16no | 7complexIdEERK15scalar_operator), |
| ise_model_type15depolarization1E) |     [\[1\                         |
| -   [cudaq::                      | ]](api/languages/cpp_api.html#_CP |
| noise_model_type::depolarization2 | Pv4N5cudaq15scalar_operatordvENSt |
|     (C++                          | 7complexIdEERR15scalar_operator), |
|     enumerator)](api/languag      |     [\[2\]](api/languages/cp      |
| es/cpp_api.html#_CPPv4N5cudaq16no | p_api.html#_CPPv4N5cudaq15scalar_ |
| ise_model_type15depolarization2E) | operatordvEdRK15scalar_operator), |
| -   [cudaq::noise_m               |     [\[3\]](api/languages/cp      |
| odel_type::depolarization_channel | p_api.html#_CPPv4N5cudaq15scalar_ |
|     (C++                          | operatordvEdRR15scalar_operator), |
|                                   |     [\[4\]](api/languages         |
|   enumerator)](api/languages/cpp_ | /cpp_api.html#_CPPv4NKR5cudaq15sc |
| api.html#_CPPv4N5cudaq16noise_mod | alar_operatordvENSt7complexIdEE), |
| el_type22depolarization_channelE) |     [\[5\]](api/languages/cpp     |
| -                                 | _api.html#_CPPv4NKR5cudaq15scalar |
|  [cudaq::noise_model_type::pauli1 | _operatordvERK15scalar_operator), |
|     (C++                          |     [\[6\]]                       |
|     enumerator)](a                | (api/languages/cpp_api.html#_CPPv |
| pi/languages/cpp_api.html#_CPPv4N | 4NKR5cudaq15scalar_operatordvEd), |
| 5cudaq16noise_model_type6pauli1E) |     [\[7\]](api/language          |
| -                                 | s/cpp_api.html#_CPPv4NO5cudaq15sc |
|  [cudaq::noise_model_type::pauli2 | alar_operatordvENSt7complexIdEE), |
|     (C++                          |     [\[8\]](api/languages/cp      |
|     enumerator)](a                | p_api.html#_CPPv4NO5cudaq15scalar |
| pi/languages/cpp_api.html#_CPPv4N | _operatordvERK15scalar_operator), |
| 5cudaq16noise_model_type6pauli2E) |     [\[9\                         |
| -   [cudaq                        | ]](api/languages/cpp_api.html#_CP |
| ::noise_model_type::phase_damping | Pv4NO5cudaq15scalar_operatordvEd) |
|     (C++                          | -   [c                            |
|     enumerator)](api/langu        | udaq::scalar_operator::operator/= |
| ages/cpp_api.html#_CPPv4N5cudaq16 |     (C++                          |
| noise_model_type13phase_dampingE) |     function)](api/languag        |
| -   [cudaq::noi                   | es/cpp_api.html#_CPPv4N5cudaq15sc |
| se_model_type::phase_flip_channel | alar_operatordVENSt7complexIdEE), |
|     (C++                          |     [\[1\]](api/languages/c       |
|     enumerator)](api/languages/   | pp_api.html#_CPPv4N5cudaq15scalar |
| cpp_api.html#_CPPv4N5cudaq16noise | _operatordVERK15scalar_operator), |
| _model_type18phase_flip_channelE) |     [\[2                          |
| -                                 | \]](api/languages/cpp_api.html#_C |
| [cudaq::noise_model_type::unknown | PPv4N5cudaq15scalar_operatordVEd) |
|     (C++                          | -   [                             |
|     enumerator)](ap               | cudaq::scalar_operator::operator= |
| i/languages/cpp_api.html#_CPPv4N5 |     (C++                          |
| cudaq16noise_model_type7unknownE) |     function)](api/languages/c    |
| -                                 | pp_api.html#_CPPv4N5cudaq15scalar |
| [cudaq::noise_model_type::x_error | _operatoraSERK15scalar_operator), |
|     (C++                          |     [\[1\]](api/languages/        |
|     enumerator)](ap               | cpp_api.html#_CPPv4N5cudaq15scala |
| i/languages/cpp_api.html#_CPPv4N5 | r_operatoraSERR15scalar_operator) |
| cudaq16noise_model_type7x_errorE) | -   [c                            |
| -                                 | udaq::scalar_operator::operator== |
| [cudaq::noise_model_type::y_error |     (C++                          |
|     (C++                          |     function)](api/languages/c    |
|     enumerator)](ap               | pp_api.html#_CPPv4NK5cudaq15scala |
| i/languages/cpp_api.html#_CPPv4N5 | r_operatoreqERK15scalar_operator) |
| cudaq16noise_model_type7y_errorE) | -   [cudaq:                       |
| -                                 | :scalar_operator::scalar_operator |
| [cudaq::noise_model_type::z_error |     (C++                          |
|     (C++                          |     func                          |
|     enumerator)](ap               | tion)](api/languages/cpp_api.html |
| i/languages/cpp_api.html#_CPPv4N5 | #_CPPv4N5cudaq15scalar_operator15 |
| cudaq16noise_model_type7z_errorE) | scalar_operatorENSt7complexIdEE), |
| -   [cudaq::num_available_gpus    |     [\[1\]](api/langu             |
|     (C++                          | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     function                      | scalar_operator15scalar_operatorE |
| )](api/languages/cpp_api.html#_CP | RK15scalar_callbackRRNSt13unorder |
| Pv4N5cudaq18num_available_gpusEv) | ed_mapINSt6stringENSt6stringEEE), |
| -   [cudaq::observe (C++          |     [\[2\                         |
|     function)]                    | ]](api/languages/cpp_api.html#_CP |
| (api/languages/cpp_api.html#_CPPv | Pv4N5cudaq15scalar_operator15scal |
| 4I00DpEN5cudaq7observeENSt6vector | ar_operatorERK15scalar_operator), |
| I14observe_resultEERR13QuantumKer |     [\[3\]](api/langu             |
| nelRK15SpinOpContainerDpRR4Args), | ages/cpp_api.html#_CPPv4N5cudaq15 |
|     [\[1\]](api/languages/cpp_ap  | scalar_operator15scalar_operatorE |
| i.html#_CPPv4I0DpEN5cudaq7observe | RR15scalar_callbackRRNSt13unorder |
| E14observe_resultNSt6size_tERR13Q | ed_mapINSt6stringENSt6stringEEE), |
| uantumKernelRK7spin_opDpRR4Args), |     [\[4\                         |
|     [\[                           | ]](api/languages/cpp_api.html#_CP |
| 2\]](api/languages/cpp_api.html#_ | Pv4N5cudaq15scalar_operator15scal |
| CPPv4I0DpEN5cudaq7observeE14obser | ar_operatorERR15scalar_operator), |
| ve_resultRK15observe_optionsRR13Q |     [\[5\]](api/language          |
| uantumKernelRK7spin_opDpRR4Args), | s/cpp_api.html#_CPPv4N5cudaq15sca |
|     [\[3\]](api/lang              | lar_operator15scalar_operatorEd), |
| uages/cpp_api.html#_CPPv4I0DpEN5c |     [\[6\]](api/languag           |
| udaq7observeE14observe_resultRR13 | es/cpp_api.html#_CPPv4N5cudaq15sc |
| QuantumKernelRK7spin_opDpRR4Args) | alar_operator15scalar_operatorEv) |
| -   [cudaq::observe_options (C++  | -   [                             |
|     st                            | cudaq::scalar_operator::to_matrix |
| ruct)](api/languages/cpp_api.html |     (C++                          |
| #_CPPv4N5cudaq15observe_optionsE) |                                   |
| -   [cudaq::observe_result (C++   |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq15scalar_ope |
| class)](api/languages/cpp_api.htm | rator9to_matrixERKNSt13unordered_ |
| l#_CPPv4N5cudaq14observe_resultE) | mapINSt6stringENSt7complexIdEEEE) |
| -                                 | -   [                             |
|    [cudaq::observe_result::counts | cudaq::scalar_operator::to_string |
|     (C++                          |     (C++                          |
|     function)](api/languages/c    |     function)](api/l              |
| pp_api.html#_CPPv4N5cudaq14observ | anguages/cpp_api.html#_CPPv4NK5cu |
| e_result6countsERK12spin_op_term) | daq15scalar_operator9to_stringEv) |
| -   [cudaq::observe_result::dump  | -   [cudaq::s                     |
|     (C++                          | calar_operator::\~scalar_operator |
|     function)                     |     (C++                          |
| ](api/languages/cpp_api.html#_CPP |     functio                       |
| v4N5cudaq14observe_result4dumpEv) | n)](api/languages/cpp_api.html#_C |
| -   [c                            | PPv4N5cudaq15scalar_operatorD0Ev) |
| udaq::observe_result::expectation | -   [cudaq::set_noise (C++        |
|     (C++                          |     function)](api/langu          |
|                                   | ages/cpp_api.html#_CPPv4N5cudaq9s |
| function)](api/languages/cpp_api. | et_noiseERKN5cudaq11noise_modelE) |
| html#_CPPv4N5cudaq14observe_resul | -   [cudaq::set_random_seed (C++  |
| t11expectationERK12spin_op_term), |     function)](api/               |
|     [\[1\]](api/la                | languages/cpp_api.html#_CPPv4N5cu |
| nguages/cpp_api.html#_CPPv4N5cuda | daq15set_random_seedENSt6size_tE) |
| q14observe_result11expectationEv) | -   [cudaq::simulation_precision  |
| -   [cuda                         |     (C++                          |
| q::observe_result::id_coefficient |     enum)                         |
|     (C++                          | ](api/languages/cpp_api.html#_CPP |
|     function)](api/langu          | v4N5cudaq20simulation_precisionE) |
| ages/cpp_api.html#_CPPv4N5cudaq14 | -   [                             |
| observe_result14id_coefficientEv) | cudaq::simulation_precision::fp32 |
| -   [cuda                         |     (C++                          |
| q::observe_result::observe_result |     enumerator)](api              |
|     (C++                          | /languages/cpp_api.html#_CPPv4N5c |
|                                   | udaq20simulation_precision4fp32E) |
|   function)](api/languages/cpp_ap | -   [                             |
| i.html#_CPPv4N5cudaq14observe_res | cudaq::simulation_precision::fp64 |
| ult14observe_resultEdRK7spin_op), |     (C++                          |
|     [\[1\]](a                     |     enumerator)](api              |
| pi/languages/cpp_api.html#_CPPv4N | /languages/cpp_api.html#_CPPv4N5c |
| 5cudaq14observe_result14observe_r | udaq20simulation_precision4fp64E) |
| esultEdRK7spin_op13sample_result) | -   [cudaq::SimulationState (C++  |
| -                                 |     c                             |
|  [cudaq::observe_result::operator | lass)](api/languages/cpp_api.html |
|     double (C++                   | #_CPPv4N5cudaq15SimulationStateE) |
|     functio                       | -   [                             |
| n)](api/languages/cpp_api.html#_C | cudaq::SimulationState::precision |
| PPv4N5cudaq14observe_resultcvdEv) |     (C++                          |
| -                                 |     enum)](api                    |
|  [cudaq::observe_result::raw_data | /languages/cpp_api.html#_CPPv4N5c |
|     (C++                          | udaq15SimulationState9precisionE) |
|     function)](ap                 | -   [cudaq:                       |
| i/languages/cpp_api.html#_CPPv4N5 | :SimulationState::precision::fp32 |
| cudaq14observe_result8raw_dataEv) |     (C++                          |
| -   [cudaq::operator_handler (C++ |     enumerator)](api/lang         |
|     cl                            | uages/cpp_api.html#_CPPv4N5cudaq1 |
| ass)](api/languages/cpp_api.html# | 5SimulationState9precision4fp32E) |
| _CPPv4N5cudaq16operator_handlerE) | -   [cudaq:                       |
| -   [cudaq::optimizable_function  | :SimulationState::precision::fp64 |
|     (C++                          |     (C++                          |
|     class)                        |     enumerator)](api/lang         |
| ](api/languages/cpp_api.html#_CPP | uages/cpp_api.html#_CPPv4N5cudaq1 |
| v4N5cudaq20optimizable_functionE) | 5SimulationState9precision4fp64E) |
| -   [cudaq::optimization_result   | -                                 |
|     (C++                          |   [cudaq::SimulationState::Tensor |
|     type                          |     (C++                          |
| )](api/languages/cpp_api.html#_CP |     struct)](                     |
| Pv4N5cudaq19optimization_resultE) | api/languages/cpp_api.html#_CPPv4 |
| -   [cudaq::optimizer (C++        | N5cudaq15SimulationState6TensorE) |
|     class)](api/languages/cpp_a   | -   [cudaq::spin_handler (C++     |
| pi.html#_CPPv4N5cudaq9optimizerE) |                                   |
| -   [cudaq::optimizer::optimize   |   class)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq12spin_handlerE) |
|                                   | -   [cudaq:                       |
|  function)](api/languages/cpp_api | :spin_handler::to_diagonal_matrix |
| .html#_CPPv4N5cudaq9optimizer8opt |     (C++                          |
| imizeEKiRR20optimizable_function) |     function)](api/la             |
| -   [cu                           | nguages/cpp_api.html#_CPPv4NK5cud |
| daq::optimizer::requiresGradients | aq12spin_handler18to_diagonal_mat |
|     (C++                          | rixERNSt13unordered_mapINSt6size_ |
|     function)](api/la             | tENSt7int64_tEEERKNSt13unordered_ |
| nguages/cpp_api.html#_CPPv4N5cuda | mapINSt6stringENSt7complexIdEEEE) |
| q9optimizer17requiresGradientsEv) | -                                 |
| -   [cudaq::orca (C++             |   [cudaq::spin_handler::to_matrix |
|     type)](api/languages/         |     (C++                          |
| cpp_api.html#_CPPv4N5cudaq4orcaE) |     function                      |
| -   [cudaq::orca::sample (C++     | )](api/languages/cpp_api.html#_CP |
|     function)](api/languages/c    | Pv4N5cudaq12spin_handler9to_matri |
| pp_api.html#_CPPv4N5cudaq4orca6sa | xERKNSt6stringENSt7complexIdEEb), |
| mpleERNSt6vectorINSt6size_tEEERNS |     [\[1                          |
| t6vectorINSt6size_tEEERNSt6vector | \]](api/languages/cpp_api.html#_C |
| IdEERNSt6vectorIdEEiNSt6size_tE), | PPv4NK5cudaq12spin_handler9to_mat |
|     [\[1\]]                       | rixERNSt13unordered_mapINSt6size_ |
| (api/languages/cpp_api.html#_CPPv | tENSt7int64_tEEERKNSt13unordered_ |
| 4N5cudaq4orca6sampleERNSt6vectorI | mapINSt6stringENSt7complexIdEEEE) |
| NSt6size_tEEERNSt6vectorINSt6size | -   [cuda                         |
| _tEEERNSt6vectorIdEEiNSt6size_tE) | q::spin_handler::to_sparse_matrix |
| -   [cudaq::orca::sample_async    |     (C++                          |
|     (C++                          |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
| function)](api/languages/cpp_api. | daq12spin_handler16to_sparse_matr |
| html#_CPPv4N5cudaq4orca12sample_a | ixERKNSt6stringENSt7complexIdEEb) |
| syncERNSt6vectorINSt6size_tEEERNS | -                                 |
| t6vectorINSt6size_tEEERNSt6vector |   [cudaq::spin_handler::to_string |
| IdEERNSt6vectorIdEEiNSt6size_tE), |     (C++                          |
|     [\[1\]](api/la                |     function)](ap                 |
| nguages/cpp_api.html#_CPPv4N5cuda | i/languages/cpp_api.html#_CPPv4NK |
| q4orca12sample_asyncERNSt6vectorI | 5cudaq12spin_handler9to_stringEb) |
| NSt6size_tEEERNSt6vectorINSt6size | -                                 |
| _tEEERNSt6vectorIdEEiNSt6size_tE) |   [cudaq::spin_handler::unique_id |
| -   [cudaq::OrcaRemoteRESTQPU     |     (C++                          |
|     (C++                          |     function)](ap                 |
|     cla                           | i/languages/cpp_api.html#_CPPv4NK |
| ss)](api/languages/cpp_api.html#_ | 5cudaq12spin_handler9unique_idEv) |
| CPPv4N5cudaq17OrcaRemoteRESTQPUE) | -   [cudaq::spin_op (C++          |
| -   [cudaq::pauli1 (C++           |     type)](api/languages/cpp      |
|     class)](api/languages/cp      | _api.html#_CPPv4N5cudaq7spin_opE) |
| p_api.html#_CPPv4N5cudaq6pauli1E) | -   [cudaq::spin_op_term (C++     |
| -                                 |                                   |
|    [cudaq::pauli1::num_parameters |    type)](api/languages/cpp_api.h |
|     (C++                          | tml#_CPPv4N5cudaq12spin_op_termE) |
|     member)]                      | -   [cudaq::state (C++            |
| (api/languages/cpp_api.html#_CPPv |     class)](api/languages/c       |
| 4N5cudaq6pauli114num_parametersE) | pp_api.html#_CPPv4N5cudaq5stateE) |
| -   [cudaq::pauli1::num_targets   | -   [cudaq::state::amplitude (C++ |
|     (C++                          |     function)](api/lang           |
|     membe                         | uages/cpp_api.html#_CPPv4N5cudaq5 |
| r)](api/languages/cpp_api.html#_C | state9amplitudeERKNSt6vectorIiEE) |
| PPv4N5cudaq6pauli111num_targetsE) | -   [cudaq::state::amplitudes     |
|                                   |     (C++                          |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq5state10amplitud |
|                                   | esERKNSt6vectorINSt6vectorIiEEEE) |
|                                   | -   [cudaq::state::dump (C++      |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4NK |
|                                   | 5cudaq5state4dumpERNSt7ostreamE), |
|                                   |                                   |
|                                   |    [\[1\]](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq5state4dumpEv) |
|                                   | -   [cudaq::state::from_data (C++ |
|                                   |     function)](api/la             |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q5state9from_dataERK10state_data) |
|                                   | -   [cudaq::state::get_num_qubits |
|                                   |     (C++                          |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq5state14get_num_qubitsEv) |
|                                   | -                                 |
|                                   |    [cudaq::state::get_num_tensors |
|                                   |     (C++                          |
|                                   |     function)](a                  |
|                                   | pi/languages/cpp_api.html#_CPPv4N |
|                                   | K5cudaq5state15get_num_tensorsEv) |
|                                   | -   [cudaq::state::get_precision  |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NK5cudaq5state13get_precisionEv) |
|                                   | -   [cudaq::state::get_tensor     |
|                                   |     (C++                          |
|                                   |     function)](api/la             |
|                                   | nguages/cpp_api.html#_CPPv4NK5cud |
|                                   | aq5state10get_tensorENSt6size_tE) |
|                                   | -   [cudaq::state::get_tensors    |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq5state11get_tensorsEv) |
|                                   | -   [cudaq::state::is_on_gpu (C++ |
|                                   |     funct                         |
|                                   | ion)](api/languages/cpp_api.html# |
|                                   | _CPPv4NK5cudaq5state9is_on_gpuEv) |
|                                   | -   [cudaq::state::operator()     |
|                                   |     (C++                          |
|                                   |     function)](api/lang           |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 5stateclENSt6size_tENSt6size_tE), |
|                                   |     [\[1\]](                      |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | NK5cudaq5stateclERKNSt16initializ |
|                                   | er_listINSt6size_tEEENSt6size_tE) |
|                                   | -   [cudaq::state::operator= (C++ |
|                                   |     fun                           |
|                                   | ction)](api/languages/cpp_api.htm |
|                                   | l#_CPPv4N5cudaq5stateaSERR5state) |
|                                   | -   [cudaq::state::operator\[\]   |
|                                   |     (C++                          |
|                                   |     functio                       |
|                                   | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4NK5cudaq5stateixENSt6size_tE) |
|                                   | -   [cudaq::state::overlap (C++   |
|                                   |     function)                     |
|                                   | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq5state7overlapERK5state) |
|                                   | -   [cudaq::state::state (C++     |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 5state5stateEP15SimulationState), |
|                                   |     [\[1\                         |
|                                   | ]](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq5state5stateERK5state), |
|                                   |     [\[2\]](api/languages/cpp_    |
|                                   | api.html#_CPPv4N5cudaq5state5stat |
|                                   | eERKNSt6vectorINSt7complexIdEEEE) |
|                                   | -   [cudaq::state::to_host (C++   |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | I0ENK5cudaq5state7to_hostEvPNSt7c |
|                                   | omplexI10ScalarTypeEENSt6size_tE) |
|                                   | -   [cudaq::state::\~state (C++   |
|                                   |     function)](api/languages/cpp_ |
|                                   | api.html#_CPPv4N5cudaq5stateD0Ev) |
|                                   | -   [cudaq::state_data (C++       |
|                                   |     type)](api/languages/cpp_api  |
|                                   | .html#_CPPv4N5cudaq10state_dataE) |
|                                   | -   [cudaq::sum_op (C++           |
|                                   |     class)](api/languages/cpp_a   |
|                                   | pi.html#_CPPv4I0EN5cudaq6sum_opE) |
|                                   | -   [cudaq::sum_op::begin (C++    |
|                                   |     fu                            |
|                                   | nction)](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4NK5cudaq6sum_op5beginEv) |
|                                   | -   [cudaq::sum_op::canonicalize  |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |  function)](api/languages/cpp_api |
|                                   | .html#_CPPv4N5cudaq6sum_op12canon |
|                                   | icalizeERKNSt3setINSt6size_tEEE), |
|                                   |     [\[1\]                        |
|                                   | ](api/languages/cpp_api.html#_CPP |
|                                   | v4N5cudaq6sum_op12canonicalizeEv) |
|                                   | -                                 |
|                                   |    [cudaq::sum_op::const_iterator |
|                                   |     (C++                          |
|                                   |     struct)]                      |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4N5cudaq6sum_op14const_iteratorE) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator!= |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq6sum_op14con |
|                                   | st_iteratorneERK14const_iterator) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator\* |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratormlEv) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator++ |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratorppEv) |
|                                   | -   [cudaq::su                    |
|                                   | m_op::const_iterator::operator-\> |
|                                   |     (C++                          |
|                                   |     function)](ap                 |
|                                   | i/languages/cpp_api.html#_CPPv4N5 |
|                                   | cudaq6sum_op14const_iteratorptEv) |
|                                   | -   [cudaq::s                     |
|                                   | um_op::const_iterator::operator== |
|                                   |     (C++                          |
|                                   |                                   |
|                                   |   function)](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NK5cudaq6sum_op14con |
|                                   | st_iteratoreqERK14const_iterator) |
|                                   | -   [cudaq::sum_op::degrees (C++  |
|                                   |     func                          |
|                                   | tion)](api/languages/cpp_api.html |
|                                   | #_CPPv4NK5cudaq6sum_op7degreesEv) |
|                                   | -                                 |
|                                   |  [cudaq::sum_op::distribute_terms |
|                                   |     (C++                          |
|                                   |     function)](api/languages      |
|                                   | /cpp_api.html#_CPPv4NK5cudaq6sum_ |
|                                   | op16distribute_termsENSt6size_tE) |
|                                   | -   [cudaq::sum_op::dump (C++     |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4NK5cudaq6sum_op4dumpEv) |
|                                   | -   [cudaq::sum_op::empty (C++    |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq6sum_op5emptyEv) |
|                                   | -   [cudaq::sum_op::end (C++      |
|                                   |                                   |
|                                   | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq6sum_op3endEv) |
|                                   | -   [cudaq::sum_op::identity (C++ |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq6sum_op8identityENSt6size_tE), |
|                                   |     [                             |
|                                   | \[1\]](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq6sum_op8identityEv) |
|                                   | -   [cudaq::sum_op::num_terms     |
|                                   |     (C++                          |
|                                   |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4NK5cudaq6sum_op9num_termsEv) |
|                                   | -   [cudaq::sum_op::operator\*    |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opmlE6sum_opI1TER |
|                                   | K15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[1\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opmlE6sum_opI1TER |
|                                   | K15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[2\]](api/languages         |
|                                   | /cpp_api.html#_CPPv4NK5cudaq6sum_ |
|                                   | opmlERK10product_opI9HandlerTyE), |
|                                   |     [\[3\]](api/lang              |
|                                   | uages/cpp_api.html#_CPPv4NK5cudaq |
|                                   | 6sum_opmlERK6sum_opI9HandlerTyE), |
|                                   |     [\[4\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opmlERK15scalar_operator), |
|                                   |     [\[5\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4NO5cu |
|                                   | daq6sum_opmlERK15scalar_operator) |
|                                   | -   [cudaq::sum_op::operator\*=   |
|                                   |     (C++                          |
|                                   |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opmLERK10product_opI9HandlerTyE), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_opmLERK15scalar_operator), |
|                                   |     [\[2\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_opmLERK6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator+     |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERK15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | K15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[2\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[3\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | K15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[4\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERR15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[5\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | R15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[6\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opplE6sum_opI1TERR15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[7\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opplE6sum_opI1TER |
|                                   | R15scalar_operatorRR6sum_opI1TE), |
|                                   |     [\[8\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opplERK10product_opI9HandlerTyE), |
|                                   |     [\[9\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opplERK15scalar_operator), |
|                                   |     [\[10\]](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opplERK6sum_opI9HandlerTyE), |
|                                   |     [\[11\]](api/languages/       |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|                                   |     [\[12\]](api/lan              |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opplERR15scalar_operator), |
|                                   |     [\[13\]](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opplERR6sum_opI9HandlerTyE), |
|                                   |                                   |
|                                   |   [\[14\]](api/languages/cpp_api. |
|                                   | html#_CPPv4NKR5cudaq6sum_opplEv), |
|                                   |     [\[15\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opplERK10product_opI9HandlerTyE), |
|                                   |     [\[16\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opplERK15scalar_operator), |
|                                   |     [\[17\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opplERK6sum_opI9HandlerTyE), |
|                                   |     [\[18\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opplERR10product_opI9HandlerTyE), |
|                                   |     [\[19\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opplERR15scalar_operator), |
|                                   |     [\[20\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opplERR6sum_opI9HandlerTyE), |
|                                   |     [\[21\]](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NO5cudaq6sum_opplEv) |
|                                   | -   [cudaq::sum_op::operator+=    |
|                                   |     (C++                          |
|                                   |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | oppLERK10product_opI9HandlerTyE), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_oppLERK15scalar_operator), |
|                                   |     [\[2\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 6sum_oppLERK6sum_opI9HandlerTyE), |
|                                   |     [\[3\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | oppLERR10product_opI9HandlerTyE), |
|                                   |     [\[4\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_oppLERR15scalar_operator), |
|                                   |     [\[5\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_oppLERR6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator-     |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERK15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[1\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERK15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[2\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERR15sc |
|                                   | alar_operatorRK10product_opI1TE), |
|                                   |     [\[3\]]                       |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4I0EN5cudaq6sum_opmiE6sum_opI1TER |
|                                   | R15scalar_operatorRK6sum_opI1TE), |
|                                   |     [\[4\]](api/                  |
|                                   | languages/cpp_api.html#_CPPv4I0EN |
|                                   | 5cudaq6sum_opmiE6sum_opI1TERR15sc |
|                                   | alar_operatorRR10product_opI1TE), |
|                                   |     [\[5\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opmiERK10product_opI9HandlerTyE), |
|                                   |     [\[6\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opmiERK15scalar_operator), |
|                                   |     [\[7\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opmiERK6sum_opI9HandlerTyE), |
|                                   |     [\[8\]](api/languages/        |
|                                   | cpp_api.html#_CPPv4NKR5cudaq6sum_ |
|                                   | opmiERR10product_opI9HandlerTyE), |
|                                   |     [\[9\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opmiERR15scalar_operator), |
|                                   |     [\[10\]](api/langu            |
|                                   | ages/cpp_api.html#_CPPv4NKR5cudaq |
|                                   | 6sum_opmiERR6sum_opI9HandlerTyE), |
|                                   |                                   |
|                                   |   [\[11\]](api/languages/cpp_api. |
|                                   | html#_CPPv4NKR5cudaq6sum_opmiEv), |
|                                   |     [\[12\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opmiERK10product_opI9HandlerTyE), |
|                                   |     [\[13\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opmiERK15scalar_operator), |
|                                   |     [\[14\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opmiERK6sum_opI9HandlerTyE), |
|                                   |     [\[15\]](api/languages        |
|                                   | /cpp_api.html#_CPPv4NO5cudaq6sum_ |
|                                   | opmiERR10product_opI9HandlerTyE), |
|                                   |     [\[16\]](api/la               |
|                                   | nguages/cpp_api.html#_CPPv4NO5cud |
|                                   | aq6sum_opmiERR15scalar_operator), |
|                                   |     [\[17\]](api/lang             |
|                                   | uages/cpp_api.html#_CPPv4NO5cudaq |
|                                   | 6sum_opmiERR6sum_opI9HandlerTyE), |
|                                   |     [\[18\]](api/languages/cpp_ap |
|                                   | i.html#_CPPv4NO5cudaq6sum_opmiEv) |
|                                   | -   [cudaq::sum_op::operator-=    |
|                                   |     (C++                          |
|                                   |     function)](api/language       |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opmIERK10product_opI9HandlerTyE), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_opmIERK15scalar_operator), |
|                                   |     [\[2\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 6sum_opmIERK6sum_opI9HandlerTyE), |
|                                   |     [\[3\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opmIERR10product_opI9HandlerTyE), |
|                                   |     [\[4\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_opmIERR15scalar_operator), |
|                                   |     [\[5\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_opmIERR6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator/     |
|                                   |     (C++                          |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4NKR5cud |
|                                   | aq6sum_opdvERK15scalar_operator), |
|                                   |     [\[1\]](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4NO5cu |
|                                   | daq6sum_opdvERK15scalar_operator) |
|                                   | -   [cudaq::sum_op::operator/=    |
|                                   |     (C++                          |
|                                   |     function)](api/               |
|                                   | languages/cpp_api.html#_CPPv4N5cu |
|                                   | daq6sum_opdVERK15scalar_operator) |
|                                   | -   [cudaq::sum_op::operator=     |
|                                   |     (C++                          |
|                                   |     functio                       |
|                                   | n)](api/languages/cpp_api.html#_C |
|                                   | PPv4I0_NSt11enable_if_tIXaantNSt7 |
|                                   | is_sameI1T9HandlerTyE5valueENSt16 |
|                                   | is_constructibleI9HandlerTy1TE5va |
|                                   | lueEEbEEEN5cudaq6sum_opaSER6sum_o |
|                                   | pI9HandlerTyERK10product_opI1TE), |
|                                   |                                   |
|                                   |  [\[1\]](api/languages/cpp_api.ht |
|                                   | ml#_CPPv4I0_NSt11enable_if_tIXaan |
|                                   | tNSt7is_sameI1T9HandlerTyE5valueE |
|                                   | NSt16is_constructibleI9HandlerTy1 |
|                                   | TE5valueEEbEEEN5cudaq6sum_opaSER6 |
|                                   | sum_opI9HandlerTyERK6sum_opI1TE), |
|                                   |     [\[2\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opaSERK10product_opI9HandlerTyE), |
|                                   |     [\[3\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4N5cudaq |
|                                   | 6sum_opaSERK6sum_opI9HandlerTyE), |
|                                   |     [\[4\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | opaSERR10product_opI9HandlerTyE), |
|                                   |     [\[5\]](api/la                |
|                                   | nguages/cpp_api.html#_CPPv4N5cuda |
|                                   | q6sum_opaSERR6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator==    |
|                                   |     (C++                          |
|                                   |     function)](api/lan            |
|                                   | guages/cpp_api.html#_CPPv4NK5cuda |
|                                   | q6sum_opeqERK6sum_opI9HandlerTyE) |
|                                   | -   [cudaq::sum_op::operator\[\]  |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq6sum_opixENSt6size_tE) |
|                                   | -   [cudaq::sum_op::sum_op (C++   |
|                                   |     function)](                   |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | I0_NSt11enable_if_tIXaaNSt7is_sam |
|                                   | eI9HandlerTy14matrix_handlerE5val |
|                                   | ueEaantNSt7is_sameI1T9HandlerTyE5 |
|                                   | valueENSt16is_constructibleI9Hand |
|                                   | lerTy1TE5valueEEbEEEN5cudaq6sum_o |
|                                   | p6sum_opERK6sum_opI1TERKN14matrix |
|                                   | _handler20commutation_behaviorE), |
|                                   |     [\[1\]](api/langu             |
|                                   | ages/cpp_api.html#_CPPv4I0_NSt11e |
|                                   | nable_if_tIXaantNSt7is_sameI1T9Ha |
|                                   | ndlerTyE5valueENSt16is_constructi |
|                                   | bleI9HandlerTy1TE5valueEEbEEEN5cu |
|                                   | daq6sum_op6sum_opERK6sum_opI1TE), |
|                                   |     [\[2\]](api/lan               |
|                                   | guages/cpp_api.html#_CPPv4IDp_NSt |
|                                   | 11enable_if_tIXaaNSt11conjunction |
|                                   | IDpNSt7is_sameI10product_opI9Hand |
|                                   | lerTyE4ArgsEEE5valueEsZ4ArgsEbEEE |
|                                   | N5cudaq6sum_op6sum_opEDpRR4Args), |
|                                   |     [\[3\]](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq6sum_op6su |
|                                   | m_opERK10product_opI9HandlerTyE), |
|                                   |     [\[4\]](api/language          |
|                                   | s/cpp_api.html#_CPPv4N5cudaq6sum_ |
|                                   | op6sum_opERK6sum_opI9HandlerTyE), |
|                                   |     [\[5\]](api/languag           |
|                                   | es/cpp_api.html#_CPPv4N5cudaq6sum |
|                                   | _op6sum_opERR6sum_opI9HandlerTyE) |
|                                   | -   [                             |
|                                   | cudaq::sum_op::to_diagonal_matrix |
|                                   |     (C++                          |
|                                   |     function)]                    |
|                                   | (api/languages/cpp_api.html#_CPPv |
|                                   | 4NK5cudaq6sum_op18to_diagonal_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -   [cudaq::sum_op::to_matrix     |
|                                   |     (C++                          |
|                                   |                                   |
|                                   | function)](api/languages/cpp_api. |
|                                   | html#_CPPv4NK5cudaq6sum_op9to_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -                                 |
|                                   |  [cudaq::sum_op::to_sparse_matrix |
|                                   |     (C++                          |
|                                   |     function                      |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4NK5cudaq6sum_op16to_sparse_mat |
|                                   | rixENSt13unordered_mapINSt6size_t |
|                                   | ENSt7int64_tEEERKNSt13unordered_m |
|                                   | apINSt6stringENSt7complexIdEEEEb) |
|                                   | -   [cudaq::sum_op::to_string     |
|                                   |     (C++                          |
|                                   |     functi                        |
|                                   | on)](api/languages/cpp_api.html#_ |
|                                   | CPPv4NK5cudaq6sum_op9to_stringEv) |
|                                   | -   [cudaq::sum_op::trim (C++     |
|                                   |     function)](api/l              |
|                                   | anguages/cpp_api.html#_CPPv4N5cud |
|                                   | aq6sum_op4trimEdRKNSt13unordered_ |
|                                   | mapINSt6stringENSt7complexIdEEEE) |
|                                   | -   [cudaq::sum_op::\~sum_op (C++ |
|                                   |                                   |
|                                   |    function)](api/languages/cpp_a |
|                                   | pi.html#_CPPv4N5cudaq6sum_opD0Ev) |
|                                   | -   [cudaq::tensor (C++           |
|                                   |     type)](api/languages/cp       |
|                                   | p_api.html#_CPPv4N5cudaq6tensorE) |
|                                   | -   [cudaq::TensorStateData (C++  |
|                                   |                                   |
|                                   | type)](api/languages/cpp_api.html |
|                                   | #_CPPv4N5cudaq15TensorStateDataE) |
|                                   | -   [cudaq::Trace (C++            |
|                                   |     class)](api/languages/c       |
|                                   | pp_api.html#_CPPv4N5cudaq5TraceE) |
|                                   | -   [cudaq::unset_noise (C++      |
|                                   |     f                             |
|                                   | unction)](api/languages/cpp_api.h |
|                                   | tml#_CPPv4N5cudaq11unset_noiseEv) |
|                                   | -   [cudaq::x_error (C++          |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7x_errorE) |
|                                   | -   [cudaq::y_error (C++          |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7y_errorE) |
|                                   | -                                 |
|                                   |   [cudaq::y_error::num_parameters |
|                                   |     (C++                          |
|                                   |     member)](                     |
|                                   | api/languages/cpp_api.html#_CPPv4 |
|                                   | N5cudaq7y_error14num_parametersE) |
|                                   | -   [cudaq::y_error::num_targets  |
|                                   |     (C++                          |
|                                   |     member                        |
|                                   | )](api/languages/cpp_api.html#_CP |
|                                   | Pv4N5cudaq7y_error11num_targetsE) |
|                                   | -   [cudaq::z_error (C++          |
|                                   |     class)](api/languages/cpp     |
|                                   | _api.html#_CPPv4N5cudaq7z_errorE) |
+-----------------------------------+-----------------------------------+

## D {#D}

+-----------------------------------+-----------------------------------+
| -   [define() (cudaq.operators    | -   [displace() (in module        |
|     method)](api/languages/python |     cudaq.operators.custo         |
| _api.html#cudaq.operators.define) | m)](api/languages/python_api.html |
|     -   [(cuda                    | #cudaq.operators.custom.displace) |
| q.operators.MatrixOperatorElement | -   [distribute_terms()           |
|         class                     |     (cu                           |
|         method)](api/langu        | daq.operators.boson.BosonOperator |
| ages/python_api.html#cudaq.operat |     method)](api/languages/pyt    |
| ors.MatrixOperatorElement.define) | hon_api.html#cudaq.operators.boso |
|     -   [(in module               | n.BosonOperator.distribute_terms) |
|         cudaq.operators.cus       |     -   [(cudaq.                  |
| tom)](api/languages/python_api.ht | operators.fermion.FermionOperator |
| ml#cudaq.operators.custom.define) |                                   |
| -   [degrees                      |    method)](api/languages/python_ |
|     (cu                           | api.html#cudaq.operators.fermion. |
| daq.operators.boson.BosonOperator | FermionOperator.distribute_terms) |
|     property)](api/lang           |     -                             |
| uages/python_api.html#cudaq.opera |  [(cudaq.operators.MatrixOperator |
| tors.boson.BosonOperator.degrees) |         method)](api/language     |
|     -   [(cudaq.ope               | s/python_api.html#cudaq.operators |
| rators.boson.BosonOperatorElement | .MatrixOperator.distribute_terms) |
|                                   |     -   [(                        |
|        property)](api/languages/p | cudaq.operators.spin.SpinOperator |
| ython_api.html#cudaq.operators.bo |         method)](api/languages/p  |
| son.BosonOperatorElement.degrees) | ython_api.html#cudaq.operators.sp |
|     -   [(cudaq.                  | in.SpinOperator.distribute_terms) |
| operators.boson.BosonOperatorTerm |     -   [(cuda                    |
|         property)](api/language   | q.operators.spin.SpinOperatorTerm |
| s/python_api.html#cudaq.operators |                                   |
| .boson.BosonOperatorTerm.degrees) |      method)](api/languages/pytho |
|     -   [(cudaq.                  | n_api.html#cudaq.operators.spin.S |
| operators.fermion.FermionOperator | pinOperatorTerm.distribute_terms) |
|         property)](api/language   | -   [draw() (in module            |
| s/python_api.html#cudaq.operators |     cudaq)](api/lang              |
| .fermion.FermionOperator.degrees) | uages/python_api.html#cudaq.draw) |
|     -   [(cudaq.operato           | -   [dump() (cudaq.ComplexMatrix  |
| rs.fermion.FermionOperatorElement |                                   |
|                                   |   method)](api/languages/python_a |
|    property)](api/languages/pytho | pi.html#cudaq.ComplexMatrix.dump) |
| n_api.html#cudaq.operators.fermio |     -   [(cudaq.ObserveResult     |
| n.FermionOperatorElement.degrees) |                                   |
|     -   [(cudaq.oper              |   method)](api/languages/python_a |
| ators.fermion.FermionOperatorTerm | pi.html#cudaq.ObserveResult.dump) |
|                                   |     -   [(cu                      |
|       property)](api/languages/py | daq.operators.boson.BosonOperator |
| thon_api.html#cudaq.operators.fer |         method)](api/l            |
| mion.FermionOperatorTerm.degrees) | anguages/python_api.html#cudaq.op |
|     -                             | erators.boson.BosonOperator.dump) |
|  [(cudaq.operators.MatrixOperator |     -   [(cudaq.                  |
|         property)](api            | operators.boson.BosonOperatorTerm |
| /languages/python_api.html#cudaq. |         method)](api/langu        |
| operators.MatrixOperator.degrees) | ages/python_api.html#cudaq.operat |
|     -   [(cuda                    | ors.boson.BosonOperatorTerm.dump) |
| q.operators.MatrixOperatorElement |     -   [(cudaq.                  |
|         property)](api/langua     | operators.fermion.FermionOperator |
| ges/python_api.html#cudaq.operato |         method)](api/langu        |
| rs.MatrixOperatorElement.degrees) | ages/python_api.html#cudaq.operat |
|     -   [(c                       | ors.fermion.FermionOperator.dump) |
| udaq.operators.MatrixOperatorTerm |     -   [(cudaq.oper              |
|         property)](api/lan        | ators.fermion.FermionOperatorTerm |
| guages/python_api.html#cudaq.oper |         method)](api/languages    |
| ators.MatrixOperatorTerm.degrees) | /python_api.html#cudaq.operators. |
|     -   [(                        | fermion.FermionOperatorTerm.dump) |
| cudaq.operators.spin.SpinOperator |     -                             |
|         property)](api/la         |  [(cudaq.operators.MatrixOperator |
| nguages/python_api.html#cudaq.ope |         method)](                 |
| rators.spin.SpinOperator.degrees) | api/languages/python_api.html#cud |
|     -   [(cudaq.o                 | aq.operators.MatrixOperator.dump) |
| perators.spin.SpinOperatorElement |     -   [(c                       |
|         property)](api/languages  | udaq.operators.MatrixOperatorTerm |
| /python_api.html#cudaq.operators. |         method)](api/             |
| spin.SpinOperatorElement.degrees) | languages/python_api.html#cudaq.o |
|     -   [(cuda                    | perators.MatrixOperatorTerm.dump) |
| q.operators.spin.SpinOperatorTerm |     -   [(                        |
|         property)](api/langua     | cudaq.operators.spin.SpinOperator |
| ges/python_api.html#cudaq.operato |         method)](api              |
| rs.spin.SpinOperatorTerm.degrees) | /languages/python_api.html#cudaq. |
| -   [Depolarization1 (class in    | operators.spin.SpinOperator.dump) |
|     cudaq)](api/languages/pytho   |     -   [(cuda                    |
| n_api.html#cudaq.Depolarization1) | q.operators.spin.SpinOperatorTerm |
| -   [Depolarization2 (class in    |         method)](api/lan          |
|     cudaq)](api/languages/pytho   | guages/python_api.html#cudaq.oper |
| n_api.html#cudaq.Depolarization2) | ators.spin.SpinOperatorTerm.dump) |
| -   [DepolarizationChannel (class |     -   [(cudaq.Resources         |
|     in                            |                                   |
|                                   |       method)](api/languages/pyth |
| cudaq)](api/languages/python_api. | on_api.html#cudaq.Resources.dump) |
| html#cudaq.DepolarizationChannel) |     -   [(cudaq.SampleResult      |
| -   [description (cudaq.Target    |                                   |
|                                   |    method)](api/languages/python_ |
| property)](api/languages/python_a | api.html#cudaq.SampleResult.dump) |
| pi.html#cudaq.Target.description) |     -   [(cudaq.State             |
| -   [deserialize()                |         method)](api/languages/   |
|     (cudaq.SampleResult           | python_api.html#cudaq.State.dump) |
|     meth                          |                                   |
| od)](api/languages/python_api.htm |                                   |
| l#cudaq.SampleResult.deserialize) |                                   |
+-----------------------------------+-----------------------------------+

## E {#E}

+-----------------------------------+-----------------------------------+
| -   [ElementaryOperator (in       | -   [evaluate()                   |
|     module                        |                                   |
|     cudaq.operators)]             |   (cudaq.operators.ScalarOperator |
| (api/languages/python_api.html#cu |     method)](api/                 |
| daq.operators.ElementaryOperator) | languages/python_api.html#cudaq.o |
| -   [empty()                      | perators.ScalarOperator.evaluate) |
|     (cu                           | -   [evaluate_coefficient()       |
| daq.operators.boson.BosonOperator |     (cudaq.                       |
|     static                        | operators.boson.BosonOperatorTerm |
|     method)](api/la               |     m                             |
| nguages/python_api.html#cudaq.ope | ethod)](api/languages/python_api. |
| rators.boson.BosonOperator.empty) | html#cudaq.operators.boson.BosonO |
|     -   [(cudaq.                  | peratorTerm.evaluate_coefficient) |
| operators.fermion.FermionOperator |     -   [(cudaq.oper              |
|         static                    | ators.fermion.FermionOperatorTerm |
|         method)](api/langua       |         metho                     |
| ges/python_api.html#cudaq.operato | d)](api/languages/python_api.html |
| rs.fermion.FermionOperator.empty) | #cudaq.operators.fermion.FermionO |
|     -                             | peratorTerm.evaluate_coefficient) |
|  [(cudaq.operators.MatrixOperator |     -   [(c                       |
|         static                    | udaq.operators.MatrixOperatorTerm |
|         method)](a                |                                   |
| pi/languages/python_api.html#cuda |     method)](api/languages/python |
| q.operators.MatrixOperator.empty) | _api.html#cudaq.operators.MatrixO |
|     -   [(                        | peratorTerm.evaluate_coefficient) |
| cudaq.operators.spin.SpinOperator |     -   [(cuda                    |
|         static                    | q.operators.spin.SpinOperatorTerm |
|         method)](api/             |                                   |
| languages/python_api.html#cudaq.o |  method)](api/languages/python_ap |
| perators.spin.SpinOperator.empty) | i.html#cudaq.operators.spin.SpinO |
|     -   [(in module               | peratorTerm.evaluate_coefficient) |
|                                   | -   [evolve() (in module          |
|     cudaq.boson)](api/languages/p |     cudaq)](api/langua            |
| ython_api.html#cudaq.boson.empty) | ges/python_api.html#cudaq.evolve) |
|     -   [(in module               | -   [evolve_async() (in module    |
|                                   |     cudaq)](api/languages/py      |
| cudaq.fermion)](api/languages/pyt | thon_api.html#cudaq.evolve_async) |
| hon_api.html#cudaq.fermion.empty) | -   [EvolveResult (class in       |
|     -   [(in module               |     cudaq)](api/languages/py      |
|         cudaq.operators.cu        | thon_api.html#cudaq.EvolveResult) |
| stom)](api/languages/python_api.h | -   [expectation()                |
| tml#cudaq.operators.custom.empty) |     (cudaq.ObserveResult          |
|     -   [(in module               |     metho                         |
|                                   | d)](api/languages/python_api.html |
|       cudaq.spin)](api/languages/ | #cudaq.ObserveResult.expectation) |
| python_api.html#cudaq.spin.empty) |     -   [(cudaq.SampleResult      |
| -   [empty_op()                   |         meth                      |
|     (                             | od)](api/languages/python_api.htm |
| cudaq.operators.spin.SpinOperator | l#cudaq.SampleResult.expectation) |
|     static                        | -   [expectation_values()         |
|     method)](api/lan              |     (cudaq.EvolveResult           |
| guages/python_api.html#cudaq.oper |     method)](ap                   |
| ators.spin.SpinOperator.empty_op) | i/languages/python_api.html#cudaq |
| -   [enable_return_to_log()       | .EvolveResult.expectation_values) |
|     (cudaq.PyKernelDecorator      | -   [expectation_z()              |
|     method)](api/langu            |     (cudaq.SampleResult           |
| ages/python_api.html#cudaq.PyKern |     method                        |
| elDecorator.enable_return_to_log) | )](api/languages/python_api.html# |
| -   [estimate_resources() (in     | cudaq.SampleResult.expectation_z) |
|     module                        | -   [expected_dimensions          |
|                                   |     (cuda                         |
|    cudaq)](api/languages/python_a | q.operators.MatrixOperatorElement |
| pi.html#cudaq.estimate_resources) |                                   |
|                                   | property)](api/languages/python_a |
|                                   | pi.html#cudaq.operators.MatrixOpe |
|                                   | ratorElement.expected_dimensions) |
+-----------------------------------+-----------------------------------+

## F {#F}

+-----------------------------------+-----------------------------------+
| -   [FermionOperator (class in    | -   [from_json()                  |
|                                   |     (                             |
|    cudaq.operators.fermion)](api/ | cudaq.gradients.CentralDifference |
| languages/python_api.html#cudaq.o |     static                        |
| perators.fermion.FermionOperator) |     method)](api/lang             |
| -   [FermionOperatorElement       | uages/python_api.html#cudaq.gradi |
|     (class in                     | ents.CentralDifference.from_json) |
|     cuda                          |     -   [(                        |
| q.operators.fermion)](api/languag | cudaq.gradients.ForwardDifference |
| es/python_api.html#cudaq.operator |         static                    |
| s.fermion.FermionOperatorElement) |         method)](api/lang         |
| -   [FermionOperatorTerm (class   | uages/python_api.html#cudaq.gradi |
|     in                            | ents.ForwardDifference.from_json) |
|     c                             |     -                             |
| udaq.operators.fermion)](api/lang |  [(cudaq.gradients.ParameterShift |
| uages/python_api.html#cudaq.opera |         static                    |
| tors.fermion.FermionOperatorTerm) |         method)](api/l            |
| -   [final_expectation_values()   | anguages/python_api.html#cudaq.gr |
|     (cudaq.EvolveResult           | adients.ParameterShift.from_json) |
|     method)](api/lang             |     -   [(                        |
| uages/python_api.html#cudaq.Evolv | cudaq.operators.spin.SpinOperator |
| eResult.final_expectation_values) |         static                    |
| -   [final_state()                |         method)](api/lang         |
|     (cudaq.EvolveResult           | uages/python_api.html#cudaq.opera |
|     meth                          | tors.spin.SpinOperator.from_json) |
| od)](api/languages/python_api.htm |     -   [(cuda                    |
| l#cudaq.EvolveResult.final_state) | q.operators.spin.SpinOperatorTerm |
| -   [finalize() (in module        |         static                    |
|     cudaq.mpi)](api/languages/py  |         method)](api/language     |
| thon_api.html#cudaq.mpi.finalize) | s/python_api.html#cudaq.operators |
| -   [for_each_pauli()             | .spin.SpinOperatorTerm.from_json) |
|     (                             |     -   [(cudaq.optimizers.COBYLA |
| cudaq.operators.spin.SpinOperator |         static                    |
|     method)](api/languages        |         method)                   |
| /python_api.html#cudaq.operators. | ](api/languages/python_api.html#c |
| spin.SpinOperator.for_each_pauli) | udaq.optimizers.COBYLA.from_json) |
|     -   [(cuda                    |     -   [                         |
| q.operators.spin.SpinOperatorTerm | (cudaq.optimizers.GradientDescent |
|                                   |         static                    |
|        method)](api/languages/pyt |         method)](api/lan          |
| hon_api.html#cudaq.operators.spin | guages/python_api.html#cudaq.opti |
| .SpinOperatorTerm.for_each_pauli) | mizers.GradientDescent.from_json) |
| -   [for_each_term()              |     -   [(cudaq.optimizers.LBFGS  |
|     (                             |         static                    |
| cudaq.operators.spin.SpinOperator |         method                    |
|     method)](api/language         | )](api/languages/python_api.html# |
| s/python_api.html#cudaq.operators | cudaq.optimizers.LBFGS.from_json) |
| .spin.SpinOperator.for_each_term) |                                   |
| -   [ForwardDifference (class in  | -   [(cudaq.optimizers.NelderMead |
|     cudaq.gradients)              |         static                    |
| ](api/languages/python_api.html#c |         method)](ap               |
| udaq.gradients.ForwardDifference) | i/languages/python_api.html#cudaq |
| -   [from_data() (cudaq.State     | .optimizers.NelderMead.from_json) |
|     static                        |     -   [(cudaq.PyKernelDecorator |
|     method)](api/languages/pytho  |         static                    |
| n_api.html#cudaq.State.from_data) |         method)                   |
|                                   | ](api/languages/python_api.html#c |
|                                   | udaq.PyKernelDecorator.from_json) |
|                                   | -   [from_word()                  |
|                                   |     (                             |
|                                   | cudaq.operators.spin.SpinOperator |
|                                   |     static                        |
|                                   |     method)](api/lang             |
|                                   | uages/python_api.html#cudaq.opera |
|                                   | tors.spin.SpinOperator.from_word) |
+-----------------------------------+-----------------------------------+

## G {#G}

+-----------------------------------+-----------------------------------+
| -   [get()                        | -   [get_register_counts()        |
|     (cudaq.AsyncEvolveResult      |     (cudaq.SampleResult           |
|     m                             |     method)](api                  |
| ethod)](api/languages/python_api. | /languages/python_api.html#cudaq. |
| html#cudaq.AsyncEvolveResult.get) | SampleResult.get_register_counts) |
|                                   | -   [get_sequential_data()        |
|    -   [(cudaq.AsyncObserveResult |     (cudaq.SampleResult           |
|         me                        |     method)](api                  |
| thod)](api/languages/python_api.h | /languages/python_api.html#cudaq. |
| tml#cudaq.AsyncObserveResult.get) | SampleResult.get_sequential_data) |
|     -   [(cudaq.AsyncStateResult  | -   [get_spin()                   |
|                                   |     (cudaq.ObserveResult          |
| method)](api/languages/python_api |     me                            |
| .html#cudaq.AsyncStateResult.get) | thod)](api/languages/python_api.h |
| -   [get_binary_symplectic_form() | tml#cudaq.ObserveResult.get_spin) |
|     (cuda                         | -   [get_state() (in module       |
| q.operators.spin.SpinOperatorTerm |     cudaq)](api/languages         |
|     metho                         | /python_api.html#cudaq.get_state) |
| d)](api/languages/python_api.html | -   [get_state_async() (in module |
| #cudaq.operators.spin.SpinOperato |     cudaq)](api/languages/pytho   |
| rTerm.get_binary_symplectic_form) | n_api.html#cudaq.get_state_async) |
| -   [get_channels()               | -   [get_state_refval()           |
|     (cudaq.NoiseModel             |     (cudaq.State                  |
|     met                           |     me                            |
| hod)](api/languages/python_api.ht | thod)](api/languages/python_api.h |
| ml#cudaq.NoiseModel.get_channels) | tml#cudaq.State.get_state_refval) |
| -   [get_coefficient()            | -   [get_target() (in module      |
|     (                             |     cudaq)](api/languages/        |
| cudaq.operators.spin.SpinOperator | python_api.html#cudaq.get_target) |
|     method)](api/languages/       | -   [get_targets() (in module     |
| python_api.html#cudaq.operators.s |     cudaq)](api/languages/p       |
| pin.SpinOperator.get_coefficient) | ython_api.html#cudaq.get_targets) |
|     -   [(cuda                    | -   [get_term_count()             |
| q.operators.spin.SpinOperatorTerm |     (                             |
|                                   | cudaq.operators.spin.SpinOperator |
|       method)](api/languages/pyth |     method)](api/languages        |
| on_api.html#cudaq.operators.spin. | /python_api.html#cudaq.operators. |
| SpinOperatorTerm.get_coefficient) | spin.SpinOperator.get_term_count) |
| -   [get_marginal_counts()        | -   [get_total_shots()            |
|     (cudaq.SampleResult           |     (cudaq.SampleResult           |
|     method)](api                  |     method)]                      |
| /languages/python_api.html#cudaq. | (api/languages/python_api.html#cu |
| SampleResult.get_marginal_counts) | daq.SampleResult.get_total_shots) |
| -   [get_ops()                    | -   [getTensor() (cudaq.State     |
|     (cudaq.KrausChannel           |     method)](api/languages/pytho  |
|                                   | n_api.html#cudaq.State.getTensor) |
| method)](api/languages/python_api | -   [getTensors() (cudaq.State    |
| .html#cudaq.KrausChannel.get_ops) |     method)](api/languages/python |
| -   [get_pauli_word()             | _api.html#cudaq.State.getTensors) |
|     (cuda                         | -   [gradient (class in           |
| q.operators.spin.SpinOperatorTerm |     cudaq.g                       |
|     method)](api/languages/pyt    | radients)](api/languages/python_a |
| hon_api.html#cudaq.operators.spin | pi.html#cudaq.gradients.gradient) |
| .SpinOperatorTerm.get_pauli_word) | -   [GradientDescent (class in    |
| -   [get_precision()              |     cudaq.optimizers              |
|     (cudaq.Target                 | )](api/languages/python_api.html# |
|                                   | cudaq.optimizers.GradientDescent) |
| method)](api/languages/python_api |                                   |
| .html#cudaq.Target.get_precision) |                                   |
| -   [get_qubit_count()            |                                   |
|     (                             |                                   |
| cudaq.operators.spin.SpinOperator |                                   |
|     method)](api/languages/       |                                   |
| python_api.html#cudaq.operators.s |                                   |
| pin.SpinOperator.get_qubit_count) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|                                   |                                   |
|       method)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.spin. |                                   |
| SpinOperatorTerm.get_qubit_count) |                                   |
| -   [get_raw_data()               |                                   |
|     (                             |                                   |
| cudaq.operators.spin.SpinOperator |                                   |
|     method)](api/languag          |                                   |
| es/python_api.html#cudaq.operator |                                   |
| s.spin.SpinOperator.get_raw_data) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         method)](api/languages/p  |                                   |
| ython_api.html#cudaq.operators.sp |                                   |
| in.SpinOperatorTerm.get_raw_data) |                                   |
+-----------------------------------+-----------------------------------+

## H {#H}

+-----------------------------------+-----------------------------------+
| -   [handle_call_arguments()      | -   [has_target() (in module      |
|     (cudaq.PyKernelDecorator      |     cudaq)](api/languages/        |
|     method)](api/langua           | python_api.html#cudaq.has_target) |
| ges/python_api.html#cudaq.PyKerne |                                   |
| lDecorator.handle_call_arguments) |                                   |
+-----------------------------------+-----------------------------------+

## I {#I}

+-----------------------------------+-----------------------------------+
| -   [i() (in module               | -   [initialize_cudaq() (in       |
|     cudaq.spin)](api/langua       |     module                        |
| ges/python_api.html#cudaq.spin.i) |     cudaq)](api/languages/python  |
| -   [id                           | _api.html#cudaq.initialize_cudaq) |
|     (cuda                         | -   [InitialState (in module      |
| q.operators.MatrixOperatorElement |     cudaq.dynamics.helpers)](     |
|     property)](api/l              | api/languages/python_api.html#cud |
| anguages/python_api.html#cudaq.op | aq.dynamics.helpers.InitialState) |
| erators.MatrixOperatorElement.id) | -   [InitialStateType (class in   |
| -   [identities() (in module      |     cudaq)](api/languages/python  |
|     c                             | _api.html#cudaq.InitialStateType) |
| udaq.boson)](api/languages/python | -   [instantiate()                |
| _api.html#cudaq.boson.identities) |     (cudaq.operators              |
|     -   [(in module               |     m                             |
|         cudaq                     | ethod)](api/languages/python_api. |
| .fermion)](api/languages/python_a | html#cudaq.operators.instantiate) |
| pi.html#cudaq.fermion.identities) |     -   [(in module               |
|     -   [(in module               |         cudaq.operators.custom)]  |
|         cudaq.operators.custom)   | (api/languages/python_api.html#cu |
| ](api/languages/python_api.html#c | daq.operators.custom.instantiate) |
| udaq.operators.custom.identities) | -   [intermediate_states()        |
|     -   [(in module               |     (cudaq.EvolveResult           |
|                                   |     method)](api                  |
|  cudaq.spin)](api/languages/pytho | /languages/python_api.html#cudaq. |
| n_api.html#cudaq.spin.identities) | EvolveResult.intermediate_states) |
| -   [identity()                   | -   [IntermediateResultSave       |
|     (cu                           |     (class in                     |
| daq.operators.boson.BosonOperator |     c                             |
|     static                        | udaq)](api/languages/python_api.h |
|     method)](api/langu            | tml#cudaq.IntermediateResultSave) |
| ages/python_api.html#cudaq.operat | -   [is_constant()                |
| ors.boson.BosonOperator.identity) |                                   |
|     -   [(cudaq.                  |   (cudaq.operators.ScalarOperator |
| operators.fermion.FermionOperator |     method)](api/lan              |
|         static                    | guages/python_api.html#cudaq.oper |
|         method)](api/languages    | ators.ScalarOperator.is_constant) |
| /python_api.html#cudaq.operators. | -   [is_emulated() (cudaq.Target  |
| fermion.FermionOperator.identity) |                                   |
|     -                             |   method)](api/languages/python_a |
|  [(cudaq.operators.MatrixOperator | pi.html#cudaq.Target.is_emulated) |
|         static                    | -   [is_identity()                |
|         method)](api/             |     (cudaq.                       |
| languages/python_api.html#cudaq.o | operators.boson.BosonOperatorTerm |
| perators.MatrixOperator.identity) |     method)](api/languages/py     |
|     -   [(                        | thon_api.html#cudaq.operators.bos |
| cudaq.operators.spin.SpinOperator | on.BosonOperatorTerm.is_identity) |
|         static                    |     -   [(cudaq.oper              |
|         method)](api/lan          | ators.fermion.FermionOperatorTerm |
| guages/python_api.html#cudaq.oper |                                   |
| ators.spin.SpinOperator.identity) |     method)](api/languages/python |
|     -   [(in module               | _api.html#cudaq.operators.fermion |
|                                   | .FermionOperatorTerm.is_identity) |
|  cudaq.boson)](api/languages/pyth |     -   [(c                       |
| on_api.html#cudaq.boson.identity) | udaq.operators.MatrixOperatorTerm |
|     -   [(in module               |         method)](api/languag      |
|         cud                       | es/python_api.html#cudaq.operator |
| aq.fermion)](api/languages/python | s.MatrixOperatorTerm.is_identity) |
| _api.html#cudaq.fermion.identity) |     -   [(                        |
|     -   [(in module               | cudaq.operators.spin.SpinOperator |
|                                   |         method)](api/langua       |
|    cudaq.spin)](api/languages/pyt | ges/python_api.html#cudaq.operato |
| hon_api.html#cudaq.spin.identity) | rs.spin.SpinOperator.is_identity) |
| -   [initial_parameters           |     -   [(cuda                    |
|     (cudaq.optimizers.COBYLA      | q.operators.spin.SpinOperatorTerm |
|     property)](api/lan            |         method)](api/languages/   |
| guages/python_api.html#cudaq.opti | python_api.html#cudaq.operators.s |
| mizers.COBYLA.initial_parameters) | pin.SpinOperatorTerm.is_identity) |
|     -   [                         | -   [is_initialized() (in module  |
| (cudaq.optimizers.GradientDescent |     c                             |
|                                   | udaq.mpi)](api/languages/python_a |
|       property)](api/languages/py | pi.html#cudaq.mpi.is_initialized) |
| thon_api.html#cudaq.optimizers.Gr | -   [is_on_gpu() (cudaq.State     |
| adientDescent.initial_parameters) |     method)](api/languages/pytho  |
|     -   [(cudaq.optimizers.LBFGS  | n_api.html#cudaq.State.is_on_gpu) |
|         property)](api/la         | -   [is_remote() (cudaq.Target    |
| nguages/python_api.html#cudaq.opt |     method)](api/languages/python |
| imizers.LBFGS.initial_parameters) | _api.html#cudaq.Target.is_remote) |
|                                   | -   [is_remote_simulator()        |
| -   [(cudaq.optimizers.NelderMead |     (cudaq.Target                 |
|         property)](api/languag    |     method                        |
| es/python_api.html#cudaq.optimize | )](api/languages/python_api.html# |
| rs.NelderMead.initial_parameters) | cudaq.Target.is_remote_simulator) |
| -   [initialize() (in module      | -   [items() (cudaq.SampleResult  |
|                                   |                                   |
|    cudaq.mpi)](api/languages/pyth |   method)](api/languages/python_a |
| on_api.html#cudaq.mpi.initialize) | pi.html#cudaq.SampleResult.items) |
+-----------------------------------+-----------------------------------+

## K {#K}

+-----------------------------------+-----------------------------------+
| -   [Kernel (in module            | -   [KrausChannel (class in       |
|     cudaq)](api/langua            |     cudaq)](api/languages/py      |
| ges/python_api.html#cudaq.Kernel) | thon_api.html#cudaq.KrausChannel) |
| -   [kernel() (in module          | -   [KrausOperator (class in      |
|     cudaq)](api/langua            |     cudaq)](api/languages/pyt     |
| ges/python_api.html#cudaq.kernel) | hon_api.html#cudaq.KrausOperator) |
+-----------------------------------+-----------------------------------+

## L {#L}

+-----------------------------------+-----------------------------------+
| -   [launch_args_required()       | -   [lower_bounds                 |
|     (cudaq.PyKernelDecorator      |     (cudaq.optimizers.COBYLA      |
|     method)](api/langu            |     property)](a                  |
| ages/python_api.html#cudaq.PyKern | pi/languages/python_api.html#cuda |
| elDecorator.launch_args_required) | q.optimizers.COBYLA.lower_bounds) |
| -   [LBFGS (class in              |     -   [                         |
|     cudaq.                        | (cudaq.optimizers.GradientDescent |
| optimizers)](api/languages/python |         property)](api/langua     |
| _api.html#cudaq.optimizers.LBFGS) | ges/python_api.html#cudaq.optimiz |
| -   [left_multiply()              | ers.GradientDescent.lower_bounds) |
|     (cudaq.SuperOperator static   |     -   [(cudaq.optimizers.LBFGS  |
|     method)                       |         property)](               |
| ](api/languages/python_api.html#c | api/languages/python_api.html#cud |
| udaq.SuperOperator.left_multiply) | aq.optimizers.LBFGS.lower_bounds) |
| -   [left_right_multiply()        |                                   |
|     (cudaq.SuperOperator static   | -   [(cudaq.optimizers.NelderMead |
|     method)](api/                 |         property)](api/l          |
| languages/python_api.html#cudaq.S | anguages/python_api.html#cudaq.op |
| uperOperator.left_right_multiply) | timizers.NelderMead.lower_bounds) |
|                                   | -   [lower_quake_to_codegen()     |
|                                   |     (cudaq.PyKernelDecorator      |
|                                   |     method)](api/languag          |
|                                   | es/python_api.html#cudaq.PyKernel |
|                                   | Decorator.lower_quake_to_codegen) |
+-----------------------------------+-----------------------------------+

## M {#M}

+-----------------------------------+-----------------------------------+
| -   [make_kernel() (in module     | -   [min_degree                   |
|     cudaq)](api/languages/p       |     (cu                           |
| ython_api.html#cudaq.make_kernel) | daq.operators.boson.BosonOperator |
| -   [MatrixOperator (class in     |     property)](api/languag        |
|     cudaq.operato                 | es/python_api.html#cudaq.operator |
| rs)](api/languages/python_api.htm | s.boson.BosonOperator.min_degree) |
| l#cudaq.operators.MatrixOperator) |     -   [(cudaq.                  |
| -   [MatrixOperatorElement (class | operators.boson.BosonOperatorTerm |
|     in                            |                                   |
|     cudaq.operators)](ap          |        property)](api/languages/p |
| i/languages/python_api.html#cudaq | ython_api.html#cudaq.operators.bo |
| .operators.MatrixOperatorElement) | son.BosonOperatorTerm.min_degree) |
| -   [MatrixOperatorTerm (class in |     -   [(cudaq.                  |
|     cudaq.operators)]             | operators.fermion.FermionOperator |
| (api/languages/python_api.html#cu |                                   |
| daq.operators.MatrixOperatorTerm) |        property)](api/languages/p |
| -   [max_degree                   | ython_api.html#cudaq.operators.fe |
|     (cu                           | rmion.FermionOperator.min_degree) |
| daq.operators.boson.BosonOperator |     -   [(cudaq.oper              |
|     property)](api/languag        | ators.fermion.FermionOperatorTerm |
| es/python_api.html#cudaq.operator |                                   |
| s.boson.BosonOperator.max_degree) |    property)](api/languages/pytho |
|     -   [(cudaq.                  | n_api.html#cudaq.operators.fermio |
| operators.boson.BosonOperatorTerm | n.FermionOperatorTerm.min_degree) |
|                                   |     -                             |
|        property)](api/languages/p |  [(cudaq.operators.MatrixOperator |
| ython_api.html#cudaq.operators.bo |         property)](api/la         |
| son.BosonOperatorTerm.max_degree) | nguages/python_api.html#cudaq.ope |
|     -   [(cudaq.                  | rators.MatrixOperator.min_degree) |
| operators.fermion.FermionOperator |     -   [(c                       |
|                                   | udaq.operators.MatrixOperatorTerm |
|        property)](api/languages/p |         property)](api/langua     |
| ython_api.html#cudaq.operators.fe | ges/python_api.html#cudaq.operato |
| rmion.FermionOperator.max_degree) | rs.MatrixOperatorTerm.min_degree) |
|     -   [(cudaq.oper              |     -   [(                        |
| ators.fermion.FermionOperatorTerm | cudaq.operators.spin.SpinOperator |
|                                   |         property)](api/langu      |
|    property)](api/languages/pytho | ages/python_api.html#cudaq.operat |
| n_api.html#cudaq.operators.fermio | ors.spin.SpinOperator.min_degree) |
| n.FermionOperatorTerm.max_degree) |     -   [(cuda                    |
|     -                             | q.operators.spin.SpinOperatorTerm |
|  [(cudaq.operators.MatrixOperator |         property)](api/languages  |
|         property)](api/la         | /python_api.html#cudaq.operators. |
| nguages/python_api.html#cudaq.ope | spin.SpinOperatorTerm.min_degree) |
| rators.MatrixOperator.max_degree) | -   [minimal_eigenvalue()         |
|     -   [(c                       |     (cudaq.ComplexMatrix          |
| udaq.operators.MatrixOperatorTerm |     method)](api                  |
|         property)](api/langua     | /languages/python_api.html#cudaq. |
| ges/python_api.html#cudaq.operato | ComplexMatrix.minimal_eigenvalue) |
| rs.MatrixOperatorTerm.max_degree) | -   [minus() (in module           |
|     -   [(                        |     cudaq.spin)](api/languages/   |
| cudaq.operators.spin.SpinOperator | python_api.html#cudaq.spin.minus) |
|         property)](api/langu      | -   module                        |
| ages/python_api.html#cudaq.operat |     -   [cudaq](api/langua        |
| ors.spin.SpinOperator.max_degree) | ges/python_api.html#module-cudaq) |
|     -   [(cuda                    |     -                             |
| q.operators.spin.SpinOperatorTerm |    [cudaq.boson](api/languages/py |
|         property)](api/languages  | thon_api.html#module-cudaq.boson) |
| /python_api.html#cudaq.operators. |     -   [                         |
| spin.SpinOperatorTerm.max_degree) | cudaq.fermion](api/languages/pyth |
| -   [max_iterations               | on_api.html#module-cudaq.fermion) |
|     (cudaq.optimizers.COBYLA      |     -   [cudaq.operators.cu       |
|     property)](api                | stom](api/languages/python_api.ht |
| /languages/python_api.html#cudaq. | ml#module-cudaq.operators.custom) |
| optimizers.COBYLA.max_iterations) |                                   |
|     -   [                         |  -   [cudaq.spin](api/languages/p |
| (cudaq.optimizers.GradientDescent | ython_api.html#module-cudaq.spin) |
|         property)](api/language   | -   [momentum() (in module        |
| s/python_api.html#cudaq.optimizer |                                   |
| s.GradientDescent.max_iterations) |  cudaq.boson)](api/languages/pyth |
|     -   [(cudaq.optimizers.LBFGS  | on_api.html#cudaq.boson.momentum) |
|         property)](ap             |     -   [(in module               |
| i/languages/python_api.html#cudaq |         cudaq.operators.custo     |
| .optimizers.LBFGS.max_iterations) | m)](api/languages/python_api.html |
|                                   | #cudaq.operators.custom.momentum) |
| -   [(cudaq.optimizers.NelderMead | -   [most_probable()              |
|         property)](api/lan        |     (cudaq.SampleResult           |
| guages/python_api.html#cudaq.opti |     method                        |
| mizers.NelderMead.max_iterations) | )](api/languages/python_api.html# |
| -   [mdiag_sparse_matrix (C++     | cudaq.SampleResult.most_probable) |
|     type)](api/languages/cpp_api. |                                   |
| html#_CPPv419mdiag_sparse_matrix) |                                   |
| -   [merge_kernel()               |                                   |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](a                    |                                   |
| pi/languages/python_api.html#cuda |                                   |
| q.PyKernelDecorator.merge_kernel) |                                   |
| -   [merge_quake_source()         |                                   |
|     (cudaq.PyKernelDecorator      |                                   |
|     method)](api/lan              |                                   |
| guages/python_api.html#cudaq.PyKe |                                   |
| rnelDecorator.merge_quake_source) |                                   |
+-----------------------------------+-----------------------------------+

## N {#N}

+-----------------------------------+-----------------------------------+
| -   [name (cudaq.PyKernel         | -   [num_qpus() (cudaq.Target     |
|     attribute)](api/languages/pyt |     method)](api/languages/pytho  |
| hon_api.html#cudaq.PyKernel.name) | n_api.html#cudaq.Target.num_qpus) |
|                                   | -   [num_qubits() (cudaq.State    |
|   -   [(cudaq.SimulationPrecision |     method)](api/languages/python |
|         proper                    | _api.html#cudaq.State.num_qubits) |
| ty)](api/languages/python_api.htm | -   [num_ranks() (in module       |
| l#cudaq.SimulationPrecision.name) |     cudaq.mpi)](api/languages/pyt |
|     -   [(cudaq.spin.Pauli        | hon_api.html#cudaq.mpi.num_ranks) |
|                                   | -   [num_rows()                   |
|    property)](api/languages/pytho |     (cudaq.ComplexMatrix          |
| n_api.html#cudaq.spin.Pauli.name) |     me                            |
|     -   [(cudaq.Target            | thod)](api/languages/python_api.h |
|                                   | tml#cudaq.ComplexMatrix.num_rows) |
|        property)](api/languages/p | -   [number() (in module          |
| ython_api.html#cudaq.Target.name) |                                   |
| -   [NelderMead (class in         |    cudaq.boson)](api/languages/py |
|     cudaq.optim                   | thon_api.html#cudaq.boson.number) |
| izers)](api/languages/python_api. |     -   [(in module               |
| html#cudaq.optimizers.NelderMead) |         c                         |
| -   [NoiseModel (class in         | udaq.fermion)](api/languages/pyth |
|     cudaq)](api/languages/        | on_api.html#cudaq.fermion.number) |
| python_api.html#cudaq.NoiseModel) |     -   [(in module               |
| -   [num_available_gpus() (in     |         cudaq.operators.cus       |
|     module                        | tom)](api/languages/python_api.ht |
|                                   | ml#cudaq.operators.custom.number) |
|    cudaq)](api/languages/python_a | -   [nvqir::MPSSimulationState    |
| pi.html#cudaq.num_available_gpus) |     (C++                          |
| -   [num_columns()                |     class)]                       |
|     (cudaq.ComplexMatrix          | (api/languages/cpp_api.html#_CPPv |
|     metho                         | 4I0EN5nvqir18MPSSimulationStateE) |
| d)](api/languages/python_api.html | -                                 |
| #cudaq.ComplexMatrix.num_columns) |  [nvqir::TensorNetSimulationState |
|                                   |     (C++                          |
|                                   |     class)](api/l                 |
|                                   | anguages/cpp_api.html#_CPPv4I0EN5 |
|                                   | nvqir24TensorNetSimulationStateE) |
+-----------------------------------+-----------------------------------+

## O {#O}

+-----------------------------------+-----------------------------------+
| -   [observe() (in module         | -   [OptimizationResult (class in |
|     cudaq)](api/languag           |                                   |
| es/python_api.html#cudaq.observe) |    cudaq)](api/languages/python_a |
| -   [observe_async() (in module   | pi.html#cudaq.OptimizationResult) |
|     cudaq)](api/languages/pyt     | -   [optimize()                   |
| hon_api.html#cudaq.observe_async) |     (cudaq.optimizers.COBYLA      |
| -   [ObserveResult (class in      |     method                        |
|     cudaq)](api/languages/pyt     | )](api/languages/python_api.html# |
| hon_api.html#cudaq.ObserveResult) | cudaq.optimizers.COBYLA.optimize) |
| -   [OperatorSum (in module       |     -   [                         |
|     cudaq.oper                    | (cudaq.optimizers.GradientDescent |
| ators)](api/languages/python_api. |         method)](api/la           |
| html#cudaq.operators.OperatorSum) | nguages/python_api.html#cudaq.opt |
| -   [ops_count                    | imizers.GradientDescent.optimize) |
|     (cudaq.                       |     -   [(cudaq.optimizers.LBFGS  |
| operators.boson.BosonOperatorTerm |         metho                     |
|     property)](api/languages/     | d)](api/languages/python_api.html |
| python_api.html#cudaq.operators.b | #cudaq.optimizers.LBFGS.optimize) |
| oson.BosonOperatorTerm.ops_count) |                                   |
|     -   [(cudaq.oper              | -   [(cudaq.optimizers.NelderMead |
| ators.fermion.FermionOperatorTerm |         method)](a                |
|                                   | pi/languages/python_api.html#cuda |
|     property)](api/languages/pyth | q.optimizers.NelderMead.optimize) |
| on_api.html#cudaq.operators.fermi | -   [optimizer (class in          |
| on.FermionOperatorTerm.ops_count) |     cudaq.opti                    |
|     -   [(c                       | mizers)](api/languages/python_api |
| udaq.operators.MatrixOperatorTerm | .html#cudaq.optimizers.optimizer) |
|         property)](api/langu      | -   [overlap() (cudaq.State       |
| ages/python_api.html#cudaq.operat |     method)](api/languages/pyt    |
| ors.MatrixOperatorTerm.ops_count) | hon_api.html#cudaq.State.overlap) |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         property)](api/language   |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .spin.SpinOperatorTerm.ops_count) |                                   |
+-----------------------------------+-----------------------------------+

## P {#P}

+-----------------------------------+-----------------------------------+
| -   [parameters                   | -   [Pauli1 (class in             |
|     (cu                           |     cudaq)](api/langua            |
| daq.operators.boson.BosonOperator | ges/python_api.html#cudaq.Pauli1) |
|     property)](api/languag        | -   [Pauli2 (class in             |
| es/python_api.html#cudaq.operator |     cudaq)](api/langua            |
| s.boson.BosonOperator.parameters) | ges/python_api.html#cudaq.Pauli2) |
|     -   [(cudaq.                  | -   [PhaseDamping (class in       |
| operators.boson.BosonOperatorTerm |     cudaq)](api/languages/py      |
|                                   | thon_api.html#cudaq.PhaseDamping) |
|        property)](api/languages/p | -   [PhaseFlipChannel (class in   |
| ython_api.html#cudaq.operators.bo |     cudaq)](api/languages/python  |
| son.BosonOperatorTerm.parameters) | _api.html#cudaq.PhaseFlipChannel) |
|     -   [(cudaq.                  | -   [platform (cudaq.Target       |
| operators.fermion.FermionOperator |                                   |
|                                   |    property)](api/languages/pytho |
|        property)](api/languages/p | n_api.html#cudaq.Target.platform) |
| ython_api.html#cudaq.operators.fe | -   [plus() (in module            |
| rmion.FermionOperator.parameters) |     cudaq.spin)](api/languages    |
|     -   [(cudaq.oper              | /python_api.html#cudaq.spin.plus) |
| ators.fermion.FermionOperatorTerm | -   [position() (in module        |
|                                   |                                   |
|    property)](api/languages/pytho |  cudaq.boson)](api/languages/pyth |
| n_api.html#cudaq.operators.fermio | on_api.html#cudaq.boson.position) |
| n.FermionOperatorTerm.parameters) |     -   [(in module               |
|     -                             |         cudaq.operators.custo     |
|  [(cudaq.operators.MatrixOperator | m)](api/languages/python_api.html |
|         property)](api/la         | #cudaq.operators.custom.position) |
| nguages/python_api.html#cudaq.ope | -   [pre_compile()                |
| rators.MatrixOperator.parameters) |     (cudaq.PyKernelDecorator      |
|     -   [(cuda                    |     method)](                     |
| q.operators.MatrixOperatorElement | api/languages/python_api.html#cud |
|         property)](api/languages  | aq.PyKernelDecorator.pre_compile) |
| /python_api.html#cudaq.operators. | -   [probability()                |
| MatrixOperatorElement.parameters) |     (cudaq.SampleResult           |
|     -   [(c                       |     meth                          |
| udaq.operators.MatrixOperatorTerm | od)](api/languages/python_api.htm |
|         property)](api/langua     | l#cudaq.SampleResult.probability) |
| ges/python_api.html#cudaq.operato | -   [ProductOperator (in module   |
| rs.MatrixOperatorTerm.parameters) |     cudaq.operator                |
|     -                             | s)](api/languages/python_api.html |
|  [(cudaq.operators.ScalarOperator | #cudaq.operators.ProductOperator) |
|         property)](api/la         | -   [PyKernel (class in           |
| nguages/python_api.html#cudaq.ope |     cudaq)](api/language          |
| rators.ScalarOperator.parameters) | s/python_api.html#cudaq.PyKernel) |
|     -   [(                        | -   [PyKernelDecorator (class in  |
| cudaq.operators.spin.SpinOperator |     cudaq)](api/languages/python_ |
|         property)](api/langu      | api.html#cudaq.PyKernelDecorator) |
| ages/python_api.html#cudaq.operat |                                   |
| ors.spin.SpinOperator.parameters) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         property)](api/languages  |                                   |
| /python_api.html#cudaq.operators. |                                   |
| spin.SpinOperatorTerm.parameters) |                                   |
| -   [ParameterShift (class in     |                                   |
|     cudaq.gradien                 |                                   |
| ts)](api/languages/python_api.htm |                                   |
| l#cudaq.gradients.ParameterShift) |                                   |
| -   [parity() (in module          |                                   |
|     cudaq.operators.cus           |                                   |
| tom)](api/languages/python_api.ht |                                   |
| ml#cudaq.operators.custom.parity) |                                   |
| -   [Pauli (class in              |                                   |
|     cudaq.spin)](api/languages/   |                                   |
| python_api.html#cudaq.spin.Pauli) |                                   |
+-----------------------------------+-----------------------------------+

## Q {#Q}

+-----------------------------------+-----------------------------------+
| -   [qreg (in module              | -   [qubit_count                  |
|     cudaq)](api/lang              |     (                             |
| uages/python_api.html#cudaq.qreg) | cudaq.operators.spin.SpinOperator |
| -   [QuakeValue (class in         |     property)](api/langua         |
|     cudaq)](api/languages/        | ges/python_api.html#cudaq.operato |
| python_api.html#cudaq.QuakeValue) | rs.spin.SpinOperator.qubit_count) |
| -   [qubit (class in              |     -   [(cuda                    |
|     cudaq)](api/langu             | q.operators.spin.SpinOperatorTerm |
| ages/python_api.html#cudaq.qubit) |         property)](api/languages/ |
|                                   | python_api.html#cudaq.operators.s |
|                                   | pin.SpinOperatorTerm.qubit_count) |
|                                   | -   [qvector (class in            |
|                                   |     cudaq)](api/languag           |
|                                   | es/python_api.html#cudaq.qvector) |
+-----------------------------------+-----------------------------------+

## R {#R}

+-----------------------------------+-----------------------------------+
| -   [random()                     | -   [reset_target() (in module    |
|     (                             |     cudaq)](api/languages/py      |
| cudaq.operators.spin.SpinOperator | thon_api.html#cudaq.reset_target) |
|     static                        | -   [Resources (class in          |
|     method)](api/l                |     cudaq)](api/languages         |
| anguages/python_api.html#cudaq.op | /python_api.html#cudaq.Resources) |
| erators.spin.SpinOperator.random) | -   [right_multiply()             |
| -   [rank() (in module            |     (cudaq.SuperOperator static   |
|     cudaq.mpi)](api/language      |     method)]                      |
| s/python_api.html#cudaq.mpi.rank) | (api/languages/python_api.html#cu |
| -   [register_names               | daq.SuperOperator.right_multiply) |
|     (cudaq.SampleResult           | -   [row_count                    |
|     attribute)                    |     (cudaq.KrausOperator          |
| ](api/languages/python_api.html#c |     prope                         |
| udaq.SampleResult.register_names) | rty)](api/languages/python_api.ht |
| -                                 | ml#cudaq.KrausOperator.row_count) |
|   [register_set_target_callback() | -   [run() (in module             |
|     (in module                    |     cudaq)](api/lan               |
|     cudaq)]                       | guages/python_api.html#cudaq.run) |
| (api/languages/python_api.html#cu | -   [run_async() (in module       |
| daq.register_set_target_callback) |     cudaq)](api/languages         |
| -   [requires_gradients()         | /python_api.html#cudaq.run_async) |
|     (cudaq.optimizers.COBYLA      | -   [RydbergHamiltonian (class in |
|     method)](api/lan              |     cudaq.operators)]             |
| guages/python_api.html#cudaq.opti | (api/languages/python_api.html#cu |
| mizers.COBYLA.requires_gradients) | daq.operators.RydbergHamiltonian) |
|     -   [                         |                                   |
| (cudaq.optimizers.GradientDescent |                                   |
|         method)](api/languages/py |                                   |
| thon_api.html#cudaq.optimizers.Gr |                                   |
| adientDescent.requires_gradients) |                                   |
|     -   [(cudaq.optimizers.LBFGS  |                                   |
|         method)](api/la           |                                   |
| nguages/python_api.html#cudaq.opt |                                   |
| imizers.LBFGS.requires_gradients) |                                   |
|                                   |                                   |
| -   [(cudaq.optimizers.NelderMead |                                   |
|         method)](api/languag      |                                   |
| es/python_api.html#cudaq.optimize |                                   |
| rs.NelderMead.requires_gradients) |                                   |
+-----------------------------------+-----------------------------------+

## S {#S}

+-----------------------------------+-----------------------------------+
| -   [sample() (in module          | -   [set_target() (in module      |
|     cudaq)](api/langua            |     cudaq)](api/languages/        |
| ges/python_api.html#cudaq.sample) | python_api.html#cudaq.set_target) |
|     -   [(in module               | -   [signatureWithCallables()     |
|                                   |     (cudaq.PyKernelDecorator      |
|      cudaq.orca)](api/languages/p |     method)](api/languag          |
| ython_api.html#cudaq.orca.sample) | es/python_api.html#cudaq.PyKernel |
| -   [sample_async() (in module    | Decorator.signatureWithCallables) |
|     cudaq)](api/languages/py      | -   [SimulationPrecision (class   |
| thon_api.html#cudaq.sample_async) |     in                            |
| -   [SampleResult (class in       |                                   |
|     cudaq)](api/languages/py      |   cudaq)](api/languages/python_ap |
| thon_api.html#cudaq.SampleResult) | i.html#cudaq.SimulationPrecision) |
| -   [ScalarOperator (class in     | -   [simulator (cudaq.Target      |
|     cudaq.operato                 |                                   |
| rs)](api/languages/python_api.htm |   property)](api/languages/python |
| l#cudaq.operators.ScalarOperator) | _api.html#cudaq.Target.simulator) |
| -   [Schedule (class in           | -   [slice() (cudaq.QuakeValue    |
|     cudaq)](api/language          |     method)](api/languages/python |
| s/python_api.html#cudaq.Schedule) | _api.html#cudaq.QuakeValue.slice) |
| -   [serialize()                  | -   [SpinOperator (class in       |
|     (                             |     cudaq.operators.spin)         |
| cudaq.operators.spin.SpinOperator | ](api/languages/python_api.html#c |
|     method)](api/lang             | udaq.operators.spin.SpinOperator) |
| uages/python_api.html#cudaq.opera | -   [SpinOperatorElement (class   |
| tors.spin.SpinOperator.serialize) |     in                            |
|     -   [(cuda                    |     cudaq.operators.spin)](api/l  |
| q.operators.spin.SpinOperatorTerm | anguages/python_api.html#cudaq.op |
|         method)](api/language     | erators.spin.SpinOperatorElement) |
| s/python_api.html#cudaq.operators | -   [SpinOperatorTerm (class in   |
| .spin.SpinOperatorTerm.serialize) |     cudaq.operators.spin)](ap     |
|     -   [(cudaq.SampleResult      | i/languages/python_api.html#cudaq |
|         me                        | .operators.spin.SpinOperatorTerm) |
| thod)](api/languages/python_api.h | -   [squeeze() (in module         |
| tml#cudaq.SampleResult.serialize) |     cudaq.operators.cust          |
| -   [set_noise() (in module       | om)](api/languages/python_api.htm |
|     cudaq)](api/languages         | l#cudaq.operators.custom.squeeze) |
| /python_api.html#cudaq.set_noise) | -   [State (class in              |
| -   [set_random_seed() (in module |     cudaq)](api/langu             |
|     cudaq)](api/languages/pytho   | ages/python_api.html#cudaq.State) |
| n_api.html#cudaq.set_random_seed) | -   [SuperOperator (class in      |
|                                   |     cudaq)](api/languages/pyt     |
|                                   | hon_api.html#cudaq.SuperOperator) |
+-----------------------------------+-----------------------------------+

## T {#T}

+-----------------------------------+-----------------------------------+
| -   [Target (class in             | -   [to_numpy()                   |
|     cudaq)](api/langua            |     (cudaq.ComplexMatrix          |
| ges/python_api.html#cudaq.Target) |     me                            |
| -   [target                       | thod)](api/languages/python_api.h |
|     (cudaq.ope                    | tml#cudaq.ComplexMatrix.to_numpy) |
| rators.boson.BosonOperatorElement | -   [to_sparse_matrix()           |
|     property)](api/languages/     |     (cu                           |
| python_api.html#cudaq.operators.b | daq.operators.boson.BosonOperator |
| oson.BosonOperatorElement.target) |     method)](api/languages/pyt    |
|     -   [(cudaq.operato           | hon_api.html#cudaq.operators.boso |
| rs.fermion.FermionOperatorElement | n.BosonOperator.to_sparse_matrix) |
|                                   |     -   [(cudaq.                  |
|     property)](api/languages/pyth | operators.boson.BosonOperatorTerm |
| on_api.html#cudaq.operators.fermi |                                   |
| on.FermionOperatorElement.target) |    method)](api/languages/python_ |
|     -   [(cudaq.o                 | api.html#cudaq.operators.boson.Bo |
| perators.spin.SpinOperatorElement | sonOperatorTerm.to_sparse_matrix) |
|         property)](api/language   |     -   [(cudaq.                  |
| s/python_api.html#cudaq.operators | operators.fermion.FermionOperator |
| .spin.SpinOperatorElement.target) |                                   |
| -   [Tensor (class in             |    method)](api/languages/python_ |
|     cudaq)](api/langua            | api.html#cudaq.operators.fermion. |
| ges/python_api.html#cudaq.Tensor) | FermionOperator.to_sparse_matrix) |
| -   [term_count                   |     -   [(cudaq.oper              |
|     (cu                           | ators.fermion.FermionOperatorTerm |
| daq.operators.boson.BosonOperator |         m                         |
|     property)](api/languag        | ethod)](api/languages/python_api. |
| es/python_api.html#cudaq.operator | html#cudaq.operators.fermion.Ferm |
| s.boson.BosonOperator.term_count) | ionOperatorTerm.to_sparse_matrix) |
|     -   [(cudaq.                  |     -   [(                        |
| operators.fermion.FermionOperator | cudaq.operators.spin.SpinOperator |
|                                   |         method)](api/languages/p  |
|        property)](api/languages/p | ython_api.html#cudaq.operators.sp |
| ython_api.html#cudaq.operators.fe | in.SpinOperator.to_sparse_matrix) |
| rmion.FermionOperator.term_count) |     -   [(cuda                    |
|     -                             | q.operators.spin.SpinOperatorTerm |
|  [(cudaq.operators.MatrixOperator |                                   |
|         property)](api/la         |      method)](api/languages/pytho |
| nguages/python_api.html#cudaq.ope | n_api.html#cudaq.operators.spin.S |
| rators.MatrixOperator.term_count) | pinOperatorTerm.to_sparse_matrix) |
|     -   [(                        | -   [to_string()                  |
| cudaq.operators.spin.SpinOperator |     (cudaq.ope                    |
|         property)](api/langu      | rators.boson.BosonOperatorElement |
| ages/python_api.html#cudaq.operat |     method)](api/languages/pyt    |
| ors.spin.SpinOperator.term_count) | hon_api.html#cudaq.operators.boso |
| -   [term_id                      | n.BosonOperatorElement.to_string) |
|     (cudaq.                       |     -   [(cudaq.operato           |
| operators.boson.BosonOperatorTerm | rs.fermion.FermionOperatorElement |
|     property)](api/language       |                                   |
| s/python_api.html#cudaq.operators |    method)](api/languages/python_ |
| .boson.BosonOperatorTerm.term_id) | api.html#cudaq.operators.fermion. |
|     -   [(cudaq.oper              | FermionOperatorElement.to_string) |
| ators.fermion.FermionOperatorTerm |     -   [(cuda                    |
|                                   | q.operators.MatrixOperatorElement |
|       property)](api/languages/py |         method)](api/language     |
| thon_api.html#cudaq.operators.fer | s/python_api.html#cudaq.operators |
| mion.FermionOperatorTerm.term_id) | .MatrixOperatorElement.to_string) |
|     -   [(c                       |     -   [(                        |
| udaq.operators.MatrixOperatorTerm | cudaq.operators.spin.SpinOperator |
|         property)](api/lan        |         method)](api/lang         |
| guages/python_api.html#cudaq.oper | uages/python_api.html#cudaq.opera |
| ators.MatrixOperatorTerm.term_id) | tors.spin.SpinOperator.to_string) |
|     -   [(cuda                    |     -   [(cudaq.o                 |
| q.operators.spin.SpinOperatorTerm | perators.spin.SpinOperatorElement |
|         property)](api/langua     |         method)](api/languages/p  |
| ges/python_api.html#cudaq.operato | ython_api.html#cudaq.operators.sp |
| rs.spin.SpinOperatorTerm.term_id) | in.SpinOperatorElement.to_string) |
| -   [to_dict() (cudaq.Resources   |     -   [(cuda                    |
|                                   | q.operators.spin.SpinOperatorTerm |
|    method)](api/languages/python_ |         method)](api/language     |
| api.html#cudaq.Resources.to_dict) | s/python_api.html#cudaq.operators |
| -   [to_json()                    | .spin.SpinOperatorTerm.to_string) |
|     (                             | -   [translate() (in module       |
| cudaq.gradients.CentralDifference |     cudaq)](api/languages         |
|     method)](api/la               | /python_api.html#cudaq.translate) |
| nguages/python_api.html#cudaq.gra | -   [trim()                       |
| dients.CentralDifference.to_json) |     (cu                           |
|     -   [(                        | daq.operators.boson.BosonOperator |
| cudaq.gradients.ForwardDifference |     method)](api/l                |
|         method)](api/la           | anguages/python_api.html#cudaq.op |
| nguages/python_api.html#cudaq.gra | erators.boson.BosonOperator.trim) |
| dients.ForwardDifference.to_json) |     -   [(cudaq.                  |
|     -                             | operators.fermion.FermionOperator |
|  [(cudaq.gradients.ParameterShift |         method)](api/langu        |
|         method)](api              | ages/python_api.html#cudaq.operat |
| /languages/python_api.html#cudaq. | ors.fermion.FermionOperator.trim) |
| gradients.ParameterShift.to_json) |     -                             |
|     -   [(                        |  [(cudaq.operators.MatrixOperator |
| cudaq.operators.spin.SpinOperator |         method)](                 |
|         method)](api/la           | api/languages/python_api.html#cud |
| nguages/python_api.html#cudaq.ope | aq.operators.MatrixOperator.trim) |
| rators.spin.SpinOperator.to_json) |     -   [(                        |
|     -   [(cuda                    | cudaq.operators.spin.SpinOperator |
| q.operators.spin.SpinOperatorTerm |         method)](api              |
|         method)](api/langua       | /languages/python_api.html#cudaq. |
| ges/python_api.html#cudaq.operato | operators.spin.SpinOperator.trim) |
| rs.spin.SpinOperatorTerm.to_json) | -   [type_to_str()                |
|     -   [(cudaq.optimizers.COBYLA |     (cudaq.PyKernelDecorator      |
|         metho                     |     static                        |
| d)](api/languages/python_api.html |     method)](                     |
| #cudaq.optimizers.COBYLA.to_json) | api/languages/python_api.html#cud |
|     -   [                         | aq.PyKernelDecorator.type_to_str) |
| (cudaq.optimizers.GradientDescent |                                   |
|         method)](api/l            |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| timizers.GradientDescent.to_json) |                                   |
|     -   [(cudaq.optimizers.LBFGS  |                                   |
|         meth                      |                                   |
| od)](api/languages/python_api.htm |                                   |
| l#cudaq.optimizers.LBFGS.to_json) |                                   |
|                                   |                                   |
| -   [(cudaq.optimizers.NelderMead |                                   |
|         method)](                 |                                   |
| api/languages/python_api.html#cud |                                   |
| aq.optimizers.NelderMead.to_json) |                                   |
|     -   [(cudaq.PyKernelDecorator |                                   |
|         metho                     |                                   |
| d)](api/languages/python_api.html |                                   |
| #cudaq.PyKernelDecorator.to_json) |                                   |
| -   [to_matrix()                  |                                   |
|     (cu                           |                                   |
| daq.operators.boson.BosonOperator |                                   |
|     method)](api/langua           |                                   |
| ges/python_api.html#cudaq.operato |                                   |
| rs.boson.BosonOperator.to_matrix) |                                   |
|     -   [(cudaq.ope               |                                   |
| rators.boson.BosonOperatorElement |                                   |
|                                   |                                   |
|        method)](api/languages/pyt |                                   |
| hon_api.html#cudaq.operators.boso |                                   |
| n.BosonOperatorElement.to_matrix) |                                   |
|     -   [(cudaq.                  |                                   |
| operators.boson.BosonOperatorTerm |                                   |
|         method)](api/languages/   |                                   |
| python_api.html#cudaq.operators.b |                                   |
| oson.BosonOperatorTerm.to_matrix) |                                   |
|     -   [(cudaq.                  |                                   |
| operators.fermion.FermionOperator |                                   |
|         method)](api/languages/   |                                   |
| python_api.html#cudaq.operators.f |                                   |
| ermion.FermionOperator.to_matrix) |                                   |
|     -   [(cudaq.operato           |                                   |
| rs.fermion.FermionOperatorElement |                                   |
|                                   |                                   |
|    method)](api/languages/python_ |                                   |
| api.html#cudaq.operators.fermion. |                                   |
| FermionOperatorElement.to_matrix) |                                   |
|     -   [(cudaq.oper              |                                   |
| ators.fermion.FermionOperatorTerm |                                   |
|                                   |                                   |
|       method)](api/languages/pyth |                                   |
| on_api.html#cudaq.operators.fermi |                                   |
| on.FermionOperatorTerm.to_matrix) |                                   |
|     -                             |                                   |
|  [(cudaq.operators.MatrixOperator |                                   |
|         method)](api/l            |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| erators.MatrixOperator.to_matrix) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.MatrixOperatorElement |                                   |
|         method)](api/language     |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .MatrixOperatorElement.to_matrix) |                                   |
|     -   [(c                       |                                   |
| udaq.operators.MatrixOperatorTerm |                                   |
|         method)](api/langu        |                                   |
| ages/python_api.html#cudaq.operat |                                   |
| ors.MatrixOperatorTerm.to_matrix) |                                   |
|     -                             |                                   |
|  [(cudaq.operators.ScalarOperator |                                   |
|         method)](api/l            |                                   |
| anguages/python_api.html#cudaq.op |                                   |
| erators.ScalarOperator.to_matrix) |                                   |
|     -   [(                        |                                   |
| cudaq.operators.spin.SpinOperator |                                   |
|         method)](api/lang         |                                   |
| uages/python_api.html#cudaq.opera |                                   |
| tors.spin.SpinOperator.to_matrix) |                                   |
|     -   [(cudaq.o                 |                                   |
| perators.spin.SpinOperatorElement |                                   |
|         method)](api/languages/p  |                                   |
| ython_api.html#cudaq.operators.sp |                                   |
| in.SpinOperatorElement.to_matrix) |                                   |
|     -   [(cuda                    |                                   |
| q.operators.spin.SpinOperatorTerm |                                   |
|         method)](api/language     |                                   |
| s/python_api.html#cudaq.operators |                                   |
| .spin.SpinOperatorTerm.to_matrix) |                                   |
+-----------------------------------+-----------------------------------+

## U {#U}

+-----------------------------------------------------------------------+
| -   [unregister_set_target_callback() (in module                      |
|     cudaq)                                                            |
| ](api/languages/python_api.html#cudaq.unregister_set_target_callback) |
| -   [unset_noise() (in module                                         |
|     cudaq)](api/languages/python_api.html#cudaq.unset_noise)          |
| -   [upper_bounds (cudaq.optimizers.COBYLA                            |
|     property)                                                         |
| ](api/languages/python_api.html#cudaq.optimizers.COBYLA.upper_bounds) |
|     -   [(cudaq.optimizers.GradientDescent                            |
|         property)](api/lan                                            |
| guages/python_api.html#cudaq.optimizers.GradientDescent.upper_bounds) |
|     -   [(cudaq.optimizers.LBFGS                                      |
|         property                                                      |
| )](api/languages/python_api.html#cudaq.optimizers.LBFGS.upper_bounds) |
|     -   [(cudaq.optimizers.NelderMead                                 |
|         property)](ap                                                 |
| i/languages/python_api.html#cudaq.optimizers.NelderMead.upper_bounds) |
+-----------------------------------------------------------------------+

## V {#V}

+-----------------------------------+-----------------------------------+
| -   [values() (cudaq.SampleResult | -   [vqe() (in module             |
|                                   |     cudaq)](api/lan               |
|  method)](api/languages/python_ap | guages/python_api.html#cudaq.vqe) |
| i.html#cudaq.SampleResult.values) |                                   |
+-----------------------------------+-----------------------------------+

## X {#X}

+-----------------------------------+-----------------------------------+
| -   [x() (in module               | -   [XError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.x) | ges/python_api.html#cudaq.XError) |
+-----------------------------------+-----------------------------------+

## Y {#Y}

+-----------------------------------+-----------------------------------+
| -   [y() (in module               | -   [YError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.y) | ges/python_api.html#cudaq.YError) |
+-----------------------------------+-----------------------------------+

## Z {#Z}

+-----------------------------------+-----------------------------------+
| -   [z() (in module               | -   [ZError (class in             |
|     cudaq.spin)](api/langua       |     cudaq)](api/langua            |
| ges/python_api.html#cudaq.spin.z) | ges/python_api.html#cudaq.ZError) |
+-----------------------------------+-----------------------------------+
:::
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
